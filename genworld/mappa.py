"""Import di una mappa disegnata: da immagine a classi di terreno e quote.

E' la modalita' "interpretativa": la mappa non contiene una heightmap, quindi
le quote vanno DEDOTTE. Il presupposto e' che una mappa di etichette contenga
gia' l'informazione altimetrica in forma implicita - "montagna", "foresta",
"mare" sono affermazioni sulla quota, non categorie di colore.

Tre difficolta' specifiche delle mappe disegnate a mano o da generatore
artistico, che una mappa a colori piatti non ha:

1. L'ombreggiatura del rilievo e' DIPINTA DENTRO i colori, quindi lo stesso
   terreno ha molte tonalita' e il versante in ombra si confonde con la
   foresta. Si classifica in HSV, non in RGB, e si usa la tinta piu' del valore.

2. Quella stessa ombreggiatura e' anche un'INFORMAZIONE: i rilievi dipinti sono
   texture ad alto contrasto locale, le pianure sono lisce. Il contrasto locale
   diventa un canale "rugosita'" che alza le quote dove ci sono montagne,
   qualunque sia la tinta.

3. Le SCRITTE sono pixel scuri che diventerebbero canyon. Vanno mascherate e
   ricoperte prima di classificare.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_dilation, gaussian_filter, median_filter,
                           uniform_filter)

# --------------------------------------------------------------------------
# Classi di terreno
# --------------------------------------------------------------------------
(OCEANO, MARE, SPIAGGIA, DESERTO, PIANURA, PRATERIA, FORESTA, MONTAGNA, NEVE,
 FIUME, VULCANO, CRATERE) = range(12)

NOMI = {
    OCEANO: "oceano", MARE: "mare", SPIAGGIA: "spiaggia", DESERTO: "deserto",
    PIANURA: "pianura", PRATERIA: "prateria", FORESTA: "foresta",
    MONTAGNA: "montagna", NEVE: "neve", FIUME: "fiume",
    VULCANO: "vulcano", CRATERE: "cratere",
}

# Colore per l'anteprima della classificazione (non sono i blocchi Minecraft).
COLORI_CLASSE = {
    OCEANO: (28, 52, 104), MARE: (62, 118, 170), SPIAGGIA: (222, 208, 165),
    DESERTO: (226, 199, 136), PIANURA: (176, 190, 120), PRATERIA: (128, 166, 88),
    FORESTA: (52, 102, 58), MONTAGNA: (128, 120, 112), NEVE: (238, 242, 246),
    FIUME: (86, 146, 196), VULCANO: (54, 46, 46), CRATERE: (214, 86, 24),
}

# Acqua in generale: niente alberi, niente erosione, niente insediamenti.
ACQUA = (OCEANO, MARE, FIUME)
# Solo acqua MARINA. Distinzione necessaria: il tetto al livello del mare vale
# per oceano e mare, non per un fiume, che scorre in quota e deve restare
# dov'e' invece di essere schiacciato a 62.
MARINO = (OCEANO, MARE)


@dataclass
class Quote:
    """Quota target per classe, in blocchi Minecraft (livello del mare a 62)."""

    livello_mare: int = 62
    base: dict[int, float] = field(default_factory=lambda: {
        OCEANO: 30.0, MARE: 54.0, SPIAGGIA: 64.0, DESERTO: 68.0,
        PIANURA: 71.0, PRATERIA: 79.0, FORESTA: 86.0,
        MONTAGNA: 120.0, NEVE: 140.0, FIUME: 70.0,
        VULCANO: 130.0, CRATERE: 150.0,
    })
    # quanto la rugosita' dipinta alza le quote (blocchi a rugosita' piena)
    guadagno_rugosita: float = 62.0


# --------------------------------------------------------------------------
# Lettura e pulizia
# --------------------------------------------------------------------------

def carica(percorso: str, lato: int | None = None,
           ritaglio: float = 0.0) -> np.ndarray:
    """Carica l'immagine come RGB float 0..1, opzionalmente ridimensionata.

    `ritaglio` toglie una frazione di bordo da ogni lato. Serve piu' spesso di
    quanto sembri: le mappe disegnate hanno quasi sempre una cornice o una
    vignettatura chiara, e il bianco sbiadito del bordo viene classificato come
    neve, producendo una calotta glaciale lungo i margini del mondo.
    """
    im = Image.open(percorso).convert("RGB")
    if ritaglio > 0:
        w, h = im.size
        dx, dy = int(w * ritaglio), int(h * ritaglio)
        im = im.crop((dx, dy, w - dx, h - dy))
    if lato is not None:
        im = im.resize((lato, lato), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def a_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB 0..1 -> (tinta 0..360, saturazione 0..1, valore 0..1), vettoriale."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(2), rgb.min(2)
    d = mx - mn
    v = mx
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    dd = np.maximum(d, 1e-6)
    h = np.select(
        [mx == r, mx == g, mx == b],
        [((g - b) / dd) % 6.0, (b - r) / dd + 2.0, (r - g) / dd + 4.0],
        default=0.0,
    ) * 60.0
    return h, s, np.where(d < 1e-6, v, v)


def maschera_testo(rgb: np.ndarray, soglia_valore: float = 0.52,
                   soglia_sat: float = 0.32, dilata: int = 3) -> np.ndarray:
    """Pixel di scritte e cornice: scuri e poco saturi.

    La soglia e' generosa di proposito. Il testo e' antialiasato, e l'alone
    grigio attorno a una lettera e' proprio cio' che sopravvive alla riparazione
    e diventa rilievo. Meglio mascherare qualche pixel di troppo: l'acqua
    profonda, che ha una luminanza simile, e' molto piu' satura e non viene
    presa.
    """
    _, s, v = a_hsv(rgb)
    m = (v < soglia_valore) & (s < soglia_sat)
    if dilata:
        m = binary_dilation(m, iterations=dilata)
    return m


def ripara(rgb: np.ndarray, maschera: np.ndarray, finestra: int = 13) -> np.ndarray:
    """Ricopre i pixel mascherati con la mediana del vicinato."""
    if not maschera.any():
        return rgb
    fondo = np.stack([median_filter(rgb[..., c], size=finestra) for c in range(3)], -1)
    return np.where(maschera[..., None], fondo, rgb)


# --------------------------------------------------------------------------
# Classificazione
# --------------------------------------------------------------------------

def rugosita(rgb: np.ndarray, finestra: int = 7, maschera: np.ndarray | None = None,
             soglia: float = 0.42) -> np.ndarray:
    """Contrasto locale normalizzato 0..1: marca i rilievi dipinti.

    Due accorgimenti imparati sul campo:

    - `maschera` azzera le zone delle scritte. Ricoprire il colore non basta:
      il bordo di una lettera resta un salto di luminanza e diventa una catena
      montuosa a forma di parola.
    - `soglia` scarta il contrasto di fondo. Un dipinto ha grana ovunque; senza
      un pavimento, ogni terra emersa risulta rugosa e l'intero continente si
      solleva. Sotto soglia la rugosita' e' zero, sopra viene riscalata su
      tutto l'intervallo.
    """
    lum = rgb.mean(2)
    media = uniform_filter(lum, finestra)
    var = uniform_filter(lum * lum, finestra) - media * media
    r = np.sqrt(np.maximum(var, 0.0))
    rif = float(np.percentile(r, 99)) or 1.0
    r = np.clip(r / rif, 0.0, 1.0)

    if maschera is not None and maschera.any():
        r = np.where(binary_dilation(maschera, iterations=3), 0.0, r)

    # Il bordo dell'immagine e' sempre un salto di luminanza enorme, quindi
    # contrasto massimo, quindi "montagna": senza questo, ogni mappa nasce
    # circondata da una muraglia. Il ritaglio della cornice non basta, perche'
    # il bordo esiste comunque dopo il ritaglio.
    margine = max(3, int(round(min(r.shape) * 0.012)))
    r[:margine, :] = 0.0
    r[-margine:, :] = 0.0
    r[:, :margine] = 0.0
    r[:, -margine:] = 0.0

    return np.clip((r - soglia) / max(1e-6, 1.0 - soglia), 0.0, 1.0)


def classifica(rgb: np.ndarray, rug: np.ndarray | None = None,
               soglia_montagna: float = 0.30, pulisci: int = 5) -> np.ndarray:
    """Da RGB alla mappa di etichette.

    L'ordine dei test conta: si decide prima acqua/non acqua sulla tinta, poi
    si differenzia la terra. La montagna NON viene dal colore ma dalla
    rugosita', perche' su una mappa dipinta i rilievi sono riconoscibili dalla
    texture molto piu' che dalla tinta.
    """
    h, s, v = a_hsv(rgb)
    if rug is None:
        rug = rugosita(rgb)

    cls = np.full(h.shape, PRATERIA, dtype=np.uint8)

    acqua = (h >= 170) & (h <= 260) & (s > 0.15)
    profonda = acqua & ((v < 0.62) | (h > 206))
    cls[acqua] = MARE
    cls[profonda] = OCEANO

    terra = ~acqua
    cls[terra & (h >= 95) & (h < 175) & (v < 0.70)] = FORESTA
    cls[terra & (h >= 36) & (h < 62) & (v > 0.80)] = DESERTO
    cls[terra & (h >= 62) & (h < 95) & (v > 0.74)] = PIANURA
    cls[terra & (v > 0.90) & (s < 0.14)] = NEVE

    # rilievi: dalla texture, non dalla tinta
    cls[terra & (rug > soglia_montagna)] = MONTAGNA
    cls[terra & (rug > soglia_montagna) & (v > 0.88)] = NEVE

    # Regolarizzazione spaziale. Senza, restano pixel isolati di terra sparsi
    # in mezzo al mare: ognuno diventa un isolotto alto un blocco, e la semina
    # ci pianta sopra un albero. Il risultato sono boschetti in mezzo
    # all'oceano aperto, visibili solo nell'anteprima del mondo finito.
    if pulisci > 1:
        cls = median_filter(cls, size=pulisci)
        acqua = np.isin(cls, ACQUA)
        terra = ~acqua

    # la spiaggia e' la terra a contatto con l'acqua
    bordo = binary_dilation(acqua, iterations=2) & terra
    cls[bordo & (cls != MONTAGNA) & (cls != NEVE)] = SPIAGGIA
    return cls


def statistiche(cls: np.ndarray) -> list[tuple[str, float]]:
    tot = cls.size
    out = [(NOMI[k], float((cls == k).sum()) / tot * 100) for k in sorted(NOMI)]
    return sorted(out, key=lambda x: -x[1])


# --------------------------------------------------------------------------
# Deduzione delle quote
# --------------------------------------------------------------------------

def altimetria(cls: np.ndarray, rug: np.ndarray, q: Quote | None = None,
               morbidezza: float = 6.0, seed: int = 0) -> np.ndarray:
    """Dalle etichette a una heightmap continua in blocchi.

    1. quota target per classe
    2. sfocatura forte: le transizioni fra classi diventano pendii, non muri
    3. spinta dalla rugosita': dove il disegno ha rilievo, il terreno sale
    4. si reimpone l'acqua sotto il livello del mare, che la sfocatura alzerebbe
    """
    q = q or Quote()
    h = np.zeros(cls.shape, dtype=np.float32)
    for k, quota in q.base.items():
        h[cls == k] = quota

    h = gaussian_filter(h, morbidezza)

    # La spinta di rilievo vale solo sulla terra: la grana dipinta del mare
    # non deve creare secche, e le coste devono restare coste.
    spinta = gaussian_filter(rug.astype(np.float32), 2.0)
    spinta = np.where(np.isin(cls, MARINO), 0.0, spinta)
    h = h + q.guadagno_rugosita * spinta

    # L'acqua non puo' stare sopra il pelo del mare: la sfocatura e la spinta
    # di rugosita' la alzerebbero, e le coste si chiuderebbero.
    marino = np.isin(cls, MARINO)
    tetto = np.where(cls == OCEANO, q.livello_mare - 12.0, q.livello_mare - 3.0)
    h = np.where(marino, np.minimum(h, tetto), h)
    return h


def anteprima_classi(cls: np.ndarray) -> Image.Image:
    rgb = np.zeros((*cls.shape, 3), dtype=np.uint8)
    for k, c in COLORI_CLASSE.items():
        rgb[cls == k] = c
    return Image.fromarray(rgb, "RGB")


def anteprima_quote(h: np.ndarray, livello_mare: int = 62) -> Image.Image:
    """Grigio per la terra, blu per il fondale, con ombreggiatura."""
    gz, gx = np.gradient(h)
    luce = np.clip(0.55 + 0.35 * (gx + gz), 0.2, 1.3)
    terra = h > livello_mare
    lo, hi = float(h[terra].min()) if terra.any() else 0.0, float(h.max())
    n = np.clip((h - lo) / max(1.0, hi - lo), 0, 1)
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.where(terra, 70 + 185 * n, 20 + 40 * n)
    rgb[..., 1] = np.where(terra, 80 + 175 * n, 50 + 60 * n)
    rgb[..., 2] = np.where(terra, 60 + 170 * n, 110 + 80 * n)
    rgb *= luce[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
