"""Pre-elaborazione: portare qualunque mappa a una forma standard.

Il problema che risolve: le soglie HSV scritte a mano funzionano su UNA
immagine. Cambia il disegnatore, lo stile, la resa dei colori, e la
classificazione crolla. Tarare a mano ogni mappa non e' una strategia.

La soluzione non e' trovare soglie migliori: e' smettere di usare soglie
assolute. Qui si normalizza l'immagine e si misurano i colori RELATIVAMENTE
alle statistiche dell'immagine stessa, cosi' che "il piu' blu", "il piu'
luminoso", "il piu' verde" restino veri qualunque sia la tavolozza.

Tre stadi:
  1. ritaglio automatico della cornice
  2. bilanciamento: l'istogramma viene steso in modo confrontabile
  3. riconoscimento della famiglia di mappa, che decide come procedere
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

# --------------------------------------------------------------------------
# Spazio colore percettivo
# --------------------------------------------------------------------------

_M_RGB_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_BIANCO_D65 = np.array([0.95047, 1.00000, 1.08883])


def a_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0..1 -> CIELAB (L 0..100, a e b tipicamente -100..100).

    Si lavora in Lab e non in HSV perche' in Lab le distanze corrispondono a
    differenze percepite: due verdi che a occhio sono lo stesso verde restano
    vicini anche se la tinta HSV oscilla. Per raggruppare i colori di un
    dipinto e' la differenza fra un raggruppamento sensato e uno casuale.
    """
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB_XYZ.T / _BIANCO_D65
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(np.maximum(xyz, 1e-12)), xyz / (3 * d * d) + 4.0 / 29.0)
    return np.stack([
        116.0 * f[..., 1] - 16.0,
        500.0 * (f[..., 0] - f[..., 1]),
        200.0 * (f[..., 1] - f[..., 2]),
    ], axis=-1)


# --------------------------------------------------------------------------
# 1. Ritaglio automatico della cornice
# --------------------------------------------------------------------------

def trova_cornice(rgb: np.ndarray, tolleranza: float = 0.05,
                  massimo: float = 0.12) -> tuple[int, int, int, int]:
    """Quante righe/colonne di cornice ci sono su ciascun lato.

    Una cornice e' una fascia di bordo uniforme e diversa dall'interno. Si
    procede dal bordo verso il centro finche' la riga somiglia alla prima piu'
    che all'interno della mappa.

    Il ritaglio per lato conta: molte mappe hanno il bordo bianco solo in alto,
    dove sta il titolo, e ritagliare in modo uniforme butterebbe via mappa buona
    sugli altri tre lati.
    """
    h, w = rgb.shape[:2]
    interno = np.median(rgb[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3), axis=0)

    def scansiona(linee: np.ndarray, limite: int) -> int:
        rif = np.median(linee[0], axis=0)
        if np.abs(rif - interno).mean() < tolleranza:
            return 0
        for i in range(limite):
            m = np.median(linee[i], axis=0)
            if np.abs(m - rif).mean() > tolleranza and np.abs(m - interno).mean() < tolleranza * 2:
                return i
        return limite

    lim_v, lim_o = int(h * massimo), int(w * massimo)
    alto = scansiona(rgb, lim_v)
    basso = scansiona(rgb[::-1], lim_v)
    sinistra = scansiona(rgb.transpose(1, 0, 2), lim_o)
    destra = scansiona(rgb.transpose(1, 0, 2)[::-1], lim_o)
    return alto, basso, sinistra, destra


def ritaglia_cornice(rgb: np.ndarray, margine_extra: int = 2) -> np.ndarray:
    a, b, s, d = trova_cornice(rgb)
    h, w = rgb.shape[:2]
    a, b = a + margine_extra, b + margine_extra
    s, d = s + margine_extra, d + margine_extra
    return rgb[a:h - b, s:w - d]


# --------------------------------------------------------------------------
# 2. Bilanciamento
# --------------------------------------------------------------------------

def bilancia(rgb: np.ndarray, percentile: float = 1.0,
             forza: float = 0.7) -> np.ndarray:
    """Stende l'istogramma di ciascun canale fra i percentili indicati.

    Serve a rendere confrontabili mappe chiare e mappe scure, o con dominanti
    seppia. `forza` < 1 attenua l'intervento: una correzione totale
    distruggerebbe le dominanti volute (una mappa artica *deve* essere fredda).
    """
    out = np.empty_like(rgb)
    for c in range(3):
        canale = rgb[..., c]
        lo, hi = np.percentile(canale, [percentile, 100 - percentile])
        if hi - lo < 1e-4:
            out[..., c] = canale
            continue
        steso = np.clip((canale - lo) / (hi - lo), 0.0, 1.0)
        out[..., c] = (1 - forza) * canale + forza * steso
    return out


# --------------------------------------------------------------------------
# 3. Famiglia di mappa
# --------------------------------------------------------------------------

def famiglia(rgb: np.ndarray) -> tuple[str, dict]:
    """Riconosce con che tipo di mappa si ha a che fare.

    Non esiste una pipeline unica: una mappa a colori piatti si legge con una
    tabella esatta, una dipinta va raggruppata, uno schizzo a matita non ha
    colore e va trattato come disegno al tratto. Sbagliare famiglia significa
    applicare lo strumento sbagliato, quindi va deciso prima di tutto il resto.
    """
    lab = a_lab(rgb)
    croma = np.hypot(lab[..., 1], lab[..., 2])
    lum = lab[..., 0]

    # quanti colori distinti: poche decine = colori piatti, migliaia = dipinta
    q = (rgb * 31).astype(np.int16)
    codici = q[..., 0] * 1024 + q[..., 1] * 32 + q[..., 2]
    conteggi = np.bincount(codici.ravel())
    significativi = int((conteggi > codici.size * 0.0005).sum())

    # texture: grana locale della luminanza
    media = uniform_filter(lum, 5)
    grana = float(np.sqrt(np.maximum(
        uniform_filter(lum * lum, 5) - media * media, 0)).mean())

    info = {
        "colori_significativi": significativi,
        "croma_media": float(croma.mean()),
        "grana": round(grana, 3),
    }

    if croma.mean() < 8:
        nome = "grigi"          # schizzo a matita, mappa in bianco e nero
    elif significativi <= 24 and grana < 1.2:
        nome = "colori_piatti"  # mappa politica o a legenda
    elif grana > 4.5:
        nome = "fotografica"    # satellitare o resa molto dettagliata
    else:
        nome = "dipinta"        # mappa artistica con ombreggiatura
    return nome, info
