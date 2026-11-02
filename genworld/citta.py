"""Citta': pianta regolare, mura, fosso, isolati, lotti sulla strada.

Una citta' e' il contrario di un mucchio di case: prima esiste il **vuoto** -
piazza, assi, circonvallazione - e le case vengono dopo, appoggiate a quel
vuoto, ognuna con la facciata verso la strada che la serve.

La pianta e' **regolare**: un quadrato con due assi che si incrociano nella
piazza e, nelle citta', una cinta con quattro porte, il fosso e un ponte per
ogni porta. Una versione precedente, a maglia organica (punti sparsi,
triangolati), dava isolati di ogni forma ma anche citta' "incasinate e non
ordinate" - cosi' le definiva chi le guardava in gioco - con mura che
seguivano un contorno a caso.

E soprattutto **il terreno si adatta alla citta', non il contrario**: dove la
pianta passa sopra un fiume o una collina il terreno viene spianato e il fiume
sparisce; la citta' non si deforma per rispettare l'ambiente. Le copie di
`cls` e `livello` che `pianifica` restituisce sono la mappa dopo questo
cambiamento.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import edifici as E
from .mappa import (ACQUA, CRATERE, FIUME, FORESTA, MARINO, MONTAGNA, NEVE,
                    PIANURA, PRATERIA, SPIAGGIA, VULCANO)

ABITABILI = (PIANURA, PRATERIA, FORESTA, SPIAGGIA)

# larghezza della carreggiata per rango
VICOLO, SECONDARIA, ASSE = 1, 2, 3
LARGHEZZA = {VICOLO: 1, SECONDARIA: 2, ASSE: 3}

# oltre questo raggio un insediamento e' una citta': ha mura e piazza vera
RAGGIO_CITTA = 30

# Chi sta dove. Una citta' non e' un dormitorio: le botteghe stanno sulla
# piazza e sulle vie principali, gli artigiani rumorosi e puzzolenti un po'
# piu' in la', e chi lavora la terra o il pesce sta al bordo, vicino a quello
# che lavora. E' la ragione per cui i mestieri hanno dato i nomi alle vie di
# mezza Europa.
BOTTEGHE_CENTRO = ("libraio", "cartografo", "speziale", "fruttivendolo",
                   "macellaio", "fabbro")
BOTTEGHE_MEDIO = ("armaiolo", "corazzaio", "falegname", "scalpellino",
                  "conciatore", "pastore")
BOTTEGHE_BORDO = ("fruttivendolo", "pastore", "falegname")


def _mestiere(t: float, sull_acqua: bool, rng: np.random.Generator) -> str:
    """Il mestiere di una casa, dedotto da dove sta.

    Ritorna "" per una semplice abitazione: una citta' fatta di sole botteghe
    e' un centro commerciale, non una citta'.
    """
    if sull_acqua:
        return "pescatore"
    if t < 0.3:
        quota, elenco = 0.55, BOTTEGHE_CENTRO
    elif t < 0.65:
        quota, elenco = 0.35, BOTTEGHE_MEDIO
    else:
        quota, elenco = 0.20, BOTTEGHE_BORDO
    if rng.random() > quota:
        return ""
    return elenco[int(rng.integers(0, len(elenco)))]


@dataclass
class Citta:
    z: int
    x: int
    raggio: int
    piazza: tuple[int, int]
    porte: list[tuple[int, int]] = field(default_factory=list)
    murata: bool = False
    edifici: int = 0
    lato: int = 0
    # la pianta: quadrato (predefinito), rettangolo, ovale o tondo. `lato` e' la
    # mezza estensione piu' lunga; `lz` e `lx` le due mezze estensioni (0 = lato)
    forma: str = "quadrato"
    lz: int = 0
    lx: int = 0

    @property
    def raggio_piazza(self) -> int:
        """Raggio del lastricato tondo della piazza."""
        return dati_piazza(self.lato, self.murata)[0]

    @property
    def e_citta(self) -> bool:
        return self.murata


# --------------------------------------------------------------------------
# Pianta regolare
# --------------------------------------------------------------------------
#
# Una citta' si progetta PRIMA e il mondo si adatta, non il contrario: dove
# la pianta passa sopra un fiume, una collina o un laghetto, il terreno viene
# spianato e il fiume sparisce sotto la citta'. E' il verso giusto - chi
# costruisce una citta' non la deforma per rispettare un ruscello - e toglie
# alla radice tutta la famiglia di difetti nata dal verso opposto (case che
# scartano lotti in riva all'acqua, cinte a pezzi, sponde da ripuntellare).
#
# La pianta e' un QUADRATO con lo stesso schema ovunque:
#
#   * due assi larghi tre celle che si incrociano nella piazza (11x11);
#   * se ci sono le mura, una via di circonvallazione a ridosso della cinta;
#   * i lotti riempiono gli isolati che i due assi e la circonvallazione
#     lasciano liberi, con la facciata verso la strada.
#
# Una citta' ha in piu' la cinta (tre celle di spessore, torri agli angoli, ai
# lati delle porte e a meta' dei lati), il fosso e il ponte: gli assi escono
# dalle quattro porte, attraversano il fosso e li' diventano ponte. Un
# villaggio ha la stessa pianta, piu' piccola, senza cinta.

SPESSORE_MURA = 3
SEMILARGHEZZA_PORTA = 2       # il portale e' largo 5: due pilastri e tre di passaggio
LARGHEZZA_FOSSO = 3
PROFONDITA_FOSSO = 3          # di quanto il fondo sta sotto la piana
MARGINE_RACCORDO = 10         # fascia in cui la piana sfuma nel terreno vero
PENDENZA_RACCORDO = 0.5       # quote per cella di scarpata: 1 blocco ogni 2
FASCIA_RACCORDO = 56          # fin dove puo' arrivare una scarpata (dislivello ~28)
SALTO_MASSIMO = 20            # dislivello (5-95 percentile) oltre cui il sito e' troppo aspro
MARINO_MASSIMO = 0.12         # quota di mare ammessa dentro la piana
# Distanza libera fra le piane (cinta e fossato compresi) di due abitati. Due citta'
# murate una attaccata all'altra - viste in gioco - non hanno senso: fra due
# cinte ci sta una campagna intera.
DISTANZA_FRA_CITTA_MURATE = 40
DISTANZA_FRA_ABITATI = 12
DISTANZA_DAL_VULCANO = 28        # dal bordo della piana: la scarpata arriva lontano e non deve tagliare un cono
SPOSTAMENTO_MASSIMO = 60      # di quanto un sito puo' spostarsi per trovare posto
PIAZZA = 5                    # mezzo lato della piazza
MIN_LOTTI_VILLAGGIO = 4       # sulle mappe grandi i lotti sono 12 e 16: vedi `lotti`
MIN_LOTTI_CITTA = 4


LATI_VILLAGGIO = (38, 30, 24, 20)
LATI_CITTA = (46, 38, 30)


# --------------------------------------------------------------------------
# Forme della pianta
# --------------------------------------------------------------------------
#
# La pianta resta sempre la stessa (due assi che si incrociano nella piazza, una
# circonvallazione dentro la cinta, i lotti negli isolati); cambia solo il
# CONTORNO. Una forma e' fissata per ogni citta' al momento della scelta, mai
# deformata dopo - le prime piante organiche venivano tutte scombussolate.
#
# `_metrica` da' la distanza dal contorno IN BLOCCHI, rimessa nella scala di
# `lato`: D <= lato e' l'interno, D in lato+1..lato+3 la cinta, e cosi' via,
# esattamente come con il quadrato (per cui coincide con la distanza di
# Chebyshev di prima). Cosi' piana, raccordo, fosso, cinta e strade fuori porta
# funzionano per ogni forma senza saperlo.

FORME = ("quadrato", "rettangolo", "ovale", "tondo")
PESI_FORME_MISTE = (0.40, 0.20, 0.20, 0.20)
RAPPORTI_ALLUNGAMENTO = (0.65, 0.75)
LATO_MIN_ALLUNGATA = 36          # sotto, un rettangolo o un ovale non ha posto per i lotti


@dataclass(frozen=True)
class Forma:
    tipo: str = "quadrato"
    lz: int = 0          # mezza estensione lungo z (nord-sud)
    lx: int = 0          # mezza estensione lungo x (est-ovest)


def forma_quadrata(lato: int) -> Forma:
    return Forma("quadrato", lato, lato)


def scegli_forma(forme: str, lato: int, murata: bool, rng: np.random.Generator) -> Forma:
    """La forma di una citta'. `forme` e' "quadrato" (sempre), "miste" (a sorte) o
    il nome di una forma. Solo le citta' murate cambiano forma: un villaggio senza
    cinta e' gia' un quadrato di case. Rettangolo e ovale servono lato >= 36."""
    if forme == "quadrato" or not murata:
        return forma_quadrata(lato)
    if forme == "miste":
        tipo = FORME[int(rng.choice(len(FORME), p=PESI_FORME_MISTE))]
    elif forme in FORME:
        tipo = forme
    else:
        raise ValueError(f"forma sconosciuta: {forme!r}")
    if tipo in ("rettangolo", "ovale") and lato < LATO_MIN_ALLUNGATA:
        tipo = "quadrato" if tipo == "rettangolo" else "tondo"
    if tipo in ("quadrato", "tondo"):
        return Forma(tipo, lato, lato)
    corta = int(round(lato * RAPPORTI_ALLUNGAMENTO[int(rng.integers(0, len(RAPPORTI_ALLUNGAMENTO)))]))
    return Forma(tipo, corta, lato) if rng.random() < 0.5 else Forma(tipo, lato, corta)


def _metrica(forma: Forma, lato: int, dz: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Distanza (intera) dal centro nella scala di `lato`, secondo la forma."""
    az, ax = np.abs(dz), np.abs(dx)
    if forma.tipo in ("quadrato", "rettangolo"):
        return (lato + np.maximum(az - forma.lz, ax - forma.lx)).astype(np.int64)
    azf, axf = az.astype(np.float64), ax.astype(np.float64)
    if forma.tipo == "tondo":
        return np.rint(np.hypot(azf, axf)).astype(np.int64)
    # ovale: distanza LUNGO IL RAGGIO dall'ellisse, |v| - R(theta). Cresce sempre
    # allontanandosi dal centro (le curve di livello sono ellissi concentriche); su
    # un contorno ripido la misura e' un po' piu' lunga della normale, per i
    # rapporti usati (>= 0,65) di poco. Spessore di cinta e fosso restano 3.
    e = np.sqrt((azf / forma.lz) ** 2 + (axf / forma.lx) ** 2)
    modulo = np.hypot(azf, axf)
    with np.errstate(divide="ignore", invalid="ignore"):
        raggio_ell = np.where(e > 1e-9, modulo / np.where(e > 1e-9, e, 1.0),
                              float(min(forma.lz, forma.lx)))
    return np.rint(lato + modulo - raggio_ell).astype(np.int64)


def piante_possibili(raggio: int) -> list[tuple[int, bool]]:
    """(mezzo lato dell'abitato, murata) che un sito di raggio dato puo'
    avere, dalla migliore. Un sito da citta' prova le misure da citta' e,
    se la mappa non ha posto, ripiega su un villaggio: meno case e niente
    mura, ma un abitato. Lo stesso vale, in piccolo, per un sito da villaggio.
    """
    villaggio = [(lato, False) for lato in LATI_VILLAGGIO]
    if raggio >= RAGGIO_CITTA:
        return [(lato, True) for lato in LATI_CITTA] + villaggio
    return villaggio


def _distanza(forma: tuple[int, int], z: int, x: int):
    """Distanza di Chebyshev dal centro (i quadrati sono le sue curve di livello)
    e le due coordinate relative."""
    zz, xx = np.ogrid[:forma[0], :forma[1]]
    dz = np.broadcast_to(zz - z, forma)
    dx = np.broadcast_to(xx - x, forma)
    return np.maximum(np.abs(dz), np.abs(dx)), dz, dx


def _raggio_piana(lato: int, murata: bool) -> int:
    """Fin dove arriva la piana: con le mura, cinta + berma + fosso + argine."""
    if murata:
        return lato + SPESSORE_MURA + 1 + LARGHEZZA_FOSSO + 1
    return lato + 3


def _valuta(cls, h, urbano, z, x, raggio_piana, livello_mare,
            fissa=None, fissa_mura=None, murata=False,
            mare_max: float = None, salto_max: float = None) -> tuple[int, float] | None:
    """(quota di spianamento, costo) di una piana centrata in (z, x), o None.

    Il terreno si piega alla citta', ma non senza limiti: dentro il quadrato
    non ci puo' essere troppo mare (non si interra un golfo), niente vulcano,
    ne' un dislivello da montagna, ne' un'altra citta'. Il costo e' quanto
    mare e quanto dislivello bisogna sistemare: a parita' di sito, meno e'
    meglio. Si lavora su un ritaglio, perche' se ne valutano centinaia.
    """
    H, W = cls.shape
    s = raggio_piana + MARGINE_RACCORDO + 2
    if z - s < 0 or x - s < 0 or z + s >= H or x + s >= W:
        return None
    r_v = raggio_piana + DISTANZA_DAL_VULCANO
    finestra = cls[max(0, z - r_v):z + r_v + 1, max(0, x - r_v):x + r_v + 1]
    if np.isin(finestra, (VULCANO, CRATERE)).any():
        return None
    for maschera, gap in ((fissa, DISTANZA_FRA_ABITATI),
                          (fissa_mura if murata else None, DISTANZA_FRA_CITTA_MURATE)):
        if maschera is None:
            continue
        r2 = raggio_piana + gap
        z0, z1, x0, x1 = max(0, z - r2), min(H, z + r2 + 1), max(0, x - r2), min(W, x + r2 + 1)
        if maschera[z0:z1, x0:x1].any():
            return None
    fetta = (slice(z - s, z + s + 1), slice(x - s, x + s + 1))
    c, q, u = cls[fetta], h[fetta], urbano[fetta]
    D = _distanza(c.shape, s, s)[0]
    zona = D <= raggio_piana
    if np.isin(c, (VULCANO, CRATERE))[D <= raggio_piana + MARGINE_RACCORDO].any():
        return None
    if u[D <= raggio_piana + MARGINE_RACCORDO].any():
        return None
    mare = float(np.isin(c, MARINO)[zona].mean())
    if mare > (MARINO_MASSIMO if mare_max is None else mare_max):
        return None
    terra = zona & ~np.isin(c, ACQUA)
    if terra.sum() < zona.sum() * 0.5:
        return None
    quote = q[terra]
    lo, hi = np.percentile(quote, (5, 95))
    if hi - lo > (SALTO_MASSIMO if salto_max is None else salto_max):
        return None
    base = max(int(np.median(quote)), livello_mare + PROFONDITA_FOSSO + 2)
    return base, mare * 200.0 + float(hi - lo)


def _cerca_sito(cls, h, urbano, z, x, raggio_piana, livello_mare,
                fissa=None, fissa_mura=None, murata=False, mare_max=None, salto_max=None):
    """Il posto migliore per la piana vicino al sito scelto: (z, x, quota).

    I siti sono scelti vicino all'acqua, quindi spesso a ridosso di una costa
    o di un lago. Invece di scartarli si prova a spostarli: la citta' cerca il
    suo posto, poi lo spiana.
    """
    meglio = None
    for oz in range(-SPOSTAMENTO_MASSIMO, SPOSTAMENTO_MASSIMO + 1, 4):
        for ox in range(-SPOSTAMENTO_MASSIMO, SPOSTAMENTO_MASSIMO + 1, 4):
            v = _valuta(cls, h, urbano, z + oz, x + ox, raggio_piana, livello_mare,
                        fissa, fissa_mura, murata, mare_max, salto_max)
            if v is None:
                continue
            costo = v[1] + 0.15 * float(np.hypot(oz, ox))
            if meglio is None or costo < meglio[0]:
                meglio = (costo, z + oz, x + ox, v[0])
    return None if meglio is None else meglio[1:]


def _spiana(h, cls, livello, cls_originale, D, lato, murata, base, livello_mare,
            fissa=None):
    """Spiana la piana a `base`, scava il fosso, raccorda con una scarpata.

    Opera sulle copie di lavoro `h`, `cls`, `livello`. L'acqua dentro la piana
    sparisce (diventa pianura): e' il terreno che si adatta alla citta'.

    Il raccordo non e' una sfumatura a larghezza fissa: sul bordo della piana
    il terreno viene tagliato (o riempito) con una scarpata a pendenza
    costante, tanto larga quanto serve. Una sfumatura di 10 celle su un
    dislivello di 15 blocchi lasciava uno strapiombo dritto sopra il fiume o
    il mare. Nell'acqua il fondale si alza lungo la stessa scarpata, ma mai
    oltre il pelo dell'acqua: e' una sponda che scende, non terra nuova.
    `fissa` sono le celle che un'altra citta' ha gia' spianato: non si toccano.
    """
    r = _raggio_piana(lato, murata)
    dentro = D <= r
    h[dentro] = base
    cls[dentro] = PIANURA
    livello[dentro] = livello_mare

    fascia = (D > r) & (D <= r + FASCIA_RACCORDO)
    # un vulcano non si taglia: la scarpata lo lascerebbe con un fianco piatto e
    # la lava del cratere in piedi come un pilastro (visto in gioco)
    fascia &= ~np.isin(cls_originale, (VULCANO, CRATERE))
    if fissa is not None:
        fascia &= ~fissa
    zz, xx = np.nonzero(fascia)
    d = D[fascia] - r
    # una scarpata dritta come un righello non e' naturale: la pendenza e la
    # quota ondeggiano piano lungo il perimetro, con una fase diversa per ogni
    # citta' (deterministica: stesso mondo, stesse rive)
    rng = np.random.default_rng(int(base) * 131 + int(lato))
    f = rng.uniform(0.05, 0.12, 4)
    ph = rng.uniform(0, 2 * np.pi, 4)
    n1 = 0.5 * (np.sin(zz * f[0] + ph[0]) + np.sin(xx * f[1] + ph[1]))
    n2 = 0.5 * (np.sin((zz + xx) * f[2] + ph[2]) + np.sin((zz - xx) * f[3] + ph[3]))
    limite = d * PENDENZA_RACCORDO * (1.0 + 0.45 * n1) + 1.6 * np.abs(n2) * np.minimum(d, 6) / 6
    q = h[fascia].astype(np.float32)
    acqua = np.isin(cls_originale[fascia], ACQUA)
    nuova = np.clip(q, base - limite, base + limite)
    # nell'acqua si puo' solo riempire: il fondale sale lungo la scarpata e,
    # dove la scarpata supera il pelo dell'acqua, diventa riva di terra - il
    # terreno si adatta alla citta', non lascia una parete di roccia a picco
    # sull'acqua
    pelo = livello[fascia]
    terra_nuova = acqua & ((base - limite) > pelo - 1.0)
    nuova = np.where(acqua, np.maximum(q, base - limite), nuova)
    h[fascia] = np.round(nuova).astype(h.dtype)
    if terra_nuova.any():
        idx = (zz[terra_nuova], xx[terra_nuova])
        cls[idx] = PIANURA
        livello[idx] = livello_mare

    fosso = np.zeros_like(dentro)
    if murata:
        primo = lato + SPESSORE_MURA + 2
        # Un fossato scavato col righello e' un canale: largo sempre tre, sempre
        # alla stessa distanza dal muro. Qui la sponda interna (0 o 1 cella piu'
        # fuori) e la larghezza (2 o 3) ondeggiano piano lungo il perimetro; il
        # bordo esterno resta dentro la piana (D <= primo + LARGHEZZA_FOSSO).
        H, W = D.shape
        zz, xx = np.mgrid[:H, :W]
        rng_f = np.random.default_rng(int(base) * 977 + int(lato))
        f = rng_f.uniform(0.05, 0.13, 4)
        ph = rng_f.uniform(0, 2 * np.pi, 4)
        u1 = 0.5 + 0.25 * (np.sin(zz * f[0] + ph[0]) + np.sin(xx * f[1] + ph[1]))
        u2 = 0.5 + 0.25 * (np.sin(zz * f[2] + ph[2]) + np.sin(xx * f[3] + ph[3]))
        sponda = (u1 > 0.5).astype(np.int32)                  # 0 o 1
        larghezza = np.where(u2 > 0.62, LARGHEZZA_FOSSO - 1, LARGHEZZA_FOSSO)
        larghezza = np.minimum(larghezza, LARGHEZZA_FOSSO + 1 - sponda)
        fosso = (D >= primo + sponda) & (D < primo + sponda + larghezza)
        h[fosso] = base - PROFONDITA_FOSSO
        cls[fosso] = FIUME
        livello[fosso] = base - 2
    return fosso


def _strade(D, dz, dx, lato, murata, C=None) -> tuple[np.ndarray, np.ndarray]:
    """Le vie dell'abitato (rango per cella) e il prolungamento degli assi fuori
    dalla cinta, che passa il fosso e arriva alla campagna. `C` e' la distanza di
    Chebyshev dal centro (la piazza e' sempre un quadrato); senza, vale `D`."""
    if C is None:
        C = D
    rango = np.zeros(D.shape, np.uint8)
    dentro = D <= lato - 1
    asse = (np.abs(dz) <= 1) | (np.abs(dx) <= 1)
    rango[asse & dentro] = ASSE
    if murata:
        giro = ((D == lato - 3) | (D == lato - 4)) & (rango == 0)
        rango[giro] = SECONDARIA
    rango[C <= PIAZZA] = ASSE
    fuori = lato + (10 if murata else 2)
    esterno = asse & (D > lato - 1) & (D <= fuori)
    return rango, esterno


def _mura(D, dz, dx, lato, forma: Forma | None = None) -> np.ndarray:
    """La cinta: 1 muro, 2 porta, 3 torre. Quattro porte sugli assi; torri accanto
    alle porte e, secondo la forma, agli angoli e a meta' lato (quadrato e
    rettangolo) o ogni 22,5 gradi (ovale e tondo)."""
    forma = forma or forma_quadrata(lato)
    anello = (D >= lato + 1) & (D <= lato + SPESSORE_MURA)
    muro = np.zeros(D.shape, np.uint8)
    muro[anello] = 1
    az, ax = np.abs(dz), np.abs(dx)
    porta = anello & ((az <= SEMILARGHEZZA_PORTA) | (ax <= SEMILARGHEZZA_PORTA))
    lungo = np.minimum(az, ax)
    torre = (lungo > SEMILARGHEZZA_PORTA) & (lungo <= SEMILARGHEZZA_PORTA + 3)   # accanto alle porte
    if forma.tipo in ("quadrato", "rettangolo"):
        torre |= (az >= forma.lz - 1) & (ax >= forma.lx - 1)                  # angoli
        sul_z = (az - forma.lz) >= (ax - forma.lx)       # lato nord o sud
        lungo_lato = np.where(sul_z, ax, az)             # posizione lungo il lato
        meta = np.where(sul_z, forma.lx, forma.lz) // 2
        torre |= np.abs(lungo_lato - meta) <= 1                               # a meta' lato
    else:
        theta = np.arctan2(az / forma.lz, ax / forma.lx)         # 0 .. pi/2
        for k in (1, 2, 3):
            torre |= np.abs(theta - k * np.pi / 8) * lato <= 1.5
    muro[anello & torre] = 3
    muro[porta] = 2
    return muro


PIOLO_BASE = 4         # in `muro` i valori 4..7 sono una scala a pioli (verso: nord, est, sud, ovest)
SCALA_BASE = 16        # in `muro` i valori da qui in su sono gradini di una scalinata
LUNGHEZZA_SCALA = 7    # un gradino per ogni blocco di altezza della cinta
_NORMALI = ((-1, 0), (0, 1), (1, 0), (0, -1))        # nord, est, sud, ovest (dz, dx)


def _scale(centro: tuple[int, int], lato: int, forma: tuple[int, int],
           muro: np.ndarray, edifici: list,
           contorno: Forma | None = None) -> list[tuple[int, int, int]]:
    """Le scalinate che salgono dall'interno sul cammino di ronda: una per lato,
    a meta' fra la torre accanto alla porta e quella di mezzo, addossata alla
    faccia interna della cinta e in salita lungo di essa.

    Ritorna [(z, x, valore)] con `valore = SCALA_BASE + (salita * 4 + lato) * 8 +
    gradino`: la direzione in cui si sale, il lato della cinta cui e' addossata
    e il numero del gradino (0..6). Si salta una scalinata se un edificio o la
    cinta le stanno sopra.
    """
    cz, cx = centro
    fuori: list[tuple[int, int, int]] = []
    contorno = contorno or forma_quadrata(lato)
    if contorno.tipo in ("ovale", "tondo"):
        return fuori             # su una cinta curva le scalinate dritte non si addossano
    primo = SEMILARGHEZZA_PORTA + 4                      # oltre la torre accanto alla porta
    for lato_i, (nz, nx) in enumerate(_NORMALI):
        semi = contorno.lz if nz else contorno.lx        # distanza della cinta dal centro
        tang = contorno.lx if nz else contorno.lz        # semi-lunghezza di questo lato
        ultimo = tang // 2 - 2                           # prima della torre di mezzo
        if ultimo - primo + 1 < LUNGHEZZA_SCALA:
            continue
        inizio = primo + (ultimo - primo + 1 - LUNGHEZZA_SCALA) // 2
        salita = (lato_i + 1) % 4                       # si sale girando in senso orario
        tz, tx = _NORMALI[salita]
        celle = [(cz + nz * semi + tz * (inizio + k), cx + nx * semi + tx * (inizio + k), k)
                 for k in range(LUNGHEZZA_SCALA)]
        if any(not (0 <= z < forma[0] and 0 <= x < forma[1]) or muro[z, x] for z, x, _ in celle):
            continue
        sopra = False
        for e in edifici:
            x0, z0, x1, z1 = e.ingombro_tetto()
            if any(x0 - 1 <= x <= x1 + 1 and z0 - 1 <= z <= z1 + 1 for z, x, _ in celle):
                sopra = True
                break
        if sopra:
            continue
        for z, x, k in celle:
            fuori.append((z, x, SCALA_BASE + (salita * 4 + lato_i) * 8 + k))
    return fuori


ANGOLI_PIOLI = (3 * np.pi / 16, 5 * np.pi / 16)      # due per quadrante, fra una torre e l'altra


def _pioli(centro: tuple[int, int], lato: int, forma: tuple[int, int],
           muro: np.ndarray, edifici: list, contorno: Forma) -> list[tuple[int, int, int]]:
    """Sulle cinte curve le scalinate dritte non si addossano al muro: si sale con
    scale a pioli, otto in tutto (due per quadrante, fra una torre e l'altra).

    Ogni scala sta nella cella appena dentro la cinta (la berma, D == lato), ha
    sul retro una cella di muro e guarda verso l'interno. Ritorna
    [(z, x, PIOLO_BASE + verso)] con verso 0..3 = nord, est, sud, ovest (verso
    dove guarda la scala). Si salta un punto se sotto c'e' un edificio, un muro
    o altro."""
    cz, cx = centro
    H, W = forma
    fuori: list[tuple[int, int, int]] = []
    scelti: set = set()
    for sz in (1, -1):
        for sx in (1, -1):
            for theta in ANGOLI_PIOLI:
                # il punto ideale: sul raggio a quell'angolo, dove la distanza vale `lato`
                ideale = None
                for r in np.arange(lato - 8.0, lato + 3.0, 0.25):
                    # angolo nella scala della forma (come per le torri): sull'ovale
                    # il raggio a 3/16 di pi greco non e' quello della circonferenza
                    dz = int(round(sz * (r / lato) * contorno.lz * np.sin(theta)))
                    dx = int(round(sx * (r / lato) * contorno.lx * np.cos(theta)))
                    if int(_metrica(contorno, lato, np.array(dz), np.array(dx))) == lato:
                        ideale = (cz + dz, cx + dx)
                        break
                if ideale is None:
                    continue
                cella = None
                migliore = None
                for z in range(ideale[0] - 3, ideale[0] + 4):
                    for x in range(ideale[1] - 3, ideale[1] + 4):
                        if not (1 <= z < H - 1 and 1 <= x < W - 1) or muro[z, x] or (z, x) in scelti:
                            continue
                        if int(_metrica(contorno, lato, np.array(z - cz), np.array(x - cx))) != lato:
                            continue
                        # il muro dietro: un vicino che sia cinta (non torre)
                        dietro = [v for v, (vz, vx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1)))
                                  if muro[z + vz, x + vx] == 1]
                        if not dietro:
                            continue
                        d2 = (z - ideale[0]) ** 2 + (x - ideale[1]) ** 2
                        if migliore is None or d2 < migliore:
                            # la scala guarda dalla parte opposta al muro
                            migliore = d2
                            cella = (z, x, PIOLO_BASE + (dietro[0] + 2) % 4)
                if cella is None:
                    continue
                z, x, _ = cella
                if any(ex0 - 1 <= x <= ex1 + 1 and ez0 - 1 <= z <= ez1 + 1
                       for ex0, ez0, ex1, ez1 in (e.ingombro_tetto() for e in edifici)):
                    continue
                scelti.add((z, x))
                fuori.append(cella)
    return fuori


def _porte_esterne(z: int, x: int, lato: int, murata: bool,
                   forma: tuple[int, int],
                   contorno: Forma | None = None) -> list[tuple[int, int]]:
    """I quattro punti, in campagna, da cui partono le strade fra gli abitati."""
    contorno = contorno or forma_quadrata(lato)
    extra = 10 if murata else 2
    dz_, dx_ = contorno.lz + extra, contorno.lx + extra
    punti = [(z - dz_, x), (z + dz_, x), (z, x - dx_), (z, x + dx_)]
    return [(pz, px) for pz, px in punti if 0 <= pz < forma[0] and 0 <= px < forma[1]]


# --------------------------------------------------------------------------
# Lotti
# --------------------------------------------------------------------------

# Un isolato e' il quadrato che gli assi e la circonvallazione lasciano libero
# in ciascun quadrante. Lo si divide in quattro lotti, separati da due celle
# (le gronde di due case vicine non si toccano), e ogni lotto guarda la strada
# a cui sta appoggiato. Lasciare che la misura dei lotti la decida l'isolato,
# come faceva la pianta organica, dava case di ogni taglia ma quasi mai della
# taglia dei modelli scaricati (misurato: con lotti da 16x16 e gronda entrano
# 20 modelli diversi su 40, con 13x13 solo 10), quindi qui e' la pianta a
# partire dalla misura del lotto, non il contrario.
MARCIAPIEDE = 4        # dal centro dell'asse al primo lotto
LATO_PIAZZA_GRANDE = 38  # da qui in su la piazza e' larga: fontana al centro, banchi attorno


def dati_piazza(lato: int, murata: bool = False) -> tuple[int, int, int]:
    """(raggio del lastricato, sgombro dei lotti d'angolo, distanza dei banchi).

    Nelle citta' e nei villaggi grandi la piazza e' un disco di nove celle di
    raggio: la fontana (nove per nove) sta al centro, la campana accanto e i
    banchi dei mercanti in cerchio attorno, in diagonale, fuori dagli assi
    che restano liberi. Nei borghi piccoli un pozzo e poco altro.
    """
    if lato >= LATO_PIAZZA_GRANDE:
        return 9, 10, 7
    return (5 if lato >= 30 else 4), 7, 3

STACCO = 2             # fra due lotti
LOTTO_MINIMO = 11      # sotto questa misura l'isolato non si divide


def lotti(cls: np.ndarray, h: np.ndarray, livello: np.ndarray,
          centro: tuple[int, int], lato: int, murata: bool, livello_mare: int,
          rng: np.random.Generator, forma: Forma | None = None) -> list[E.Edificio]:
    """Quattro lotti per quadrante (tre in un villaggio: l'ultimo non
    toccherebbe nessuna strada), con la facciata verso la strada vicina.

    Con una forma diversa dal quadrato i quadranti hanno due misure (est-ovest e
    nord-sud) e, in un ovale o in un cerchio, un lotto il cui angolo esterno
    sbordasse dalla circonvallazione si rimpicciolisce verso il centro, o
    sparisce se non resta una casa."""
    cz, cx = centro
    forma = forma or forma_quadrata(lato)

    def misure(semi: int) -> tuple[int, tuple[int, ...]]:
        w = (semi - 9 - STACCO) // 2
        # un isolato troppo piccolo per quattro case ne regge una sola
        if w < LOTTO_MINIMO:
            return semi - 9, (0,)
        return w, (0, 1)

    wu, div_a = misure(forma.lx)         # lungo x
    wv, div_b = misure(forma.lz)         # lungo z
    curva = forma.tipo in ("ovale", "tondo")
    fuori: list[E.Edificio] = []
    for sv in (1, -1):
        for su in (1, -1):
            for b in div_b:
                for a in div_a:
                    if a == 1 and b == 1 and not murata:
                        continue
                    u0 = MARCIAPIEDE + a * (wu + STACCO)
                    v0 = MARCIAPIEDE + b * (wv + STACCO)
                    ul, vl = wu, wv
                    if a == 0 and b == 0:
                        sgombro = dati_piazza(lato, murata)[1]
                        u0, ul = sgombro, wu - (sgombro - MARCIAPIEDE)
                    if curva:
                        # l'angolo esterno deve stare dentro la circonvallazione
                        while ul >= 9 and vl >= 9 and int(_metrica(
                                forma, lato, np.array(v0 + vl - 1),
                                np.array(u0 + ul - 1))) > lato - 5:
                            if ul >= vl:
                                ul -= 1
                            else:
                                vl -= 1
                        if ul < 9 or vl < 9:
                            continue
                    if a == 1 and b == 1:
                        verso = E.EST if su > 0 else E.OVEST
                    elif b == 0:
                        verso = E.NORD if sv > 0 else E.SUD
                    else:
                        verso = E.OVEST if su > 0 else E.EST
                    x0 = cx + u0 if su > 0 else cx - u0 - ul + 1
                    z0 = cz + v0 if sv > 0 else cz - v0 - vl + 1
                    t = float(np.hypot(u0 + ul / 2, v0 + vl / 2)) / (lato * 1.25)
                    ed = _fabbrica(cls, h, livello, x0, z0, ul, vl, verso, t,
                                   livello_mare, rng, 1, _mestiere(t, False, rng))
                    if ed is not None:
                        fuori.append(ed)
    return fuori


def _fabbrica(cls, h, livello, x0, z0, larg, prof, verso, t,
              livello_mare, rng, gronda: int = 1,
              mestiere: str = "") -> E.Edificio | None:
    fetta_c = cls[z0:z0 + prof, x0:x0 + larg]
    fetta_h = h[z0:z0 + prof, x0:x0 + larg]
    if fetta_c.size == 0 or np.isin(fetta_c, ACQUA).any():
        return None
    # ...e nemmeno proprio sul filo dell'acqua. Una cella di rispetto basta:
    # il raccordo del lotto arriva piu' in la', ma le sponde sono nella
    # maschera `intoccabile` e il raccordo non le tocca. Pretenderne tre
    # faceva sparire un terzo delle case di una citta' di fiume.
    if np.isin(cls[max(0, z0 - 1):z0 + prof + 1,
                   max(0, x0 - 1):x0 + larg + 1], ACQUA).any():
        return None
    # Dislivello ROBUSTO, non massimo meno minimo: il terreno porta addosso
    # il dettaglio frattale, e un solo pixel fuori posto bocciava il lotto.
    # Il sedime viene spianato comunque; quello che va evitato e' la casa sul
    # dirupo, non la casa sul dosso.
    lo, hi = np.percentile(fetta_h, (10, 90))
    dislivello = float(hi - lo)
    if dislivello > 7:
        return None
    base = int(np.median(fetta_h))
    if base <= livello_mare:
        return None
    classe = int(np.bincount(fetta_c.ravel()).argmax())
    if classe in (MONTAGNA, NEVE) and dislivello > 5:
        return None
    # In centro si costruisce alto, in periferia no. Il terzo piano e' raro:
    # in una citta' medievale e' un'eccezione, e messo a meta' delle case del
    # centro trasformava i vicoli in trincee.
    piani = 3 if t < 0.18 and rng.random() < 0.3 else (2 if t < 0.55 else 1)
    return E.Edificio(
        x=x0, z=z0, larghezza=larg, profondita=prof, base=base,
        piani=piani, porta=verso,
        stile=E.PALETTE_PER_CLASSE.get(classe, "prato"),
        palafitta=False, fondale=base, gronda=gronda, mestiere=mestiere,
        seme=int(rng.integers(0, 2 ** 31 - 1)),
    )


# --------------------------------------------------------------------------
# Mercato
# --------------------------------------------------------------------------

MERCI = ("frutta", "carne", "pesce", "verdura")


def mercato(centro: tuple[int, int], lato: int, murata: bool, h: np.ndarray,
            rng: np.random.Generator) -> list[E.Banco]:
    """Banchi sulla piazza, agli angoli, rivolti verso il centro.

    Il banco sta SUL selciato (la piazza e' il posto del mercato) e agli
    angoli, fuori dagli assi, che devono restare liberi. Una citta' ne ha
    quattro, un villaggio due (sulla diagonale), un borgo piccolo nessuno.
    """
    if lato < 30:
        return []
    cz, cx = centro
    angoli = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
    quanti = 4 if murata else 2
    fuori: list[E.Banco] = []
    for k, (sv, su) in enumerate(angoli[:quanti]):
        d = dati_piazza(lato, murata)[2]
        bz, bx = cz + sv * d - 1, cx + su * d - 1            # 3x3 centrato a (+-d, +-d)
        verso = E.OVEST if su > 0 else E.EST
        fuori.append(E.Banco(x=bx, z=bz, base=int(h[bz:bz + 3, bx:bx + 3].min()),
                             verso=verso, merce=MERCI[k % len(MERCI)],
                             seme=int(rng.integers(0, 2 ** 31 - 1))))
    return fuori


def scarpate(h: np.ndarray, urbano: np.ndarray, salto: int = 2) -> np.ndarray:
    """Le celle che formano la fronte di un gradone.

    Servono al motore per vestirle di pietra invece che di erba: un muro di
    sostegno si vede, un taglio di terra nuda sembra un difetto.
    """
    hh = h.astype(np.int32)
    piu_basso = np.minimum.reduce([
        np.roll(hh, 1, 0), np.roll(hh, -1, 0),
        np.roll(hh, 1, 1), np.roll(hh, -1, 1),
    ])
    return urbano & ((hh - piu_basso) >= salto)


def pianifica(
    cls: np.ndarray,
    altezze: np.ndarray,
    livello: np.ndarray,
    siti: list[tuple[int, int, int]],
    livello_mare: int = 62,
    seed: int = 0,
    forme: str = "quadrato",
) -> tuple[list[E.Edificio], np.ndarray, np.ndarray, np.ndarray,
           list[Citta], list[E.Banco], np.ndarray, np.ndarray, np.ndarray]:
    """Progetta gli insediamenti.

    Ritorna (edifici, altezze spianate, vie, mura, citta', banchi, urbano,
    classi, livello dell'acqua). Le ultime due sono le COPIE della mappa dopo
    che le citta' l'hanno cambiata: dove una pianta passa sopra un fiume il
    fiume non c'e' piu', e dove c'e' il fosso c'e' acqua che sulla mappa non
    c'era. Chi disegna il mondo deve leggere quelle, non le originali.

    Un sito che non si presta (troppo mare, vulcano, terreno da montagna,
    un'altra citta' accanto) o che non produce abbastanza lotti viene
    scartato per intero: un abitato di due case non e' un abitato.
    """
    h = altezze.astype(np.int32).copy()
    cls = cls.copy()
    livello = livello.astype(np.float32).copy()
    H, W = cls.shape
    vie = np.zeros((H, W), np.uint8)
    muro = np.zeros((H, W), np.uint8)
    urbano = np.zeros((H, W), bool)
    fissa = np.zeros((H, W), bool)
    fissa_mura = np.zeros((H, W), bool)
    fuori: list[E.Edificio] = []
    citta: list[Citta] = []
    banchi: list[E.Banco] = []

    for n, sito in enumerate(siti):
        sz, sx, raggio = sito[:3]
        # un sito puo' avere in coda le sue tolleranze: (mare ammesso, dislivello ammesso) - le citta'
        # VERE (capitali sulla costa o in valle) ne hanno di piu' di un villaggio scelto a caso
        mare_max = sito[3] if len(sito) > 3 else None
        salto_max = sito[4] if len(sito) > 4 else None
        rng = np.random.default_rng(seed * 977 + n)
        for lato, murata in piante_possibili(raggio):
            posto = _cerca_sito(cls, h, urbano, sz, sx, _raggio_piana(lato, murata),
                                livello_mare, fissa, fissa_mura, murata, mare_max, salto_max)
            if posto is not None:
                break
        else:
            continue
        sz, sx, base = posto
        C, dz, dx = _distanza((H, W), sz, sx)
        # la forma si sceglie per prima e non cambia piu'; se non produce
        # abbastanza lotti si ripiega sul quadrato, che sul sito ci sta di sicuro
        # `forme` puo' essere un elenco ("tondo,ovale,rettangolo"): le citta' murate
        # lo percorrono in ordine, per vedere tutte le forme in una mappa sola
        scelta = forme
        if "," in forme:
            elenco = forme.split(",")
            scelta = elenco[sum(1 for c_ in citta if c_.murata) % len(elenco)] if murata else "quadrato"
        candidate = [scegli_forma(scelta, lato, murata, rng)]
        if candidate[0].tipo != "quadrato":
            candidate.append(forma_quadrata(lato))
        for contorno in candidate:
            D = _metrica(contorno, lato, dz, dx)
            # tutto si prova su copie: se il sito non da' abbastanza lotti non
            # deve lasciare traccia (una piana spianata e un fosso senza citta')
            h_t, cls_t, liv_t = h.copy(), cls.copy(), livello.copy()
            _spiana(h_t, cls_t, liv_t, cls, D, lato, murata, base, livello_mare, fissa)
            rango, esterno = _strade(D, dz, dx, lato, murata, C)
            ed = lotti(cls_t, h_t, liv_t, (sz, sx), lato, murata, livello_mare, rng, contorno)
            if len(ed) >= (MIN_LOTTI_CITTA if murata else MIN_LOTTI_VILLAGGIO):
                break
        else:
            continue
        for e in ed:
            e.villaggio = n

        b = mercato((sz, sx), lato, murata, h_t, rng)

        h, cls, livello = h_t, cls_t, liv_t
        vie = np.maximum(vie, rango)
        vie[esterno] = ASSE
        urbano |= D <= lato
        fissa |= D <= _raggio_piana(lato, murata)
        if murata:
            fissa_mura |= D <= _raggio_piana(lato, murata)
        fuori.extend(ed)
        banchi.extend(b)
        if murata:
            m = _mura(D, dz, dx, lato, contorno)
            muro[m > 0] = m[m > 0]
            for z, x, v in _scale((sz, sx), lato, (H, W), muro, ed, contorno):
                muro[z, x] = v
            if contorno.tipo in ("ovale", "tondo"):
                for z, x, v in _pioli((sz, sx), lato, (H, W), muro, ed, contorno):
                    muro[z, x] = v
        citta.append(Citta(
            z=sz, x=sx, raggio=_raggio_piana(lato, murata), piazza=(sz, sx),
            porte=_porte_esterne(sz, sx, lato, murata, (H, W), contorno),
            murata=murata, edifici=len(ed), lato=lato,
            forma=contorno.tipo, lz=contorno.lz, lx=contorno.lx))

    return fuori, h, vie, muro, citta, banchi, urbano, cls, livello


def statistiche(edifici: list[E.Edificio], vie: np.ndarray, muro: np.ndarray,
                citta: list[Citta], banchi: list | None = None) -> dict:
    return {
        "insediamenti": len(citta),
        "citta": sum(1 for c in citta if c.e_citta),
        "edifici": len(edifici),
        "assi": int((vie == ASSE).sum()),
        "secondarie": int((vie == SECONDARIA).sum()),
        "vicoli": int((vie == VICOLO).sum()),
        "mura": int((muro == 1).sum()),
        "porte": int((muro == 2).sum()),
        "torri": int((muro == 3).sum()),
        "botteghe": sum(1 for e in edifici if e.mestiere),
        "banchi": len(banchi or []),
    }


def conteggio_mestieri(edifici: list[E.Edificio]) -> dict[str, int]:
    fuori: dict[str, int] = {}
    for e in edifici:
        if e.mestiere:
            fuori[e.mestiere] = fuori.get(e.mestiere, 0) + 1
    return dict(sorted(fuori.items(), key=lambda t: -t[1]))
