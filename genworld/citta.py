"""Citta': pianta organica, isolati, lotti sulla strada.

Fino a qui un "villaggio" era una griglia sfalsata di case buttate dentro un
cerchio. Dall'alto si riconosce subito: le case non guardano niente, non c'e'
un dentro e un fuori, e lo spazio fra loro non e' uno spazio, e' l'avanzo.

Una citta' e' il contrario: prima esiste il **vuoto** - piazza, strade,
vicoli - e le case vengono dopo, appoggiate a quel vuoto. Quindi qui si
disegna prima la pianta e solo alla fine si mettono gli edifici, ognuno con
la facciata verso la strada che lo serve.

Impianto **organico medievale**, non a griglia: le strade non sono rette
ortogonali ma i collegamenti fra i luoghi. Si spargono dei punti nell'area
urbana, si triangolano, e le strade sono gli spigoli di quella
triangolazione. Viene fuori una maglia irregolare, piena di isolati chiusi di
forma diversa, che e' esattamente il modo in cui cresce un borgo: prima i
sentieri fra i posti dove sta la gente, poi le case lungo i sentieri.

Gerarchia in tre livelli, dedotta e non dichiarata: sono **assi** gli spigoli
che stanno sul cammino piu' breve fra una porta e la piazza, **secondarie**
quelle lunghe, **vicoli** tutto il resto.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from scipy.spatial import Delaunay

from . import edifici as E
from .mappa import (ACQUA, FIUME, FORESTA, MARINO, MONTAGNA, NEVE, PIANURA,
                    PRATERIA, SPIAGGIA)

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

    @property
    def e_citta(self) -> bool:
        return self.raggio >= RAGGIO_CITTA


# --------------------------------------------------------------------------
# Area urbana
# --------------------------------------------------------------------------

def contorno(cls: np.ndarray, altezze: np.ndarray, z: int, x: int, raggio: int,
             livello_mare: int = 62, seed: int = 0) -> np.ndarray:
    """L'area che la citta' occupa: non un cerchio.

    Il raggio e' deformato da tre onde lente - una citta' col perimetro
    circolare si legge come un timbro - e poi tagliato su quello che il
    terreno concede: niente acqua, niente pendii da capre.
    """
    H, W = cls.shape
    zz, xx = np.ogrid[:H, :W]
    d = np.sqrt((zz - z) ** 2 + (xx - x) ** 2)
    ang = np.arctan2(zz - z, xx - x)
    rng = np.random.default_rng(seed)
    deforma = sum(a * np.sin(f * ang + float(rng.random() * 6.28))
                  for f, a in ((2, 0.12), (3, 0.08), (5, 0.05)))
    dentro = d <= raggio * (1.0 + deforma)

    pend = np.hypot(*np.gradient(gaussian_filter(altezze.astype(np.float32), 1.5)))
    costruibile = (np.isin(cls, ABITABILI) & (altezze > livello_mare)
                   & (pend < 1.6))
    return dentro & costruibile


def _piazza(area: np.ndarray, altezze: np.ndarray, z: int, x: int) -> tuple[int, int]:
    """Il posto piu' piano e piu' interno dell'area.

    La piazza non sta al centro geometrico ma dove si puo' stare: su un sito
    tagliato da un fiume il centro geometrico puo' cadere in acqua.
    """
    if not area.any():
        return z, x
    dist = distance_transform_edt(area)
    pend = np.hypot(*np.gradient(gaussian_filter(altezze.astype(np.float32), 2)))
    punteggio = np.where(area, dist - pend * 6.0, -1e9)
    i = int(np.argmax(punteggio))
    return divmod(i, area.shape[1])


# --------------------------------------------------------------------------
# Rete viaria
# --------------------------------------------------------------------------

def _semi(area: np.ndarray, piazza: tuple[int, int], passo: int,
          rng: np.random.Generator) -> np.ndarray:
    """Punti sparsi nell'area, a griglia sfalsata. La piazza e' sempre uno.

    La griglia sfalsata e' la stessa idea della semina degli alberi: una
    griglia pura darebbe isolati tutti uguali, il caso puro darebbe grumi e
    vuoti.
    """
    zs, xs = np.nonzero(area)
    if zs.size == 0:
        return np.zeros((0, 2), int)
    z0, z1, x0, x1 = zs.min(), zs.max(), xs.min(), xs.max()
    punti = [piazza]
    for gz in range(z0, z1 + 1, passo):
        for gx in range(x0, x1 + 1, passo):
            pz = gz + int(rng.integers(-passo // 3, passo // 3 + 1))
            px = gx + int(rng.integers(-passo // 3, passo // 3 + 1))
            if 0 <= pz < area.shape[0] and 0 <= px < area.shape[1] and area[pz, px]:
                if abs(pz - piazza[0]) + abs(px - piazza[1]) > passo // 2:
                    punti.append((pz, px))
    return np.array(punti, int)


def _spigoli(punti: np.ndarray, lunghezza_max: float) -> list[tuple[int, int]]:
    """Spigoli della triangolazione di Delaunay, senza quelli troppo lunghi.

    Delaunay collega ogni punto ai suoi vicini naturali: ne esce una maglia
    planare con isolati chiusi di forma irregolare. Gli spigoli lunghi sono
    quelli che scavalcano il bordo concavo dell'area e vanno tolti,
    altrimenti la citta' ha strade che passano fuori da se stessa.
    """
    if len(punti) < 4:
        return [(i, j) for i in range(len(punti)) for j in range(i + 1, len(punti))]
    tri = Delaunay(punti.astype(float))
    fuori = set()
    for a, b, c in tri.simplices:
        for i, j in ((a, b), (b, c), (c, a)):
            i, j = (i, j) if i < j else (j, i)
            if np.hypot(*(punti[i] - punti[j])) <= lunghezza_max:
                fuori.add((int(i), int(j)))
    return sorted(fuori)


def _cammino(punti: np.ndarray, spigoli: list[tuple[int, int]],
             da: int, a: int) -> list[tuple[int, int]]:
    """Dijkstra sul grafo delle strade: quali spigoli si percorrono."""
    vicini: dict[int, list[tuple[int, float]]] = {}
    for i, j in spigoli:
        d = float(np.hypot(*(punti[i] - punti[j])))
        vicini.setdefault(i, []).append((j, d))
        vicini.setdefault(j, []).append((i, d))
    dist = {da: 0.0}
    prima: dict[int, int] = {}
    coda = [(0.0, da)]
    visti = set()
    while coda:
        d, n = heapq.heappop(coda)
        if n in visti:
            continue
        visti.add(n)
        if n == a:
            break
        for m, w in vicini.get(n, ()):
            nd = d + w
            if nd < dist.get(m, 1e18):
                dist[m] = nd
                prima[m] = n
                heapq.heappush(coda, (nd, m))
    if a not in visti:
        return []
    percorso = [a]
    while percorso[-1] != da:
        if percorso[-1] not in prima:
            return []
        percorso.append(prima[percorso[-1]])
    percorso.reverse()
    return [(percorso[k], percorso[k + 1]) for k in range(len(percorso) - 1)]


def _linea(a: tuple[int, int], b: tuple[int, int],
           curva: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Segmento con una piega. Una strada medievale non e' un righello.

    Si sposta il punto di mezzo di lato e si passa per quello: due tratti
    invece di uno, e la strada si incurva quel tanto che basta.
    """
    (z0, x0), (z1, x1) = a, b
    lung = float(np.hypot(z1 - z0, x1 - x0))
    if lung < 2:
        return [a, b]
    off = (rng.random() - 0.5) * 2.0 * curva * lung
    mz = (z0 + z1) / 2 - (x1 - x0) / lung * off
    mx = (x0 + x1) / 2 + (z1 - z0) / lung * off
    fuori: list[tuple[int, int]] = []
    for (pa, pb) in (((z0, x0), (mz, mx)), ((mz, mx), (z1, x1))):
        n = max(2, int(np.hypot(pb[0] - pa[0], pb[1] - pa[1])) + 1)
        for t in np.linspace(0, 1, n):
            fuori.append((int(round(pa[0] + (pb[0] - pa[0]) * t)),
                          int(round(pa[1] + (pb[1] - pa[1]) * t))))
    return fuori


def rete(area: np.ndarray, piazza: tuple[int, int], raggio: int,
         seed: int = 0) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Traccia le vie. Ritorna (rango per cella, porte, punti del grafo)."""
    H, W = area.shape
    rango = np.zeros((H, W), np.uint8)
    rng = np.random.default_rng(seed)

    # Il passo decide la dimensione degli ISOLATI, ed e' il numero piu'
    # delicato di tutto il modulo. Con passo 11 su raggio 33 le vie si
    # mangiavano il 38% dell'area e nessun lotto ci stava piu' dentro: zero
    # case costruite su una pianta perfetta.
    # Alzato da 11-20 a 14-24 quando sono arrivate le case da template: con
    # isolati da undici celle, fra la sede stradale e le due celle di
    # distacco, al lotto ne restavano tre o quattro di profondita'. Il
    # generatore parametrico ci costruiva lo stesso - prende le misure dal
    # lotto - e venivano casotti di due per due; un template, che le misure
    # ce le ha sue, non ci entrava mai.
    passo = int(np.clip(raggio // 2, 14, 24))
    punti = _semi(area, piazza, passo, rng)
    if len(punti) < 3:
        return rango, [], punti
    spigoli = _spigoli(punti, lunghezza_max=passo * 1.9)
    if not spigoli:
        return rango, [], punti

    # Le porte sono i punti piu' esterni, sparsi attorno: si prende il punto
    # piu' lontano dalla piazza in ciascuno di N settori angolari, cosi' non
    # escono tutte dallo stesso lato.
    dz = punti[:, 0] - piazza[0]
    dx = punti[:, 1] - piazza[1]
    dist = np.hypot(dz, dx)
    ang = np.arctan2(dz, dx)
    n_porte = 4 if raggio >= RAGGIO_CITTA else 3
    porte_i: list[int] = []
    for k in range(n_porte):
        lo = -np.pi + 2 * np.pi * k / n_porte
        hi = -np.pi + 2 * np.pi * (k + 1) / n_porte
        nel_settore = np.nonzero((ang >= lo) & (ang < hi) & (dist > raggio * 0.55))[0]
        if nel_settore.size:
            porte_i.append(int(nel_settore[np.argmax(dist[nel_settore])]))

    # rango: prima tutto vicolo, poi si promuove
    rango_spigolo = {s: VICOLO for s in spigoli}
    for i, j in spigoli:
        if np.hypot(*(punti[i] - punti[j])) > passo * 1.55:
            rango_spigolo[(i, j)] = SECONDARIA
    for p in porte_i:
        for i, j in _cammino(punti, spigoli, p, 0):     # 0 = piazza
            chiave = (i, j) if (i, j) in rango_spigolo else (j, i)
            if chiave in rango_spigolo:
                rango_spigolo[chiave] = ASSE

    for (i, j), r in rango_spigolo.items():
        curva = 0.10 if r == ASSE else 0.18
        for z, x in _linea(tuple(punti[i]), tuple(punti[j]), curva, rng):
            if 0 <= z < H and 0 <= x < W:
                rango[z, x] = max(rango[z, x], r)

    # allarga secondo il rango
    for r in (SECONDARIA, ASSE):
        largo = binary_dilation(rango >= r, iterations=LARGHEZZA[r] - 1)
        rango[largo & (rango == 0)] = r

    # la piazza: uno slargo, non un incrocio
    lato = 3 if raggio < RAGGIO_CITTA else 5
    z0 = max(0, piazza[0] - lato); z1 = min(H, piazza[0] + lato + 1)
    x0 = max(0, piazza[1] - lato); x1 = min(W, piazza[1] + lato + 1)
    rango[z0:z1, x0:x1] = ASSE

    rango[~area] = 0
    return rango, [tuple(punti[p]) for p in porte_i], punti


# --------------------------------------------------------------------------
# Lotti
# --------------------------------------------------------------------------

def _verso_strada(vie: np.ndarray, z: int, x: int, portata: int = 4) -> int | None:
    """Da che parte sta la strada piu' vicina, in quattro direzioni."""
    H, W = vie.shape
    for d in range(1, portata + 1):
        for verso, (dz, dx) in ((E.NORD, (-1, 0)), (E.SUD, (1, 0)),
                                (E.OVEST, (0, -1)), (E.EST, (0, 1))):
            pz, px = z + dz * d, x + dx * d
            if 0 <= pz < H and 0 <= px < W and vie[pz, px]:
                return verso
    return None


PASSO = {E.NORD: (1, 0), E.SUD: (-1, 0), E.OVEST: (0, 1), E.EST: (0, -1)}
LATO = {E.NORD: (0, 1), E.SUD: (0, 1), E.OVEST: (1, 0), E.EST: (1, 0)}


# Un sedime di quattro per quattro non e' una casa, e' un ripostiglio: dentro
# ci sta una stanza di due per due. Serviva anche a un altro scopo - riempire
# gli angoli degli isolati - e il prezzo era un paese di casotti. Sei e' il
# minimo perche' ci stia una stanza di quattro per quattro, che e' anche la
# misura sotto la quale nessun template di casa entra.
LATO_MINIMO = 6

# Il tetto massimo che un lotto puo' raggiungere (poi `_rettangolo` cerca il
# piu' grande che ci sta davvero, fino a qui - un isolato piccolo resta
# piccolo comunque). Con 13/9 quasi nessuno dei template scaricati
# (`templates/strutture`) entrava - la maggior parte e' piu' larga o piu'
# profonda - e la scelta (allora `template.scegli()`) finiva quasi sempre sull'unico modello
# che ci stava: non un bug di scelta, un tetto troppo basso per il catalogo.
#
# Misurato su Arda vera (832x832, 39 modelli contando le rotazioni): con
# 13/9 sceglieva sempre lo stesso template su 36 case (1 modello distinto).
# Con questi valori diventano 6 modelli distinti su 28 case - le case
# restano comunque per lo piu' piccole (la maggioranza degli isolati e'
# stretta di suo, non solo per via del tetto) ma quelle nei blocchi piu'
# larghi ora possono davvero diventare piu' grandi, invece di restare
# tagliate allo stesso modello minuscolo. Alzare ulteriormente il tetto
# (provato fino a 22/18) non migliora la varieta' - anzi la fa scendere,
# perche' toglie spazio a piu' lotti di quanti template in piu' fa entrare -
# quindi non basta da solo a risolvere la ripetizione: il resto e' la forma
# stessa degli isolati, non questo tetto.
#
# La PROFONDITA' resta comunque piu' bassa del fronte apposta: un isolato ha
# due file di case, una per lato, schiena contro schiena, e un lotto troppo
# profondo mangia l'isolato intero lasciando la seconda fila senza spazio
# (misurato anni fa: a profondita' 13, 1.956 lotti scartati su 2.100. Qui
# si arriva a 15 comunque, ma solo perche' e' anche il punto sopra al quale
# la varieta' smette di migliorare - vedi sopra).
FRONTE_MAX_CENTRO = 18
FRONTE_MAX_PERIFERIA = 15
FONDO_MAX = 15


def _rettangolo(occupato: np.ndarray, area: np.ndarray, cz: int, cx: int,
                verso: int, fronte_max: int, fondo_max: int,
                ) -> tuple[int, int, int, int] | None:
    """Il piu' grande lotto che ci sta, a partire da una cella sul fronte.

    Si CERCA la misura invece di proporla. Proponendo un rettangolo a caso e
    scartandolo se non entra si buttavano via 288 posizioni su 830 e restavano
    otto case: gli isolati di una pianta organica non sono mai della misura
    che ti aspetti. Una casa a schiera fa esattamente questo - prende quello
    che l'isolato le lascia - ed e' il motivo per cui nei centri storici i
    lotti sono stretti, lunghi e tutti diversi.
    """
    H, W = occupato.shape
    dz, dx = PASSO[verso]        # via dalla strada
    pz, px = LATO[verso]         # lungo la strada

    def libera(d: int, off: int) -> bool:
        z, x = cz + dz * d + pz * off, cx + dx * d + px * off
        return (0 <= z < H and 0 <= x < W and area[z, x] and not occupato[z, x])

    fondo = 0
    while fondo < fondo_max and libera(fondo, 0):
        fondo += 1
    if fondo < LATO_MINIMO:
        return None

    def colonna(off: int) -> bool:
        return all(libera(d, off) for d in range(fondo))

    sinistra = destra = 0
    while sinistra + destra + 1 < fronte_max:
        cresciuto = False
        if sinistra <= destra and colonna(-(sinistra + 1)):
            sinistra += 1
            cresciuto = True
        elif colonna(destra + 1):
            destra += 1
            cresciuto = True
        elif colonna(-(sinistra + 1)):
            sinistra += 1
            cresciuto = True
        if not cresciuto:
            break
    if sinistra + destra + 1 < LATO_MINIMO:
        return None

    zs, xs = [], []
    for d in (0, fondo - 1):
        for off in (-sinistra, destra):
            zs.append(cz + dz * d + pz * off)
            xs.append(cx + dx * d + px * off)
    z0, z1 = min(zs), max(zs)
    x0, x1 = min(xs), max(xs)
    return x0, z0, x1 - x0 + 1, z1 - z0 + 1


def lotti(area: np.ndarray, vie: np.ndarray, cls: np.ndarray, h: np.ndarray,
          livello: np.ndarray, piazza: tuple[int, int], raggio: int,
          livello_mare: int, rng: np.random.Generator) -> list[E.Edificio]:
    """Riempie gli isolati con case che guardano la strada.

    Non si cercano le facce del grafo: si scorrono le celle che confinano con
    una strada e si prova a posarci un lotto, che si allunga lungo la
    carreggiata e si approfondisce fin dove l'isolato lo lascia fare. Il
    risultato e' lo stesso - case in fila sul fronte strada - ma funziona
    anche su isolati di forma qualunque, che e' tutto il punto di una pianta
    organica.

    La densita' cala verso la periferia: al centro le case si toccano e hanno
    due o tre piani, fuori sono staccate e basse. Senza questo gradiente si
    ottiene un quartiere residenziale caduto dal cielo.
    """
    H, W = area.shape
    dist_via = distance_transform_edt(~vie)
    occupato = binary_dilation(vie, iterations=1).copy()
    fuori: list[E.Edificio] = []

    candidate = np.nonzero(area & (dist_via >= 2) & (dist_via <= 3))
    d_centro = np.hypot(candidate[0] - piazza[0], candidate[1] - piazza[1])
    ordine = np.argsort(d_centro)

    for k in ordine:
        cz, cx = int(candidate[0][k]), int(candidate[1][k])
        if occupato[cz, cx]:
            continue
        t = float(d_centro[k]) / max(raggio, 1)         # 0 centro, 1 bordo
        if rng.random() > 1.35 - 0.45 * t:             # piu' rado in periferia
            continue
        verso = _verso_strada(vie, cz, cx)
        if verso is None:
            continue

        # il fronte e' quello che conta, la profondita' e' quel che avanza -
        # vedi `FRONTE_MAX_CENTRO`/`FRONTE_MAX_PERIFERIA`/`FONDO_MAX` sopra
        # per il perche' di questi valori.
        fronte_max = FRONTE_MAX_PERIFERIA if t > 0.6 else FRONTE_MAX_CENTRO
        misura = _rettangolo(occupato, area, cz, cx, verso, fronte_max, FONDO_MAX)
        if misura is None:
            continue
        x0, z0, larg, prof = misura

        # QUI STAVA UN DIFETTO che si vedeva solo dall'alto: i tetti si
        # sovrapponevano. Il sedime di due case adiacenti non si tocca mai -
        # c'e' un test che lo controlla - ma il tetto sporge di un blocco
        # oltre i muri, e due gronde in un vicolo stretto finiscono nella
        # stessa cella. L'ingombro vero di una casa vista dall'alto non e' il
        # sedime, e' il sedime PIU' la gronda.
        #
        # In centro le case restano attaccate e rinunciano alla gronda, che e'
        # esattamente quello che succede in una schiera; in periferia tengono
        # la gronda e si lasciano due blocchi, cosi' i tetti non si toccano.
        #
        # Le soglie sono state ritarate guardando una citta' dall'alto: con
        # la schiera fino a meta' raggio e tre piani in tutto il centro, i
        # vicoli venivano larghi un blocco e profondi dodici. Non e' un
        # centro storico, e' un pozzo di ventilazione. La schiera resta, ma
        # solo nel cuore vero, e subito fuori le case si staccano.
        if t < 0.35:
            stacco, gronda = 0, 0
        else:
            stacco, gronda = 1, 1
        if not _bordo_mappa_libero(occupato, x0, z0, larg, prof):
            continue
        # La gronda fa parte della sagoma, non e' un dettaglio del tetto. Se
        # non c'e' posto anche per quella pero' non si butta via la casa: si
        # butta via la gronda. Pretendere lo spazio faceva scendere Arda da 91
        # edifici a 59 - un terzo del paese demolito per un blocco di
        # sporgenza.
        if gronda and occupato[max(0, z0 - gronda):z0 + prof + gronda,
                               max(0, x0 - gronda):x0 + larg + gronda].any():
            gronda = 0
        # Pescatore solo se l'acqua e' DAVVERO a due passi e siamo al bordo:
        # con sei celle di margine, in una citta' costiera facevano i
        # pescatori tutti, quindici su quarantuno.
        vicino_acqua = bool(np.isin(
            cls[max(0, z0 - 2):z0 + prof + 2, max(0, x0 - 2):x0 + larg + 2],
            ACQUA).any())
        ed = _fabbrica(cls, h, livello, x0, z0, larg, prof, verso, t,
                       livello_mare, rng, gronda,
                       _mestiere(t, vicino_acqua and t > 0.5
                                 and rng.random() < 0.5, rng))
        if ed is None:
            continue
        bordo = gronda + stacco
        occupato[max(0, z0 - bordo):z0 + prof + bordo,
                 max(0, x0 - bordo):x0 + larg + bordo] = True
        fuori.append(ed)
    return fuori


def _bordo_mappa_libero(occupato, x0, z0, larg, prof) -> bool:
    H, W = occupato.shape
    return x0 >= 2 and z0 >= 2 and x0 + larg < W - 2 and z0 + prof < H - 2


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


def mercato(area: np.ndarray, vie: np.ndarray, occupato: np.ndarray,
            h: np.ndarray, piazza: tuple[int, int], raggio: int,
            rng: np.random.Generator) -> list[E.Banco]:
    """Banchi attorno alla piazza.

    Il mercato non si mette al centro della piazza - la piazza serve a stare -
    ma sul suo bordo, dove passa la gente. E guarda verso l'interno, perche'
    un banco che da' le spalle alla piazza non vende niente.
    """
    if raggio < 24:
        return []                      # un borgo non ha un mercato
    pz, px = piazza
    # Il banco sta SULLA piazza, sul suo bordo. La prima versione lo cercava
    # su terreno libero e non ne trovava mai nessuno: la piazza e' selciato,
    # quindi per il codice era "strada occupata". Un mercato che aspetta un
    # prato non si apre mai.
    lato = 3 if raggio < RAGGIO_CITTA else 5
    quanti = 3 if raggio < RAGGIO_CITTA else 5
    fuori: list[E.Banco] = []
    tentativi = 0
    while len(fuori) < quanti and tentativi < 80:
        tentativi += 1
        ang = rng.random() * 2 * np.pi
        d = lato - 1 + rng.random() * 2.5
        bz = int(round(pz + d * np.sin(ang))) - 1
        bx = int(round(px + d * np.cos(ang))) - 1
        if bz < 1 or bx < 1 or bz + 3 >= area.shape[0] or bx + 3 >= area.shape[1]:
            continue
        fetta_occ = occupato[bz - 1:bz + 4, bx - 1:bx + 4]
        if fetta_occ.any() or not area[bz:bz + 3, bx:bx + 3].all():
            continue
        quote = h[bz:bz + 3, bx:bx + 3]
        if int(quote.max() - quote.min()) > 1:
            continue
        # il banco guarda la piazza
        if abs(bz + 1 - pz) > abs(bx + 1 - px):
            verso = E.NORD if bz + 1 > pz else E.SUD
        else:
            verso = E.OVEST if bx + 1 > px else E.EST
        fuori.append(E.Banco(x=bx, z=bz, base=int(np.median(quote)) + 1,
                             verso=verso,
                             merce=MERCI[len(fuori) % len(MERCI)],
                             seme=int(rng.integers(0, 2 ** 31 - 1))))
        occupato[bz - 1:bz + 4, bx - 1:bx + 4] = True
    return fuori


# --------------------------------------------------------------------------
# Mura
# --------------------------------------------------------------------------

def mura(area: np.ndarray, vie: np.ndarray, sedimi: np.ndarray,
         cls: np.ndarray, porte_grafo: list[tuple[int, int]],
         altezze: np.ndarray | None = None, riva: int = 4,
         pendenza_massima: float = 1.6,
         ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Cinta attorno all'abitato. Ritorna (muro, porta, punti delle porte,
    torri). `porta` include sia i varchi delle strade sia, quando un fiume
    attraversa l'abitato, il varco sul fiume - vedi piu' sotto.

    Non segue il contorno dell'area - che comprende anche i campi - ma
    l'inviluppo di cio' che e' costruito: una cinta che gira larghissima
    attorno a quattro case non e' una cinta, e' una recinzione.

    Le porte non si piazzano a occhio: sono il punto in cui la cinta incontra
    un asse. Il primo tentativo cercava le celle di muro gia' toccate da una
    strada e ne trovava zero - le strade finiscono dentro l'abitato, la cinta
    gira fuori - quindi ora e' la strada che viene prolungata fino al muro,
    che e' anche quello che succede davvero: la porta esiste perche' ci passa
    la via, non viceversa.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes

    vuoto = np.zeros_like(sedimi)
    costruito = sedimi | (vie >= SECONDARIA)
    if costruito.sum() < 300:
        return vuoto, vuoto.copy(), [], vuoto.copy()
    dentro = binary_closing(binary_dilation(costruito, iterations=3),
                            iterations=3, border_value=0)
    dentro = binary_fill_holes(dentro)
    dentro = binary_dilation(dentro, iterations=1)
    # TRE celle di spessore, non una e nemmeno due. Una cinta di una cella
    # sola e' connessa solo in diagonale: sul terreno si vede una fila di
    # cubi staccati che si toccano per lo spigolo, ci si passa in mezzo, e da
    # lontano sembra muratura caduta in giro a caso invece che una cinta.
    # Due era meglio ma ancora sottile per una muraglia vera - vista in
    # gioco si legge come un cordolo, non come qualcosa che tiene fuori un
    # assedio. Tre e' il minimo che si legge come un muro con del corpo.
    anello = binary_dilation(dentro, iterations=3) & ~dentro
    # il filo dell'anello dove un fiume lo attraversa - serve piu' sotto per
    # aprire un varco vero (con architrave) invece di un buco senza spiegazione
    varco_fiume = anello & (cls == FIUME)
    anello &= ~np.isin(cls, ACQUA)          # una cinta non cammina sull'acqua

    # LA CINTA SI FERMA SULLA RIVA - ma solo quella del MARE, non di un fiume.
    #
    # Negli screenshot le mura scendevano fino in acqua e continuavano dentro
    # il mare, a gradoni, come muratura buttata giu' dalla scogliera. Togliere
    # le sole celle d'acqua non basta: il guaio e' l'ultimo tratto, quello che
    # corre sulla battigia e sul fianco della falesia, dove un muro alto sette
    # blocchi viene su a pezzi e non sta in piedi ne' con gli occhi ne' con la
    # testa. Una citta' di mare, del resto, le mura dalla parte del mare non le
    # ha mai avute: il mare e' gia' la difesa, e dove si apre il porto il muro
    # smette. Quindi si toglie la fascia di cinta a ridosso del mare, e il
    # varco che resta e' il fronte a mare.
    #
    # Un FIUME che attraversa l'abitato non e' la stessa cosa: non e' una
    # difesa naturale, e' solo un corso d'acqua che passa in mezzo. Applicargli
    # la stessa fascia larga apriva un buco ingiustificato nel mezzo della
    # cinta, ovunque il fiume la sfiorasse - "un effetto poco normale",
    # segnalato dall'utente. La cinta ora resta piena fino alla riva del
    # fiume su entrambi i lati, e il solo punto dove il fiume la attraversa
    # (`varco_fiume`, sopra) diventa un vero varco con architrave, come una
    # porta - vedi piu' sotto.
    if riva > 0:
        vicino_mare = binary_dilation(np.isin(cls, MARINO), iterations=int(riva))
        anello &= ~vicino_mare

    # E NON SI COSTRUISCE SUL DIRUPO. Un muro appoggiato a un pendio da tre
    # blocchi per cella e' una fila di cubi sfalsati, non una cortina.
    if pendenza_massima > 0 and altezze is not None:
        gz_, gx_ = np.gradient(altezze.astype(np.float32))
        anello &= np.hypot(gz_, gx_) <= pendenza_massima

    if not anello.any():
        return vuoto, vuoto.copy(), [], vuoto.copy()

    muro = anello.copy()
    porta = np.zeros_like(anello)
    zs, xs = np.nonzero(anello)
    punti: list[tuple[int, int]] = []
    for gz, gx in porte_grafo:
        d = np.hypot(zs - gz, xs - gx)
        k = int(np.argmin(d))
        mz, mx = int(zs[k]), int(xs[k])
        if d[k] > 40:
            continue
        # il varco nel muro
        porta[max(0, mz - 1):mz + 2, max(0, mx - 1):mx + 2] = True
        # e la via che ci arriva
        for z, x in _linea((gz, gx), (mz, mx), 0.0, np.random.default_rng(0)):
            if 0 <= z < vie.shape[0] and 0 <= x < vie.shape[1]:
                vie[z, x] = max(vie[z, x], ASSE)
        punti.append((mz, mx))
    porta |= varco_fiume          # il varco sul fiume, vedi sopra
    muro &= ~porta

    # --- torri ------------------------------------------------------------
    # Una cinta senza torri e' un recinto. Le torri sono anche l'unica cosa
    # che, da dentro, dice dove finisce la citta': un muro alto cinque lo si
    # perde di vista dietro una casa, una torre no.
    torre = np.zeros_like(anello)
    zc, xc = float(zs.mean()), float(xs.mean())
    ordine = np.argsort(np.arctan2(zs - zc, xs - xc))
    passo_torri = max(18, int(len(ordine) / 14))
    for k in range(0, len(ordine), passo_torri):
        tz, tx = int(zs[ordine[k]]), int(xs[ordine[k]])
        torre[max(0, tz - 1):tz + 2, max(0, tx - 1):tx + 2] = True
    # e due torri a fianco di ogni porta, che e' il posto in cui le torri
    # sono servite davvero
    for mz, mx in punti:
        torre[max(0, mz - 3):mz + 4, max(0, mx - 3):mx + 4] |= \
            anello[max(0, mz - 3):mz + 4, max(0, mx - 3):mx + 4]
        torre[max(0, mz - 1):mz + 2, max(0, mx - 1):mx + 2] = False
    torre &= anello & ~porta
    return muro, porta, punti, torre


# --------------------------------------------------------------------------
# Pianificazione
# --------------------------------------------------------------------------

PASSO_TERRAZZA = 4      # altezza di un gradone, in blocchi


def _riquadro(h, area, margine):
    zs, xs = np.nonzero(area)
    if zs.size == 0:
        return None
    z0, z1 = max(0, int(zs.min()) - margine), min(h.shape[0], int(zs.max()) + margine + 1)
    x0, x1 = max(0, int(xs.min()) - margine), min(h.shape[1], int(xs.max()) + margine + 1)
    return z0, z1, x0, x1


def terrazza_abitato(h: np.ndarray, area: np.ndarray, intoccabile: np.ndarray,
                     passo: int = PASSO_TERRAZZA, sfocatura: float = 5.0) -> None:
    """Spezza il terreno dell'abitato in pochi ripiani piani.

    La spianata dolce di prima toglieva la rugosita' ma lasciava la pendenza,
    e su un fianco ripido non bastava: le case restavano a cinquanta quote
    diverse e il paese veniva un mappazzone di gradini casuali.

    Un paese vero in collina non segue il pendio: lo **terrazza**. Pochi
    ripiani piani, ciascuno alto qualche blocco, e fra l'uno e l'altro una
    scarpata o un muro di sostegno. Le case di un ripiano stanno tutte alla
    stessa quota, le strade salgono in rampa da un ripiano al successivo, e
    la citta' si legge.

    Tecnicamente e' una quantizzazione: si sfoca il terreno e si arrotonda al
    multiplo del passo. I bordi dei ripiani vengono da soli lungo le curve di
    livello, che e' esattamente dove un contadino avrebbe messo il muretto.
    """
    r = _riquadro(h, area, int(np.ceil(sfocatura * 3)))
    if r is None:
        return
    z0, z1, x0, x1 = r
    porzione = h[z0:z1, x0:x1].astype(np.float32)
    liscio = gaussian_filter(porzione, sfocatura)
    ripiani = np.round(liscio / passo) * passo
    # un ripiano non puo' allontanarsi dal terreno vero piu' di un passo:
    # altrimenti in fondo a una conca si scava un pozzo
    ripiani = np.clip(ripiani, liscio - passo, liscio + passo)

    peso = gaussian_filter(area[z0:z1, x0:x1].astype(np.float32), 3.0)
    peso = np.clip(peso * 1.4, 0.0, 1.0)
    peso[intoccabile[z0:z1, x0:x1]] = 0.0
    h[z0:z1, x0:x1] = np.round(porzione * (1 - peso)
                               + ripiani * peso).astype(np.int32)


def raccorda_mura(h: np.ndarray, muro: np.ndarray, intoccabile: np.ndarray,
                  raggio: int = 1, sfocatura: float = 1.5) -> None:
    """Toglie il seghettato dalla base della cinta.

    Le mura non hanno una quota propria: in `motore._posa_mura` ogni colonna
    parte dalla quota del terreno SOTTO di se' e sale di un'altezza fissa. Se
    quella quota e' quella finale - con il dither e la rugosita' fine che
    servono bene a un prato ma non a una cortina di pietra - due merli vicini
    nascono anche solo un blocco piu' alti o piu' bassi l'uno dell'altro, e da
    lontano la cinta non si legge come un muro ma come una fila di macerie
    ammucchiate: e' esattamente il "mura di pietra... in ordine sparso" degli
    screenshot.

    Qui si sfoca il terreno SOLO sotto l'anello (e una cella di rispetto
    intorno, cosi' anche le fondamenta scendono con continuita'): la cinta
    puo' ancora arrampicarsi su un pendio vero, ma senza il dente-si'-dente-no
    che viene dal rumore a grana fine. Va chiamata DOPO `mura()`, che ha gia'
    usato il terreno non sfocato per il controllo di pendenza - toglier loro
    la grana qui non cambia dove passa la cinta, solo com'e' fatta sotto.
    """
    sede = binary_dilation(muro, iterations=raggio) & ~intoccabile
    if not sede.any():
        return
    r = _riquadro(h, sede, 4)
    if r is None:
        return
    z0, z1, x0, x1 = r
    porzione = h[z0:z1, x0:x1].astype(np.float32)
    liscio = gaussian_filter(porzione, sfocatura)
    m = sede[z0:z1, x0:x1]
    h[z0:z1, x0:x1] = np.where(m, np.round(liscio), porzione).astype(np.int32)


def raccorda_vie(h: np.ndarray, vie: np.ndarray, intoccabile: np.ndarray,
                 raggio: int = 2) -> None:
    """Trasforma i gradoni attraversati da una strada in rampe.

    Le terrazze fanno bene alle case e male alle strade: una via che incontra
    un salto di quattro blocchi diventa una parete. Qui si risfoca il terreno
    lungo la sede stradale e un paio di celle intorno, cosi' il salto si
    distribuisce su qualche cella e diventa una salita percorribile. Le case
    non se ne accorgono: sono gia' state messe, e stanno dentro i ripiani.
    """
    sede = binary_dilation(vie > 0, iterations=raggio) & ~intoccabile
    if not sede.any():
        return
    r = _riquadro(h, sede, 6)
    if r is None:
        return
    z0, z1, x0, x1 = r
    porzione = h[z0:z1, x0:x1].astype(np.float32)
    rampa = gaussian_filter(porzione, 2.2)
    m = sede[z0:z1, x0:x1]
    h[z0:z1, x0:x1] = np.where(m, np.round(rampa), porzione).astype(np.int32)


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
) -> tuple[list[E.Edificio], np.ndarray, np.ndarray, np.ndarray,
           list[Citta], list[E.Banco], np.ndarray]:
    """Progetta gli insediamenti.

    Ritorna (edifici, altezze spianate, vie, mura, citta', banchi, urbano).
    """
    from .insediamenti import _terrazza

    h = altezze.astype(np.int32).copy()
    H, W = cls.shape
    vie = np.zeros((H, W), np.uint8)
    muro = np.zeros((H, W), np.uint8)
    urbano = np.zeros((H, W), bool)
    fuori: list[E.Edificio] = []
    citta: list[Citta] = []
    banchi: list[E.Banco] = []

    # L'acqua e le sue sponde non si toccano: ne' la spianata dell'abitato ne'
    # il raccordo dei lotti. Tre celle di rispetto, che sono il raccordo meno
    # due: cosi' il peso e' gia' quasi nullo quando arriva alla riva.
    intoccabile = binary_dilation(np.isin(cls, ACQUA), iterations=3)

    for n, (sz, sx, raggio) in enumerate(siti):
        rng = np.random.default_rng(seed * 977 + n)
        area = contorno(cls, h, sz, sx, raggio, livello_mare, seed=seed * 31 + n)
        if area.sum() < 200:
            continue
        terrazza_abitato(h, area, intoccabile)
        pz, px = _piazza(area, h, sz, sx)
        rango, porte, _ = rete(area, (pz, px), raggio, seed=seed * 61 + n)
        if not rango.any():
            continue
        # le strade PRIMA delle case: il raccordo delle rampe muove il
        # terreno, e una casa gia' posata si ritroverebbe il pavimento storto
        raccorda_vie(h, rango, intoccabile)
        vie = np.maximum(vie, rango)
        urbano |= area

        ed = lotti(area, rango > 0, cls, h, livello, (pz, px), raggio,
                   livello_mare, rng)
        for e in ed:
            e.villaggio = n
            _terrazza(h, e, intoccabile=intoccabile)
        fuori.extend(ed)

        # il mercato dopo le case: gli serve sapere cosa e' gia' occupato
        # occupato = solo gli edifici: il selciato della piazza e' dove il
        # mercato DEVE stare, non un ostacolo
        preso = np.zeros_like(area)
        for e in ed:
            x0, z0, x1, z1 = e.ingombro_tetto()
            preso[max(0, z0 - 1):z1 + 1, max(0, x0 - 1):x1 + 1] = True
        banchi.extend(mercato(area, rango, preso, h, (pz, px), raggio, rng))

        c = Citta(z=sz, x=sx, raggio=raggio, piazza=(pz, px), porte=porte,
                  edifici=len(ed))
        if c.e_citta and ed:
            sedimi = np.zeros((H, W), bool)
            for e in ed:
                sedimi[e.z:e.z1, e.x:e.x1] = True
            m, p, varchi, t = mura(area, rango, sedimi, cls, porte, altezze=h)
            if m.any():
                # via la grana fine da sotto la cinta PRIMA di fissarla nel
                # muro condiviso: mura() ha gia' deciso dove passa usando il
                # terreno vero, qui si cambia solo la quota su cui poggia.
                raccorda_mura(h, m | p, intoccabile)
                muro[m] = 1
                muro[t] = 3
                muro[p] = 2
                c.murata = True
                if varchi:
                    c.porte = varchi
                vie = np.maximum(vie, rango)
        citta.append(c)

    return fuori, h, vie, muro, citta, banchi, urbano


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
