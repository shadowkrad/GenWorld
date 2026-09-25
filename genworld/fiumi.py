"""Fiumi: estrazione dalla mappa e livellamento del corso.

Sulle mappe disegnate i fiumi sono blu come il mare, quindi il classificatore
li mette nella stessa classe. Separarli per colore e' impossibile: sono lo
stesso colore. Si separano per FORMA.

Un mare e' largo, un fiume e' sottile. L'apertura morfologica con un disco
cancella tutto cio' che e' piu' stretto del disco e lascia intatto il resto:
quello che sparisce e' fiume, quello che resta e' mare. Nessuna soglia di
colore, nessuna taratura sulla tavolozza.

Il secondo problema e' che un fiume deve SCENDERE. Una maschera presa da
un'immagine non sa niente di quote: tagliata dentro il terreno cosi' com'e',
produce un corso che sale e scende, con l'acqua che risale le colline. Il
livellamento risolve questo, propagando dalla foce verso monte con la regola
che il pelo dell'acqua non puo' mai calare andando a monte.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, gaussian_filter, label


def _disco(raggio: int) -> np.ndarray:
    r = max(1, int(raggio))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def estrai(acqua: np.ndarray, larghezza_mare: int = 5,
           lunghezza_minima: int = 40) -> np.ndarray:
    """Separa i fiumi dal mare per forma, non per colore.

    `larghezza_mare` e' il raggio del disco: una distesa d'acqua che non
    contiene un disco di quel raggio viene considerata fiume.
    `lunghezza_minima` scarta i frammenti: una pozza di dieci pixel non e' un
    fiume, e' rumore di classificazione.
    """
    if not acqua.any():
        return np.zeros_like(acqua)

    mare = binary_opening(acqua, _disco(larghezza_mare))
    fiumi = acqua & ~mare

    cc, n = label(fiumi, structure=np.ones((3, 3), bool))
    if n == 0:
        return fiumi
    dim = np.bincount(cc.ravel())
    dim[0] = 0
    tieni = np.flatnonzero(dim >= lunghezza_minima)
    return np.isin(cc, tieni)


def livella(
    fiumi: np.ndarray,
    altezze: np.ndarray,
    mare: np.ndarray,
    livello_mare: int = 62,
    incassamento: float = 2.0,
    profondita: int = 2,
    scavo_rive: int = 2,
    scavo_massimo: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcola il pelo dell'acqua e scava l'alveo.

    Ritorna (livello_acqua, altezze_scavate). `livello_acqua` vale
    `livello_mare` ovunque tranne sui fiumi, dove sale seguendo il terreno.

    La propagazione parte dalle FOCI - le celle di fiume che toccano il mare -
    e risale. La regola e' una sola: andando a monte il pelo dell'acqua non
    puo' scendere. Senza, si ottengono fiumi che scorrono in salita, che e' il
    difetto piu' evidente di un corso d'acqua disegnato a mano e poi tagliato
    dentro il terreno.
    """
    H, W = altezze.shape
    livello = np.full((H, W), float(livello_mare), np.float32)
    h = altezze.astype(np.float32).copy()
    if not fiumi.any():
        return livello, h

    # il terreno sfocato evita che una singola cella rumorosa alzi il corso
    liscio = gaussian_filter(h, 1.5)

    visto = np.zeros((H, W), bool)
    coda: deque = deque()

    # foci: celle di fiume a contatto col mare
    foci = fiumi & binary_dilation(mare, iterations=1)
    for z, x in zip(*np.nonzero(foci)):
        livello[z, x] = float(livello_mare)
        visto[z, x] = True
        coda.append((int(z), int(x)))

    # componenti senza foce: si parte dalla cella piu' bassa
    if fiumi.any():
        cc, n = label(fiumi, structure=np.ones((3, 3), bool))
        toccate = set(np.unique(cc[visto])) - {0}
        for k in range(1, n + 1):
            if k in toccate:
                continue
            m = cc == k
            zz, xx = np.nonzero(m)
            i = int(np.argmin(liscio[zz, xx]))
            z, x = int(zz[i]), int(xx[i])
            livello[z, x] = max(float(livello_mare), liscio[z, x] - incassamento)
            visto[z, x] = True
            coda.append((z, x))

    vicini = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while coda:
        z, x = coda.popleft()
        base = livello[z, x]
        for dz, dx in vicini:
            nz, nx = z + dz, x + dx
            if not (0 <= nz < H and 0 <= nx < W):
                continue
            if visto[nz, nx] or not fiumi[nz, nx]:
                continue
            # a monte il pelo dell'acqua non puo' scendere
            livello[nz, nx] = max(base, liscio[nz, nx] - incassamento)
            visto[nz, nx] = True
            coda.append((nz, nx))

    # Alveo: sotto il pelo dell'acqua, ma con un tetto allo scavo. Senza, dove
    # il corso passa su un dosso locale il terreno sfocato sta molto piu' in
    # basso di quello vero e si apre un canyon da decine di blocchi.
    alveo = np.maximum(livello - profondita, h - scavo_massimo)
    h = np.where(fiumi, alveo, h)

    # rive: si abbassa la fascia adiacente cosi' il fiume sta in un solco e
    # non su un rilevato, che e' l'effetto di scavare senza toccare i bordi
    if scavo_rive > 0:
        riva = binary_dilation(fiumi, iterations=scavo_rive) & ~fiumi
        tetto = np.where(riva, livello + 2.0, np.inf)
        h = np.minimum(h, tetto)

    return livello, h


def statistiche(fiumi: np.ndarray, livello: np.ndarray,
                livello_mare: int = 62) -> dict:
    n_celle = int(fiumi.sum())
    if n_celle == 0:
        return {"celle": 0, "corsi": 0, "quota_max": 0.0, "dislivello": 0.0}
    cc, n = label(fiumi, structure=np.ones((3, 3), bool))
    liv = livello[fiumi]
    return {
        "celle": n_celle,
        "corsi": int(n),
        "quota_max": float(liv.max()),
        "dislivello": float(liv.max() - livello_mare),
    }


# --------------------------------------------------------------------------
# Fiumi calcolati dal terreno
# --------------------------------------------------------------------------
# Estrarre i fiumi disegnati sulla mappa si e' rivelato impraticabile: a
# risoluzione di lavoro sono larghi uno o due pixel, tenui, e il loro colore si
# confonde con le ombre bluastre del rilievo. Tre criteri provati - forma
# sull'acqua classificata, "piu' blu del contorno", e i due combinati - danno
# insenature costiere, creste montuose, o niente.
#
# Ma un fiume non e' un segno sulla carta: e' dove l'acqua si raccoglie. Avendo
# la heightmap si puo' calcolare, ed e' meglio sotto ogni aspetto: scorre in
# discesa per costruzione, arriva sempre al mare, e funziona su qualunque mappa
# senza dipendere dalla tavolozza.

def riempi_depressioni(altezze: np.ndarray, mare: np.ndarray) -> np.ndarray:
    """Priority-flood: elimina le conche chiuse alzandole fino allo sfioro.

    Senza, il calcolo del deflusso si ferma dentro ogni buca e non si formano
    corsi continui. Il terreno riempito serve solo a calcolare le direzioni:
    non e' quello che finisce nel mondo.
    """
    import heapq

    H, W = altezze.shape
    riempito = altezze.astype(np.float32).copy()
    visto = np.zeros((H, W), bool)
    heap: list = []

    bordo = np.zeros((H, W), bool)
    bordo[0, :] = bordo[-1, :] = bordo[:, 0] = bordo[:, -1] = True
    semi = bordo | mare
    for z, x in zip(*np.nonzero(semi)):
        heapq.heappush(heap, (float(riempito[z, x]), int(z), int(x)))
        visto[z, x] = True

    vicini = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while heap:
        q, z, x = heapq.heappop(heap)
        for dz, dx in vicini:
            nz, nx = z + dz, x + dx
            if not (0 <= nz < H and 0 <= nx < W) or visto[nz, nx]:
                continue
            visto[nz, nx] = True
            nq = max(riempito[nz, nx], q)
            riempito[nz, nx] = nq
            heapq.heappush(heap, (float(nq), nz, nx))
    return riempito


def accumulo(altezze: np.ndarray, mare: np.ndarray) -> np.ndarray:
    """Area drenata da ogni cella, in celle.

    Direzione di deflusso D8 - ogni cella scarica nel vicino di massima
    pendenza - e poi una passata sola in ordine di quota decrescente: quando
    si arriva a una cella, tutto cio' che sta a monte l'ha gia' alimentata.
    """
    H, W = altezze.shape
    q = riempi_depressioni(altezze, mare)

    # vicino piu' basso, con la pendenza corretta per la diagonale
    dz8 = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dx8 = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    dist = np.hypot(dz8, dx8).astype(np.float32)

    qp = np.pad(q, 1, mode="edge")
    migliore = np.full((H, W), -1, np.int8)
    pend_max = np.zeros((H, W), np.float32)
    for k in range(8):
        vic = qp[1 + dz8[k]: 1 + dz8[k] + H, 1 + dx8[k]: 1 + dx8[k] + W]
        pend = (q - vic) / dist[k]
        meglio = pend > pend_max
        pend_max = np.where(meglio, pend, pend_max)
        migliore = np.where(meglio, k, migliore)

    acc = np.ones(H * W, np.float32)
    ordine = np.argsort(q.ravel())[::-1]          # dal piu' alto al piu' basso
    mig = migliore.ravel()
    zz = (np.arange(H * W) // W)
    xx = (np.arange(H * W) % W)
    for i in ordine:
        k = mig[i]
        if k < 0:
            continue
        nz = zz[i] + dz8[k]
        nx = xx[i] + dx8[k]
        if 0 <= nz < H and 0 <= nx < W:
            acc[nz * W + nx] += acc[i]
    return acc.reshape(H, W)


def da_terreno(
    altezze: np.ndarray,
    mare: np.ndarray,
    soglia: float = 150.0,
    sfocatura: float = 4.0,
    larghezza_per_portata: int = 1,
    lunghezza_minima: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Fiumi dove l'acqua si raccoglie. Ritorna (maschera, accumulo).

    Il deflusso si calcola su un terreno SFOCATO: il reticolo idrografico
    segue la topografia regionale, non i dossi da un blocco. Senza, ogni
    increspatura devia l'acqua e i corsi si frantumano in rivoli scollegati.
    """
    base = gaussian_filter(altezze.astype(np.float32), sfocatura) if sfocatura > 0 \
        else altezze.astype(np.float32)
    acc = accumulo(base, mare)
    f = (acc >= soglia) & ~mare

    if larghezza_per_portata > 0 and f.any():
        # i corsi maggiori si allargano: la foce e' piu' larga della sorgente
        grandi = (acc >= soglia * 6) & ~mare
        if grandi.any():
            f = f | (binary_dilation(grandi, iterations=int(larghezza_per_portata))
                     & ~mare)

    if lunghezza_minima > 1 and f.any():
        cc, n = label(f, structure=np.ones((3, 3), bool))
        if n:
            dim = np.bincount(cc.ravel()); dim[0] = 0
            f = np.isin(cc, np.flatnonzero(dim >= lunghezza_minima))
    return f, acc
