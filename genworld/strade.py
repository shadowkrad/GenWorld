"""Strade e ponti.

I ponti non si progettano: EMERGONO. Era la scelta di progetto del primo
documento e si e' rivelata giusta. Non esiste da nessuna parte una funzione
"metti un ponte qui": si traccia una strada, e ogni tratto che finisce
sull'acqua o su un avvallamento profondo diventa ponte perche' e' l'unico modo
di stare alla quota della strada. Il ponte e' una conseguenza della geometria,
non un oggetto da piazzare.

Il tracciato si cerca su una griglia di costo con A*: piano costa poco, ripido
costa molto, acqua costa parecchio ma non e' proibita. Cosi' la strada gira
attorno alle colline e attraversa il fiume nel punto piu' stretto, che e'
esattamente quello che fa una strada vera.

Il percorso si calcola su una griglia RIDOTTA: su 832x832 un A* per ogni coppia
di villaggi sarebbe milioni di celle per nulla, e la strada non ha bisogno di
decidere blocco per blocco - le serve la linea generale, il dettaglio lo da'
il terreno.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, uniform_filter

from .mappa import ACQUA, MONTAGNA, NEVE

NIENTE, STRADA, PONTE, PARAPETTO, LASTRICATO = 0, 1, 2, 3, 4


def griglia_costo(cls: np.ndarray, altezze: np.ndarray,
                  costo_acqua: float = 16.0, peso_pendenza: float = 6.0,
                  costo_montagna: float = 3.0,
                  campi: np.ndarray | None = None,
                  costo_campi: float = 25.0) -> np.ndarray:
    """Quanto costa attraversare ogni cella.

    I campi costano CARO ma non sono vietati: una strada che deve passare di
    li' passa, e in campagna succede. Senza questo costo pero' l'A* tagliava
    dritto per i poderi, perche' sono la cosa piu' piana che ci sia - li
    avevamo appena spianati noi.
    """
    pend = np.hypot(*np.gradient(gaussian_filter(altezze.astype(np.float32), 1.5)))
    c = 1.0 + peso_pendenza * pend
    c = np.where(np.isin(cls, ACQUA), c + costo_acqua, c)
    c = np.where(np.isin(cls, (MONTAGNA, NEVE)), c + costo_montagna, c)
    if campi is not None:
        c = np.where(campi > 0, c + costo_campi, c)
    return c.astype(np.float32)


def _riduci(a: np.ndarray, fattore: int) -> np.ndarray:
    H, W = a.shape
    h, w = H // fattore, W // fattore
    return a[:h * fattore, :w * fattore].reshape(h, fattore, w, fattore).mean((1, 3))


def _astar(costo: np.ndarray, partenza: tuple[int, int],
           arrivo: tuple[int, int]) -> list[tuple[int, int]]:
    """Cammino di costo minimo, 8 direzioni."""
    H, W = costo.shape
    pz, px = partenza
    az, ax = arrivo
    if not (0 <= pz < H and 0 <= px < W and 0 <= az < H and 0 <= ax < W):
        return []

    def stima(z, x):
        return float(np.hypot(z - az, x - ax))

    visto = np.zeros((H, W), bool)
    da_dove = np.full((H, W, 2), -1, np.int32)
    finora = np.full((H, W), np.inf, np.float32)
    finora[pz, px] = 0.0
    coda = [(stima(pz, px), pz, px)]
    passi = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    while coda:
        _, z, x = heapq.heappop(coda)
        if visto[z, x]:
            continue
        visto[z, x] = True
        if (z, x) == (az, ax):
            break
        for dz, dx, l in passi:
            nz, nx = z + dz, x + dx
            if not (0 <= nz < H and 0 <= nx < W) or visto[nz, nx]:
                continue
            nuovo = finora[z, x] + costo[nz, nx] * l
            if nuovo < finora[nz, nx]:
                finora[nz, nx] = nuovo
                da_dove[nz, nx] = (z, x)
                heapq.heappush(coda, (nuovo + stima(nz, nx), nz, nx))

    if not visto[az, ax]:
        return []
    cammino = [(az, ax)]
    while cammino[-1] != (pz, px):
        z, x = cammino[-1]
        p = da_dove[z, x]
        if p[0] < 0:
            return []
        cammino.append((int(p[0]), int(p[1])))
    return cammino[::-1]


def _segmento(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Celle fra due punti (Bresenham)."""
    z0, x0 = a; z1, x1 = b
    dz, dx = abs(z1 - z0), abs(x1 - x0)
    sz = 1 if z1 > z0 else -1
    sx = 1 if x1 > x0 else -1
    err = dz - dx
    fuori = []
    z, x = z0, x0
    while True:
        fuori.append((z, x))
        if (z, x) == (z1, x1):
            return fuori
        e2 = 2 * err
        if e2 > -dx:
            err -= dx; z += sz
        if e2 < dz:
            err += dz; x += sx


def pianifica(
    cls: np.ndarray,
    altezze: np.ndarray,
    livello: np.ndarray,
    siti: list[tuple[int, int, int]],
    edifici: list,
    livello_mare: int = 62,
    fattore: int = 4,
    larghezza: int = 1,
    vie: np.ndarray | None = None,
    ancore: list[list[tuple[int, int]]] | None = None,
    campi: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Traccia la rete. Ritorna (tipo, quota, altezze spianate).

    `tipo` vale NIENTE / STRADA / PONTE / PARAPETTO per ogni cella.
    """
    H, W = cls.shape
    tipo = np.zeros((H, W), np.uint8)
    quota = np.zeros((H, W), np.int32)
    h = altezze.astype(np.int32).copy()
    if not siti and not edifici and vie is None:
        return tipo, quota, h

    celle: list[tuple[int, int]] = []

    # --- collegamenti fra villaggi, su griglia ridotta ------------------
    if len(siti) > 1:
        costo = _riduci(griglia_costo(cls, h, campi=campi),
                        fattore).astype(np.float32)
        ridotti = [(z // fattore, x // fattore) for z, x, _ in siti]
        # ogni villaggio al piu' vicino non ancora collegato (albero minimo)
        collegati = {0}
        while len(collegati) < len(ridotti):
            meglio = None
            for i in collegati:
                for j in range(len(ridotti)):
                    if j in collegati:
                        continue
                    d = np.hypot(ridotti[i][0] - ridotti[j][0],
                                 ridotti[i][1] - ridotti[j][1])
                    if meglio is None or d < meglio[0]:
                        meglio = (d, i, j)
            if meglio is None:
                break
            _, i, j = meglio
            collegati.add(j)
            # La strada esterna punta alla PORTA piu' vicina, non al centro:
            # senza, il tracciato entrava in citta' e passava in mezzo alle
            # case, che e' esattamente il motivo per cui le porte esistono.
            pa = _ancora(ancore, i, siti[j], ridotti[i], fattore)
            pb = _ancora(ancore, j, siti[i], ridotti[j], fattore)
            cammino = _astar(costo, pa, pb)
            for k in range(len(cammino) - 1):
                a = (cammino[k][0] * fattore + fattore // 2,
                     cammino[k][1] * fattore + fattore // 2)
                b = (cammino[k + 1][0] * fattore + fattore // 2,
                     cammino[k + 1][1] * fattore + fattore // 2)
                celle.extend(_segmento(a, b))

    # I vicoli interni non si tracciano piu' qui: li disegna `citta.py`,
    # che sa dove sono gli isolati. Prima ogni casa veniva collegata al
    # centro del villaggio con un raggio dritto, e il risultato era una
    # ruota di bicicletta.
    if not celle:
        return tipo, quota, h

    linea = np.zeros((H, W), bool)
    for z, x in celle:
        if 0 <= z < H and 0 <= x < W:
            linea[z, x] = True
    strada = binary_dilation(linea, iterations=larghezza) if larghezza else linea
    if vie is not None:
        strada = strada | (vie > 0)

    # --- quota della sede stradale --------------------------------------
    # La strada segue il terreno ma non i suoi sobbalzi: si prende la media
    # locale. Senza, il tracciato copia ogni dosso e diventa una scalinata.
    liscio = uniform_filter(h.astype(np.float32), 9)
    q = np.where(strada, np.maximum(np.round(liscio), livello_mare + 1), 0)

    # --- dove serve un ponte --------------------------------------------
    acqua = np.isin(cls, ACQUA)
    sopra_acqua = strada & acqua
    in_aria = strada & (h < q - 3)
    ponte = sopra_acqua | in_aria
    # sull'acqua la sede sale sopra il pelo, altrimenti il ponte e' sommerso
    q = np.where(sopra_acqua, np.maximum(q, np.round(livello) + 2), q)

    tipo[strada] = STRADA
    if vie is not None:
        # i viali di rango >= secondaria restano una classe a parte - serve
        # a `motore._posa_strada` per distinguerli dai vicoli - ma non sono
        # piu' lastricati di pietra: vedi il commento li' per il perche'.
        tipo[(vie >= 2) & ~ponte] = LASTRICATO
    tipo[ponte] = PONTE
    # parapetto: cella di ponte che confina con il vuoto
    fuori_ponte = binary_dilation(ponte, iterations=1) & ~ponte
    orlo = ponte & binary_dilation(fuori_ponte, iterations=1)
    tipo[orlo] = PARAPETTO
    quota = q.astype(np.int32)

    # --- spianamento della sede (solo dove non e' ponte) -----------------
    solo_strada = strada & ~ponte
    h = np.where(solo_strada, quota, h)
    return tipo, quota, h


def _ancora(ancore, indice, verso_sito, predefinito, fattore):
    """La porta dell'insediamento `indice` piu' vicina all'altro sito."""
    if not ancore or indice >= len(ancore) or not ancore[indice]:
        return predefinito
    tz, tx = verso_sito[0], verso_sito[1]
    pz, px = min(ancore[indice], key=lambda p: (p[0] - tz) ** 2 + (p[1] - tx) ** 2)
    return (pz // fattore, px // fattore)


def statistiche(tipo: np.ndarray) -> dict[str, int]:
    return {
        "strada": int((tipo == STRADA).sum()),
        "lastricato": int((tipo == LASTRICATO).sum()),
        "ponte": int(((tipo == PONTE) | (tipo == PARAPETTO)).sum()),
        "parapetto": int((tipo == PARAPETTO).sum()),
    }
