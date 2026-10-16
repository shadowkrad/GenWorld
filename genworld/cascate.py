"""Cascate: dove il pelo di un fiume scende di colpo, l'acqua cade davvero.

Il livello dell'acqua e' per cella (vedi `fiumi.livella`) e dove il corso scende
di qualche blocco in una sola cella il file contiene due colonne d'acqua di
altezza diversa, con aria accanto. In un mondo scritto da un programma i fluidi
non hanno ancora avuto un "tick": quell'aria resta vuota finche' qualcosa non
la sveglia, e il salto sembra un muro d'acqua tagliato. Qui la caduta si scrive
gia' fatta:

  - nella cella bassa, dal pelo di valle a quello di monte, acqua *che cade*
    (stato 8, quello che il gioco stesso scrive per l'acqua in caduta);
  - sui lati, dove c'e' terra e non acqua, una parete di roccia (muschiosa a
    tratti) alta fino al pelo di monte: l'acqua non scappa sul prato e non
    scava la riva;
  - la pozza ai piedi e' il fiume stesso, che li' e' piu' basso.

Solo fiume -> acqua con un salto di almeno SALTO_MINIMO blocchi.
"""
from __future__ import annotations

import numpy as np

from . import mappa as M

SALTO_MINIMO = 3
VICINI = ((-1, 0), (1, 0), (0, -1), (0, 1))


def trova(livello: np.ndarray, cls: np.ndarray, mare: int = 62,
          lava: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(caduta, parete), due mappe int16: `caduta[z, x]` e' il pelo di monte
    (intero) nelle celle d'acqua che stanno sotto un salto, 0 altrove;
    `parete[z, x]` e' lo stesso valore nelle celle di terra che affiancano una
    caduta."""
    H, W = cls.shape
    liv = np.maximum(np.floor(livello).astype(np.int32), mare)
    acqua = np.isin(cls, M.ACQUA)
    if lava is not None:
        acqua &= ~np.asarray(lava, bool)
    fiume = cls == M.FIUME
    caduta = np.zeros((H, W), np.int16)
    for dz, dx in VICINI:
        # (z, x) = cella di monte, (z+dz, x+dx) = quella di valle
        zs, ze = max(0, -dz), H - max(0, dz)
        xs, xe = max(0, -dx), W - max(0, dx)
        a = (slice(zs, ze), slice(xs, xe))
        b = (slice(zs + dz, ze + dz), slice(xs + dx, xe + dx))
        salto = fiume[a] & acqua[b] & (liv[a] - liv[b] >= SALTO_MINIMO)
        alto = np.where(salto, liv[a], 0).astype(np.int16)
        caduta[b] = np.maximum(caduta[b], alto)
    parete = np.zeros((H, W), np.int16)
    for dz, dx in VICINI:
        zs, ze = max(0, -dz), H - max(0, dz)
        xs, xe = max(0, -dx), W - max(0, dx)
        src = caduta[zs:ze, xs:xe]
        dst = (slice(zs + dz, ze + dz), slice(xs + dx, xe + dx))
        parete[dst] = np.maximum(parete[dst], src)
    parete[acqua] = 0
    return caduta, parete


def posa(out: np.ndarray, scrittore, h_c: np.ndarray, liv_c: np.ndarray,
         caduta_c: np.ndarray, parete_c: np.ndarray, ox: int, oz: int, y0: int,
         mare: int = 62) -> int:
    """Scrive caduta e pareti in un chunk (`out` ha forma (16, alt, 16)).
    Ritorna quante colonne d'acqua ha scritto."""
    if not (np.any(caduta_c) or np.any(parete_c)):
        return 0
    aria = scrittore.id_aria
    cade = scrittore.blocco("water", falling="true", flowing="true", level="0")
    pietra = (scrittore.blocco("mossy_cobblestone"), scrittore.blocco("stone"),
              scrittore.blocco("mossy_cobblestone"), scrittore.blocco("andesite"))
    alt = out.shape[1]
    colonne = 0
    liv = np.maximum(np.floor(np.asarray(liv_c)).astype(np.int32), mare)
    for lx, lz in zip(*np.nonzero(caduta_c)):
        basso, alto = int(liv[lx, lz]) + 1, int(caduta_c[lx, lz])
        a, b = max(0, basso - y0), min(alt, alto - y0 + 1)
        if a >= b:
            continue
        col = out[lx, a:b, lz]
        col[col == aria] = cade
        colonne += 1
    for lx, lz in zip(*np.nonzero(parete_c)):
        alto = int(parete_c[lx, lz])
        base = int(h_c[lx, lz])
        a, b = max(0, base - y0), min(alt, alto - y0 + 1)
        for iy in range(a, b):
            if out[lx, iy, lz] == aria:
                out[lx, iy, lz] = pietra[(ox + lx + 3 * (oz + lz) + 5 * iy) % 4]
    return colonne
