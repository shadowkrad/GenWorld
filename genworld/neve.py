"""Neve sottile sui tetti e sulle chiome (#28).

Nei biomi innevati il manto non si ferma al terreno: la neve si posa anche sulle
foglie degli alberi e sui tetti in blocchi interi, come in Minecraft. Si fa a
chunk finito, quando alberi e case sono gia' stati stampati: si cerca il blocco
piu' alto di ogni colonna e, se e' un cubo pieno adatto e sopra c'e' aria, ci si
posa uno strato di neve.
"""
from __future__ import annotations

import numpy as np

from . import biomi as B

# biomi dove nevica (acqua gelata esclusa: ci pensa `acque`)
BIOMI_NEVOSI = ("snowy_tundra", "snowy_taiga", "snowy_slopes", "frozen_peaks",
                "snowy_beach")

# blocchi (nomi universali) su cui la neve si posa: cubi pieni
PIENI = frozenset((
    "leaves", "planks", "log", "stone", "cobblestone", "mossy_cobblestone",
    "stone_bricks", "bricks", "terracotta", "stained_terracotta", "wool",
    "sandstone", "concrete", "grass_block", "dirt", "podzol", "coarse_dirt",
    "deepslate", "andesite", "diorite", "granite", "hay_block", "gravel",
))

DENSITA = 0.9


def neve_sui_tetti(out: np.ndarray, s, biomi_c, ox: int, oz: int) -> int:
    """Posa il manto sottile sui tetti e sulle chiome del chunk. Ritorna quanti."""
    nevosi = np.isin(np.asarray(biomi_c), [B.INDICE[n] for n in BIOMI_NEVOSI])
    if not nevosi.any():
        return 0
    aria = s.id_aria
    manto = s.blocco("snow", layers="1")
    H = out.shape[1]
    adatto: dict[int, bool] = {}
    posati = 0
    for lx, lz in zip(*np.nonzero(nevosi)):
        col = out[lx, :, lz]
        pieni = np.nonzero(col != aria)[0]
        if pieni.size == 0:
            continue
        top = int(pieni[-1])
        if top + 1 >= H or col[top + 1] != aria:
            continue
        b = int(col[top])
        if b not in adatto:
            adatto[b] = s.nome_blocco(b) in PIENI
        if not adatto[b]:
            continue
        h = ((ox + int(lx)) * 73856093) ^ ((oz + int(lz)) * 19349663)
        if (h & 0xFFFF) / 65535.0 < DENSITA:
            out[lx, top + 1, lz] = manto
            posati += 1
    return posati
