"""Vita dell'acqua (#28): alghe, ninfee, coralli, ghiaccio, argilla e fango.

Il mare e i fiumi erano acqua su sabbia o ghiaia. Qui, dopo che l'acqua e' stata
scritta, ogni colonna bagnata si veste secondo il BIOMA: nei mari gelati la
superficie e' ghiaccio, nei mari caldi crescono i coralli, altrove l'erba di
mare e le alghe; sui fiumi e sul basso fondo compaiono argilla e fango, e sui
fiumi e i laghi bassi le ninfee.

Tutto e' deterministico sulle coordinate DI MONDO (come le sponde dei fossati):
con quelle locali al chunk il motivo si ripeterebbe a ogni confine.
"""
from __future__ import annotations

import numpy as np

from . import biomi as B
from . import mappa as M

LIVELLO_MARE = 62

GELATI = ("frozen_ocean", "deep_frozen_ocean", "frozen_river")
CALDI = ("warm_ocean", "deep_warm_ocean")
CON_ALGHE = ("ocean", "cold_ocean", "deep_ocean", "deep_cold_ocean")
CORALLI = ("brain", "tube", "fire", "horn", "bubble")


def _rumore(x: np.ndarray, z: np.ndarray, sale: int) -> np.ndarray:
    """0..1, a grana fine, dalle coordinate di mondo."""
    h = (x.astype(np.int64) * 73856093) ^ (z.astype(np.int64) * 19349663) \
        ^ (np.int64(sale) * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def _grumo(x: np.ndarray, z: np.ndarray, sale: int, lato: int = 9) -> np.ndarray:
    """0..1, a chiazze larghe `lato` celle: fa raggruppare le piante."""
    return _rumore(x // lato, z // lato, sale + 1000)


def decora(out: np.ndarray, s, h_c, liv_c, cls_c, biomi_c, ox: int, oz: int,
           y0: int) -> int:
    """Veste l'acqua del chunk. Ritorna quante colonne sono state toccate."""
    h = np.asarray(h_c).astype(np.int32)
    liv = np.maximum(np.asarray(liv_c).astype(np.int32), LIVELLO_MARE)
    cls = np.asarray(cls_c)
    vulcanico = np.isin(cls, (M.CRATERE, M.VULCANO))
    bagnata = (h <= liv) & ~vulcanico
    if not bagnata.any():
        return 0

    gx = ox + np.arange(16)[:, None] + np.zeros((1, 16), np.int64)
    gz = oz + np.arange(16)[None, :] + np.zeros((16, 1), np.int64)
    fine, fine2 = _rumore(gx, gz, 1), _rumore(gx, gz, 2)
    grumo, grumo2 = _grumo(gx, gz, 3), _grumo(gx, gz, 4, 13)

    acqua = s.blocco("water")
    H = out.shape[1]
    nomi = np.array(B.BIOMI)
    bioma = nomi[np.asarray(biomi_c)]

    ghiaccio, pieno = s.blocco("ice"), s.blocco("packed_ice")
    argilla, fango = s.blocco("clay"), s.blocco("mud")
    erba_mare = s.blocco("seagrass")
    alta_b = s.blocco("tall_seagrass", half="lower")
    alta_a = s.blocco("tall_seagrass", half="upper")
    stelo = s.blocco("kelp_plant", age="25")
    ninfea = s.blocco("lily_pad")
    pickle = s.blocco("sea_pickle", pickles="2", dead="false")

    toccate = 0
    for lx, lz in zip(*np.nonzero(bagnata)):
        prof = int(liv[lx, lz] - h[lx, lz] + 1)
        sopra = int(liv[lx, lz]) - y0              # ultima cella d'acqua
        fondo = int(h[lx, lz]) - 1 - y0            # primo blocco solido sotto
        if not (0 < fondo and sopra < H - 1) or out[lx, sopra, lz] != acqua:
            continue
        b = str(bioma[lx, lz])
        f, f2, g, g2 = (float(fine[lx, lz]), float(fine2[lx, lz]),
                        float(grumo[lx, lz]), float(grumo2[lx, lz]))

        if b in GELATI:
            out[lx, sopra, lz] = pieno if (prof <= 2 and f < 0.45) else ghiaccio
            toccate += 1
            continue

        if b in CALDI:
            if 2 <= prof <= 14 and g > 0.62 and f < 0.55:
                tipo = CORALLI[int(g2 * len(CORALLI)) % len(CORALLI)]
                out[lx, fondo, lz] = s.blocco("coral_block", coral_type=tipo,
                                              dead="false")
                if out[lx, fondo + 1, lz] == acqua:
                    r = f2
                    if r < 0.45:
                        out[lx, fondo + 1, lz] = s.blocco(
                            "coral", coral_type=tipo, dead="false")
                    elif r < 0.75:
                        out[lx, fondo + 1, lz] = s.blocco(
                            "coral_fan", coral_type=tipo, facing="up",
                            dead="false")
                    elif r < 0.85:
                        out[lx, fondo + 1, lz] = pickle
                toccate += 1
                continue
            if prof >= 2 and f < 0.10 and out[lx, fondo + 1, lz] == acqua:
                out[lx, fondo + 1, lz] = erba_mare
                toccate += 1
            continue

        # -- fondo di argilla o di fango: sui fiumi e sul basso fondo --------
        fiume = b == "river" or int(cls[lx, lz]) == M.FIUME
        if prof <= 5 and g2 > 0.74 and (fiume or int(cls[lx, lz]) == M.MARE):
            out[lx, fondo, lz] = fango if (fiume and g2 > 0.90) else argilla
            toccate += 1

        suolo = out[lx, fondo, lz]
        sabbioso = suolo in (s.blocco("sand"), s.blocco("gravel"), argilla)

        # -- alghe: a gruppi sul fondo profondo del mare -------------------
        if (b in CON_ALGHE and 5 <= prof <= 26 and g > 0.55 and f < 0.22
                and sabbioso):
            altezza = min(prof - 1, 3 + int(f2 * 9))
            for k in range(1, altezza):
                out[lx, fondo + k, lz] = stelo
            out[lx, fondo + altezza, lz] = s.blocco("kelp", age=str(int(f2 * 25)))
            toccate += 1
            continue

        # -- erba di mare, singola e alta ---------------------------------
        soglia = 0.16 if g > 0.40 else 0.025
        if f < soglia and sabbioso and out[lx, fondo + 1, lz] == acqua:
            if prof >= 2 and f2 < 0.45 and out[lx, fondo + 2, lz] == acqua:
                out[lx, fondo + 1, lz] = alta_b
                out[lx, fondo + 2, lz] = alta_a
            else:
                out[lx, fondo + 1, lz] = erba_mare
            toccate += 1
            continue

        # -- ninfee: pelo dell'acqua di fiumi e laghi bassi -------------------
        if (fiume and prof <= 4 and g > 0.50 and f < 0.10
                and out[lx, sopra + 1, lz] == s.id_aria):
            out[lx, sopra + 1, lz] = ninfea
            toccate += 1
    return toccate
