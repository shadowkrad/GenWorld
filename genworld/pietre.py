"""Pietre e terracotta per bioma (#30): la mesa a fasce e il vulcano lucido.

* **mesa** - nei deserti alti e aridi (bioma `badlands` su classe DESERTO) la
  superficie e' sabbia rossa e sotto la roccia e' a strati di terracotta di
  colori diversi, orizzontali e continui da un chunk all'altro (dipendono solo
  dalla quota): dove il pendio e' una parete si vedono le fasce.
* **vulcano** - sulle pendici e sull'orlo del cratere, pietra nera lucida,
  qualche blocco dorato e obsidiana che piange vicino alla lava.

Come per le sponde dei fossati, il disordine viene da un hash sulle coordinate
DI MONDO: con quelle locali al chunk il motivo si ripeterebbe a ogni confine.
"""
from __future__ import annotations

import numpy as np

from . import biomi as B
from . import mappa as M

# colori delle fasce dall'alto verso il basso, ripetuti con periodo 24 in quota
FASCE = ("terracotta", "terracotta", "orange", "orange", "terracotta", "yellow",
         "terracotta", "terracotta", "white", "terracotta", "brown", "brown",
         "terracotta", "red", "red", "terracotta", "light_gray", "terracotta",
         "orange", "terracotta", "terracotta", "yellow", "white", "terracotta")
PROFONDITA_FASCE = 22


def _rumore(x, z, sale):
    h = (x.astype(np.int64) * 73856093) ^ (z.astype(np.int64) * 19349663) \
        ^ (np.int64(sale) * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def _fascia(s, nome: str) -> int:
    if nome == "terracotta":
        return s.blocco("terracotta")
    return s.blocco("stained_terracotta", color=nome)


def mesa(out: np.ndarray, s, h_c, cls_c, biomi_c, nuda_c, y0: int) -> int:
    """Sabbia rossa in superficie e fasce di terracotta sotto. Ritorna le
    colonne cambiate."""
    mesa_m = (np.asarray(cls_c) == M.DESERTO) & \
        (np.asarray(biomi_c) == B.INDICE["badlands"])
    if not mesa_m.any():
        return 0
    H = out.shape[1]
    sabbia = s.blocco("red_sand")
    arenaria = s.blocco("red_sandstone", variant="normal")
    colori = [_fascia(s, n) for n in FASCE]
    nuda = (np.zeros((16, 16), bool) if nuda_c is None
            else np.asarray(nuda_c).astype(bool))
    n = 0
    for lx, lz in zip(*np.nonzero(mesa_m)):
        cima = int(h_c[lx, lz]) - 1 - y0
        if not (2 <= cima < H):
            continue
        for k in range(0, PROFONDITA_FASCE):
            ly = cima - k
            if ly < 1:
                break
            quota = ly + y0
            if k == 0 and not nuda[lx, lz]:
                out[lx, ly, lz] = sabbia
            elif k in (1, 2) and not nuda[lx, lz]:
                out[lx, ly, lz] = arenaria
            else:
                out[lx, ly, lz] = colori[quota % len(colori)]
        n += 1
    return n


def palude(out: np.ndarray, s, h_c, cls_c, biomi_c, ox: int, oz: int, y0: int) -> int:
    """Chiazze di fango e di argilla sul terreno della palude di mangrovie."""
    pal = np.asarray(biomi_c) == B.INDICE["mangrove_swamp"]
    pal &= ~np.isin(np.asarray(cls_c), M.ACQUA)
    if not pal.any():
        return 0
    H = out.shape[1]
    gx = ox + np.arange(16)[:, None] + np.zeros((1, 16), np.int64)
    gz = oz + np.arange(16)[None, :] + np.zeros((16, 1), np.int64)
    r, grumo = _rumore(gx, gz, 21), _rumore(gx // 6, gz // 6, 22)
    fango, argilla = s.blocco("mud"), s.blocco("clay")
    n = 0
    for lx, lz in zip(*np.nonzero(pal)):
        cima = int(h_c[lx, lz]) - 1 - y0
        if not (1 <= cima < H):
            continue
        if grumo[lx, lz] > 0.55 and r[lx, lz] < 0.75:
            out[lx, cima, lz] = fango
            n += 1
        elif r[lx, lz] < 0.03:
            out[lx, cima, lz] = argilla
            n += 1
    return n


def vulcano(out: np.ndarray, s, h_c, cls_c, lava_c, ox: int, oz: int, y0: int) -> int:
    """Pietra lucida sui fianchi, obsidiana attorno al cratere."""
    cls = np.asarray(cls_c)
    cono = np.isin(cls, (M.VULCANO, M.CRATERE))
    if not cono.any():
        return 0
    H = out.shape[1]
    gx = ox + np.arange(16)[:, None] + np.zeros((1, 16), np.int64)
    gz = oz + np.arange(16)[None, :] + np.zeros((16, 1), np.int64)
    r = _rumore(gx, gz, 11)
    lucida = s.blocco("polished_blackstone")
    dorata = s.blocco("gilded_blackstone")
    ossidiana = s.blocco("obsidian")
    piange = s.blocco("crying_obsidian")
    lava = (np.zeros((16, 16), bool) if lava_c is None else np.asarray(lava_c).astype(bool))
    n = 0
    for lx, lz in zip(*np.nonzero(cono)):
        cima = int(h_c[lx, lz]) - 1 - y0
        if not (1 <= cima < H) or lava[lx, lz]:
            continue
        v = float(r[lx, lz])
        if int(cls[lx, lz]) == M.CRATERE:
            out[lx, cima, lz] = piange if v < 0.12 else (ossidiana if v < 0.55 else out[lx, cima, lz])
        elif v < 0.07:
            out[lx, cima, lz] = lucida
        elif v < 0.075:
            out[lx, cima, lz] = dorata
        else:
            continue
        n += 1
    return n
