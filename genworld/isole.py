"""Isole del tesoro: un'isoletta in mare aperto con una croce di sabbia rossa.

In mezzo al mare, lontano da ogni costa, sale dal fondale un'isola piccola: un
disco di sabbia con un cuore d'erba, due palme e, su un lato, una croce di sabbia
rossa. Sotto la croce, a due blocchi di profondita', c'e' un forziere col tesoro
(vedi `bauli.BOTTINO_ZONA["isola"]`). E' una tappa della catena degli indizi come
le rovine: le mappe del tesoro (vedi `mappe.py`) mostrano la croce.

L'isola si costruisce nel TERRENO (`applica` alza il fondale fino alla spiaggia,
con un fianco dolce), non si posa come una struttura: l'acqua attorno resta
quella del mare. La posa (`posa`) mette solo palme, croce e forziere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import mappa as M

LATO_PIATTO = 5          # raggio del disco in piano (sabbia e erba)
FIANCO = 14              # celle di pendio dal bordo del disco al fondale
RAGGIO_TOTALE = LATO_PIATTO + FIANCO
PROFONDITA_MINIMA = 5    # il fondale deve stare almeno cosi' sotto il mare, ovunque
DISTANZA_DALLA_COSTA = 14
DISTANZA_FRA_ISOLE = 140
CROCE = 2                # mezza diagonale della croce


@dataclass
class Isola:
    x: int                  # centro, coordinate di mappa
    z: int
    cima: int               # quota della superficie (primo blocco libero)
    seme: int = 0

    @property
    def croce(self) -> tuple[int, int]:
        """Il centro della croce: due celle a ovest e una a sud del centro."""
        return self.x - 2, self.z + 1

    def forzieri(self) -> list[tuple[int, int, int, str]]:
        """(x, y, z, tipo): il forziere e' due blocchi sotto la superficie della croce."""
        cx, cz = self.croce
        return [(cx, self.cima - 3, cz, "tesoro")]


def _disturbo(x: np.ndarray, z: np.ndarray, seme: int) -> np.ndarray:
    """Un rumore in [-1, 1] che dipende solo dalle coordinate di mondo."""
    h = (x.astype(np.int64) * 73856093) ^ (z.astype(np.int64) * 19349663) ^ (np.int64(seme) * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF) / 32767.5 - 1.0


def pianifica(cls: np.ndarray, h: np.ndarray, livello_mare: int = 62,
              evita: np.ndarray | None = None, densita: float = 1.0, seed: int = 0,
              celle_per: int = 250_000) -> list[Isola]:
    """Siti in mare aperto: tutto attorno, fino a RAGGIO_TOTALE, e' mare profondo
    almeno PROFONDITA_MINIMA, e a DISTANZA_DALLA_COSTA oltre quel raggio non c'e'
    terra. Una ogni `celle_per` celle di mappa."""
    from scipy.ndimage import distance_transform_edt, maximum_filter
    H, W = cls.shape
    quanti = int(max(0, round(H * W / celle_per * densita)))
    if quanti <= 0:
        return []
    marino = np.isin(cls, M.MARINO)
    terra = ~marino
    dalla_costa = distance_transform_edt(marino)            # celle di mare fino alla terra piu' vicina
    prof_ok = maximum_filter(h, size=2 * RAGGIO_TOTALE + 1) <= livello_mare - PROFONDITA_MINIMA
    ok = marino & prof_ok & (dalla_costa >= RAGGIO_TOTALE + DISTANZA_DALLA_COSTA)
    ok[:RAGGIO_TOTALE + 20, :] = False
    ok[-(RAGGIO_TOTALE + 20):, :] = False
    ok[:, :RAGGIO_TOTALE + 20] = False
    ok[:, -(RAGGIO_TOTALE + 20):] = False
    if evita is not None:
        from scipy.ndimage import binary_dilation
        ok &= ~binary_dilation(evita, iterations=RAGGIO_TOTALE)
    zz, xx = np.nonzero(ok)
    fuori: list[Isola] = []
    if zz.size == 0:
        return fuori
    rng = np.random.default_rng(seed + 9090)
    for k in rng.permutation(zz.size):
        x, z = int(xx[k]), int(zz[k])
        if any((x - i.x) ** 2 + (z - i.z) ** 2 < DISTANZA_FRA_ISOLE ** 2 for i in fuori):
            continue
        fuori.append(Isola(x=x, z=z, cima=livello_mare + 3,
                           seme=int(rng.integers(0, 2 ** 31 - 1))))
        if len(fuori) >= quanti:
            break
    return fuori


def applica(h: np.ndarray, cls: np.ndarray, isole: list[Isola], livello_mare: int = 62) -> None:
    """Alza il fondale in un'isola, sul posto (`h` e `cls`, coordinate di mappa).
    Disco piano a `cima` con un bordo irregolare, poi un fianco che scende al
    fondale. La sabbia dove la quota e' bassa, l'erba nel cuore."""
    H, W = h.shape
    for isl in isole:
        z0, z1 = max(0, isl.z - RAGGIO_TOTALE - 2), min(H, isl.z + RAGGIO_TOTALE + 3)
        x0, x1 = max(0, isl.x - RAGGIO_TOTALE - 2), min(W, isl.x + RAGGIO_TOTALE + 3)
        zz, xx = np.mgrid[z0:z1, x0:x1]
        d = np.hypot(xx - isl.x, zz - isl.z) + 1.2 * _disturbo(xx, zz, isl.seme)
        t = np.clip((d - LATO_PIATTO) / FIANCO, 0.0, 1.0)
        base = h[z0:z1, x0:x1].astype(np.float32)
        nuova = np.round(base + (isl.cima - base) * (1.0 - t) ** 1.5).astype(np.int32)
        nuova = np.where(d <= LATO_PIATTO, isl.cima, nuova)
        nuova = np.where(d <= 2.0, isl.cima + 1, nuova)         # una piccola duna al centro
        alza = nuova > h[z0:z1, x0:x1]
        h[z0:z1, x0:x1] = np.where(alza, nuova, h[z0:z1, x0:x1])
        terra = alza & (nuova > livello_mare)
        sub = cls[z0:z1, x0:x1]
        sub[terra] = M.SPIAGGIA
        sub[terra & (d <= LATO_PIATTO - 2.5)] = M.PRATERIA


def maschera(isole: list[Isola], forma: tuple[int, int], margine: int = 0) -> np.ndarray:
    occ = np.zeros(forma, bool)
    H, W = forma
    r = RAGGIO_TOTALE + margine
    for isl in isole:
        occ[max(0, isl.z - r):isl.z + r + 1, max(0, isl.x - r):isl.x + r + 1] = True
    return occ


def indice_per_chunk(isole: list[Isola], passo: int = 16) -> dict:
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, isl in enumerate(isole):
        r = LATO_PIATTO + 2
        for cz in range((isl.z - r) // passo, (isl.z + r) // passo + 1):
            for cx in range((isl.x - r) // passo, (isl.x + r) // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


class Tavolozza:
    def __init__(self, s):
        self.aria = s.id_aria
        self.sabbia_rossa = s.blocco("red_sand")
        self.tronco = s.blocco("log", axis="y", material="jungle", stripped="false")
        self.foglie = s.blocco("leaves", material="jungle", distance="1", persistent="true",
                               check_decay="false")
        self.forziere = s.blocco("chest", facing="south", connection="none", material="wood")
        self.cespuglio = s.blocco("plant", plant_type="dead_bush")
        self.bauli: list = []


def posa(out: np.ndarray, tav: Tavolozza, h_c: np.ndarray, ox: int, oz: int, y0: int,
         isole: list[Isola], quali: list[int], meta: int = 0) -> int:
    """Palme, croce e forziere dentro il chunk (16, H, 16). `ox, oz`: angolo del
    chunk in coordinate di GIOCO; le isole sono in coordinate di mappa (`meta`)."""
    H = out.shape[1]
    messi = 0

    def metti(x: int, y: int, z: int, blocco: int, solo_aria: bool = False) -> None:
        lx, lz, ly = x - meta - ox, z - meta - oz, y - y0
        if 0 <= lx < 16 and 0 <= lz < 16 and 1 <= ly < H:
            if not solo_aria or out[lx, ly, lz] == tav.aria:
                out[lx, ly, lz] = blocco

    for i in quali:
        isl = isole[i]
        sup = isl.cima                       # il primo blocco libero; la superficie sta a sup - 1
        # --- due palme: tronco di giungla e chioma a ventaglio ---
        for (dx, dz, alt) in ((0, 0, 6), (3, -2, 5)):
            px, pz = isl.x + dx, isl.z + dz
            base = sup + (1 if (dx, dz) == (0, 0) else 0)
            for k in range(alt):
                metti(px, base + k, pz, tav.tronco)
            cima = base + alt
            for ax, az in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                metti(px + ax, cima, pz + az, tav.foglie, solo_aria=True)
            for ax, az in ((2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                metti(px + ax, cima - 1, pz + az, tav.foglie, solo_aria=True)
            metti(px, cima + 1, pz, tav.foglie, solo_aria=True)
            messi += 1
        # --- cespugli secchi sulla spiaggia ---
        for (dx, dz) in ((-3, -2), (2, 3), (4, 1)):
            metti(isl.x + dx, sup, isl.z + dz, tav.cespuglio, solo_aria=True)
        # --- la croce di sabbia rossa ---
        cx, cz = isl.croce
        for k in range(-CROCE, CROCE + 1):
            metti(cx + k, sup - 1, cz + k, tav.sabbia_rossa)
            metti(cx + k, sup - 1, cz - k, tav.sabbia_rossa)
        # --- il forziere, sepolto sotto il centro della croce ---
        for (x, y, z, tipo) in isl.forzieri():
            metti(x, y, z, tav.forziere)
            lx, lz = x - meta - ox, z - meta - oz
            if 0 <= lx < 16 and 0 <= lz < 16:
                tav.bauli.append((x, y, z, tipo))
    return messi
