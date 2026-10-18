"""Mondi da un modello di quote reali (vedi `esempi/prepara_dem.py`).

Al posto dell'immagine colorata, la forma del terreno viene da un `.npz` con le quote in
metri su una griglia di blocchi quadrati (proiezione LAEA Europa, EPSG:3035). Qui si
passa da metri a blocchi, da quote a classi di terreno, e dai laghi livellati del DEM al
`livello` dell'acqua del motore.

Scala verticale: 1 blocco ogni `METRI_PER_BLOCCO` metri sopra il mare (livello 62).
Il fondale e' compresso (radice), cosi' una piattaforma di 100 m e un abisso di 3500 m
restano distinguibili in meno di 60 blocchi.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from . import mappa as M

METRI_PER_BLOCCO = 28.0
LIVELLO_MARE = 62
PROFONDITA_MASSIMA = 56            # blocchi sotto il mare (fondo a y = 6)


def carica(percorso: str) -> dict:
    d = np.load(percorso, allow_pickle=False)
    return {"elev": d["elev"].astype(np.float32), "lago": d["lago"].astype(bool),
            "x0": float(d["x0"]), "y1": float(d["y1"]), "passo": float(d["passo"])}


def _adatta(a: np.ndarray, lato: int, riempi, ordine: int) -> tuple[np.ndarray, int, int, float]:
    """Porta la griglia a `lato` sul lato lungo (se serve) e la centra nel quadrato.
    Ritorna (array lato x lato, scarto z, scarto x, fattore di scala)."""
    from scipy.ndimage import zoom
    ny, nx = a.shape
    f = lato / max(ny, nx)
    if abs(f - 1.0) > 1e-6:
        a = zoom(a, f, order=ordine)
        ny, nx = a.shape
    fuori = np.full((lato, lato), riempi, a.dtype)
    oz, ox = (lato - ny) // 2, (lato - nx) // 2
    fuori[oz:oz + ny, ox:ox + nx] = a
    return fuori, oz, ox, f


def a_mondo(d: dict, lato: int) -> dict:
    """Quote in metri -> (classi, quote in blocchi, laghi) su una griglia lato x lato."""
    elev, oz, ox, f = _adatta(d["elev"], lato, -3000.0, 1)
    lago, _, _, _ = _adatta(d["lago"].astype(np.uint8), lato, 0, 0)
    lago = lago > 0
    passo = d["passo"] / f                                      # metri per cella
    terra = elev > 0.5

    # quote in blocchi: sopra il mare lineari, sotto compresse
    h = np.where(terra, LIVELLO_MARE + elev / METRI_PER_BLOCCO,
                 LIVELLO_MARE - np.minimum(PROFONDITA_MASSIMA, 3.0 * np.sqrt(np.maximum(-elev, 0) / 10.0)))
    h = gaussian_filter(h.astype(np.float32), 0.6)              # toglie la grana del campionamento
    h = np.where(terra, np.maximum(h, LIVELLO_MARE + 1), np.minimum(h, LIVELLO_MARE - 1))

    gz, gx = np.gradient(gaussian_filter(elev, 1.0))
    pend = np.hypot(gz, gx) / passo                              # metri su metro
    dal_mare = distance_transform_edt(terra)

    cls = np.full(elev.shape, M.OCEANO, np.uint8)
    cls[~terra & (elev > -200)] = M.MARE
    cls[terra] = M.PIANURA
    prateria = terra & (elev >= 200)
    cls[prateria] = M.PRATERIA
    foresta = terra & (elev >= 450) & (pend > 0.07)
    cls[foresta] = M.FORESTA
    montagna = terra & ((elev >= 1700) | (pend > 0.45))
    cls[montagna] = M.MONTAGNA
    cls[terra & (elev >= 2800)] = M.NEVE
    spiaggia = terra & (elev < 4.0) & (dal_mare <= 3) & (pend < 0.05)
    cls[spiaggia] = M.SPIAGGIA
    return {"cls": cls, "h": np.round(h).astype(np.int32), "lago": lago,
            "elev": elev, "passo": passo, "oz": oz, "ox": ox, "fattore": f}


def applica_laghi(cls: np.ndarray, h: np.ndarray, livello: np.ndarray, base: dict):
    """I laghi del DEM: acqua piatta alla quota vera del lago, con un fondo che scende
    verso il centro. Ritorna (cls, h, livello)."""
    lago = base["lago"]
    if not lago.any():
        return cls, h, livello
    elev = base["elev"]
    # il pelo e' costante in un lago: la mediana delle sue celle, per ogni specchio d'acqua
    from scipy.ndimage import label
    lab, n = label(lago)
    pelo = np.zeros(cls.shape, np.float32)
    for i in range(1, n + 1):
        m = lab == i
        pelo[m] = np.median(LIVELLO_MARE + elev[m] / METRI_PER_BLOCCO)
    dentro = distance_transform_edt(lago)
    fondo = pelo - np.minimum(12.0, 2.0 + dentro * 0.8)
    cls = cls.copy()
    cls[lago] = M.FIUME
    h = np.where(lago, np.round(fondo), h).astype(np.int32)
    livello = np.where(lago, pelo, livello).astype(np.float32)
    return cls, h, livello


def frazioni_da_lonlat(d: dict, lato: int, punti: list) -> list:
    """Da (lon, lat) alle frazioni 0..1 del mondo `lato` x `lato` (centrato)."""
    from rasterio.warp import transform
    xs, ys = transform("EPSG:4326", "EPSG:3035", [p[0] for p in punti], [p[1] for p in punti])
    ny, nx = d["elev"].shape
    f = lato / max(ny, nx)
    oz, ox = (lato - round(ny * f)) // 2, (lato - round(nx * f)) // 2
    fuori = []
    for x, y in zip(xs, ys):
        cx = (x - d["x0"]) / d["passo"] * f + ox
        cz = (d["y1"] - y) / d["passo"] * f + oz
        fuori.append((cx / lato, cz / lato))
    return fuori
