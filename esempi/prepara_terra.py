"""Prepara la TERRA INTERA per GenWorld (quote e fondali di ETOPO 2022).

Un solo file globale a 60 secondi d'arco (circa 1,85 km): terra, mare e ghiacci in un'unica
griglia, che a 2-5 km per blocco e' piu' che sufficiente (a questa scala i riquadri da 30 m
di Copernicus sarebbero migliaia di richieste per niente). Si scarica "ice surface": dove c'e'
la calotta (Groenlandia, Antartide) vale la quota del GHIACCIO, non della roccia sotto.

Fonte: NOAA National Centers for Environmental Information, ETOPO 2022 Global Relief Model,
DOI 10.25921/fd45-gt74 (da citare). ~470 MB, scaricati una volta in `input/dati_terra/`.

Uso:
    python esempi/prepara_terra.py --lato 8192 --uscita input/terra_dem.npz
    python esempi/genera_mappa.py --dem input/terra_dem.npz --lato 8192 ...

La griglia e' equirettangolare (longitudine e latitudine in linea retta, rapporto 2:1): le alte
latitudini sono stirate in larghezza, come nelle carte piane. Il file contiene la latitudine di
ogni cella (ridotta) e la scala verticale: 36 m per blocco, cosi' l'Everest (8848 m) sta a
y = 62 + 246 = 308, sotto il limite di 320.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import numpy as np

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = ("https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/"
       "ETOPO_2022_v1_60s_N90W180_surface.tif")


def scarica(destinazione: str) -> None:
    if os.path.exists(destinazione) and os.path.getsize(destinazione) > 400_000_000:
        return
    print(f"scarico {URL} (circa 470 MB)...", flush=True)
    tmp = destinazione + ".parte"
    with urllib.request.urlopen(URL, timeout=600) as r, open(tmp, "wb") as f:
        totale = int(r.headers.get("content-length", 0))
        letti = 0
        while True:
            blocco = r.read(1 << 20)
            if not blocco:
                break
            f.write(blocco)
            letti += len(blocco)
            if totale and letti % (50 << 20) < (1 << 20):
                print(f"  {letti >> 20} / {totale >> 20} MB", flush=True)
    os.replace(tmp, destinazione)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lato", type=int, default=8192, help="celle in larghezza (la meta' in altezza)")
    ap.add_argument("--proiezione", choices=("mercatore", "equirettangolare"), default="mercatore",
                    help="mercatore (predefinita): la Terra tra 85 gradi N e S in un quadrato, senza "
                         "spazio sprecato; equirettangolare: 2:1, il mondo quadrato resta mezzo oceano")
    ap.add_argument("--sud", type=float, default=-90.0)
    ap.add_argument("--nord", type=float, default=90.0)
    ap.add_argument("--uscita", default=os.path.join(RADICE, "input", "terra_dem.npz"))
    ap.add_argument("--cache", default=os.path.join(RADICE, "input", "dati_terra"))
    ap.add_argument("--metri-per-blocco", type=float, default=36.0)
    a = ap.parse_args()

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

    os.makedirs(a.cache, exist_ok=True)
    sorgente = os.path.join(a.cache, "ETOPO_2022_v1_60s_N90W180_surface.tif")
    scarica(sorgente)

    nx = a.lato
    if a.proiezione == "mercatore":
        # Web Mercator: un quadrato da -85,05 a +85,05 gradi, i blocchi sono quadrati in ogni punto
        # ma vicino ai poli rappresentano meno terra (a 60 N circa la meta' che all'equatore)
        R = 6378137.0
        mezzo = np.pi * R
        ny = nx
        T = from_bounds(-mezzo, -mezzo, mezzo, mezzo, nx, ny)
        crs = "EPSG:3857"
        passo_val = 2 * mezzo / nx
        print(f"griglia {nx} x {ny} (Mercatore): {passo_val / 1000:.2f} km per blocco all'equatore", flush=True)
    else:
        ny = int(round(nx * (a.nord - a.sud) / 360.0))
        T = from_bounds(-180.0, a.sud, 180.0, a.nord, nx, ny)
        crs = "EPSG:4326"
        passo_val = 360.0 / nx
        print(f"griglia {nx} x {ny}: {360.0 / nx * 111.32:.2f} km per blocco all'equatore", flush=True)
    with rasterio.open(sorgente) as src:
        elev = np.zeros((ny, nx), np.float32)
        reproject(rasterio.band(src, 1), elev, dst_transform=T, dst_crs=crs,
                  resampling=Resampling.average, src_nodata=src.nodata, dst_nodata=0.0)
    print(f"quote {elev.min():.0f}..{elev.max():.0f} m, terra {float((elev > 0.5).mean()) * 100:.0f}%",
          flush=True)

    # latitudine di ogni cella, a risoluzione ridotta (1 ogni 32)
    sub = 32
    righe = np.arange(0, ny, sub)
    if a.proiezione == "mercatore":
        y = mezzo - (righe + 0.5) * passo_val
        lat = np.degrees(np.arctan(np.sinh(y / R))).astype(np.float32)
        x0, y1 = -mezzo, mezzo
    else:
        lat = (a.nord - (righe + 0.5) * (a.nord - a.sud) / ny).astype(np.float32)
        x0, y1 = -180.0, a.nord
    lat_g = np.tile(lat[:, None], (1, len(np.arange(0, nx, sub))))
    np.savez_compressed(a.uscita, elev=elev, lago=np.zeros((ny, nx), bool), x0=x0, y1=y1,
                        passo=passo_val, nx=nx, ny=ny, crs=crs, lat=lat_g, lat_sub=sub,
                        metri_per_blocco=a.metri_per_blocco)
    print(f"scritto {a.uscita}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
