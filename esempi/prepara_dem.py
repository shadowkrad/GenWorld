"""Prepara un modello di quote reali (Copernicus GLO-30 + batimetria EMODnet) per GenWorld.

Scarica SOLO cio' che serve, a risoluzione ridotta: i riquadri GLO-30 sono COG con
sovracampioni, quindi si leggono a blocchi grossolani via HTTP; la batimetria arriva a
pezzi da 3 gradi dal WCS di EMODnet (il server rifiuta richieste troppo grandi). Tutto
finisce in un solo `.npz` che `genera_mappa.py --dem` legge al posto dell'immagine.

Fonti e licenze:
  - Copernicus DEM GLO-30 (ESA): uso libero con la licenza Copernicus, va citato.
    https://registry.opendata.aws/copernicus-dem/
  - EMODnet Bathymetry DTM: https://emodnet.ec.europa.eu/en/bathymetry (citare EMODnet)

Uso:
    python esempi/prepara_dem.py --lato 4096 --uscita input/italia_dem.npz
    (serve `pip install rasterio`)

La griglia e' in proiezione LAEA Europa (EPSG:3035): blocchi quadrati in metri veri.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

import numpy as np

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_"
           "{ew}{lon:03d}_00_DEM/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
URL_BATI = ("https://ows.emodnet-bathymetry.eu/wcs?service=WCS&version=1.0.0&request=GetCoverage"
            "&coverage=emodnet:mean&crs=EPSG:4326&bbox={w},{s},{e},{n}&format=image/tiff"
            "&width={nx}&height={ny}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lato", type=int, default=4096, help="celle sul lato lungo")
    ap.add_argument("--uscita", default=os.path.join(RADICE, "input", "italia_dem.npz"))
    ap.add_argument("--ovest", type=float, default=6.4)
    ap.add_argument("--est", type=float, default=18.9)
    ap.add_argument("--sud", type=float, default=36.3)
    ap.add_argument("--nord", type=float, default=47.2)
    ap.add_argument("--cache", default=os.path.join(RADICE, "input", "dati_italia"))
    a = ap.parse_args()

    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, transform_bounds

    os.makedirs(a.cache, exist_ok=True)
    dst_crs = CRS.from_epsg(3035)
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", dst_crs, a.ovest, a.sud, a.est, a.nord)
    larg, alt = x1 - x0, y1 - y0
    passo = max(larg, alt) / a.lato
    nx, ny = int(round(larg / passo)), int(round(alt / passo))
    T = from_origin(x0, y1, passo, passo)
    print(f"griglia {nx} x {ny}, {passo:.0f} m per blocco")

    dem = np.full((ny, nx), np.nan, np.float32)
    t0 = time.time()
    lat0, lat1 = int(np.floor(a.sud)), int(np.floor(a.nord))
    lon0, lon1 = int(np.floor(a.ovest)), int(np.floor(a.est))
    trovati = mancanti = 0
    for lat in range(lat0, lat1 + 1):
        for lon in range(lon0, lon1 + 1):
            url = URL_DEM.format(ns="N", lat=lat, ew="E" if lon >= 0 else "W", lon=abs(lon))
            try:
                with rasterio.open(url) as src:
                    parte = np.full((ny, nx), np.nan, np.float32)
                    reproject(rasterio.band(src, 1), parte, dst_transform=T, dst_crs=dst_crs,
                              resampling=Resampling.average, src_nodata=None, dst_nodata=np.nan)
                ok = ~np.isnan(parte)
                dem[ok] = parte[ok]
                trovati += 1
            except Exception as e:                      # il mare non ha riquadri
                if "No such file" in str(e) or "HTTP response code: 404" in str(e) or "not recognized" in str(e) \
                        or "does not exist" in str(e):
                    mancanti += 1
                    continue
                print(f"  ATTENZIONE {lat},{lon}: {e}")
                mancanti += 1
        print(f"  riga N{lat}: {trovati} riquadri finora ({time.time() - t0:.0f} s)")
    print(f"quote: {trovati} riquadri letti, {mancanti} senza dati (mare)")

    # batimetria a pezzi da 3 gradi
    bati = np.full((ny, nx), np.nan, np.float32)
    passo_g = 3.0
    s = a.sud
    while s < a.nord:
        n = min(s + passo_g, a.nord)
        w = a.ovest
        while w < a.est:
            e = min(w + passo_g, a.est)
            nome = os.path.join(a.cache, f"emodnet_{w:.1f}_{s:.1f}.tif")
            if not os.path.exists(nome):
                nxp, nyp = int((e - w) * 960), int((n - s) * 960)       # ~ 1/16 di primo
                url = URL_BATI.format(w=w, s=s, e=e, n=n, nx=nxp, ny=nyp)
                with urllib.request.urlopen(url, timeout=300) as r, open(nome, "wb") as f:
                    f.write(r.read())
            try:
                with rasterio.open(nome) as src:
                    parte = np.full((ny, nx), np.nan, np.float32)
                    reproject(rasterio.band(src, 1), parte, dst_transform=T, dst_crs=dst_crs,
                              resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=np.nan)
                ok = ~np.isnan(parte)
                bati[ok] = parte[ok]
            except Exception as ex:
                print(f"  ATTENZIONE batimetria {w},{s}: {ex}")
            w = e
        s = n
        print(f"  batimetria fino a {s:.1f}N ({time.time() - t0:.0f} s)")

    # unione: terra dove il DEM ha quota vera, altrimenti la batimetria (sempre sotto zero)
    terra = np.nan_to_num(dem, nan=0.0) > 0.5
    mare = np.where(np.isnan(bati), -50.0, np.minimum(bati, -1.0))
    elev = np.where(terra, np.nan_to_num(dem, nan=0.0), mare).astype(np.float32)

    # laghi: acqua piatta nel DEM (le superfici d'acqua sono livellate da ESA) = celle uguali
    # alle vicine a meno di pochi centimetri, a terra, in una regione abbastanza grande
    from scipy.ndimage import label, maximum_filter, minimum_filter
    piatto = (maximum_filter(elev, size=3) - minimum_filter(elev, size=3) < 0.06) & terra & (elev > 1.0)
    lab, n = label(piatto)
    if n:
        aree = np.bincount(lab.ravel())
        grandi = np.nonzero(aree >= 25)[0]
        grandi = grandi[grandi > 0]
        lago = np.isin(lab, grandi)
    else:
        lago = np.zeros_like(piatto)
    print(f"laghi trovati: {int(lago.sum())} celle")

    np.savez_compressed(a.uscita, elev=elev, lago=lago, x0=x0, y1=y1, passo=passo,
                        nx=nx, ny=ny, crs="EPSG:3035")
    print(f"scritto {a.uscita}: quote {elev.min():.0f}..{elev.max():.0f} m, "
          f"terra {float(terra.mean()) * 100:.0f}%")


if __name__ == "__main__":
    sys.exit(main())
