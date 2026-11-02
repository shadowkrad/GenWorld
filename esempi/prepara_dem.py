"""Prepara un modello di quote reali (Copernicus GLO-30 + batimetria EMODnet) per GenWorld.

Scarica SOLO cio' che serve, a risoluzione ridotta: i riquadri GLO-30 sono COG con
sovracampioni, quindi si leggono a blocchi grossolani via HTTP; la batimetria arriva a
pezzi da 3 gradi dal WCS di EMODnet (il server rifiuta richieste troppo grandi). Tutto
finisce in un solo `.npz` che `genera_mappa.py --dem` legge al posto dell'immagine.

Fonti e licenze:
  - Copernicus DEM GLO-30 (ESA): uso libero con la licenza Copernicus, va citato.
    https://registry.opendata.aws/copernicus-dem/
  - EMODnet Bathymetry DTM: https://emodnet.ec.europa.eu/en/bathymetry (citare EMODnet)

Uso (Italia, il predefinito):
    python esempi/prepara_dem.py --lato 4096 --uscita input/italia_dem.npz
Uso (Europa):
    python esempi/prepara_dem.py --lato 4096 --ovest -25 --est 45 --sud 34 --nord 71.5 \\
        --uscita input/europa_dem.npz --cache input/dati_europa
    (serve `pip install "rasterio<1.5"` con numpy<2)

La griglia e' in proiezione LAEA Europa (EPSG:3035): blocchi quadrati in metri veri.
I riquadri si leggono in parallelo (--thread).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Windows (schannel) a volte non raggiunge i server di revoca dei certificati e rifiuta TUTTO
# ("CERT_TRUST_REVOCATION_STATUS_UNKNOWN"): si salta solo quel controllo, la catena resta verificata.
os.environ.setdefault("GDAL_HTTP_SSL_OPTIONS", "NO_REVOKE")

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
    ap.add_argument("--thread", type=int, default=4)
    a = ap.parse_args()

    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, transform, transform_bounds
    from rasterio.windows import Window  # noqa: F401  (documenta l'idea: si scrive in una finestra)

    os.makedirs(a.cache, exist_ok=True)
    dst_crs = CRS.from_epsg(3035)
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", dst_crs, a.ovest, a.sud, a.est, a.nord, densify_pts=41)
    larg, alt = x1 - x0, y1 - y0
    passo = max(larg, alt) / a.lato
    nx, ny = int(round(larg / passo)), int(round(alt / passo))
    T = from_origin(x0, y1, passo, passo)
    print(f"griglia {nx} x {ny}, {passo:.0f} m per blocco", flush=True)

    dem = np.full((ny, nx), np.nan, np.float32)
    blocco = threading.Lock()
    contatori = {"letti": 0, "mancanti": 0}
    t0 = time.time()

    def finestra(w, s, e, n):
        """Righe/colonne della griglia coperte da un riquadro in gradi (con un po' di margine)."""
        bx0, by0, bx1, by1 = transform_bounds("EPSG:4326", dst_crs, w, s, e, n, densify_pts=21)
        c0 = max(0, int((bx0 - x0) / passo) - 2)
        c1 = min(nx, int((bx1 - x0) / passo) + 3)
        r0 = max(0, int((y1 - by1) / passo) - 2)
        r1 = min(ny, int((y1 - by0) / passo) + 3)
        return r0, r1, c0, c1

    def leggi_riquadro(lat_lon):
        lat, lon = lat_lon
        url = URL_DEM.format(ns="N", lat=lat, ew="E" if lon >= 0 else "W", lon=abs(lon))
        r0, r1, c0, c1 = finestra(lon, lat, lon + 1, lat + 1)
        if r1 <= r0 or c1 <= c0:
            return
        for tentativo in range(6):
            try:
                with rasterio.Env(GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2"):
                    with rasterio.open(url) as src:
                        parte = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
                        Ts = rasterio.Affine(passo, 0, x0 + c0 * passo, 0, -passo, y1 - r0 * passo)
                        reproject(rasterio.band(src, 1), parte, dst_transform=Ts, dst_crs=dst_crs,
                                  resampling=Resampling.average, src_nodata=None, dst_nodata=np.nan)
                with blocco:
                    sub = dem[r0:r1, c0:c1]
                    ok = ~np.isnan(parte)
                    sub[ok] = parte[ok]
                    contatori["letti"] += 1
                return
            except Exception as e:
                testo = str(e)
                if any(k in testo for k in ("No such file", "404", "not recognized", "does not exist")):
                    with blocco:
                        contatori["mancanti"] += 1          # il mare non ha riquadri
                    return
                time.sleep(2 + 3 * tentativo)               # errore di rete: si riprova
        print(f"  ERRORE: riquadro {lat},{lon} non letto dopo 6 tentativi", flush=True)
        with blocco:
            contatori["errori"] = contatori.get("errori", 0) + 1

    riquadri = [(lat, lon) for lat in range(int(np.floor(a.sud)), int(np.floor(a.nord)) + 1)
                for lon in range(int(np.floor(a.ovest)), int(np.floor(a.est)) + 1)]
    print(f"{len(riquadri)} riquadri da provare", flush=True)
    with ThreadPoolExecutor(a.thread) as pool:
        for i, _ in enumerate(pool.map(leggi_riquadro, riquadri)):
            if i % 200 == 0:
                print(f"  {i}/{len(riquadri)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"quote: {contatori['letti']} riquadri letti, {contatori['mancanti']} senza dati (mare), "
          f"{contatori.get('errori', 0)} errori", flush=True)
    if contatori.get("errori"):
        raise SystemExit("alcuni riquadri non si sono letti: rilancia (la rete ha fallito)")

    # batimetria a pezzi da 3 gradi, a risoluzione adatta alla griglia
    bati = np.full((ny, nx), np.nan, np.float32)
    passo_g = 3.0
    lavori = []
    s = a.sud
    while s < a.nord:
        n = min(s + passo_g, a.nord)
        w = a.ovest
        while w < a.est:
            lavori.append((w, s, min(w + passo_g, a.est), n))
            w += passo_g
        s = n

    def leggi_bati(lav):
        w, s_, e, n = lav
        nome = os.path.join(a.cache, f"emodnet_{w:.1f}_{s_:.1f}.tif")
        try:
            if not os.path.exists(nome):
                # ~ 1,5 volte la risoluzione della griglia: basta e il server legge meno
                nxp = max(64, min(2880, int((e - w) * 111_000 * 0.74 / passo * 1.5)))
                nyp = max(64, min(2880, int((n - s_) * 111_000 / passo * 1.5)))
                url = URL_BATI.format(w=w, s=s_, e=e, n=n, nx=nxp, ny=nyp)
                with urllib.request.urlopen(url, timeout=600) as r:
                    dati = r.read()
                with open(nome, "wb") as f:
                    f.write(dati)
            r0, r1, c0, c1 = finestra(w, s_, e, n)
            with rasterio.open(nome) as src:
                parte = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
                Ts = rasterio.Affine(passo, 0, x0 + c0 * passo, 0, -passo, y1 - r0 * passo)
                reproject(rasterio.band(src, 1), parte, dst_transform=Ts, dst_crs=dst_crs,
                          resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=np.nan)
            with blocco:
                sub = bati[r0:r1, c0:c1]
                ok = ~np.isnan(parte)
                sub[ok] = parte[ok]
        except Exception as ex:
            print(f"  ATTENZIONE batimetria {w},{s_}: {ex}", flush=True)

    with ThreadPoolExecutor(min(4, a.thread)) as pool:
        for i, _ in enumerate(pool.map(leggi_bati, lavori)):
            if i % 20 == 0:
                print(f"  batimetria {i}/{len(lavori)} ({time.time() - t0:.0f} s)", flush=True)

    # unione: terra dove il DEM ha una quota vera. Il mare dentro i riquadri vale ESATTAMENTE 0;
    # la terra sotto il livello del mare (polder, delta) ha valori negativi non nulli: e' terra
    d0 = np.nan_to_num(dem, nan=0.0)
    sotto = (d0 < -0.01) & (d0 > -60.0)
    terra = (d0 > 0.5) | sotto
    mare = np.where(np.isnan(bati), -50.0, np.minimum(bati, -1.0))
    elev = np.where(d0 > 0.5, d0, np.where(sotto, 0.6, mare)).astype(np.float32)

    # laghi: acqua piatta nel DEM (le superfici d'acqua sono livellate da ESA) = celle uguali
    # alle vicine a meno di pochi centimetri, a terra, in una regione abbastanza grande
    from scipy.ndimage import label, maximum_filter, minimum_filter
    piatto = (maximum_filter(elev, size=3) - minimum_filter(elev, size=3) < 0.06) & (d0 > 1.0)
    lab, n = label(piatto)
    if n:
        aree = np.bincount(lab.ravel())
        grandi = np.nonzero(aree >= 25)[0]
        grandi = grandi[grandi > 0]
        lago = np.isin(lab, grandi)
    else:
        lago = np.zeros_like(piatto)
    print(f"laghi trovati: {int(lago.sum())} celle", flush=True)

    # latitudine di ogni cella, a risoluzione ridotta (serve per limite delle nevi e degli alberi)
    sub = 32
    cx = x0 + (np.arange(0, nx, sub) + 0.5) * passo
    cy = y1 - (np.arange(0, ny, sub) + 0.5) * passo
    gx, gy = np.meshgrid(cx, cy)
    lon_g, lat_g = transform(dst_crs, "EPSG:4326", gx.ravel().tolist(), gy.ravel().tolist())
    lat_g = np.asarray(lat_g, np.float32).reshape(gx.shape)

    np.savez_compressed(a.uscita, elev=elev, lago=lago, x0=x0, y1=y1, passo=passo,
                        nx=nx, ny=ny, crs="EPSG:3035", lat=lat_g, lat_sub=sub)
    print(f"scritto {a.uscita}: quote {elev.min():.0f}..{elev.max():.0f} m, "
          f"terra {float(terra.mean()) * 100:.0f}%", flush=True)


if __name__ == "__main__":
    sys.exit(main())
