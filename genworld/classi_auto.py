"""Classificazione adattiva: nessuna soglia scritta a mano.

Il classificatore a soglie HSV di `mappa.py` funziona sulla mappa su cui e'
stato tarato. Qui l'approccio e' rovesciato: si raggruppano i colori che
l'immagine contiene davvero, e poi si decide che cos'e' ciascun gruppo con
criteri RELATIVI - "il piu' blu", "il piu' luminoso e smorto", "il piu' verde
e scuro" - che restano veri qualunque sia la tavolozza.

Perche' Lab e non HSV: in Lab l'asse b va dal blu (negativo) al giallo
(positivo) e l'asse a dal verde (negativo) al rosso. "Acqua = b negativo" e
"vegetazione = a negativo" sono affermazioni percettive, non numeri tarati su
un'immagine. E' il piu' vicino a una regola universale che si possa avere
senza una legenda dichiarata.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_dilation, uniform_filter

from .mappa import (DESERTO, FORESTA, MARE, MONTAGNA, NEVE, OCEANO, PIANURA,
                    PRATERIA, SPIAGGIA)
from .normalizza import a_lab


def raggruppa(rgb: np.ndarray, n: int = 14, seed: int = 0,
              campione: int = 60000) -> tuple[np.ndarray, np.ndarray]:
    """k-means in Lab. Ritorna (mappa dei gruppi, centri Lab)."""
    lab = a_lab(rgb)
    piatto = lab.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    idx = rng.choice(piatto.shape[0], min(campione, piatto.shape[0]), replace=False)
    centri, _ = kmeans2(piatto[idx], n, minit="++", seed=seed, iter=40)

    # assegna ogni pixel al centro piu' vicino, a blocchi per non esplodere in RAM
    gruppi = np.empty(piatto.shape[0], dtype=np.int16)
    passo = 500_000
    for i in range(0, piatto.shape[0], passo):
        blocco = piatto[i:i + passo]
        d = ((blocco[:, None, :] - centri[None, :, :]) ** 2).sum(2)
        gruppi[i:i + passo] = d.argmin(1)
    return gruppi.reshape(lab.shape[:2]), centri


def _statistiche(gruppi: np.ndarray, centri: np.ndarray,
                 rgb: np.ndarray) -> list[dict]:
    """Per ogni gruppo: colore, area, contatto col bordo, grana."""
    lum = a_lab(rgb)[..., 0]
    media = uniform_filter(lum, 5)
    grana = np.sqrt(np.maximum(uniform_filter(lum * lum, 5) - media * media, 0))

    bordo = np.zeros(gruppi.shape, bool)
    m = max(2, int(min(gruppi.shape) * 0.01))
    bordo[:m, :] = bordo[-m:, :] = bordo[:, :m] = bordo[:, -m:] = True

    out = []
    for k in range(len(centri)):
        msk = gruppi == k
        area = float(msk.mean())
        L, a, b = centri[k]
        out.append({
            "id": k, "L": float(L), "a": float(a), "b": float(b),
            "croma": float(np.hypot(a, b)), "area": area,
            "bordo": float((msk & bordo).sum() / max(1, msk.sum())) if area > 0 else 0.0,
            "grana": float(grana[msk].mean()) if area > 0 else 0.0,
        })
    return out


def assegna(stat: list[dict]) -> dict[int, int]:
    """Da gruppi di colore a classi di terreno, con criteri relativi.

    L'ordine e' quello della confidenza decrescente: prima l'acqua, che e' il
    segnale piu' netto e la piu' estesa; poi la neve, che e' un estremo di
    luminosita'; poi la vegetazione e l'arido, che si distinguono lungo l'asse
    verde-giallo. Cio' che resta e' pianura.
    """
    m: dict[int, int] = {}

    # --- acqua: b negativo (versante blu). Il contatto col bordo conferma. ---
    acqua = [s for s in stat if s["b"] < -2.0]
    if not acqua:   # mappa senza blu: si prende comunque il gruppo piu' freddo
        acqua = [min(stat, key=lambda s: s["b"])]
    # profonda contro bassa: si separano sulla luminosita' interna all'acqua
    if len(acqua) > 1:
        soglia = float(np.median([s["L"] for s in acqua]))
        for s in acqua:
            m[s["id"]] = OCEANO if s["L"] <= soglia else MARE
    else:
        m[acqua[0]["id"]] = MARE

    terra = [s for s in stat if s["id"] not in m]
    if not terra:
        return m

    croma_max = max(s["croma"] for s in terra) or 1.0
    L_terra = [s["L"] for s in terra]
    L_max = max(L_terra)

    # --- neve: estremo luminoso e smorto ---
    for s in terra:
        if s["L"] > L_max - 8 and s["croma"] < 0.30 * croma_max:
            m[s["id"]] = NEVE

    resto = [s for s in terra if s["id"] not in m]
    if not resto:
        return m

    # --- arido contro vegetato, lungo l'asse verde-giallo ---
    # "giallezza" alta = sabbia/steppa secca; "verdezza" alta = vegetazione
    for s in resto:
        s["_giallo"] = s["b"] - (-s["a"])     # giallo, al netto del verde
        s["_verde"] = -s["a"]

    gialli = sorted(resto, key=lambda s: -s["_giallo"])
    verdi = sorted(resto, key=lambda s: -s["_verde"])

    n_des = max(1, round(len(resto) * 0.30))
    for s in gialli[:n_des]:
        m[s["id"]] = DESERTO

    rimasti = [s for s in resto if s["id"] not in m]
    if rimasti:
        # foresta: fra i verdi, i piu' scuri
        L_med = float(np.median([s["L"] for s in rimasti]))
        for s in rimasti:
            if s["_verde"] > 0 and s["L"] < L_med:
                m[s["id"]] = FORESTA
        for s in rimasti:
            if s["id"] not in m:
                m[s["id"]] = PRATERIA if s["L"] < L_med else PIANURA
    return m


def classifica_adattiva(rgb: np.ndarray, rug: np.ndarray | None = None,
                        n_gruppi: int = 14, soglia_montagna: float = 0.30,
                        seed: int = 0) -> tuple[np.ndarray, list[dict]]:
    """Classificazione senza soglie tarate a mano."""
    from .mappa import ACQUA, rugosita
    if rug is None:
        rug = rugosita(rgb)

    gruppi, centri = raggruppa(rgb, n=n_gruppi, seed=seed)
    stat = _statistiche(gruppi, centri, rgb)
    mappa_classi = assegna(stat)

    cls = np.zeros(gruppi.shape, dtype=np.uint8)
    for k, c in mappa_classi.items():
        cls[gruppi == k] = c

    # montagna e spiaggia restano geometriche: non dipendono dalla tavolozza
    acqua = np.isin(cls, ACQUA)
    cls[(~acqua) & (rug > soglia_montagna)] = MONTAGNA
    bordo = binary_dilation(acqua, iterations=2) & ~acqua
    cls[bordo & (cls != MONTAGNA) & (cls != NEVE)] = SPIAGGIA

    for s in stat:
        s["classe"] = mappa_classi.get(s["id"])
    return cls, stat
