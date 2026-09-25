"""Classificazione guidata da pochi campioni indicati dall'utente.

Perche' esiste. Provata su cinque mappe reali di stili diversi, la
classificazione automatica funziona bene su due, mediocre su una e fallisce su
due. Su una mappa seppia con cornice illustrata trova l'1% di acqua invece del
30%: non e' una taratura da aggiustare, e' che in quell'immagine il colore
dell'acqua e quello della terra si sovrappongono, e nessuna regola cromatica
puo' separarli.

L'informazione mancante ce l'ha la persona che guarda la mappa. Sei clic con il
contagocce - "questo e' mare", "questo e' montagna", "questo e' decorazione da
ignorare" - valgono piu' di qualunque euristica, perche' portano la semantica
che l'immagine da sola non contiene.

E' classificazione supervisionata con pochissime etichette: si prende il
campione piu' vicino in CIELAB, dove le distanze corrispondono a differenze
percepite. Nessuna soglia, nessuna assunzione sulla tavolozza.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, median_filter

from .mappa import ACQUA, MONTAGNA, NEVE, SPIAGGIA
from .normalizza import a_lab

# Classe speciale: cornici decorative, legende, cartigli, rose dei venti.
# Non e' terreno e non deve finire nel mondo.
DECORO = 250


def campiona(rgb: np.ndarray, punti: dict[int, list[tuple[float, float]]],
             raggio: int = 4) -> list[tuple[np.ndarray, int]]:
    """Dai clic ai colori di riferimento.

    `punti` mappa una classe alle posizioni normalizzate (x, y) in 0..1.
    Si prende la MEDIANA di un intorno, non il pixel singolo: su una mappa
    dipinta un pixel isolato puo' essere un granello di rumore.
    """
    h, w = rgb.shape[:2]
    lab = a_lab(rgb)
    out = []
    for classe, lista in punti.items():
        for x, y in lista:
            cx, cy = int(x * w), int(y * h)
            f = lab[max(0, cy - raggio): cy + raggio + 1,
                    max(0, cx - raggio): cx + raggio + 1].reshape(-1, 3)
            if len(f):
                out.append((np.median(f, axis=0), classe))
    return out


def classifica_guidata(rgb: np.ndarray, campioni: list[tuple[np.ndarray, int]],
                       rug: np.ndarray | None = None,
                       soglia_montagna: float | None = 0.30,
                       pulisci: int = 5) -> np.ndarray:
    """Assegna a ogni pixel la classe del campione piu' vicino in Lab."""
    if not campioni:
        raise ValueError("servono almeno due campioni")
    lab = a_lab(rgb)
    rif = np.stack([c for c, _ in campioni])
    classi = np.array([k for _, k in campioni], dtype=np.uint8)

    piatto = lab.reshape(-1, 3)
    fuori = np.empty(piatto.shape[0], dtype=np.uint8)
    passo = 400_000
    for i in range(0, piatto.shape[0], passo):
        b = piatto[i:i + passo]
        d = ((b[:, None, :] - rif[None, :, :]) ** 2).sum(2)
        fuori[i:i + passo] = classi[d.argmin(1)]
    cls = fuori.reshape(lab.shape[:2])

    # filtro di mediana: toglie il sale-e-pepe senza spostare i confini
    if pulisci > 1:
        cls = median_filter(cls, size=pulisci)

    # montagna e spiaggia restano geometriche, come nel percorso automatico
    if soglia_montagna is not None and rug is not None:
        terra = ~np.isin(cls, ACQUA) & (cls != DECORO)
        cls[terra & (rug > soglia_montagna)] = MONTAGNA
    acqua = np.isin(cls, ACQUA)
    bordo = binary_dilation(acqua, iterations=2) & ~acqua & (cls != DECORO)
    cls[bordo & (cls != MONTAGNA) & (cls != NEVE)] = SPIAGGIA
    return cls
