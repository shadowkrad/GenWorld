"""Il fondale del mare.

Misurato su Arda prima di questo modulo: il **53% di tutta l'acqua** stava a
esattamente y=54, un unico piano a otto blocchi di profondita' che copriva
meta' del mare; un altro 7% a y=30. Non erano un fondale, erano due terrazze,
e fra l'una e l'altra un muro - 2.836 celle con un salto di piu' di otto
blocchi in una cella, con punte di 67. Nuotando verso il largo non si
scendeva: si camminava su un pavimento, si cadeva da una rupe, si camminava
su un altro pavimento.

La causa era che la profondita' non si calcolava, si **tagliava**:
`altimetria` imponeva un tetto (-3 al mare, -12 all'oceano) e sotto non c'era
niente che generasse rilievo, quindi tutto si appiattiva contro il tetto. La
profondita' discendeva dalla CLASSE, e la rupe era il confine fra due classi.

Qui la profondita' discende dalla **distanza dalla costa**, che e' quello che
la determina anche in mare vero, e ha tre regimi:

* **piattaforma continentale** - dolce, fino a una decina di blocchi. E' dove
  si vede il fondo, dove si ancora, dove crescono le alghe.
* **scarpata** - il salto vero, ma disteso su decine di celle invece che su
  una. E' ripida, non verticale.
* **piana abissale** - profonda e quasi piatta, ma non piatta: il rilievo
  fBm ci mette dorsali e fosse.

Il rilievo cresce con la profondita': sulla piattaforma resta poco, cosi' le
spiagge non diventano un terreno accidentato, e al largo diventa molto, dove
nessuno inciampa.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .mappa import MARINO, OCEANO
from .rumore import fbm

# I tre regimi, in celle di distanza dalla costa e in blocchi di profondita'.
FINE_PIATTAFORMA = 34        # celle: dove finisce la piattaforma
FINE_SCARPATA = 70           # celle: dove comincia la piana
PROF_RIVA = 2.0              # blocchi appena staccati da terra
PROF_PIATTAFORMA = 11.0      # blocchi al bordo della piattaforma
PROF_PIANA = 36.0            # blocchi sulla piana abissale


def profondita(cls: np.ndarray, livello_mare: int = 62,
               rilievo: float = 1.0, seed: int = 0) -> np.ndarray:
    """Profondita' in blocchi per ogni cella marina (0 sulla terra)."""
    marino = np.isin(cls, MARINO)
    if not marino.any():
        return np.zeros(cls.shape, np.float32)

    # distanza dalla costa: la misura da cui discende tutto
    d = distance_transform_edt(marino).astype(np.float32)

    # Le isobate non seguono la costa come un'aureola. Presa cosi' com'e', la
    # distanza disegna anelli concentrici attorno a ogni isolotto - si vedono
    # benissimo su una carta batimetrica finta. Si deforma la distanza stessa
    # con un rumore largo, ed e' lo stesso gesto del raggio deformato del
    # cono vulcanico: il difetto e' la regolarita', non il rumore che manca.
    if rilievo > 0:
        lato = max(cls.shape)
        w = fbm(lato, ottave=3, celle_base=max(3, lato // 160),
                persistenza=0.55, seed=seed + 7) - 0.5
        d = d * (1.0 + 0.55 * w[:cls.shape[0], :cls.shape[1]])
        d = np.maximum(d, 0.0)

    # --- i tre regimi ---------------------------------------------------
    # Ogni tratto e' interpolato con una curva a S (smoothstep) invece che
    # con una retta: le giunzioni fra un regime e l'altro devono essere
    # morbide, altrimenti si ricrea a meta' strada lo stesso scalino che
    # stiamo togliendo, solo piu' piccolo.
    p = np.zeros_like(d)

    t1 = np.clip(d / FINE_PIATTAFORMA, 0, 1)
    p = PROF_RIVA + (PROF_PIATTAFORMA - PROF_RIVA) * _s(t1)

    in_scarpata = d > FINE_PIATTAFORMA
    t2 = np.clip((d - FINE_PIATTAFORMA) / (FINE_SCARPATA - FINE_PIATTAFORMA), 0, 1)
    p = np.where(in_scarpata,
                 PROF_PIATTAFORMA + (PROF_PIANA - PROF_PIATTAFORMA) * _s(t2), p)

    # oltre la scarpata la piana continua a scendere, ma pianissimo
    oltre = d > FINE_SCARPATA
    p = np.where(oltre, PROF_PIANA + (d - FINE_SCARPATA) * 0.05, p)

    # L'oceano disegnato resta un indizio, non una regola: sposta la
    # profondita' di qualche blocco, non crea piu' il muro.
    p = np.where(cls == OCEANO, p + 4.0, p)

    # --- rilievo ---------------------------------------------------------
    if rilievo > 0:
        lato = max(cls.shape)
        # due scale: dorsali larghe e secche piccole
        largo = fbm(lato, ottave=4, celle_base=max(4, lato // 120),
                    persistenza=0.55, seed=seed) - 0.5
        fine = fbm(lato, ottave=3, celle_base=max(8, lato // 40),
                   persistenza=0.5, seed=seed + 1) - 0.5
        n = (largo[:cls.shape[0], :cls.shape[1]] * 1.6
             + fine[:cls.shape[0], :cls.shape[1]] * 0.6)
        # l'ampiezza cresce con la profondita': niente scogli sotto la riva
        ampiezza = rilievo * np.clip(p / PROF_PIANA, 0, 1.2) * 14.0
        p = p + n * ampiezza

    p = np.where(marino, np.maximum(p, 1.0), 0.0)
    return gaussian_filter(p.astype(np.float32), 1.0) * marino


def _s(t: np.ndarray) -> np.ndarray:
    """Smoothstep: parte piano, arriva piano."""
    return t * t * (3.0 - 2.0 * t)


def applica(h: np.ndarray, cls: np.ndarray, livello_mare: int = 62,
            rilievo: float = 1.0, seed: int = 0) -> np.ndarray:
    """Sostituisce la quota delle celle marine con il fondale calcolato.

    Sostituisce, non taglia: il vecchio `minimum(h, tetto)` e' esattamente il
    gesto che appiattiva tutto contro il tetto.
    """
    marino = np.isin(cls, MARINO)
    if not marino.any():
        return h
    p = profondita(cls, livello_mare, rilievo, seed)
    fondo = np.round(livello_mare - p).astype(h.dtype)
    return np.where(marino, fondo, h)


def statistiche(h: np.ndarray, cls: np.ndarray,
                livello_mare: int = 62) -> dict:
    marino = np.isin(cls, MARINO)
    if not marino.any():
        return {"acqua": 0}
    d = (livello_mare - h[marino]).astype(np.float32)
    gz, gx = np.gradient(h.astype(np.float32))
    pend = np.hypot(gz, gx)[marino]
    q, n = np.unique(h[marino], return_counts=True)
    return {
        "acqua": int(marino.sum()),
        "prof_media": float(d.mean()),
        "prof_max": float(d.max()),
        "quote_distinte": int(len(q)),
        # la quota piu' diffusa: era il 53% prima di questo modulo
        "quota_piu_diffusa": float(n.max() / marino.sum()),
        "pendenza_media": float(pend.mean()),
        "scalini": int((pend > 8).sum()),
    }
