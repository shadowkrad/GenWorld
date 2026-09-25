"""Rumore frattale (fBm) per il dettaglio del terreno.

Serve a rompere le terrazze: una heightmap generata da una funzione liscia,
quantizzata a blocchi interi, produce curve di livello concentriche
perfettamente visibili in gioco. Sono l'artefatto piu' riconoscibile di un
terreno generato male.

Il punto non e' aggiungere rumore ovunque, ma **modularne l'ampiezza con la
semantica e la pendenza**: piena sui versanti, quasi nulla in pianura, zero
sulle aree urbane. Il rumore uniforme fa sembrare tutto finto.

Implementazione: value noise a piu' ottave, solo numpy, deterministico a
parita' di seed.
"""

from __future__ import annotations

import numpy as np


def _ottava(lato: int, celle: int, rng: np.random.Generator) -> np.ndarray:
    """Una singola ottava: griglia grossolana interpolata con smoothstep."""
    celle = max(1, int(celle))
    grezza = rng.random((celle + 1, celle + 1)).astype(np.float32)

    pos = np.linspace(0, celle, lato, endpoint=False, dtype=np.float32)
    i0 = np.floor(pos).astype(np.int32)
    t = pos - i0
    t = t * t * (3.0 - 2.0 * t)              # smoothstep: derivata nulla ai nodi
    i1 = i0 + 1

    # interpolazione separabile: prima lungo un asse, poi lungo l'altro
    a = grezza[i0, :] * (1 - t)[:, None] + grezza[i1, :] * t[:, None]
    return a[:, i0] * (1 - t)[None, :] + a[:, i1] * t[None, :]


def fbm(
    lato: int,
    ottave: int = 5,
    celle_base: int = 4,
    persistenza: float = 0.5,
    lacunarita: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Fractal Brownian motion normalizzato in 0..1, forma (lato, lato)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((lato, lato), dtype=np.float32)
    ampiezza, totale, celle = 1.0, 0.0, float(celle_base)
    for _ in range(ottave):
        out += ampiezza * _ottava(lato, celle, rng)
        totale += ampiezza
        ampiezza *= persistenza
        celle *= lacunarita
    return out / totale


def pendenza(altezze: np.ndarray) -> np.ndarray:
    """Modulo del gradiente, normalizzato in 0..1."""
    gx, gz = np.gradient(altezze.astype(np.float32))
    p = np.hypot(gx, gz)
    massimo = float(p.max())
    return p / massimo if massimo > 0 else p


def dettaglio(
    altezze: np.ndarray,
    ampiezza_base: float = 1.2,
    ampiezza_pendenza: float = 4.0,
    ottave: int = 6,
    celle_base: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Rumore da sommare a una heightmap, modulato dalla pendenza locale.

    `ampiezza_base` agisce ovunque e rompe le terrazze anche in piano;
    `ampiezza_pendenza` si aggiunge in proporzione alla pendenza, cosi' i
    versanti diventano irregolari e le pianure restano pianure.
    """
    lato = altezze.shape[0]
    n = fbm(lato, ottave=ottave, celle_base=celle_base, seed=seed) - 0.5
    amp = ampiezza_base + ampiezza_pendenza * pendenza(altezze)
    return n * 2.0 * amp


def quantizza(altezze: np.ndarray, forza: float = 1.0, seed: int = 0) -> np.ndarray:
    """Arrotonda a blocchi interi con dithering.

    Le terrazze non sono un difetto del rilievo: sono un artefatto della
    QUANTIZZAZIONE. Una superficie liscia che attraversa lentamente il confine
    fra due interi lo attraversa lungo una curva di livello, e quella curva
    diventa un gradino continuo lungo decine di blocchi.

    Sommare rumore frattale prima di arrotondare non basta: in un fBm le ottave
    fini portano un'ampiezza pari a persistenza^n, cioe' centesimi di blocco,
    mentre per rompere un gradino serve variazione dell'ordine del blocco
    proprio a frequenza alta.

    La soluzione e' ditherare la soglia di arrotondamento: si sposta il confine
    fra un intero e l'altro con rumore ad alta frequenza di ampiezza ~1 blocco.
    Il gradino si frastaglia e sparisce, senza spostare la quota media.
    """
    lato = altezze.shape[0]
    # due ottave molto fini: dettaglio a 2-4 blocchi, non a decine
    fine = fbm(lato, ottave=2, celle_base=max(1, lato // 3),
               persistenza=0.6, seed=seed + 991)
    # fbm() e' normalizzato sull'AMPIEZZA, non sull'intervallo: in pratica resta
    # dentro ~0,2-0,8. Va riportato a tutto [0,1], altrimenti il dithering non
    # copre l'intero passo di quantizzazione e i gradini sopravvivono.
    lo, hi = float(fine.min()), float(fine.max())
    if hi > lo:
        fine = (fine - lo) / (hi - lo)
    return np.floor(altezze + 0.5 + forza * (fine - 0.5)).astype(np.int32)


# NOTA. Qui c'era una metrica automatica di terrazzamento. Ne sono state
# provate quattro - frazione di pianori, lunghezza media dei tratti costanti,
# coerenza dei bordi, catene di bordo connesse - e NESSUNA distingue un pendio
# terrazzato da uno ditherato: le statistiche locali sono identiche, cio' che
# cambia e' la forma globale delle curve di livello. Sono state tolte invece di
# lasciare in giro numeri che non misurano quello che dichiarano.
#
# Per ora il giudice e' il confronto visivo a risoluzione piena
# (mondi/confronto_terrazze.png) e l'aspetto in gioco. Una metrica che funzioni
# probabilmente deve guardare la curvatura delle curve di livello, non il
# vicinato di un pixel.
