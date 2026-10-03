"""Case isolate: le costruzioni grandi, fuori dai villaggi.

I template piu' alti di `ALTEZZA_MAX_VILLAGGIO` blocchi (manieri, case
sull'albero, torri) non entrano nei lotti stretti di un paese e, dove entrano,
sono troppo per una strada di case a schiera. Fuori dai villaggi invece sono
esattamente quello che si vorrebbe incontrare: una casa nella prateria, un
maniero nel bosco, una fattoria isolata. Ognuna sta su un terreno abbastanza
piano, lontano da tutto il resto, e ogni modello compare al massimo UNA volta
per mappa - sono pochi e diversi, e due manieri uguali si notano.

Come gli avamposti, si piazzano a caso su tutta la mappa (con seme), e
come le case dei villaggi si posano con `template.costruisci` e la fondazione.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import edifici as E
from .mappa import DESERTO, FORESTA, PIANURA, PRATERIA, SPIAGGIA

# oltre questa altezza un template non e' una casa di villaggio
ALTEZZA_MAX_VILLAGGIO = 24

# classi su cui una casa isolata puo' stare
_TERRENI = (PIANURA, PRATERIA, FORESTA, SPIAGGIA, DESERTO)


@dataclass
class CasaIsolata:
    x: int             # angolo minimo, coordinate di mappa
    z: int
    larghezza: int     # ingombro GIRATO (lungo x)
    profondita: int    # (lungo z)
    base: int          # quota del primo strato
    modello: int       # indice nel catalogo delle case
    quarti: int
    stile: str = "prato"


def candidati(modelli: list, lato_mappa: int) -> list[int]:
    """I modelli riservati alle case isolate: quelli troppo alti per un lotto,
    non esclusi a mano, e non piu' larghi di un quinto della mappa (un modello
    da 109 blocchi su una mappa da 256 la riempirebbe da solo)."""
    return [k for k, m in enumerate(modelli)
            if m.dy > ALTEZZA_MAX_VILLAGGIO and "_escluso" not in m.stili
            and max(m.dx, m.dz) <= lato_mappa // 5]


def pianifica(modelli: list, h: np.ndarray, cls: np.ndarray, evita: np.ndarray,
              livello_mare: int, seed: int = 0, densita: float = 1.0,
              celle_per: int = 120_000, distanza_min: int = 60,
              margine: int = 5, dislivello_max: int = 7,
              tentativi: int = 300) -> list[CasaIsolata]:
    """Sceglie dove mettere le case isolate.

    `evita` e' una maschera di dove non si puo' stare (villaggi, strade, campi,
    acqua, vulcano, avamposti, miniere...): si tiene `margine` celle da essa.
    Ogni modello si prova in punti a caso, in rotazioni a caso, fino a
    `tentativi` volte; se non trova posto per lui va oltre - una mappa piccola
    o piena semplicemente ne avra' meno.
    """
    if densita <= 0:
        return []
    H, W = h.shape
    elenco = candidati(modelli, min(H, W))
    if not elenco:
        return []
    rng = np.random.default_rng(seed * 6151 + 29)
    quanti = min(len(elenco), int(round(H * W / celle_per * densita)))
    rng.shuffle(elenco)
    occ = evita.copy()
    fuori: list[CasaIsolata] = []
    for k in elenco:
        if len(fuori) >= quanti:
            break
        m = modelli[k]
        for _ in range(tentativi):
            q = int(rng.integers(0, 4))
            ix, iz = m.ingombro(q)
            if ix + 2 * margine >= W or iz + 2 * margine >= H:
                break
            x0 = int(rng.integers(margine, W - ix - margine))
            z0 = int(rng.integers(margine, H - iz - margine))
            fp_cls = cls[z0:z0 + iz, x0:x0 + ix]
            if not np.isin(fp_cls, _TERRENI).all():
                continue
            if occ[max(0, z0 - margine):z0 + iz + margine,
                   max(0, x0 - margine):x0 + ix + margine].any():
                continue
            fp_h = h[z0:z0 + iz, x0:x0 + ix]
            lo, hi = np.percentile(fp_h, (10, 90))
            if hi - lo > dislivello_max:
                continue
            base = int(np.median(fp_h))
            if base <= livello_mare:
                continue
            classe = int(np.bincount(fp_cls.ravel()).argmax())
            stile = E.PALETTE_PER_CLASSE.get(classe, "prato")
            if m.stili and stile not in m.stili:
                continue
            cx, cz = x0 + ix // 2, z0 + iz // 2
            if any((cx - c.x - c.larghezza // 2) ** 2 + (cz - c.z - c.profondita // 2) ** 2
                   < distanza_min ** 2 for c in fuori):
                continue
            fuori.append(CasaIsolata(x=x0, z=z0, larghezza=ix, profondita=iz,
                                     base=base, modello=k, quarti=q, stile=stile))
            occ[max(0, z0 - margine):z0 + iz + margine,
                max(0, x0 - margine):x0 + ix + margine] = True
            break
    return fuori


def indice_per_chunk(case: list[CasaIsolata], passo: int = 16) -> dict:
    """Ogni casa e' registrata in TUTTI i chunk che il suo ingombro tocca: un
    maniero da 40 blocchi ne attraversa parecchi."""
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(case):
        for cz in range(c.z // passo, (c.z + c.profondita - 1) // passo + 1):
            for cx in range(c.x // passo, (c.x + c.larghezza - 1) // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def maschera(case: list[CasaIsolata], shape: tuple[int, int], margine: int = 0) -> np.ndarray:
    H, W = shape
    m = np.zeros(shape, bool)
    for c in case:
        m[max(0, c.z - margine):min(H, c.z + c.profondita + margine),
          max(0, c.x - margine):min(W, c.x + c.larghezza + margine)] = True
    return m
