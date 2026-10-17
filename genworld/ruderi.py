"""Megaliti e ruderi antichi sulle colline.

Un paesaggio senza storia e' un tappeto. Qui si lasciano in giro tre cose che
c'erano prima di tutto il resto, come gli arredi dei paesi (`arredi.py`): un
`Disegno` per ognuna, tradotto una volta e posato dove c'e' posto.

  - il cerchio di pietre (cromlech): undici menhir in cerchio, qualcuno caduto,
    e al centro un trilite con una candela;
  - il tempio in rovina: uno zoccolo di mattoni con due file di colonne spezzate a
    altezze diverse, un architrave ancora in piedi e rocchi sparsi;
  - il dolmen: due pietre in piedi con una lastra sopra, su un letto di muschio.

Ogni esemplare e' diverso (pietre, altezze, cadute) ma deterministico sul seme.
Solo su terra asciutta e piana, lontano da paesi e strade.
"""
from __future__ import annotations

import numpy as np

from . import mappa as M
from .arredi import Arredo, Disegno

PIETRE = ("andesite", "mossy_cobblestone", "cobblestone", "stone", "mossy_stone_bricks")
MATTONI = ("stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks")
DISTANZA_FRA_RUDERI = 110
DISTANZA_DAL_PAESE = 40
TIPI = ("cromlech", "tempio", "dolmen")
PESI_TIPI = (0.40, 0.35, 0.25)
DIMENSIONI = {"cromlech": (17, 7, 17), "tempio": (11, 8, 7), "dolmen": (7, 4, 7)}


def _muschio(d: Disegno, rng, x: int, z: int, prob: float = 0.35) -> None:
    if (x, 0, z) not in d.blocchi and rng.random() < prob:
        d.metti(x, 0, z, "moss_carpet" if rng.random() < 0.7 else "moss_block")


def cromlech(seme: int = 0) -> Disegno:
    """Undici pietre in cerchio (raggio 7), tre cadute, al centro un trilite e una
    candela. 17 x 7 x 17."""
    rng = np.random.default_rng(seme)
    d = Disegno(f"cromlech{seme}", 17, 7, 17)
    n = 11
    off = rng.random() * 2 * np.pi / n
    for k in range(n):
        a = off + 2 * np.pi * k / n
        x = int(round(8 + 7 * np.cos(a)))
        z = int(round(8 + 7 * np.sin(a)))
        pietra = PIETRE[int(rng.integers(0, len(PIETRE)))]
        if rng.random() < 0.27:                                  # una pietra caduta, distesa a terra
            asse = int(rng.integers(0, 2))
            for j in range(3):
                xx, zz = (x + j, z) if asse == 0 else (x, z + j)
                if 0 <= xx < 17 and 0 <= zz < 17:
                    d.metti(xx, 0, zz, pietra)
            continue
        alt = int(rng.integers(3, 6))
        for y in range(alt):
            d.metti(x, y, z, pietra)
        if rng.random() < 0.4 and x + 1 < 17:                   # una pietra larga due
            for y in range(alt - 1):
                d.metti(x + 1, y, z, pietra)
    # al centro: trilite (due pilastri e un architrave) e un altare con la candela
    for z in (7, 9):
        for y in range(4):
            d.metti(8, y, z, "stone_bricks" if rng.random() < 0.7 else "mossy_stone_bricks")
    for z in (7, 8, 9):
        d.metti(8, 4, z, "stone_brick_slab", type="bottom", waterlogged="false")
    d.metti(7, 0, 8, "chiseled_stone_bricks")
    d.metti(7, 1, 8, "candle", candles="2", lit="true", waterlogged="false")
    for x in range(17):
        for z in range(17):
            if np.hypot(x - 8, z - 8) <= 7.8:
                _muschio(d, rng, x, z)
    return d


def tempio(seme: int = 0) -> Disegno:
    """Uno zoccolo 11 x 7 con due file di quattro colonne spezzate a quote diverse,
    un architrave ancora in cima a due colonne, rocchi caduti. 11 x 8 x 7."""
    rng = np.random.default_rng(seme)
    d = Disegno(f"tempio{seme}", 11, 8, 7)
    for x in range(11):
        for z in range(7):
            if rng.random() < 0.82:                               # lo zoccolo, con qualche buco
                d.metti(x, 0, z, MATTONI[int(rng.integers(0, len(MATTONI)))])
    colonne = [(x, z) for x in (1, 4, 7, 10) for z in (1, 5)]
    alte = []
    for (x, z) in colonne:
        alt = int(rng.integers(1, 7))
        for y in range(1, alt + 1):
            d.metti(x, y, z, "stone_bricks" if y % 3 else "chiseled_stone_bricks")
        alte.append((alt, x, z))
        if alt <= 4 and rng.random() < 0.5:                      # troncata: un mezzo blocco in cima
            d.metti(x, alt + 1, z, "stone_brick_slab", type="top", waterlogged="false")
    # un architrave ancora in piedi fra le due colonne piu' alte della stessa fila
    for z in (1, 5):
        fila = sorted((a for a in alte if a[2] == z), reverse=True)
        (a1, x1, _), (a2, x2, _) = fila[0], fila[1]
        if min(a1, a2) >= 4:
            y = min(a1, a2) + 1                 # appoggiato alla colonna piu' bassa
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if (x, y, z) not in d.blocchi:
                    d.metti(x, y, z, "stone_bricks")
    # rocchi caduti e muschio
    for _ in range(5):
        x, z = int(rng.integers(0, 11)), int(rng.integers(0, 7))
        if (x, 1, z) not in d.blocchi and (x, 0, z) in d.blocchi:
            d.metti(x, 1, z, "stone_bricks")
    for x in range(11):
        for z in range(7):
            if (x, 0, z) in d.blocchi and (x, 1, z) not in d.blocchi and rng.random() < 0.25:
                d.metti(x, 1, z, "moss_carpet")
    return d


def dolmen(seme: int = 0) -> Disegno:
    """Due pietre in piedi e una lastra sopra, su un anello di muschio. 7 x 4 x 7."""
    rng = np.random.default_rng(seme)
    d = Disegno(f"dolmen{seme}", 7, 4, 7)
    pietra = PIETRE[int(rng.integers(0, len(PIETRE)))]
    for x in (1, 2, 4, 5):
        for y in range(3):
            d.metti(x, y, 3, pietra)
    for x in range(7):
        for z in (2, 3, 4):
            if abs(x - 3) <= 3 and not (rng.random() < 0.12):
                d.metti(x, 3, z, pietra if (x + z) % 2 else "stone")
    for x in range(7):
        for z in range(7):
            if (x, 0, z) not in d.blocchi and np.hypot(x - 3, z - 3) <= 3.2 and rng.random() < 0.55:
                d.metti(x, 0, z, "moss_carpet" if rng.random() < 0.6 else "moss_block")
    return d


FABBRICHE = {"cromlech": cromlech, "tempio": tempio, "dolmen": dolmen}


def pianifica(ris, cls: np.ndarray, h: np.ndarray, strada: np.ndarray | None, citta: list,
              occupato: np.ndarray | None, ver, seed: int = 0, densita: float = 1.0,
              celle_per: int = 150_000) -> dict:
    """Aggiunge a `ris` (un `arredi.Risultato`) i ruderi. Ritorna {tipo: quanti}."""
    conteggio = {t: 0 for t in TIPI}
    H, W = h.shape
    quanti = int(max(0, round(H * W / celle_per * densita)))
    if quanti <= 0:
        return conteggio
    rng = np.random.default_rng(seed * 8191 + 77)
    occ = np.zeros((H, W), bool) if occupato is None else occupato.copy()
    occ |= np.isin(cls, M.ACQUA)
    occ |= ~np.isin(cls, (M.PIANURA, M.PRATERIA, M.FORESTA, M.MONTAGNA))
    zz, xx = np.mgrid[0:H, 0:W]
    for c in citta:
        r = (c.lato or c.raggio) + DISTANZA_DAL_PAESE
        occ |= (np.abs(zz - c.z) <= r) & (np.abs(xx - c.x) <= r)
    if strada is not None:
        from scipy.ndimage import binary_dilation
        occ |= binary_dilation(strada > 0, iterations=4)
    posti: list[tuple[int, int]] = []
    zs, xs = np.nonzero(~occ)
    if zs.size == 0:
        return conteggio
    for k in rng.permutation(zs.size)[:30000]:
        if sum(conteggio.values()) >= quanti:
            break
        x, z = int(xs[k]), int(zs[k])
        if any((x - px) ** 2 + (z - pz) ** 2 < DISTANZA_FRA_RUDERI ** 2 for px, pz in posti):
            continue
        tipo = TIPI[int(rng.choice(len(TIPI), p=PESI_TIPI))]
        dx, dy, dz = DIMENSIONI[tipo]
        x0, z0 = x - dx // 2, z - dz // 2
        if x0 < 3 or z0 < 3 or x0 + dx >= W - 3 or z0 + dz >= H - 3:
            continue
        if occ[z0 - 2:z0 + dz + 2, x0 - 2:x0 + dx + 2].any():
            continue
        q = h[z0:z0 + dz, x0:x0 + dx]
        if int(q.max() - q.min()) > 2:
            continue
        modello = FABBRICHE[tipo](int(rng.integers(0, 2 ** 31 - 1))).modello(ver)
        ris.modelli.append(modello)
        ris.arredi.append(Arredo(tipo, x0, z0, dx, dz, int(np.median(q)), len(ris.modelli) - 1))
        occ[z0 - 3:z0 + dz + 3, x0 - 3:x0 + dx + 3] = True
        posti.append((x, z))
        conteggio[tipo] += 1
    return conteggio
