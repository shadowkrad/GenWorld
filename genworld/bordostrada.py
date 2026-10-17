"""Vita lungo le strade fuori dai paesi: pietre miliari, santelle e carovane.

Una strada di campagna senza niente accanto e' una riga sul prato. Qui le si
mettono accanto tre cose, tutte come gli arredi dei paesi (`arredi.py`): un
`Disegno` che si traduce una volta e si posa dove c'e' posto.

  - le pietre miliari, ogni cinquanta blocchi circa: un pilastrino di mattoni;
  - le santelle, ogni centocinquanta: un'edicola di pietra con un lume e una
    candela, rivolta verso la strada;
  - le carovane, poche: un carro con il carico, un fuoco acceso e un mercante
    ambulante con due lama, fermi in uno slargo a una decina di blocchi dalla strada.

Solo fuori dai paesi, mai su acqua, su pendii o dove c'e' gia' qualcosa.
"""
from __future__ import annotations

import numpy as np

from . import mappa as M
from .arredi import Arredo, Disegno

DIREZIONI = {"+x": (1, 0), "-x": (-1, 0), "+z": (0, 1), "-z": (0, -1)}
DISTANZA_MILIARI = 50
DISTANZA_SANTELLE = 150
DISTANZA_CAROVANE = 220
MARGINE_PAESE = 10


def miliare() -> Disegno:
    """Un pilastrino 1 x 3 x 1: base, mattone scolpito, lastra."""
    d = Disegno("miliare", 1, 3, 1)
    d.metti(0, 0, 0, "stone_bricks")
    d.metti(0, 1, 0, "chiseled_stone_bricks")
    d.metti(0, 2, 0, "stone_brick_slab", type="bottom", waterlogged="false")
    return d


def _frame(verso: str):
    """(fx, fz): dove guarda la facciata. Mappa (u, v) -> cella nel 3 x 3:
    u va dal fondo (0) alla facciata (2), v e' laterale."""
    fx, fz = DIREZIONI[verso]

    def cella(u: int, v: int) -> tuple[int, int]:
        du, dv = u - 1, v - 1
        return 1 + fx * du - fz * dv, 1 + fz * du + fx * dv
    return cella


def santella(verso: str = "+z") -> Disegno:
    """Un'edicola 3 x 4 x 3, aperta verso `verso` (la strada): zoccolo, tre lati
    di pietra, tetto di lastre, un altare con la candela e una lanterna appesa."""
    d = Disegno(f"santella{verso}", 3, 4, 3)
    cella = _frame(verso)
    for u in range(3):
        for v in range(3):
            x, z = cella(u, v)
            d.metti(x, 0, z, "stone_bricks")
            d.metti(x, 3, z, "stone_brick_slab", type="bottom", waterlogged="false")
    for v in range(3):                                   # il fondo
        for y in (1, 2):
            d.metti(*_xyz(cella(0, v), y), "stone_bricks")
    for u in (1, 2):                                     # i fianchi (la facciata resta aperta)
        for v in (0, 2):
            for y in (1, 2):
                d.metti(*_xyz(cella(u, v), y), "stone_bricks")
    x, z = cella(1, 1)
    d.metti(x, 1, z, "chiseled_stone_bricks")            # l'altare
    d.metti(x, 2, z, "candle", candles="3", lit="true", waterlogged="false")
    x, z = cella(2, 1)
    d.metti(x, 2, z, "lantern", hanging="true", waterlogged="false")
    return d


def _xyz(xz: tuple[int, int], y: int) -> tuple[int, int, int]:
    return xz[0], y, xz[1]


LUNGHEZZA_CAROVANA, LARGHEZZA_CAROVANA = 7, 4


def _cella_carovana(verso: str, u: int, v: int) -> tuple[int, int]:
    W, D = LUNGHEZZA_CAROVANA, LARGHEZZA_CAROVANA
    if verso == "+x":
        return u, v
    if verso == "-x":
        return W - 1 - u, D - 1 - v
    if verso == "+z":
        return v, u
    return D - 1 - v, W - 1 - u


def dimensioni_carovana(verso: str) -> tuple[int, int]:
    W, D = LUNGHEZZA_CAROVANA, LARGHEZZA_CAROVANA
    return (W, D) if verso in ("+x", "-x") else (D, W)


def carovana(verso: str = "+x") -> Disegno:
    """Il campo di un mercante: un carro (pianale, sponde, balle di fieno, botte,
    ruote di staccionata) e davanti un fuoco con due ceppi. 7 x 3 x 4. `verso`
    e' dove guarda il fuoco (la strada)."""
    dx, dz = dimensioni_carovana(verso)
    d = Disegno(f"carovana{verso}", dx, 3, dz)

    def metti(u, v, y, nome, **prop):
        x, z = _cella_carovana(verso, u, v)
        d.metti(x, y, z, nome, **prop)

    for u in range(4):                                   # il pianale 4 x 2
        for v in (1, 2):
            metti(u, v, 0, "oak_planks")
    for u in (0, 3):                                     # le sponde ai due capi
        for v in (1, 2):
            metti(u, v, 1, "oak_fence", waterlogged="false")
    metti(1, 1, 1, "hay_block", axis="y")
    metti(1, 2, 1, "hay_block", axis="y")
    metti(2, 2, 1, "barrel", facing="up", open="false")
    for u in (1, 2):                                     # le ruote
        for v in (0, 3):
            metti(u, v, 0, "dark_oak_fence", waterlogged="false")
    metti(5, 1, 0, "campfire", facing="north", lit="true", signal_fire="false", waterlogged="false")
    metti(5, 0, 0, "stripped_oak_log", axis="y")         # i ceppi
    metti(5, 2, 0, "stripped_oak_log", axis="y")
    return d


def posti_carovana(verso: str, x0: int, z0: int) -> list[tuple[float, float]]:
    """Dove stanno il mercante e i due lama (coordinate di mappa)."""
    celle = [(4.0, 1.5), (3.0, -0.6), (1.5, -0.6)]
    dx, dz = dimensioni_carovana(verso)
    fuori = []
    for u, v in celle:
        # stessa mappa di `_cella_carovana`, in coordinate continue
        W, D = LUNGHEZZA_CAROVANA, LARGHEZZA_CAROVANA
        if verso == "+x":
            x, z = u, v
        elif verso == "-x":
            x, z = W - 1 - u, D - 1 - v
        elif verso == "+z":
            x, z = v, u
        else:
            x, z = D - 1 - v, W - 1 - u
        fuori.append((x0 + x + 0.5, z0 + z + 0.5))
    return fuori


def pianifica(ris, tipo_strada: np.ndarray | None, citta: list, h: np.ndarray,
              cls: np.ndarray, occupato: np.ndarray | None, ver, seed: int = 0,
              densita: float = 1.0) -> dict:
    """Aggiunge a `ris` (un `arredi.Risultato`) pietre miliari, santelle e
    carovane lungo le strade fuori dai paesi, e i mercanti con i lama in
    `ris.animali`. Ritorna {tipo: quanti}."""
    from .fauna import Animale
    conteggio = {"miliare": 0, "santella": 0, "carovana": 0}
    if tipo_strada is None or densita <= 0:
        return conteggio
    H, W = h.shape
    rng = np.random.default_rng(seed * 7727 + 31)
    strada = tipo_strada > 0
    # `occupato` (di solito `protetto`) contiene GIA' le strade: toglierle, altrimenti
    # non resterebbe nessuna cella di strada da cui partire. Il margine dalla strada
    # lo controlla `libero` con la maschera `strada`.
    occ = np.zeros((H, W), bool) if occupato is None else (occupato & ~strada)
    occ |= np.isin(cls, M.ACQUA)
    # il paese e i suoi dintorni
    zz, xx = np.mgrid[0:H, 0:W]
    for c in citta:
        r = (c.lato or c.raggio) + MARGINE_PAESE
        occ |= (np.abs(zz - c.z) <= r) & (np.abs(xx - c.x) <= r)
    cache: dict[str, int] = {}

    def indice(nome: str, fabbrica) -> int:
        if nome not in cache:
            ris.modelli.append(fabbrica().modello(ver))
            cache[nome] = len(ris.modelli) - 1
        return cache[nome]

    def libero(x0: int, z0: int, dx: int, dz: int, margine: int = 1) -> bool:
        if x0 < 2 or z0 < 2 or x0 + dx >= W - 2 or z0 + dz >= H - 2:
            return False
        xa, za, xb, zb = x0 - margine, z0 - margine, x0 + dx + margine, z0 + dz + margine
        if occ[za:zb, xa:xb].any() or strada[za:zb, xa:xb].any():
            return False
        q = h[z0:z0 + dz, x0:x0 + dx]
        return int(q.max() - q.min()) <= 1

    def posa(tipo: str, x0: int, z0: int, dx: int, dz: int, modello: int) -> None:
        base = int(np.median(h[z0:z0 + dz, x0:x0 + dx]))
        ris.arredi.append(Arredo(tipo, x0, z0, dx, dz, base, modello))
        occ[max(0, z0 - 1):z0 + dz + 1, max(0, x0 - 1):x0 + dx + 1] = True

    zs, xs = np.nonzero(strada & ~occ)
    if zs.size == 0:
        return conteggio
    ordine = rng.permutation(zs.size)[:20000]
    miliari: list[tuple[int, int]] = []
    santelle: list[tuple[int, int]] = []
    carovane: list[tuple[int, int]] = []
    limite = {"carovana": max(1, int(round(H * W / 160_000 * densita)))}

    def lontano(p, lista, dist):
        return all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= dist * dist for q in lista)

    for k in ordine:
        z, x = int(zs[k]), int(xs[k])
        p = (x, z)
        dirs = list(DIREZIONI.items())
        rng.shuffle(dirs)
        for verso_str, (dx, dz) in dirs:
            # (dx, dz) e' la direzione dalla strada al sito
            toward_road = {"+x": "-x", "-x": "+x", "+z": "-z", "-z": "+z"}[verso_str]
            # santella: a 3 celle dalla strada, aperta verso di essa
            if lontano(p, santelle, DISTANZA_SANTELLE) and lontano(p, miliari, 12):
                cx, cz = x + dx * 3, z + dz * 3
                if libero(cx - 1, cz - 1, 3, 3):
                    posa("santella", cx - 1, cz - 1, 3, 3,
                         indice(f"santella{toward_road}", lambda v=toward_road: santella(v)))
                    santelle.append(p)
                    conteggio["santella"] += 1
                    break
            # carovana: a una decina di blocchi, il fuoco verso la strada
            if (len(carovane) < limite["carovana"] and lontano(p, carovane, DISTANZA_CAROVANE)
                    and lontano(p, santelle, 40)):
                ex, ez = dimensioni_carovana(toward_road)
                cx, cz = x + dx * 9, z + dz * 9
                x0, z0 = cx - ex // 2, cz - ez // 2
                if libero(x0, z0, ex, ez, margine=2):
                    posa("carovana", x0, z0, ex, ez,
                         indice(f"carovana{toward_road}", lambda v=toward_road: carovana(v)))
                    base = float(np.median(h[z0:z0 + ez, x0:x0 + ex]))
                    punti = posti_carovana(toward_road, x0, z0)
                    for specie, (px, pz) in zip(("mercante", "lama_mercante", "lama_mercante"), punti):
                        ris.animali.append(Animale(x=px, y=base, z=pz, specie=specie,
                                                   seme=int(rng.integers(0, 2 ** 31 - 1))))
                    carovane.append(p)
                    conteggio["carovana"] += 1
                    break
            # miliare: a due celle, un pilastrino
            if lontano(p, miliari, DISTANZA_MILIARI) and lontano(p, santelle, 12):
                cx, cz = x + dx * 2, z + dz * 2
                if libero(cx, cz, 1, 1):
                    posa("miliare", cx, cz, 1, 1, indice("miliare", miliare))
                    miliari.append(p)
                    conteggio["miliare"] += 1
                    break
    return conteggio
