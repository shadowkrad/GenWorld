"""Ferrovie, stazioni di legno e locomotive a vapore.

La rete ferroviaria segue le strade gia' tracciate: un Dijkstra sulle celle di strada fra
le citta' piu' grandi (le strade sanno gia' attraversare fiumi e colline), con una
penalita' per le curve - un treno non gira a ogni isolato - e il divieto di curvare su
una pendenza (un binario in salita non puo' essere anche una curva). I binari stanno
sulla sede stradale, un blocco sopra la superficie; sulle salite sono tutti alimentati,
in piano uno ogni `PASSO_ALIMENTATO` (con un blocco di redstone sotto).

Ogni citta' collegata ha una STAZIONE (un `siti.Sito` di tipo "stazione"): banchina di
legno su palafitte, edificio con tetto, e su un binario morto la locomotiva a vapore -
caldaia di carbone, ciminiera con un fuoco da segnalazione per il fumo, cabina, tender e
due carrelli. In Sottosopra e' il treno fantasma: lo specchio e il decadimento fanno il
resto.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from . import strade as ST

PASSO_ALIMENTATO = 12
MEZZO_STAZIONE = 13
LUNGHEZZA_RETTILINEO = 10              # celle dritte e piane per lato richieste a una stazione
DISTANZA_MAX = 900                     # oltre, due citta' non si collegano

# direzioni (dz, dx) -> nome
N, S, E, O = (-1, 0), (1, 0), (0, 1), (0, -1)
NOME_DIR = {N: "north", S: "south", E: "east", O: "west"}
FORMA = {frozenset((N, S)): "north_south", frozenset((E, O)): "east_west",
         frozenset((S, E)): "south_east", frozenset((S, O)): "south_west",
         frozenset((N, O)): "north_west", frozenset((N, E)): "north_east"}

# codici dei binari nei dati del piano: forma, alimentato
FORME = ("north_south", "east_west", "south_east", "south_west", "north_west", "north_east",
         "ascending_north", "ascending_south", "ascending_east", "ascending_west")


@dataclass
class Ferrovia:
    # {(chunk x, chunk z) di GIOCO: [(x locale, z locale, y, forma, alimentato)]}
    rotaie: dict = field(default_factory=dict)
    stazioni: list = field(default_factory=list)
    lunghezza: int = 0
    linee: list = field(default_factory=list)      # [(nome citta' A, nome citta' B, celle)]


# --------------------------------------------------------------------------
# Tracciato
# --------------------------------------------------------------------------

def _percorso(utile: np.ndarray, quota: np.ndarray, a: tuple, b: tuple,
              margine: int = 30):
    """Cammino piu' breve da a a b (z, x) sulle celle `utile`. Stato = (cella, direzione, salto
    in ingresso): si cambia direzione solo dove la quota e' la stessa prima e dopo. None se manca."""
    H, W = utile.shape
    z0, z1 = max(0, min(a[0], b[0]) - margine), min(H, max(a[0], b[0]) + margine + 1)
    x0, x1 = max(0, min(a[1], b[1]) - margine), min(W, max(a[1], b[1]) + margine + 1)
    sub_u = utile[z0:z1, x0:x1]
    sub_q = quota[z0:z1, x0:x1]
    if not sub_u[a[0] - z0, a[1] - x0] or not sub_u[b[0] - z0, b[1] - x0]:
        return None
    inizio = (a[0] - z0, a[1] - x0, None, 0)
    fine = (b[0] - z0, b[1] - x0)
    dist = {inizio: 0.0}
    prima = {}
    coda = [(0.0, 0, inizio)]
    n = 0
    while coda:
        d, _, st = heapq.heappop(coda)
        if d > dist.get(st, 1e18):
            continue
        z, x, dr, dy_in = st
        if (z, x) == fine:
            cammino = [(z + z0, x + x0)]
            while st in prima:
                st = prima[st]
                cammino.append((st[0] + z0, st[1] + x0))
            return cammino[::-1]
        for nd in (N, S, E, O):
            if dr is not None and nd == (-dr[0], -dr[1]):
                continue
            nz, nx = z + nd[0], x + nd[1]
            if not (0 <= nz < sub_u.shape[0] and 0 <= nx < sub_u.shape[1]) or not sub_u[nz, nx]:
                continue
            dy = int(sub_q[nz, nx]) - int(sub_q[z, x])
            if abs(dy) > 1:
                continue
            costo = 1.0 + 3.0 * abs(dy)
            if dr is not None and nd != dr:
                if dy_in != 0 or dy != 0:                   # niente curve sulle pendenze
                    continue
                costo += 3.0
            if dy_in * dy < 0:                              # una valle: il binario non puo' salire e scendere
                if dy_in < 0:
                    continue
            ns = (nz, nx, nd, dy)
            nd_ = d + costo
            if nd_ < dist.get(ns, 1e18):
                dist[ns] = nd_
                prima[ns] = st
                n += 1
                heapq.heappush(coda, (nd_, n, ns))
    return None


def _forme(cammino: list, quota: np.ndarray) -> list:
    """Per ogni cella (z, x): la forma del binario. I due vicini sono prev e next del cammino."""
    fuori = []
    n = len(cammino)
    for i, (z, x) in enumerate(cammino):
        prec = cammino[i - 1] if i > 0 else None
        succ = cammino[i + 1] if i < n - 1 else None
        if prec is None and succ is None:
            fuori.append("north_south")
            continue
        if prec is None:
            prec = (2 * z - succ[0], 2 * x - succ[1])
        if succ is None:
            succ = (2 * z - prec[0], 2 * x - prec[1])
        dp = (prec[0] - z, prec[1] - x)
        ds = (succ[0] - z, succ[1] - x)
        q = int(quota[z, x])
        sale_s = i < n - 1 and int(quota[succ[0], succ[1]]) > q
        sale_p = i > 0 and int(quota[prec[0], prec[1]]) > q
        if dp == (-ds[0], -ds[1]):
            if sale_s:
                fuori.append("ascending_" + NOME_DIR[ds])
            elif sale_p:
                fuori.append("ascending_" + NOME_DIR[dp])
            else:
                fuori.append(FORMA[frozenset((dp, ds))])
        else:
            fuori.append(FORMA.get(frozenset((dp, ds)), "north_south"))
    return fuori


def _scegli_ancora(tipo: np.ndarray, c, interdetto: np.ndarray):
    """La cella di strada (non ponte, non dentro altre citta') piu' vicina al bordo della citta'."""
    H, W = tipo.shape
    r = max(c.lato, c.raggio, c.lz, c.lx) + 12
    z0, z1 = max(0, c.z - r - 40), min(H, c.z + r + 41)
    x0, x1 = max(0, c.x - r - 40), min(W, c.x + r + 41)
    sub = (tipo[z0:z1, x0:x1] == ST.STRADA) & ~interdetto[z0:z1, x0:x1]
    zz, xx = np.nonzero(sub)
    if zz.size == 0:
        return None
    d = np.hypot(zz + z0 - c.z, xx + x0 - c.x)
    ok = d >= r
    if not ok.any():
        return None
    k = int(np.argmin(np.where(ok, d, 1e9)))
    return int(zz[k] + z0), int(xx[k] + x0)


def pianifica(tipo_strada, quota_strada, citta: list, meta: int, nomi_citta: dict | None = None,
              densita: float = 1.0, seed: int = 0, max_stazioni: int = 14) -> Ferrovia:
    """Rete e stazioni. Coordinate in uscita: i binari sono in coordinate di GIOCO (chunk locali),
    le stazioni sono `Sito` in coordinate di MAPPA come tutti gli altri luoghi."""
    fer = Ferrovia()
    if tipo_strada is None or densita <= 0 or len(citta) < 2:
        return fer
    H, W = tipo_strada.shape
    nomi_citta = nomi_citta or {}
    grandi = sorted(citta, key=lambda c: -(c.raggio + c.lato))[:max(2, int(round(max_stazioni * densita)))]

    # dentro le citta' non passa il treno
    interdetto = np.zeros((H, W), bool)
    for c in citta:
        r = max(c.lato, c.raggio, c.lz, c.lx) + 4
        interdetto[max(0, c.z - r):c.z + r + 1, max(0, c.x - r):c.x + r + 1] = True
    utile = (np.isin(tipo_strada, (ST.STRADA, ST.PONTE))) & ~interdetto
    quota = np.asarray(quota_strada, np.int32)

    ancore = {}
    for c in grandi:
        # l'ancora sta nell'area interdetta ai bordi: la cella utile piu' vicina
        a = _scegli_ancora(tipo_strada, c, np.zeros((H, W), bool))
        if a is None:
            continue
        ancore[(c.x, c.z)] = a
    elenco = [c for c in grandi if (c.x, c.z) in ancore]
    if len(elenco) < 2:
        return fer

    # archi: i 3 piu' vicini di ogni citta'
    coppie = set()
    for i, ci in enumerate(elenco):
        vicini = sorted((j for j in range(len(elenco)) if j != i),
                        key=lambda j: (elenco[j].x - ci.x) ** 2 + (elenco[j].z - ci.z) ** 2)[:3]
        coppie.update((min(i, j), max(i, j)) for j in vicini)
    archi = []
    for i, j in coppie:
        d = float(np.hypot(elenco[j].x - elenco[i].x, elenco[j].z - elenco[i].z))
        if d <= DISTANZA_MAX:
            archi.append((d, i, j))
    archi.sort()
    padre = list(range(len(elenco)))

    def radice(k):
        while padre[k] != k:
            padre[k] = padre[padre[k]]
            k = padre[k]
        return k

    percorsi = []
    extra = 0
    for d, i, j in archi:
        ri, rj = radice(i), radice(j)
        if ri == rj and extra >= max(1, len(elenco) // 6):
            continue
        cam = _percorso(utile, quota, ancore[(elenco[i].x, elenco[i].z)],
                        ancore[(elenco[j].x, elenco[j].z)])
        if cam is None or len(cam) < 6:
            continue
        if ri == rj:
            extra += 1
        padre[ri] = rj
        percorsi.append((i, j, cam))
    if not percorsi:
        return fer

    # le celle di binario: la prima forma scritta vince (incroci)
    celle: dict = {}
    for i, j, cam in percorsi:
        forme = _forme(cam, quota)
        for k, ((z, x), f) in enumerate(zip(cam, forme)):
            if (z, x) in celle:
                continue
            sale = f.startswith("ascending")
            alim = sale or (k % PASSO_ALIMENTATO == PASSO_ALIMENTATO // 2 and f in ("north_south", "east_west"))
            celle[(z, x)] = (f, alim)
        fer.linee.append((elenco[i], elenco[j], len(cam)))

    # stazioni: un rettilineo piano vicino a ogni citta' collegata
    usate = {k for i, j, _ in percorsi for k in (i, j)}
    for i in sorted(usate):
        c = elenco[i]
        s = _stazione(celle, quota, c, tipo_strada, seed + i)
        if s is not None:
            s.extra["citta"] = (c.x, c.z)
            s.nome = "Stazione di " + nomi_citta.get((c.x, c.z), "")
            fer.stazioni.append(s)

    for (z, x), (f, alim) in celle.items():
        gx, gz = x - meta, z - meta
        fer.rotaie.setdefault((gx >> 4, gz >> 4), []).append(
            (gx & 15, gz & 15, int(quota[z, x]), f, alim))
    fer.lunghezza = len(celle)
    return fer


def _stazione(celle: dict, quota: np.ndarray, c, tipo_strada: np.ndarray, seme: int):
    """Un tratto rettilineo e piano di binario, a distanza utile dalla citta'."""
    from .siti import Sito
    r = max(c.lato, c.raggio, c.lz, c.lx) + 14
    migliori = None
    for (z, x), (f, _) in celle.items():
        if f not in ("north_south", "east_west"):
            continue
        d = float(np.hypot(z - c.z, x - c.x))
        if d < r or d > r + 70:
            continue
        dz, dx = (1, 0) if f == "north_south" else (0, 1)
        ok = True
        q = int(quota[z, x])
        for t in range(-LUNGHEZZA_RETTILINEO, LUNGHEZZA_RETTILINEO + 1):
            cella = (z + dz * t, x + dx * t)
            v = celle.get(cella)
            if v is None or v[0] != f or int(quota[cella]) != q or tipo_strada[cella] != ST.STRADA:
                ok = False
                break
        if ok and (migliori is None or d < migliori[0]):
            migliori = (d, z, x, f, q)
    if migliori is None:
        return None
    _, z, x, f, q = migliori
    return Sito(tipo="stazione", x=x, z=z, y=q, seme=seme * 7919 + 13,
                extra={"asse": "z" if f == "north_south" else "x", "citta": None})


# --------------------------------------------------------------------------
# Scrittura dei binari
# --------------------------------------------------------------------------

def blocchi_rotaia(scrittore) -> dict:
    """{(forma, alimentato): id}, piu' 'redstone' per il blocco sotto i binari alimentati."""
    t = {}
    for f in FORME:
        t[(f, False)] = scrittore.blocco("rail", shape=f)
        if f in ("north_south", "east_west") or f.startswith("ascending"):
            t[(f, True)] = scrittore.blocco("powered_rail", shape=f, powered="true")
    t["redstone"] = scrittore.blocco("redstone_block")
    return t


def posa(blocchi: np.ndarray, tav: dict, aria: int, ox: int, oz: int, y0: int,
         ferrovia: Ferrovia, cx: int, cz: int) -> None:
    """Scrive i binari nel chunk (16, H, 16) di coordinate di gioco (cx, cz)."""
    H = blocchi.shape[1]
    for lx, lz, y, forma, alim in ferrovia.rotaie.get((cx, cz), ()):
        ya = y - y0
        if not (2 <= ya < H - 2):
            continue
        blocchi[lx, ya + 1:ya + 3, lz] = aria
        if alim and (forma, True) in tav:
            blocchi[lx, ya - 1, lz] = tav["redstone"]
            blocchi[lx, ya, lz] = tav[(forma, True)]
        else:
            blocchi[lx, ya, lz] = tav[(forma, False)]
