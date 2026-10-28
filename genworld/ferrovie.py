"""Ferrovie, stazioni di legno e locomotive a vapore.

Il binario ha un TRACCIATO TUTTO SUO, separato dalle strade e da ogni altra costruzione:
un A* sulle celle libere (fuori da edifici, mura, campi, luoghi, acqua, ponti e dalle citta')
con una quota di posa liscia - mai piu' di un blocco di dislivello per cella, e mai una curva
su una pendenza. Le strade si attraversano solo in linea retta, con un passaggio a livello.
Ogni linea riserva il suo corridoio: le linee successive non lo toccano, e dopo la
pianificazione il corridoio entra in `protetto` (niente alberi, luoghi o arredi).

Lungo il tracciato si costruisce il RILEVATO: sotto il binario un letto di ghiaia, e dove il
terreno e' piu' basso si riempie fino al suolo; dove e' piu' alto si scava (trincea). Le
curve e le salite hanno la forma giusta (`rail[shape=...]`): una salita e' una rampa che
sale verso la cella dopo, che sta un blocco piu' in alto. In salita tutti i binari sono
alimentati; in piano uno ogni `PASSO_ALIMENTATO`, con un blocco di redstone sotto.

Ogni citta' collegata ha una STAZIONE (un `siti.Sito` di tipo "stazione") su un rettilineo
piano della propria linea. In Sottosopra e' il treno fantasma.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter

from . import mappa as M
from . import strade as ST

PASSO_ALIMENTATO = 12
LUNGHEZZA_RETTILINEO = 10              # celle dritte e piane per lato richieste a una stazione
DISTANZA_MAX = 600                     # oltre, due citta' non si collegano
MARGINE_FINESTRA = 40
SCAVO_MAX = 7                          # dislivello massimo fra binario e terreno

# direzioni (dz, dx) -> nome
N, S, E, O = (-1, 0), (1, 0), (0, 1), (0, -1)
NOME_DIR = {N: "north", S: "south", E: "east", O: "west"}
FORMA = {frozenset((N, S)): "north_south", frozenset((E, O)): "east_west",
         frozenset((S, E)): "south_east", frozenset((S, O)): "south_west",
         frozenset((N, O)): "north_west", frozenset((N, E)): "north_east"}

FORME = ("north_south", "east_west", "south_east", "south_west", "north_west", "north_east",
         "ascending_north", "ascending_south", "ascending_east", "ascending_west")


@dataclass
class Ferrovia:
    # {(chunk x, chunk z) di GIOCO: [(x locale, z locale, y del binario, forma, alimentato, y del suolo)]}
    rotaie: dict = field(default_factory=dict)
    stazioni: list = field(default_factory=list)
    lunghezza: int = 0
    linee: list = field(default_factory=list)
    celle: set = field(default_factory=set)        # (z, x) di mappa
    maschera_celle: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Ricerca del percorso
# --------------------------------------------------------------------------

def _percorso(passabile: np.ndarray, costo: np.ndarray, quota: np.ndarray, strada: np.ndarray,
              a: tuple, b: tuple):
    """A* da a a b (z, x locali alla finestra). Stato = (cella, direzione, salto in ingresso).
    Si curva solo in piano e fuori dalle strade; una valle (scende e poi sale) e' vietata.
    Ritorna la lista di celle o None."""
    h, w = passabile.shape
    if not passabile[a] or not passabile[b]:
        return None
    inizio = (a[0], a[1], None, 0)
    dist = {inizio: 0.0}
    prima = {}
    coda = [(0.0, 0, inizio)]
    n = 0
    while coda:
        _, _, st = heapq.heappop(coda)
        z, x, dr, dy_in = st
        d = dist[st]
        if (z, x) == b:
            cam = [(z, x)]
            while st in prima:
                st = prima[st]
                cam.append((st[0], st[1]))
            return cam[::-1]
        for nd in (N, S, E, O):
            if dr is not None and nd == (-dr[0], -dr[1]):
                continue
            nz, nx = z + nd[0], x + nd[1]
            if not (0 <= nz < h and 0 <= nx < w) or not passabile[nz, nx]:
                continue
            dy = int(quota[nz, nx]) - int(quota[z, x])
            if abs(dy) > 1:
                continue
            c = float(costo[nz, nx]) + 3.0 * abs(dy)
            if dr is not None and nd != dr:
                if dy_in != 0 or dy != 0 or strada[z, x]:      # niente curve sulle pendenze o sulle strade
                    continue
                c += 2.0
            if strada[nz, nx] and (dr is not None and nd != dr):
                continue
            if dy_in < 0 and dy > 0:                           # una valle
                continue
            ns = (nz, nx, nd, dy)
            nd_ = d + c
            if nd_ < dist.get(ns, 1e18):
                dist[ns] = nd_
                prima[ns] = st
                n += 1
                heapq.heappush(coda, (nd_ + abs(nz - b[0]) + abs(nx - b[1]), n, ns))
    return None


def _forme(cammino: list, quota) -> list:
    """Per ogni cella (z, x): la forma del binario, dai due vicini del cammino."""
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


# --------------------------------------------------------------------------
# Finestre di lavoro
# --------------------------------------------------------------------------

class _Finestra:
    """Una porzione della mappa con tutto cio' che serve al percorso."""

    def __init__(self, z0, z1, x0, x1, tipo, quota_s, h, cls, protetto, citta, occupate):
        self.z0, self.x0 = z0, x0
        sl = (slice(z0, z1), slice(x0, x1))
        tipo_w = np.asarray(tipo[sl])
        strada = tipo_w == ST.STRADA
        fisso = (tipo_w > 0) & ~strada                                # ponti, parapetti, viali
        acqua = np.isin(np.asarray(cls[sl]), M.ACQUA)
        occ = np.asarray(protetto[sl]) & ~(tipo_w > 0)                # costruito, strade escluse
        ostacolo = binary_dilation(occ | acqua, iterations=1) | fisso
        for c in citta:
            r = max(c.lato, c.raggio, c.lz, c.lx) + 5
            ostacolo[max(0, c.z - r - z0):max(0, c.z + r + 1 - z0),
                     max(0, c.x - r - x0):max(0, c.x + r + 1 - x0)] = True
        riservato = np.zeros(ostacolo.shape, bool)
        for (z, x) in occupate:
            if z0 <= z < z1 and x0 <= x < x1:
                riservato[z - z0, x - x0] = True
        ostacolo |= binary_dilation(riservato, iterations=2)
        vicino_strada = binary_dilation(strada, iterations=2) & ~strada
        hw = np.asarray(h[sl]).astype(np.float32)
        liscio = uniform_filter(hw, 7)
        self.quota = np.where(strada, np.asarray(quota_s[sl]), np.round(liscio)).astype(np.int32)
        self.quota = np.maximum(self.quota, 63)
        scavo = np.abs(self.quota - hw)
        self.passabile = ~ostacolo & (scavo <= SCAVO_MAX)
        self.costo = (1.0 + 0.35 * scavo + np.where(vicino_strada, 1.5, 0.0)
                      + np.where(strada, 6.0, 0.0)).astype(np.float32)
        self.strada = strada
        self.h = hw

    def locale(self, cella):
        return cella[0] - self.z0, cella[1] - self.x0


def _scegli_ancora(fin: _Finestra, c, verso, usate: list):
    """Una cella libera appena fuori dalla citta' `c`, dal lato di `verso`, lontana dalle altre ancore."""
    r0 = max(c.lato, c.raggio, c.lz, c.lx) + 10
    migliore = None
    for r in range(r0, r0 + 32, 2):
        for k in range(72):
            a = 2 * np.pi * k / 72
            z, x = int(round(c.z + r * np.sin(a))), int(round(c.x + r * np.cos(a)))
            lz, lx = z - fin.z0, x - fin.x0
            if not (0 <= lz < fin.passabile.shape[0] and 0 <= lx < fin.passabile.shape[1]):
                continue
            if not fin.passabile[lz, lx] or fin.strada[lz, lx]:
                continue
            if any(abs(z - uz) + abs(x - ux) < 14 for uz, ux in usate):
                continue
            d = (z - verso.z) ** 2 + (x - verso.x) ** 2
            if migliore is None or d < migliore[0]:
                migliore = (d, z, x)
    return None if migliore is None else (migliore[1], migliore[2])


def pianifica(tipo_strada, quota_strada, h, cls, protetto, citta: list, meta: int,
              nomi_citta: dict | None = None, densita: float = 1.0, seed: int = 0,
              max_stazioni: int = 14) -> Ferrovia:
    """Rete e stazioni. I binari escono in coordinate di GIOCO (per chunk), le stazioni sono
    `Sito` in coordinate di MAPPA."""
    fer = Ferrovia()
    if tipo_strada is None or densita <= 0 or len(citta) < 2:
        return fer
    H, W = h.shape
    nomi_citta = nomi_citta or {}
    grandi = sorted(citta, key=lambda c: -(c.raggio + c.lato))[:max(2, int(round(max_stazioni * densita)))]

    coppie = set()
    for i, ci in enumerate(grandi):
        vicini = sorted((j for j in range(len(grandi)) if j != i),
                        key=lambda j: (grandi[j].x - ci.x) ** 2 + (grandi[j].z - ci.z) ** 2)[:3]
        coppie.update((min(i, j), max(i, j)) for j in vicini)
    archi = sorted((float(np.hypot(grandi[j].x - grandi[i].x, grandi[j].z - grandi[i].z)), i, j)
                   for i, j in coppie)
    archi = [a for a in archi if a[0] <= DISTANZA_MAX]

    padre = list(range(len(grandi)))

    def radice(k):
        while padre[k] != k:
            padre[k] = padre[padre[k]]
            k = padre[k]
        return k

    occupate: set = set()
    ancore_usate = {i: [] for i in range(len(grandi))}
    linee = []                                            # (i, j, cammino, quota locale->globale)
    extra = 0
    for d, i, j in archi:
        ri, rj = radice(i), radice(j)
        if ri == rj and extra >= max(1, len(grandi) // 6):
            continue
        ci, cj = grandi[i], grandi[j]
        z0 = max(0, min(ci.z, cj.z) - MARGINE_FINESTRA - 30)
        z1 = min(H, max(ci.z, cj.z) + MARGINE_FINESTRA + 31)
        x0 = max(0, min(ci.x, cj.x) - MARGINE_FINESTRA - 30)
        x1 = min(W, max(ci.x, cj.x) + MARGINE_FINESTRA + 31)
        fin = _Finestra(z0, z1, x0, x1, tipo_strada, quota_strada, h, cls, protetto, citta, occupate)
        ai = _scegli_ancora(fin, ci, cj, ancore_usate[i])
        aj = _scegli_ancora(fin, cj, ci, ancore_usate[j])
        if ai is None or aj is None:
            continue
        cam = _percorso(fin.passabile, fin.costo, fin.quota, fin.strada, fin.locale(ai), fin.locale(aj))
        if cam is None or len(cam) < 12:
            continue
        if ri == rj:
            extra += 1
        padre[ri] = rj
        ancore_usate[i].append(ai)
        ancore_usate[j].append(aj)
        forme = _forme(cam, fin.quota)
        glob = [(z + z0, x + x0) for z, x in cam]
        occupate.update(glob)
        linee.append((i, j, cam, glob, forme, fin))
    if not linee:
        return fer

    # le celle: forma, alimentazione, quota di posa e del suolo
    celle: dict = {}
    for i, j, cam, glob, forme, fin in linee:
        for k, ((lz, lx), (z, x), f) in enumerate(zip(cam, glob, forme)):
            sale = f.startswith("ascending")
            alim = sale or (k % PASSO_ALIMENTATO == PASSO_ALIMENTATO // 2 and f in ("north_south", "east_west"))
            celle[(z, x)] = (f, alim, int(fin.quota[lz, lx]), int(round(float(fin.h[lz, lx]))))
        fer.linee.append((grandi[i], grandi[j], len(cam)))

    # stazioni: un rettilineo piano in testa a ogni linea, fuori da tutto il resto
    for i, j, cam, glob, forme, fin in linee:
        for estremo, c in ((0, grandi[i]), (1, grandi[j])):
            s = _stazione(glob if estremo == 0 else glob[::-1], celle, c, fin, seed + 31 * i + j + estremo)
            if s is not None and not any(abs(s.x - t.x) + abs(s.z - t.z) < 30 for t in fer.stazioni):
                s.extra["citta"] = (c.x, c.z)
                s.nome = "Stazione di " + nomi_citta.get((c.x, c.z), "")
                fer.stazioni.append(s)

    for (z, x), (f, alim, y, suolo) in celle.items():
        gx, gz = x - meta, z - meta
        fer.rotaie.setdefault((gx >> 4, gz >> 4), []).append((gx & 15, gz & 15, y, f, alim, suolo))
    fer.celle = set(celle)
    fer.lunghezza = len(celle)
    return fer


def _stazione(glob: list, celle: dict, c, fin: _Finestra, seme: int):
    """Un tratto rettilineo e piano nei primi 60 passi della linea, con lo spazio per la stazione."""
    from .siti import Sito
    sito = None
    for k in range(LUNGHEZZA_RETTILINEO, min(len(glob) - LUNGHEZZA_RETTILINEO, 70)):
        z, x = glob[k]
        f = celle[(z, x)][0]
        if f not in ("north_south", "east_west"):
            continue
        dz, dx = (1, 0) if f == "north_south" else (0, 1)
        q = celle[(z, x)][2]
        ok = True
        for t in range(-LUNGHEZZA_RETTILINEO, LUNGHEZZA_RETTILINEO + 1):
            v = celle.get((z + dz * t, x + dx * t))
            if v is None or v[0] != f or v[2] != q:
                ok = False
                break
        if not ok:
            continue
        # l'ingombro: u -12..12 lungo il binario, v da -10 (banchina) a +8 (binario morto)
        libero = True
        for u in range(-12, 13, 2):
            for v_ in range(-10, 9, 2):
                pz, px = (z + dz * u + dx * v_, x + dx * u + dz * v_)
                lz_, lx_ = pz - fin.z0, px - fin.x0
                if not (0 <= lz_ < fin.passabile.shape[0] and 0 <= lx_ < fin.passabile.shape[1]):
                    libero = False
                    break
                if abs(v_) > 1 and ((not fin.passabile[lz_, lx_]) or (pz, px) in celle):
                    libero = False
                    break
            if not libero:
                break
        if libero:
            sito = Sito(tipo="stazione", x=x, z=z, y=q, seme=seme * 7919 + 13,
                        extra={"asse": "z" if f == "north_south" else "x", "citta": None})
            break
    return sito


# --------------------------------------------------------------------------
# Scrittura dei binari
# --------------------------------------------------------------------------

def blocchi_rotaia(scrittore) -> dict:
    """{(forma, alimentato): id}, piu' 'redstone' e 'ghiaia' per il letto."""
    t = {}
    for f in FORME:
        t[(f, False)] = scrittore.blocco("rail", shape=f)
        if f in ("north_south", "east_west") or f.startswith("ascending"):
            t[(f, True)] = scrittore.blocco("powered_rail", shape=f, powered="true")
    t["redstone"] = scrittore.blocco("redstone_block")
    t["ghiaia"] = scrittore.blocco("gravel")
    return t


def posa(blocchi: np.ndarray, tav: dict, aria: int, ox: int, oz: int, y0: int,
         ferrovia: Ferrovia, cx: int, cz: int) -> None:
    """Rilevato e binari nel chunk (16, H, 16) di coordinate di gioco (cx, cz)."""
    H = blocchi.shape[1]
    for lx, lz, y, forma, alim, suolo in ferrovia.rotaie.get((cx, cz), ()):
        ya = y - y0
        cima = suolo - 1 - y0                  # l'ultimo blocco pieno del suolo
        if not (3 <= ya < H - 4):
            continue
        if cima >= ya:                         # trincea: si scava fin sopra il binario
            blocchi[lx, ya:min(H, cima + 4), lz] = aria
        elif cima < ya - 1:                    # rilevato: ghiaia fino al suolo
            blocchi[lx, max(1, cima + 1):ya, lz] = tav["ghiaia"]
        blocchi[lx, ya + 1:ya + 4, lz] = aria
        if alim and (forma, True) in tav:
            blocchi[lx, ya - 1, lz] = tav["redstone"]
            blocchi[lx, ya, lz] = tav[(forma, True)]
        else:
            blocchi[lx, ya - 1, lz] = tav["ghiaia"]
            blocchi[lx, ya, lz] = tav[(forma, False)]
