"""L'End di Arda: isole sospese nel vuoto, con le cittadelle di purpur.

L'isola centrale e il drago restano del gioco (qui non si scrive niente a meno di 150
blocchi dall'origine). Intorno, a 170-460 blocchi, galleggiano sette isole di pietra
dell'End: cinque con soli cori e due con una CITTADELLA - una torre di purpur con la scala a
pioli, shulker sul tetto, endermen di guardia e forzieri con le cose buone; in una, nel
forziere in cima, un paio di ali (elytra) per tornare giu'.

DIMENSIONE X. In mezzo a tutto sorge la CITTADELLA DI VECNA, un'isola di ardesia e ossidiana
piangente: la Camera del Grande Pendolo, con l'orologio fermo alle 4:44, il pendolo d'oro
appeso a una catena e sotto un forziere con le ali (elytra). Dalla cittadella partono i PONTI
TENTACOLARI, serpenti di sculk che si arcuano nel vuoto fino alle altre isole; sulle isole di
cori ci sono i BOZZOLI di sculk (con un forziere dentro) e gli OBELISCHI D'OMBRA.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nether import ALT, _applica, _s

RAGGIO_MIN = 170
RAGGIO_MAX = 460
VECNA_RAGGIO = 58
NOMI_CITTADELLA = ("Cittadella di Purpur", "Torre degli Sguardi", "Guglia del Silenzio",
                   "Faro del Vuoto")

BLOCCHI: dict[str, tuple[str, dict]] = {
    "aria": ("air", {}), "pietra": ("end_stone", {}), "purpur": ("purpur_block", {}),
    "colonna": ("purpur_pillar", {"axis": "y"}), "mattoni": ("end_stone_bricks", {}),
    "asta": ("end_rod", {"facing": "up"}),
    "cor_pianta": ("chorus_plant", {"up": "true", "down": "true", "north": "false",
                                    "south": "false", "east": "false", "west": "false"}),
    "cor_fiore": ("chorus_flower", {"age": "5"}),
    "scala": ("ladder", {"facing": "east", "waterlogged": "false"}),
    "obsidiana": ("obsidian", {}),
    "ds_mattoni": ("deepslate_bricks", {}), "ds_tegole": ("deepslate_tiles", {}),
    "ds_liscia": ("polished_deepslate", {}), "piange": ("crying_obsidian", {}),
    "sculk": ("sculk", {}), "catalizzatore": ("sculk_catalyst", {"bloom": "false"}),
    "catena": ("chain", {"axis": "y"}), "oro": ("gold_block", {}),
    "bianco": ("concrete", {"color": "white"}), "nero": ("concrete", {"color": "black"}),
    "lanterna_anime": ("soul_lantern", {"hanging": "true"}),
}


@dataclass
class Isola:
    x: int
    z: int
    y: int
    raggio: int
    spessore: int
    cittadella: bool = False
    nome: str = ""
    tesoro: bool = False
    vecna: bool = False
    bozzoli: list = field(default_factory=list)       # [(dx, dz)] rispetto al centro
    obelischi: list = field(default_factory=list)     # [(dx, dz, altezza)]

    def bbox(self) -> tuple[int, int, int, int]:
        r = self.raggio + 2
        return self.x - r, self.z - r, self.x + r, self.z + r


@dataclass
class PianoEnd:
    seed: int
    isole: list
    creature: list
    ponti: list = field(default_factory=list)          # [array (n, 5): x, y, z, raggio, corpo]
    indice_ponti: dict = field(default_factory=dict)   # {(cx, cz): [(x, y, z, raggio, corpo)]}


class Tavolozza:
    def __init__(self, scrittore) -> None:
        self.id = {k: scrittore.blocco(n, **pr) for k, (n, pr) in BLOCCHI.items()}
        self.id["aria"] = scrittore.id_aria
        from .bauli import Tavolozza as TB
        self.bauli = TB(scrittore)
        self.forziere = {f: scrittore.blocco("chest", facing=f, connection="none", material="wood")
                         for f in ("north", "south", "east", "west")}


def pianifica(lato: int, seed: int = 0) -> PianoEnd:
    rng = np.random.default_rng(seed + 5150)
    n = 7
    fuori: list[Isola] = []
    for k in range(n):
        ang = 2 * np.pi * (k + rng.random() * 0.5) / n
        d = float(rng.integers(RAGGIO_MIN, RAGGIO_MAX))
        cit = k in (1, 4)
        fuori.append(Isola(x=int(d * np.cos(ang)), z=int(d * np.sin(ang)),
                           y=int(rng.integers(55, 76)),
                           raggio=int(rng.integers(30, 44) if cit else rng.integers(18, 34)),
                           spessore=int(rng.integers(8, 15)), cittadella=cit,
                           nome=NOMI_CITTADELLA[k % len(NOMI_CITTADELLA)] if cit else "",
                           tesoro=False))
        i = fuori[-1]
        if not cit:
            for _ in range(int(rng.integers(1, 3))):
                i.bozzoli.append((int(rng.integers(-i.raggio // 2, i.raggio // 2 + 1)),
                                  int(rng.integers(-i.raggio // 2, i.raggio // 2 + 1))))
            a = float(rng.random() * 2 * np.pi)
            for _ in range(int(rng.integers(1, 3))):
                a += 2.4
                i.obelischi.append((int(i.raggio * 0.6 * np.cos(a)), int(i.raggio * 0.6 * np.sin(a)),
                                    int(rng.integers(10, 17))))
    # la Cittadella di Vecna: dove c'e' piu' spazio, in mezzo alle altre
    migliore = None
    for k in range(24):
        ang = 2 * np.pi * k / 24 + rng.random() * 0.2
        d = float(rng.integers(300, 420))
        x, z = int(d * np.cos(ang)), int(d * np.sin(ang))
        libero = min(np.hypot(x - i.x, z - i.z) - i.raggio for i in fuori)
        if migliore is None or libero > migliore[0]:
            migliore = (libero, x, z)
    _, vx, vz = migliore
    fuori.append(Isola(x=vx, z=vz, y=72, raggio=VECNA_RAGGIO, spessore=18, vecna=True, tesoro=True,
                       nome="Cittadella di Vecna"))
    p = PianoEnd(seed=seed, isole=fuori, creature=[])
    _ponti(p, rng)
    p.creature = _creature(p)
    return p


def _ponti(p: PianoEnd, rng: np.random.Generator) -> None:
    """I ponti tentacolari: da Vecna alle cittadelle e alle due isole piu' vicine."""
    vecna = next(i for i in p.isole if i.vecna)
    altre = sorted((i for i in p.isole if not i.vecna),
                   key=lambda i: (i.x - vecna.x) ** 2 + (i.z - vecna.z) ** 2)
    mete = [i for i in altre if i.cittadella] + [i for i in altre if not i.cittadella][:2]
    for m in mete:
        pts = _ponte(vecna, m, rng)
        p.ponti.append(pts)
        for (x, y, z, r, corpo) in pts:
            r_ = int(np.ceil(r))
            for cx in range((x - r_) >> 4, ((x + r_) >> 4) + 1):
                for cz in range((z - r_) >> 4, ((z + r_) >> 4) + 1):
                    p.indice_ponti.setdefault((cx, cz), []).append((x, y, z, float(r), int(corpo)))


def _ponte(a: Isola, b: Isola, rng: np.random.Generator) -> list:
    """Punti (x, y, z, raggio, corpo) di un ponte serpeggiante da `a` a `b`, con tentacoli
    che pendono nel vuoto. `corpo` = 1: tratto percorribile; 0: tentacolo."""
    lung = float(np.hypot(b.x - a.x, b.z - a.z))
    ux, uz = (b.x - a.x) / lung, (b.z - a.z) / lung
    sx, sz = a.x + ux * a.raggio * 0.85, a.z + uz * a.raggio * 0.85
    ex, ez = b.x - ux * b.raggio * 0.85, b.z - uz * b.raggio * 0.85
    L = float(np.hypot(ex - sx, ez - sz))
    n = max(2, int(L * 1.5))
    onde = float(rng.integers(2, 5))
    ampiezza = float(rng.integers(4, 8))
    sag = float(rng.integers(6, 14))
    fase = float(rng.random() * 2 * np.pi)
    visti: set = set()
    fuori: list = []
    for k in range(n + 1):
        t = k / n
        lat = ampiezza * np.sin(2 * np.pi * onde * t + fase) * np.sin(np.pi * t)
        x = sx + (ex - sx) * t - uz * lat
        z = sz + (ez - sz) * t + ux * lat
        y = a.y + (b.y - a.y) * t - sag * 2 * t * (1 - t)
        r = 2.3 + 0.7 * np.sin(np.pi * 3 * t)
        punto = (int(round(x)), int(round(y)), int(round(z)))
        if punto in visti:
            continue
        visti.add(punto)
        fuori.append((*punto, float(r), 1))
        if k % 11 == 5:                                    # un tentacolo che pende
            for j in range(1, int(rng.integers(5, 10))):
                fuori.append((punto[0] + (j % 2), punto[1] - 2 - j, punto[2], max(0.6, 1.8 - j * 0.2), 0))
    return fuori


def _creature(p: PianoEnd) -> list:
    from .torri import Combattente
    fuori = []
    n = 0
    for i in p.isole:
        if i.vecna:
            for dx, dz in ((-6, -4), (6, -4), (-6, 4), (6, 4), (0, 7), (0, -7)):
                n += 1
                fuori.append(Combattente(x=i.x + dx + 0.5, y=float(i.y + 2), z=i.z + dz + 0.5,
                                         specie="enderman", seme=p.seed * 977 + 700000 + n))
            for dx, dz in ((-16, -16), (16, -16), (-16, 16), (16, 16)):
                n += 1
                fuori.append(Combattente(x=i.x + dx + 0.5, y=float(i.y + 26), z=i.z + dz + 0.5,
                                         specie="shulker", seme=p.seed * 977 + 700000 + n))
            continue
        if i.cittadella:
            top = i.y + 26
            for dx, dz in ((-3, -3), (3, -3), (-3, 3), (3, 3)):
                n += 1
                fuori.append(Combattente(x=i.x + dx + 0.5, y=float(top), z=i.z + dz + 0.5,
                                         specie="shulker", seme=p.seed * 977 + 700000 + n))
            for dx, dz in ((6, 0), (-6, 0), (0, 6)):
                n += 1
                fuori.append(Combattente(x=i.x + dx + 0.5, y=float(i.y + 1), z=i.z + dz + 0.5,
                                         specie="enderman", seme=p.seed * 977 + 700000 + n))
        else:
            for dx, dz in i.bozzoli:
                n += 1
                fuori.append(Combattente(x=i.x + dx + 0.5, y=float(i.y + 2), z=i.z + dz + 0.5,
                                         specie="enderman", seme=p.seed * 977 + 700000 + n))
            for dx, dz in ((4, 4), (-5, 2)):
                n += 1
                fuori.append(Combattente(x=i.x + dx + 0.5, y=float(i.y + 2), z=i.z + dz + 0.5,
                                         specie="enderman", seme=p.seed * 977 + 700000 + n))
    return fuori


def _scatole_cittadella(i: Isola) -> tuple[list, list]:
    cx, cz, y = i.x, i.z, i.y
    sc = []
    sc.append(_s(cx - 8, y, cz - 8, cx + 8, y, cz + 8, "purpur"))                # il basamento
    sc.append(_s(cx - 4, y + 1, cz - 4, cx + 4, y + 24, cz + 4, "mattoni"))      # la torre
    sc.append(_s(cx - 3, y + 1, cz - 3, cx + 3, y + 23, cz + 3, "aria"))
    for sx in (-4, 4):
        for sz in (-4, 4):
            sc.append(_s(cx + sx, y + 1, cz + sz, cx + sx, y + 24, cz + sz, "colonna"))
    for fy in (y + 8, y + 16):                                                   # i solai
        sc.append(_s(cx - 3, fy, cz - 3, cx + 3, fy, cz + 3, "purpur"))
    sc.append(_s(cx - 3, y + 1, cz, cx - 3, y + 25, cz, "scala"))                # la scala a pioli
    sc.append(_s(cx + 4, y + 1, cz - 1, cx + 4, y + 3, cz + 1, "aria"))          # la porta
    sc.append(_s(cx - 4, y + 25, cz - 4, cx + 4, y + 25, cz + 4, "purpur"))      # il tetto
    sc.append(_s(cx - 3, y + 25, cz, cx - 3, y + 25, cz, "scala"))               # il pozzetto
    for sx in (-4, 4):
        for sz in (-4, 4):
            sc.append(_s(cx + sx, y + 26, cz + sz, cx + sx, y + 27, cz + sz, "colonna"))
            sc.append(_s(cx + sx, y + 28, cz + sz, cx + sx, y + 28, cz + sz, "asta"))
    for fy in (y + 4, y + 12, y + 20):
        sc.append(_s(cx + 3, fy, cz + 3, cx + 3, fy, cz + 3, "asta"))
    forzieri = [(cx + 2, y + 1, cz + 2, "end_citta", "west"),
                (cx + 2, y + 9, cz - 2, "end_citta", "west"),
                (cx + 2, y + 17, cz + 2, "end_tesoro" if i.tesoro else "end_citta", "west")]
    return sc, forzieri


def _scatole_vecna(i: Isola) -> tuple[list, list]:
    """La cittadella: sala del Pendolo (21 x 21 x 15), torre centrale, quattro torrette."""
    cx, cz, y = i.x, i.z, i.y
    sc = []
    sc.append(_s(cx - 22, y, cz - 22, cx + 22, y, cz + 22, "ds_tegole"))             # la piazza
    sc.append(_s(cx - 12, y + 1, cz - 12, cx + 12, y + 16, cz + 12, "ds_mattoni"))   # il mastio
    sc.append(_s(cx - 10, y + 1, cz - 10, cx + 10, y + 15, cz + 10, "aria"))         # la sala
    sc.append(_s(cx - 10, y, cz - 10, cx + 10, y, cz + 10, "ds_liscia"))
    sc.append(_s(cx + 11, y + 1, cz - 2, cx + 12, y + 4, cz + 2, "aria"))            # la porta, a est
    for fz in (-8, -4, 4, 8):                                                       # feritoie piangenti
        for fx in (-12, 12):
            sc.append(_s(cx + fx, y + 6, cz + fz, cx + fx, y + 10, cz + fz, "piange"))
    for px in (-8, 8):                                                              # colonne
        for pz in (-8, 8):
            sc.append(_s(cx + px, y + 1, cz + pz, cx + px, y + 14, cz + pz, "ds_liscia"))
            sc.append(_s(cx + px, y + 15, cz + pz, cx + px, y + 15, cz + pz, "piange"))
    sc.append(_s(cx - 13, y + 16, cz - 13, cx + 13, y + 16, cz + 13, "ds_tegole"))   # il tetto
    sc.append(_s(cx - 4, y + 16, cz - 4, cx + 4, y + 16, cz + 4, "aria"))            # il foro della torre
    sc.append(_s(cx - 5, y + 17, cz - 5, cx + 5, y + 44, cz + 5, "ds_mattoni"))      # la torre centrale
    sc.append(_s(cx - 4, y + 17, cz - 4, cx + 4, y + 44, cz + 4, "aria"))
    sc.append(_s(cx - 4, y + 1, cz, cx - 4, y + 44, cz, "scala"))                    # i pioli, dalla sala alla cima
    sc.append(_s(cx - 6, y + 45, cz - 6, cx + 6, y + 45, cz + 6, "ds_tegole"))
    sc.append(_s(cx - 4, y + 45, cz, cx - 4, y + 45, cz, "aria"))                    # il pozzetto
    for sx in (-6, 6):
        for sz in (-6, 6):
            sc.append(_s(cx + sx, y + 46, cz + sz, cx + sx, y + 49, cz + sz, "ds_mattoni"))
            sc.append(_s(cx + sx, y + 50, cz + sz, cx + sx, y + 50, cz + sz, "piange"))
    sc.append(_s(cx, y + 46, cz, cx, y + 60, cz, "ds_liscia"))
    sc.append(_s(cx, y + 61, cz, cx, y + 61, cz, "asta"))
    for sx in (-16, 16):                                                            # le quattro torrette
        for sz in (-16, 16):
            sc.append(_s(cx + sx - 2, y + 1, cz + sz - 2, cx + sx + 2, y + 24, cz + sz + 2, "ds_mattoni"))
            sc.append(_s(cx + sx, y + 25, cz + sz, cx + sx, y + 28, cz + sz, "piange"))
    # --- la Camera del Grande Pendolo: l'orologio fermo alle 4:44 sulla parete nord
    ocy = y + 9
    zc = cz - 10
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            r = (dx * dx + dy * dy) ** 0.5
            if r <= 5.5:
                sc.append(_s(cx + dx, ocy + dy, zc, cx + dx, ocy + dy, zc, "bianco"))
            elif r <= 6.5:
                sc.append(_s(cx + dx, ocy + dy, zc, cx + dx, ocy + dy, zc, "nero"))
    for ang in range(0, 360, 30):                                                   # le dodici ore
        a = np.radians(ang)
        hx, hy = cx + int(round(4.6 * np.sin(a))), ocy + int(round(4.6 * np.cos(a)))
        sc.append(_s(hx, hy, zc, hx, hy, zc, "oro"))
    for lung, ang in ((5, 2 * np.pi * 44 / 60), (3, 2 * np.pi * (4 + 44 / 60) / 12)):   # minuti, ore
        for t in range(0, lung + 1):
            hx, hy = cx + int(round(t * np.sin(ang))), ocy + int(round(t * np.cos(ang)))
            sc.append(_s(hx, hy, zc + 1, hx, hy, zc + 1, "nero"))
    # il pendolo: una catena dal soffitto e il peso d'oro, sopra il forziere
    sc.append(_s(cx, y + 8, cz + 3, cx, y + 15, cz + 3, "catena"))
    sc.append(_s(cx - 1, y + 6, cz + 2, cx + 1, y + 7, cz + 4, "oro"))
    sc.append(_s(cx - 2, y + 1, cz + 1, cx + 2, y + 1, cz + 5, "ds_liscia"))          # il palchetto sotto
    forzieri = [(cx, y + 2, cz + 3, "end_tesoro", "west"),
                (cx + 3, y + 46, cz + 3, "end_citta", "west"),
                (cx - 3, y + 46, cz - 3, "end_citta", "east")]
    return sc, forzieri


def _bozzolo(out: np.ndarray, tav: Tavolozza, ox: int, oz: int, bx: int, by: int, bz: int) -> None:
    """Guscio ellittico di sculk (raggi 4 x 5 x 4): l'interno resta com'e'."""
    xs = (np.arange(16) + ox - bx)[:, None, None] / 4.0
    ys = (np.arange(ALT) - by)[None, :, None] / 5.0
    zs = (np.arange(16) + oz - bz)[None, None, :] / 4.0
    d2 = xs ** 2 + ys ** 2 + zs ** 2
    out[(d2 <= 1.0) & (d2 > 0.62)] = tav.id["sculk"]


def _ponti_chunk(out: np.ndarray, tav: Tavolozza, p: PianoEnd, cx: int, cz: int) -> None:
    pts = p.indice_ponti.get((cx, cz))
    if not pts:
        return
    I = tav.id
    ox, oz = cx * 16, cz * 16
    corpo = []
    for (x, y, z, r, c) in pts:
        r_ = int(np.ceil(r))
        x0, x1 = max(x - r_, ox), min(x + r_, ox + 15)
        z0, z1 = max(z - r_, oz), min(z + r_, oz + 15)
        y0, y1 = max(y - r_, 0), min(y + r_, ALT - 1)
        if x0 > x1 or z0 > z1 or y0 > y1:
            continue
        gx = (np.arange(x0, x1 + 1) - x)[:, None, None]
        gy = (np.arange(y0, y1 + 1) - y)[None, :, None]
        gz = (np.arange(z0, z1 + 1) - z)[None, None, :]
        dentro = gx ** 2 + gy ** 2 + gz ** 2 <= r * r
        X = np.arange(x0, x1 + 1)[:, None, None]
        Y = np.arange(y0, y1 + 1)[None, :, None]
        Z = np.arange(z0, z1 + 1)[None, None, :]
        scuro = ((X * 73856093) ^ (Y * 19349663) ^ (Z * 83492791)) % 100 < 58
        sub = out[x0 - ox:x1 - ox + 1, y0:y1 + 1, z0 - oz:z1 - oz + 1]
        sub[dentro & scuro] = I["sculk"]
        sub[dentro & ~scuro] = I["ds_tegole"]
        if c:
            corpo.append((x, y, z))
    for (x, y, z) in corpo:                    # il passaggio: pavimento largo 3 e due blocchi d'aria
        x0, x1 = max(x - 1, ox), min(x + 1, ox + 15)
        z0, z1 = max(z - 1, oz), min(z + 1, oz + 15)
        if x0 > x1 or z0 > z1 or not (0 <= y < ALT - 3):
            continue
        out[x0 - ox:x1 - ox + 1, y, z0 - oz:z1 - oz + 1] = I["ds_tegole"]
        out[x0 - ox:x1 - ox + 1, y + 1:y + 3, z0 - oz:z1 - oz + 1] = I["aria"]


def blocchi_chunk(p: PianoEnd, tav: Tavolozza, cx: int, cz: int, id_bioma: int):
    """(blocchi, biomi, forzieri) di un chunk oppure None se nessuna isola ci passa."""
    ox, oz = cx * 16, cz * 16
    vicine = [i for i in p.isole
              if i.bbox()[2] >= ox and i.bbox()[0] <= ox + 15
              and i.bbox()[3] >= oz and i.bbox()[1] <= oz + 15]
    ponti = p.indice_ponti.get((cx, cz))
    if not vicine and not ponti:
        return None
    I = tav.id
    out = np.full((16, ALT, 16), I["aria"], np.uint32)
    xs = (np.arange(16) + ox)[:, None]
    zs = (np.arange(16) + oz)[None, :]
    ys = np.arange(ALT)[None, :, None]
    rng = np.random.default_rng((p.seed * 1000003 + cx * 7919 + cz * 104729) & 0x7FFFFFFF)
    forzieri = {}
    for i in vicine:
        d = np.hypot(xs - i.x, zs - i.z) / float(i.raggio)
        dentro = d < 1.0
        prof = (i.spessore * np.clip(1.0 - d ** 2, 0, 1) ** 0.7).astype(np.int32)
        # un po' di rilievo sulla cima, per non farla piatta come un tavolo
        gobba = (np.sin((xs + i.z) / 5.0) * np.cos((zs + i.x) / 6.0) * 1.2).astype(np.int32)
        if i.vecna:
            gobba = gobba * 0
        top = i.y + np.where(d < 0.6, gobba, 0)
        fondo = i.y - prof
        solido = dentro[:, None, :] & (ys <= top[:, None, :]) & (ys >= fondo[:, None, :])
        out[solido] = I["pietra"]
        if i.cittadella or i.vecna:
            sc, fo = (_scatole_vecna(i) if i.vecna else _scatole_cittadella(i))
            for scatola in sc:
                _applica(out, tav, ox, oz, scatola)
            for (fx, fy, fz, tipo, faccia) in fo:
                if ox <= fx < ox + 16 and oz <= fz < oz + 16:
                    out[fx - ox, fy, fz - oz] = tav.forziere[faccia]
                    forzieri[(fx, fy, fz)] = tipo
        else:
            for (bdx, bdz) in i.bozzoli:                                          # i bozzoli di sculk
                bx, bz = i.x + bdx, i.z + bdz
                if ox - 5 <= bx <= ox + 20 and oz - 5 <= bz <= oz + 20:
                    _bozzolo(out, tav, ox, oz, bx, i.y + 3, bz)
                    if ox <= bx < ox + 16 and oz <= bz < oz + 16:
                        ty = int(top[bx - ox, bz - oz])
                        out[bx - ox, ty, bz - oz] = I["catalizzatore"]
                        out[bx - ox, ty + 1, bz - oz] = tav.forziere["south"]
                        forzieri[(bx, ty + 1, bz)] = "end_citta"
            for (odx, odz, oh) in i.obelischi:                                     # gli obelischi d'ombra
                ob = i.x + odx, i.z + odz
                for scatola in (_s(ob[0] - 1, i.y + 1, ob[1] - 1, ob[0] + 1, i.y + 1, ob[1] + 1, "ds_mattoni"),
                                _s(ob[0], i.y + 2, ob[1], ob[0], i.y + oh, ob[1], "ds_liscia"),
                                _s(ob[0], i.y + oh + 1, ob[1], ob[0], i.y + oh + 1, ob[1], "piange"),
                                _s(ob[0], i.y + oh + 2, ob[1], ob[0], i.y + oh + 2, ob[1], "asta")):
                    _applica(out, tav, ox, oz, scatola)
            for _ in range(int(rng.integers(2, 6))):                              # i cori
                lx, lz = int(rng.integers(0, 16)), int(rng.integers(0, 16))
                if not dentro[lx, lz] or d[lx, lz] > 0.75:
                    continue
                y = int(top[lx, lz])
                if out[lx, y, lz] != I["pietra"]:
                    continue
                alto = int(rng.integers(2, 5))
                for h in range(1, alto + 1):
                    out[lx, y + h, lz] = I["cor_pianta"]
                out[lx, y + alto + 1, lz] = I["cor_fiore"]
    _ponti_chunk(out, tav, p, cx, cz)
    biomi = np.full((16, 16), id_bioma, np.uint32)
    return out, biomi, forzieri


def sezione_elenco(p: PianoEnd) -> str:
    righe = ["", "## L'End", "",
             "Isole sospese intorno a quella del drago. Per andarci: "
             "`/execute in minecraft:the_end run tp @s X Y Z`.", ""]
    for i in p.isole:
        if i.vecna:
            righe.append(f"- **{i.nome}** (Dimensione X) - `/execute in minecraft:the_end run tp @s {i.x + 16} {i.y + 3} {i.z}`"
                         " - la Camera del Grande Pendolo, con le ali (elytra) nel forziere sotto il peso d'oro")
        elif i.cittadella:
            righe.append(f"- **{i.nome}** - `/execute in minecraft:the_end run tp @s {i.x} {i.y + 30} {i.z}`"
                         + (" - con le ali (elytra) nel forziere in cima" if i.tesoro else ""))
        else:
            righe.append(f"- Isola di cori - `/execute in minecraft:the_end run tp @s {i.x} {i.y + 4} {i.z}`")
    return "\n".join(righe) + "\n"
