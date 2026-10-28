"""Luoghi da scoprire: capanne di streghe, rovine sommerse, torri di segnalazione, labirinti
di siepi, arene, santuari, grotte di cristalli, capanni di pescatori, radure di funghi e un museo.

Tutti seguono la stessa regola: un SITO (tipo, centro, quota) scelto sul terreno da una
maschera, e un DISEGNO fatto di scatole ed ellissoidi di blocchi con un nome (`BLOCCHI`).
Aggiungere un luogo vuol dire scrivere la sua maschera, il suo disegno e un nome:
posa, indice per chunk, forzieri, spawner, creature e l'elenco in `luoghi.md` sono comuni.

Coordinate: i siti stanno in coordinate di MAPPA, le quote sono assolute; `y` del sito e' il
primo blocco libero sopra il terreno (il mare, per le rovine sommerse, e' sopra).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mappa as M
from .torri import Combattente

FACCE = ("north", "south", "east", "west")


# --------------------------------------------------------------------------
# Blocchi
# --------------------------------------------------------------------------

def _tavola() -> dict[str, tuple[str, dict]]:
    t: dict[str, tuple[str, dict]] = {
        "aria": ("air", {}), "acqua": ("water", {"level": "0"}), "terra": ("dirt", {}),
        "erba": ("grass_block", {"snowy": "false"}),
        "ciottoli": ("cobblestone", {}), "pietra": ("stone", {}),
        "mattoni": ("stone_bricks", {"variant": "normal"}),
        "mattoni_c": ("stone_bricks", {"variant": "cracked"}),
        "mattoni_m": ("stone_bricks", {"variant": "mossy"}),
        "mattoni_s": ("stone_bricks", {"variant": "chiseled"}),
        "assi": ("planks", {"material": "oak"}),
        "assi_ab": ("planks", {"material": "spruce"}),
        "assi_ds": ("planks", {"material": "dark_oak"}),
        "tronco": ("log", {"axis": "y", "material": "oak", "stripped": "false"}),
        "tronco_ds": ("log", {"axis": "y", "material": "dark_oak", "stripped": "false"}),
        "tronco_ab": ("log", {"axis": "y", "material": "spruce", "stripped": "false"}),
        "lastra_ds": ("slab", {"material": "dark_oak", "type": "bottom"}),
        "lastra_ds_alta": ("slab", {"material": "dark_oak", "type": "top"}),
        "lastra": ("slab", {"material": "oak", "type": "bottom"}),
        "recinto_ds": ("fence", {"material": "dark_oak", "east": "false", "north": "false",
                                 "south": "false", "west": "false"}),
        "recinto": ("fence", {"material": "oak", "east": "false", "north": "false",
                              "south": "false", "west": "false"}),
        "vetro": ("glass", {}),
        "muretto": ("wall", {"material": "stone_brick", "up": "true", "north": "none",
                             "south": "none", "east": "none", "west": "none"}),
        "sbarre": ("bars", {"material": "iron", "north": "false", "south": "false",
                            "east": "false", "west": "false"}),
        "lanterna": ("lantern", {"material": "iron", "hanging": "false"}),
        "lanterna_app": ("lantern", {"material": "iron", "hanging": "true"}),
        "banco": ("crafting_table", {}), "libreria": ("bookshelf", {}),
        "calderone": ("cauldron", {"cauldron_liquid": "water", "level": "3"}),
        "alambicco": ("brewing_stand", {"has_bottle_0": "true", "has_bottle_1": "false",
                                        "has_bottle_2": "true"}),
        "barile": ("barrel", {"facing": "up", "open": "false"}),
        "candela": ("candle", {"color": "default", "candles": "3", "lit": "true"}),
        "tappeto": ("carpet", {"color": "red"}),
        "tappeto_nero": ("carpet", {"color": "black"}),
        "siepe": ("leaves", {"material": "oak", "distance": "1", "persistent": "true",
                             "check_decay": "false"}),
        "foglie_ab": ("leaves", {"material": "spruce", "distance": "1", "persistent": "true",
                                 "check_decay": "false"}),
        "prismarino": ("prismarine", {"variant": "normal"}),
        "prism_mattoni": ("prismarine", {"variant": "bricks"}),
        "prism_scuro": ("prismarine", {"variant": "dark"}),
        "lanterna_mare": ("sea_lantern", {}),
        "blocco_oro": ("gold_block", {}),
        "ametista": ("amethyst_block", {}),
        "ametista_madre": ("budding_amethyst", {}),
        "calcite": ("calcite", {}), "tufo": ("tuff", {}),
        "ossidiana": ("obsidian", {}),
        "muschio": ("moss_block", {}),
        "luminite": ("glowstone", {}),
        "fungo_gambo": ("mushroom_stem", {"north": "true", "south": "true", "east": "true",
                                          "west": "true", "up": "true", "down": "true"}),
        "fungo_rosso": ("red_mushroom_block", {"north": "true", "south": "true", "east": "true",
                                               "west": "true", "up": "true", "down": "true"}),
        "fungo_bruno": ("brown_mushroom_block", {"north": "true", "south": "true", "east": "true",
                                                 "west": "true", "up": "true", "down": "true"}),
        "lana": ("wool", {"color": "white"}),
        "fieno": ("hay_block", {"axis": "y"}),
        "arenaria": ("sandstone", {"variant": "normal"}),
        "arenaria_s": ("sandstone", {"variant": "smooth"}),
        "spawner": ("spawner", {}),
        "lastra_ab": ("slab", {"material": "spruce", "type": "bottom"}),
        "lastra_ab_alta": ("slab", {"material": "spruce", "type": "top"}),
        "recinto_ab": ("fence", {"material": "spruce", "east": "false", "north": "false",
                                 "south": "false", "west": "false"}),
        "tronco_ab_s": ("log", {"axis": "y", "material": "spruce", "stripped": "true"}),
        "ferro": ("iron_block", {}), "carbone": ("coal_block", {}),
        "fuoco": ("campfire", {"facing": "north", "lit": "true", "signal_fire": "true"}),
        "rotaia:ns": ("rail", {"shape": "north_south"}),
        "rotaia:ew": ("rail", {"shape": "east_west"}),
    }
    for f in FACCE:
        t[f"scala_ds:{f}"] = ("stairs", {"material": "dark_oak", "facing": f,
                                         "half": "bottom", "shape": "straight"})
        t[f"scala_pietra:{f}"] = ("stairs", {"material": "stone_brick", "facing": f,
                                             "half": "bottom", "shape": "straight"})
        t[f"scala_assi:{f}"] = ("stairs", {"material": "oak", "facing": f,
                                           "half": "bottom", "shape": "straight"})
        t[f"pioli:{f}"] = ("ladder", {"facing": f, "waterlogged": "false"})
        t[f"torcia:{f}"] = ("torch", {"facing": f})
        t[f"forziere:{f}"] = ("chest", {"facing": f, "connection": "none", "material": "wood"})
        t[f"scala_ab:{f}"] = ("stairs", {"material": "spruce", "facing": f,
                                         "half": "bottom", "shape": "straight"})
        t[f"leggio:{f}"] = ("lectern", {"facing": f, "has_book": "false", "powered": "false"})
    return t


BLOCCHI = _tavola()


class Tavolozza:
    def __init__(self, scrittore) -> None:
        self.id = {k: scrittore.blocco(n, **pr) for k, (n, pr) in BLOCCHI.items()}
        self.id["aria"] = scrittore.id_aria
        self.aria = self.id["aria"]
        self.acqua = self.id["acqua"]
        self.bauli: list = []
        self.cache: dict = {}


# --------------------------------------------------------------------------
# Disegno
# --------------------------------------------------------------------------

class Disegno:
    """Elenco di primitive in coordinate di mappa e quote assolute."""

    def __init__(self) -> None:
        self.prim: list = []
        self.forzieri: list = []           # (x, y, z, tipo)
        self.spawner: list = []            # (x, y, z, mob)
        self.creature: list = []

    def b(self, x0, y0, z0, x1, y1, z1, k, vuoto=False):
        self.prim.append(("B", min(x0, x1), min(y0, y1), min(z0, z1),
                          max(x0, x1), max(y0, y1), max(z0, z1), k, vuoto))

    def p(self, x, y, z, k, vuoto=False):
        self.b(x, y, z, x, y, z, k, vuoto)

    def ell(self, cx, cy, cz, rx, ry, rz, k, vuoto=False):
        self.prim.append(("E", cx, cy, cz, rx, ry, rz, k, vuoto))

    def forziere(self, x, y, z, tipo, faccia):
        self.p(x, y, z, f"forziere:{faccia}")
        self.forzieri.append((x, y, z, tipo))


@dataclass
class Sito:
    tipo: str
    x: int
    z: int
    y: int
    seme: int = 0
    nome: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def mezzo(self) -> int:
        return TIPI[self.tipo]["mezzo"]

    def bbox(self) -> tuple[int, int, int, int]:
        m = self.mezzo + 2
        return self.x - m, self.z - m, self.x + m, self.z + m


def _rng(s: Sito) -> np.random.Generator:
    return np.random.default_rng(s.seme & 0x7FFFFFFF)


def _spazio(d: Disegno, s: Sito, r: int, alto: int, profondo: int = 6) -> None:
    """Pulisce l'ingombro da erba alta e alberi e riempie sotto fino al terreno."""
    d.b(s.x - r, s.y - profondo, s.z - r, s.x + r, s.y - 2, s.z + r, "terra", True)
    d.b(s.x - r, s.y - 1, s.z - r, s.x + r, s.y - 1, s.z + r, "erba")
    d.b(s.x - r, s.y, s.z - r, s.x + r, s.y + alto, s.z + r, "aria")


def _combattente(x, y, z, specie, seme, **kw) -> Combattente:
    return Combattente(x=x + 0.5, y=float(y), z=z + 0.5, specie=specie, seme=seme, **kw)


# --- 1. Capanna della strega ----------------------------------------------

NOMI_STREGHE = ("Mirtilla la Muschiosa", "Nonna Ortica", "Belladonna", "Zia Verbena",
                "Ginevra dei Funghi", "Madama Acetosella")


def disegna_strega(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.y
    _spazio(d, s, 7, 9)
    p = y + 3                                    # piano della capanna, su palafitte
    for dx, dz in ((-3, -3), (3, -3), (-3, 3), (3, 3)):
        d.b(x + dx, y - 3, z + dz, x + dx, p - 1, z + dz, "tronco_ds", True)
    d.b(x - 3, p, z - 3, x + 3, p, z + 3, "assi_ds")                       # pavimento
    d.b(x - 3, p + 1, z - 3, x + 3, p + 4, z + 3, "assi_ab")               # pareti
    d.b(x - 2, p + 1, z - 2, x + 2, p + 4, z + 2, "aria")
    for dx, dz in ((-3, -3), (3, -3), (-3, 3), (3, 3)):
        d.b(x + dx, p + 1, z + dz, x + dx, p + 4, z + dz, "tronco_ds")
    d.b(x - 4, p + 5, z - 4, x + 4, p + 5, z + 4, "assi_ds")               # tetto a gradoni
    d.b(x - 3, p + 6, z - 3, x + 3, p + 6, z + 3, "assi_ds")
    d.b(x - 2, p + 7, z - 2, x + 2, p + 7, z + 2, "lastra_ds")
    d.b(x - 3, p + 5, z - 3, x + 3, p + 5, z + 3, "assi_ds")
    d.b(x - 2, p + 5, z - 2, x + 2, p + 5, z + 2, "assi_ds")
    d.b(x + 3, p + 1, z - 1, x + 3, p + 2, z, "aria")                       # la porta (aperta)
    d.b(x - 3, p + 2, z - 1, x - 3, p + 3, z, "vetro")                      # finestre
    d.b(x + 1, p + 2, z - 3, x + 2, p + 3, z - 3, "vetro")
    # la scaletta d'ingresso: quattro gradini che salgono verso ovest, a filo con il pavimento
    for k in range(4):
        d.p(x + 7 - k, y + k, z, "scala_ds:west")
        if k:
            d.b(x + 7 - k, y, z, x + 7 - k, y + k - 1, z, "assi_ds")
    # dentro
    d.p(x - 2, p + 1, z - 2, "calderone")
    d.p(x - 2, p + 1, z - 1, "alambicco")
    d.p(x - 2, p + 1, z + 1, "banco")
    d.p(x - 2, p + 1, z + 2, "libreria")
    d.p(x - 2, p + 2, z + 2, "libreria")
    d.p(x, p + 4, z, "lanterna_app")
    d.b(x - 1, p + 1, z - 1, x + 1, p + 1, z + 1, "tappeto_nero")
    d.forziere(x + 2, p + 1, z + 2, "mago_tesoro", "west")
    d.forziere(x + 2, p + 1, z - 2, "mago_stanza", "west")
    d.p(x + 4, p + 1, z + 1, "lanterna")
    d.creature.append(_combattente(x, p + 1, z, "strega", s.seme + 1, capo=True,
                                   nome=s.nome.replace("Capanna di ", ""), livello=1))
    d.creature.append(_combattente(x + 5, y, z + 3, "strega", s.seme + 2))
    return d


# --- 2. Rovina sommersa (con il guardiano anziano) -------------------------

def disegna_sommersa(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.y
    # fondale spianato e sala 13x13x7 di prismarino con l'interno allagato
    d.b(x - 7, y - 2, z - 7, x + 7, y - 1, z + 7, "prismarino")
    d.b(x - 6, y, z - 6, x + 6, y + 7, z + 6, "prism_mattoni")
    d.b(x - 5, y + 1, z - 5, x + 5, y + 6, z + 5, "acqua")
    d.b(x - 1, y, z - 6, x + 1, y + 3, z - 6, "acqua")                      # l'ingresso a nord
    d.b(x - 1, y, z + 6, x + 1, y + 3, z + 6, "acqua")
    for dx, dz in ((-6, -6), (6, -6), (-6, 6), (6, 6)):
        d.b(x + dx, y, z + dz, x + dx, y + 9, z + dz, "prism_scuro")
        d.p(x + dx, y + 10, z + dz, "lanterna_mare")
    for dx, dz in ((-3, -3), (3, -3), (-3, 3), (3, 3)):
        d.b(x + dx, y + 1, z + dz, x + dx, y + 4, z + dz, "prismarino")
        d.p(x + dx, y + 5, z + dz, "lanterna_mare")
    # la stanza del tesoro: un cubo d'oro al centro, otto blocchi attorno
    d.b(x - 1, y + 1, z - 1, x + 1, y + 2, z + 1, "prism_scuro")
    d.p(x, y + 3, z, "blocco_oro")
    d.forziere(x - 2, y + 1, z, "sommersa", "east")
    d.forziere(x + 2, y + 1, z, "sommersa", "west")
    seme = s.seme
    d.creature.append(_combattente(x, y + 3, z + 3, "guardiano_anziano", seme + 1, capo=True,
                                   nome="Il Guardiano Anziano"))
    for k, (dx, dz) in enumerate(((-4, -4), (4, 4), (-4, 4), (4, -4))):
        d.creature.append(_combattente(x + dx, y + 4, z + dz, "guardiano", seme + 10 + k))
    return d


# --- 3. Torre di segnalazione sulla costa -----------------------------------

def disegna_segnale(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.y
    _spazio(d, s, 4, 20)
    d.b(x - 3, y, z - 3, x + 3, y, z + 3, "mattoni")
    d.b(x - 2, y + 1, z - 2, x + 2, y + 14, z + 2, "mattoni")
    d.b(x - 1, y + 1, z - 1, x + 1, y + 14, z + 1, "aria")
    d.b(x - 2, y + 15, z - 2, x + 2, y + 15, z + 2, "mattoni_s")
    d.b(x - 1, y + 15, z - 1, x + 1, y + 15, z + 1, "aria")                  # il botola/pozzetto
    d.b(x - 2, y + 16, z - 2, x + 2, y + 16, z + 2, "muretto")
    d.b(x - 1, y + 16, z - 1, x + 1, y + 16, z + 1, "aria")
    d.b(x - 1, y + 1, z + 1, x - 1, y + 15, z + 1, "pioli:north")
    d.b(x, y + 1, z - 2, x, y + 2, z - 2, "aria")                           # la porta
    d.p(x, y + 17, z, "lanterna")
    d.b(x, y + 15, z, x, y + 16, z, "luminite")
    for dy in (4, 8, 12):
        d.p(x, y + dy, z - 1, f"torcia:south")
    d.forziere(x + 1, y + 1, z - 1, "faro", "west")
    d.forziere(x, y + 16, z, "faro", "south")
    d.creature.append(_combattente(x, y + 1, z, "predone", s.seme + 1))
    return d


# --- 4. Labirinto di siepi --------------------------------------------------

def _labirinto(n: int, rng: np.random.Generator) -> np.ndarray:
    """Griglia (2n+1)^2: True = siepe. Percorso con DFS iterativa."""
    g = np.ones((2 * n + 1, 2 * n + 1), bool)
    pila = [(0, 0)]
    visti = {(0, 0)}
    g[1, 1] = False
    while pila:
        cx, cz = pila[-1]
        vicini = [(cx + dx, cz + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= cx + dx < n and 0 <= cz + dz < n and (cx + dx, cz + dz) not in visti]
        if not vicini:
            pila.pop()
            continue
        nx, nz = vicini[int(rng.integers(0, len(vicini)))]
        visti.add((nx, nz))
        g[1 + cz + nz, 1 + cx + nx] = False          # il passaggio fra le due celle
        g[1 + 2 * nz, 1 + 2 * nx] = False
        pila.append((nx, nz))
    return g


N_LABIRINTO = 8                                         # celle per lato: 17 x 17 blocchi


def disegna_labirinto(s: Sito) -> Disegno:
    d = Disegno()
    rng = _rng(s)
    n = N_LABIRINTO
    g = _labirinto(n, rng)
    lato = 2 * n + 1
    x0, z0 = s.x - lato // 2, s.z - lato // 2
    y = s.y
    d.b(x0 - 1, y - 6, z0 - 1, x0 + lato, y - 2, z0 + lato, "terra", True)
    d.b(x0 - 1, y - 1, z0 - 1, x0 + lato, y - 1, z0 + lato, "erba")
    d.b(x0 - 1, y, z0 - 1, x0 + lato, y + 4, z0 + lato, "aria")
    for iz in range(lato):
        riga = g[iz]
        ix = 0
        while ix < lato:                                 # tratti continui in una sola scatola
            if riga[ix]:
                j = ix
                while j + 1 < lato and riga[j + 1]:
                    j += 1
                d.b(x0 + ix, y, z0 + iz, x0 + j, y + 2, z0 + iz, "siepe")
                ix = j + 1
            else:
                ix += 1
    # ingresso a sud-ovest, tesoro al centro (la cella piu' lontana dall'ingresso)
    d.b(x0, y, z0 + 1, x0, y + 2, z0 + 1, "aria")
    cx, cz = x0 + lato // 2, z0 + lato // 2
    d.b(cx - 1, y, cz - 1, cx + 1, y + 2, cz + 1, "aria")
    d.b(cx - 1, y - 1, cz - 1, cx + 1, y - 1, cz + 1, "mattoni_m")
    d.forziere(cx, y, cz, "mago_tesoro", "south")
    d.p(cx - 1, y, cz - 1, "lanterna")
    d.p(cx + 1, y, cz + 1, "lanterna")
    d.creature.append(_combattente(cx + 1, y, cz - 1, "ragno_grotta", s.seme + 1))
    return d


# --- 5. Arena del Devastatore -------------------------------------------------

NOMI_DEVASTATORE = ("Brontolo il Devastatore", "Zanna Rossa", "Cornogrande", "il Rovinatore")
ARENE = ("Brontolo", "Zanna Rossa", "Cornogrande", "Rovina")      # `Arena di ...`


def disegna_arena(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.y
    _spazio(d, s, 13, 10, 10)
    d.b(x - 12, y - 1, z - 12, x + 12, y - 1, z + 12, "mattoni_m")
    d.b(x - 12, y, z - 12, x + 12, y + 3, z + 12, "mattoni")                 # muro d'anello
    d.b(x - 11, y, z - 11, x + 11, y + 4, z + 11, "aria")
    for dx, dz in ((-12, -12), (12, -12), (-12, 12), (12, 12), (0, -12), (0, 12), (-12, 0), (12, 0)):
        d.b(x + dx, y, z + dz, x + dx, y + 5, z + dz, "mattoni_s")
        d.p(x + dx, y + 6, z + dz, "lanterna")
    d.b(x - 1, y, z + 12, x + 1, y + 2, z + 12, "aria")                       # un solo ingresso, a sud
    d.b(x - 11, y + 4, z - 11, x + 11, y + 4, z - 11, "muretto")
    for dx, dz in ((-6, -6), (6, -6), (-6, 6), (6, 6)):                       # colonne nell'arena
        d.b(x + dx, y, z + dz, x + dx, y + 4, z + dz, "mattoni")
    d.b(x - 2, y, z - 2, x + 2, y, z + 2, "mattoni_c")
    d.forziere(x, y, z - 10, "boss_arena", "south")
    d.forziere(x - 2, y, z - 10, "boss_arena", "south")
    d.creature.append(_combattente(x, y, z, "devastatore", s.seme + 1, capo=True,
                                   nome=NOMI_DEVASTATORE[s.seme % len(NOMI_DEVASTATORE)], livello=1))
    return d


# --- 6. Santuario -------------------------------------------------------------

def disegna_santuario(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.y
    _spazio(d, s, 4, 7)
    d.b(x - 4, y - 1, z - 4, x + 4, y - 1, z + 4, "mattoni_m")
    d.b(x - 3, y, z - 3, x + 3, y, z + 3, "mattoni")
    for dx, dz in ((-3, -3), (3, -3), (-3, 3), (3, 3)):
        d.b(x + dx, y + 1, z + dz, x + dx, y + 4, z + dz, "mattoni_s")
    d.b(x - 3, y + 5, z - 3, x + 3, y + 5, z + 3, "mattoni")
    d.b(x - 2, y + 6, z - 2, x + 2, y + 6, z + 2, "lastra")
    d.b(x - 3, y + 1, z - 3, x + 3, y + 1, z - 3, "mattoni")                 # muro di fondo
    d.b(x - 3, y + 2, z - 3, x + 3, y + 4, z - 3, "mattoni_m")
    d.p(x, y + 1, z - 2, "leggio:south")
    d.p(x - 1, y + 1, z - 2, "candela")
    d.p(x + 1, y + 1, z - 2, "candela")
    d.b(x - 1, y, z - 1, x + 1, y, z + 1, "tappeto")
    d.forziere(x, y + 1, z - 1, "santuario", "south")
    d.p(x, y + 4, z, "lanterna_app")
    return d


# --- 7. Grotta di cristalli ---------------------------------------------------

def disegna_grotta(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.extra["fondo"]                      # `y` = fondo della grotta
    sup = s.y
    rng = _rng(s)
    d.ell(x, y + 5, z, 15, 9, 15, "calcite")                   # il guscio, poi lo scavo
    d.ell(x, y + 5, z, 14, 8, 14, "aria")
    d.b(x - 10, y - 4, z - 10, x + 10, y - 3, z + 10, "tufo", True)
    d.ell(x, y, z, 8, 1.5, 8, "acqua")                         # il lago
    # ametista sulle pareti: punti presi sul guscio dell'ellissoide
    for _ in range(70):
        th = float(rng.random() * 2 * np.pi)
        ph = float(rng.random() * np.pi * 0.85)
        px = int(round(x + 15 * np.sin(ph) * np.cos(th)))
        pz = int(round(z + 15 * np.sin(ph) * np.sin(th)))
        py = int(round(y + 5 - 9 * np.cos(ph)))
        d.p(px, py, pz, "ametista_madre" if rng.random() < 0.3 else "ametista")
    d.b(x - 1, y + 1, z - 1, x + 1, y + 1, z + 1, "ametista")
    d.forziere(x, y + 2, z, "grotta", "south")
    # il pozzo d'accesso dalla superficie, con la scala a pioli
    d.b(x + 11, y + 4, z - 1, x + 12, sup + 2, z + 1, "aria")
    d.b(x + 13, y + 4, z - 1, x + 13, sup, z + 1, "ciottoli")
    d.b(x + 12, y + 4, z, x + 12, sup, z, "pioli:west")
    d.creature.append(_combattente(x - 3, y + 2, z + 3, "ragno_grotta", s.seme + 1))
    return d


# --- 8. Capanno di pescatori ---------------------------------------------------

def disegna_pesca(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.y
    _spazio(d, s, 5, 7, 4)
    d.b(x - 3, y, z - 2, x + 3, y, z + 2, "assi")
    d.b(x - 3, y + 1, z - 2, x + 3, y + 3, z + 2, "assi_ab")
    d.b(x - 2, y + 1, z - 1, x + 2, y + 3, z + 1, "aria")
    d.b(x - 4, y + 4, z - 3, x + 4, y + 4, z + 3, "lastra")
    d.b(x - 3, y + 4, z - 2, x + 3, y + 4, z + 2, "assi")
    d.b(x + 3, y + 1, z, x + 3, y + 2, z, "aria")
    d.b(x - 3, y + 2, z - 2, x - 3, y + 2, z - 2, "vetro")
    d.forziere(x - 2, y + 1, z, "pesca", "east")
    d.p(x - 2, y + 1, z + 1, "barile")
    d.p(x - 2, y + 1, z - 1, "barile")
    d.p(x, y + 3, z, "lanterna_app")
    d.b(x + 4, y, z - 1, x + 9, y, z + 1, "assi")                              # il molo
    for t in (4, 8):
        d.b(x + t, y - 3, z - 1, x + t, y - 1, z - 1, "tronco", True)
        d.b(x + t, y - 3, z + 1, x + t, y - 1, z + 1, "tronco", True)
    d.p(x + 9, y + 1, z - 1, "recinto")
    d.p(x + 9, y + 1, z + 1, "recinto")
    d.p(x + 9, y + 2, z - 1, "lanterna")
    return d


# --- 9. Radura di funghi giganti ------------------------------------------------

def disegna_radura(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.y
    rng = _rng(s)
    d.b(x - 8, y - 1, z - 8, x + 8, y - 1, z + 8, "muschio")
    d.b(x - 8, y, z - 8, x + 8, y + 12, z + 8, "aria")
    siti = [(0, 0)] + [(int(rng.integers(-6, 7)), int(rng.integers(-6, 7))) for _ in range(5)]
    for k, (dx, dz) in enumerate(siti):
        alto = 9 if k == 0 else int(rng.integers(4, 8))
        cappa = "fungo_rosso" if k % 2 == 0 else "fungo_bruno"
        d.b(x + dx, y, z + dz, x + dx, y + alto, z + dz, "fungo_gambo")
        r = 3 if k == 0 else 2
        d.b(x + dx - r, y + alto, z + dz - r, x + dx + r, y + alto, z + dz + r, cappa)
        d.b(x + dx - r + 1, y + alto + 1, z + dz - r + 1, x + dx + r - 1, y + alto + 1, z + dz + r - 1, cappa)
    d.forziere(x + 1, y, z + 1, "radura", "south")
    return d


# --- 12. Stazione ferroviaria con locomotiva a vapore --------------------------

def disegna_stazione(s: Sito) -> Disegno:
    """Il binario corre lungo `extra["asse"]` ("x" o "z") per il centro del sito. Dal lato -v
    (a nord / a ovest) c'e' la banchina con la sala d'attesa; dal lato +v un binario morto con la
    locomotiva, il tender e due carrelli. Coordinate locali: u lungo il binario, v di traverso."""
    d = Disegno()
    d.carrelli = []
    asse = s.extra.get("asse", "x")
    x, z, y = s.x, s.z, s.y            # y = il binario; la superficie della strada sta in y - 1

    def P(u, v):
        return (x + u, z + v) if asse == "x" else (x + v, z + u)

    def box(u0, v0, h0, u1, v1, h1, k, vuoto=False):
        (xa, za), (xb, zb) = P(u0, v0), P(u1, v1)
        d.b(xa, y + h0, za, xb, y + h1, zb, k, vuoto)

    def pt(u, v, h, k):
        box(u, v, h, u, v, h, k)

    ru = "rotaia:ns" if asse == "z" else "rotaia:ew"
    dietro = "east" if asse == "x" else "south"
    # spazio libero sopra tutto il sito
    box(-12, -10, 0, 12, 8, 9, "aria")
    # --- banchina su palafitte, raso la sede del binario
    box(-9, -4, -1, 9, -2, -1, "assi_ab")
    for u in range(-8, 9, 4):
        box(u, -4, -6, u, -4, -2, "tronco_ab", True)
        box(u, -2, -6, u, -2, -2, "tronco_ab", True)
    for u in range(-9, 10):
        if abs(u) > 1:                                            # la porta della sala resta libera
            pt(u, -4, 0, "recinto_ab")
    # --- sala d'attesa (u -6..6, v -9..-5)
    box(-6, -9, -1, 6, -5, -1, "assi_ab")
    box(-6, -9, 0, 6, -5, 4, "assi_ab")
    box(-5, -8, 0, 5, -6, 3, "aria")
    box(-7, -10, 5, 7, -4, 5, "lastra_ab")
    for u in (-6, 6):
        for v in (-9, -5):
            box(u, v, 0, u, v, 4, "tronco_ab_s")
    box(-1, -5, 0, 1, -5, 2, "aria")                             # la porta verso la banchina
    for u in (-4, 4):
        pt(u, -5, 2, "vetro")
        pt(u, -9, 2, "vetro")
    pt(0, -7, 4, "lanterna_app")
    box(-4, -8, 0, -3, -8, 0, "barile")
    pt(4, -8, 0, "banco")
    # --- tettoia sopra la banchina
    box(-9, -4, 4, 9, -2, 4, "lastra_ab")
    for u in (-9, -4, 4, 9):
        box(u, -4, 1, u, -4, 3, "recinto_ab")
        pt(u, -3, 3, "lanterna_app")
    # --- binario morto con la locomotiva
    box(-11, 5, 0, 12, 5, 0, "aria")
    for u in range(-11, 13):
        pt(u, 5, 0, ru)
    box(-11, 5, -1, 12, 5, -1, "assi_ab", True)
    pt(-12, 5, 0, "tronco_ab_s")                                  # il respingente
    pt(-12, 5, 1, "lanterna")
    # locomotiva: u -9..0, v 4..6 (asse del binario v = 5)
    box(-9, 4, 1, -1, 6, 3, "carbone")                            # caldaia
    box(-9, 4, 2, -1, 6, 2, "ferro")                              # fascia di ferro
    box(-9, 5, 4, -1, 5, 4, "ferro")                              # cupola superiore
    box(-8, 5, 4, -8, 5, 7, "carbone")                            # ciminiera
    pt(-8, 5, 8, "fuoco")                                         # il fumo
    pt(-9, 5, 2, "lanterna")                                      # il faro
    for u in (-7, -5, -3, 0, 2):                                  # le ruote
        pt(u, 3, 0, "ferro")
        pt(u, 7, 0, "ferro")
    box(-10, 4, 0, -10, 6, 0, f"scala_ab:{dietro}")               # il cacciapietre
    box(0, 4, 1, 3, 6, 4, "assi_ab")                              # cabina
    box(1, 5, 1, 2, 5, 3, "aria")
    box(0, 4, 5, 3, 6, 5, "lastra_ab")
    for v in (4, 6):
        pt(1, v, 2, "vetro")
        pt(2, v, 2, "vetro")
    pt(1, 5, 3, "lanterna_app")
    box(4, 4, 1, 7, 6, 2, "carbone")                              # tender col carbone
    box(4, 5, 1, 7, 5, 1, "assi_ab")
    pt(5, 5, 3, "carbone")
    # due carrelli attaccati dietro
    for u in (10, 12):
        px, pz = P(u, 5)
        d.carrelli.append((px, y, pz))
    return d


# --- 10. Mulino a vento ---------------------------------------------------------

def disegna_mulino(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.y
    _spazio(d, s, 6, 18, 4)
    d.b(x - 3, y, z - 3, x + 3, y, z + 3, "ciottoli")
    d.b(x - 2, y + 1, z - 2, x + 2, y + 9, z + 2, "mattoni")
    d.b(x - 1, y + 1, z - 1, x + 1, y + 9, z + 1, "aria")
    d.b(x - 3, y + 10, z - 3, x + 3, y + 10, z + 3, "assi_ds")             # il tetto a gradoni
    d.b(x - 2, y + 11, z - 2, x + 2, y + 11, z + 2, "assi_ds")
    d.b(x - 1, y + 12, z - 1, x + 1, y + 12, z + 1, "assi_ds")
    d.p(x, y + 13, z, "lastra_ds")
    d.b(x - 1, y + 1, z + 1, x - 1, y + 9, z + 1, "pioli:north")
    d.b(x, y + 1, z + 2, x, y + 2, z + 2, "aria")                          # la porta, a sud
    d.b(x - 1, y + 5, z - 2, x + 1, y + 6, z - 2, "vetro")
    # le pale, sul lato nord: una croce di staccionata con la tela bianca
    cy = y + 8
    d.b(x - 7, cy, z - 3, x + 7, cy, z - 3, "recinto")
    d.b(x, cy - 7, z - 3, x, cy + 7, z - 3, "recinto")
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):                    # quattro vele
        xa, xb = sorted((x + sx * 2, x + sx * 6))
        ya, yb = sorted((cy + sy * 1, cy + sy * 2))
        d.b(xa, ya, z - 4, xb, yb, z - 4, "lana")
    # dentro: sacchi di grano e il forziere del mugnaio
    d.p(x - 1, y + 1, z - 1, "fieno")
    d.p(x + 1, y + 1, z - 1, "barile")
    d.forziere(x + 1, y + 1, z + 1, "mulino", "west")
    d.p(x, y + 8, z, "lanterna_app")
    return d


# --- 11. Museo ----------------------------------------------------------------------

def disegna_museo(s: Sito) -> Disegno:
    d = Disegno()
    x, z, y = s.x, s.z, s.y
    _spazio(d, s, 8, 8, 4)
    d.b(x - 7, y, z - 5, x + 7, y, z + 5, "arenaria_s")
    d.b(x - 7, y + 1, z - 5, x + 7, y + 5, z + 5, "arenaria")
    d.b(x - 6, y + 1, z - 4, x + 6, y + 4, z + 4, "aria")
    d.b(x - 8, y + 6, z - 6, x + 8, y + 6, z + 6, "arenaria_s")
    d.b(x - 7, y + 5, z - 5, x + 7, y + 5, z + 5, "arenaria_s")
    d.b(x - 1, y + 1, z + 5, x + 1, y + 3, z + 5, "aria")                       # ingresso a sud
    for dx in (-5, -3, 3, 5):                                                    # teche
        d.p(x + dx, y + 1, z - 3, "vetro")
        d.p(x + dx, y + 2, z - 3, "libreria")
    d.b(x - 6, y + 1, z - 4, x + 6, y + 1, z - 4, "libreria")
    d.b(x - 6, y + 2, z - 4, x + 6, y + 2, z - 4, "libreria")
    d.b(x - 2, y + 1, z, x + 2, y + 1, z, "tappeto")
    d.p(x, y + 4, z, "lanterna_app")
    d.p(x - 4, y + 4, z, "lanterna_app")
    d.p(x + 4, y + 4, z, "lanterna_app")
    d.p(x, y + 1, z - 2, "leggio:south")
    d.forziere(x - 6, y + 1, z + 3, "museo", "east")
    return d


# --------------------------------------------------------------------------
# Tipi e pianificazione
# --------------------------------------------------------------------------

@dataclass
class Contesto:
    cls: np.ndarray
    h: np.ndarray
    mare: int
    evita: np.ndarray | None
    citta: list


def _rilievo(h: np.ndarray, r: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter, minimum_filter
    return maximum_filter(h, size=2 * r + 1) - minimum_filter(h, size=2 * r + 1)


def _bordo(m: np.ndarray, r: int) -> np.ndarray:
    m = m.copy()
    m[:r + 2, :] = False
    m[-(r + 2):, :] = False
    m[:, :r + 2] = False
    m[:, -(r + 2):] = False
    return m


def _m_strega(c: Contesto) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    acqua = np.isin(c.cls, M.ACQUA)
    vicino = distance_transform_edt(~acqua) <= 16
    ok = (c.cls == M.FORESTA) & vicino & (_rilievo(c.h, 7) <= 3) & (c.h > c.mare + 1)
    return ok


def _m_sommersa(c: Contesto) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt, maximum_filter
    marino = np.isin(c.cls, M.MARINO)
    costa = distance_transform_edt(marino)
    ok = (marino & (costa >= 30) & (maximum_filter(c.h, size=31) <= c.mare - 10)
          & (_rilievo(c.h, 12) <= 4) & (c.h >= c.mare - 30))
    return ok


def _m_segnale(c: Contesto) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    marino = np.isin(c.cls, M.MARINO)
    d = distance_transform_edt(~marino)
    terra = ~np.isin(c.cls, M.ACQUA)
    return terra & (d >= 3) & (d <= 7) & (_rilievo(c.h, 4) <= 2) & (c.h >= c.mare + 2)


def _m_piatto(classi, r, tol, quota_min=3):
    def f(c: Contesto) -> np.ndarray:
        return np.isin(c.cls, classi) & (_rilievo(c.h, r) <= tol) & (c.h >= c.mare + quota_min)
    return f


def _m_grotta(c: Contesto) -> np.ndarray:
    return np.isin(c.cls, (M.FORESTA, M.MONTAGNA, M.PRATERIA, M.PIANURA)) & (c.h >= c.mare + 14) \
        & (_rilievo(c.h, 3) <= 5)


def _m_pesca(c: Contesto) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    acqua = np.isin(c.cls, M.ACQUA)
    d = distance_transform_edt(~acqua)
    terra = ~acqua
    return terra & (d >= 3) & (d <= 6) & (_rilievo(c.h, 5) <= 2) & (c.h >= c.mare + 1)


NOMI = {
    "strega": ("Capanna di {}", NOMI_STREGHE),
    "sommersa": ("Rovina sommersa {}", ("del Tridente", "dell'Anziano", "di Conchiglia", "dei Sospiri")),
    "segnale": ("Torre di segnalazione {}", ("del Gabbiano", "della Nebbia", "dello Scoglio", "del Vento", "di Punta Alta")),
    "labirinto": ("Labirinto {}", ("dei Rovi", "di Verdemuro", "del Giardiniere", "delle Mille Svolte")),
    "arena": ("Arena di {}", None),
    "santuario": ("Santuario {}", ("della Quiete", "del Viandante", "delle Candele", "dei Pellegrini",
                                    "della Luce", "del Silenzio", "della Sorgente")),
    "grotta": ("Grotta {}", ("dei Cristalli", "dell'Ametista", "del Lago Nascosto", "delle Gemme")),
    "pesca": ("Capanno {}", ("dei Pescatori", "di Riva Calma", "della Barca Rossa", "dell'Anguilla")),
    "radura": ("Radura {}", ("dei Funghi Giganti", "delle Lucciole", "del Cappello Rosso")),
    "museo": ("Museo {}", ("delle Meraviglie", "del Viandante")),
    "stazione": ("Stazione {}", ("Centrale",)),
    "mulino": ("Mulino {}", ("del Vento", "della Piana", "di Spighe", "del Mugnaio", "Alto")),
}

TIPI: dict[str, dict] = {
    "strega":    {"mezzo": 8, "celle_per": 450_000, "maschera": _m_strega, "disegno": disegna_strega,
                  "nota": "capanna su palafitte, una strega e il suo bottino"},
    "sommersa":  {"mezzo": 9, "celle_per": 380_000, "maschera": _m_sommersa, "disegno": disegna_sommersa,
                  "nota": "tempio sul fondo del mare: guardiani e un Guardiano Anziano"},
    "segnale":   {"mezzo": 5, "celle_per": 300_000, "maschera": _m_segnale, "disegno": disegna_segnale,
                  "nota": "torre sulla costa con un predone e un forziere"},
    "labirinto": {"mezzo": 10, "celle_per": 420_000,
                  "maschera": _m_piatto((M.PIANURA, M.PRATERIA), 10, 2), "disegno": disegna_labirinto,
                  "nota": "labirinto di siepi: il tesoro e' nel mezzo"},
    "arena":     {"mezzo": 14, "celle_per": 500_000,
                  "maschera": _m_piatto((M.MONTAGNA, M.FORESTA, M.PRATERIA), 14, 6, 12),
                  "disegno": disegna_arena, "nota": "un devastatore a guardia di un tesoro"},
    "santuario": {"mezzo": 5, "celle_per": 200_000,
                  "maschera": _m_piatto((M.PIANURA, M.PRATERIA, M.FORESTA, M.DESERTO), 4, 2),
                  "disegno": disegna_santuario, "nota": "un altare con un forziere di pellegrini"},
    "grotta":    {"mezzo": 16, "celle_per": 480_000, "maschera": _m_grotta, "disegno": disegna_grotta,
                  "nota": "caverna di ametista con un lago, si entra dal pozzo con la scala"},
    "pesca":     {"mezzo": 8, "celle_per": 350_000, "maschera": _m_pesca, "disegno": disegna_pesca,
                  "nota": "capanno con molo"},
    "radura":    {"mezzo": 9, "celle_per": 400_000,
                  "maschera": _m_piatto((M.FORESTA,), 8, 3), "disegno": disegna_radura,
                  "nota": "funghi giganti e un forziere nascosto"},
    "mulino":    {"mezzo": 8, "celle_per": 330_000,
                  "maschera": _m_piatto((M.PIANURA, M.PRATERIA), 6, 2), "disegno": disegna_mulino,
                  "nota": "mulino a vento, con le pale rivolte a nord"},
    "stazione":  {"mezzo": 13, "celle_per": 0, "maschera": None, "disegno": disegna_stazione,
                  "nota": "stazione di legno con la locomotiva a vapore sul binario morto"},
    "museo":     {"mezzo": 9, "celle_per": 0, "maschera": None, "disegno": disegna_museo,
                  "nota": "una sala con teche e uno scrigno; il curatore compra i reperti"},
}


def pianifica(cls: np.ndarray, h: np.ndarray, livello_mare: int = 62,
              evita: np.ndarray | None = None, citta: list | None = None,
              densita: float = 1.0, seed: int = 0) -> list[Sito]:
    H, W = cls.shape
    if densita <= 0:
        return []
    c = Contesto(cls=cls, h=h, mare=livello_mare, evita=evita, citta=citta or [])
    rng = np.random.default_rng(seed + 7070)
    fuori: list[Sito] = []
    usati: dict[str, set] = {}
    from scipy.ndimage import binary_dilation
    occ = evita.copy() if evita is not None else np.zeros((H, W), bool)
    for tipo, spec in TIPI.items():
        r = spec["mezzo"]
        if spec["maschera"] is None:
            continue
        quanti = int(max(1 if H * W >= 250_000 else 0, round(H * W / spec["celle_per"] * densita)))
        if quanti <= 0:
            continue
        ok = _bordo(spec["maschera"](c), r + 4) & ~binary_dilation(occ, iterations=r + 2)
        zz, xx = np.nonzero(ok)
        if zz.size == 0:
            continue
        presi = 0
        for k in rng.permutation(zz.size)[:6000]:
            x, z = int(xx[k]), int(zz[k])
            if any((x - s.x) ** 2 + (z - s.z) ** 2 < (r + s.mezzo + 40) ** 2 for s in fuori):
                continue
            y = int(np.median(h[z - 2:z + 3, x - 2:x + 3]))
            s = Sito(tipo=tipo, x=x, z=z, y=y, seme=int(rng.integers(0, 2 ** 31 - 1)))
            if tipo == "grotta":
                s.extra["fondo"] = max(int(livello_mare) - 18, y - 30)
            s.nome = _nome(tipo, rng, usati)
            fuori.append(s)
            occ[max(0, z - r - 1):z + r + 2, max(0, x - r - 1):x + r + 2] = True
            presi += 1
            if presi >= quanti:
                break
    # il museo sta vicino alla citta' piu' grande, a 40-70 blocchi dal centro
    if citta and densita > 0:
        s = _museo(c, rng, occ, fuori, usati)
        if s is not None:
            fuori.append(s)
    return fuori


def _nome(tipo: str, rng: np.random.Generator, usati: dict) -> str:
    fmt, lista = NOMI[tipo]
    if lista is None:
        lista = ARENE
    ok = [n for n in lista if n not in usati.setdefault(tipo, set())] or list(lista)
    n = ok[int(rng.integers(0, len(ok)))]
    usati[tipo].add(n)
    return fmt.format(n)


def _museo(c: Contesto, rng, occ, fuori, usati) -> Sito | None:
    H, W = c.cls.shape
    grandi = sorted(c.citta, key=lambda q: -getattr(q, "raggio", 0))
    for q in grandi[:3]:
        for _ in range(80):
            a = float(rng.random() * 2 * np.pi)
            d = float(rng.integers(getattr(q, "raggio", 20) + 12, getattr(q, "raggio", 20) + 40))
            x, z = int(q.x + np.cos(a) * d), int(q.z + np.sin(a) * d)
            r = TIPI["museo"]["mezzo"]
            if not (r + 8 < x < W - r - 8 and r + 8 < z < H - r - 8):
                continue
            if occ[z - r:z + r + 1, x - r:x + r + 1].any():
                continue
            if np.isin(c.cls[z - r:z + r + 1, x - r:x + r + 1], M.ACQUA).any():
                continue
            zona = c.h[z - r:z + r + 1, x - r:x + r + 1]
            if int(zona.max() - zona.min()) > 3 or int(zona.min()) < c.mare + 2:
                continue
            y = int(np.median(zona))
            s = Sito("museo", x, z, y, seme=int(rng.integers(0, 2 ** 31 - 1)))
            s.nome = _nome("museo", rng, usati)
            return s
    return None


def maschera(siti: list[Sito], forma: tuple[int, int], margine: int = 2) -> np.ndarray:
    occ = np.zeros(forma, bool)
    for s in siti:
        r = s.mezzo + margine
        occ[max(0, s.z - r):s.z + r + 1, max(0, s.x - r):s.x + r + 1] = True
    return occ


def indice_per_chunk(siti: list[Sito], passo: int = 16) -> dict:
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, s in enumerate(siti):
        x0, z0, x1, z1 = s.bbox()
        for cz in range(z0 // passo, z1 // passo + 1):
            for cx in range(x0 // passo, x1 // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


# --------------------------------------------------------------------------
# Posa
# --------------------------------------------------------------------------

def disegna(s: Sito) -> Disegno:
    return TIPI[s.tipo]["disegno"](s)


def _applica(out: np.ndarray, tav: Tavolozza, ox: int, oz: int, y0: int, meta: int, p: tuple) -> None:
    H = out.shape[1]
    if p[0] == "B":
        _, x0, ya, z0, x1, yb, z1, k, vuoto = p
        a0, a1 = max(x0 - meta, ox), min(x1 - meta, ox + 15)
        b0, b1 = max(z0 - meta, oz), min(z1 - meta, oz + 15)
        if a0 > a1 or b0 > b1:
            return
        ya, yb = max(1, ya - y0), min(H - 1, yb - y0)
        if ya > yb:
            return
        sub = out[a0 - ox:a1 - ox + 1, ya:yb + 1, b0 - oz:b1 - oz + 1]
        if vuoto:
            libero = (sub == tav.aria) | (sub == tav.acqua)
            sub[libero] = tav.id[k]
        else:
            sub[...] = tav.id[k]
        return
    _, cx, cy, cz, rx, ry, rz, k, vuoto = p
    x0, x1 = max(int(cx - rx) - meta, ox), min(int(cx + rx) - meta, ox + 15)
    z0, z1 = max(int(cz - rz) - meta, oz), min(int(cz + rz) - meta, oz + 15)
    if x0 > x1 or z0 > z1:
        return
    ya, yb = max(1, int(cy - ry) - y0), min(H - 1, int(cy + ry) - y0)
    if ya > yb:
        return
    xs = (np.arange(x0, x1 + 1) + meta - cx)[:, None, None] / rx
    ys = (np.arange(ya, yb + 1) + y0 - cy)[None, :, None] / ry
    zs = (np.arange(z0, z1 + 1) + meta - cz)[None, None, :] / rz
    dentro = xs ** 2 + ys ** 2 + zs ** 2 <= 1.0
    sub = out[x0 - ox:x1 - ox + 1, ya:yb + 1, z0 - oz:z1 - oz + 1]
    if vuoto:
        dentro = dentro & ((sub == tav.aria) | (sub == tav.acqua))
    sub[dentro] = tav.id[k]


def posa(out: np.ndarray, tav: Tavolozza, ox: int, oz: int, y0: int, siti: list[Sito],
         quali: list[int], meta: int = 0) -> None:
    """Disegna i siti nel chunk (16, H, 16). I forzieri dentro il chunk vanno in `tav.bauli`
    in coordinate di MAPPA, come per le altre strutture."""
    for i in quali:
        s = siti[i]
        d = tav.cache.get(i)
        if d is None:
            d = tav.cache[i] = disegna(s)
        for p in d.prim:
            _applica(out, tav, ox, oz, y0, meta, p)
        for (x, y, z, tipo) in d.forzieri:
            if 0 <= x - meta - ox < 16 and 0 <= z - meta - oz < 16:
                tav.bauli.append((x, y, z, tipo))


def entita_blocco(siti: list[Sito], ox: int, oz: int, meta: int) -> list:
    from .dungeon import be_spawner
    fuori = []
    for s in siti:
        for (x, y, z, mob) in disegna(s).spawner:
            if ox <= x - meta < ox + 16 and oz <= z - meta < oz + 16:
                fuori.append(be_spawner(x - meta, y, z - meta, mob))
    return fuori


def creature(siti: list[Sito], trofei: dict | None = None) -> list:
    """Le creature di tutti i siti, in coordinate di MAPPA (il motore le sposta). Il capo di un
    sito porta il trofeo di un incarico, se ce n'e' uno: `trofei` {(x, z) del sito: oggetto}."""
    fuori = []
    for s in siti:
        for c in disegna(s).creature:
            if c.capo and trofei and (s.x, s.z) in trofei:
                c.trofeo = trofei[(s.x, s.z)]
            fuori.append(c)
    return fuori


def carrelli(siti: list[Sito], meta: int) -> list:
    """I carrelli attaccati ai treni delle stazioni, in coordinate di GIOCO."""
    from .miniere import Carrello
    fuori = []
    for i, s in enumerate(siti):
        for k, (x, y, z) in enumerate(getattr(disegna(s), "carrelli", ())):
            fuori.append(Carrello(x=x - meta + 0.5, y=float(y) + 0.1, z=z - meta + 0.5,
                                  seme=s.seme + k * 31 + 5))
    return fuori


def forziere_principale(s: Sito, meta: int) -> tuple:
    """(x, y, z) di GIOCO del forziere del tesoro (il primo del disegno)."""
    x, y, z, _ = disegna(s).forzieri[0]
    return (x - meta, y, z - meta)


def ha_capo(s: Sito) -> bool:
    return any(c.capo for c in disegna(s).creature)


def libri_pellegrini(siti: list[Sito], meta: int) -> dict:
    """Il SENTIERO DEI PELLEGRINI: ogni santuario ha nel forziere un libro che indica il piu'
    vicino fra quelli non ancora visitati, in coordinate di GIOCO {posizione del forziere: Libro}."""
    from .luoghi import Libro, Luogo, _passi
    sant = [s for s in siti if s.tipo == "santuario"]
    if len(sant) < 2:
        return {}
    corrente = min(sant, key=lambda s: (s.x - meta) ** 2 + (s.z - meta) ** 2)   # parte dal piu' centrale
    resto = [s for s in sant if s is not corrente]
    ordine = [corrente]
    while resto:
        prossimo = min(resto, key=lambda s: (s.x - corrente.x) ** 2 + (s.z - corrente.z) ** 2)
        resto.remove(prossimo)
        ordine.append(prossimo)
        corrente = prossimo
    fuori = {}

    def luogo(s):
        return Luogo(s.nome, "santuario", s.x - meta, s.z - meta, s.y)

    for i, s in enumerate(ordine):
        pos = (s.x - meta, s.y + 1, s.z - 1 - meta)
        titolo = f"Pellegrini, tappa {i + 1}/{len(ordine)}"
        if i + 1 < len(ordine):
            n = ordine[i + 1]
            fuori[pos] = Libro(titolo, "Un pellegrino",
                               (f"Hai trovato {s.nome}. Riposa, bevi e prendi cio' che serve.",
                                f"La strada continua: {_passi(luogo(s), luogo(n))} sorge {n.nome}."))
        else:
            fuori[pos] = Libro(titolo, "Un pellegrino",
                               (f"Questo e' {s.nome}, l'ultima tappa del sentiero.",
                                "Chi arriva fin qui ha camminato abbastanza: il resto e' tuo."))
    return fuori


def luoghi(siti: list[Sito], meta: int) -> list:
    from .luoghi import Luogo
    return [Luogo(s.nome, s.tipo, s.x - meta, s.z - meta, s.y,
                  nota=TIPI[s.tipo]["nota"]) for s in siti]
