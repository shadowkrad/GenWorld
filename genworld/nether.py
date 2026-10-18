"""Il Nether di Arda: una dimensione disegnata, non quella casuale del gioco.

Dove passa il portale c'e' un Nether vero e proprio: un mare di lava, cinque biomi
(deserti di netherrack, foreste cremisi e distorte, valli di sabbia delle anime,
delta di basalto), colonne che reggono il soffitto, ammassi di luminite appesi, e
due strutture che danno qualcosa da fare:

- le FORTEZZE: una grande sala di mattoni del Nether su pilastri, con quattro ponti,
  uno spawner di blaze, letti di verruca e un signore con il suo nome;
- i BASTIONI: un maschio di pietra nera a due piani con scala a pioli, tesori d'oro,
  brutali e piglin di guardia e un recinto di hoglin.

Le coordinate sono quelle di GIOCO del Nether (centrate sull'origine, come l'overworld).
Il gioco mette 8 blocchi del Nether per ogni blocco dell'overworld: un portale a
(x, z) sbuca qui a (x/8, z/8), quindi la zona scritta copre un overworld otto volte piu'
grande della mappa. Tutto cio' che sta fuori e' il Nether normale del gioco.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import rumore as RU

ALT = 256                    # altezza del Nether in gioco: 0..255 (la volta e' a 127)
LAVA = 32                    # il mare di lava arriva a questa quota
VOLTA = 127                  # la bedrock del soffitto
BIOMI = ("nether",       # nel formato universale i deserti del Nether (nether_wastes) si chiamano cosi'
          "crimson_forest", "warped_forest", "soul_sand_valley", "basalt_deltas")
PESI_BIOMI = (0.34, 0.20, 0.16, 0.16, 0.14)
QUOTA_MAX_STRUTTURE = 88
DISTANZA_FRA_STRUTTURE = 150
MARGINE = 40

NOMI_FORTEZZA = ("Fortezza di Brasa", "Rocca Ardente", "Cittadella delle Ceneri",
                 "Bastia dei Mattoni Neri", "Fortezza di Zolfo", "Torre del Rogo")
NOMI_SIGNORE = ("Vorag il Carbonizzato", "Ossidiana la Muta", "Karn Mangiafiamme",
                "Zeth Spezzaossa", "Ignia Cuornero", "Morthak l'Antico")
NOMI_BASTIONE = ("Bastione dell'Oro Rubato", "Maschio di Ossidiana", "Rocca dei Brutali",
                 "Tesoreria di Pietranera", "Fortino dei Predoni d'Oro")
NOMI_CAPO_BASTIONE = ("Grumb il Tesoriere", "Aurelio Fauce d'Oro", "Brakk Sette Asce",
                      "Piglion il Vorace", "Gorgol Mastro dei Conti")


# --------------------------------------------------------------------------
# Piano
# --------------------------------------------------------------------------

MEZZO = {"fortezza": 44, "bastione": 22}      # ingombro: ponti, torretta, recinto compresi


@dataclass
class Struttura:
    tipo: str                  # "fortezza" | "bastione"
    x: int                     # centro, coordinate di gioco
    z: int
    y: int                     # quota del pavimento
    nome: str
    signore: str = ""

    @property
    def mezzo(self) -> int:
        """Mezza larghezza dell'ingombro, ponti e recinto compresi."""
        return MEZZO[self.tipo]

    def bbox(self) -> tuple[int, int, int, int]:
        m = self.mezzo
        return self.x - m, self.z - m, self.x + m, self.z + m


@dataclass
class PianoNether:
    lato: int
    meta: int
    seed: int
    pav: np.ndarray            # (lato, lato) quota dell'ultimo blocco del pavimento, [z, x]
    sof: np.ndarray            # quota del primo blocco del soffitto
    bioma: np.ndarray          # (lato, lato) indice in BIOMI
    colonna: np.ndarray        # (lato, lato) bool: un pilastro dal pavimento al soffitto
    strutture: list[Struttura] = field(default_factory=list)
    creature: list = field(default_factory=list)


def _quantili(f: np.ndarray, pesi) -> np.ndarray:
    soglie = np.quantile(f, np.cumsum(pesi)[:-1])
    return np.searchsorted(soglie, f).astype(np.int8)


def pianifica(lato: int, seed: int = 0, strutture: float = 1.0) -> PianoNether:
    """Il terreno del Nether e dove stanno fortezze, bastioni e creature."""
    meta = lato // 2
    f1 = RU.fbm(lato, ottave=5, celle_base=5, seed=seed + 1)
    f1 = (f1 - f1.min()) / max(1e-6, float(f1.max() - f1.min()))
    # sotto la soglia si apre un lago di lava; sopra il pavimento sale piano piano
    pav = np.where(f1 < 0.32, 20 + f1 * 30.0, 34 + (f1 - 0.32) * 60.0).astype(np.int16)
    f3 = RU.fbm(lato, ottave=4, celle_base=6, seed=seed + 3)
    f3 = (f3 - f3.min()) / max(1e-6, float(f3.max() - f3.min()))
    sof = (112 + 10 * f3).astype(np.int16)
    f4 = RU.fbm(lato, ottave=3, celle_base=14, seed=seed + 4)
    colonna = f4 > np.quantile(f4, 0.965)
    colonna &= pav > LAVA            # i pilastri nel lago sarebbero sospesi nel vuoto
    bioma = _quantili(RU.fbm(lato, ottave=3, celle_base=3, seed=seed + 7), PESI_BIOMI)
    piano = PianoNether(lato=lato, meta=meta, seed=seed, pav=pav, sof=sof, bioma=bioma,
                        colonna=colonna)
    rng = np.random.default_rng(seed + 90)
    _pianifica_strutture(piano, rng, strutture)
    piano.creature = _creature(piano, rng)
    return piano


def _pianifica_strutture(p: PianoNether, rng: np.random.Generator, scala: float) -> None:
    area = (p.lato / 1000.0) ** 2
    quanti = {"fortezza": int(round(4 * area * scala)), "bastione": int(round(4 * area * scala))}
    if scala > 0:
        quanti = {k: max(1, v) if p.lato >= 400 else v for k, v in quanti.items()}
    nomi = {"fortezza": list(NOMI_FORTEZZA), "bastione": list(NOMI_BASTIONE)}
    capi = {"fortezza": list(NOMI_SIGNORE), "bastione": list(NOMI_CAPO_BASTIONE)}
    for tipo in ("fortezza", "bastione"):
        for _ in range(quanti[tipo]):
            for _tent in range(60):
                lim = p.meta - MARGINE - 40
                if lim <= 0:
                    break
                x = int(rng.integers(-lim, lim + 1))
                z = int(rng.integers(-lim, lim + 1))
                if any((x - s.x) ** 2 + (z - s.z) ** 2 < DISTANZA_FRA_STRUTTURE ** 2
                       for s in p.strutture):
                    continue
                if x * x + z * z < 60 * 60:           # il portale arriva qui: spazio libero
                    continue
                m = MEZZO[tipo]
                zi, xi = z + p.meta, x + p.meta
                zona = p.pav[zi - m:zi + m + 1, xi - m:xi + m + 1]
                y = int(min(QUOTA_MAX_STRUTTURE, max(int(zona.max()) + 3, 50)))
                nome = nomi[tipo].pop(int(rng.integers(0, len(nomi[tipo])))) if nomi[tipo] else f"{tipo} {len(p.strutture)}"
                capo = capi[tipo].pop(int(rng.integers(0, len(capi[tipo])))) if capi[tipo] else "Il Signore"
                p.strutture.append(Struttura(tipo, x, z, y, nome, capo))
                break


# --------------------------------------------------------------------------
# Creature
# --------------------------------------------------------------------------

def _creature(p: PianoNether, rng: np.random.Generator) -> list:
    """Creature sparse per bioma, a coordinate di gioco. Quelle delle strutture sono
    in `creature_strutture`."""
    from .torri import Combattente
    fuori: list = []
    passo = 44
    n = [0]

    def nuova(x, y, z, specie):
        n[0] += 1
        fuori.append(Combattente(x=x + 0.5, y=float(y), z=z + 0.5, specie=specie,
                                 seme=int(p.seed) * 7919 + 500000 + n[0]))

    for zi in range(passo // 2, p.lato - passo // 2, passo):
        for xi in range(passo // 2, p.lato - passo // 2, passo):
            if rng.random() > 0.55:
                continue
            xi2 = xi + int(rng.integers(-passo // 3, passo // 3))
            zi2 = zi + int(rng.integers(-passo // 3, passo // 3))
            if not (0 <= xi2 < p.lato and 0 <= zi2 < p.lato):
                continue
            x, z = xi2 - p.meta, zi2 - p.meta
            if any(abs(x - s.x) <= s.mezzo + 2 and abs(z - s.z) <= s.mezzo + 2 for s in p.strutture):
                continue
            if x * x + z * z < 30 * 30:
                continue                  # niente intorno al portale
            piano = int(p.pav[zi2, xi2])
            b = int(p.bioma[zi2, xi2])
            if piano <= LAVA:
                if rng.random() < 0.35:
                    nuova(x, LAVA + 25, z, "ghast")
                continue
            if p.colonna[zi2, xi2]:
                continue
            y = piano + 1
            if b == 0:
                specie = ("piglin", "piglin", "cubo_magma", "hoglin")[int(rng.integers(0, 4))]
            elif b == 1:
                specie = ("hoglin", "hoglin", "piglin")[int(rng.integers(0, 3))]
            elif b == 2:
                specie = ("enderman", "enderman", "piglin")[int(rng.integers(0, 3))]
            elif b == 3:
                specie = ("scheletro_nero", "scheletro_nero", "ghast")[int(rng.integers(0, 3))]
                if specie == "ghast":
                    y = piano + 14
            else:
                specie = ("cubo_magma", "cubo_magma", "piglin")[int(rng.integers(0, 3))]
            nuova(x, y, z, specie)
    return fuori


# --------------------------------------------------------------------------
# Blocchi
# --------------------------------------------------------------------------

BLOCCHI: dict[str, tuple[str, dict]] = {
    "aria": ("air", {}), "netherrack": ("netherrack", {}), "bedrock": ("bedrock", {}),
    "lava": ("lava", {"level": "0"}),
    "mattoni": ("nether_bricks", {}), "rossi": ("red_nether_bricks", {}),
    # nel formato universale lo steccato e il tronco del Nether sono `fence` e `log` con un materiale
    "recinto": ("fence", {"material": "nether_brick", "east": "false", "north": "false",
                          "south": "false", "west": "false"}),
    "nera": ("blackstone", {}), "lucida": ("polished_blackstone_bricks", {}),
    "dorata": ("gilded_blackstone", {}), "scolpita": ("chiseled_polished_blackstone", {}),
    "basalto": ("basalt", {"axis": "y"}), "basalto_liscio": ("smooth_basalt", {}),
    "sabbia": ("soul_sand", {}), "terra": ("soul_soil", {}),
    "nilio_c": ("crimson_nylium", {}), "nilio_d": ("warped_nylium", {}),
    "tronco_c": ("log", {"axis": "y", "material": "crimson", "stripped": "false"}),
    "tronco_d": ("log", {"axis": "y", "material": "warped", "stripped": "false"}),
    "verruca_c": ("nether_wart_block", {}), "verruca_d": ("warped_wart_block", {}),
    "fungolampo": ("shroomlight", {}), "luminite": ("glowstone", {}),
    "magma": ("magma_block", {}), "quarzo": ("nether_quartz_ore", {}),
    "oro_nether": ("nether_gold_ore", {}), "detrito": ("ancient_debris", {}),
    "osso": ("bone_block", {"axis": "y"}), "verruca": ("nether_wart", {"age": "3"}),
    "radici_c": ("crimson_roots", {}), "radici_d": ("warped_roots", {}),
    "germogli": ("nether_sprouts", {}), "blocco_oro": ("gold_block", {}),
    "spawner": ("spawner", {}),
    "scala": ("ladder", {"facing": "east", "waterlogged": "false"}),
}


class Tavolozza:
    """Gli id nel palette del livello dei blocchi del Nether."""

    def __init__(self, scrittore) -> None:
        self.id = {k: scrittore.blocco(n, **pr) for k, (n, pr) in BLOCCHI.items()}
        self.id["aria"] = scrittore.id_aria
        from .bauli import Tavolozza as TB
        self.bauli = TB(scrittore)
        # i forzieri come li cerca `bauli.trova`: facing + connection
        self.forziere = {f: scrittore.blocco("chest", facing=f, connection="none", material="wood")
                         for f in ("north", "south", "east", "west")}


# --------------------------------------------------------------------------
# Disegno di una struttura: elenco di scatole
# --------------------------------------------------------------------------

# (x0, y0, z0, x1, y1, z1, chiave, solo_vuoto)
Scatola = tuple


def _s(x0, y0, z0, x1, y1, z1, k, solo_vuoto=False):
    return (min(x0, x1), min(y0, y1), min(z0, z1), max(x0, x1), max(y0, y1), max(z0, z1), k,
            solo_vuoto)


def disegna_fortezza(s: Struttura) -> tuple[list, list, list, list]:
    """(scatole, forzieri, spawner, creature) della fortezza."""
    cx, cz, B = s.x, s.z, s.y
    sc: list = []
    sc.append(_s(cx - 9, B - 4, cz - 9, cx + 9, B - 1, cz + 9, "mattoni"))
    for px in (-8, 0, 8):                                     # pilastri sotto la sala
        for pz in (-8, 0, 8):
            sc.append(_s(cx + px - 1, 1, cz + pz - 1, cx + px + 1, B - 5, cz + pz + 1,
                         "mattoni", True))
    sc.append(_s(cx - 9, B, cz - 9, cx + 9, B + 11, cz + 9, "mattoni"))
    sc.append(_s(cx - 8, B + 1, cz - 8, cx + 8, B + 9, cz + 8, "aria"))
    # i quattro ponti: pavimento largo 5, parapetti, pilastri
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        L = 26
        if dx:
            xa, xb = cx + dx * 9, cx + dx * (9 + L)
            sc.append(_s(cx + dx * 9, B + 1, cz - 1, cx + dx * 8, B + 3, cz + 1, "aria"))
            sc.append(_s(xa, B, cz - 2, xb, B, cz + 2, "mattoni"))
            sc.append(_s(xa, B + 1, cz - 2, xb, B + 1, cz - 2, "recinto"))
            sc.append(_s(xa, B + 1, cz + 2, xb, B + 1, cz + 2, "recinto"))
            for t in (6, 14, 22):
                px = cx + dx * (9 + t)
                sc.append(_s(px - 1, 1, cz - 1, px + 1, B - 1, cz + 1, "mattoni", True))
        else:
            za, zb = cz + dz * 9, cz + dz * (9 + L)
            sc.append(_s(cx - 1, B + 1, cz + dz * 9, cx + 1, B + 3, cz + dz * 8, "aria"))
            sc.append(_s(cx - 2, B, za, cx + 2, B, zb, "mattoni"))
            sc.append(_s(cx - 2, B + 1, za, cx - 2, B + 1, zb, "recinto"))
            sc.append(_s(cx + 2, B + 1, za, cx + 2, B + 1, zb, "recinto"))
            for t in (6, 14, 22):
                pz = cz + dz * (9 + t)
                sc.append(_s(cx - 1, 1, pz - 1, cx + 1, B - 1, pz + 1, "mattoni", True))
    # le porte aperte sulla sala
    sc.append(_s(cx + 8, B + 1, cz - 1, cx + 9, B + 3, cz + 1, "aria"))
    sc.append(_s(cx - 9, B + 1, cz - 1, cx - 8, B + 3, cz + 1, "aria"))
    sc.append(_s(cx - 1, B + 1, cz + 8, cx + 1, B + 3, cz + 9, "aria"))
    sc.append(_s(cx - 1, B + 1, cz - 9, cx + 1, B + 3, cz - 8, "aria"))
    # in fondo al ponte est, una torretta con un forziere
    ex = cx + 9 + 26
    sc.append(_s(ex, B, cz - 3, ex + 6, B + 6, cz + 3, "mattoni"))
    sc.append(_s(ex + 1, B + 1, cz - 2, ex + 5, B + 5, cz + 2, "aria"))
    sc.append(_s(ex, B + 1, cz - 1, ex, B + 3, cz + 1, "aria"))
    # dentro la sala: piedistallo con lo spawner, letti di verruca, luci
    sc.append(_s(cx - 1, B + 1, cz - 1, cx + 1, B + 1, cz + 1, "rossi"))
    spawner = [(cx, B + 2, cz, "blaze")]
    sc.append(_s(cx, B + 2, cz, cx, B + 2, cz, "spawner"))
    for sx in (-6, 6):
        for sz in (-6, 6):
            sc.append(_s(cx + sx - 1, B, cz + sz - 1, cx + sx + 1, B, cz + sz + 1, "sabbia"))
            sc.append(_s(cx + sx - 1, B + 1, cz + sz - 1, cx + sx + 1, B + 1, cz + sz + 1, "verruca"))
    for sx in (-4, 4):
        for sz in (-4, 4):
            sc.append(_s(cx + sx, B + 9, cz + sz, cx + sx, B + 9, cz + sz, "luminite"))
    forzieri = [(cx - 7, B + 1, cz, "nether_fortezza", "east"),
                (cx + 7, B + 1, cz, "nether_fortezza", "west"),
                (ex + 3, B + 1, cz, "nether_tesoro", "west")]
    return sc, forzieri, spawner, _guardie_fortezza(s, ex)


def _guardie_fortezza(s: Struttura, ex: int) -> list:
    from .torri import Combattente
    cx, cz, B = s.x, s.z, s.y
    seme = 800000 + s.x * 31 + s.z
    c = [Combattente(x=cx + 0.5, y=float(B + 1), z=cz + 4.5, specie="scheletro_nero", capo=True,
                     nome=s.signore, livello=1, seme=seme)]
    for k, (sx, sz) in enumerate(((-5, -5), (5, -5), (-5, 5), (5, 5))):
        c.append(Combattente(x=cx + sx + 0.5, y=float(B + 1), z=cz + sz + 0.5,
                             specie="scheletro_nero", seme=seme + 1 + k))
    for k, (sx, sz) in enumerate(((-3, 0), (3, 0))):
        c.append(Combattente(x=cx + sx + 0.5, y=float(B + 3), z=cz + sz + 0.5,
                             specie="blaze", seme=seme + 10 + k))
    c.append(Combattente(x=ex + 3.5, y=float(B + 1), z=cz + 0.5, specie="scheletro_nero",
                         seme=seme + 20))
    return c


def disegna_bastione(s: Struttura) -> tuple[list, list, list, list]:
    cx, cz, B = s.x, s.z, s.y
    sc: list = []
    sc.append(_s(cx - 11, B - 4, cz - 11, cx + 11, B - 1, cz + 11, "nera"))
    for px in (-9, 0, 9):
        for pz in (-9, 0, 9):
            sc.append(_s(cx + px - 1, 1, cz + pz - 1, cx + px + 1, B - 5, cz + pz + 1, "nera", True))
    sc.append(_s(cx - 10, B, cz - 10, cx + 10, B + 15, cz + 10, "lucida"))
    sc.append(_s(cx - 9, B + 1, cz - 9, cx + 9, B + 14, cz + 9, "aria"))
    sc.append(_s(cx - 9, B + 8, cz - 9, cx + 9, B + 8, cz + 9, "nera"))          # il solaio
    rng = np.random.default_rng(((s.x * 73856093) ^ (s.z * 19349663)) & 0x7FFFFFFF)
    for _ in range(14):                                                           # pezzi d'oro nel pavimento
        gx = int(rng.integers(-8, 9))
        gz = int(rng.integers(-8, 9))
        gy = B if rng.random() < 0.5 else B + 8
        sc.append(_s(cx + gx, gy, cz + gz, cx + gx, gy, cz + gz, "dorata"))
    # le porte a nord e a sud, la scala a pioli lungo il muro ovest
    sc.append(_s(cx - 1, B + 1, cz + 9, cx + 1, B + 4, cz + 10, "aria"))
    sc.append(_s(cx - 1, B + 1, cz - 10, cx + 1, B + 4, cz - 9, "aria"))
    sc.append(_s(cx - 9, B + 1, cz, cx - 9, B + 9, cz, "scala"))
    # le luci: fungolampo agli angoli di ogni piano
    for sx in (-8, 8):
        for sz in (-8, 8):
            sc.append(_s(cx + sx, B + 6, cz + sz, cx + sx, B + 6, cz + sz, "fungolampo"))
            sc.append(_s(cx + sx, B + 13, cz + sz, cx + sx, B + 13, cz + sz, "fungolampo"))
    # il tesoro in cima: un blocco d'oro e due forzieri
    sc.append(_s(cx, B + 9, cz, cx, B + 9, cz, "blocco_oro"))
    forzieri = [(cx - 2, B + 9, cz, "nether_bastione", "east"),
                (cx + 2, B + 9, cz, "nether_bastione", "west"),
                (cx - 6, B + 1, cz - 6, "nether_bastione", "east"),
                (cx + 6, B + 1, cz + 6, "nether_bastione", "west")]
    # il recinto degli hoglin, fuori, a est
    rx0, rx1 = cx + 13, cx + 19
    sc.append(_s(rx0, B, cz - 3, rx1, B, cz + 3, "nera"))
    sc.append(_s(rx0, B + 1, cz - 3, rx1, B + 1, cz - 3, "recinto"))
    sc.append(_s(rx0, B + 1, cz + 3, rx1, B + 1, cz + 3, "recinto"))
    sc.append(_s(rx0, B + 1, cz - 3, rx0, B + 1, cz + 3, "recinto"))
    sc.append(_s(rx1, B + 1, cz - 3, rx1, B + 1, cz + 3, "recinto"))
    return sc, forzieri, [], _guardie_bastione(s)


def _guardie_bastione(s: Struttura) -> list:
    from .torri import Combattente
    cx, cz, B = s.x, s.z, s.y
    seme = 900000 + s.x * 31 + s.z
    c = [Combattente(x=cx + 0.5, y=float(B + 9), z=cz + 4.5, specie="piglin_bruto", capo=True,
                     nome=s.signore, livello=1, seme=seme)]
    for k, (sx, sz) in enumerate(((-5, -5), (5, -5), (-5, 5), (5, 5))):
        c.append(Combattente(x=cx + sx + 0.5, y=float(B + 1), z=cz + sz + 0.5,
                             specie="piglin_bruto", seme=seme + 1 + k))
    for k, (sx, sz) in enumerate(((-3, 3), (3, -3), (0, 7), (0, -7))):
        c.append(Combattente(x=cx + sx + 0.5, y=float(B + 9), z=cz + sz + 0.5,
                             specie="piglin", seme=seme + 10 + k))
    for k, dz in enumerate((-1, 0, 1)):
        c.append(Combattente(x=cx + 16.5, y=float(B + 1), z=cz + dz * 2 + 0.5,
                             specie="hoglin", seme=seme + 20 + k))
    return c


def disegna(s: Struttura):
    return disegna_fortezza(s) if s.tipo == "fortezza" else disegna_bastione(s)


# --------------------------------------------------------------------------
# Scrittura di un chunk
# --------------------------------------------------------------------------

def _applica(out: np.ndarray, tav: Tavolozza, ox: int, oz: int, sc: tuple) -> None:
    x0, y0, z0, x1, y1, z1, k, solo_vuoto = sc
    a0, a1 = max(x0, ox), min(x1, ox + 15)
    b0, b1 = max(z0, oz), min(z1, oz + 15)
    if a0 > a1 or b0 > b1:
        return
    ya, yb = max(0, y0), min(ALT - 1, y1)
    if ya > yb:
        return
    sub = out[a0 - ox:a1 - ox + 1, ya:yb + 1, b0 - oz:b1 - oz + 1]
    if solo_vuoto:
        libero = (sub == tav.id["aria"]) | (sub == tav.id.get("lava", -1))
        sub[libero] = tav.id[k]
    else:
        sub[...] = tav.id[k]


def blocchi_chunk(p: PianoNether, tav: Tavolozza, cx: int, cz: int,
                  id_biomi: list[int]):
    """(blocchi (16, 256, 16), biomi (16, 16), spawner [(x,y,z,mob)], forzieri {pos: tipo}).
    Coordinate di gioco; `cx, cz` sono le coordinate del chunk."""
    ox, oz = cx * 16, cz * 16
    zi, xi = oz + p.meta, ox + p.meta
    pav = p.pav[zi:zi + 16, xi:xi + 16].T.astype(np.int32)
    sof = p.sof[zi:zi + 16, xi:xi + 16].T.astype(np.int32)
    bio = p.bioma[zi:zi + 16, xi:xi + 16].T
    col = p.colonna[zi:zi + 16, xi:xi + 16].T
    I = tav.id
    rng = np.random.default_rng((p.seed * 1000003 + cx * 7919 + cz * 104729) & 0x7FFFFFFF)
    ys = np.arange(ALT, dtype=np.int32)[None, :, None]
    out = np.full((16, ALT, 16), I["aria"], np.uint32)
    suolo = ys <= pav[:, None, :]
    volta = (ys >= sof[:, None, :]) & (ys <= VOLTA)
    out[suolo | volta] = I["netherrack"]
    # i pilastri dal pavimento al soffitto
    pil = col[:, None, :] & (ys > pav[:, None, :]) & (ys < sof[:, None, :])
    out[pil] = I["basalto"]
    # il mare di lava
    lava = (ys <= LAVA) & (out == I["aria"]) & (ys >= 1)
    out[lava] = I["lava"]
    # la superficie, per bioma
    r2 = rng.random((16, 16))
    for k in range(16):
        for j in range(16):
            y = int(pav[k, j])
            if y <= LAVA or col[k, j]:
                continue
            b = int(bio[k, j])
            if b == 1:
                out[k, y, j] = I["nilio_c"]
            elif b == 2:
                out[k, y, j] = I["nilio_d"]
            elif b == 3:
                out[k, y, j] = I["sabbia"] if r2[k, j] < 0.7 else I["terra"]
                out[k, y - 1, j] = I["terra"]
            elif b == 4:
                v = r2[k, j]
                out[k, y, j] = (I["nera"] if v < 0.5 else I["basalto"] if v < 0.85
                                else I["magma"] if v < 0.93 else I["netherrack"])
                if r2[k, j] < 0.05:                       # una guglia di basalto
                    for h in range(1, int(2 + r2[k, j] * 80)):
                        if y + h < sof[k, j] - 4:
                            out[k, y + h, j] = I["basalto"]
            elif r2[k, j] > 0.93:
                out[k, y, j] = I["sabbia"]
            sopra = y + 1
            if b in (1, 2) and out[k, sopra, j] == I["aria"]:
                v = rng.random()
                if v < 0.10:
                    out[k, sopra, j] = I["radici_c"] if b == 1 else I["radici_d"]
                elif v < 0.14:
                    out[k, sopra, j] = I["germogli"]
    # funghi giganti nelle foreste
    _funghi(out, I, rng, pav, bio, sof, col)
    # minerali
    solido = out == I["netherrack"]
    r3 = rng.random((16, ALT, 16)).astype(np.float32)
    zona = (ys > 5) & (ys < 120)
    out[solido & zona & (r3 < 0.006)] = I["quarzo"]
    out[solido & zona & (r3 >= 0.006) & (r3 < 0.009)] = I["oro_nether"]
    out[solido & (ys > 8) & (ys < 22) & (r3 > 0.9997)] = I["detrito"]
    out[solido & (ys >= LAVA - 2) & (ys <= LAVA + 1) & (r3 > 0.97)] = I["magma"]
    # luminite appesa alla volta
    for _ in range(3):
        if rng.random() < 0.6:
            lx, lz = int(rng.integers(1, 15)), int(rng.integers(1, 15))
            top = int(sof[lx, lz])
            if out[lx, top, lz] == I["netherrack"]:
                for d in range(0, 4):
                    out[lx, top - d, lz] = I["luminite"]
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if out[lx + dx, top - 1, lz + dz] == I["aria"]:
                        out[lx + dx, top - 1, lz + dz] = I["luminite"]
    # il fondo e la volta di bedrock
    out[:, 0, :] = I["bedrock"]
    for y, pr in ((1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2)):
        m = (rng.random((16, 16)) < pr) & (out[:, y, :] == I["netherrack"])
        out[:, y, :][m] = I["bedrock"]
    out[:, VOLTA, :] = I["bedrock"]
    for y, pr in ((126, 0.8), (125, 0.6), (124, 0.4), (123, 0.2)):
        m = (rng.random((16, 16)) < pr) & (out[:, y, :] == I["netherrack"])
        out[:, y, :][m] = I["bedrock"]
    # le strutture
    spawner, forzieri = [], {}
    for s in p.strutture:
        x0, z0, x1, z1 = s.bbox()
        if x1 < ox or x0 > ox + 15 or z1 < oz or z0 > oz + 15:
            continue
        sc, fo, spw, _ = disegna(s)
        for scatola in sc:
            _applica(out, tav, ox, oz, scatola)
        for (fx, fy, fz, tipo, faccia) in fo:
            if ox <= fx < ox + 16 and oz <= fz < oz + 16:
                out[fx - ox, fy, fz - oz] = tav.forziere[faccia]
                forzieri[(fx, fy, fz)] = tipo
        for (sx, sy, sz, mob) in spw:
            if ox <= sx < ox + 16 and oz <= sz < oz + 16:
                spawner.append((sx, sy, sz, mob))
    biomi = np.array(id_biomi, np.uint32)[bio.astype(np.int64)]
    return out, biomi, spawner, forzieri


def _funghi(out, I, rng, pav, bio, sof, col) -> None:
    for _ in range(2):
        if rng.random() > 0.6:
            continue
        lx, lz = int(rng.integers(3, 13)), int(rng.integers(3, 13))
        b = int(bio[lx, lz])
        if b not in (1, 2) or col[lx, lz]:
            continue
        y = int(pav[lx, lz])
        if y <= LAVA or out[lx, y, lz] not in (I["nilio_c"], I["nilio_d"]):
            continue
        alto = int(rng.integers(6, 11))
        if y + alto + 3 >= sof[lx, lz] - 2:
            continue
        tronco, cappa = (I["tronco_c"], I["verruca_c"]) if b == 1 else (I["tronco_d"], I["verruca_d"])
        for h in range(1, alto + 1):
            out[lx, y + h, lz] = tronco
        for dy, r in ((0, 2), (1, 2), (2, 1)):
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) == r and abs(dz) == r and r == 2:
                        continue
                    if out[lx + dx, y + alto - 1 + dy, lz + dz] == I["aria"]:
                        out[lx + dx, y + alto - 1 + dy, lz + dz] = cappa
        out[lx, y + alto - 2, lz] = I["fungolampo"]


# --------------------------------------------------------------------------
# Elenco per l'utente
# --------------------------------------------------------------------------

def sezione_elenco(p: PianoNether) -> str:
    if not p.strutture:
        return ""
    righe = ["", "## Nether", "",
             "Il portale e' a 8:1: a (x, z) nell'overworld si arriva a (x/8, z/8) nel Nether. "
             "Per andarci subito: `/execute in minecraft:the_nether run tp @s X Y Z`.", ""]
    for s in p.strutture:
        righe.append(f"- **{s.nome}** ({s.tipo}) - `/execute in minecraft:the_nether run tp @s "
                     f"{s.x} {s.y + 3} {s.z}` - capo: {s.signore}")
    return "\n".join(righe) + "\n"
