"""L'End di Arda: isole sospese nel vuoto, con le cittadelle di purpur.

L'isola centrale e il drago restano del gioco (qui non si scrive niente a meno di 150
blocchi dall'origine). Intorno, a 170-460 blocchi, galleggiano sette isole di pietra
dell'End: cinque con soli cori e due con una CITTADELLA - una torre di purpur con la scala a
pioli, shulker sul tetto, endermen di guardia e forzieri con le cose buone; in una, nel
forziere in cima, un paio di ali (elytra) per tornare giu'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nether import ALT, _applica, _s

RAGGIO_MIN = 170
RAGGIO_MAX = 460
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

    def bbox(self) -> tuple[int, int, int, int]:
        r = self.raggio + 2
        return self.x - r, self.z - r, self.x + r, self.z + r


@dataclass
class PianoEnd:
    seed: int
    isole: list
    creature: list


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
                           tesoro=(k == 4)))
    p = PianoEnd(seed=seed, isole=fuori, creature=[])
    p.creature = _creature(p)
    return p


def _creature(p: PianoEnd) -> list:
    from .torri import Combattente
    fuori = []
    n = 0
    for i in p.isole:
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


def blocchi_chunk(p: PianoEnd, tav: Tavolozza, cx: int, cz: int, id_bioma: int):
    """(blocchi, biomi, forzieri) di un chunk oppure None se nessuna isola ci passa."""
    ox, oz = cx * 16, cz * 16
    vicine = [i for i in p.isole
              if i.bbox()[2] >= ox and i.bbox()[0] <= ox + 15
              and i.bbox()[3] >= oz and i.bbox()[1] <= oz + 15]
    if not vicine:
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
        top = i.y + np.where(d < 0.6, gobba, 0)
        fondo = i.y - prof
        solido = dentro[:, None, :] & (ys <= top[:, None, :]) & (ys >= fondo[:, None, :])
        out[solido] = I["pietra"]
        if i.cittadella:
            sc, fo = _scatole_cittadella(i)
            for scatola in sc:
                _applica(out, tav, ox, oz, scatola)
            for (fx, fy, fz, tipo, faccia) in fo:
                if ox <= fx < ox + 16 and oz <= fz < oz + 16:
                    out[fx - ox, fy, fz - oz] = tav.forziere[faccia]
                    forzieri[(fx, fy, fz)] = tipo
        else:
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
    biomi = np.full((16, 16), id_bioma, np.uint32)
    return out, biomi, forzieri


def sezione_elenco(p: PianoEnd) -> str:
    righe = ["", "## L'End", "",
             "Isole sospese intorno a quella del drago. Per andarci: "
             "`/execute in minecraft:the_end run tp @s X Y Z`.", ""]
    for i in p.isole:
        if i.cittadella:
            righe.append(f"- **{i.nome}** - `/execute in minecraft:the_end run tp @s {i.x} {i.y + 30} {i.z}`"
                         + (" - con le ali (elytra) nel forziere in cima" if i.tesoro else ""))
        else:
            righe.append(f"- Isola di cori - `/execute in minecraft:the_end run tp @s {i.x} {i.y + 4} {i.z}`")
    return "\n".join(righe) + "\n"
