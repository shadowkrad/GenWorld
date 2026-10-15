"""Torri isolate con un capo (#6).

Una torre di pietra a quattro piani su un pianoro, lontana da tutto: una porta
al piano terra, una scala a pioli fino in cima, feritoie, merli. Nei piani
vivono dei nemici; in cima il capo, con armatura, arma e un nome, e un forziere
con il tesoro. Piu' la torre e' lontana dalle citta', piu' il capo e' forte
(armatura di diamante, vita doppia).

Stesso schema di `dungeon.py`: coordinate locali (u, v) rispetto al centro e a
una direzione `d` (la porta sta sul lato +u), posa chunk per chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R = 4                      # mezza larghezza: 9 x 9
PIANI = 4
ALTEZZA_PIANO = 5          # solaio compreso
ALTEZZA = PIANI * ALTEZZA_PIANO
RILIEVO_MASSIMO = 3
DISTANZA_FRA_TORRI = 200
DISTANZA_TESORI = 250      # oltre questa dalle citta' il capo e' al massimo

FACCIA = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}

# nome nostro -> id di gioco
SPECIE = {"zombie": "zombie", "scheletro": "skeleton", "vindicatore": "vindicator",
          "scheletro_nero": "wither_skeleton"}
NOMI_CAPO = ("Aldric il Cupo", "Brenna Mano di Ferro", "Corvo Nero", "Dagan l'Esule",
             "Edrik Spezzaossa", "Fenna la Vedetta", "Garrick Senzaterra",
             "Hilda Cuordipietra", "Ivor il Custode", "Kael Lamarotta")
CAPO_PER_ZONA = ("vindicatore", "scheletro", "zombie", "scheletro_nero")


@dataclass
class Torre:
    x: int                      # centro, coordinate di mappa
    z: int
    ys: int                     # quota di calpestio del piano terra
    dx: int
    dz: int
    specie: str = "vindicatore"
    nome: str = "Aldric il Cupo"
    seme: int = 0

    def punto(self, u: int, v: int) -> tuple[int, int]:
        return (self.x + self.dx * u - self.dz * v, self.z + self.dz * u + self.dx * v)

    def piano(self, k: int) -> int:
        """Quota di calpestio del piano k (0 = terra)."""
        return self.ys + ALTEZZA_PIANO * k

    def bbox(self) -> tuple[int, int, int, int]:
        pts = [self.punto(u, v) for u in (-R - 2, R + 2) for v in (-R - 2, R + 2)]
        return (min(p[0] for p in pts), min(p[1] for p in pts),
                max(p[0] for p in pts), max(p[1] for p in pts))

    def forzieri(self) -> list[tuple[int, int, int, str]]:
        x1, z1 = self.punto(R - 1, R - 1)
        x2, z2 = self.punto(R - 1, -R + 1)
        return [(x1, self.piano(2), z1, "stanza"), (x2, self.piano(PIANI - 1), z2, "tesoro")]


@dataclass
class Combattente:
    """Un nemico delle torri: stessa interfaccia di `fauna.Animale`
    (`x/y/z` e `nbt_tag()`). Il capo ha un nome, armatura e piu' vita."""
    x: float
    y: float
    z: float
    specie: str = "zombie"
    capo: bool = False
    nome: str = ""
    livello: int = 0           # 0 vicino alle citta', 1 lontano
    seme: int = 0

    def nbt_tag(self):
        from amulet_nbt import (ByteTag, CompoundTag, DoubleTag, FloatTag, IntTag,
                                ListTag, StringTag)

        from .entita import _uuid
        gioco = SPECIE.get(self.specie, "zombie")
        vita = 20.0
        equip: dict = {}
        if self.capo:
            vita = 60.0 if self.livello == 0 else 100.0
            mat = "iron" if self.livello == 0 else "diamond"
            for slot, pezzo in (("head", "helmet"), ("chest", "chestplate"),
                                ("legs", "leggings"), ("feet", "boots")):
                equip[slot] = _pila(f"{mat}_{pezzo}", incantato=self.livello > 0)
            equip["mainhand"] = _pila(ARMA[self.specie][self.livello], incantato=self.livello > 0)
        elif self.specie in ("zombie", "scheletro", "vindicatore"):
            equip["mainhand"] = _pila(ARMA[self.specie][0])
            equip["head"] = _pila("chainmail_helmet")
        base = {
            "id": StringTag(f"minecraft:{gioco}"),
            "Pos": ListTag([DoubleTag(self.x), DoubleTag(self.y), DoubleTag(self.z)]),
            "Motion": ListTag([DoubleTag(0.0), DoubleTag(0.0), DoubleTag(0.0)]),
            "Rotation": ListTag([FloatTag(0.0), FloatTag(0.0)]),
            "UUID": _uuid(self.seme),
            "Health": FloatTag(vita),
            "Air": IntTag(300),
            "Fire": IntTag(-1),
            "FallDistance": FloatTag(0.0),
            "Invulnerable": ByteTag(0),
            "OnGround": ByteTag(1),
            "PersistenceRequired": ByteTag(1),
            "CanPickUpLoot": ByteTag(0),
        }
        if equip:
            base["equipment"] = CompoundTag(equip)
        if self.capo:
            base["CustomName"] = StringTag(self.nome)
            base["CustomNameVisible"] = ByteTag(1)
            base["attributes"] = ListTag([CompoundTag({
                "id": StringTag("minecraft:max_health"), "base": DoubleTag(vita)})])
            # il capo lascia cadere quasi sempre quello che porta
            base["drop_chances"] = CompoundTag({
                s: FloatTag(0.6) for s in ("head", "chest", "legs", "feet", "mainhand")})
        return CompoundTag(base)


ARMA = {
    "zombie": ("iron_sword", "diamond_sword"),
    "scheletro": ("bow", "bow"),
    "vindicatore": ("iron_axe", "diamond_axe"),
    "scheletro_nero": ("stone_sword", "diamond_sword"),
}


def _pila(nome: str, incantato: bool = False):
    from amulet_nbt import CompoundTag, IntTag, StringTag
    d = {"id": StringTag(f"minecraft:{nome}"), "count": IntTag(1)}
    if incantato:
        liv = ({"minecraft:power": 4, "minecraft:punch": 1} if nome == "bow" else
               {"minecraft:sharpness": 4} if "sword" in nome or "axe" in nome else
               {"minecraft:protection": 3, "minecraft:unbreaking": 2})
        d["components"] = CompoundTag({"minecraft:enchantments": CompoundTag(
            {k: IntTag(v) for k, v in liv.items()})})
    return CompoundTag(d)


# --------------------------------------------------------------------------
# Pianificazione
# --------------------------------------------------------------------------

def pianifica(altezze: np.ndarray, acqua: np.ndarray, evita: np.ndarray | None = None,
              densita: float = 1.0, seed: int = 0, celle_per: int = 220_000) -> list[Torre]:
    """Siti su un pianoro asciutto (rilievo <= 3 su 11 x 11), lontani da tutto."""
    H, W = altezze.shape
    rng = np.random.default_rng(seed + 404)
    quanti = int(max(0, round(H * W / celle_per * densita)))
    fuori: list[Torre] = []
    tentativi, limite = 0, quanti * 80 + 80
    while len(fuori) < quanti and tentativi < limite:
        tentativi += 1
        x = int(rng.integers(R + 8, W - R - 8))
        z = int(rng.integers(R + 8, H - R - 8))
        if any((t.x - x) ** 2 + (t.z - z) ** 2 < DISTANZA_FRA_TORRI ** 2 for t in fuori):
            continue
        r = R + 2
        quadro = altezze[z - r:z + r + 1, x - r:x + r + 1]
        if int(quadro.max() - quadro.min()) > RILIEVO_MASSIMO:
            continue
        ys = int(quadro.max())
        if ys < 64 or ys + ALTEZZA + 2 > 250:
            continue
        if acqua[z - r:z + r + 1, x - r:x + r + 1].any():
            continue
        if evita is not None and evita[z - r:z + r + 1, x - r:x + r + 1].any():
            continue
        dx, dz = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(0, 4))]
        fuori.append(Torre(x=x, z=z, ys=ys, dx=dx, dz=dz,
                           specie=CAPO_PER_ZONA[int(rng.integers(0, len(CAPO_PER_ZONA)))],
                           nome=NOMI_CAPO[int(rng.integers(0, len(NOMI_CAPO)))],
                           seme=int(rng.integers(0, 2 ** 31 - 1))))
    return fuori


def maschera(torri: list[Torre], forma: tuple[int, int], margine: int = 0) -> np.ndarray:
    occ = np.zeros(forma, bool)
    H, W = forma
    r = R + 1 + margine
    for t in torri:
        for u in range(-r, r + 1):
            for v in range(-r, r + 1):
                x, z = t.punto(u, v)
                if 0 <= x < W and 0 <= z < H:
                    occ[z, x] = True
    return occ


def indice_per_chunk(torri: list[Torre], passo: int = 16) -> dict:
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, t in enumerate(torri):
        x0, z0, x1, z1 = t.bbox()
        for cz in range(z0 // passo, z1 // passo + 1):
            for cx in range(x0 // passo, x1 // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def combattenti(torri: list[Torre], centri: list | None = None, seed: int = 0) -> list[Combattente]:
    """Il capo in cima, un nemico al secondo e uno al terzo piano. `centri` sono
    le citta' (mappa): oltre `DISTANZA_TESORI` il capo e' di diamante."""
    fuori: list[Combattente] = []
    for i, t in enumerate(torri):
        lontano = bool(centri) and min((t.x - cx) ** 2 + (t.z - cz) ** 2
                                       for cx, cz in centri) > DISTANZA_TESORI ** 2
        fuori.append(Combattente(x=t.x + 0.5, y=float(t.piano(PIANI - 1)), z=t.z + 0.5,
                                 specie=t.specie, capo=True, nome=t.nome,
                                 livello=1 if lontano else 0, seme=t.seme))
        rng = np.random.default_rng(t.seme + seed)
        for k in (1, 2):
            sp = "scheletro" if rng.random() < 0.4 else "zombie"
            ux, vz = t.punto(int(rng.integers(-2, 3)), int(rng.integers(-2, 3)))
            fuori.append(Combattente(x=ux + 0.5, y=float(t.piano(k)), z=vz + 0.5,
                                     specie=sp, seme=t.seme + 11 * (k + 1) + i))
    return fuori


def statistiche(torri: list[Torre]) -> dict:
    return {"torri": len(torri)}


# --------------------------------------------------------------------------
# Posa
# --------------------------------------------------------------------------

class Tavolozza:
    def __init__(self, scrittore):
        s = scrittore
        self.aria = s.id_aria
        self.mattoni = s.blocco("stone_bricks", variant="normal")
        self.crepati = s.blocco("stone_bricks", variant="cracked")
        self.muschiosi = s.blocco("stone_bricks", variant="mossy")
        self.assi = s.blocco("planks", material="oak")
        self.ciottoli = s.blocco("cobblestone")
        self.lanterna = s.blocco("lantern", material="iron", hanging="true")
        self.sbarre = s.blocco("bars", material="iron", north="false", south="false",
                               east="false", west="false")
        self.scala = {f: s.blocco("ladder", facing=f) for f in FACCIA.values()}
        self.forziere = {f: s.blocco("chest", facing=f, connection="none", material="wood")
                         for f in FACCIA.values()}
        self.bauli: list[tuple[int, int, int, str]] = []


def _mattone(tav, h: int):
    r = h % 100
    return tav.crepati if r < 14 else tav.muschiosi if r < 30 else tav.mattoni


def _hash(a: int, b: int, c: int) -> int:
    return ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) & 0x7FFFFFFF


def posa(out: np.ndarray, tav: Tavolozza, h_c: np.ndarray, ox: int, oz: int, y0: int,
         torri: list[Torre], quali, meta: int = 0) -> None:
    """Disegna le torri nel chunk (16, H, 16). `ox, oz` sono di GIOCO, le torri
    di MAPPA (`meta` e' lo scarto)."""
    H = out.shape[1]

    def metti(x: int, y: int, z: int, blocco: int) -> None:
        lx, lz, ly = x - meta - ox, z - meta - oz, y - y0
        if 0 <= lx < 16 and 0 <= lz < 16 and 1 <= ly < H:
            out[lx, ly, lz] = blocco

    def nel_chunk(x: int, z: int) -> bool:
        return 0 <= x - meta - ox < 16 and 0 <= z - meta - oz < 16

    for i in quali:
        t = torri[i]
        ys = t.ys
        faccia = FACCIA[(t.dx, t.dz)]          # verso la porta
        verso_porta = faccia
        indietro = FACCIA[(-t.dx, -t.dz)]
        cima = t.piano(PIANI)                   # quota del tetto

        for u in range(-R, R + 1):
            for v in range(-R, R + 1):
                x, z = t.punto(u, v)
                bordo = max(abs(u), abs(v)) == R
                lx, lz = x - meta - ox, z - meta - oz
                # fondazione fino al terreno
                if 0 <= lx < 16 and 0 <= lz < 16:
                    for y in range(int(h_c[lx, lz]) - 1, ys):
                        metti(x, y, z, _mattone(tav, _hash(x, y, z)))
                # solai: terra in mattoni, piani in assi, tetto in mattoni
                for k in range(PIANI + 1):
                    y = ys - 1 + ALTEZZA_PIANO * k
                    if k == 0:
                        blocco = tav.mattoni
                    elif bordo or k == PIANI:
                        blocco = _mattone(tav, _hash(x, y, z))
                    else:
                        blocco = tav.assi
                    metti(x, y, z, blocco)
                # aria dentro, muri sul bordo
                for k in range(PIANI):
                    base = ys + ALTEZZA_PIANO * k
                    for y in range(base, base + ALTEZZA_PIANO - 1):
                        metti(x, y, z, _mattone(tav, _hash(x, y, z)) if bordo else tav.aria)
                if bordo:
                    for y in range(cima, cima + 2):          # parapetto
                        if (u + v) % 2 == 0 or y == cima:
                            metti(x, y, z, _mattone(tav, _hash(x, y, z)))
                    # il merlo: solo i pieni alternati sopra il parapetto
                    if (u + v) % 2 != 0:
                        metti(x, cima + 1, z, tav.aria)

        # porta: un varco 1 x 2 sul lato +u, piano terra
        px, pz = t.punto(R, 0)
        metti(px, ys, pz, tav.aria)
        metti(px, ys + 1, pz, tav.aria)
        # feritoie ai piani 1..3, una per lato
        for k in range(1, PIANI):
            for u, v in ((R, 0), (-R, 0), (0, R), (0, -R)):
                fx, fz = t.punto(u, v)
                metti(fx, ys + ALTEZZA_PIANO * k + 1, fz, tav.sbarre)
        # i solai si bucano sopra la scala, poi vi si posano i pioli
        sx, sz = t.punto(-R + 1, 0)
        for k in range(1, PIANI + 1):
            metti(sx, ys - 1 + ALTEZZA_PIANO * k, sz, tav.aria)
        for y in range(ys, cima):
            metti(sx, y, sz, tav.scala[verso_porta])
        # una lanterna appesa al centro di ogni piano
        for k in range(PIANI):
            lx_, lz_ = t.punto(0, 0)
            metti(lx_, ys + ALTEZZA_PIANO * k + ALTEZZA_PIANO - 2, lz_, tav.lanterna)
        # i forzieri
        for (fx, fy, fz, tipo) in t.forzieri():
            metti(fx, fy, fz, tav.forziere[indietro])
            if nel_chunk(fx, fz):
                tav.bauli.append((fx, fy, fz, tipo))
