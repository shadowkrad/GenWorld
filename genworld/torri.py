"""Torri isolate con un capo (#6).

Una torre di pietra a quattro piani su un pianoro, lontana da tutto: porta con
stipiti e gradino, zoccolo a risvolto, fasce scolpite a ogni piano, feritoie,
merli e lanterne agli angoli del tetto. Dentro, una scala di pietra che gira
attorno alla sala (quattro rampe) e ogni piano allestito: sala d'armi, camerata
coi letti, armeria, sala del capo con scaffali, leggio e trono; torce, lanterne,
candele e stendardi nel colore della torre. Nei piani vivono dei nemici; in cima
il capo, con armatura, arma e un nome, e un forziere con il tesoro. Piu' la torre e' lontana dalle citta', piu' il capo e' forte
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
          "scheletro_nero": "wither_skeleton", "ragno": "spider",
          "ragno_grotta": "cave_spider", "husk": "husk", "pipistrello": "bat",
          # il predone (pillager) e' immune al sole: l'arciere ideale all'aperto
          "predone": "pillager",
          # la torre del mago: il capo e' un evocatore, le guardie streghe
          "evocatore": "evoker", "strega": "witch"}
MAGHI = ("evocatore", "strega")
# creature del Nether e dell'End (vedi `dimensioni.py`)
SPECIE.update({"piglin": "piglin", "piglin_bruto": "piglin_brute", "blaze": "blaze",
               "hoglin": "hoglin", "zombi_piglin": "zombified_piglin", "cubo_magma": "magma_cube", "ghast": "ghast",
               "enderman": "enderman", "shulker": "shulker",
               "devastatore": "ravager", "guardiano": "guardian",
               "guardiano_anziano": "elder_guardian"})
NOMI_CAPO = ("Aldric il Cupo", "Brenna Mano di Ferro", "Corvo Nero", "Dagan l'Esule",
             "Edrik Spezzaossa", "Fenna la Vedetta", "Garrick Senzaterra",
             "Hilda Cuordipietra", "Ivor il Custode", "Kael Lamarotta")
NOMI_MAGO = ("Malachia il Grigio", "Ysolda della Nebbia", "Teofrasto", "Elwyn Occhio di Gufo",
             "Morgause", "Baldassarre il Cieco", "Selene Ombraviva", "Ottavian Cenere")
CAPO_PER_ZONA = ("vindicatore", "scheletro", "zombie", "scheletro_nero", "evocatore")


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
        # u arriva a R + 5: il vialetto d'ingresso scende fino a quattro blocchi dalla porta
        pts = [self.punto(u, v) for u in (-R - 2, R + 5) for v in (-R - 2, R + 2)]
        return (min(p[0] for p in pts), min(p[1] for p in pts),
                max(p[0] for p in pts), max(p[1] for p in pts))

    def forzieri(self) -> list[tuple[int, int, int, str]]:
        # in un angolo d'angolo, lontano dalle rampe della scala (vedi `RAMPE`)
        x1, z1 = self.punto(-R + 1, R - 1)
        return [(x1, self.piano(2), z1, "stanza"), (x1, self.piano(PIANI - 1), z1, "tesoro")]


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
    # il trofeo che il capo lascia cadere (vedi `incarichi.py`): un oggetto con
    # nome, descrizione ed etichetta, nella mano libera con probabilita' 1
    trofeo: dict | None = None

    def nbt_tag(self):
        from amulet_nbt import (ByteTag, CompoundTag, DoubleTag, FloatTag, IntTag,
                                ListTag, StringTag)

        from .entita import _uuid
        gioco = SPECIE.get(self.specie, "zombie")
        vita = 20.0
        equip: dict = {}
        if self.capo:
            vita = 60.0 if self.livello == 0 else 100.0
            if self.specie in ARMA:             # un mago o un mostro senza mani non porta niente
                mat = "iron" if self.livello == 0 else "diamond"
                for slot, pezzo in (("head", "helmet"), ("chest", "chestplate"),
                                    ("legs", "leggings"), ("feet", "boots")):
                    equip[slot] = _pila(f"{mat}_{pezzo}", incantato=self.livello > 0)
                equip["mainhand"] = _pila(ARMA[self.specie][self.livello], incantato=self.livello > 0)
        elif self.specie in ("zombie", "scheletro", "vindicatore"):
            equip["mainhand"] = _pila(ARMA[self.specie][0])
            equip["head"] = _pila("chainmail_helmet")
        elif self.specie == "piglin_bruto":
            equip["mainhand"] = _pila(ARMA[self.specie][0])
        elif self.specie == "predone":
            equip["mainhand"] = _pila("crossbow")
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
        if self.trofeo:
            from .bauli import _tag, componenti_oggetto
            equip_tag = CompoundTag({
                "id": StringTag(f"minecraft:{self.trofeo['id']}"), "count": IntTag(1),
                "components": _tag({f"minecraft:{k}": v
                                    for k, v in componenti_oggetto(self.trofeo).items()})})
            base["equipment"] = CompoundTag({**dict(base.get("equipment", {})),
                                             "offhand": equip_tag})
        if self.capo:
            base["CustomName"] = StringTag(self.nome)
            base["CustomNameVisible"] = ByteTag(1)
            base["attributes"] = ListTag([CompoundTag({
                "id": StringTag("minecraft:max_health"), "base": DoubleTag(vita)})])
            # il capo lascia cadere quasi sempre quello che porta
            base["drop_chances"] = CompoundTag({
                s: FloatTag(0.6) for s in ("head", "chest", "legs", "feet", "mainhand")})
            if self.trofeo:
                base["drop_chances"]["offhand"] = FloatTag(1.0)
        return CompoundTag(base)


@dataclass
class Manichino:
    """Un `armor_stand` con un'armatura arrugginita: stessa interfaccia di
    `Combattente` (`x/y/z` e `nbt_tag()`), cosi' si scrive con gli altri abitanti."""
    x: float
    y: float
    z: float
    rotazione: float = 0.0
    seme: int = 0

    def nbt_tag(self):
        from amulet_nbt import (ByteTag, CompoundTag, DoubleTag, FloatTag, IntTag,
                                ListTag, StringTag)

        from .entita import _uuid
        equip = {"head": _pila("chainmail_helmet"), "chest": _pila("chainmail_chestplate"),
                 "mainhand": _pila("iron_sword" if self.seme % 2 else "stone_axe")}
        if self.seme % 3 == 0:
            equip["legs"] = _pila("chainmail_leggings")
        return CompoundTag({
            "id": StringTag("minecraft:armor_stand"),
            "Pos": ListTag([DoubleTag(self.x), DoubleTag(self.y), DoubleTag(self.z)]),
            "Motion": ListTag([DoubleTag(0.0), DoubleTag(0.0), DoubleTag(0.0)]),
            "Rotation": ListTag([FloatTag(self.rotazione), FloatTag(0.0)]),
            "UUID": _uuid(self.seme),
            "Air": IntTag(300),
            "Fire": IntTag(-1),
            "FallDistance": FloatTag(0.0),
            "Invulnerable": ByteTag(0),
            "OnGround": ByteTag(1),
            "ShowArms": ByteTag(1),
            "NoBasePlate": ByteTag(0),
            "equipment": CompoundTag(equip),
        })


ARMA = {
    "zombie": ("iron_sword", "diamond_sword"),
    "scheletro": ("bow", "bow"),
    "vindicatore": ("iron_axe", "diamond_axe"),
    "scheletro_nero": ("stone_sword", "diamond_sword"),
    "predone": ("crossbow", "crossbow"),
    "piglin_bruto": ("golden_axe", "diamond_axe"),
    "zombi_piglin": ("golden_sword", "diamond_sword"),
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
    nomi_liberi = list(NOMI_CAPO)
    nomi_maghi = list(NOMI_MAGO)
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
        specie = CAPO_PER_ZONA[int(rng.integers(0, len(CAPO_PER_ZONA)))]
        if len(fuori) == 2:                 # almeno una torre del mago ogni tre, se ce ne sono
            specie = "evocatore"
        nome = (_nome_unico(rng, nomi_maghi) if specie == "evocatore" and nomi_maghi
                else _nome_unico(rng, nomi_liberi))
        fuori.append(Torre(x=x, z=z, ys=ys, dx=dx, dz=dz, specie=specie, nome=nome,
                           seme=int(rng.integers(0, 2 ** 31 - 1))))
    return fuori


def _nome_unico(rng, liberi: list) -> str:
    """Un nome di capo mai usato; esauriti, ne compone uno con un numero."""
    if liberi:
        return liberi.pop(int(rng.integers(0, len(liberi))))
    return f"{NOMI_CAPO[int(rng.integers(0, len(NOMI_CAPO)))]} il {int(rng.integers(2, 9))}°"


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


def combattenti(torri: list[Torre], centri: list | None = None, seed: int = 0,
                trofei: dict | None = None) -> list[Combattente]:
    """Il capo in cima, un nemico al secondo e uno al terzo piano. `centri` sono
    le citta' (mappa): oltre `DISTANZA_TESORI` il capo e' di diamante."""
    fuori: list[Combattente] = []
    for i, t in enumerate(torri):
        lontano = bool(centri) and min((t.x - cx) ** 2 + (t.z - cz) ** 2
                                       for cx, cz in centri) > DISTANZA_TESORI ** 2
        fuori.append(Combattente(x=t.x + 0.5, y=float(t.piano(PIANI - 1)), z=t.z + 0.5,
                                 specie=t.specie, capo=True, nome=t.nome,
                                 livello=1 if lontano else 0, seme=t.seme,
                                 trofeo=(trofei or {}).get((t.x, t.z))))
        rng = np.random.default_rng(t.seme + seed)
        for k in (1, 2):
            sp = ("strega" if t.specie == "evocatore"
                  else "scheletro" if rng.random() < 0.4 else "zombie")
            ux, vz = t.punto(*CELLE_GUARDIE[k])
            fuori.append(Combattente(x=ux + 0.5, y=float(t.piano(k)), z=vz + 0.5,
                                     specie=sp, seme=t.seme + 11 * (k + 1) + i))
    return fuori


def statistiche(torri: list[Torre]) -> dict:
    return {"torri": len(torri)}


# --------------------------------------------------------------------------
# Posa
# --------------------------------------------------------------------------

STENDARDI = ("red", "blue", "green", "purple", "orange", "cyan", "yellow", "magenta")

# Le quattro rampe della scala interna, una per piano, che girano attorno alla
# sala: (cella di partenza (u, v), passo (du, dv)). Ogni rampa sale di cinque
# blocchi lungo un muro; chi esce dal gradino piu' alto e' su un angolo, da cui
# parte la rampa del piano dopo.
RAMPE = {0: ((-3, -3), (0, 1)),
         1: ((-2, 3), (1, 0)),
         2: ((3, 2), (0, -1)),
         3: ((2, -3), (-1, 0))}

# dove stanno le guardie ai piani 1 e 2 (celle libere dai mobili)
CELLE_GUARDIE = {1: (1, 1), 2: (-1, -1)}


class Tavolozza:
    def __init__(self, scrittore):
        s = scrittore
        self.aria = s.id_aria
        self.mattoni = s.blocco("stone_bricks", variant="normal")
        self.crepati = s.blocco("stone_bricks", variant="cracked")
        self.muschiosi = s.blocco("stone_bricks", variant="mossy")
        self.scolpiti = s.blocco("stone_bricks", variant="chiseled")
        self.assi = s.blocco("planks", material="oak")
        self.ciottoli = s.blocco("cobblestone")
        self.lanterna = s.blocco("lantern", material="iron", hanging="true")
        self.lanterna_su = s.blocco("lantern", material="iron", hanging="false")
        self.sbarre = s.blocco("bars", material="iron", north="false", south="false",
                               east="false", west="false")
        self.muretto = s.blocco("wall", material="stone_brick", up="true", north="none",
                                south="none", east="none", west="none")
        facce = tuple(FACCIA.values())
        self.scala_pietra = {f: s.blocco("stairs", material="stone_brick", facing=f,
                                         half="bottom", shape="straight") for f in facce}
        self.scala_legno = {f: s.blocco("stairs", material="spruce", facing=f,
                                        half="bottom", shape="straight") for f in facce}
        self.lastra_legno = s.blocco("slab", material="spruce", type="top")
        self.forziere = {f: s.blocco("chest", facing=f, connection="none", material="wood")
                         for f in facce}
        self.torcia = {f: s.blocco("torch", facing=f) for f in facce}
        self.porta = {(f, m): s.blocco("door", material="spruce", facing=f, half=m,
                                       hinge="left", open="false", powered="false")
                      for f in facce for m in ("lower", "upper")}
        self.letto = {(f, p): s.blocco("bed", color="red", facing=f, part=p, occupied="false")
                      for f in facce for p in ("head", "foot")}
        self.stendardo = {(c, f): s.blocco("wall_banner", color=c, facing=f)
                          for c in STENDARDI for f in facce}
        self.barile = s.blocco("barrel", facing="up", open="false")
        # la torre del mago: alambicchi, calderoni, un tavolo per gli incantesimi
        self.alambicco = s.blocco("brewing_stand", has_bottle_0="true", has_bottle_1="false",
                                  has_bottle_2="true")
        self.calderone = s.blocco("cauldron", cauldron_liquid="water", level="3")
        self.incantesimi = s.blocco("enchanting_table")
        self.parafulmine = s.blocco("lightning_rod", facing="up", material="copper",
                                    powered="false")
        self.banco_lavoro = s.blocco("crafting_table")
        self.fucina = s.blocco("smithing_table")
        self.fletcher = s.blocco("fletching_table")
        self.cartografo = s.blocco("cartography_table")
        self.forno = {f: s.blocco("furnace", facing=f, lit="false") for f in facce}
        self.mola = {f: s.blocco("grindstone", face="floor", facing=f) for f in facce}
        self.incudine = {f: s.blocco("anvil", facing=f, damage="normal") for f in facce}
        self.libreria = s.blocco("bookshelf")
        self.leggio = {f: s.blocco("lectern", facing=f, has_book="false", powered="false")
                       for f in facce}
        self.tappeto = s.blocco("carpet", color="red")
        self.candela = s.blocco("candle", color="default", candles="3", lit="true")
        self.bauli: list[tuple[int, int, int, str]] = []


def _mattone(tav, h: int):
    r = h % 100
    return tav.crepati if r < 14 else tav.muschiosi if r < 30 else tav.mattoni


def _hash(a: int, b: int, c: int) -> int:
    return ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) & 0x7FFFFFFF


def posa(out: np.ndarray, tav: Tavolozza, h_c: np.ndarray, ox: int, oz: int, y0: int,
         torri: list[Torre], quali, meta: int = 0) -> None:
    """Disegna le torri nel chunk (16, H, 16). `ox, oz` sono di GIOCO, le torri
    di MAPPA (`meta` e' lo scarto).

    Dentro, in coordinate locali (u, v): u va verso la porta, v e' la
    perpendicolare. La sala di ogni piano e' 7 x 7 (u, v da -3 a 3)."""
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
        cima = t.piano(PIANI)                   # quota di calpestio sul tetto
        colore = STENDARDI[t.seme % len(STENDARDI)]

        def P(u: int, v: int) -> tuple[int, int]:
            return t.punto(u, v)

        def faccia(du: int, dv: int) -> str:
            """Il nome della direzione mondiale di uno spostamento locale (du, dv)."""
            return FACCIA[(t.dx * du - t.dz * dv, t.dz * du + t.dx * dv)]

        def alza(u: int, v: int, y: int, blocco: int) -> None:
            x, z = P(u, v)
            metti(x, y, z, blocco)

        fuori = faccia(1, 0)                   # dove guarda la facciata (la porta)

        # ---- muri, solai, fondazioni, interni vuoti ----------------------------
        for u in range(-R, R + 1):
            for v in range(-R, R + 1):
                x, z = P(u, v)
                bordo = max(abs(u), abs(v)) == R
                lx, lz = x - meta - ox, z - meta - oz
                if 0 <= lx < 16 and 0 <= lz < 16:
                    for y in range(int(h_c[lx, lz]) - 1, ys):
                        metti(x, y, z, _mattone(tav, _hash(x, y, z)))
                for k in range(PIANI + 1):
                    y = ys - 1 + ALTEZZA_PIANO * k
                    if k == 0:
                        blocco = tav.mattoni
                    elif bordo:
                        # la fascia di ogni piano: un filare scolpito sul filo
                        blocco = tav.scolpiti if k < PIANI else _mattone(tav, _hash(x, y, z))
                    elif k == PIANI:
                        blocco = _mattone(tav, _hash(x, y, z))
                    else:
                        blocco = tav.assi
                    metti(x, y, z, blocco)
                for k in range(PIANI):
                    base = t.piano(k)
                    for y in range(base, base + ALTEZZA_PIANO - 1):
                        metti(x, y, z, _mattone(tav, _hash(x, y, z)) if bordo else tav.aria)
                if bordo:
                    for y in range(cima, cima + 2):          # parapetto e merli
                        if (u + v) % 2 == 0 or y == cima:
                            metti(x, y, z, _mattone(tav, _hash(x, y, z)))
                    if (u + v) % 2 != 0:
                        metti(x, cima + 1, z, tav.aria)

        # ---- zoccolo: un risvolto di gradini appoggiato al piede della torre -----
        for u in range(-R - 1, R + 2):
            for v in range(-R - 1, R + 2):
                if max(abs(u), abs(v)) != R + 1 or (abs(u) == R + 1 and abs(v) == R + 1):
                    continue
                if u == R + 1 and abs(v) <= 1:
                    continue                                   # lo spazio dei gradini d'ingresso
                x, z = P(u, v)
                lx, lz = x - meta - ox, z - meta - oz
                if 0 <= lx < 16 and 0 <= lz < 16:
                    for y in range(int(h_c[lx, lz]) - 1, ys):
                        metti(x, y, z, _mattone(tav, _hash(x, y, z)))
                # appoggiato al muro: il dorso alto contro la torre
                if abs(u) == R + 1:
                    verso = faccia(-1 if u > 0 else 1, 0)
                else:
                    verso = faccia(0, -1 if v > 0 else 1)
                metti(x, ys, z, tav.scala_pietra[verso])

        # ---- porta, gradini, stipiti, torce e stendardi all'ingresso -------------
        alza(R, 0, ys, tav.porta[(fuori, "lower")])
        alza(R, 0, ys + 1, tav.porta[(fuori, "upper")])
        for v in (-1, 1):
            for y in range(ys, ys + 3):
                alza(R, v, y, tav.scolpiti)                    # gli stipiti
            alza(R + 1, v, ys + 2, tav.torcia[fuori])
        alza(R, 0, ys + 2, tav.scolpiti)                       # l'architrave
        # il vialetto: la soglia a livello della porta (nessun gradino davanti, che
        # la tapperebbe), poi scende a gradini fino al terreno, largo tre blocchi
        for v in (-1, 0, 1):
            for i in range(4):
                x, z = P(R + 1 + i, v)
                lx, lz = x - meta - ox, z - meta - oz
                if not (0 <= lx < 16 and 0 <= lz < 16):
                    continue
                suolo = int(h_c[lx, lz])                    # dove si cammina sul terreno
                y_s = ys - i                                 # quota del piano del vialetto qui
                if i == 0 or suolo <= y_s:
                    for y in range(suolo - 1, y_s):
                        metti(x, y, z, tav.mattoni)
                    for y in range(y_s + (0 if i == 0 else 1), ys + 2):
                        metti(x, y, z, tav.aria)             # due blocchi di aria per la testa
                    if i > 0:
                        metti(x, y_s, z, tav.scala_pietra[faccia(-1, 0)])
                else:
                    for y in range(suolo, ys + 2):           # il terreno qui e' piu' alto: si cammina sopra
                        metti(x, y, z, tav.aria)
        for v in (-3, 3):
            alza(R + 1, v, ys + 3, tav.stendardo[(colore, fuori)])

        # ---- feritoie ai piani 1..3, una per lato -----------------------------------
        for k in range(1, PIANI):
            for u, v in ((R, 0), (-R, 0), (0, R), (0, -R)):
                alza(u, v, t.piano(k) + 1, tav.sbarre)
                alza(u, v, t.piano(k) + 2, tav.sbarre)

        # ---- il parafulmine di rame al centro del tetto (la torre e' la cosa piu' alta intorno)
        alza(0, 0, cima, tav.parafulmine)

        # ---- tetto: pali agli angoli con lanterne -----------------------------------
        for u, v in ((R, R), (R, -R), (-R, R), (-R, -R)):
            alza(u, v, cima + 1, tav.muretto)
            alza(u, v, cima + 2, tav.muretto)
            alza(u, v, cima + 3, tav.lanterna_su)

        # ---- la scala interna: quattro rampe che girano --------------------------------
        for k, ((u0, v0), (du, dv)) in RAMPE.items():
            y_k = t.piano(k)
            sale = faccia(du, dv)                      # `facing` di una scala e' il verso in cui si sale
            for j in range(5):
                u, v = u0 + du * j, v0 + dv * j
                # il solaio sopra si buca dove c'e' lo spazio per la testa
                # (da j=1: salendo, la testa arriva a un blocco dal solaio prima che
                # i piedi siano nella cella col buco, e il corpo e' largo 0,6)
                if j in (1, 2, 3):
                    alza(u, v, y_k + 4, tav.aria)
                alza(u, v, y_k + j, tav.scala_pietra[sale])

        # ---- una lanterna appesa al centro di ogni piano -----------------------------
        for k in range(PIANI):
            alza(0, 0, t.piano(k) + ALTEZZA_PIANO - 2, tav.lanterna)

        # ---- piano terra: sala d'armi -------------------------------------------------
        y = t.piano(0)
        for v in (-1, 0, 1):
            alza(0, v, y, tav.lastra_legno)              # il tavolo lungo...
        alza(0, 0, y + 1, tav.candela)
        for v in (-1, 1):
            alza(1, v, y, tav.scala_legno[faccia(1, 0)])   # ...con le sedie, schienale verso il fondo
            alza(-1, v, y, tav.scala_legno[faccia(-1, 0)])
        alza(3, -3, y, tav.barile)
        alza(3, 3, y, tav.barile)
        alza(3, 2, y, tav.barile)
        alza(2, 3, y, tav.banco_lavoro)
        alza(1, 3, y, tav.forno[faccia(0, -1)])
        alza(-2, 3, y, tav.incudine[faccia(0, 1)])
        for u in (1, 2, 3):
            alza(u, 0, y, tav.tappeto)
        alza(2, -3, y + 2, tav.stendardo[(colore, faccia(0, 1))])
        alza(2, 3, y + 2, tav.stendardo[(colore, faccia(0, -1))])
        alza(3, -2, y + 2, tav.torcia[faccia(-1, 0)])
        alza(3, 2, y + 2, tav.torcia[faccia(-1, 0)])

        # ---- primo piano: la camerata (o, dal mago, il laboratorio di alchimia) -------------
        y = t.piano(1)
        testa = faccia(0, -1)                              # i letti hanno la testata contro il muro
        if t.specie == "evocatore":
            for u in (-2, 0, 2):
                alza(u, -3, y, tav.libreria)
                alza(u, -3, y + 1, tav.libreria)
                alza(u, -2, y, tav.alambicco)
            alza(3, -3, y, tav.calderone)
            alza(3, -2, y, tav.calderone)
        else:
            for u in (-2, 0, 2):
                alza(u, -3, y, tav.letto[(testa, "head")])
                alza(u, -2, y, tav.letto[(testa, "foot")])
            alza(3, -3, y, tav.barile)
            alza(3, -2, y, tav.barile)
        alza(-3, -1, y + 2, tav.torcia[faccia(1, 0)])        # non davanti alle feritoie
        alza(3, 1, y + 2, tav.torcia[faccia(-1, 0)])
        alza(-1, 0, y, tav.lastra_legno)
        alza(-1, 0, y + 1, tav.candela)

        # ---- secondo piano: l'armeria (dal mago, la biblioteca degli incantesimi) ---------
        y = t.piano(2)
        if t.specie == "evocatore":
            for v in (-3, -2, -1, 1):
                alza(-3, v, y, tav.libreria)
                alza(-3, v, y + 1, tav.libreria)
            alza(-3, 0, y, tav.leggio[faccia(1, 0)])
            alza(0, -1, y, tav.incantesimi)
            alza(0, 1, y, tav.incantesimi)
        else:
            alza(-3, -3, y, tav.fletcher)
            alza(-3, -2, y, tav.fucina)
            alza(-3, -1, y, tav.mola[faccia(1, 0)])
            alza(-3, 0, y, tav.incudine[faccia(0, 1)])
            alza(-3, 1, y, tav.cartografo)
        alza(2, -3, y + 2, tav.stendardo[(colore, faccia(0, 1))])
        alza(2, 3, y + 2, tav.stendardo[(colore, faccia(0, -1))])
        alza(-2, 3, y + 2, tav.torcia[faccia(0, -1)])
        alza(-2, -3, y + 2, tav.torcia[faccia(0, 1)])

        # ---- terzo piano: la sala del capo -----------------------------------------------
        y = t.piano(3)
        for v in (-2, -1, 0, 1):
            alza(-3, v, y, tav.libreria)
            alza(-3, v, y + 1, tav.libreria)
        alza(-3, 2, y, tav.leggio[faccia(1, 0)])
        for u in range(-1, 4):
            alza(u, 0, y, tav.tappeto)
        alza(-3, 0, y + 2, tav.candela)
        alza(2, 3, y + 2, tav.stendardo[(colore, faccia(0, -1))])
        alza(3, -2, y + 2, tav.torcia[faccia(-1, 0)])
        alza(3, 2, y + 2, tav.torcia[faccia(-1, 0)])
        alza(-2, 3, y + 2, tav.torcia[faccia(0, -1)])

        # ---- i forzieri --------------------------------------------------------------------
        for (fx, fy, fz, tipo) in t.forzieri():
            metti(fx, fy, fz, tav.forziere[fuori])
            if nel_chunk(fx, fz):
                # nella torre di un mago il bottino e' di un mago (vedi `bauli.BOTTINO_MAGO`)
                tav.bauli.append((fx, fy, fz, f"mago_{tipo}" if t.specie == "evocatore" else tipo))
