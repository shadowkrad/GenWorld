"""Rovine con cripta (#5): una rovina in superficie, una scala che scende e, sotto
terra, una successione di stanze: anticamera, sala dello spawner con due
forzieri, tesoro finale con un forziere vero e due trappole (forzieri-trappola
sopra il TNT).

Il bottino cresce con la distanza dalle citta' (vedi `bauli.trova`, tabella
`BOTTINO_DUNGEON` e `TESORI`). Sta fra le miniere e gli avamposti: poche, lontane
da tutto, un luogo che si trova.

Tutto e' in coordinate (u, v) rispetto all'ingresso e a una direzione `d`:
`u` corre lungo `d`, `v` e' la perpendicolare. Le stanze sono scatole; ogni
scatola ha un guscio di mattoni attorno, che si posa solo dove c'e' roccia.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---- geometria (celle) ------------------------------------------------------
PROFONDITA = 14            # quanto scende la scala (anche: quote sotto la superficie)
RUINA = 4                  # mezza larghezza della rovina in superficie (9 x 9)
U_ANTICAMERA = (PROFONDITA + 1, PROFONDITA + 9)
U_CORR1 = (PROFONDITA + 10, PROFONDITA + 22)
U_SPAWNER = (PROFONDITA + 23, PROFONDITA + 31)
U_CORR2 = (PROFONDITA + 32, PROFONDITA + 42)
U_TESORO = (PROFONDITA + 43, PROFONDITA + 51)
LUNGHEZZA = U_TESORO[1] + 2
COPERTURA = 3              # blocchi di terra minimi sopra il soffitto delle stanze
DISTANZA_FRA_RUINE = 160
RILIEVO_MASSIMO = 3
MOB = ("zombie", "skeleton", "spider")

# direzione (dx, dz) -> nome della faccia
FACCIA = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}


@dataclass
class Scatola:
    u0: int
    u1: int
    v0: int
    v1: int
    y0: int        # quota (assoluta) della prima cella libera
    y1: int        # ultima cella libera
    tipo: str


@dataclass
class Dungeon:
    x: int                      # ingresso: il centro della rovina, mappa
    z: int
    ys: int                     # quota di calpestio della rovina
    dx: int
    dz: int
    mob: str = "zombie"
    seme: int = 0
    scatole: list = field(default_factory=list)

    def punto(self, u: int, v: int) -> tuple[int, int]:
        """(x, z) di mappa per le coordinate locali (u, v)."""
        return (self.x + self.dx * u - self.dz * v, self.z + self.dz * u + self.dx * v)

    @property
    def yf(self) -> int:
        return self.ys - PROFONDITA

    def bbox(self) -> tuple[int, int, int, int]:
        """(x0, z0, x1, z1) inclusivi, con un margine di 2 celle."""
        punti = []
        for s in self.scatole:
            for u in (s.u0 - 2, s.u1 + 2):
                for v in (s.v0 - 2, s.v1 + 2):
                    punti.append(self.punto(u, v))
        for u in (-RUINA - 1, RUINA + 1):
            for v in (-RUINA - 1, RUINA + 1):
                punti.append(self.punto(u, v))
        xs = [p[0] for p in punti]
        zs = [p[1] for p in punti]
        return min(xs), min(zs), max(xs), max(zs)

    def forzieri(self) -> list[tuple[int, int, int, str]]:
        """(x, y, z, tipo) in coordinate di MAPPA: 'stanza' o 'tesoro'."""
        fuori = []
        yf = self.yf
        for u, v in ((U_SPAWNER[0] + 1, -2), (U_SPAWNER[0] + 1, 2)):
            x, z = self.punto(u, v)
            fuori.append((x, yf, z, "stanza"))
        x, z = self.punto(U_TESORO[1] - 1, 0)
        fuori.append((x, yf, z, "tesoro"))
        return fuori

    def spawner(self) -> tuple[int, int, int]:
        x, z = self.punto((U_SPAWNER[0] + U_SPAWNER[1]) // 2, 0)
        return x, self.yf + 1, z


def _scatole(ys: int) -> list[Scatola]:
    yf = ys - PROFONDITA
    return [
        Scatola(1, PROFONDITA, -1, 1, 0, 0, "scala"),   # y gestito a parte (scala)
        Scatola(*U_ANTICAMERA, -4, 4, yf, yf + 4, "anticamera"),
        Scatola(*U_CORR1, -1, 1, yf, yf + 2, "corridoio"),
        Scatola(*U_SPAWNER, -4, 4, yf, yf + 4, "spawner"),
        Scatola(*U_CORR2, -1, 1, yf, yf + 2, "corridoio"),
        Scatola(*U_TESORO, -4, 4, yf, yf + 4, "tesoro"),
    ]


# --------------------------------------------------------------------------
# Pianificazione
# --------------------------------------------------------------------------

def pianifica(altezze: np.ndarray, acqua: np.ndarray, evita: np.ndarray | None = None,
              densita: float = 1.0, seed: int = 0, celle_per: int = 160_000) -> list[Dungeon]:
    """Sceglie i siti. Una rovina serve un pianoro (rilievo <= 3 su 9 x 9) senza
    acqua ne' costruzioni, e un sottosuolo asciutto e coperto lungo tutto il
    percorso."""
    H, W = altezze.shape
    rng = np.random.default_rng(seed + 303)
    quanti = int(max(0, round(H * W / celle_per * densita)))
    fuori: list[Dungeon] = []
    limite = quanti * 80 + 80
    tentativi = 0
    margine = LUNGHEZZA + 6
    while len(fuori) < quanti and tentativi < limite:
        tentativi += 1
        x = int(rng.integers(margine, W - margine))
        z = int(rng.integers(margine, H - margine))
        if any((d.x - x) ** 2 + (d.z - z) ** 2 < DISTANZA_FRA_RUINE ** 2 for d in fuori):
            continue
        quadro = altezze[z - RUINA:z + RUINA + 1, x - RUINA:x + RUINA + 1]
        if int(quadro.max() - quadro.min()) > RILIEVO_MASSIMO:
            continue
        ys = int(quadro.max())
        if ys < 66 or ys - PROFONDITA < -40:
            continue
        dx, dz = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(0, 4))]
        d = Dungeon(x=x, z=z, ys=ys, dx=dx, dz=dz, mob=MOB[int(rng.integers(0, len(MOB)))],
                    seme=int(rng.integers(0, 2 ** 31 - 1)))
        d.scatole = _scatole(ys)
        if _percorso_valido(d, altezze, acqua, evita):
            fuori.append(d)
    return fuori


def _percorso_valido(d: Dungeon, altezze, acqua, evita) -> bool:
    H, W = altezze.shape
    # la rovina in superficie
    for u in range(-RUINA - 1, RUINA + 2):
        for v in range(-RUINA - 1, RUINA + 2):
            x, z = d.punto(u, v)
            if not (0 <= x < W and 0 <= z < H) or acqua[z, x]:
                return False
            if evita is not None and evita[z, x]:
                return False
    # le stanze e la scala, con la copertura di terra sopra
    for s in d.scatole:
        for u in range(s.u0 - 1, s.u1 + 2):
            for v in range(s.v0 - 1, s.v1 + 2):
                x, z = d.punto(u, v)
                if not (0 <= x < W and 0 <= z < H) or acqua[z, x]:
                    return False
                if evita is not None and evita[z, x]:
                    return False
                if u > RUINA + 1:
                    # la scala ha solo bisogno di non sbucare dal terreno
                    tetto, cop = ((s.y1, COPERTURA) if s.tipo != "scala"
                                  else (d.ys - u + 3, 1))
                    if int(altezze[z, x]) - 1 < tetto + cop:
                        return False
    return True


def maschera(dungeons: list[Dungeon], forma: tuple[int, int], margine: int = 0) -> np.ndarray:
    """Le colonne della rovina in superficie (e un margine): lì niente alberi,
    arredi, case o animali."""
    occ = np.zeros(forma, bool)
    H, W = forma
    r = RUINA + margine
    for d in dungeons:
        for u in range(-r, r + 1):
            for v in range(-r, r + 1):
                x, z = d.punto(u, v)
                if 0 <= x < W and 0 <= z < H:
                    occ[z, x] = True
    return occ


def indice_per_chunk(dungeons: list[Dungeon], passo: int = 16) -> dict:
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(dungeons):
        x0, z0, x1, z1 = d.bbox()
        for cz in range(z0 // passo, z1 // passo + 1):
            for cx in range(x0 // passo, x1 // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def statistiche(dungeons: list[Dungeon]) -> dict:
    return {"rovine": len(dungeons),
            "mob": {m: sum(1 for d in dungeons if d.mob == m) for m in MOB
                    if any(d.mob == m for d in dungeons)}}


# --------------------------------------------------------------------------
# Posa
# --------------------------------------------------------------------------

class Tavolozza:
    """Id dei blocchi della cripta, risolti sul livello aperto."""

    def __init__(self, scrittore):
        s = scrittore
        self.aria = s.id_aria
        self.mattoni = s.blocco("stone_bricks", variant="normal")
        self.crepati = s.blocco("stone_bricks", variant="cracked")
        self.muschiosi = s.blocco("stone_bricks", variant="mossy")
        self.scolpiti = s.blocco("stone_bricks", variant="chiseled")
        self.ciottoli = s.blocco("mossy_cobblestone")
        self.ragnatela = s.blocco("cobweb")
        self.scale = {f: s.blocco("stairs", material="stone_brick", facing=f,
                                  half="bottom", shape="straight")
                      for f in FACCIA.values()}
        self.lanterna = s.blocco("soul_lantern", hanging="true")
        self.spawner = s.blocco("spawner")
        self.tnt = s.blocco("tnt", unstable="false", underwater="false")
        self.sbarre = s.blocco("bars", material="iron", north="false", south="false",
                               east="false", west="false")
        self.ossa = s.blocco("bone_block", axis="y")
        self.forziere = {f: s.blocco("chest", facing=f, connection="none", material="wood")
                         for f in FACCIA.values()}
        self.trappola = {f: s.blocco("trapped_chest", facing=f, connection="none")
                         for f in FACCIA.values()}
        # i forzieri posati, (x, y, z, tipo) in coordinate di MAPPA: serve a
        # `bauli.trova` per scegliere il bottino
        self.bauli: list[tuple[int, int, int, str]] = []


def _mattone(tav, h: int):
    r = h % 100
    return (tav.crepati if r < 18 else tav.muschiosi if r < 40 else tav.mattoni)


def _hash(a: int, b: int, c: int) -> int:
    return ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) & 0x7FFFFFFF


def posa(out: np.ndarray, tav: Tavolozza, h_c: np.ndarray, ox: int, oz: int, y0: int,
         dungeons: list[Dungeon], quali, meta: int = 0, protetto_c=None) -> None:
    """Disegna rovina e cripta nel chunk (16, H, 16). `ox, oz` sono coordinate
    di GIOCO dell'angolo del chunk, i dungeon di MAPPA: `meta` e' lo scarto."""
    H = out.shape[1]

    def metti(x: int, y: int, z: int, blocco: int, solo_pieno: bool = False) -> None:
        lx, lz, ly = x - meta - ox, z - meta - oz, y - y0
        if not (0 <= lx < 16 and 0 <= lz < 16 and 1 <= ly < H):
            return
        if solo_pieno and out[lx, ly, lz] == tav.aria:
            return
        out[lx, ly, lz] = blocco

    def nel_chunk(x: int, z: int) -> bool:
        return 0 <= x - meta - ox < 16 and 0 <= z - meta - oz < 16

    def cella(x: int, y: int, z: int) -> int:
        lx, lz, ly = x - meta - ox, z - meta - oz, y - y0
        if not (0 <= lx < 16 and 0 <= lz < 16 and 0 <= ly < H):
            return -1
        return int(out[lx, ly, lz])

    for i in quali:
        d = dungeons[i]
        rng = np.random.default_rng(d.seme)
        ys, yf = d.ys, d.yf
        faccia = FACCIA[(d.dx, d.dz)]
        indietro = FACCIA[(-d.dx, -d.dz)]

        # ---- guscio di mattoni attorno a tutte le scatole e alla scala ------
        for s in d.scatole:
            for u in range(s.u0 - 1, s.u1 + 2):
                for v in range(s.v0 - 1, s.v1 + 2):
                    x, z = d.punto(u, v)
                    if s.tipo == "scala":
                        if u < 1:
                            continue
                        bassi = ys - u - 1
                        alti = ys - u + 4
                    else:
                        bassi, alti = s.y0 - 1, s.y1 + 1
                    for y in range(bassi, alti + 1):
                        metti(x, y, z, _mattone(tav, _hash(x, y, z)), solo_pieno=True)

        # ---- la rovina in superficie ------------------------------------------
        for u in range(-RUINA, RUINA + 1):
            for v in range(-RUINA, RUINA + 1):
                x, z = d.punto(u, v)
                # fondazione: dal terreno fino al piano della rovina
                lx, lz = x - meta - ox, z - meta - oz
                if 0 <= lx < 16 and 0 <= lz < 16:
                    for y in range(int(h_c[lx, lz]) - 1, ys):
                        metti(x, y, z, _mattone(tav, _hash(x, y, z)))
                if not (u >= 1 and abs(v) <= 1):
                    metti(x, ys - 1, z, _mattone(tav, _hash(x, ys, z)))
                perimetro = max(abs(u), abs(v)) == RUINA
                if perimetro:
                    angolo = abs(u) == RUINA and abs(v) == RUINA
                    r = _hash(x, 3, z) % 100
                    # lato della scala: il varco in cima a destra e sinistra
                    altezza = 4 + (r % 2) if angolo else (r % 5)
                    if abs(v) <= 1 and u == -RUINA:
                        altezza = 0                  # l'ingresso, sul lato opposto
                    for y in range(ys, ys + altezza):
                        metti(x, y, z, _mattone(tav, _hash(x, y, z)))
                elif rng.random() < 0.05:
                    metti(x, ys, z, tav.ragnatela)

        # ---- scavo: scala, stanze, corridoi ------------------------------------
        for s in d.scatole:
            for u in range(s.u0, s.u1 + 1):
                for v in range(s.v0, s.v1 + 1):
                    x, z = d.punto(u, v)
                    if s.tipo == "scala":
                        for y in range(ys - u + 1, ys - u + 4):
                            metti(x, y, z, tav.aria)
                        # `facing` di una scala e' il verso in cui si SALE: si scende
                        # verso +d, quindi la scala guarda l'ingresso (-d)
                        metti(x, ys - u, z, tav.scale[indietro])
                    else:
                        for y in range(s.y0, s.y1 + 1):
                            metti(x, y, z, tav.aria)
                        metti(x, s.y0 - 1, z, _mattone(tav, _hash(x, 7, z)))

        # ---- arredo delle stanze ---------------------------------------------
        for s in d.scatole:
            if s.tipo == "scala":
                continue
            if s.tipo in ("anticamera", "tesoro", "spawner"):
                # quattro colonne scolpite agli angoli
                for u in (s.u0 + 1, s.u1 - 1):
                    for v in (s.v0 + 1, s.v1 - 1):
                        x, z = d.punto(u, v)
                        for y in range(s.y0, s.y1 + 1):
                            metti(x, y, z, tav.scolpiti)
            # ragnatele nei soffitti e ossa a terra, a caso
            for u in range(s.u0, s.u1 + 1):
                for v in range(s.v0, s.v1 + 1):
                    x, z = d.punto(u, v)
                    r = _hash(x, 5, z) % 100
                    if r < 6 and cella(x, s.y1, z) == tav.aria:
                        metti(x, s.y1, z, tav.ragnatela)
                    elif r < 9 and s.tipo != "corridoio":
                        if cella(x, s.y0, z) == tav.aria:
                            metti(x, s.y0, z, tav.ossa)
            if s.tipo in ("anticamera", "corridoio", "tesoro"):
                # una lanterna dell'anima appesa al centro
                x, z = d.punto((s.u0 + s.u1) // 2, 0)
                metti(x, s.y1, z, tav.lanterna)

        # ---- sala dello spawner: piedistallo e due forzieri -------------------
        x, z = d.punto((U_SPAWNER[0] + U_SPAWNER[1]) // 2, 0)
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                px, pz = d.punto((U_SPAWNER[0] + U_SPAWNER[1]) // 2 + du, dv)
                metti(px, yf, pz, tav.mattoni)
        sx, sy, sz = d.spawner()
        metti(sx, sy, sz, tav.spawner)
        for (fx, fy, fz, tipo) in d.forzieri():
            if tipo == "stanza":
                metti(fx, fy, fz, tav.forziere[indietro])
                if nel_chunk(fx, fz):
                    tav.bauli.append((fx, fy, fz, tipo))

        # ---- tesoro: forziere vero, due trappole (forziere-trappola sul TNT) --------
        ux = U_TESORO[1] - 1
        fx, fz = d.punto(ux, 0)
        metti(fx, yf, fz, tav.forziere[indietro])
        if nel_chunk(fx, fz):
            tav.bauli.append((fx, yf, fz, "tesoro"))
        for v in (-2, 2):
            tx, tz = d.punto(ux, v)
            metti(tx, yf - 1, tz, tav.tnt)
            metti(tx, yf, tz, tav.trappola[indietro])
        # un cuscino di mattoni scolpiti attorno al forziere vero
        for v in (-1, 1):
            cx, cz = d.punto(ux, v)
            metti(cx, yf, cz, tav.scolpiti)


def be_spawner(gx: int, y: int, gz: int, mob: str):
    """Il `BlockEntity` (formato universale di amulet) di uno spawner di `mob`.
    Coordinate di GIOCO."""
    from amulet.api.block_entity import BlockEntity
    from amulet_nbt import (ByteTag, CompoundTag, FloatTag, IntTag, ListTag,
                            NamedTag, ShortTag, StringTag)
    entita = CompoundTag({"entity": CompoundTag({"id": StringTag(f"minecraft:{mob}")})})
    utags = CompoundTag({
        "isMovable": ByteTag(1),
        "MaxNearbyEntities": ShortTag(6),
        "RequiredPlayerRange": ShortTag(16),
        "SpawnCount": ShortTag(4),
        # formato del gioco recente (1.21.5+): `entity` dentro SpawnData e
        # `data`/`weight` in SpawnPotentials. Senza SpawnPotentials amulet
        # ne aggiunge uno di maiali nel vecchio formato, e dopo il primo
        # giro lo spawner farebbe maiali.
        "SpawnData": entita,
        "SpawnPotentials": ListTag([CompoundTag({"data": entita, "weight": IntTag(1)})]),
        "EntityIdentifier": StringTag(""),
        "MaxSpawnDelay": ShortTag(800),
        "Delay": ShortTag(20),
        "SpawnRange": ShortTag(4),
        "MinSpawnDelay": ShortTag(200),
        "DisplayEntityHeight": FloatTag(1.8),
        "DisplayEntityScale": FloatTag(1.0),
        "DisplayEntityWidth": FloatTag(0.8),
    })
    return BlockEntity("universal_minecraft", "spawner", gx, y, gz,
                       NamedTag(CompoundTag({"utags": utags})))


def be_dispenser(gx: int, y: int, gz: int, oggetto: str = "arrow", quanti: int = 64):
    """Un distributore con una pila di `oggetto` (frecce, di solito) in ogni slot
    che ne vale la pena: sparera' finche' ne ha. Coordinate di GIOCO."""
    from amulet.api.block_entity import BlockEntity
    from amulet_nbt import (ByteTag, CompoundTag, IntTag, ListTag, NamedTag,
                            StringTag)
    voci = ListTag([CompoundTag({"Slot": ByteTag(i), "id": StringTag(f"minecraft:{oggetto}"),
                                 "count": IntTag(quanti)}) for i in range(4)])
    utags = CompoundTag({"isMovable": ByteTag(1), "Items": voci, "Lock": StringTag("")})
    return BlockEntity("universal_minecraft", "dispenser", gx, y, gz,
                       NamedTag(CompoundTag({"utags": utags})))


def entita_blocco(dungeons: list[Dungeon], ox: int, oz: int, meta: int) -> list:
    """I `BlockEntity` degli spawner che cadono nel chunk di angolo (ox, oz),
    coordinate di GIOCO."""
    fuori = []
    for d in dungeons:
        x, y, z = d.spawner()
        gx, gz = x - meta, z - meta
        if not (ox <= gx < ox + 16 and oz <= gz < oz + 16):
            continue
        fuori.append(be_spawner(gx, y, gz, d.mob))
    return fuori
