"""Rovine con cripta (#5): una rovina in superficie, una scala che scende e, sotto
terra, una successione di stanze: anticamera, sala dello spawner con due
forzieri, tesoro finale con un forziere vero e due trappole (forzieri-trappola
sopra il TNT).

Il bottino cresce con la distanza dalle citta' (vedi `bauli.trova`, tabella
`BOTTINO_DUNGEON` e `TESORI`). Sta fra le miniere e gli avamposti: poche, lontane
da tutto, un luogo che si trova.

Non e' un corridoio dritto: dopo l'anticamera il percorso fa degli SNODI (il
corridoio si sposta di lato e la sala dello spawner e quella del tesoro stanno
fuori asse) e dai corridoi e dalle sale si diramano CUNICOLI bassi che portano a
piccole stanze laterali, ognuna con un forziere. Tutto cio' e' deciso dal seme.

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

# Compagni occasionali di uno spawner: il mob principale pesa 4, ognuno dei
# compagni 1, quindi su cinque comparse una e' diversa. Cosi' una cripta di
# zombie ogni tanto lascia uscire anche uno scheletro o un ragno.
COMPAGNI = {"zombie": ("skeleton", "spider", "husk"),
            "skeleton": ("zombie", "cave_spider", "stray"),
            "spider": ("cave_spider", "skeleton", "zombie"),
            "cave_spider": ("spider", "skeleton"),
            "skeleton_nero": ("skeleton", "zombie"),
            "vindicatore": ("zombie", "skeleton")}

# Le trappole dei corridoi: dardi (un filo d'inciampo e due distributori di frecce nei
# muri), fossa (con le punte e una sponda stretta), ragnatele, pavimento di magma,
# muri infestati (pesciolini d'argento) e piastre sopra il TNT.
TIPI_TRAPPOLA = ("dardi", "fossa", "ragnatele", "magma", "infestati", "tnt")
PESI_TRAPPOLA = (0.28, 0.20, 0.14, 0.12, 0.12, 0.14)

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
class Stanza:
    """Una stanza laterale, raggiunta da un cunicolo: centro (u, v) e il lato
    (+1 / -1) su cui sta, rispetto alla sala o al corridoio da cui parte."""
    u: int
    v: int
    lato: int

    def forziere(self) -> tuple[int, int]:
        """(u, v) del forziere: contro la parete di fondo."""
        return self.u, self.v + self.lato * 2


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
    vs: int = 0                 # di quanto la sala dello spawner sta fuori asse (v)
    vt: int = 0                 # idem per la sala del tesoro
    laterali: list = field(default_factory=list)      # `Stanza`

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
        for u, v in ((U_SPAWNER[0] + 1, self.vs - 2), (U_SPAWNER[0] + 1, self.vs + 2)):
            x, z = self.punto(u, v)
            fuori.append((x, yf, z, "stanza"))
        x, z = self.punto(U_TESORO[1] - 1, self.vt)
        fuori.append((x, yf, z, "tesoro"))
        # i forzieri delle stanze laterali, dopo i tre principali (gli indici dei
        # primi tre non cambiano: `motore` usa il terzo come tesoro)
        for st in self.laterali:
            x, z = self.punto(*st.forziere())
            fuori.append((x, yf, z, "stanza"))
        return fuori

    def spawner(self) -> tuple[int, int, int]:
        x, z = self.punto((U_SPAWNER[0] + U_SPAWNER[1]) // 2, self.vs)
        return x, self.yf + 1, z

    def trappole(self) -> list[tuple[str, int, int]]:
        """(tipo, u, v del centro): una trappola in quasi ogni tratto dritto e lungo
        di corridoio. Sei tipi (vedi `TIPI_TRAPPOLA`), scelti dal seme; il TNT sotto le
        piastre al massimo uno per cripta, e ce n'e' sempre almeno una."""
        rng = np.random.default_rng(self.seme + 17)
        fuori: list[tuple[str, int, int]] = []
        tnt = False
        tratti = [s for s in self.scatole if s.tipo == "corridoio" and s.u1 - s.u0 >= 4
                  and s.v1 - s.v0 == 2]
        for k, s in enumerate(tratti):
            if rng.random() < 0.15 and not (k == len(tratti) - 1 and not fuori):
                continue
            tipo = TIPI_TRAPPOLA[int(rng.choice(len(TIPI_TRAPPOLA), p=PESI_TRAPPOLA))]
            if tipo == "tnt":
                tipo = "ragnatele" if tnt else "tnt"
                tnt = tnt or tipo == "tnt"
            fuori.append((tipo, (s.u0 + s.u1) // 2, (s.v0 + s.v1) // 2))
        return fuori

    def _faccia(self, du: int, dv: int) -> str:
        return FACCIA[(self.dx * du - self.dz * dv, self.dz * du + self.dx * dv)]

    def asse_v(self) -> str:
        """L'asse del MONDO lungo cui corre v (il filo d'inciampo sta di traverso)."""
        return "z" if self.dx else "x"

    def distributori(self) -> list[tuple[int, int, int, str]]:
        """(x, y, z, facing) dei distributori di frecce delle trappole `dardi`."""
        fuori = []
        for tipo, u, vc in self.trappole():
            if tipo != "dardi":
                continue
            for lato in (1, -1):
                x, z = self.punto(u, vc + 2 * lato)
                fuori.append((x, self.yf + 1, z, self._faccia(0, -lato)))
        return fuori


def _corridoio(u0: int, u1: int, v_da: int, v_a: int, yf: int) -> list[Scatola]:
    """Un corridoio da `v_da` a `v_a`: un tratto dritto, uno snodo che sposta di
    lato (se i due v differiscono) e un altro tratto dritto. Largo 3, alto 3."""
    if v_da == v_a:
        return [Scatola(u0, u1, v_da - 1, v_da + 1, yf, yf + 2, "corridoio")]
    um = u0 + 4
    return [
        Scatola(u0, um, v_da - 1, v_da + 1, yf, yf + 2, "corridoio"),
        Scatola(um, um + 2, min(v_da, v_a) - 1, max(v_da, v_a) + 1, yf, yf + 2, "corridoio"),
        Scatola(um + 2, u1, v_a - 1, v_a + 1, yf, yf + 2, "corridoio"),
    ]


def _scatole(ys: int, vs: int = 0, vt: int = 0, laterali: list | None = None) -> list[Scatola]:
    """Scala, anticamera, corridoio (con snodo), sala dello spawner, corridoio,
    sala del tesoro, e per ogni stanza laterale il suo cunicolo e la stanza."""
    yf = ys - PROFONDITA
    scatole = [
        Scatola(1, PROFONDITA, -1, 1, 0, 0, "scala"),   # y gestito a parte (scala)
        Scatola(*U_ANTICAMERA, -4, 4, yf, yf + 4, "anticamera"),
        *_corridoio(*U_CORR1, 0, vs, yf),
        Scatola(*U_SPAWNER, vs - 4, vs + 4, yf, yf + 4, "spawner"),
        *_corridoio(*U_CORR2, vs, vt, yf),
        Scatola(*U_TESORO, vt - 4, vt + 4, yf, yf + 4, "tesoro"),
    ]
    for st in laterali or ():
        # il cunicolo attraversa le due celle fra il bordo della sala (o del corridoio)
        # e quello della stanza (5 x 5): a 4 e 3 celle dal centro della stanza
        v0, v1 = sorted((st.v - st.lato * 4, st.v - st.lato * 3))
        scatole.append(Scatola(st.u, st.u, v0, v1, yf, yf + 1, "cunicolo"))
        scatole.append(Scatola(st.u - 2, st.u + 2, st.v - 2, st.v + 2, yf, yf + 3, "stanza_laterale"))
    return scatole


def _sovrappone(a: Scatola, b: Scatola, margine: int = 2) -> bool:
    return not (a.u1 + margine < b.u0 or b.u1 + margine < a.u0
                or a.v1 + margine < b.v0 or b.v1 + margine < a.v0)


def _stanze_laterali(rng, vs: int, vt: int, base: list[Scatola]) -> list[Stanza]:
    """Da 0 a 3 stanze laterali, appese alla sala d'ingresso, a quella dello spawner,
    a quella del tesoro o al secondo tratto del primo corridoio. Una stanza si
    scarta se tocca il resto (con un margine per il guscio di mattoni)."""
    um1 = (U_ANTICAMERA[0] + U_ANTICAMERA[1]) // 2
    um2 = (U_SPAWNER[0] + U_SPAWNER[1]) // 2
    um3 = (U_TESORO[0] + U_TESORO[1]) // 2
    um4 = U_CORR1[1] - 2
    ancore = [(um1, 0, 4), (um2, vs, 4), (um3, vt, 4), (um4, vs, 1)]
    quante = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.40, 0.30, 0.15]))
    fuori: list[Stanza] = []
    scelte = rng.permutation(len(ancore))
    for k in scelte:
        if len(fuori) >= quante:
            break
        u, v, mezza = ancore[int(k)]
        lato = 1 if rng.random() < 0.5 else -1
        # il centro della stanza: oltre il bordo della sala (mezza) piu' il cunicolo
        centro_v = v + lato * (mezza + 5)
        st = Stanza(u=u, v=centro_v, lato=lato)
        nuova = Scatola(u - 2, u + 2, centro_v - 2, centro_v + 2, 0, 0, "stanza_laterale")
        if any(_sovrappone(nuova, b) for b in base if b.tipo not in ("scala",)
               and not (b.u0 <= u <= b.u1 and abs(b.v0 - centro_v) > 0 and False)):
            # la sala di partenza non conta: il cunicolo la tocca
            if any(_sovrappone(nuova, b, 1) for b in base):
                continue
        if any(_sovrappone(nuova, Scatola(o.u - 2, o.u + 2, o.v - 2, o.v + 2, 0, 0, ""))
               for o in fuori):
            continue
        fuori.append(st)
    return fuori


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
        # il layout: sala dello spawner e del tesoro fuori asse (snodi) e stanze laterali;
        # se il terreno non lo permette si ripiega su uno piu' semplice, fino al dritto
        lr = np.random.default_rng(d.seme + 5)
        vs = int(lr.choice([-9, -7, -5, 0, 5, 7, 9]))
        vt = int(lr.choice([-9, -6, 0, 6, 9]))
        if vs == 0 and vt == 0:
            vt = int(lr.choice([-7, 7]))
        for tentativo in range(4):
            if tentativo == 0:
                d.vs, d.vt = vs, vt
            elif tentativo == 1:
                d.vs, d.vt = vs // 2 * 2, 0
            elif tentativo == 2:
                d.vs, d.vt = 0, 0
            base = _scatole(ys, d.vs, d.vt)
            d.laterali = _stanze_laterali(lr, d.vs, d.vt, base) if tentativo < 3 else []
            d.scatole = _scatole(ys, d.vs, d.vt, d.laterali)
            if _percorso_valido(d, altezze, acqua, evita):
                fuori.append(d)
                break
            if tentativo == 2:
                d.vs = d.vt = 0
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
        self.piastra = s.blocco("pressure_plate", material="stone", powered="false")
        self.gancio = {f: s.blocco("tripwire_hook", facing=f, attached="true", powered="false")
                       for f in FACCIA.values()}
        self.filo = {ass: s.blocco("tripwire", attached="true", disarmed="false", powered="false",
                                   suspended="false",
                                   east=str(ass == "x").lower(), west=str(ass == "x").lower(),
                                   north=str(ass == "z").lower(), south=str(ass == "z").lower())
                     for ass in ("x", "z")}
        self.distributore = {f: s.blocco("dispenser", facing=f, triggered="false")
                             for f in FACCIA.values()}
        self.punta = s.blocco("pointed_dripstone", thickness="tip", vertical_direction="up")
        self.magma = s.blocco("magma_block")
        self.infestati = (s.blocco("infested_block", material="stone_bricks"),
                          s.blocco("infested_block", material="cracked_stone_bricks"),
                          s.blocco("infested_block", material="mossy_stone_bricks"))
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
            if s.tipo in ("anticamera", "corridoio", "tesoro", "stanza_laterale"):
                # una lanterna dell'anima appesa al centro (nei corridoi, in cima:
                # il soffitto sta a tre blocchi, la testa a uno e otto)
                x, z = d.punto((s.u0 + s.u1) // 2, (s.v0 + s.v1) // 2)
                metti(x, s.y1, z, tav.lanterna)

        # ---- sala dello spawner: piedistallo e due forzieri -------------------
        x, z = d.punto((U_SPAWNER[0] + U_SPAWNER[1]) // 2, d.vs)
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                px, pz = d.punto((U_SPAWNER[0] + U_SPAWNER[1]) // 2 + du, d.vs + dv)
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
        fx, fz = d.punto(ux, d.vt)
        metti(fx, yf, fz, tav.forziere[indietro])
        if nel_chunk(fx, fz):
            tav.bauli.append((fx, yf, fz, "tesoro"))
        for v in (d.vt - 2, d.vt + 2):
            tx, tz = d.punto(ux, v)
            metti(tx, yf - 1, tz, tav.tnt)
            metti(tx, yf, tz, tav.trappola[indietro])
        # un cuscino di mattoni scolpiti attorno al forziere vero
        for v in (d.vt - 1, d.vt + 1):
            cx, cz = d.punto(ux, v)
            metti(cx, yf, cz, tav.scolpiti)

        # ---- le trappole dei corridoi ------------------------------------------
        def alza(u: int, v: int, y: int, blocco: int) -> None:
            x, z = d.punto(u, v)
            metti(x, y, z, blocco)

        for tipo, u, vc in d.trappole():
            if tipo == "dardi":
                alza(u, vc + 1, yf, tav.gancio[d._faccia(0, -1)])
                alza(u, vc - 1, yf, tav.gancio[d._faccia(0, 1)])
                alza(u, vc, yf, tav.filo[d.asse_v()])
                for lato in (1, -1):
                    alza(u, vc + 2 * lato, yf + 1, tav.distributore[d._faccia(0, -lato)])
            elif tipo == "fossa":
                for uu in (u - 1, u, u + 1):
                    for v in (vc, vc + 1):                     # la sponda e' vc - 1
                        for y in range(yf - 4, yf):
                            alza(uu, v, y, tav.aria)
                        alza(uu, v, yf - 5, _mattone(tav, _hash(uu, 9, v)))
                        alza(uu, v, yf - 4, tav.punta)
                    alza(uu, vc - 1, yf - 1, tav.crepati)       # la sponda, stretta e crepata
                for uu in (u - 2, u + 2):
                    for v in (vc, vc + 1):
                        alza(uu, v, yf, tav.ragnatela)
            elif tipo == "ragnatele":
                for uu in (u - 1, u, u + 1):
                    for v in (vc - 1, vc, vc + 1):
                        for y in (yf, yf + 1):
                            alza(uu, v, y, tav.ragnatela)
            elif tipo == "magma":
                for uu in (u - 1, u, u + 1):
                    for v in (vc - 1, vc, vc + 1):
                        alza(uu, v, yf - 1, tav.magma)
            elif tipo == "infestati":
                for uu in range(u - 2, u + 3):
                    for lato in (1, -1):
                        for y in (yf, yf + 1):
                            alza(uu, vc + 2 * lato, y, tav.infestati[(uu + y) % 3])
            elif tipo == "tnt":
                for uu in (u, u + 1):
                    for v in (vc - 1, vc, vc + 1):
                        alza(uu, v, yf - 1, tav.tnt)
                        alza(uu, v, yf, tav.piastra)


def be_spawner(gx: int, y: int, gz: int, mob: str, misto: bool = False):
    """Il `BlockEntity` (formato universale di amulet) di uno spawner di `mob`.
    Coordinate di GIOCO."""
    from amulet.api.block_entity import BlockEntity
    from amulet_nbt import (ByteTag, CompoundTag, FloatTag, IntTag, ListTag,
                            NamedTag, ShortTag, StringTag)
    def _ent(m):
        return CompoundTag({"entity": CompoundTag({"id": StringTag(f"minecraft:{m}")})})
    entita = _ent(mob)
    potenziali = [CompoundTag({"data": entita, "weight": IntTag(4 if misto else 1)})]
    if misto:
        potenziali += [CompoundTag({"data": _ent(m), "weight": IntTag(1)})
                       for m in COMPAGNI.get(mob, ())]
    utags = CompoundTag({
        "isMovable": ByteTag(1),
        # Meno asfissiante dei valori del gioco (4 per volta, 6 vicini, ogni 10-40 s,
        # da 16 blocchi): una cripta che rigenera mostri senza tregua non si finisce mai.
        # Qui 2 per volta, al massimo 3 vicini, ogni 35-90 s, solo con il giocatore a
        # 10 blocchi, e compaiono a 3 blocchi dallo spawner: restano nella sala.
        "MaxNearbyEntities": ShortTag(3),
        "RequiredPlayerRange": ShortTag(10),
        "SpawnCount": ShortTag(2),
        # formato del gioco recente (1.21.5+): `entity` dentro SpawnData e
        # `data`/`weight` in SpawnPotentials. Senza SpawnPotentials amulet
        # ne aggiunge uno di maiali nel vecchio formato, e dopo il primo
        # giro lo spawner farebbe maiali.
        "SpawnData": entita,
        "SpawnPotentials": ListTag(potenziali),
        "EntityIdentifier": StringTag(""),
        "MaxSpawnDelay": ShortTag(1800),
        "Delay": ShortTag(20),
        "SpawnRange": ShortTag(3),
        "MinSpawnDelay": ShortTag(700),
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
    """I `BlockEntity` degli spawner e dei distributori delle trappole che cadono nel
    chunk di angolo (ox, oz), coordinate di GIOCO."""
    fuori = []
    for d in dungeons:
        x, y, z = d.spawner()
        gx, gz = x - meta, z - meta
        if ox <= gx < ox + 16 and oz <= gz < oz + 16:
            fuori.append(be_spawner(gx, y, gz, d.mob, misto=True))
        for (x, y, z, _f) in d.distributori():
            if ox <= x - meta < ox + 16 and oz <= z - meta < oz + 16:
                fuori.append(be_dispenser(x - meta, y, z - meta))
    return fuori
