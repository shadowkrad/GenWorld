"""Castello in rovina (#6): un castello diroccato da esplorare, con mostri, trappole
e una cripta sotterranea.

In superficie: una cinta con brecce e torri spezzate, un portale sfondato, un
cortile invaso da erba, macerie, alberi morti, tombe e ossa, e il maschio
sventrato con la sala del trono, la scala rotta, le ragnatele. Il capo siede sul
trono. Sotto, una cripta che si apre dalla sala: anticamera, corridoi con
trappole (filo d'inciampo e distributori di frecce, fossa con le punte, piastre
sopra il TNT, forzieri-esca), la sala di uno spawner, le celle dei prigionieri e
il caveau del tesoro.

Coordinate locali (u, v) dal centro; u va verso il portale. La cinta e' un
quadrato di mezzo lato `K`. La cripta corre lungo +u, sull'asse v = `VC`.
Tutto il disordine (cosa e' crollato, dov'e' l'erba) viene da un rumore sulle
coordinate di MONDO: stesso castello, stesse rovine, e i confini fra chunk non si
vedono.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dungeon as DG
from . import torri as TR

K = 19                      # mezzo lato della cinta (39 x 39)
ALTEZZA_MURA = 8
CENTRO_MASCHIO = -6
MEZZO_MASCHIO = 7           # il maschio e' 15 x 15
PROFONDITA = 14             # quanto scende la scala della cripta
VC = -5                     # l'asse della cripta, lungo v
RILIEVO_MASSIMO = 5
DISTANZA_FRA_CASTELLI = 400
COPERTURA = 3

FACCIA = TR.FACCIA
NOMI_CASTELLO = ("Castel Malvento", "Rocca dei Corvi", "Castel Cenere", "Rocca Spezzata",
                 "Castel Nerofumo", "Rocca delle Ombre")
NOMI_RE = ("Re Spettro Aldovrando", "La Regina Pallida", "Lord Vaelor il Caduto",
           "Il Cavaliere Cavo", "Il Castellano Senzavolto", "Madama Orsolina Ossa")
CAPO_PER_ZONA = ("scheletro_nero", "vindicatore", "zombie", "scheletro")

# la cripta: (u0, u1) di ogni sala lungo +u (la scala occupa u da -10 a 3)
U_ANTICAMERA = (4, 12)
U_TRAPPOLA = (13, 28)
U_FOSSA = (29, 39)
U_SPAWNER = (40, 48)
U_PRIGIONI = (49, 63)
U_CAVEAU = (64, 72)
U_FINE = U_CAVEAU[1] + 2
STAZIONI_FILO = (17, 24)                # u dei due fili d'inciampo
FILO_SALA_U = CENTRO_MASCHIO + 4        # il filo della sala del trono, di traverso
FOSSA_U = (32, 36)                      # dove manca il pavimento (con la sponda libera)
CELLE_U = ((51, 53), (58, 60))
# celle del cortile dove nascono soldati e manichini: niente macerie
CORTILE_LIBERO = ((8, -4), (10, 5), (5, -9), (-5, 11), (12, 1), (-4, -11), (5, -4), (8, 8))


@dataclass
class Castello:
    x: int                      # centro della cinta, mappa
    z: int
    ys: int                     # quota del cortile
    dx: int
    dz: int
    nome: str = "Castel Malvento"
    capo: str = "Re Spettro Aldovrando"
    specie: str = "scheletro_nero"
    seme: int = 0

    def punto(self, u: int, v: int) -> tuple[int, int]:
        return (self.x + self.dx * u - self.dz * v, self.z + self.dz * u + self.dx * v)

    @property
    def yf(self) -> int:
        return self.ys - PROFONDITA

    def bbox(self) -> tuple[int, int, int, int]:
        pts = []
        for u in (-K - 4, K + 4, U_FINE + 3):
            for v in (-K - 4, K + 4):
                pts.append(self.punto(u, v))
        return (min(p[0] for p in pts), min(p[1] for p in pts),
                max(p[0] for p in pts), max(p[1] for p in pts))

    # -- posti notevoli, in coordinate locali ----------------------------------
    def posto_capo(self) -> tuple[int, int, int]:
        return (CENTRO_MASCHIO - 4, 0, self.ys + 1)

    def forzieri_locali(self) -> list[tuple[int, int, int, str]]:
        """(u, v, y, tipo) di tutti i forzieri veri."""
        yf = self.yf
        return [
            (CENTRO_MASCHIO - 6, 3, self.ys + 1, "tesoro"),           # dietro il trono
            (CENTRO_MASCHIO + 3, 5, self.ys, "stanza"),               # in un angolo della sala
            (-(K - 3), -(K - 3), self.ys, "stanza"),                  # nella torre dello spawner
            (U_SPAWNER[0] + 1, VC - 2, yf, "stanza"),
            (U_SPAWNER[0] + 1, VC + 2, yf, "stanza"),
            (CELLE_U[1][1], VC - 4, yf, "stanza"),                          # in fondo a una cella
            (U_CAVEAU[1] - 1, VC, yf, "tesoro"),
            (CENTRO_MASCHIO + MEZZO_MASCHIO, 4, self.ys, "stanza"),          # murato dietro uno scaffale
            (6, 6, self.ys - 4, "stanza"),                                  # sul fondo del pozzo
        ]

    def forzieri(self) -> list[tuple[int, int, int, str]]:
        fuori = []
        for u, v, y, tipo in self.forzieri_locali():
            x, z = self.punto(u, v)
            fuori.append((x, y, z, tipo))
        return fuori

    def spawner(self) -> list[tuple[int, int, int, str]]:
        """(x, y, z, mob) di MAPPA."""
        s1 = self.punto(-(K - 3) + 1, -(K - 3) + 1)
        s2 = self.punto((U_SPAWNER[0] + U_SPAWNER[1]) // 2, VC)
        s3 = self.punto(U_CAVEAU[0] + 4, VC)
        mob = "skeleton" if self.seme % 2 else "zombie"
        return [(s1[0], self.ys, s1[1], "zombie"), (s2[0], self.yf + 1, s2[1], mob),
                (s3[0], self.yf + 1, s3[1], "cave_spider")]

    def distributori(self) -> list[tuple[int, int, int, str]]:
        """(x, y, z, facing): i distributori di frecce nei muri: due per ognuno
        dei due fili della cripta e due per quello della sala del trono."""
        fuori = []
        for u in STAZIONI_FILO:
            for lato in (1, -1):
                x, z = self.punto(u, VC + 2 * lato)
                fuori.append((x, self.yf + 1, z, self._faccia(0, -lato)))
        for lato in (1, -1):
            x, z = self.punto(FILO_SALA_U, MEZZO_MASCHIO * lato)
            fuori.append((x, self.ys + 2, z, self._faccia(0, -lato)))
        return fuori

    def asse_v(self) -> str:
        """L'asse del MONDO lungo cui corre v: serve al filo d'inciampo."""
        return "z" if self.dx else "x"

    def _faccia(self, du: int, dv: int) -> str:
        return FACCIA[(self.dx * du - self.dz * dv, self.dz * du + self.dx * dv)]


# --------------------------------------------------------------------------
# Pianificazione
# --------------------------------------------------------------------------

def pianifica(altezze: np.ndarray, acqua: np.ndarray, evita: np.ndarray | None = None,
              densita: float = 1.0, seed: int = 0, celle_per: int = 900_000) -> list[Castello]:
    """Siti su un pianoro asciutto (rilievo <= 5 su 47 x 47), lontani da tutto, con
    terra sufficiente sopra la cripta."""
    H, W = altezze.shape
    rng = np.random.default_rng(seed + 606)
    quanti = int(max(0, round(H * W / celle_per * densita)))
    fuori: list[Castello] = []
    nomi, re = list(NOMI_CASTELLO), list(NOMI_RE)
    tentativi, limite = 0, quanti * 150 + 150
    r = K + 4
    margine = U_FINE + 8
    while len(fuori) < quanti and tentativi < limite:
        tentativi += 1
        x = int(rng.integers(margine, W - margine))
        z = int(rng.integers(margine, H - margine))
        if any((c.x - x) ** 2 + (c.z - z) ** 2 < DISTANZA_FRA_CASTELLI ** 2 for c in fuori):
            continue
        quadro = altezze[z - r:z + r + 1, x - r:x + r + 1]
        if int(quadro.max() - quadro.min()) > RILIEVO_MASSIMO:
            continue
        ys = int(quadro.max())
        if ys < 66 or ys - PROFONDITA < -40 or ys + 24 > 250:
            continue
        dx, dz = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(0, 4))]
        c = Castello(x=x, z=z, ys=ys, dx=dx, dz=dz,
                     nome=nomi.pop(int(rng.integers(0, len(nomi)))) if nomi else "Castello Perduto",
                     capo=re.pop(int(rng.integers(0, len(re)))) if re else "Il Re Senzanome",
                     specie=CAPO_PER_ZONA[int(rng.integers(0, len(CAPO_PER_ZONA)))],
                     seme=int(rng.integers(0, 2 ** 31 - 1)))
        if not _valido(c, altezze, acqua, evita):
            nomi.append(c.nome)
            re.append(c.capo)
            continue
        fuori.append(c)
    return fuori


def _valido(c: Castello, altezze, acqua, evita) -> bool:
    H, W = altezze.shape
    for u in range(-K - 4, K + 5):
        for v in range(-K - 4, K + 5):
            x, z = c.punto(u, v)
            if not (0 <= x < W and 0 <= z < H) or acqua[z, x]:
                return False
            if evita is not None and evita[z, x]:
                return False
    # la cripta fuori dalla cinta: asciutta, con terra sopra il soffitto
    for u in range(K + 1, U_FINE + 3):
        for v in range(VC - 5, VC + 6):
            x, z = c.punto(u, v)
            if not (0 <= x < W and 0 <= z < H) or acqua[z, x]:
                return False
            if evita is not None and evita[z, x]:
                return False
            if int(altezze[z, x]) - 1 < c.yf + 4 + COPERTURA:
                return False
    return True


def maschera(castelli: list[Castello], forma: tuple[int, int], margine: int = 0) -> np.ndarray:
    """Le colonne del castello (la cinta e un margine): niente alberi, case o animali."""
    occ = np.zeros(forma, bool)
    H, W = forma
    r = K + 3 + margine
    for c in castelli:
        for u in range(-r, r + 1):
            for v in range(-r, r + 1):
                x, z = c.punto(u, v)
                if 0 <= x < W and 0 <= z < H:
                    occ[z, x] = True
    return occ


def indice_per_chunk(castelli: list[Castello], passo: int = 16) -> dict:
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(castelli):
        x0, z0, x1, z1 = c.bbox()
        for cz in range(z0 // passo, z1 // passo + 1):
            for cx in range(x0 // passo, x1 // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def statistiche(castelli: list[Castello]) -> dict:
    return {"castelli": len(castelli)}


# --------------------------------------------------------------------------
# Abitanti
# --------------------------------------------------------------------------

def combattenti(castelli: list[Castello], centri: list | None = None,
                seed: int = 0, trofei: dict | None = None) -> list[TR.Combattente]:
    """Il capo sul trono; sentinelle nella sala; zombie, scheletri e ragni nel
    cortile e nelle torri; guardie e prigionieri nella cripta."""
    fuori: list[TR.Combattente] = []
    for i, c in enumerate(castelli):
        lontano = bool(centri) and min((c.x - cx) ** 2 + (c.z - cz) ** 2
                                       for cx, cz in centri) > TR.DISTANZA_TESORI ** 2
        u, v, y = c.posto_capo()
        x, z = c.punto(u, v)
        fuori.append(TR.Combattente(x=x + 0.5, y=float(y), z=z + 0.5, specie=c.specie,
                                    capo=True, nome=c.capo, livello=1 if lontano else 0,
                                    seme=c.seme, trofeo=(trofei or {}).get((c.x, c.z))))
        elenco = []                         # (u, v, y, specie)
        for k, (uu, vv) in enumerate(((5, 2), (5, -2), (-1, 4))):
            elenco.append((CENTRO_MASCHIO + uu, vv, c.ys, "scheletro" if k != 1 else "zombie"))
        for uu, vv, sp in ((8, -4, "zombie"), (10, 5, "zombie"), (5, -9, "husk"), (-5, 11, "zombie"),
                           (12, 1, "ragno"), (-4, -11, "ragno")):
            elenco.append((uu, vv, c.ys, sp))
        for su, sv in ((1, 1), (1, -1), (-1, 1)):                  # le torri senza spawner
            elenco.append((su * (K - 3), sv * (K - 3), c.ys, "scheletro"))
        for (uu, vv) in posti_sulle_mura(c):                       # arcieri sul camminamento
            elenco.append((uu, vv, c.ys + ALTEZZA_MURA, "scheletro"))
        yf = c.yf
        for uu, vv, sp in ((7, VC - 2, "scheletro"), (9, VC + 2, "scheletro"),
                           (20, VC, "zombie"), (26, VC, "zombie"), (38, VC, "ragno_grotta"),
                           (43, VC - 3, "scheletro"), (45, VC + 3, "zombie")):
            elenco.append((uu, vv, yf, sp))
        for uu, vv in ((8, VC), (44, VC - 2), (46, VC + 2), (20, VC + 1), (U_CAVEAU[0] + 6, VC - 3)):
            elenco.append((uu, vv, yf + 2, "pipistrello"))
        # il luogotenente: un secondo capo, con un nome, a guardia del tesoro
        x, z = c.punto(U_CAVEAU[0] + 5, VC + 3)
        fuori.append(TR.Combattente(x=x + 0.5, y=float(yf), z=z + 0.5, specie="vindicatore",
                                    capo=True, nome="Il Custode del Caveau", livello=0,
                                    seme=c.seme + 977))
        for k, (uu, vv, ang) in enumerate(((CENTRO_MASCHIO - 1, 5, 200.0), (CENTRO_MASCHIO + 1, -5, 90.0),
                                           (5, -4, 20.0), (8, 8, 310.0))):
            x, z = c.punto(uu, vv)
            fuori.append(TR.Manichino(x=x + 0.5, y=float(c.ys), z=z + 0.5, rotazione=ang,
                                      seme=c.seme + 53 * (k + 1)))
        for (u0, u1) in CELLE_U:                                    # i prigionieri
            for lato in (1, -1):
                elenco.append(((u0 + u1) // 2, VC + 3 * lato, yf,
                               "zombie" if (u0 + lato) % 2 else "scheletro"))
        for k, (uu, vv, yy, sp) in enumerate(elenco):
            x, z = c.punto(uu, vv)
            fuori.append(TR.Combattente(x=x + 0.5, y=float(yy), z=z + 0.5, specie=sp,
                                        seme=c.seme + 37 * (k + 1) + i + seed))
    return fuori


# --------------------------------------------------------------------------
# Posa
# --------------------------------------------------------------------------

class Tavolozza:
    """I blocchi della torre curata piu' quelli del castello in rovina."""

    def __init__(self, scrittore):
        s = scrittore
        self.torre = TR.Tavolozza(s)
        self.aria = s.id_aria
        self.mattoni = s.blocco("stone_bricks", variant="normal")
        self.crepati = s.blocco("stone_bricks", variant="cracked")
        self.muschiosi = s.blocco("stone_bricks", variant="mossy")
        self.scolpiti = s.blocco("stone_bricks", variant="chiseled")
        self.ciottoli = s.blocco("cobblestone")
        self.ciottoli_muschio = s.blocco("mossy_cobblestone")
        self.ghiaia = s.blocco("gravel")
        self.erba = s.blocco("grass_block")
        self.terra_grossa = s.blocco("dirt", variant="coarse")
        self.podzol = s.blocco("podzol")
        self.assi = s.blocco("planks", material="oak")
        self.erba_alta = s.blocco("plant", plant_type="grass")
        self.felce = s.blocco("plant", plant_type="fern")
        self.cespuglio_secco = s.blocco("plant", plant_type="dead_bush")
        self.ragnatela = s.blocco("cobweb")
        self.ossa = s.blocco("bone_block", axis="y")
        self.ossa_x = s.blocco("bone_block", axis="x")
        self.teschio = {r: s.blocco("head", mob="skeleton", rotation=str(r), no_drop="false",
                                    powered="false") for r in (0, 4, 8, 12)}
        facce = tuple(FACCIA.values())
        self.scala_pietra = {f: s.blocco("stairs", material="stone_brick", facing=f,
                                         half="bottom", shape="straight") for f in facce}
        self.scala_ciottoli = {f: s.blocco("stairs", material="cobblestone", facing=f,
                                           half="bottom", shape="straight") for f in facce}
        self.scala_legno = {f: s.blocco("stairs", material="spruce", facing=f,
                                        half="bottom", shape="straight") for f in facce}
        self.lastra_legno = s.blocco("slab", material="spruce", type="top")
        self.lastra_pietra = s.blocco("slab", material="stone_brick", type="bottom")
        self.muretto = s.blocco("wall", material="cobblestone", up="true", north="none",
                                south="none", east="none", west="none")
        self.sbarre = s.blocco("bars", material="iron", north="false", south="false",
                               east="false", west="false")
        self.recinto = s.blocco("fence", material="spruce", north="false", south="false",
                                east="false", west="false")
        self.tronco = s.blocco("log", axis="y", material="oak", stripped="false")
        self.ramo = {"x": s.blocco("log", axis="x", material="oak", stripped="false"),
                     "z": s.blocco("log", axis="z", material="oak", stripped="false")}
        self.falo_spento = s.blocco("campfire", facing="north", lit="false", signal_fire="false",
                                    waterlogged="false")
        self.falo = s.blocco("campfire", facing="north", lit="true", signal_fire="false",
                             waterlogged="false")
        self.anima = s.blocco("soul_lantern", hanging="true")
        self.anima_su = s.blocco("soul_lantern", hanging="false")
        self.annerito = [s.blocco("blackstone"), s.blocco("magma_block"), s.blocco("dirt", variant="coarse"),
                         s.blocco("coal_block")]
        self.muschio = s.blocco("moss_block")
        self.tappeto_muschio = s.blocco("moss_carpet")
        self.fungo = (s.blocco("brown_mushroom"), s.blocco("red_mushroom"))
        self.infestato = s.blocco("infested_block", material="stone_bricks")
        self.candela = s.blocco("candle", color="default", candles="2", lit="true")
        self.vite = {f: s.blocco("vine", north=str(f == "north").lower(),
                                 south=str(f == "south").lower(), east=str(f == "east").lower(),
                                 west=str(f == "west").lower(), up="false") for f in facce}
        self.tappeto = s.blocco("carpet", color="red")
        self.forziere = {f: s.blocco("chest", facing=f, connection="none", material="wood")
                         for f in facce}
        self.trappola = {f: s.blocco("trapped_chest", facing=f, connection="none") for f in facce}
        self.tnt = s.blocco("tnt", unstable="false", underwater="false")
        self.piastra = s.blocco("pressure_plate", material="stone", powered="false")
        self.gancio = {f: s.blocco("tripwire_hook", facing=f, attached="true", powered="false")
                       for f in facce}
        self.filo = {ass: s.blocco("tripwire", attached="true", disarmed="false", powered="false",
                                   suspended="false",
                                   east=str(ass == "x").lower(), west=str(ass == "x").lower(),
                                   north=str(ass == "z").lower(), south=str(ass == "z").lower())
                     for ass in ("x", "z")}
        self.distributore = {f: s.blocco("dispenser", facing=f, triggered="false") for f in facce}
        self.punta = s.blocco("pointed_dripstone", thickness="tip", vertical_direction="up")
        self.spawner = s.blocco("spawner")
        self.acqua = s.blocco("water")
        self.fieno = s.blocco("hay_block", axis="y")
        self.libreria = s.blocco("bookshelf")
        self.bauli: list[tuple[int, int, int, str]] = []


def _hash(a: int, b: int, c: int) -> int:
    return ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) & 0x7FFFFFFF


def _rum(x: int, z: int, sale: int) -> float:
    return (_hash(x, sale, z) % 10000) / 10000.0


def _liscio(x: int, z: int, sale: int, lato: int = 5) -> float:
    """Rumore 0..1 morbido (interpolato) sulle coordinate di mondo."""
    xi, zi = x // lato, z // lato
    fx, fz = (x % lato) / lato, (z % lato) / lato
    a, b = _rum(xi, zi, sale), _rum(xi + 1, zi, sale)
    c, d = _rum(xi, zi + 1, sale), _rum(xi + 1, zi + 1, sale)
    return (a * (1 - fx) + b * fx) * (1 - fz) + (c * (1 - fx) + d * fx) * fz


def _altezza_muro(x: int, z: int) -> int:
    """L'altezza della cinta in una cella: 0 = breccia, 1..7 = spezzata, 8 = intera."""
    n = _liscio(x, z, 1, 5)
    if n < 0.16:
        return 0
    if n < 0.38:
        return int(1 + (n - 0.16) / 0.22 * 6)
    return ALTEZZA_MURA


def posti_sulle_mura(c: Castello, quanti: int = 4) -> list[tuple[int, int]]:
    """(u, v) di celle del camminamento ancora intere (altezza 8), distanziate: li'
    stanno gli arcieri. Fuori dal portale."""
    cand = []
    for t in range(-K + 8, K - 7):
        for (u, v) in ((t, K - 1), (-K + 1, t), (t, -K + 1)):
            x, z = c.punto(u, v)
            if _altezza_muro(x, z) == ALTEZZA_MURA and not (u >= K - 6 and abs(v) <= 6):
                cand.append((u, v))
    scelti: list[tuple[int, int]] = []
    for (u, v) in cand:
        if all(abs(u - a) + abs(v - b) >= 12 for a, b in scelti):
            scelti.append((u, v))
        if len(scelti) == quanti:
            break
    return scelti


def posa(out: np.ndarray, tav: Tavolozza, h_c: np.ndarray, ox: int, oz: int, y0: int,
         castelli: list[Castello], quali, meta: int = 0) -> None:
    """Disegna i castelli nel chunk (16, H, 16): `ox, oz` di GIOCO, i castelli di MAPPA
    (`meta` e' lo scarto)."""
    H = out.shape[1]
    tt = tav.torre

    def metti(x: int, y: int, z: int, blocco: int) -> None:
        lx, lz, ly = x - meta - ox, z - meta - oz, y - y0
        if 0 <= lx < 16 and 0 <= lz < 16 and 1 <= ly < H:
            out[lx, ly, lz] = blocco

    def nel_chunk(x: int, z: int) -> bool:
        return 0 <= x - meta - ox < 16 and 0 <= z - meta - oz < 16

    def leggi(x: int, y: int, z: int) -> int:
        lx, lz, ly = x - meta - ox, z - meta - oz, y - y0
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 <= ly < H:
            return int(out[lx, ly, lz])
        return -1

    for i in quali:
        c = castelli[i]
        ys, yf = c.ys, c.yf

        def P(u, v):
            return c.punto(u, v)

        def faccia(du, dv):
            return c._faccia(du, dv)

        def alza(u, v, y, blocco):
            x, z = P(u, v)
            metti(x, y, z, blocco)

        def mattone(x, y, z):
            r = _hash(x, y, z) % 100
            return tav.crepati if r < 22 else tav.muschiosi if r < 48 else tav.mattoni

        altezza_muro = _altezza_muro

        # ===================== superficie: una colonna alla volta ===================
        r = K + 3
        for u in range(-r, r + 1):
            for v in range(-r, r + 1):
                x, z = P(u, v)
                if not nel_chunk(x, z):
                    continue
                lx, lz = x - meta - ox, z - meta - oz
                d = max(abs(u), abs(v))
                hh = int(h_c[lx, lz])
                if d <= K + 1:
                    for y in range(hh - 1, ys - 1):
                        metti(x, y, z, tav.ciottoli)
                in_torre = abs(u) >= K - 6 and abs(v) >= K - 6 and d <= K
                perimetro_t = in_torre and (abs(u) in (K - 6, K) or abs(v) in (K - 6, K))
                interno_t = in_torre and not perimetro_t
                in_mura = K - 1 <= d <= K and not in_torre
                in_maschio = (abs(u - CENTRO_MASCHIO) <= MEZZO_MASCHIO and abs(v) <= MEZZO_MASCHIO)
                perimetro_m = in_maschio and (abs(u - CENTRO_MASCHIO) == MEZZO_MASCHIO
                                              or abs(v) == MEZZO_MASCHIO)
                r1 = _hash(x, 7, z) % 100

                # ---- il suolo (cortile, torri, maschio): invaso dall'erba ----
                if d <= K - 2 or interno_t:
                    if in_maschio and not perimetro_m:
                        suolo = mattone(x, ys - 1, z)
                    elif r1 < 28:
                        suolo = tav.erba
                    elif r1 < 36:
                        suolo = tav.podzol
                    elif r1 < 46:
                        suolo = tav.terra_grossa
                    elif r1 < 60:
                        suolo = tav.ciottoli_muschio
                    elif r1 < 78:
                        suolo = tav.ghiaia
                    else:
                        suolo = tav.ciottoli
                    if abs(v) <= 1 and u >= 0 and not in_maschio:
                        suolo = tav.ghiaia if r1 < 60 else tav.ciottoli      # il viale, mezzo sparito
                    metti(x, ys - 1, z, suolo)
                    if suolo == tav.erba and not in_maschio:
                        r2 = _hash(x, 9, z) % 100
                        if r2 < 22:
                            metti(x, ys, z, tav.erba_alta)
                        elif r2 < 28:
                            metti(x, ys, z, tav.felce)
                        elif r2 < 33:
                            metti(x, ys, z, tav.cespuglio_secco)

                # ---- la cinta, con brecce -------------------------------------------
                if in_mura:
                    hw = altezza_muro(x, z)
                    # il portale: un'apertura larga cinque, con un architrave spezzato
                    if u >= K - 1 and abs(v) <= 2:
                        hw = 0
                    # i pilastri del portale, piu' alti
                    if u >= K - 3 and 3 <= abs(v) <= 5:
                        hw = max(hw, 6 + int(_liscio(x, z, 2, 3) * 6))
                    for y in range(ys, ys + hw):
                        metti(x, y, z, mattone(x, y, z))
                    if hw == ALTEZZA_MURA and d == K and (u + v) % 2 == 0:
                        metti(x, ys + hw, z, mattone(x, ys + hw, z))        # un merlo
                    nu = 1 if u >= K - 1 else (-1 if u <= -(K - 1) else 0)
                    nv = 1 if v >= K - 1 else (-1 if v <= -(K - 1) else 0)
                    if (nu == 0) != (nv == 0):
                        # sui lati dritti: cio' che e' crollato resta ai piedi, dentro e fuori
                        if hw < ALTEZZA_MURA:
                            for (du, dv) in ((nu, nv), (-nu, -nv)):
                                rr = _hash(x + du, 19, z + dv) % 100
                                if rr < 45:
                                    alza(u + du, v + dv, ys, tav.ciottoli_muschio if rr % 2
                                         else tav.scala_ciottoli[("north", "south", "east", "west")[rr % 4]])
                                if rr < 15:
                                    alza(u + du, v + dv, ys + 1, tav.ciottoli)
                            if hw > 0 and _hash(x, 20, z) % 100 < 50:
                                metti(x, ys + hw, z, tav.scala_ciottoli[("north", "south", "east", "west")[_hash(x, 21, z) % 4]])
                        # un velo di edera sulla faccia interna delle parti ancora alte
                        if hw >= 4 and _hash(x, 11, z) % 100 < 16 and abs(u if nu else v) <= K - 1:
                            for y in range(ys + 1, ys + min(hw, 7)):
                                alza(u - nu, v - nv, y, tav.vite[faccia(nu, nv)])

                # ---- le torri d'angolo: spezzate in cima --------------------------------
                if perimetro_t:
                    ht = 5 + int(_liscio(x, z, 3, 3) * 10) + (_hash(x, 12, z) % 3 - 1)
                    for y in range(ys, ys + ht):
                        metti(x, y, z, mattone(x, y, z))
                if interno_t:
                    for y in range(ys, ys + 12):
                        metti(x, y, z, tav.aria)
                    # un po' di edera sulle pareti interne
                    if abs(u) == K - 1 or abs(v) == K - 1 or abs(u) == K - 5 or abs(v) == K - 5:
                        if abs(u) == K - 1:
                            dd = (1 if u > 0 else -1, 0)
                        elif abs(u) == K - 5:
                            dd = (-1 if u > 0 else 1, 0)
                        elif abs(v) == K - 1:
                            dd = (0, 1 if v > 0 else -1)
                        else:
                            dd = (0, -1 if v > 0 else 1)
                        if _hash(x, 14, z) % 100 < 22:
                            for y in range(ys + 2, ys + 2 + 1 + _hash(x, 18, z) % 4):
                                alza(u, v, y, tav.vite[faccia(*dd)])
                    if _liscio(x, z, 4, 3) > 0.45:
                        metti(x, ys + 6, z, tav.assi)                     # un solaio sopravvissuto
                    if _hash(x, 13, z) % 100 < 9:
                        metti(x, ys + 11, z, tav.ragnatela)

                # ---- il maschio: pareti sventrate ------------------------------------------
                if perimetro_m:
                    hk = 20 - int(_liscio(x, z, 5, 4) * 14)
                    hk = max(hk, 3)
                    if u - CENTRO_MASCHIO == MEZZO_MASCHIO and abs(v) <= 3:
                        hk = int(_liscio(x, z, 6, 2) * 4)               # lo squarcio sulla fronte
                    for y in range(ys, ys + hk):
                        metti(x, y, z, mattone(x, y, z))
                elif in_maschio:
                    for y in range(ys, ys + 21):
                        metti(x, y, z, tav.aria)
                    # il primo solaio, bucato
                    if _liscio(x, z, 7, 3) > 0.5 and abs(u - CENTRO_MASCHIO) <= 5 and abs(v) <= 5:
                        metti(x, ys + 6, z, tav.assi)
                    if _liscio(x, z, 8, 3) > 0.62 and abs(u - CENTRO_MASCHIO) <= 5 and abs(v) <= 5:
                        metti(x, ys + 13, z, tav.assi)
                    if _hash(x, 15, z) % 100 < 6 and (abs(u - CENTRO_MASCHIO) >= 5 or abs(v) >= 5):
                        for y in (ys + 3 + _hash(x, 16, z) % 3, ys + 5):
                            metti(x, y, z, tav.ragnatela)

                # ---- macerie e ossa nel cortile ----------------------------------------------
                if (d <= K - 2 and not in_maschio and not in_torre
                        and not any(abs(u - a_) <= 1 and abs(v - b_) <= 1 for a_, b_ in CORTILE_LIBERO)):
                    r3 = _hash(x, 17, z) % 1000
                    if r3 < 22:
                        metti(x, ys, z, tav.ciottoli if r3 % 2 else tav.ciottoli_muschio)
                    elif r3 < 30:
                        metti(x, ys, z, tav.scala_ciottoli[("north", "south", "east", "west")[r3 % 4]])
                    elif r3 < 36:
                        metti(x, ys, z, tav.ossa_x if r3 % 2 else tav.ossa)
                    elif r3 < 38:
                        metti(x, ys, z, tav.teschio[(0, 4, 8, 12)[r3 % 4]])

        # ===================== elementi singoli =============================================
        # il pozzo, spezzato
        for uu in range(5, 8):
            for vv in range(5, 8):
                if (uu, vv) == (6, 6):
                    alza(uu, vv, ys - 1, tav.aria)
                    for y in range(ys - 4, ys - 1):
                        alza(uu, vv, y, tav.aria)
                    alza(uu, vv, ys - 5, tav.ossa)
                elif _hash(uu, 21, vv) % 100 < 70:
                    alza(uu, vv, ys, tav.muretto)
        # gli alberi morti
        for (uu, vv) in ((11, -9), (-9, 12), (13, 11)):
            alt = 4 + _hash(uu, 23, vv) % 3
            for y in range(ys, ys + alt):
                alza(uu, vv, y, tav.tronco)
            alza(uu + 1, vv, ys + alt - 1, tav.ramo["x"])
            alza(uu, vv - 1, ys + alt - 2, tav.ramo["z"])
            alza(uu - 1, vv, ys + alt - 3, tav.ramo["x"])
        # l'accampamento dei predoni, spento, e le tombe
        alza(3, -13, ys, tav.falo_spento)
        for (uu, vv) in ((2, -13), (4, -13), (3, -12)):
            alza(uu, vv, ys, tav.ramo["x"])
        for k, (uu, vv) in enumerate(((-15, 4), (-15, 6), (-15, 8), (-13, 5), (-13, 7))):
            if abs(uu) < K - 2 and abs(vv) < K - 2:
                alza(uu, vv, ys - 1, tav.terra_grossa)
                alza(uu, vv, ys, tav.muretto)
                alza(uu, vv, ys + 1, tav.lastra_pietra)

        # ---- i pezzi del portale caduto: sbarre sparse nell'apertura ------------------------
        for uu in (K - 1, K):
            for vv in range(-2, 3):
                x, z = P(uu, vv)
                for y in range(ys, ys + 6):
                    r = _hash(x, y, z) % 100
                    if r < 9:
                        metti(x, y, z, tav.sbarre)
                    elif y <= ys + 1 and r < 22:
                        metti(x, y, z, tav.ciottoli_muschio)

        # ===================== il maschio, dentro ===============================================
        mc = CENTRO_MASCHIO
        # la scala rotta lungo il muro di destra (v = +6), verso la fronte
        for j in range(7):
            if _hash(j, 31, c.seme) % 100 < 26:
                continue                                                     # un gradino mancante
            for y in range(ys, ys + j):
                alza(mc - 5 + j, 6, y, tav.mattoni)
            alza(mc - 5 + j, 6, ys + j, tav.scala_pietra[faccia(1, 0)])
        # il rialzo del trono
        for uu in range(mc - 6, mc - 3):
            for vv in range(-3, 4):
                alza(uu, vv, ys, tav.mattoni if _hash(uu, 33, vv) % 100 < 80 else tav.crepati)
                if _hash(uu, 34, vv) % 100 < 55:
                    alza(uu, vv, ys + 1, tav.tappeto)
        alza(mc - 5, 0, ys + 1, tav.scala_legno[faccia(-1, 0)])             # il trono, schienale al muro
        for vv in (-1, 1):
            alza(mc - 5, vv, ys + 1, tav.lastra_pietra)
        for vv in (-2, 2):
            alza(mc - 5, vv, ys + 1, tav.candela)
        # la tavola lunga, mezza distrutta, con le sedie rovesciate
        for uu in range(mc - 2, mc + 4):
            if _hash(uu, 35, c.seme) % 100 < 70:
                alza(uu, 0, ys, tav.lastra_legno)
        for uu in range(mc - 2, mc + 4, 2):
            for vv, f in ((2, faccia(0, 1)), (-2, faccia(0, -1))):
                if _hash(uu, vv + 40, c.seme) % 100 < 55:
                    alza(uu, vv, ys, tav.scala_legno[f])
        alza(mc + 1, 0, ys + 1, tav.candela)
        # un braciere spento, i barili, gli scaffali
        alza(mc + 5, 0, ys, tav.falo_spento)
        for (uu, vv) in ((mc + 3, 4), (mc + 4, 4), (mc - 3, 4)):
            alza(uu, vv, ys, tt.barile)
        for vv in (3, 4, 5):
            alza(mc + 6, vv, ys, tav.libreria)
            alza(mc + 6, vv, ys + 1, tav.libreria)
        # ragnatele al posto degli stendardi: in un castello infestato i colori
        # non dicono niente, le ragnatele si'
        for (uu, vv, (du, dv)) in ((mc - 3, 6, (0, -1)), (mc + 2, -6, (0, 1)),
                                   (mc - 3, -6, (0, 1)), (mc - 6, 3, (1, 0))):
            bx, bz = P(uu - du, vv - dv)
            if leggi(bx, ys + 4, bz) not in (-1, tav.aria):
                alza(uu, vv, ys + 4, tav.ragnatela)
        # i forzieri del maschio
        for (uu, vv, yy, tipo) in c.forzieri_locali()[:2]:
            x, z = P(uu, vv)
            verso = faccia(1, 0) if tipo == "tesoro" else faccia(0, -1)
            metti(x, yy, z, tav.forziere[verso])
            if nel_chunk(x, z):
                tav.bauli.append((x, yy, z, tipo))
        # la torre dello spawner (SW): un piedistallo, ragnatele, bottino
        su = sv = -1
        sx, sz = P(su * (K - 3) + 1, sv * (K - 3) + 1)
        for dxx in (-1, 0, 1):
            for dzz in (-1, 0, 1):
                alza(su * (K - 3) + 1 + dxx, sv * (K - 3) + 1 + dzz, ys - 1, tav.mattoni)
        metti(sx, ys, sz, tav.spawner)
        for (uu, vv) in ((-(K - 5), -(K - 5)), (-(K - 1), -(K - 1))):
            alza(uu, vv, ys + 1, tav.ragnatela)
        (uu, vv, yy, tipo) = c.forzieri_locali()[2]
        x, z = P(uu, vv)
        metti(x, yy, z, tav.forziere[faccia(0, 1)])
        if nel_chunk(x, z):
            tav.bauli.append((x, yy, z, tipo))

        # ---- segni di battaglia: terra bruciata presso le brecce, un carro rovesciato ----
        for t in range(-K + 7, K - 6):
            for (u, v) in ((K, t), (-K, t), (t, K), (t, -K)):
                x, z = P(u, v)
                if _altezza_muro(x, z) != 0 or (u >= K - 1 and abs(v) <= 2):
                    continue
                for k in (-3, -2, -1, 1, 2, 3):
                    du, dv = (k, 0) if abs(u) == K else (0, k)
                    if _hash(x + du, 25, z + dv) % 100 < 45:
                        alza(u + du, v + dv, ys - 1, tav.annerito[_hash(x + du, 26, z + dv) % 4])
        alza(9, 9, ys, tt.barile)
        alza(10, 9, ys, tav.fieno)
        alza(10, 10, ys, tav.fieno)
        alza(9, 10, ys, tav.lastra_legno)
        alza(11, 9, ys, tav.recinto)
        alza(9, 11, ys, tav.recinto)
        for (uu, vv) in ((3, 6), (-3, -12), (12, -3)):
            alza(uu, vv, ys, tav.anima_su)
        for vv in (4, 6, 8):
            alza(-14, vv, ys, tav.candela)

        # ---- la trappola della sala del trono: un filo d'inciampo di traverso -----------
        uf = FILO_SALA_U
        alza(uf, MEZZO_MASCHIO - 1, ys + 1, tav.gancio[faccia(0, -1)])
        alza(uf, -MEZZO_MASCHIO + 1, ys + 1, tav.gancio[faccia(0, 1)])
        for vv in range(-MEZZO_MASCHIO + 2, MEZZO_MASCHIO - 1):
            alza(uf, vv, ys + 1, tav.filo[c.asse_v()])
        for lato in (1, -1):
            alza(uf, MEZZO_MASCHIO * lato, ys + 1, tav.mattoni)
            alza(uf, MEZZO_MASCHIO * lato, ys + 2, tav.distributore[faccia(0, -lato)])

        # ---- i nascondigli: un forziere murato dietro lo scaffale, uno in fondo al pozzo --
        (uu, vv, yy, tipo) = c.forzieri_locali()[7]
        x, z = P(uu, vv)
        metti(x, yy, z, tav.forziere[faccia(-1, 0)])
        if nel_chunk(x, z):
            tav.bauli.append((x, yy, z, tipo))
        (uu, vv, yy, tipo) = c.forzieri_locali()[8]
        x, z = P(uu, vv)
        metti(x, yy, z, tav.forziere[faccia(1, 0)])
        if nel_chunk(x, z):
            tav.bauli.append((x, yy, z, tipo))

        # ===================== la cripta ==========================================================
        _cripta(c, tav, metti, nel_chunk, P, faccia, alza, ys, yf)


def _cripta(c: Castello, tav: Tavolozza, metti, nel_chunk, P, faccia, alza, ys: int, yf: int) -> None:
    """Scala, sale e trappole. Prima i gusci, poi lo scavo, poi l'allestimento."""
    tt = tav.torre
    scatole = [
        # (u0, u1, v0, v1, y0, y1, tipo)
        (4, 12, VC - 4, VC + 4, yf, yf + 4, "anticamera"),
        (13, 28, VC - 1, VC + 1, yf, yf + 2, "trappola"),
        (29, 39, VC - 1, VC + 1, yf, yf + 2, "fossa"),
        (40, 48, VC - 4, VC + 4, yf, yf + 4, "spawner"),
        (49, 63, VC - 1, VC + 1, yf, yf + 2, "prigioni"),
        (64, 72, VC - 4, VC + 4, yf, yf + 4, "caveau"),
    ]
    celle = []                      # cavita' delle prigioni
    for (u0, u1) in CELLE_U:
        for lato in (1, -1):
            lo, hi = sorted((VC + 2 * lato, VC + 4 * lato))
            celle.append((u0, u1, lo, hi, yf, yf + 2))

    def mattone(x, y, z):
        r = _hash(x, y, z) % 100
        if r < 3:
            return tav.infestato          # un mattone che sembra come gli altri: dentro c'e' un lepisma
        return tav.crepati if r < 24 else tav.muschiosi if r < 46 else tav.mattoni

    # ---- gusci ---------------------------------------------------------------
    for (u0, u1, v0, v1, y0_, y1_, tipo) in scatole:
        for u in range(u0 - 1, u1 + 2):
            for v in range(v0 - 1, v1 + 2):
                x, z = P(u, v)
                if not nel_chunk(x, z):
                    continue
                for y in range(y0_ - 1, y1_ + 2):
                    metti(x, y, z, mattone(x, y, z))
    for (u0, u1, v0, v1, y0_, y1_) in celle:
        for u in range(u0 - 1, u1 + 2):
            for v in range(v0 - 1, v1 + 2):
                x, z = P(u, v)
                if nel_chunk(x, z):
                    for y in range(y0_ - 1, y1_ + 2):
                        metti(x, y, z, mattone(x, y, z))
    # la scala: da (-10, VC) a (3, VC), un blocco in discesa a ogni cella
    for s in range(1, PROFONDITA + 1):
        u = -11 + s
        for v in range(VC - 2, VC + 3):
            x, z = P(u, v)
            if not nel_chunk(x, z):
                continue
            for y in range(ys - s - 1, min(ys - s + 5, ys)):
                if abs(v - VC) == 2 or (y == ys - s - 1):
                    metti(x, y, z, mattone(x, y, z))
    # ---- scavo ---------------------------------------------------------------------
    for s in range(1, PROFONDITA + 1):
        u = -11 + s
        for v in range(VC - 1, VC + 2):
            x, z = P(u, v)
            if not nel_chunk(x, z):
                continue
            for y in range(ys - s + 1, ys - s + 4):
                metti(x, y, z, tav.aria)
            # `facing` di una scala e' il verso in cui si SALE: si scende verso +u
            metti(x, ys - s, z, tav.scala_pietra[faccia(-1, 0)])
    for (u0, u1, v0, v1, y0_, y1_, tipo) in scatole:
        for u in range(u0, u1 + 1):
            for v in range(v0, v1 + 1):
                x, z = P(u, v)
                if not nel_chunk(x, z):
                    continue
                for y in range(y0_, y1_ + 1):
                    metti(x, y, z, tav.aria)
                metti(x, y0_ - 1, z, mattone(x, y0_ - 1, z))
    for (u0, u1, v0, v1, y0_, y1_) in celle:
        for u in range(u0, u1 + 1):
            for v in range(v0, v1 + 1):
                x, z = P(u, v)
                if nel_chunk(x, z):
                    for y in range(y0_, y1_ + 1):
                        metti(x, y, z, tav.aria)
                    metti(x, y0_ - 1, z, mattone(x, y0_ - 1, z))
    # ---- allestimento ---------------------------------------------------------------
    # ragnatele e ossa ovunque, qualche lanterna dell'anima
    for (u0, u1, v0, v1, y0_, y1_, tipo) in scatole:
        for u in range(u0, u1 + 1):
            for v in range(v0, v1 + 1):
                x, z = P(u, v)
                if not nel_chunk(x, z):
                    continue
                r = _hash(x, 41, z) % 100
                if r < 7:
                    metti(x, y1_, z, tav.ragnatela)
                elif r < 10 and tipo not in ("fossa",):
                    metti(x, y0_, z, tav.ossa if r % 2 else tav.teschio[(0, 4, 8, 12)[r % 4]])
        if tipo in ("anticamera", "spawner", "caveau"):
            for (uu, vv) in ((u0 + 1, v0 + 1), (u0 + 1, v1 - 1), (u1 - 1, v0 + 1), (u1 - 1, v1 - 1)):
                for y in range(y0_, y1_ + 1):
                    alza(uu, vv, y, tav.scolpiti)
            alza((u0 + u1) // 2, VC, y1_, tav.anima)
        if tipo in ("trappola", "fossa", "prigioni"):
            for uu in range(u0 + 3, u1, 6):
                alza(uu, VC, y1_, tav.anima)
        if tipo in ("anticamera", "spawner", "caveau", "prigioni"):
            # muschio e funghi sul pavimento, un lume di candele in un angolo
            for u in range(u0, u1 + 1):
                for v in range(v0, v1 + 1):
                    x, z = P(u, v)
                    if not nel_chunk(x, z):
                        continue
                    r = _hash(x, 43, z) % 100
                    if r < 9:
                        metti(x, y0_ - 1, z, tav.muschio)
                        if _hash(x, 44, z) % 100 < 60:
                            metti(x, y0_, z, tav.tappeto_muschio)
                    elif r < 12:
                        metti(x, y0_, z, tav.fungo[r % 2])
            alza(u0 + 2, v0 + 2, y0_, tav.candela)
    # -- trappola 1: due fili d'inciampo con i distributori di frecce ---------------------
    for u in STAZIONI_FILO:
        alza(u, VC + 1, yf, tav.gancio[faccia(0, -1)])
        alza(u, VC - 1, yf, tav.gancio[faccia(0, 1)])
        alza(u, VC, yf, tav.filo[c.asse_v()])
        for lato in (1, -1):
            # il distributore e' appoggiato al blocco di muro a cui e' attaccato il gancio
            alza(u, VC + 2 * lato, yf + 1, tav.distributore[faccia(0, -lato)])
    # -- trappola 2: la fossa con le punte, e una sponda libera lungo un lato ------------------
    for u in range(FOSSA_U[0], FOSSA_U[1] + 1):
        for v in (VC, VC + 1):                                    # la sponda e' v = VC - 1
            for y in range(yf - 4, yf):
                alza(u, v, y, tav.aria)
            alza(u, v, yf - 5, mattone(u, 0, v))
            alza(u, v, yf - 4, tav.punta)
        for v in (VC - 1,):
            alza(u, v, yf - 1, tav.crepati)                        # la sponda, stretta e crepata
    for u in (FOSSA_U[0] - 1, FOSSA_U[1] + 1):
        for v in (VC, VC + 1):
            alza(u, v, yf, tav.ragnatela)
    # -- sala dello spawner -----------------------------------------------------------------------
    um = (U_SPAWNER[0] + U_SPAWNER[1]) // 2
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            alza(um + du, VC + dv, yf, tav.mattoni)
    for (x_, y_, z_, mob) in c.spawner()[1:2]:
        metti(x_, y_, z_, tav.spawner)
    # -- le celle: le sbarre sul lato del corridoio -------------------------------------------------
    for (u0, u1) in CELLE_U:
        for lato in (1, -1):
            for u in range(u0, u1 + 1):
                for y in range(yf, yf + 3):
                    alza(u, VC + 2 * lato, y, tav.sbarre)
            alza((u0 + u1) // 2, VC + 4 * lato, yf, tav.fieno)
    # -- caveau: piastre sopra il TNT all'ingresso, spawner di ragni, esche e il vero tesoro -------
    ui = U_CAVEAU[0]
    for uu in (ui + 1, ui + 2):
        for v in range(VC - 2, VC + 3):
            alza(uu, v, yf - 1, tav.tnt)
            alza(uu, v, yf, tav.piastra)
    for (x_, y_, z_, mob) in c.spawner()[2:3]:
        metti(x_, y_, z_, tav.spawner)
    us = U_CAVEAU[0] + 4
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            alza(us + du, VC + dv, yf, tav.mattoni)
    uc = U_CAVEAU[1] - 1
    for v in (VC - 2, VC + 2):
        alza(uc, v, yf - 1, tav.tnt)
        alza(uc, v, yf, tav.trappola[faccia(-1, 0)])
    # -- i forzieri veri della cripta (gli ultimi cinque) -----------------------------------------
    for (uu, vv, yy, tipo) in c.forzieri_locali()[3:7]:
        x, z = P(uu, vv)
        metti(x, yy, z, tav.forziere[faccia(-1, 0)])
        if nel_chunk(x, z):
            tav.bauli.append((x, yy, z, tipo))


def entita_blocco(castelli: list[Castello], ox: int, oz: int, meta: int) -> list:
    """I `BlockEntity` (spawner e distributori) che cadono nel chunk di angolo
    (ox, oz), coordinate di GIOCO."""
    fuori = []
    for c in castelli:
        for (x, y, z, mob) in c.spawner():
            gx, gz = x - meta, z - meta
            if ox <= gx < ox + 16 and oz <= gz < oz + 16:
                fuori.append(DG.be_spawner(gx, y, gz, mob))
        for (x, y, z, _f) in c.distributori():
            gx, gz = x - meta, z - meta
            if ox <= gx < ox + 16 and oz <= gz < oz + 16:
                fuori.append(DG.be_dispenser(gx, y, gz))
    return fuori
