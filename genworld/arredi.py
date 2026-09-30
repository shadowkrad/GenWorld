"""Arredi urbani: lampioni, campana, fontana o pozzo, giardini, recinti, bazar.

Un lotto in cui non entra nessuna casa libera (vedi `template.assegna`) non
deve restare un buco nella fila, e una casa che sporge e viene tagliata dal
vicino non e' un rimedio: il vuoto si riempie con quello che riempie i vuoti
in un paese vero. Un giardino, un recinto per le bestie, un banco da mercato,
una piazzetta con un pozzo. E in ogni insediamento, indipendentemente dai
lotti, cio' che fa un paese di un gruppo di case: una campana (in Minecraft e'
anche il punto d'incontro degli abitanti), una fontana in citta' o un pozzo in
un borgo, i lampioni lungo le vie principali.

Come i template delle case, ogni arredo e' un `template.Modello`: un
parallelepipedo di celle con -1 dove non si tocca niente. Si disegnano qui in
codice, con i nomi di gioco dei blocchi (`minecraft:oak_fence`), e si
traducono UNA volta con PyMCTranslate; un blocco che il traduttore non
conosce solleva un errore invece di sparire in gioco - la trappola di
`universal_minecraft:oak_log`, che un mondo si scrive benissimo e non compare.
Cosi' la posa e' quella gia' collaudata delle case (`template.costruisci`), e
un arredo che sta a cavallo di due chunk viene ritagliato come loro.

Niente e' casuale senza seme: stessa mappa, stesso paese.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation

from . import edifici as E
from . import strade as ST
from . import template as TM
from .citta import ACQUA, MERCI
from .fauna import CORTILE, Animale

# in che direzione guarda l'ingresso di un lotto: dove sta la strada.
# Riga/colonna del lotto (in coordinate locali) che confina con la via.
NORD, EST, SUD, OVEST = E.NORD, E.EST, E.SUD, E.OVEST


# --------------------------------------------------------------------------
# Un disegno a blocchi, da tradurre in Modello
# --------------------------------------------------------------------------

class Disegno:
    """Celle (x, y, z) -> (blocco di gioco, proprieta'). y=0 e' il primo strato
    sopra il terreno. Il resto della scatola resta -1: non si tocca."""

    def __init__(self, nome: str, dx: int, dy: int, dz: int):
        self.nome = nome
        self.dim = (dx, dy, dz)
        self.blocchi: dict[tuple[int, int, int], tuple[str, tuple]] = {}

    def metti(self, x: int, y: int, z: int, nome: str, **prop: str) -> None:
        dx, dy, dz = self.dim
        if not (0 <= x < dx and 0 <= y < dy and 0 <= z < dz):
            raise ValueError(f"{self.nome}: ({x},{y},{z}) fuori da {self.dim}")
        self.blocchi[(x, y, z)] = (nome, tuple(sorted(prop.items())))

    def incolla(self, altro: "Disegno", ox: int, oy: int, oz: int) -> None:
        for (x, y, z), v in altro.blocchi.items():
            if 0 <= ox + x < self.dim[0] and 0 <= oy + y < self.dim[1] \
                    and 0 <= oz + z < self.dim[2]:
                self.blocchi[(ox + x, oy + y, oz + z)] = v

    def _connetti(self) -> None:
        """Le staccionate si collegano ai vicini. Un mondo scritto a mano non
        ricalcola le forme dei blocchi al caricamento: una staccionata scritta
        senza collegamenti resta una fila di pali isolati."""
        def e_recinto(pos) -> bool:
            v = self.blocchi.get(pos)
            return v is not None and (v[0].endswith("_fence") or v[0].endswith("_fence_gate"))

        for (x, y, z), (nome, prop) in list(self.blocchi.items()):
            if not nome.endswith("_fence"):
                continue
            p = dict(prop)
            p.update({"east": str(e_recinto((x + 1, y, z))).lower(),
                      "west": str(e_recinto((x - 1, y, z))).lower(),
                      "south": str(e_recinto((x, y, z + 1))).lower(),
                      "north": str(e_recinto((x, y, z - 1))).lower(),
                      "waterlogged": "false"})
            self.blocchi[(x, y, z)] = (nome, tuple(sorted(p.items())))

    def modello(self, ver) -> TM.Modello:
        self._connetti()
        chiavi = sorted(set(self.blocchi.values()))
        indice = {k: i for i, k in enumerate(chiavi)}
        tavolozza: list[tuple[str, dict]] = []
        for nome, prop in chiavi:
            base, p, tradotto = TM.traduci_blocco(ver, nome, dict(prop))
            if not tradotto:
                raise ValueError(f"{self.nome}: PyMCTranslate non traduce "
                                 f"{nome} {dict(prop)}")
            tavolozza.append((base, p))
        dx, dy, dz = self.dim
        celle = np.full((dx, dy, dz), -1, np.int32)
        for pos, v in self.blocchi.items():
            celle[pos] = indice[v]
        return TM.Modello(nome=self.nome, celle=celle, tavolozza=tavolozza)


_FOGLIE = {"distance": "1", "persistent": "true", "waterlogged": "false"}
_FIORI = ("poppy", "dandelion", "cornflower", "azure_bluet", "oxeye_daisy",
          "allium", "red_tulip", "pink_tulip", "lily_of_the_valley", "blue_orchid")
_MURETTO = {"up": "true", "north": "none", "south": "none", "east": "none",
            "west": "none"}


def _acqua() -> tuple[str, dict]:
    return "water", {"level": "0"}


# --------------------------------------------------------------------------
# Arredi di misura fissa
# --------------------------------------------------------------------------

def lampione() -> Disegno:
    """Un palo di steccato scuro con la lanterna in cima."""
    d = Disegno("lampione", 1, 4, 1)
    for y in range(3):
        d.metti(0, y, 0, "dark_oak_fence")
    d.metti(0, 3, 0, "lantern", hanging="false", waterlogged="false")
    return d


def campana() -> Disegno:
    """Una campana appesa a una trave fra due pali: 3 x 3 x 1.

    Qualunque campana e' un punto d'incontro per gli abitanti in Minecraft;
    e' anche il segnale d'allarme di un paese.
    """
    d = Disegno("campana", 3, 3, 1)
    for x in (0, 2):
        d.metti(x, 0, 0, "oak_fence")
        d.metti(x, 1, 0, "oak_fence")
    for x in range(3):
        d.metti(x, 2, 0, "oak_planks")
    d.metti(1, 1, 0, "bell", attachment="ceiling", facing="north", powered="false")
    return d


def fontana() -> Disegno:
    """Una vasca di 5 x 5 con un pilastro al centro. L'acqua sta dentro un
    anello pieno: sorgenti chiuse su tutti i lati, non scorrono via."""
    d = Disegno("fontana", 5, 3, 5)
    nome, prop = _acqua()
    for x in range(5):
        for z in range(5):
            anello = x in (0, 4) or z in (0, 4)
            if anello:
                d.metti(x, 0, z, "stone_bricks")
            elif (x, z) == (2, 2):
                d.metti(x, 0, z, "stone_brick_wall", waterlogged="true", **_MURETTO)
            else:
                d.metti(x, 0, z, nome, **prop)
    for x, z in ((0, 0), (4, 0), (0, 4), (4, 4)):
        d.metti(x, 1, z, "stone_brick_wall", waterlogged="false", **_MURETTO)
    d.metti(2, 1, 2, "stone_brick_wall", waterlogged="false", **_MURETTO)
    d.metti(2, 2, 2, "stone_brick_slab", type="bottom", waterlogged="false")
    return d


def pozzo() -> Disegno:
    """Un pozzo coperto di 3 x 3: anello di ciottoli, acqua al centro, quattro
    pali e un tettuccio. E' la fontana dei borghi."""
    d = Disegno("pozzo", 3, 4, 3)
    nome, prop = _acqua()
    for x in range(3):
        for z in range(3):
            if (x, z) == (1, 1):
                d.metti(x, 0, z, nome, **prop)
            else:
                d.metti(x, 0, z, "cobblestone")
    for x, z in ((0, 0), (2, 0), (0, 2), (2, 2)):
        d.metti(x, 1, z, "oak_fence")
        d.metti(x, 2, z, "oak_fence")
    for x in range(3):
        for z in range(3):
            d.metti(x, 3, z, "oak_planks")
    return d


def panchina() -> Disegno:
    d = Disegno("panchina", 3, 1, 1)
    d.metti(0, 0, 0, "oak_stairs", facing="east", half="bottom", shape="straight",
            waterlogged="false")
    d.metti(1, 0, 0, "oak_slab", type="bottom", waterlogged="false")
    d.metti(2, 0, 0, "oak_stairs", facing="west", half="bottom", shape="straight",
            waterlogged="false")
    return d


# --------------------------------------------------------------------------
# Riempitivi: prendono la misura del lotto
# --------------------------------------------------------------------------

def _lato_strada(verso: int, w: int, d: int):
    """Le celle di bordo del lotto dal lato della via, e il verso del cancello.
    (Riga z=0 se la strada e' a nord, colonna x=0 se e' a ovest, ecc.)"""
    if verso == NORD:
        return [(x, 0) for x in range(w)], "north", "x"
    if verso == SUD:
        return [(x, d - 1) for x in range(w)], "south", "x"
    if verso == OVEST:
        return [(0, z) for z in range(d)], "west", "z"
    return [(w - 1, z) for z in range(d)], "east", "z"


def _perimetro(w: int, d: int):
    return [(x, z) for x in range(w) for z in range(d)
            if x in (0, w - 1) or z in (0, d - 1)]


def giardino(w: int, d: int, verso: int, rng: np.random.Generator) -> Disegno:
    """Siepe bassa tutt'intorno con un varco sul lato della strada, fiori e
    qualche cespuglio di bacche dentro, un lampione se c'e' spazio."""
    g = Disegno("giardino", w, 4, d)
    lato, _, _ = _lato_strada(verso, w, d)
    centro = len(lato) // 2
    varco = {lato[centro], lato[max(0, centro - 1)]}
    for (x, z) in _perimetro(w, d):
        if (x, z) in varco:
            continue
        angolo = x in (0, w - 1) and z in (0, d - 1)
        foglia = "flowering_azalea_leaves" if rng.random() < 0.25 else "oak_leaves"
        g.metti(x, 0, z, foglia, **_FOGLIE)
        if angolo:
            g.metti(x, 1, z, foglia, **_FOGLIE)
    for x in range(1, w - 1):
        for z in range(1, d - 1):
            r = rng.random()
            if r < 0.55:
                g.metti(x, 0, z, _FIORI[int(rng.integers(0, len(_FIORI)))])
            elif r < 0.66:
                g.metti(x, 0, z, "sweet_berry_bush", age="3")
    if w >= 8 and d >= 8:
        g.incolla(lampione(), w // 2, 0, d // 2)
    return g


def recinto(w: int, d: int, verso: int, rng: np.random.Generator
            ) -> tuple[Disegno, list[tuple[float, float]]]:
    """Steccato con un cancello sul lato della strada, fieno e un abbeveratoio
    dentro. Ritorna anche le posizioni (locali) dove stanno le bestie."""
    g = Disegno("recinto", w, 2, d)
    lato, facing, _ = _lato_strada(verso, w, d)
    cancello = lato[len(lato) // 2]
    for (x, z) in _perimetro(w, d):
        if (x, z) == cancello:
            g.metti(x, 0, z, "oak_fence_gate", facing=facing, in_wall="false",
                    open="false", powered="false")
        else:
            g.metti(x, 0, z, "oak_fence")
    interni = [(x, z) for x in range(1, w - 1) for z in range(1, d - 1)]
    rng.shuffle(interni)
    occupate: set[tuple[int, int]] = set()
    if len(interni) >= 3:
        for (x, z) in interni[:2]:
            g.metti(x, 0, z, "hay_block", axis="y")
            occupate.add((x, z))
        x, z = interni[2]
        g.metti(x, 0, z, "water_cauldron", level="3")
        occupate.add((x, z))
    liberi = [c for c in interni if c not in occupate]
    return g, [(x + 0.5, z + 0.5) for (x, z) in liberi[:max(1, min(4, len(liberi) // 3))]]


def piazzetta(w: int, d: int, verso: int, rng: np.random.Generator) -> Disegno:
    """Un pozzo al centro, due panchine e una siepe agli angoli. Per lotti di
    almeno 7 x 7: le due righe delle panchine restano fuori dal pozzo."""
    g = Disegno("piazzetta", w, 4, d)
    g.incolla(pozzo(), (w - 3) // 2, 0, (d - 3) // 2)
    for z in (1, d - 2):
        g.incolla(panchina(), (w - 3) // 2, 0, z)
    for x, z in ((0, 0), (w - 1, 0), (0, d - 1), (w - 1, d - 1)):
        g.metti(x, 0, z, "oak_leaves", **_FOGLIE)
    return g


# --------------------------------------------------------------------------
# Pianificazione
# --------------------------------------------------------------------------

@dataclass
class Arredo:
    tipo: str
    x: int                  # angolo minimo, coordinate di mappa
    z: int
    larghezza: int
    profondita: int
    base: int               # quota del primo strato (primo blocco libero)
    modello: int            # indice in `Risultato.modelli`


@dataclass
class Risultato:
    arredi: list[Arredo] = field(default_factory=list)
    modelli: list = field(default_factory=list)
    banchi: list = field(default_factory=list)     # bazar: `Banco` nuovi
    animali: list = field(default_factory=list)    # bestie dei recinti


def indice_per_chunk(arredi: list[Arredo], passo: int = 16) -> dict:
    """Ogni arredo e' registrato in TUTTI i chunk che il suo ingombro tocca."""
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, a in enumerate(arredi):
        for cz in range(a.z // passo, (a.z + a.profondita - 1) // passo + 1):
            for cx in range(a.x // passo, (a.x + a.larghezza - 1) // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def maschera(arredi: list[Arredo], shape: tuple[int, int], margine: int = 0) -> np.ndarray:
    H, W = shape
    m = np.zeros(shape, bool)
    for a in arredi:
        m[max(0, a.z - margine):min(H, a.z + a.profondita + margine),
          max(0, a.x - margine):min(W, a.x + a.larghezza + margine)] = True
    return m


def statistiche(r: Risultato) -> dict:
    tipi: dict[str, int] = {}
    for a in r.arredi:
        tipi[a.tipo] = tipi.get(a.tipo, 0) + 1
    tipi["bazar"] = len(r.banchi)
    return tipi


def pianifica(edifici: list, scelte: dict, citta: list, h: np.ndarray, cls: np.ndarray,
              vie: np.ndarray | None, tipo_strada: np.ndarray | None,
              muro: np.ndarray | None, campi: np.ndarray | None, banchi: list,
              seed: int = 0, densita: float = 1.0, versione=(1, 21, 4),
              evita: np.ndarray | None = None) -> Risultato:
    """Arreda i lotti rimasti senza casa e gli spazi civici dei paesi.

    `scelte` e' `{indice edificio: (modello, rotazione)}` (vedi
    `template.assegna`): i lotti che non ci sono restano senza casa. `evita` e'
    una maschera in piu' (per esempio la zona del vulcano). `densita` scala i
    lampioni e la probabilita' del bazar; 0 spegne tutto.
    """
    ris = Risultato()
    if densita <= 0 or not citta:
        return ris
    rng = np.random.default_rng(seed * 4093 + 17)
    H, W = h.shape
    ver = TM.traduttore(versione)
    cache: dict[str, int] = {}

    def indice(nome: str, fabbrica) -> int:
        if nome not in cache:
            ris.modelli.append(fabbrica().modello(ver))
            cache[nome] = len(ris.modelli) - 1
        return cache[nome]

    # occupato: dove un arredo civico NON puo' stare
    occ = np.zeros((H, W), bool)
    for e in edifici:
        x0, z0, x1, z1 = e.ingombro_tetto()
        occ[max(0, z0 - 1):z1 + 1, max(0, x0 - 1):x1 + 1] = True
    for b in banchi:
        # solo il banco: il margine di rispetto lo mette `libero`, e un banco
        # che si tirava dietro il proprio margine non lasciava piu' posto a
        # una fontana in una piazza di undici celle
        occ[max(0, b.z):b.z1, max(0, b.x):b.x1] = True
    if muro is not None:
        occ |= muro > 0
    if campi is not None:
        occ |= campi > 0
    occ |= np.isin(cls, ACQUA)
    if evita is not None:
        occ |= evita
    strada = (tipo_strada > 0) if tipo_strada is not None else np.zeros((H, W), bool)
    # la piazza e' selciato: ci si puo' stare. Il resto della carreggiata no.
    carreggiata = strada & (tipo_strada != ST.LASTRICATO) if tipo_strada is not None \
        else np.zeros((H, W), bool)

    def libero(x0: int, z0: int, dx: int, dz: int, margine: int = 1) -> bool:
        xa, za = max(0, x0 - margine), max(0, z0 - margine)
        xb, zb = min(W, x0 + dx + margine), min(H, z0 + dz + margine)
        if x0 < 1 or z0 < 1 or x0 + dx >= W or z0 + dz >= H:
            return False
        if occ[za:zb, xa:xb].any() or carreggiata[z0:z0 + dz, x0:x0 + dx].any():
            return False
        q = h[z0:z0 + dz, x0:x0 + dx]
        return int(q.max() - q.min()) <= 1

    def posa(tipo: str, x0: int, z0: int, dx: int, dz: int, modello: int,
             base: int | None = None) -> None:
        if base is None:
            base = int(np.median(h[z0:z0 + dz, x0:x0 + dx]))
        ris.arredi.append(Arredo(tipo, x0, z0, dx, dz, base, modello))
        occ[max(0, z0 - 1):z0 + dz + 1, max(0, x0 - 1):x0 + dx + 1] = True

    def cerca(cx: int, cz: int, dx: int, dz: int, r_min: int, r_max: int) -> tuple[int, int] | None:
        """Il punto libero piu' vicino a (cx, cz), per anelli."""
        for r in range(r_min, r_max + 1):
            scelte_r = []
            for ox in range(-r, r + 1):
                for oz in range(-r, r + 1):
                    if max(abs(ox), abs(oz)) != r:
                        continue
                    x0, z0 = cx + ox - dx // 2, cz + oz - dz // 2
                    if libero(x0, z0, dx, dz):
                        scelte_r.append((x0, z0))
            if scelte_r:
                return scelte_r[int(rng.integers(0, len(scelte_r)))]
        return None

    # 1) piazza: fontana (citta') o pozzo (borgo), poi la campana
    for c in citta:
        pz, px = c.piazza
        # una citta' vuole la fontana; se in piazza non ce n'e' il posto (i
        # banchi del mercato le stanno sul bordo) si ripiega sul pozzo
        opzioni = ([("fontana", 5, fontana), ("pozzo", 3, pozzo)] if c.raggio >= 24
                   else [("pozzo", 3, pozzo)])
        for civico, dx, fabbrica in opzioni:
            p = cerca(px, pz, dx, dx, 0, 8)
            if p is not None:
                posa(civico, p[0], p[1], dx, dx, indice(civico, fabbrica))
                break
        q = cerca(px, pz, 3, 1, 2, 12)
        if q is not None:
            posa("campana", q[0], q[1], 3, 1, indice("campana", campana))

    # 2) lotti senza casa
    banchi_nuovi = 0
    for i, e in enumerate(edifici):
        if i in scelte or e.palafitta:
            continue
        w, d = e.larghezza, e.profondita
        # dove passa una strada o c'e' acqua l'arredio non si disegna (verrebbe
        # posato sopra la carreggiata): si tolgono quelle celle. Se mancano
        # piu' di un terzo del lotto, o e' dentro la zona del vulcano, si lascia
        # com'e'.
        inutile = (strada[e.z:e.z + d, e.x:e.x + w]
                   | np.isin(cls[e.z:e.z + d, e.x:e.x + w], ACQUA))
        if inutile.mean() > 1 / 3 or (evita is not None
                                      and evita[e.z:e.z + d, e.x:e.x + w].any()):
            continue
        cx, cz = e.x + w // 2, e.z + d // 2
        c = min(citta, key=lambda c: (c.x - cx) ** 2 + (c.z - cz) ** 2)
        t = float(np.hypot(c.x - cx, c.z - cz)) / max(c.raggio, 1)
        r = rng.random()
        base = e.base
        if min(w, d) >= 5 and t < 0.6 and r < 0.4 * min(densita, 1.0) and banchi_nuovi < 6:
            bx, bz = e.x + (w - 3) // 2, e.z + (d - 3) // 2
            q = h[bz:bz + 3, bx:bx + 3]
            ris.banchi.append(E.Banco(x=bx, z=bz, base=int(np.median(q)) + 1,
                                      verso=e.porta,
                                      merce=MERCI[banchi_nuovi % len(MERCI)],
                                      seme=int(rng.integers(0, 2 ** 31 - 1))))
            banchi_nuovi += 1
            continue
        bestie: list[tuple[float, float]] = []
        if t < 0.5 and min(w, d) >= 7 and r < 0.75:
            tipo, disegno = "piazzetta", piazzetta(w, d, e.porta, rng)
        elif t >= 0.5 and r < 0.35:
            tipo = "recinto"
            disegno, bestie = recinto(w, d, e.porta, rng)
        else:
            tipo, disegno = "giardino", giardino(w, d, e.porta, rng)
        modello = disegno.modello(ver)
        if inutile.any():
            # `inutile` e' (z, x), le celle sono (x, y, z)
            modello.celle = np.where(inutile.T[:, None, :], -1, modello.celle)
            bestie = [(lx, lz) for lx, lz in bestie if not inutile[int(lz), int(lx)]]
        ris.modelli.append(modello)
        ris.arredi.append(Arredo(tipo, e.x, e.z, w, d, base, len(ris.modelli) - 1))
        if bestie:
            specie = CORTILE[int(rng.integers(0, len(CORTILE)))]
            for (lx, lz) in bestie:
                ris.animali.append(Animale(x=e.x + lx, y=float(base), z=e.z + lz,
                                           specie=specie,
                                           seme=int(rng.integers(0, 2 ** 31 - 1))))
        occ[e.z:e.z + d, e.x:e.x + w] = True

    # 3) lampioni lungo le vie principali
    if vie is not None and vie.any():
        principali = binary_dilation(vie >= 2, iterations=3)
        bordo = binary_dilation(carreggiata, iterations=1) & ~carreggiata & principali
        lampioni: list[tuple[int, int]] = []
        for c in citta:
            zz, xx = np.nonzero(bordo)
            dist = np.hypot(zz - c.z, xx - c.x)
            dentro = dist <= c.raggio * 0.95
            ordine = np.argsort(dist[dentro])
            zz, xx = zz[dentro][ordine], xx[dentro][ordine]
            massimo = max(3, int(c.raggio / 5 * densita))
            presi = 0
            for z, x in zip(zz.tolist(), xx.tolist()):
                if presi >= massimo:
                    break
                if occ[z, x] or carreggiata[z, x]:
                    continue
                if any((x - lx) ** 2 + (z - lz) ** 2 < 81 for lx, lz in lampioni):
                    continue
                lampioni.append((x, z))
                posa("lampione", x, z, 1, 1, indice("lampione", lampione),
                     base=int(h[z, x]))
                presi += 1
    return ris
