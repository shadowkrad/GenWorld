"""Miniere artificiali: condotti scavati da qualcuno, non dall'acqua.

`sottosuolo.py` fa le caverne - cunicoli organici, un cammino casuale con
inerzia - e i filoni ambientali. Una miniera e' il contrario per costruzione:
dritta dove una caverna serpeggia, con un pozzo d'ingresso invece di sbucare
dal nulla, un binario nel mezzo, delle travi che la tengono su. E' la stessa
distinzione che passa fra un fiume e un canale.

Una miniera e':

* un **pozzo** verticale con la scala, dalla superficie fino al fondo;
* dal fondo, alcuni **bracci** - gallerie dritte che ogni tanto svoltano ad
  angolo retto, mai a caso: la svolta e' l'unica scelta, la direzione fra le
  due prima di ripartire;
* da un braccio, spesso, una **diramazione** secondaria che si stacca a
  meta' percorso - un vero albero di gallerie, non solo raggi dal pozzo;
* in fondo a ogni braccio e a ogni diramazione, un **giacimento**: una
  saletta scavata con un grumo grande di UN minerale solo incastonato
  nelle sue pareti, ben visibile appena si arriva - il motivo per cui quel
  ramo e' stato scavato, non un filoncino a caso lungo il tragitto.

Le gallerie si scavano alla cieca rispetto al rilievo sopra di loro tranne
per un controllo: prima di allungare un tratto si verifica che passi sotto
il cappello di sicurezza e non sotto un corso d'acqua, esattamente come le
caverne - altrimenti un tratto lungo puo' sbucare su un fianco di collina
lontano dal pozzo, o allagarsi sotto un fiume che al momento della
pianificazione della miniera esiste gia'.

Lungo il percorso, ogni tanto, anche un piccolo filone incidentale - stessa
tavola di `sottosuolo`, stessa rarita' (il diamante resta il piu' raro): un
extra in piu', non il motivo per cui la miniera e' degna di essere stata
scavata - quello lo sono i giacimenti in fondo ai rami.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sottosuolo import MINERALI, QUOTA_ARDESIA, FASCIA_ARDESIA, grumo

_MINERALE_PER_NOME = {m.nome: m for m in MINERALI}

LARGHEZZA = 1           # meta' larghezza della galleria oltre il centro (3 celle)
CAPPELLO = 4            # non scavare piu' vicino della superficie di questo
PASSO_TRAVE = 6         # ogni quanti blocchi un paio di travi di sostegno
DISTANZA_MINIMA_TRATTO = 6
LUNGHEZZA_TRATTO = (9, 22)
LUNGHEZZA_DIRAMAZIONE = (6, 14)     # le diramazioni secondarie sono piu' corte
PROB_DIRAMAZIONE = 0.55             # probabilita' che un braccio ne stacchi una
RAGGIO_GIACIMENTO_VUOTO = 2         # raggio della saletta scavata vuota
GIACIMENTO_QUANTI = (28, 55)        # blocchi di minerale nel grumo che la incastona

DIREZIONI = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Le due perpendicolari di ogni direzione: una svolta e' sempre di 90 gradi,
# mai dritta e mai indietro - tornare sui propri passi non scava niente di
# nuovo, e proseguire dritti renderebbe inutile la lista di tratti.
_PERPENDICOLARI = {
    (1, 0): ((0, 1), (0, -1)),
    (-1, 0): ((0, 1), (0, -1)),
    (0, 1): ((1, 0), (-1, 0)),
    (0, -1): ((1, 0), (-1, 0)),
}

_LATO = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}
_CURVE = {
    frozenset(("north", "east")): "north_east",
    frozenset(("north", "west")): "north_west",
    frozenset(("south", "east")): "south_east",
    frozenset(("south", "west")): "south_west",
}


@dataclass
class Segmento:
    """Un tratto dritto di galleria, in quota costante."""
    x0: int
    z0: int
    x1: int
    z1: int
    y: int
    forma_inizio: str | None = None   # forma del binario in (x0, z0), se e' una curva
    forma_fine: str | None = None     # forma del binario in (x1, z1), se e' una curva

    @property
    def lungo_x(self) -> bool:
        return self.z0 == self.z1


@dataclass
class Giacimento:
    """Una saletta a fine diramazione, con un grumo grande di UN minerale
    solo incastonato nelle sue pareti - il punto d'arrivo di un ramo di
    galleria, non un filoncino a caso lungo il percorso (quello resta,
    vedi `_carica_segmento`, ma e' un extra: questo e' il motivo per cui
    la diramazione e' stata scavata)."""
    x: int
    z: int
    y: int
    minerale: str    # `Minerale.nome`, vedi `sottosuolo.MINERALI`


@dataclass
class Miniera:
    x: int
    z: int
    y_superficie: int
    y_fondo: int
    segmenti: list[Segmento] = field(default_factory=list)
    giacimenti: list[Giacimento] = field(default_factory=list)


@dataclass
class Carrello:
    """Un minecart, per la stessa strada degli abitanti e della fauna -
    vedi `entita.scrivi_regioni`: basta avere `.x/.y/.z` e `.nbt_tag()`."""
    x: float
    y: float
    z: float
    seme: int = 0

    def nbt_tag(self):
        from amulet_nbt import (ByteTag, CompoundTag, DoubleTag, FloatTag,
                                IntTag, ListTag, StringTag)
        import random
        from amulet_nbt import IntArrayTag
        r = random.Random(self.seme)
        uuid = IntArrayTag([r.randint(-2 ** 31, 2 ** 31 - 1) for _ in range(4)])
        return CompoundTag({
            "id": StringTag("minecraft:minecart"),
            "Pos": ListTag([DoubleTag(self.x), DoubleTag(self.y), DoubleTag(self.z)]),
            "Motion": ListTag([DoubleTag(0.0), DoubleTag(0.0), DoubleTag(0.0)]),
            "Rotation": ListTag([FloatTag(0.0), FloatTag(0.0)]),
            "UUID": uuid,
            "Air": IntTag(300),
            "Fire": IntTag(-1),
            "FallDistance": FloatTag(0.0),
            "Invulnerable": ByteTag(0),
            "OnGround": ByteTag(1),
            "PersistenceRequired": ByteTag(1),
        })


# --------------------------------------------------------------------------
# Pianificazione
# --------------------------------------------------------------------------

def _sicuro(altezze: np.ndarray, acqua: np.ndarray, x0: int, z0: int,
           x1: int, z1: int, y: int, cappello: int) -> bool:
    """Il tratto passa abbastanza sotto la superficie e non sotto l'acqua.

    Stessa idea del cappello delle caverne (`sottosuolo.CAPPELLO`): senza
    questo controllo un tratto lungo puo' sbucare su un fianco di collina
    lontano dal pozzo, o aprirsi sotto un fiume o un lago.
    """
    H, W = altezze.shape
    n = max(2, int(max(abs(x1 - x0), abs(z1 - z0))))
    for t in np.linspace(0.0, 1.0, n):
        x = int(round(x0 + (x1 - x0) * t))
        z = int(round(z0 + (z1 - z0) * t))
        if not (0 <= z < H and 0 <= x < W):
            return False
        if acqua[z, x]:
            return False
        if altezze[z, x] - y < cappello:
            return False
    return True


def _scava_ramo(x: int, z: int, direzione: tuple[int, int], y: int, n_tratti: int,
                altezze: np.ndarray, acqua: np.ndarray, rng: np.random.Generator,
                lunghezze: tuple) -> tuple[list[Segmento], int, int, tuple[int, int]]:
    """Un ramo di galleria: fino a `n_tratti` tratti dritti, con una svolta a
    90 gradi (mai dritta, mai indietro) fra un tratto e il successivo - si
    ferma prima se non trova piu' un tratto sicuro. Usata sia per i bracci
    principali dal pozzo sia per le diramazioni secondarie che si staccano
    da un braccio: stessa meccanica, lunghezze diverse (`lunghezze`).

    Ritorna i segmenti scavati, il punto dove il ramo finisce e la
    direzione dell'ultimo tratto - quello che serve a `_galleria` per
    piazzare il giacimento subito oltre l'ultima rotaia, non a caso.
    """
    segmenti: list[Segmento] = []
    precedente: Segmento | None = None
    direzione_prec = direzione
    for _tratto in range(n_tratti):
        lunghezza = int(rng.integers(*lunghezze))
        dx, dz = direzione
        nx, nz = x, z
        trovato = False
        while lunghezza >= DISTANZA_MINIMA_TRATTO:
            nx, nz = x + dx * lunghezza, z + dz * lunghezza
            if _sicuro(altezze, acqua, x, z, nx, nz, y, CAPPELLO):
                trovato = True
                break
            lunghezza -= 4
        if not trovato:
            break
        seg = Segmento(x0=x, z0=z, x1=nx, z1=nz, y=y)
        if precedente is not None:
            entra = _LATO[(-direzione_prec[0], -direzione_prec[1])]
            esce = _LATO[direzione]
            forma = _CURVE.get(frozenset((entra, esce)))
            precedente.forma_fine = forma
            seg.forma_inizio = forma
        segmenti.append(seg)
        precedente = seg
        direzione_prec = direzione
        x, z = nx, nz
        # svolta di 90 gradi, mai dritti e mai indietro
        perp = _PERPENDICOLARI[direzione]
        direzione = perp[int(rng.integers(0, 2))]
    return segmenti, x, z, direzione_prec


def _scegli_minerale(y: int, rng: np.random.Generator) -> str | None:
    """Il minerale per un giacimento a fine diramazione: fra quelli validi
    alla quota `y`, pesato sulla stessa rarita' (`per_chunk`) del campo
    ambientale di `sottosuolo` - il "criterio" e' lo stesso di sempre, solo
    applicato a colpo sicuro a fine ramo invece che a un dado per colonna
    lungo tutta la galleria. `None` se nessun minerale copre questa quota
    (capita solo ai bordi delle fasce, mine molto superficiali o profonde)."""
    candidati = [m for m in MINERALI if m.y_min <= y <= m.y_max]
    if not candidati:
        return None
    pesi = np.array([m.per_chunk for m in candidati], np.float64)
    return candidati[int(rng.choice(len(candidati), p=pesi / pesi.sum()))].nome


def _galleria(cx: int, cz: int, y: int, altezze: np.ndarray, acqua: np.ndarray,
             rng: np.random.Generator) -> tuple[list[Segmento], list[Giacimento]]:
    """I bracci che partono dal fondo del pozzo, con le loro svolte, le
    diramazioni secondarie che si staccano da un braccio a meta' percorso, e
    un giacimento in fondo a ognuno - un vero albero di gallerie che porta
    da qualche parte, non un mazzo di raggi dal pozzo che finiscono nel
    nulla."""
    segmenti: list[Segmento] = []
    giacimenti: list[Giacimento] = []
    n_bracci = int(rng.integers(2, 5))
    usate: set[tuple[int, int]] = set()

    def termina(x: int, z: int, direzione: tuple[int, int]) -> None:
        nome = _scegli_minerale(y, rng)
        if nome is None:
            return
        giacimenti.append(Giacimento(x=x + direzione[0], z=z + direzione[1],
                                     y=y, minerale=nome))

    for _ in range(n_bracci):
        libere = [d for d in DIREZIONI if d not in usate]
        scelte = libere if libere else list(DIREZIONI)
        direzione = scelte[int(rng.integers(0, len(scelte)))]
        usate.add(direzione)
        ramo, fx, fz, fdir = _scava_ramo(cx, cz, direzione, y, int(rng.integers(1, 4)),
                                         altezze, acqua, rng, LUNGHEZZA_TRATTO)
        if not ramo:
            continue
        segmenti.extend(ramo)
        termina(fx, fz, fdir)

        if rng.random() < PROB_DIRAMAZIONE:
            # una diramazione secondaria: parte da un punto a caso lungo il
            # braccio appena scavato, non dalla sua fine - e' quello che fa
            # un ALBERO di gallerie invece di un semplice raggio piu' lungo.
            base_seg = ramo[int(rng.integers(0, len(ramo)))]
            t = float(rng.uniform(0.25, 0.85))
            bx = int(round(base_seg.x0 + (base_seg.x1 - base_seg.x0) * t))
            bz = int(round(base_seg.z0 + (base_seg.z1 - base_seg.z0) * t))
            perp = _PERPENDICOLARI[_asse(base_seg)]
            dir_dirama = perp[int(rng.integers(0, 2))]
            dirama, dfx, dfz, dfdir = _scava_ramo(bx, bz, dir_dirama, y,
                                                  int(rng.integers(1, 3)),
                                                  altezze, acqua, rng,
                                                  LUNGHEZZA_DIRAMAZIONE)
            if dirama:
                segmenti.extend(dirama)
                termina(dfx, dfz, dfdir)

    return segmenti, giacimenti


def pianifica(altezze: np.ndarray, mare: np.ndarray, acqua: np.ndarray | None = None,
             evita: np.ndarray | None = None, densita: float = 1.0,
             seed: int = 0) -> list[Miniera]:
    """Sceglie ingressi sparsi e scava pozzo e gallerie da ciascuno.

    `acqua` e' la maschera di TUTTA l'acqua (mare, fiumi, laghi): le
    gallerie non devono passare sotto nessuna delle tre. `evita` e'
    opzionale - lotti, mura, strade: un ingresso in mezzo a una casa si
    vede, anche se la miniera stessa non ha altri legami con gli
    insediamenti (sono sparse a caso sulla mappa, non vicino ai paesi).
    """
    H, W = altezze.shape
    if acqua is None:
        acqua = mare
    rng = np.random.default_rng(seed)
    quanti = int(max(0, round(H * W / 14_000 * densita)))
    fuori: list[Miniera] = []
    tentativi = 0
    limite = quanti * 8 + 30
    while len(fuori) < quanti and tentativi < limite:
        tentativi += 1
        z = int(rng.integers(12, H - 12))
        x = int(rng.integers(12, W - 12))
        if mare[z, x] or (evita is not None and evita[z, x]):
            continue
        y_sup = int(altezze[z, x])
        y_fondo = y_sup - int(rng.integers(20, 48))
        y_fondo = max(y_fondo, -58)
        if y_sup - y_fondo < 16:
            continue
        segmenti, giacimenti = _galleria(x, z, y_fondo, altezze, acqua, rng)
        if not segmenti:
            continue
        fuori.append(Miniera(x=x, z=z, y_superficie=y_sup, y_fondo=y_fondo,
                             segmenti=segmenti, giacimenti=giacimenti))
    return fuori


def indice_per_chunk(miniere: list[Miniera], passo: int = 16) -> dict:
    """Ogni miniera va registrata in tutti i chunk che pozzo e gallerie
    toccano - stessa idea di `insediamenti.indice_per_chunk`."""
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, mn in enumerate(miniere):
        # margine di RAGGIO_INGRESSO (la capanna d'ingresso, vedi
        # `_carica_ingresso`): senza, un pozzo vicino al bordo di un chunk
        # registra la miniera solo li', e la capanna che sporge nel chunk
        # accanto non verrebbe mai disegnata - "smezzata", il difetto che
        # ebbero le case da template.
        celle = {((mn.x + dx) // passo, (mn.z + dz) // passo)
                for dx in (-RAGGIO_INGRESSO, RAGGIO_INGRESSO)
                for dz in (-RAGGIO_INGRESSO, RAGGIO_INGRESSO)}
        for s in mn.segmenti:
            x0, x1 = sorted((s.x0, s.x1))
            z0, z1 = sorted((s.z0, s.z1))
            m = LARGHEZZA + 2
            for cz in range((z0 - m) // passo, (z1 + m) // passo + 1):
                for cx in range((x0 - m) // passo, (x1 + m) // passo + 1):
                    celle.add((cx, cz))
        # margine del giacimento (saletta vuota + il grumo di minerale che
        # la incastona, vedi `_carica_giacimento`): stesso motivo della
        # capanna sopra - senza margine una saletta a cavallo di un confine
        # di chunk si vedrebbe "smezzata" nel chunk dove non e' registrata.
        mg = RAGGIO_GIACIMENTO_VUOTO + 3
        for g in mn.giacimenti:
            for cz in range((g.z - mg) // passo, (g.z + mg) // passo + 1):
                for cx in range((g.x - mg) // passo, (g.x + mg) // passo + 1):
                    celle.add((cx, cz))
        for cella in celle:
            fuori.setdefault(cella, []).append(i)
    return fuori


def carrelli(miniere: list[Miniera], seed: int = 0) -> list[Carrello]:
    """Un carrello ogni tanto, appoggiato sul binario di una galleria."""
    rng = np.random.default_rng(seed)
    fuori: list[Carrello] = []
    for i, mn in enumerate(miniere):
        for s in mn.segmenti:
            if rng.random() > 0.35:
                continue
            t = float(rng.uniform(0.2, 0.8))
            x = s.x0 + (s.x1 - s.x0) * t
            z = s.z0 + (s.z1 - s.z0) * t
            fuori.append(Carrello(x=x + 0.5, y=float(s.y + 1), z=z + 0.5,
                                  seme=i * 97 + int(t * 1000)))
    return fuori


def statistiche(miniere: list[Miniera]) -> dict:
    if not miniere:
        return {"miniere": 0, "gallerie": 0, "lunghezza": 0}
    gallerie = sum(len(mn.segmenti) for mn in miniere)
    lunghezza = sum(abs(s.x1 - s.x0) + abs(s.z1 - s.z0)
                    for mn in miniere for s in mn.segmenti)
    giacimenti = sum(len(mn.giacimenti) for mn in miniere)
    return {
        "miniere": len(miniere),
        "gallerie": gallerie,
        "lunghezza": int(lunghezza),
        "giacimenti": giacimenti,
        "quota_media_fondo": float(np.mean([mn.y_fondo for mn in miniere])),
    }


# --------------------------------------------------------------------------
# Posa dentro il chunk
# --------------------------------------------------------------------------

class Tavolozza:
    """Id dei blocchi della miniera, risolti sul livello aperto."""

    def __init__(self, scrittore):
        s = scrittore
        self.aria = s.id_aria
        self.scala = {lato: s.blocco("ladder", facing=lato)
                      for lato in ("north", "south", "east", "west")}
        self.rotaia = {forma: s.blocco("rail", shape=forma) for forma in (
            "north_south", "east_west", "north_east", "north_west",
            "south_east", "south_west")}
        self.palo = s.blocco("log", axis="y", material="oak", stripped="true")
        # ATTENZIONE al nome universale. Non e' "wall_torch": nel formato
        # universale di PyMCTranslate torcia a terra e torcia a muro sono
        # LO STESSO blocco ("torch"), distinti solo dalla proprieta'
        # `facing` ("up" per quella a terra). Scriverlo come "wall_torch"
        # supera il controllo di andata (si salva senza errori) ma fallisce
        # al ritorno - lo stesso tranello gia' descritto in
        # `mondo.ScrittoreMondo.blocco` per i blocchi senza proprieta': qui
        # le proprieta' c'erano, ma il nome era comunque sbagliato, e in
        # gioco la torcia non compariva.
        self.torcia = {lato: s.blocco("torch", facing=lato)
                       for lato in ("north", "south", "east", "west")}
        # Capanna d'ingresso (vedi `_carica_ingresso`): una stanza chiusa di
        # base in pietra e pareti di assi, con un tetto pieno - non piu' un
        # castelletto aperto. `self.palo` (sopra) fa gia' da pilone d'angolo.
        self.base_muro = s.blocco("stone_bricks", variant="normal")
        self.parete = s.blocco("planks", material="oak")
        self.pavimento = s.blocco("planks", material="oak")
        self.tetto = s.blocco("stone_bricks", variant="normal")
        self.vetro = s.blocco("glass_pane", north="false", south="false",
                              east="false", west="false")
        self.lanterna = s.blocco("lantern", hanging="true", waterlogged="false")
        self.minerale = {}
        for m in MINERALI:
            p = m.proprieta or {}
            self.minerale[(m.nome, False)] = s.blocco(m.nome, **p)
            self.minerale[(m.nome, True)] = s.blocco(m.nome_ardesia, **p)


def _asse(seg: Segmento) -> tuple[int, int]:
    """Direzione unitaria dal primo al secondo estremo del tratto."""
    dx = 0 if seg.x0 == seg.x1 else (1 if seg.x1 > seg.x0 else -1)
    dz = 0 if seg.z0 == seg.z1 else (1 if seg.z1 > seg.z0 else -1)
    return dx, dz


def _carica_pozzo(out, tav, h_c, mn: Miniera, ox: int, oz: int, y0: int,
                  protetto_c=None) -> None:
    lx, lz = mn.x - ox, mn.z - oz
    if not (0 <= lx < 16 and 0 <= lz < 16):
        return
    if protetto_c is not None and protetto_c[lx, lz]:
        return
    H = out.shape[1]
    top = mn.y_superficie - 1 - y0
    fondo = mn.y_fondo - y0
    for ly in range(max(1, fondo), min(top + 1, H)):
        out[lx, ly, lz] = tav.aria
    # la scala e' appesa al lato sud del pozzo, che resta pieno apposta
    for ly in range(max(1, fondo), min(top, H)):
        if ly % 5 == 0:
            continue    # ogni tanto un gradino libero, non tutta scala
        out[lx, ly, lz] = tav.scala["north"]


RAGGIO_INGRESSO = 2     # meta' lato della capanna (5x5) intorno al pozzo
ALTEZZA_INGRESSO = 3    # altezza delle pareti, dal pavimento al tetto

_LATI_CARDINALI = ("north", "south", "east", "west")
_DELTA_LATO = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def _verso_ingresso(mn: Miniera) -> str:
    """Il lato dove si apre la porta della capanna - deterministico sulla
    posizione, cosi' la stessa miniera ha sempre lo stesso ingresso a ogni
    rigenerazione con lo stesso seed, senza dover aggiungere un campo alla
    dataclass `Miniera` solo per questo."""
    h = (mn.x * 374761393) ^ (mn.z * 668265263)
    return _LATI_CARDINALI[h & 0x3]


def _carica_ingresso(out, tav, mn: Miniera, ox: int, oz: int, y0: int,
                     protetto_c=None) -> None:
    """La capanna d'ingresso: pareti chiuse in pietra e assi su tre lati,
    una porta sul quarto, una finestra sul lato opposto, un tetto pieno e
    due lanterne appese - non piu' il castelletto aperto segnalato come "un
    gazebo con un buco nel mezzo" (screenshot dell'utente), ma un vero
    ripostiglio di minatori con la botola nel pavimento.

    Gira su TUTTO il chunk, esattamente come `_carica_segmento`, non solo
    sulla colonna del pozzo: la capanna e' 5x5, quindi un pozzo vicino al
    bordo di un chunk la mette a cavallo di due. Funziona solo se
    `indice_per_chunk` registra la miniera anche nel chunk vicino - senza
    quel margine si vedrebbe "smezzata", il difetto che ebbero le case da
    template.

    Non conosce l'altezza vera del terreno colonna per colonna (`h_c` non
    arriva qui, solo `mn.y_superficie`, la quota nel punto del pozzo): su un
    terreno inclinato la capanna resta comunque un parallelepipedo piatto,
    stessa semplificazione gia' presente nel castelletto precedente. Pareti,
    porta e pavimento vengono pero' sempre ripuliti esplicitamente (non solo
    disegnati sopra), cosi' la stanza resta vuota e la porta resta aperta
    anche quando sotto c'era gia' qualcosa.
    """
    H = out.shape[1]
    base = mn.y_superficie - y0
    cima = base + ALTEZZA_INGRESSO
    verso = _verso_ingresso(mn)
    dxp, dzp = _DELTA_LATO[verso]
    px, pz = dxp * RAGGIO_INGRESSO, dzp * RAGGIO_INGRESSO   # cella della porta
    fx, fz = -px, -pz                                       # cella della finestra

    for lx in range(16):
        wx = ox + lx
        dx = wx - mn.x
        if abs(dx) > RAGGIO_INGRESSO:
            continue
        for lz in range(16):
            wz = oz + lz
            dz = wz - mn.z
            if abs(dz) > RAGGIO_INGRESSO:
                continue
            if protetto_c is not None and protetto_c[lx, lz]:
                continue          # qui sotto c'e' gia' altro in superficie

            if 0 <= cima < H:
                out[lx, cima, lz] = tav.tetto      # tetto pieno, pozzo e porta compresi

            if dx == 0 and dz == 0:
                continue          # il pozzo stesso, gia' scavato da _carica_pozzo

            bordo = max(abs(dx), abs(dz)) == RAGGIO_INGRESSO
            angolo = bordo and abs(dx) == RAGGIO_INGRESSO and abs(dz) == RAGGIO_INGRESSO
            porta = bordo and not angolo and dx == px and dz == pz
            finestra = bordo and not angolo and not porta and dx == fx and dz == fz

            if angolo:
                for ly in range(base, base + ALTEZZA_INGRESSO):
                    if 0 <= ly < H:
                        out[lx, ly, lz] = tav.palo
                continue

            if bordo and not porta:
                if 0 <= base < H:
                    out[lx, base, lz] = tav.base_muro
                for ly in (base + 1, base + 2):
                    if not (0 <= ly < H):
                        continue
                    out[lx, ly, lz] = tav.vetro if (finestra and ly == base + 1) else tav.parete
                continue

            # interno della capanna, o vano della porta: pavimento, e vuoto
            # sopra - una vera stanza, non un blocco pieno con un'etichetta.
            if 0 <= base < H:
                out[lx, base, lz] = tav.pavimento
            for ly in (base + 1, base + 2):
                if 0 <= ly < H:
                    out[lx, ly, lz] = tav.aria

    # lanterne: una appesa al tetto sopra il pozzo, una sopra la soglia
    for lx0, lz0 in ((mn.x - ox, mn.z - oz), (mn.x + px - ox, mn.z + pz - oz)):
        if not (0 <= lx0 < 16 and 0 <= lz0 < 16):
            continue
        if protetto_c is not None and protetto_c[lx0, lz0]:
            continue
        if 0 <= cima - 1 < H:
            out[lx0, cima - 1, lz0] = tav.lanterna


def _carica_segmento(out, tav, rng, h_c, seg: Segmento, ox: int, oz: int,
                     y0: int, scavabile: tuple, protetto_c=None) -> None:
    H = out.shape[1]
    dx, dz = _asse(seg)
    lungo_x = dz == 0
    x0, x1 = sorted((seg.x0, seg.x1))
    z0, z1 = sorted((seg.z0, seg.z1))
    fy = seg.y - y0
    a1, a2 = fy + 1, fy + 2

    for lx in range(16):
        wx = ox + lx
        if not (x0 - LARGHEZZA <= wx <= x1 + LARGHEZZA):
            continue
        for lz in range(16):
            wz = oz + lz
            if not (z0 - LARGHEZZA <= wz <= z1 + LARGHEZZA):
                continue
            if protetto_c is not None and protetto_c[lx, lz]:
                # sopra questa colonna c'e' gia' una struttura in superficie:
                # niente scavo, niente binario, niente filone qui.
                continue
            if lungo_x:
                if not (x0 <= wx <= x1):
                    continue
                scarto = wz - seg.z0
            else:
                if not (z0 <= wz <= z1):
                    continue
                scarto = wx - seg.x0
            if abs(scarto) > LARGHEZZA:
                continue
            for ly in (a1, a2):
                if 1 <= ly < H:
                    out[lx, ly, lz] = tav.aria
            if scarto != 0:
                continue

            forma = "east_west" if lungo_x else "north_south"
            if (wx, wz) == (seg.x0, seg.z0) and seg.forma_inizio:
                forma = seg.forma_inizio
            elif (wx, wz) == (seg.x1, seg.z1) and seg.forma_fine:
                forma = seg.forma_fine
            if 0 <= a1 < H:
                out[lx, a1, lz] = tav.rotaia[forma]

            avanzamento = (wx - seg.x0) if lungo_x else (wz - seg.z0)
            if avanzamento % PASSO_TRAVE == 0:
                for offset in (LARGHEZZA, -LARGHEZZA):
                    if lungo_x:
                        plx, plz = lx, lz + offset
                    else:
                        plx, plz = lx + offset, lz
                    if 0 <= plx < 16 and 0 <= plz < 16:
                        for ly in (a1, a2):
                            if 0 <= ly < H:
                                out[plx, ly, plz] = tav.palo
                if avanzamento % (PASSO_TRAVE * 2) == 0 and 0 <= a2 < H:
                    lato = "east" if lungo_x else "south"
                    out[lx, a2, lz] = tav.torcia[lato]

            if rng.random() < 0.10:
                candidati = [m for m in MINERALI if m.y_min <= seg.y <= m.y_max]
                if candidati:
                    pesi = np.array([m.per_chunk for m in candidati], np.float64)
                    m = candidati[int(rng.choice(len(candidati),
                                                 p=pesi / pesi.sum()))]
                    lato_dx, lato_dz = (0, 1) if lungo_x else (1, 0)
                    verso = 1 if rng.random() < 0.5 else -1
                    dist = LARGHEZZA + int(rng.integers(2, 4))
                    gx = lx + lato_dx * dist * verso
                    gz = lz + lato_dz * dist * verso
                    gy = seg.y + int(rng.integers(-1, 2))
                    profondo = gy < QUOTA_ARDESIA - FASCIA_ARDESIA // 2
                    quanti = int(rng.integers(m.grumo[0], m.grumo[1] + 1))
                    grumo(out, gx, gz, gy, y0, h_c,
                         tav.minerale[(m.nome, profondo)], quanti, scavabile)


def _carica_giacimento(out, tav, h_c, rng, g: Giacimento, ox: int, oz: int,
                       y0: int, scavabile: tuple, protetto_c=None) -> None:
    """La saletta a fine diramazione: si scava una piccola stanza vuota,
    poi si incastona intorno un grumo grande del minerale scelto in
    `_galleria`. Il grumo rispetta `scavabile` (non riscrive l'aria appena
    scavata), quindi il minerale resta incastonato proprio sulle pareti
    della saletta - ben visibile appena si arriva in fondo al binario,
    invece di un filoncino nascosto che tocca scavare per trovare.
    """
    H = out.shape[1]
    gx, gz, gy = g.x - ox, g.z - oz, g.y
    r = RAGGIO_GIACIMENTO_VUOTO
    for dx in range(-r, r + 1):
        lx = gx + dx
        if not (0 <= lx < 16):
            continue
        for dz in range(-r, r + 1):
            lz = gz + dz
            if not (0 <= lz < 16):
                continue
            if protetto_c is not None and protetto_c[lx, lz]:
                continue
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy + dz * dz > r * r:
                    continue
                ly = gy + dy - y0
                if not (1 <= ly < H):
                    continue
                if gy + dy > int(h_c[lx, lz]) - 2:
                    continue          # mai piu' vicino della superficie del cappello
                out[lx, ly, lz] = tav.aria

    m = _MINERALE_PER_NOME.get(g.minerale)
    if m is None:
        return
    profondo = gy < QUOTA_ARDESIA - FASCIA_ARDESIA // 2
    quanti = int(rng.integers(*GIACIMENTO_QUANTI))
    grumo(out, gx, gz, gy, y0, h_c, tav.minerale[(m.nome, profondo)], quanti,
         scavabile, protetto=protetto_c)


def posa(out: np.ndarray, tav: Tavolozza, h_c: np.ndarray, ox: int, oz: int,
        y0: int, miniere: list[Miniera], quali, scavabile: tuple,
        seed: int = 0, protetto_c: np.ndarray | None = None) -> None:
    """Pozzi, gallerie, binari, travi, filoni e giacimenti dentro il chunk
    (16, H, 16).

    `scavabile` e' la stessa idea di `sottosuolo.Tavolozza.scavabile`: solo
    quello che e' gia' pietra (o ardesia, o uno strato) prende un filone -
    un filone non deve mai bucare l'aria di una galleria o il pavimento di
    una casa capitata sopra per caso.

    `protetto_c` (16x16, opzionale) e' la stessa maschera di
    `sottosuolo.posa()`: un ingresso non puo' nascere dentro un
    insediamento (`evita` in `pianifica()`), ma una galleria puo' comunque
    attraversarne uno durante il percorso - qui si evita di scavare aria
    sotto cio' che e' gia' disegnato in superficie.
    """
    for i in quali:
        mn = miniere[i]
        rng = np.random.default_rng(
            ((mn.x * 73856093) ^ (mn.z * 19349663) ^ (seed * 83492791)) & 0x7FFFFFFF)
        _carica_pozzo(out, tav, h_c, mn, ox, oz, y0, protetto_c)
        _carica_ingresso(out, tav, mn, ox, oz, y0, protetto_c)
        for seg in mn.segmenti:
            _carica_segmento(out, tav, rng, h_c, seg, ox, oz, y0, scavabile,
                             protetto_c)
        for g in mn.giacimenti:
            _carica_giacimento(out, tav, h_c, rng, g, ox, oz, y0, scavabile,
                               protetto_c)
