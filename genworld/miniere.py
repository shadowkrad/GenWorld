"""Miniere artificiali: gallerie scavate da qualcuno, non dall'acqua.

`sottosuolo.py` fa le caverne - cunicoli organici, un cammino casuale con
inerzia - e i filoni ambientali. Una miniera e' il contrario per costruzione:
dritta dove una caverna serpeggia, con un ingresso costruito invece di
sbucare dal nulla, un binario nel mezzo, delle travi che la tengono su.

Una miniera e' fatta cosi', come quelle americane di collina:

* un **portale** di travi di tronco sul fianco di un rilievo - due pali, un
  architrave, i muretti di pietra ai lati, le lanterne - con il binario che
  esce all'aperto e un carrello fermo davanti;
* una **rampa**: la galleria entra nella collina e SCENDE piano, un blocco ogni
  tre di cammino, con le travi di sostegno ogni quattro blocchi e i binari in
  pendenza. All'inizio, dove la roccia sopra e' poca, e' una trincea aperta; poi
  si copre. Non c'e' un pozzo: per arrivare in fondo si cammina (o si va in
  carrello) per decine di blocchi;
* in fondo alla rampa, alcuni **bracci** - gallerie dritte che ogni tanto
  svoltano ad angolo retto, mai a caso - e da un braccio, spesso, una
  **diramazione** secondaria che si stacca a meta' percorso: un albero di
  gallerie, non solo raggi. E dalla fine di un braccio, spesso, **un'altra
  rampa** che scende a un secondo livello, poi a un terzo: la miniera ha piani,
  e ogni piano e' piu' ricco del precedente;
* in fondo a ogni braccio, un **giacimento**: una saletta con un grumo grande di
  UN minerale solo incastonato nelle pareti. Il minerale dipende dalla
  profondita': poco sotto la superficie carbone e rame, in fondo - dove la
  discesa e' costata fatica - i minerali rari. Sono la ricompensa di arrivare
  fin laggiu'.

Le gallerie si scavano alla cieca rispetto al rilievo sopra di loro tranne per
un controllo: prima di allungare un tratto si verifica che passi sotto il
cappello di sicurezza e non sotto un corso d'acqua, esattamente come le
caverne.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sottosuolo import MINERALI, QUOTA_ARDESIA, FASCIA_ARDESIA, grumo

_MINERALE_PER_NOME = {m.nome: m for m in MINERALI}

LARGHEZZA = 1           # meta' larghezza della galleria oltre il centro (3 celle)
CAPPELLO = 4            # non scavare piu' vicino della superficie di questo
PASSO_TRAVE = 6         # ogni quanti blocchi un paio di travi di sostegno (bracci)
DISTANZA_MINIMA_TRATTO = 6
LUNGHEZZA_TRATTO = (9, 22)
LUNGHEZZA_DIRAMAZIONE = (6, 14)     # le diramazioni secondarie sono piu' corte
PROB_DIRAMAZIONE = 0.55             # probabilita' che un braccio ne stacchi una
RAGGIO_GIACIMENTO_VUOTO = 2         # raggio della saletta scavata vuota
GIACIMENTO_QUANTI = (28, 55)        # blocchi di minerale nel grumo che la incastona

# La rampa
PASSO_RAMPA = 3                     # un blocco di discesa ogni PASSO_RAMPA di cammino
PROFONDITA_RAMPA = (16, 32)         # di quanto scende, al massimo
# Celle libere sopra il pavimento della rampa. Il telaio mette la trave in cima e
# la lanterna appesa un blocco sotto: con tre celle restava UN blocco di luce
# sotto la lanterna e il personaggio (alto due) non passava. Con quattro ne
# restano due.
ALTEZZA_GALLERIA = 4
COPERTURA_RAMPA = 2                 # terra sopra il soffitto perche' sia una galleria
LUNGHEZZA_RAMPA_MIN = 30            # una rampa piu' corta non e' una discesa graduale
PASSO_TRAVE_RAMPA = 4               # un telaio di travi ogni 4 celle di rampa
QUOTA_MINIMA_RAMPA = -50            # sotto, il fondo e' troppo vicino alla bedrock
LUNGHEZZA_IMBOCCO = 7                # quanto la roccia dell'imbocco si addentra
LARGHEZZA_IMBOCCO = 5                # e quanto si allarga ai lati del portale
ALTEZZA_IMBOCCO = 7                  # altezza della roccia sopra il pavimento, sulla facciata
BINARIO_FUORI = 6                   # celle di binario all'aperto, davanti al portale
DISTANZA_FRA_MINIERE = 90           # nessuna miniera a portata d'occhio di un'altra
LIVELLI_MAX = 3                     # piani di una miniera, il primo compreso
PROB_LIVELLO_SOTTO = 0.92           # probabilita' di scendere ancora, per ogni piano
PROFONDITA_RAMPA_INTERNA = (16, 28) # di quanto scende una rampa fra due piani

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
class Rampa:
    """La galleria in discesa che parte dal portale.

    La cella `i` e' `(x + dx*i, z + dz*i)` e il suo pavimento sta a
    `y - i // PASSO_RAMPA`: scende di un blocco ogni `PASSO_RAMPA`. Il binario
    sta un blocco sopra il pavimento, ed e' in pendenza solo sulla prima cella
    dopo ogni gradino.
    """
    x: int
    z: int
    dx: int
    dz: int
    y: int              # quota del pavimento nella cella 0, dove sta il portale
    lunghezza: int      # indice dell'ultima cella: la rampa ne ha lunghezza + 1

    def cella(self, i: int) -> tuple[int, int, int]:
        return (self.x + self.dx * i, self.z + self.dz * i, self.y - i // PASSO_RAMPA)

    @property
    def fine(self) -> tuple[int, int, int]:
        return self.cella(self.lunghezza)

    def e_gradino(self, i: int) -> bool:
        """La cella `i` e' la prima dopo un gradino: il suo binario scende."""
        return i > 0 and i % PASSO_RAMPA == 0


@dataclass
class Miniera:
    x: int                      # il portale
    z: int
    y_superficie: int           # quota del pavimento del portale
    y_fondo: int                # quota delle gallerie in fondo alla rampa
    rampa: Rampa | None = None
    # le rampe fra un piano e il successivo: partono dalla fine di un braccio
    rampe_interne: list[Rampa] = field(default_factory=list)
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
    lontano dall'imbocco, o aprirsi sotto un fiume o un lago.
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
    principali dal fondo della rampa sia per le diramazioni secondarie che si
    staccano da un braccio: stessa meccanica, lunghezze diverse (`lunghezze`).

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


def _profondita_relativa(y: int, y_superficie: int) -> float:
    """0 in superficie, 1 sulla bedrock (y = -60): quanto e' profonda una quota."""
    return float(np.clip((y_superficie - y) / max(1.0, y_superficie + 60.0), 0.0, 1.0))


def _scegli_minerale(y: int, rng: np.random.Generator,
                     profondita: float = 0.5) -> str | None:
    """Il minerale per un giacimento: fra quelli validi alla quota `y`.

    Il peso e' la rarita' elevata a un esponente che dipende dalla
    PROFONDITA: vicino alla superficie (0) vince il comune - carbone, rame,
    ferro; in fondo (1) il rapporto si rovescia e vincono i rari - oro,
    redstone, lapislazzuli, diamante, smeraldo. E' il motivo per cui conviene
    scendere fino in fondo. `None` se nessun minerale copre questa quota.
    """
    candidati = [m for m in MINERALI if m.y_min <= y <= m.y_max]
    if not candidati:
        return None
    esponente = 1.5 - 3.0 * float(np.clip(profondita, 0.0, 1.0))
    pesi = np.array([m.per_chunk for m in candidati], np.float64) ** esponente
    return candidati[int(rng.choice(len(candidati), p=pesi / pesi.sum()))].nome


def _galleria(cx: int, cz: int, y: int, altezze: np.ndarray, acqua: np.ndarray,
              rng: np.random.Generator, escludi: tuple = (),
              y_superficie: int | None = None) -> tuple[list[Segmento], list[Giacimento]]:
    """I bracci che partono dal fondo della rampa, con le loro svolte, le
    diramazioni secondarie che si staccano da un braccio a meta' percorso, e
    un giacimento in fondo a ognuno - un vero albero di gallerie che porta
    da qualche parte, non un mazzo di raggi che finiscono nel nulla.

    `escludi` sono le direzioni da non prendere (la rampa da cui si arriva).
    """
    segmenti: list[Segmento] = []
    giacimenti: list[Giacimento] = []
    n_bracci = int(rng.integers(2, 5))
    usate: set[tuple[int, int]] = set(escludi)
    if y_superficie is None:
        y_superficie = int(altezze[min(max(cz, 0), altezze.shape[0] - 1),
                                   min(max(cx, 0), altezze.shape[1] - 1)])
    prof = _profondita_relativa(y, y_superficie)

    def termina(x: int, z: int, direzione: tuple[int, int]) -> None:
        nome = _scegli_minerale(y, rng, prof)
        if nome is None:
            return
        giacimenti.append(Giacimento(x=x + direzione[0], z=z + direzione[1],
                                     y=y, minerale=nome))

    for _ in range(n_bracci):
        libere = [d for d in DIREZIONI if d not in usate]
        scelte = libere if libere else [d for d in DIREZIONI if d not in escludi]
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


def _direzione_in_salita(altezze: np.ndarray, x: int, z: int,
                         rng: np.random.Generator) -> tuple[int, int]:
    """La direzione cardinale verso cui il terreno sale di piu': e' li' che
    un portale entra in una collina. Su terreno piatto, una a caso."""
    H, W = altezze.shape
    migliore, salita = None, -1e9
    ordine = list(DIREZIONI)
    rng.shuffle(ordine)
    for dx, dz in ordine:
        xx = min(max(x + dx * 16, 0), W - 1)
        zz = min(max(z + dz * 16, 0), H - 1)
        s = float(altezze[zz, xx] - altezze[z, x])
        if s > salita:
            migliore, salita = (dx, dz), s
    return migliore


def _pianifica_rampa(x: int, z: int, direzione: tuple[int, int], altezze: np.ndarray,
                     acqua: np.ndarray, mare: np.ndarray, evita: np.ndarray | None,
                     rng: np.random.Generator) -> Rampa | None:
    """La rampa che parte dal portale in `(x, z)` e scende nella direzione data.

    Cella per cella, finche' non ha la profondita' voluta: si ferma (e si
    accorcia) dove la galleria incontrerebbe acqua, uscirebbe dal bordo della
    mappa o non avrebbe abbastanza terra sopra. Le prime celle possono avere
    poca copertura - sono la trincea d'ingresso - ma non dentro un lotto o una
    strada (`evita`). Ritorna None se la rampa viene troppo corta per essere
    una discesa graduale.
    """
    H, W = altezze.shape
    dx, dz = direzione
    y0 = int(altezze[z, x]) - 1                  # pavimento al livello del terreno
    if evita is not None:
        # il portale, la piazzola, il binario che lo precede e la collina
        # dell'imbocco (larga LARGHEZZA_IMBOCCO per parte, lunga
        # LUNGHEZZA_IMBOCCO verso l'interno): niente strutture, campi
        # compresi, con un giro di margine
        for j in range(-BINARIO_FUORI - 2, LUNGHEZZA_IMBOCCO + 2):
            for k in range(-LARGHEZZA_IMBOCCO - 2, LARGHEZZA_IMBOCCO + 3):
                lx, lz = x + dx * j - dz * k, z + dz * j + dx * k
                if not (0 <= lx < W and 0 <= lz < H) or evita[lz, lx]:
                    return None
    profondita = int(rng.integers(*PROFONDITA_RAMPA))
    massima = profondita * PASSO_RAMPA
    ultima = -1
    for i in range(0, massima + 1):
        cx, cz = x + dx * i, z + dz * i
        yi = y0 - i // PASSO_RAMPA
        if yi < QUOTA_MINIMA_RAMPA:
            break
        coperto = True
        for k in (-1, 0, 1):
            lx, lz = cx - dz * k, cz + dx * k      # lato perpendicolare
            if not (2 <= lx < W - 2 and 2 <= lz < H - 2):
                coperto = None
                break
            if acqua[lz, lx] or mare[lz, lx]:
                coperto = None
                break
            copertura = altezze[lz, lx] - 1 - (yi + ALTEZZA_GALLERIA)
            if copertura < COPERTURA_RAMPA:
                coperto = False
        if coperto is None:
            break
        if not coperto:
            # trincea aperta: ammessa solo in apertura, e fuori dalle strutture
            if i >= 22:
                break
            if evita is not None and any(
                    evita[cz + dx * k, cx - dz * k] for k in (-1, 0, 1)):
                break
        ultima = i
    if ultima < LUNGHEZZA_RAMPA_MIN:
        return None
    # la rampa deve finire coperta: l'ultima cella deve avere terra sopra
    while ultima >= LUNGHEZZA_RAMPA_MIN:
        cx, cz = x + dx * ultima, z + dz * ultima
        yi = y0 - ultima // PASSO_RAMPA
        if altezze[cz, cx] - 1 - (yi + ALTEZZA_GALLERIA) >= CAPPELLO:
            break
        ultima -= 1
    if ultima < LUNGHEZZA_RAMPA_MIN:
        return None
    return Rampa(x=x, z=z, dx=dx, dz=dz, y=y0, lunghezza=ultima)


def _rampa_interna(x: int, z: int, direzione: tuple[int, int], y: int,
                   altezze: np.ndarray, acqua: np.ndarray,
                   rng: np.random.Generator) -> Rampa | None:
    """Una rampa fra due piani: parte dalla fine di un braccio e scende ancora,
    tutta sotto terra (stesso cappello delle gallerie). None se non c'e' posto
    per almeno sette blocchi di discesa."""
    H, W = altezze.shape
    dx, dz = direzione
    massima = int(rng.integers(*PROFONDITA_RAMPA_INTERNA)) * PASSO_RAMPA
    ultima = -1
    for i in range(0, massima + 1):
        cx, cz = x + dx * i, z + dz * i
        yi = y - i // PASSO_RAMPA
        if yi < QUOTA_MINIMA_RAMPA:
            break
        ok = True
        for k in (-1, 0, 1):
            lx, lz = cx - dz * k, cz + dx * k
            if not (2 <= lx < W - 2 and 2 <= lz < H - 2) or acqua[lz, lx]:
                ok = False
                break
            if altezze[lz, lx] - 1 - (yi + ALTEZZA_GALLERIA) < CAPPELLO:
                ok = False
                break
        if not ok:
            break
        ultima = i
    if ultima < 7 * PASSO_RAMPA:
        return None
    return Rampa(x=x, z=z, dx=dx, dz=dz, y=y, lunghezza=ultima)


def _scendi_ancora(mn: Miniera, altezze: np.ndarray, acqua: np.ndarray,
                   rng: np.random.Generator) -> bool:
    """Da un braccio dell'ultimo piano parte una rampa verso il piano sotto: si
    prende il giacimento in fondo a un braccio, lo si toglie (la rampa passa di
    li') e si scava da quel punto in giu'. Poi i bracci del nuovo piano.
    Ritorna True se ha aggiunto un piano."""
    ultimo = mn.y_fondo
    candidati = []
    for g in mn.giacimenti:
        if g.y != ultimo:
            continue
        for s in mn.segmenti:
            if s.y != ultimo:
                continue
            dx, dz = _asse(s)
            if (s.x1 + dx, s.z1 + dz) == (g.x, g.z):
                candidati.append((g, (dx, dz)))
                break
    if not candidati:
        return False
    # si prova da ogni fine di braccio, in ordine sparso, finche' una ha posto
    for k in rng.permutation(len(candidati)):
        g, d = candidati[int(k)]
        rampa = _rampa_interna(g.x, g.z, d, ultimo, altezze, acqua, rng)
        if rampa is None:
            continue
        fx, fz, fy = rampa.fine
        segmenti, giacimenti = _galleria(fx, fz, fy, altezze, acqua, rng,
                                         escludi=((-d[0], -d[1]),),
                                         y_superficie=mn.y_superficie + 1)
        if not segmenti:
            continue
        mn.giacimenti.remove(g)
        mn.rampe_interne.append(rampa)
        mn.segmenti.extend(segmenti)
        mn.giacimenti.extend(giacimenti)
        mn.y_fondo = fy
        return True
    return False


def pianifica(altezze: np.ndarray, mare: np.ndarray, acqua: np.ndarray | None = None,
              evita: np.ndarray | None = None, densita: float = 1.0,
              seed: int = 0, celle_per: int = 45_000,
              distanza_min: int = DISTANZA_FRA_MINIERE) -> list[Miniera]:
    """Sceglie dove mettere i portali e scava rampa e gallerie da ciascuno.

    `acqua` e' la maschera di TUTTA l'acqua (mare, fiumi, laghi): le
    gallerie non devono passare sotto nessuna delle tre. `evita` e'
    opzionale - lotti, mura, strade: un portale in mezzo a una casa si
    vede, anche se la miniera stessa non ha altri legami con gli
    insediamenti (sono sparse a caso sulla mappa, non vicino ai paesi).

    Sono poche e lontane (`celle_per`, `distanza_min`): una miniera e' un
    luogo che si trova, non qualcosa che si inciampa ogni dieci passi.
    """
    H, W = altezze.shape
    if acqua is None:
        acqua = mare
    rng = np.random.default_rng(seed)
    quanti = int(max(0, round(H * W / celle_per * densita)))
    fuori: list[Miniera] = []
    tentativi = 0
    limite = quanti * 40 + 60
    while len(fuori) < quanti and tentativi < limite:
        tentativi += 1
        z = int(rng.integers(20, H - 20))
        x = int(rng.integers(20, W - 20))
        if mare[z, x] or acqua[z, x] or (evita is not None and evita[z, x]):
            continue
        if any((m.x - x) ** 2 + (m.z - z) ** 2 < distanza_min ** 2 for m in fuori):
            continue
        direzione = _direzione_in_salita(altezze, x, z, rng)
        rampa = _pianifica_rampa(x, z, direzione, altezze, acqua, mare, evita, rng)
        if rampa is None:
            continue
        fx, fz, fy = rampa.fine
        segmenti, giacimenti = _galleria(fx, fz, fy, altezze, acqua, rng,
                                         escludi=((-direzione[0], -direzione[1]),),
                                         y_superficie=rampa.y + 1)
        if not segmenti:
            continue
        mn = Miniera(x=x, z=z, y_superficie=rampa.y, y_fondo=fy, rampa=rampa,
                     segmenti=segmenti, giacimenti=giacimenti)
        # i piani sotto: finche' c'e' posto e la sorte vuole, si scende ancora
        for _ in range(LIVELLI_MAX - 1):
            if rng.random() > PROB_LIVELLO_SOTTO or not _scendi_ancora(mn, altezze, acqua, rng):
                break
        fuori.append(mn)
    return fuori


def _celle_chunk(x0: int, x1: int, z0: int, z1: int, m: int, passo: int) -> set:
    return {(cx, cz) for cz in range((z0 - m) // passo, (z1 + m) // passo + 1)
            for cx in range((x0 - m) // passo, (x1 + m) // passo + 1)}


RAGGIO_PORTALE = 3      # il portale e' largo 7 (3 + 1 + 3) e l'apron lo precede


def indice_per_chunk(miniere: list[Miniera], passo: int = 16) -> dict:
    """Ogni miniera va registrata in tutti i chunk che portale, rampa e
    gallerie toccano - stessa idea di `insediamenti.indice_per_chunk`."""
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, mn in enumerate(miniere):
        celle: set = set()
        r = mn.rampa
        if r is not None:
            # il portale e il binario all'aperto che lo precede
            ax, az, _ = r.cella(-BINARIO_FUORI - 2)
            celle |= _celle_chunk(min(ax, r.x), max(ax, r.x), min(az, r.z),
                                  max(az, r.z), RAGGIO_PORTALE + 1, passo)
            for k in range(0, r.lunghezza + 1):
                cx, cz, _ = r.cella(k)
                celle |= _celle_chunk(cx, cx, cz, cz, LARGHEZZA + 2, passo)
        for ri in mn.rampe_interne:
            for k in range(0, ri.lunghezza + 1):
                cx, cz, _ = ri.cella(k)
                celle |= _celle_chunk(cx, cx, cz, cz, LARGHEZZA + 2, passo)
        for s in mn.segmenti:
            x0, x1 = sorted((s.x0, s.x1))
            z0, z1 = sorted((s.z0, s.z1))
            celle |= _celle_chunk(x0, x1, z0, z1, LARGHEZZA + 2, passo)
        # margine del giacimento (saletta vuota + il grumo di minerale che
        # la incastona, vedi `_carica_giacimento`): senza margine una saletta a
        # cavallo di un confine di chunk si vedrebbe "smezzata".
        mg = RAGGIO_GIACIMENTO_VUOTO + 3
        for g in mn.giacimenti:
            celle |= _celle_chunk(g.x, g.x, g.z, g.z, mg, passo)
        for cella in celle:
            fuori.setdefault(cella, []).append(i)
    return fuori


def maschera_ingresso(miniere: list[Miniera], shape: tuple[int, int],
                      margine: int = LARGHEZZA_IMBOCCO + 1) -> np.ndarray:
    """Dove sta in superficie quello che si vede di una miniera: il portale, il
    binario che lo precede e la trincea d'ingresso. Serve a tenerci fuori gli
    alberi (un tronco dentro la trincea resterebbe a mezz'aria)."""
    H, W = shape
    m = np.zeros(shape, bool)
    for mn in miniere:
        r = mn.rampa
        if r is None:
            continue
        for i in range(-BINARIO_FUORI - 2, min(r.lunghezza, 24) + 1):
            cx, cz, _ = r.cella(i)
            m[max(0, cz - margine):min(H, cz + margine + 1),
              max(0, cx - margine):min(W, cx + margine + 1)] = True
    return m


def carrelli(miniere: list[Miniera], seed: int = 0) -> list[Carrello]:
    """Un carrello fermo davanti al portale, uno ogni tanto lungo la rampa e
    uno ogni tanto sul binario delle gallerie in fondo."""
    rng = np.random.default_rng(seed)
    fuori: list[Carrello] = []
    for i, mn in enumerate(miniere):
        r = mn.rampa
        if r is not None:
            cx, cz, cy = r.cella(-3)
            fuori.append(Carrello(x=cx + 0.5, y=float(r.y + 1), z=cz + 0.5, seme=i * 97 + 1))
            for k in range(8, r.lunghezza - 4, 14):
                if r.e_gradino(k) or r.e_gradino(k + 1) or rng.random() > 0.55:
                    continue
                cx, cz, cy = r.cella(k)
                fuori.append(Carrello(x=cx + 0.5, y=float(cy + 1), z=cz + 0.5,
                                      seme=i * 97 + 10 + k))
        for ri in mn.rampe_interne:
            for k in range(6, ri.lunghezza - 4, 16):
                if ri.e_gradino(k) or ri.e_gradino(k + 1) or rng.random() > 0.5:
                    continue
                cx, cz, cy = ri.cella(k)
                fuori.append(Carrello(x=cx + 0.5, y=float(cy + 1), z=cz + 0.5,
                                      seme=i * 97 + 500 + k))
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
    rampe = [mn.rampa.lunghezza for mn in miniere if mn.rampa is not None]
    giacimenti = sum(len(mn.giacimenti) for mn in miniere)
    return {
        "miniere": len(miniere),
        "gallerie": gallerie,
        "lunghezza": int(lunghezza),
        "rampa_media": float(np.mean(rampe)) if rampe else 0.0,
        "piani": float(np.mean([1 + len(mn.rampe_interne) for mn in miniere])),
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
        self.rotaia = {forma: s.blocco("rail", shape=forma) for forma in (
            "north_south", "east_west", "north_east", "north_west",
            "south_east", "south_west",
            "ascending_north", "ascending_south", "ascending_east", "ascending_west")}
        # il binario esterno finisce con una fermata: una rotaia frenante (non
        # alimentata) e un paraurti di pietra in testa
        self.freno = {forma: s.blocco("powered_rail", shape=forma, powered="false")
                      for forma in ("east_west", "north_south")}
        self.palo = s.blocco("log", axis="y", material="oak", stripped="true")
        # le travi orizzontali sono tronchi interi, non scortecciati: si
        # distinguono dai pali verticali e danno il colore del legno grezzo
        self.trave = {"x": s.blocco("log", axis="x", material="oak", stripped="false"),
                      "z": s.blocco("log", axis="z", material="oak", stripped="false")}
        # ATTENZIONE al nome universale. Non e' "wall_torch": nel formato
        # universale di PyMCTranslate torcia a terra e torcia a muro sono
        # LO STESSO blocco ("torch"), distinti solo dalla proprieta'
        # `facing` ("up" per quella a terra). Scriverlo come "wall_torch"
        # supera il controllo di andata (si salva senza errori) ma fallisce
        # al ritorno, e in gioco la torcia non compariva.
        self.torcia = {lato: s.blocco("torch", facing=lato)
                       for lato in ("north", "south", "east", "west")}
        self.base_muro = s.blocco("stone_bricks", variant="normal")
        self.muretto = s.blocco("wall", material="cobblestone", up="true", north="none",
                                south="none", east="none", west="none")
        self.ghiaia = s.blocco("gravel")
        self.lastra = s.blocco("slab", material="oak", type="bottom")
        self.lanterna = s.blocco("lantern", hanging="true", waterlogged="false")
        # la roccia attorno all'imbocco: pietra con un po' di variazione
        self.roccia = [s.blocco("stone"), s.blocco("stone"), s.blocco("cobblestone"),
                       s.blocco("andesite"), s.blocco("mossy_cobblestone")]
        # il resto della collina: terra con l'erba in cima
        self.terra = s.blocco("dirt")
        self.erba = s.blocco("grass_block")
        self.assi = s.blocco("planks", material="oak")
        self.cassa = {v: s.blocco("chest", facing=v, type="single", waterlogged="false")
                      for v in ("north", "south", "east", "west")}
        self.falda = {v: s.blocco("stairs", facing=v, half="bottom", shape="straight",
                                  material="oak", waterlogged="false")
                      for v in ("north", "south", "east", "west")}
        # --- arredo delle gallerie: attrezzi, ragnatele, bauli, gemme ---
        self.barile = s.blocco("barrel", facing="up", open="false")
        self.tavolo = s.blocco("crafting_table")
        self.fabbro = s.blocco("smithing_table")
        self.incudine = {v: s.blocco("anvil", facing=v) for v in ("north", "south", "east", "west")}
        self.mola = {v: s.blocco("grindstone", face="floor", facing=v)
                     for v in ("north", "south", "east", "west")}
        self.ragnatela = s.blocco("cobweb")
        self.ametista = s.blocco("amethyst_cluster", facing="up")
        self.germoglio = s.blocco("small_amethyst_bud", facing="up")
        self.grezzo = [s.blocco("raw_iron_block"), s.blocco("raw_copper_block"),
                       s.blocco("raw_gold_block"), s.blocco("coal_block")]
        self.forziere = {v: s.blocco("chest", facing=v, connection="none", material="wood")
                         for v in ("north", "south", "east", "west")}
        self.decora = True            # spento solo dai test che misurano la sezione nuda
        # posizioni (coordinate di MAPPA) dei forzieri messi nel chunk appena
        # scritto: chi scrive il chunk le legge e le svuota (`bauli.trova`)
        self.bauli: list[tuple[int, int, int]] = []
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


_VERSO_SALITA = {(1, 0): "ascending_west", (-1, 0): "ascending_east",
                 (0, 1): "ascending_north", (0, -1): "ascending_south"}


def _arreda_lato(out, tav, rng, lx: int, lz: int, y_piano: int, verso: str,
                 ox: int, oz: int, y0: int) -> None:
    """Un oggetto di mestiere sul bordo della galleria, a fianco del binario.

    `y_piano` e' lo strato dell'aria sopra il pavimento della cella laterale
    (lx, lz), `verso` la direzione verso il centro della galleria (dove guarda
    un forziere o un'incudine). Si sceglie a caso: un barile, un banco da
    lavoro, un'incudine, una mola, un mucchio di minerale grezzo, un
    forziere col bottino, un cristallo di ametista; e ogni tanto una
    ragnatela nell'angolo del soffitto. Mai sopra qualcosa che c'e' gia'.
    """
    H = out.shape[1]
    if not tav.decora or not (0 <= lx < 16 and 0 <= lz < 16 and 1 <= y_piano + 2 < H):
        return
    r = rng.random()
    if r < 0.12:
        if out[lx, y_piano, lz] == tav.aria:
            out[lx, y_piano, lz] = tav.barile
    elif r < 0.17:
        if out[lx, y_piano, lz] == tav.aria:
            out[lx, y_piano, lz] = tav.tavolo
    elif r < 0.21:
        if out[lx, y_piano, lz] == tav.aria:
            out[lx, y_piano, lz] = tav.incudine[verso]
    elif r < 0.26:
        if out[lx, y_piano, lz] == tav.aria:
            out[lx, y_piano, lz] = tav.mola[verso]
    elif r < 0.30:
        if out[lx, y_piano, lz] == tav.aria:
            out[lx, y_piano, lz] = tav.fabbro
    elif r < 0.37:
        if out[lx, y_piano, lz] == tav.aria:
            out[lx, y_piano, lz] = tav.grezzo[int(rng.integers(0, len(tav.grezzo)))]
            if rng.random() < 0.4 and out[lx, y_piano + 1, lz] == tav.aria:
                out[lx, y_piano + 1, lz] = tav.grezzo[int(rng.integers(0, len(tav.grezzo)))]
    elif r < 0.42:
        if out[lx, y_piano, lz] == tav.aria:
            out[lx, y_piano, lz] = tav.forziere[verso]
            tav.bauli.append((ox + lx, y0 + y_piano, oz + lz))
    elif r < 0.48:
        if out[lx, y_piano, lz] == tav.aria:
            out[lx, y_piano, lz] = tav.ametista if rng.random() < 0.5 else tav.germoglio
    # la ragnatela sta in alto, dove non intralcia chi cammina: il soffitto e'
    # a y_piano + 2 (3 celle di luce), la ragnatela nell'ultima
    if rng.random() < 0.12 and out[lx, y_piano + 2, lz] == tav.aria:
        out[lx, y_piano + 2, lz] = tav.ragnatela


def _carica_rampa(out, tav, rng, h_c, r: Rampa, y_superficie: int, ox: int, oz: int,
                  y0: int, scavabile: tuple, protetto_c=None) -> None:
    """La rampa: si scava cella per cella (tre di larghezza, tre di altezza),
    dove la terra sopra e' poca si apre in trincea, il pavimento e' sempre
    pieno, il binario segue i gradini con una rotaia in pendenza sulla prima
    cella dopo ognuno, e ogni quattro celle un telaio di travi regge il
    soffitto.

    Si cammina lungo le celle della rampa (non su tutto il chunk): un chunk
    ne tocca poche decine.
    """
    H = out.shape[1]
    lat = (-r.dz, r.dx)                           # il lato, perpendicolare alla rampa
    asse_trave = "x" if r.dx == 0 else "z"        # la trave corre sul lato
    for i in range(0, r.lunghezza + 1):
        cx, cz, fy = r.cella(i)
        ly_f = fy - y0
        # il primo telaio e' il portale stesso: niente pali dentro l'apertura
        telaio = i > 0 and i % PASSO_TRAVE_RAMPA == 0
        for k in (-1, 0, 1):
            wx, wz = cx + lat[0] * k, cz + lat[1] * k
            lx, lz = wx - ox, wz - oz
            if not (0 <= lx < 16 and 0 <= lz < 16):
                continue
            if protetto_c is not None and protetto_c[lx, lz]:
                continue
            sup = int(h_c[lx, lz])                 # primo blocco libero sopra il terreno
            # pavimento: sempre pieno. Dove il terreno sta piu' in basso (il
            # davanti di una collina) si riempie di ghiaia fino a terra.
            if 1 <= ly_f < H:
                if out[lx, ly_f, lz] == tav.aria:
                    for d in range(0, 8):
                        if not (1 <= ly_f - d < H) or out[lx, ly_f - d, lz] != tav.aria:
                            break
                        out[lx, ly_f - d, lz] = tav.ghiaia
            # vuoto: tre celle, o fino alla superficie se la terra sopra e' poca
            alto = max(fy + ALTEZZA_GALLERIA, sup - 1)
            aperta = sup - 1 - (fy + ALTEZZA_GALLERIA) < COPERTURA_RAMPA
            fino = alto if aperta else fy + ALTEZZA_GALLERIA
            for y in range(fy + 1, fino + 1):
                ly = y - y0
                if 1 <= ly < H:
                    out[lx, ly, lz] = tav.aria
            if k != 0:
                if telaio and 1 <= ly_f + 1 < H:
                    for y in range(fy + 1, fy + ALTEZZA_GALLERIA):
                        ly = y - y0
                        if 1 <= ly < H:
                            out[lx, ly, lz] = tav.palo
                elif i > 3 and not (i % PASSO_TRAVE_RAMPA in (1, PASSO_TRAVE_RAMPA - 1))                         and rng.random() < 0.15:
                    vx, vz = -lat[0] * k, -lat[1] * k          # verso il centro
                    verso = (("east" if vx > 0 else "west") if vx
                             else ("south" if vz > 0 else "north"))
                    _arreda_lato(out, tav, rng, lx, lz, ly_f + 1, verso, ox, oz, y0)
                continue
            # il binario e le travi sul centro
            forma = "east_west" if r.dz == 0 else "north_south"
            if r.e_gradino(i):
                forma = _VERSO_SALITA[(r.dx, r.dz)]
            if 1 <= ly_f + 1 < H:
                out[lx, ly_f + 1, lz] = tav.rotaia[forma]
            ly_t = fy + ALTEZZA_GALLERIA - y0
            if telaio and 1 <= ly_t < H:
                out[lx, ly_t, lz] = tav.trave[asse_trave]
                # una lanterna ogni due telai, appesa sotto la trave
                if i % (PASSO_TRAVE_RAMPA * 2) == PASSO_TRAVE_RAMPA and 1 <= ly_t - 1 < H:
                    out[lx, ly_t - 1, lz] = tav.lanterna
        if telaio:
            # la trave si completa sui due lati (il centro e' gia' stato fatto)
            for k in (-1, 1):
                wx, wz = cx + lat[0] * k, cz + lat[1] * k
                lx, lz = wx - ox, wz - oz
                if not (0 <= lx < 16 and 0 <= lz < 16):
                    continue
                if protetto_c is not None and protetto_c[lx, lz]:
                    continue
                ly_t = fy + ALTEZZA_GALLERIA - y0
                if 1 <= ly_t < H:
                    out[lx, ly_t, lz] = tav.trave[asse_trave]
        # un filone ogni tanto sulla parete, piu' ricco quanto piu' si scende
        if i > 10 and rng.random() < 0.07:
            prof = _profondita_relativa(fy, y_superficie + 1)
            nome = _scegli_minerale(fy, rng, prof)
            m = _MINERALE_PER_NOME.get(nome) if nome else None
            if m is not None:
                lato = 1 if rng.random() < 0.5 else -1
                gx = cx + lat[0] * (LARGHEZZA + 2) * lato - ox
                gz = cz + lat[1] * (LARGHEZZA + 2) * lato - oz
                profondo = fy < QUOTA_ARDESIA - FASCIA_ARDESIA // 2
                quanti = int(rng.integers(m.grumo[0], m.grumo[1] + 1))
                grumo(out, gx, gz, fy + int(rng.integers(0, 2)), y0, h_c,
                      tav.minerale[(m.nome, profondo)], quanti, scavabile,
                      protetto=protetto_c)


def _carica_portale(out, tav, h_c, mn: Miniera, ox: int, oz: int, y0: int,
                    protetto_c=None) -> None:
    """Il portale: due pali di tronco e un architrave, i muretti di pietra ai
    lati, un tettuccio, le lanterne. Davanti, una piazzola piana col binario
    che esce all'aperto (il carrello fermo lo mette `carrelli()`).

    Come la rampa lavora sulle celle, non sul chunk intero: il portale e'
    largo sette e la piazzola lunga sei, quindi puo' stare a cavallo di due
    chunk.
    """
    r = mn.rampa
    if r is None:
        return
    H = out.shape[1]
    lat = (-r.dz, r.dx)
    asse_trave = "x" if r.dx == 0 else "z"
    fy = r.y

    def posa(wx: int, wz: int, y: int, blocco: int, solo_aria: bool = False) -> None:
        lx, lz, ly = wx - ox, wz - oz, y - y0
        if not (0 <= lx < 16 and 0 <= lz < 16 and 1 <= ly < H):
            return
        if protetto_c is not None and protetto_c[lx, lz]:
            return
        if solo_aria and out[lx, ly, lz] != tav.aria:
            return
        out[lx, ly, lz] = blocco

    # il portale a i = 0
    for k in range(-RAGGIO_PORTALE, RAGGIO_PORTALE + 1):
        wx, wz = r.x + lat[0] * k, r.z + lat[1] * k
        if abs(k) == 2:
            for y in range(fy + 1, fy + 5):
                posa(wx, wz, y, tav.palo)                    # i due pali
        if abs(k) <= 2:
            posa(wx, wz, fy + 4, tav.trave[asse_trave])      # architrave
            posa(wx, wz, fy + 5, tav.lastra)                 # tettuccio
        if abs(k) == 3:
            for y in range(fy, fy + 3):
                posa(wx, wz, y, tav.base_muro)               # i muretti laterali
            posa(wx, wz, fy + 3, tav.muretto)
        if abs(k) == 1:
            posa(wx, wz, fy + 3, tav.lanterna)               # sotto l'architrave

    def superficie(wx: int, wz: int, k: int, i: int) -> tuple[int, int]:
        """(blocco in cima, blocco sotto) del territorio: erba e terra nella
        foresta e in pianura, sabbia e arenaria nel deserto, neve, pietra...

        Si legge cio' che il chunk ha gia' scritto sulla superficie. Dentro la
        trincea della rampa quel blocco e' stato scavato: si prende quello di
        una colonna piu' laterale, alla stessa distanza dal portale.
        """
        candidati = ([k] if abs(k) > 1 else []) + [2, -2, 3, -3, 4, -4, 5, -5]
        for kk in candidati:
            ex, ez = r.x + r.dx * i + lat[0] * kk, r.z + r.dz * i + lat[1] * kk
            lx, lz = ex - ox, ez - oz
            if not (0 <= lx < 16 and 0 <= lz < 16):
                continue
            ly = int(h_c[lx, lz]) - 1 - y0
            if not (2 <= ly < H):
                continue
            alto, basso_ = out[lx, ly, lz], out[lx, ly - 1, lz]
            if alto != tav.aria and basso_ != tav.aria and alto != tav.ghiaia:
                return int(alto), int(basso_)
        return tav.erba, tav.terra

    # l'imbocco: il portale non sta in mezzo alla pianura ma nella roccia
    for i in range(0, LUNGHEZZA_IMBOCCO + 1):
        cx, cz, fy_i = r.cella(i)
        for k in range(-LARGHEZZA_IMBOCCO, LARGHEZZA_IMBOCCO + 1):
            wx, wz = cx + lat[0] * k, cz + lat[1] * k
            lx, lz = wx - ox, wz - oz
            if not (0 <= lx < 16 and 0 <= lz < 16):
                continue
            a = abs(k)
            if a == 2 and i == 0:
                continue                                  # i pali del portale
            cima = r.y + ALTEZZA_IMBOCCO - i // 2 - max(0, a - 1)
            basso = fy_i + ALTEZZA_GALLERIA + 1 if a <= 1 else fy_i
            if i == 0 and a <= 2:
                basso = fy + 6                            # sopra architrave e tettuccio
            hv = ((wx * 73856093) ^ (wz * 19349663)) & 0x7fffffff
            sopra, sotto = superficie(wx, wz, k, i)
            for y in range(basso, cima + 1):
                # la facciata e i fianchi bassi sono roccia viva; sopra, la
                # collina, fatta di quello che c'e' in quel territorio
                facciata = a >= 2 and (i <= 2 or y <= r.y + 2) and y <= r.y + 3 + (3 - a)
                if facciata:
                    # nel deserto la roccia e' arenaria, in montagna pietra
                    b = (sotto if sotto != tav.terra
                         else tav.roccia[(hv + y * 31) % len(tav.roccia)])
                else:
                    b = sopra if y == cima else sotto
                posa(wx, wz, y, b, solo_aria=True)

    # il portico di legno: tettuccio a due falde sopra l'architrave, e una
    # cassa a lato davanti all'ingresso
    def verso(vx: int, vz: int) -> str:
        return ("east" if vx > 0 else "west") if vx else ("south" if vz > 0 else "north")

    # (le scale salgono verso il lato indicato: le falde salgono verso il colmo)
    for k in range(-2, 3):
        wx, wz = r.x + lat[0] * k, r.z + lat[1] * k
        if abs(k) == 2:
            sgn = 1 if k < 0 else -1
            posa(wx, wz, fy + 5, tav.falda[verso(lat[0] * sgn, lat[1] * sgn)])
        else:
            posa(wx, wz, fy + 5, tav.assi)
    # la cassa sta di lato, un paio di passi fuori, rivolta verso la strada
    posa(r.x - lat[0] * 3 - r.dx * 2, r.z - lat[1] * 3 - r.dz * 2, fy + 1,
         tav.cassa[verso(-r.dx, -r.dz)])

    # la piazzola e il binario all'aperto, davanti al portale
    for j in range(1, BINARIO_FUORI + 1):
        cx, cz = r.x - r.dx * j, r.z - r.dz * j
        for k in (-2, -1, 0, 1, 2):
            wx, wz = cx + lat[0] * k, cz + lat[1] * k
            lx, lz = wx - ox, wz - oz
            if not (0 <= lx < 16 and 0 <= lz < 16):
                continue
            if protetto_c is not None and protetto_c[lx, lz]:
                continue
            sup = int(h_c[lx, lz])
            # La piazzola e' del TERRITORIO, non di ghiaia: sabbia nel deserto,
            # erba nella foresta, neve sulla neve. Il blocco di superficie e quello
            # sotto si leggono prima di toccare la colonna.
            sopra, sotto = superficie(wx, wz, k, -j)
            # si porta il suolo alla quota del pavimento: riempiendo se sta
            # piu' in basso, scavando se piu' in alto
            for y in range(max(1 + y0, fy - 6), fy + 1):
                ly = y - y0
                if 1 <= ly < H and out[lx, ly, lz] == tav.aria:
                    out[lx, ly, lz] = sotto
            ly_f = fy - y0
            if 1 <= ly_f < H:
                out[lx, ly_f, lz] = sopra
            for y in range(fy + 1, max(fy + 5, sup)):
                ly = y - y0
                if 1 <= ly < H:
                    out[lx, ly, lz] = tav.aria
            if k == 0 and 1 <= ly_f + 1 < H:
                forma = "east_west" if r.dz == 0 else "north_south"
                if j == BINARIO_FUORI:
                    out[lx, ly_f + 1, lz] = tav.muretto           # il paraurti, in testa
                elif j == BINARIO_FUORI - 1:
                    out[lx, ly_f + 1, lz] = tav.freno[forma]      # la fermata
                else:
                    out[lx, ly_f + 1, lz] = tav.rotaia[forma]


def _carica_segmento(out, tav, rng, h_c, seg: Segmento, ox: int, oz: int,
                     y0: int, scavabile: tuple, protetto_c=None,
                     profondita: float = 0.5) -> None:
    H = out.shape[1]
    dx, dz = _asse(seg)
    lungo_x = dz == 0
    x0, x1 = sorted((seg.x0, seg.x1))
    z0, z1 = sorted((seg.z0, seg.z1))
    fy = seg.y - y0
    a1, a2, a3 = fy + 1, fy + 2, fy + 3      # tre di luce: si cammina senza chinarsi

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
            for ly in (a1, a2, a3):
                if 1 <= ly < H:
                    out[lx, ly, lz] = tav.aria
            if scarto != 0:
                # il bordo della galleria: ogni tanto un attrezzo, un barile,
                # un forziere, una ragnatela - mai dove ci sono le travi
                lungo = (wx - seg.x0) if lungo_x else (wz - seg.z0)
                if lungo % PASSO_TRAVE not in (0, 1, PASSO_TRAVE - 1) and rng.random() < 0.18:
                    if lungo_x:
                        verso = "north" if scarto > 0 else "south"
                    else:
                        verso = "west" if scarto > 0 else "east"
                    _arreda_lato(out, tav, rng, lx, lz, a1, verso, ox, oz, y0)
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
                        for ly in (a1, a2, a3):
                            if 0 <= ly < H:
                                out[plx, ly, plz] = tav.palo
                if avanzamento % (PASSO_TRAVE * 2) == 0 and 0 <= a2 < H:
                    lato = "east" if lungo_x else "south"
                    out[lx, a2, lz] = tav.torcia[lato]

            if rng.random() < 0.10:
                nome = _scegli_minerale(seg.y, rng, profondita)
                m = _MINERALE_PER_NOME.get(nome) if nome else None
                if m is not None:
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
    """Portali, rampe, gallerie, binari, travi, filoni e giacimenti dentro il
    chunk (16, H, 16).

    `scavabile` e' la stessa idea di `sottosuolo.Tavolozza.scavabile`: solo
    quello che e' gia' pietra (o ardesia, o uno strato) prende un filone -
    un filone non deve mai bucare l'aria di una galleria o il pavimento di
    una casa capitata sopra per caso.

    `protetto_c` (16x16, opzionale) e' la stessa maschera di
    `sottosuolo.posa()`: un portale non puo' nascere dentro un insediamento
    (`evita` in `pianifica()`), ma una galleria puo' comunque attraversarne
    uno durante il percorso - qui si evita di scavare aria sotto cio' che e'
    gia' disegnato in superficie.
    """
    for i in quali:
        mn = miniere[i]
        rng = np.random.default_rng(
            ((mn.x * 73856093) ^ (mn.z * 19349663) ^ (seed * 83492791)) & 0x7FFFFFFF)
        prof = _profondita_relativa(mn.y_fondo, mn.y_superficie + 1)
        if mn.rampa is not None:
            _carica_rampa(out, tav, rng, h_c, mn.rampa, mn.y_superficie, ox, oz, y0,
                          scavabile, protetto_c)
        for ri in mn.rampe_interne:
            _carica_rampa(out, tav, rng, h_c, ri, mn.y_superficie, ox, oz, y0,
                          scavabile, protetto_c)
        _carica_portale(out, tav, h_c, mn, ox, oz, y0, protetto_c)
        for seg in mn.segmenti:
            _carica_segmento(out, tav, rng, h_c, seg, ox, oz, y0, scavabile,
                             protetto_c, prof)
        for g in mn.giacimenti:
            _carica_giacimento(out, tav, h_c, rng, g, ox, oz, y0, scavabile,
                               protetto_c)
