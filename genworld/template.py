"""Case da template: file `.nbt` dei blocchi struttura di Minecraft.

Il generatore parametrico di `edifici.py` fa case corrette e anonime. Corrette
perche' non sbagliano mai un tetto e non restano mai appese in aria; anonime
perche' una funzione di cinque parametri produce cinque parametri di varieta',
e cinquanta case in fila si riconoscono come cinquanta volte la stessa casa.

Questo modulo prende la strada opposta: la casa la disegna una persona, dentro
Minecraft, e il programma si limita a posarla. Il formato e' quello che
Minecraft stesso scrive con il **blocco struttura** - `.nbt`, gzip, con
`size`, `palette` e `blocks` - quindi non c'e' niente da imparare e niente da
convertire: si costruisce, si salva, si butta il file in `templates/case/`.

Tre cose non ovvie, che sono poi il grosso del modulo.

**La traduzione.** Nel file i blocchi hanno i nomi di gioco
(`minecraft:oak_stairs`); il mondo che scriviamo parla il namespace
universale di amulet. Fra i due c'e' PyMCTranslate, ed e' la stessa trappola
di `oak_log` vista da dentro: un nome non tradotto si scrive benissimo e in
gioco non c'e'. Qui pero' la traduzione e' obbligata dal formato, quindi la
si fa una volta sola al caricamento e si tengono gli id di palette.

**La rotazione.** Una casa va girata verso la strada, e girare una struttura
non e' girare un array: un tronco con `axis=x` diventa `axis=z`, una scala
che guarda a nord guarda a est, uno steccato collegato a ovest si collega a
nord. Le proprieta' che portano una direzione vanno ruotate insieme ai
blocchi, altrimenti si ottiene una casa giusta fatta di pezzi storti.

**Il vuoto.** Una struttura salva anche l'aria, ed e' giusto che la riposi:
serve a svuotare la stanza. Ma `structure_void` vuol dire il contrario -
"qui non toccare niente" - e va saltato, altrimenti il blocco struttura non
serve a niente quando si vuole una casa di pianta non rettangolare.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

# Rotazione di 90 gradi in senso orario visti dall'alto:
#   (x, z) -> (dz - 1 - z, x)
# che manda est in sud, sud in ovest, ovest in nord, nord in est.
GIRA = {"north": "east", "east": "south", "south": "west", "west": "north"}
GIRA_ASSE = {"x": "z", "z": "x", "y": "y"}
# I quattro booleani di collegamento di steccati, muretti e vetrate.
LATI = ("north", "east", "south", "west")


def ruota_proprieta(prop: dict, quarti: int) -> dict:
    """Le proprieta' di un blocco girate di `quarti` quarti di giro."""
    quarti %= 4
    if quarti == 0:
        return dict(prop)
    fuori = dict(prop)
    for _ in range(quarti):
        passo = dict(fuori)
        if "facing" in fuori and fuori["facing"] in GIRA:
            passo["facing"] = GIRA[fuori["facing"]]
        if "axis" in fuori and fuori["axis"] in GIRA_ASSE:
            passo["axis"] = GIRA_ASSE[fuori["axis"]]
        if any(l in fuori for l in LATI):
            for l in LATI:
                if l in fuori:
                    # il valore che stava a nord finisce a est
                    passo[GIRA[l]] = fuori[l]
        if "rotation" in fuori:
            try:
                passo["rotation"] = str((int(fuori["rotation"]) + 4) % 16)
            except ValueError:
                pass
        fuori = passo
    return fuori


@dataclass
class Modello:
    """Una casa letta da un `.nbt`, gia' tradotta e pronta da posare."""
    nome: str
    # indici nella tavolozza del modello, forma (dx, dy, dz); -1 = non toccare
    celle: np.ndarray
    # (nome universale, proprieta') per ogni indice
    tavolozza: list[tuple[str, dict]]
    porta: int | None = None          # verso della porta nel modello (0..3)
    # In quali stili (bosco/montagna/deserto/prato) ha senso comparire.
    # Vuoto = nessun vincolo: compare ovunque. E' cosi' che un file buttato
    # dentro la cartella senza toccare nient'altro funziona subito, come
    # promette il README - il vincolo e' un'aggiunta esplicita, non un
    # obbligo.
    stili: frozenset[str] = field(default_factory=frozenset)

    @property
    def dx(self) -> int: return int(self.celle.shape[0])
    @property
    def dy(self) -> int: return int(self.celle.shape[1])
    @property
    def dz(self) -> int: return int(self.celle.shape[2])

    def ingombro(self, quarti: int) -> tuple[int, int]:
        return (self.dz, self.dx) if quarti % 2 else (self.dx, self.dz)


# --------------------------------------------------------------------------
# Lettura
# --------------------------------------------------------------------------

NORD, EST, SUD, OVEST = range(4)
_VERSO = {"north": NORD, "east": EST, "south": SUD, "west": OVEST}


def carica(percorso: str, versione=(1, 21, 4)) -> Modello:
    """Legge un `.nbt` di blocco struttura e lo traduce al namespace universale."""
    import PyMCTranslate
    from amulet.api.block import Block
    from amulet_nbt import load as nbt_load

    radice = nbt_load(percorso).compound
    dim = [int(v) for v in radice["size"]]
    if len(dim) != 3 or min(dim) <= 0:
        raise ValueError(f"{percorso}: dimensione non valida {dim}")
    dx, dy, dz = dim

    tavola = radice.get("palette")
    if tavola is None:
        tavole = radice.get("palettes")
        if tavole is None or len(tavole) == 0:
            raise ValueError(f"{percorso}: manca la palette")
        tavola = tavole[0]

    ver = PyMCTranslate.new_translation_manager().get_version("java", versione)

    tavolozza: list[tuple[str, dict]] = []
    vuoto: set[int] = set()
    for i, voce in enumerate(tavola):
        nome = str(voce["Name"])
        prop = {k: str(v) for k, v in dict(voce.get("Properties", {})).items()}
        if nome.endswith("structure_void"):
            vuoto.add(i)
            tavolozza.append(("air", {}))
            continue
        spazio, _, _ = ver.block.to_universal(
            Block(*nome.split(":", 1) if ":" in nome else ("minecraft", nome),
                  _prop_nbt(prop)))
        tavolozza.append((spazio.base_name,
                          {k: str(v.py_str if hasattr(v, "py_str") else v)
                           for k, v in spazio.properties.items()}))

    celle = np.full((dx, dy, dz), -1, np.int32)
    for voce in radice.get("blocks", []):
        px, py, pz = (int(v) for v in voce["pos"])
        stato = int(voce["state"])
        if stato in vuoto:
            continue
        if 0 <= px < dx and 0 <= py < dy and 0 <= pz < dz:
            celle[px, py, pz] = stato

    m = Modello(nome=os.path.splitext(os.path.basename(percorso))[0],
                celle=celle, tavolozza=tavolozza)
    m.porta = _trova_porta(m)
    return m


def _prop_nbt(prop: dict):
    from amulet_nbt import StringTag
    return {k: StringTag(v) for k, v in prop.items()}


def traduttore(versione=(1, 21, 4)):
    """Il traduttore di PyMCTranslate per una versione Java: caricarlo costa,
    quindi lo si prende una volta e lo si passa a `traduci_blocco`."""
    import PyMCTranslate
    return PyMCTranslate.new_translation_manager().get_version("java", versione)


def traduci_blocco(ver, nome: str, prop: dict | None = None
                   ) -> tuple[str, dict, bool]:
    """Un blocco col nome di gioco (`minecraft:oak_fence`, proprieta' di
    gioco) nella forma universale che `ScrittoreMondo.blocco` si aspetta.

    Ritorna (nome universale, proprieta', tradotto). `tradotto` e' False se
    PyMCTranslate non conosce il blocco: lo restituisce tale e quale, senza un
    errore, e in gioco poi non compare - la trappola di sempre. Chi disegna
    un blocco a mano deve controllarlo.
    """
    from amulet.api.block import Block
    spazio, _, _ = ver.block.to_universal(
        Block(*(nome.split(":", 1) if ":" in nome else ("minecraft", nome)),
              _prop_nbt(prop or {})))
    proprieta = {k: str(v.py_str if hasattr(v, "py_str") else v)
                 for k, v in spazio.properties.items()}
    return spazio.base_name, proprieta, spazio.namespace.startswith("universal")


def _trova_porta(m: Modello) -> int | None:
    """Da che parte guarda la casa.

    Si cerca la porta e si guarda il suo `facing`, che in Minecraft punta
    verso l'interno: il fronte e' dalla parte opposta. Senza questo, una casa
    su due darebbe le spalle alla strada - ed e' il genere di cosa che non si
    nota guardando un render dall'alto e si nota subito camminandoci.
    """
    for i, (nome, prop) in enumerate(m.tavolozza):
        if nome != "door" or "facing" not in prop:
            continue
        if not (m.celle == i).any():
            continue
        dentro = _VERSO.get(prop["facing"])
        if dentro is None:
            continue
        return (dentro + 2) % 4
    return None


def carica_stili(percorso: str) -> dict[str, frozenset[str]]:
    """Legge `stili.json` da una cartella di template, se c'e'.

    Formato: {"nome_file_senza_estensione": ["bosco", "prato"], ...}. Un file
    non nominato nel manifesto non ha vincoli - vedi il commento su
    `Modello.stili`. Un manifesto assente o rotto non e' un errore: vuol dire
    solo che quella cartella non ha vincoli di stile.
    """
    percorso_json = os.path.join(percorso, "stili.json")
    if not os.path.isfile(percorso_json):
        return {}
    try:
        with open(percorso_json, encoding="utf-8") as f:
            grezzo = json.load(f)
        return {nome: frozenset(stili) for nome, stili in grezzo.items()}
    except Exception as guaio:                           # noqa: BLE001
        print(f"  stili.json ignorato in {percorso}: {guaio}")
        return {}


def carica_cartella(percorso: str, versione=(1, 21, 4)) -> list[Modello]:
    """Tutti i `.nbt` di una cartella. Un file rotto non ferma gli altri."""
    if not os.path.isdir(percorso):
        return []
    stili = carica_stili(percorso)
    fuori: list[Modello] = []
    for nome in sorted(os.listdir(percorso)):
        if not nome.lower().endswith(".nbt"):
            continue
        try:
            m = carica(os.path.join(percorso, nome), versione)
            m.stili = stili.get(m.nome, frozenset())
            fuori.append(m)
        except Exception as guaio:                      # noqa: BLE001
            print(f"  template saltato: {nome} ({guaio})")
    return fuori


def carica_cartelle(percorsi: list[str], versione=(1, 21, 4)) -> list[Modello]:
    """Come `carica_cartella`, ma unendo piu' cartelle in un unico catalogo."""
    fuori: list[Modello] = []
    for percorso in percorsi:
        fuori.extend(carica_cartella(percorso, versione))
    return fuori


# --------------------------------------------------------------------------
# Scelta e posa
# --------------------------------------------------------------------------

class Catalogo:
    """I modelli caricati, con gli id di palette risolti sul livello aperto.

    Gli id si risolvono una volta sola per modello e per rotazione: girare le
    proprieta' costa poco, ma farlo per ogni casa di ogni chunk - e una casa
    sta in quattro chunk - vorrebbe dire rifarlo migliaia di volte.
    """

    def __init__(self, modelli: list[Modello], scrittore):
        self.modelli = modelli
        self._id: dict[tuple[int, int], np.ndarray] = {}
        self._scrittore = scrittore

    def __len__(self) -> int:
        return len(self.modelli)

    def id_palette(self, k: int, quarti: int) -> np.ndarray:
        chiave = (k, quarti % 4)
        if chiave not in self._id:
            m = self.modelli[k]
            self._id[chiave] = np.array(
                [self._scrittore.blocco(nome, **ruota_proprieta(prop, quarti))
                 for nome, prop in m.tavolozza], np.uint32)
        return self._id[chiave]

    def celle(self, k: int, quarti: int) -> np.ndarray:
        """Le celle girate di `quarti`, in ordine (x, y, z)."""
        c = self.modelli[k].celle
        for _ in range(quarti % 4):
            # (x, z) -> (dz - 1 - z, x): rot90 sul piano x-z
            c = np.rot90(c, k=-1, axes=(2, 0))
        return np.ascontiguousarray(c)


def candidati_lotto(modelli: list[Modello], larghezza: int, profondita: int,
                    verso_porta: int, altezza_massima: int = 24,
                    stile: str | None = None,
                    escludi: frozenset[int] | set[int] = frozenset()
                    ) -> list[tuple[int, int]]:
    """Tutti i (modello, rotazione) che ENTRANO nel lotto, per intero.

    Prima quelli con la porta dalla parte della strada; se nessuno guarda
    dalla parte giusta, quelli che entrano comunque, girati male: meglio una
    casa girata male che un buco nella fila. Lo stile e' un vincolo rigido.
    `escludi` sono indici di modello gia' usati (vedi `assegna`). Lista
    vuota = nessun modello ci sta: il lotto non ha casa.
    """
    ok_stile = [k for k, m in enumerate(modelli)
                if k not in escludi
                and (not m.stili or stile is None or stile in m.stili)
                and m.dy <= altezza_massima]
    dentro = []
    for k in ok_stile:
        m = modelli[k]
        for quarti in range(4):
            ix, iz = m.ingombro(quarti)
            if ix <= larghezza and iz <= profondita:
                dentro.append((k, quarti))
    giusti = [(k, q) for k, q in dentro
              if modelli[k].porta is None or (modelli[k].porta + q) % 4 == verso_porta]
    return giusti or dentro


def assegna(modelli: list[Modello], edifici: list, rng: np.random.Generator,
            altezza_massima: int = 24) -> dict[int, tuple[int, int]]:
    """Il modello di ogni lotto, senza ripetizioni dentro lo stesso villaggio.

    Ritorna {indice edificio: (modello, rotazione)}. Un lotto in cui non
    entra nessun modello ancora libero NON compare: non ha casa (chi chiama
    ci mette un arredo). Niente sporgenze: una casa che eccede il lotto
    verrebbe tagliata dove incontra quello del vicino.

    Si comincia dai lotti con meno scelta, cosi' un lotto piccolo non resta
    senza il suo unico modello perche' uno grande, che ne avrebbe avuti molti,
    l'ha preso per primo. Due villaggi diversi possono avere la stessa casa.
    `edifici` sono `Edificio` (larghezza, profondita, gronda, porta, stile,
    palafitta, villaggio).
    """
    gruppi: dict[int, list[int]] = {}
    for i, e in enumerate(edifici):
        if not e.palafitta:                # e' un modificatore, non una casa
            gruppi.setdefault(e.villaggio, []).append(i)
    fuori: dict[int, tuple[int, int]] = {}
    for _, indici in sorted(gruppi.items()):
        cand = {i: candidati_lotto(modelli, edifici[i].larghezza + 2 * edifici[i].gronda,
                                   edifici[i].profondita + 2 * edifici[i].gronda,
                                   edifici[i].porta, altezza_massima, edifici[i].stile)
                for i in indici}
        usati: set[int] = set()
        for i in sorted(indici, key=lambda i: (len({k for k, _ in cand[i]}), i)):
            liberi = [(k, q) for k, q in cand[i] if k not in usati]
            if not liberi:
                continue
            k, q = liberi[int(rng.integers(0, len(liberi)))]
            fuori[i] = (k, q)
            usati.add(k)
    return fuori


def scegli(modelli: list[Modello], larghezza: int, profondita: int,
           verso_porta: int, rng: np.random.Generator,
           altezza_massima: int = 24, stile: str | None = None,
           sporgere: bool = True
           ) -> tuple[int, int] | None:
    """Un modello che entra nel lotto, girato in modo da guardare la strada.

    Lo stile e' un vincolo rigido, non una preferenza: una casa deserto non
    deve mai spuntare in un bosco. Un modello senza `stili` (il caso comune,
    vedi `Modello.stili`) passa sempre. L'orientamento e la misura invece
    sono vincoli morbidi - vedi i due fallback sotto. Non esiste piu' un
    generatore parametrico a cui tornare quando niente entra: le case sono
    tutte da template, quindi la scelta deve arrendersi per ultima, non per
    prima. Ritorna None solo se lo stile richiesto non ha NESSUN modello
    (caso che con almeno un template universale per stile non succede mai).

    Con `sporgere=False` l'ultimo ripiego (il modello che eccede il lotto)
    non c'e': se nessun modello entra per intero ritorna None. E' quello che
    usa il motore, tramite `assegna`: una casa tagliata non ha rimedio, un
    lotto senza casa si riempie con un arredo.
    """
    ok_stile = [k for k, m in enumerate(modelli)
                if not m.stili or stile is None or stile in m.stili]
    if not ok_stile:
        return None

    def _giusti(k: int):
        m = modelli[k]
        if m.dy > altezza_massima:
            return
        for quarti in range(4):
            ix, iz = m.ingombro(quarti)
            if ix > larghezza or iz > profondita:
                continue
            if m.porta is not None and (m.porta + quarti) % 4 != verso_porta:
                continue
            yield quarti

    candidati = [(k, q) for k in ok_stile for q in _giusti(k)]
    if not candidati:
        # nessuno guarda dalla parte giusta: meglio una casa girata male che
        # un buco nella fila di case. Lo stile invece resta rigido anche qui.
        candidati = [(k, quarti) for k in ok_stile for quarti in range(4)
                    if modelli[k].dy <= altezza_massima
                    and modelli[k].ingombro(quarti)[0] <= larghezza
                    and modelli[k].ingombro(quarti)[1] <= profondita]
    if not candidati and not sporgere:
        return None
    if not candidati:
        # nemmeno il piu' piccolo dei modelli giusti per stile entra nel
        # lotto (lotto minuscolo, o l'unico modello dello stile e' grande -
        # capita col deserto, che per ora ha una sola casa). Meglio una casa
        # vera che sporge un po' fuori dal lotto che un buco nella fila, e
        # meglio ancora del generatore parametrico che non c'e' piu': si
        # prende il modello con la minor eccedenza rispetto al lotto.
        def _eccesso(k: int, quarti: int) -> int:
            ix, iz = modelli[k].ingombro(quarti)
            return max(0, ix - larghezza) + max(0, iz - profondita)
        tutti = [(k, quarti) for k in ok_stile for quarti in range(4)
                if modelli[k].dy <= altezza_massima]
        if not tutti:
            # neanche per altezza: nessun modello dello stile giusto entra
            # sotto il limite di altezza. Non dovrebbe succedere con un
            # limite ragionevole, ma non e' un errore, solo un lotto senza
            # casa - come i buchi che l'orientamento gia' tollerava.
            return None
        minimo = min(_eccesso(k, q) for k, q in tutti)
        candidati = [(k, q) for k, q in tutti if _eccesso(k, q) == minimo]
    return candidati[int(rng.integers(0, len(candidati)))]


def costruisci(out: np.ndarray, y0: int, ox: int, oz: int, cat: Catalogo,
               k: int, quarti: int, x: int, z: int, base: int,
               vietato: np.ndarray | None = None) -> None:
    """Posa il modello dentro l'array di chunk (16, H, 16).

    `x`, `z` sono l'angolo minimo in coordinate di mappa; `base` e' la quota
    del pavimento, cioe' dove va lo strato y=0 del modello.

    `vietato` (16x16, opzionale) marca le colonne [lx, lz] che appartengono
    al lotto di un ALTRO edificio (vedi `proprietario` in
    `motore.pianifica()`): un modello puo' sporgere oltre il proprio lotto
    (l'ultimo ripiego di `scegli()`, "meglio una casa vera che sporge un
    po'"), ma non deve mai sporgere DENTRO quello del vicino - altrimenti la
    seconda casa disegnata mangia un pezzo di quella disegnata per prima.
    """
    celle = cat.celle(k, quarti)
    ids = cat.id_palette(k, quarti)
    dx, dy, dz = celle.shape
    H = out.shape[1]

    # ritaglio sul chunk: si tocca solo la parte che cade qui dentro
    lx0, lx1 = max(0, ox - x), min(dx, ox + 16 - x)
    lz0, lz1 = max(0, oz - z), min(dz, oz + 16 - z)
    ly0, ly1 = max(0, y0 - base), min(dy, y0 + H - base)
    if lx0 >= lx1 or lz0 >= lz1 or ly0 >= ly1:
        return

    pezzo = celle[lx0:lx1, ly0:ly1, lz0:lz1]
    dentro = pezzo >= 0
    if not dentro.any():
        return
    valori = ids[np.where(dentro, pezzo, 0)]

    ax0, az0 = x + lx0 - ox, z + lz0 - oz
    ay0 = base + ly0 - y0
    if vietato is not None:
        sotto = vietato[ax0:ax0 + (lx1 - lx0), az0:az0 + (lz1 - lz0)]
        if sotto.any():
            dentro = dentro & ~sotto[:, None, :]
            if not dentro.any():
                return
    bersaglio = out[ax0:ax0 + (lx1 - lx0),
                    ay0:ay0 + (ly1 - ly0),
                    az0:az0 + (lz1 - lz0)]
    bersaglio[dentro] = valori[dentro]


def fondazione(out: np.ndarray, y0: int, ox: int, oz: int, cat: Catalogo,
               k: int, quarti: int, x: int, z: int, base: int,
               blocco: int, aria: int, giu: int = 8,
               vietato: np.ndarray | None = None) -> None:
    """Riempie il vuoto sotto il modello.

    Il lotto e' spianato, ma il raccordo lascia il bordo un po' piu' basso, e
    una casa di template non ha il basamento che si costruisce da sola come
    quella parametrica: senza questo, lungo un lato appoggia sull'aria.

    `vietato`: stessa maschera di `costruisci()` - niente basamento sotto il
    lotto di un altro edificio.
    """
    celle = cat.celle(k, quarti)
    dx, _, dz = celle.shape
    H = out.shape[1]
    for px in range(dx):
        lx = x + px - ox
        if not (0 <= lx < 16):
            continue
        for pz in range(dz):
            lz = z + pz - oz
            if not (0 <= lz < 16):
                continue
            if vietato is not None and vietato[lx, lz]:
                continue
            for d in range(1, giu + 1):
                ly = base - d - y0
                if not (0 <= ly < H) or out[lx, ly, lz] != aria:
                    break
                out[lx, ly, lz] = blocco


def statistiche(modelli: list[Modello]) -> dict:
    if not modelli:
        return {"modelli": 0}
    return {
        "modelli": len(modelli),
        "ingombro_min": min(min(m.dx, m.dz) for m in modelli),
        "ingombro_max": max(max(m.dx, m.dz) for m in modelli),
        "altezza_max": max(m.dy for m in modelli),
        "con_porta": sum(1 for m in modelli if m.porta is not None),
    }
