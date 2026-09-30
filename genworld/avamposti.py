"""Accampamenti, cimiteri e portali: dove i nemici presidiano di giorno e si
moltiplicano di notte, e il landmark raro del portale del nether.

Tre strutture sparse a caso sulla mappa, lontane dagli insediamenti (stessa
idea di `miniere.pianifica`: `evita` dice dove NON possono nascere, non c'e'
nessun legame positivo con i paesi):

* un **campo** (`raggio=RAGGIO_CAMPO`, 7x7) - un fuoco acceso al centro,
  quattro tronchi come panche ai lati, due barili di bottino sugli angoli,
  una staccionata rada intorno con un varco. Resta procedurale, disegnato
  colonna per colonna qui sotto (`_carica_campo`).
* un **cimitero**, da template `.nbt` (vedi `carica_cimiteri`) - lo stesso
  meccanismo di `template.py` gia' usato per le case, riusato qui per una
  struttura sola invece che scelta fra tante per un lotto. Il cimitero
  disegnato interamente da codice (`_carica_cimitero_procedurale`, muro di
  cinta, lapidi in griglia, cripta 3x3) resta come RIPIEGO per quando
  nessun template e' disponibile - non piaceva all'utente ("sostituisci
  quello che hai creato tu"), quindi non e' piu' la scelta di default.
* un **portale**, da template `.nbt` (vedi `carica_portali`) - stesso
  meccanismo del cimitero da template, ma SENZA ripiego procedurale: un
  portale non ha una versione disegnata da codice, quindi senza un template
  configurato (`Opzioni.templates_portale`) semplicemente non ne compare
  nessuno, densita' o no. E' voluto un landmark raro, non uno per paese:
  vedi la spaziatura di `pianifica` (`celle_per` molto piu' alto degli
  altri due).

Niente torce ne' lanterne nel campo (vedi `Tavolozza`): e' voluto. Un
blocco gia' scritto qui - il forziere del campo, i muri della cripta -
non ha bisogno di uno spawner per popolarsi di nemici: due o tre arrivano
gia' piazzati (`nemici()`, stesso meccanismo di `miniere.Carrello` o
`fauna.Animale`), e il resto lo fa Minecraft da solo, di notte, perche' il
buio fra le staccionate del campo non lo abbiamo tolto con una luce di
troppo - "la sera escono fuori zombie e scheletri" e' un comportamento
naturale del gioco, non qualcosa che va simulato a mano. Il cimitero da
template puo' avere le sue proprie luci (e' la scelta di chi l'ha
disegnato, non la si corregge), quindi lo spawn naturale li' dipende dal
modello. Il portale non ha nemici piazzati a mano: non c'e' una convenzione
come la ragnatela del cimitero da cui dedurre un punto al coperto, e non e'
comunque lo scopo della struttura - resta un landmark, non un presidio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import template as TM

RAGGIO_CAMPO = 3        # meta' lato del campo (7x7)
RAGGIO_CIMITERO = 5     # meta' lato del cimitero PROCEDURALE (11x11) - solo
                        # per il ripiego, vedi il commento sopra. Un cimitero
                        # da template usa le dimensioni vere del modello,
                        # non questa costante (vedi `pianifica`).
RAGGIO_PORTALE = 4      # ripiego SOLO per chi costruisce un Avamposto a mano
                        # senza passare da `pianifica` (es. nei test): in
                        # pratica un portale usa sempre le dimensioni vere
                        # del modello, come il cimitero da template.

_RAGGIO_DI_DEFAULT = {"campo": RAGGIO_CAMPO, "cimitero": RAGGIO_CIMITERO,
                      "portale": RAGGIO_PORTALE}


@dataclass
class Avamposto:
    x: int
    z: int
    y: int              # quota di superficie nel punto centrale
    tipo: str            # "campo", "cimitero" o "portale"
    # indice nel catalogo dei modelli da template (cimiteri o portali, vedi
    # `carica_cimiteri`/`carica_portali` + `pianifica`); -1 vuol dire
    # "nessun template, usa il ripiego procedurale" per un cimitero, o
    # "niente da disegnare" per un portale (che non ne ha uno) - sempre -1
    # per un campo.
    modello: int = -1
    # mezza estensione (assi x, z): 0 vuol dire "usa il raggio di default
    # del tipo" (`__post_init__` lo risolve). Un cimitero o un portale da
    # template puo' non essere quadrato - vedi `pianifica`, che qui passa le
    # dimensioni vere del modello invece del default.
    mezzo_x: int = 0
    mezzo_z: int = 0

    def __post_init__(self) -> None:
        if not self.mezzo_x:
            self.mezzo_x = _RAGGIO_DI_DEFAULT.get(self.tipo, RAGGIO_CIMITERO)
        if not self.mezzo_z:
            self.mezzo_z = self.mezzo_x

    @property
    def raggio(self) -> int:
        """Mezza estensione massima fra i due assi - un solo numero, per
        chi ha bisogno solo di tenere due avamposti staccati (`_libero()`)
        e non di un ingombro preciso per asse."""
        return max(self.mezzo_x, self.mezzo_z)


@dataclass
class Nemico:
    """Un mostro piazzato a mano, stessa interfaccia di `entita.Abitante` e
    `fauna.Animale` - basta `.x/.y/.z` e `.nbt_tag()`."""
    x: float
    y: float
    z: float
    specie: str = "zombie"     # "zombie" o "scheletro"
    seme: int = 0

    def nbt_tag(self):
        from amulet_nbt import (ByteTag, CompoundTag, DoubleTag, FloatTag,
                                IntTag, ListTag, StringTag)

        from .entita import _uuid
        nome_gioco = "skeleton" if self.specie == "scheletro" else "zombie"
        return CompoundTag({
            "id": StringTag(f"minecraft:{nome_gioco}"),
            "Pos": ListTag([DoubleTag(self.x), DoubleTag(self.y), DoubleTag(self.z)]),
            "Motion": ListTag([DoubleTag(0.0), DoubleTag(0.0), DoubleTag(0.0)]),
            "Rotation": ListTag([FloatTag(0.0), FloatTag(0.0)]),
            "UUID": _uuid(self.seme),
            "Health": FloatTag(20.0),
            "Air": IntTag(300),
            "Fire": IntTag(-1),
            "FallDistance": FloatTag(0.0),
            "Invulnerable": ByteTag(0),
            "OnGround": ByteTag(1),
            # senza questo un nemico lontano dal giocatore puo' sparire nel
            # giro di pochi minuti - stessa ragione di abitanti e animali
            "PersistenceRequired": ByteTag(1),
            # niente raccolta di oggetti caduti: un cimitero pieno di
            # scheletri che si armano con quello che trovano per terra
            # cambierebbe difficolta' senza che l'abbia deciso nessuno
            "CanPickUpLoot": ByteTag(0),
        })


# --------------------------------------------------------------------------
# Pianificazione
# --------------------------------------------------------------------------

def _libero(occupati: list[tuple[int, int, int]], x: int, z: int, raggio: int,
           margine: int = 5) -> bool:
    """Nessun avamposto - campo o cimitero, non importa - troppo vicino a
    un altro gia' piazzato: due strutture a contatto si vedrebbero come una
    sola cosa confusa."""
    for ox, oz, r in occupati:
        if abs(x - ox) < raggio + r + margine and abs(z - oz) < raggio + r + margine:
            return False
    return True


def _mezza_estensione(modelli: list | None, ripiego: int) -> tuple[int, int]:
    """La spaziatura si tara sul piu' grande fra i modelli disponibili, cosi'
    nessuna coppia si tocca qualunque variante venga scelta poi. Senza
    modelli, il ripiego (usato solo dal cimitero procedurale)."""
    if not modelli:
        return ripiego, ripiego
    return (max((m.dx + 1) // 2 for m in modelli),
            max((m.dz + 1) // 2 for m in modelli))


def pianifica(altezze: np.ndarray, mare: np.ndarray, evita: np.ndarray | None = None,
             campi: float = 1.0, cimiteri: float = 1.0, portali: float = 1.0,
             seed: int = 0, modelli_cimitero: list | None = None,
             modelli_portale: list | None = None) -> list[Avamposto]:
    """Sceglie posizioni sparse per campi, cimiteri e portali.

    `evita` e' la stessa maschera di `miniere.pianifica` (mura, strade,
    lotti): un accampamento di nemici dentro le mura di una citta' non ha
    senso quanto un pozzo di miniera nel salotto di una casa.

    `modelli_cimitero` (opzionale): i modelli da template caricati con
    `carica_cimiteri`. Se dato, ogni cimitero piazzato ne sceglie uno a caso
    (`Avamposto.modello`) e la spaziatura usa le dimensioni VERE del modello
    piu' grande, non `RAGGIO_CIMITERO` - un cimitero da template puo' essere
    molto piu' grande degli 11x11 del ripiego procedurale. Senza, si piazzano
    comunque cimiteri (con `modello=-1`, il ripiego) alla vecchia misura.

    `modelli_portale` (opzionale): stessa idea per il portale (vedi
    `carica_portali`), ma senza ripiego - senza modelli non si piazza NESSUN
    portale, qualunque sia `portali`: non c'e' niente da disegnare al suo
    posto (vedi il commento in cima al modulo).
    """
    H, W = altezze.shape
    rng = np.random.default_rng(seed)
    fuori: list[Avamposto] = []
    occupati: list[tuple[int, int, int]] = []
    mezzo_cx, mezzo_cz = _mezza_estensione(modelli_cimitero, RAGGIO_CIMITERO)
    mezzo_px, mezzo_pz = _mezza_estensione(modelli_portale, RAGGIO_PORTALE)
    # celle_per e' tarato per essere molto piu' rado delle miniere: un
    # accampamento di nemici e' un evento raro sulla mappa, non un dettaglio
    # sparso come un pozzo. Il portale e' rado anche rispetto a campo e
    # cimitero: e' un landmark, non una struttura che ci si aspetta di
    # incontrare spesso - da qui il `celle_per` molto piu' alto.
    piani = [("campo", campi, RAGGIO_CAMPO, RAGGIO_CAMPO, 40_000),
             ("cimitero", cimiteri, mezzo_cx, mezzo_cz, 55_000)]
    if modelli_portale:
        piani.append(("portale", portali, mezzo_px, mezzo_pz, 600_000))
    modelli_per_tipo = {"cimitero": modelli_cimitero, "portale": modelli_portale}
    for tipo, densita, mezzo_x, mezzo_z, celle_per in piani:
        if densita <= 0:
            continue
        raggio = max(mezzo_x, mezzo_z)
        quanti = int(max(0, round(H * W / celle_per * densita)))
        piazzati = 0
        tentativi = 0
        limite = quanti * 10 + 30
        margine = raggio + 6
        while piazzati < quanti and tentativi < limite:
            tentativi += 1
            z = int(rng.integers(margine, H - margine))
            x = int(rng.integers(margine, W - margine))
            if mare[z, x] or (evita is not None and evita[z, x]):
                continue
            if not _libero(occupati, x, z, raggio):
                continue
            y = int(altezze[z, x])
            modelli = modelli_per_tipo.get(tipo)
            modello = int(rng.integers(0, len(modelli))) if modelli else -1
            fuori.append(Avamposto(x=x, z=z, y=y, tipo=tipo, modello=modello,
                                   mezzo_x=mezzo_x, mezzo_z=mezzo_z))
            occupati.append((x, z, raggio))
            piazzati += 1
    return fuori


def indice_per_chunk(avamposti: list[Avamposto], passo: int = 16) -> dict:
    """Ogni avamposto va registrato in TUTTI i chunk che il suo ingombro
    tocca - stessa idea di `insediamenti.indice_per_chunk`: un cimitero puo'
    essere largo diversi blocchi, e attraversare piu' di due chunk, non solo
    quelli dei suoi angoli."""
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, av in enumerate(avamposti):
        x0, x1 = av.x - av.mezzo_x, av.x + av.mezzo_x
        z0, z1 = av.z - av.mezzo_z, av.z + av.mezzo_z
        for cz in range(z0 // passo, z1 // passo + 1):
            for cx in range(x0 // passo, x1 // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def maschera(avamposti: list[Avamposto], shape: tuple[int, int],
            margine: int = 0) -> np.ndarray:
    """Ingombro di tutti gli avamposti in una maschera booleana - per
    escludere alberi e fauna dal loro perimetro e per proteggere il
    sottosuolo (`protetto`, vedi `motore.pianifica`), stessa idea del
    `ingombro_tetto` degli edifici."""
    H, W = shape
    fuori = np.zeros(shape, bool)
    for av in avamposti:
        rx, rz = av.mezzo_x + margine, av.mezzo_z + margine
        z0, z1 = max(0, av.z - rz), min(H, av.z + rz + 1)
        x0, x1 = max(0, av.x - rx), min(W, av.x + rx + 1)
        fuori[z0:z1, x0:x1] = True
    return fuori


def _punto_da_ragnatela(m) -> tuple[int, int, int] | None:
    """Un posto vuoto vicino a una ragnatela nel modello (coordinate LOCALI
    del template, x,y,z).

    La ragnatela e' la convenzione di Minecraft per "qui e' un interno, un
    dungeon" - un punto li' vicino e' quasi certamente al coperto. Serve
    proprio per questo: un punto scelto a caso dentro l'ingombro del
    template potrebbe cadere in pieno cortile, e uno scheletro piazzato a
    mano alla luce diretta del sole prende fuoco appena il mondo si apre.
    Nessuna ragnatela nel modello (puo' succedere con un template futuro
    disegnato senza) -> None, e `nemici()` rinuncia al piazzamento a mano
    per quell'avamposto: resta comunque lo spawn naturale notturno, vedi il
    commento in cima al modulo."""
    nomi = [n for n, _ in m.tavolozza]
    ragnatela = {i for i, n in enumerate(nomi) if n == "cobweb"}
    if not ragnatela:
        return None
    dx, dy, dz = m.celle.shape
    posizioni = np.argwhere(np.isin(m.celle, list(ragnatela)))
    if len(posizioni) == 0:
        return None
    x, y, z = (int(v) for v in posizioni[0])
    for ddx, ddz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, nz = x + ddx, z + ddz
        if 0 <= nx < dx and 0 <= nz < dz and m.celle[nx, y, nz] == -1:
            return nx, y, nz
    return None


def nemici(avamposti: list[Avamposto], seed: int = 0,
          modelli_cimitero: list | None = None) -> list[Nemico]:
    """Chi presidia l'avamposto appena il mondo si apre. Posizioni scelte a
    mano dentro le zone lasciate libere dal disegno (mai dentro un muro, mai
    su una lapide, mai alla luce diretta del sole - vedi
    `_punto_da_ragnatela` per il caso del cimitero da template). Il portale
    non ha nessuno piazzato a mano (vedi il commento in cima al modulo): resta
    fuori dal ciclo, non finisce per sbaglio nel ripiego del cimitero."""
    rng = np.random.default_rng(seed)
    fuori: list[Nemico] = []
    for i, av in enumerate(avamposti):
        posti: list[tuple[str, float, float, float]] = []
        if av.tipo == "campo":
            # gli angoli liberi accanto al fuoco (vedi `_carica_campo`: ne'
            # centro, ne' panca, ne' barile, ne' staccionata)
            for specie, dx, dz in (("zombie", 1, 1), ("zombie", -1, -1)):
                posti.append((specie, av.x + dx + 0.5, float(av.y + 1), av.z + dz + 0.5))
        elif av.tipo == "cimitero":
            if av.modello >= 0 and modelli_cimitero:
                m = modelli_cimitero[av.modello]
                punto = _punto_da_ragnatela(m)
                if punto is not None:
                    lx, ly, lz = punto
                    x0, z0 = av.x - m.dx // 2, av.z - m.dz // 2
                    posti.append(("scheletro", x0 + lx + 0.5, float(av.y + ly),
                                 z0 + lz + 0.5))
            else:
                # ripiego procedurale: uno nella cripta, uno sul vialetto
                # centrale verso il cancello (colonna dx=0, mai una lapide -
                # vedi `_carica_cimitero_procedurale`)
                for specie, dx, dz in (("scheletro", 0, 0), ("zombie", 0, av.raggio - 2)):
                    posti.append((specie, av.x + dx + 0.5, float(av.y + 1), av.z + dz + 0.5))
        # "portale": nessun posto, niente piazzato a mano.
        for k, (specie, x, y, z) in enumerate(posti):
            fuori.append(Nemico(x=x, y=y, z=z, specie=specie,
                                seme=int(rng.integers(0, 2 ** 31 - 1)) + i * 131 + k))
    return fuori


def statistiche(avamposti: list[Avamposto]) -> dict:
    campi = sum(1 for a in avamposti if a.tipo == "campo")
    cimiteri = sum(1 for a in avamposti if a.tipo == "cimitero")
    portali = sum(1 for a in avamposti if a.tipo == "portale")
    return {"avamposti": len(avamposti), "campi": campi,
            "cimiteri": cimiteri, "portali": portali}


# --------------------------------------------------------------------------
# Cimitero da template
# --------------------------------------------------------------------------

# Blocchi che il template puo' contenere ma che PyMCTranslate non traduce
# ancora (in nessuna delle due direzioni) sotto la 1.21.9 - mentre il
# progetto punta di default alla 1.21.4 (vedi `Opzioni.versione`). Senza
# sostituzione uscirebbero blocchi sconosciuti, invisibili in gioco: la
# stessa trappola gia' documentata in `template.py` ("un nome non tradotto
# si scrive benissimo e in gioco non c'e'"), verificata qui scrivendo e
# rileggendo un mondo di prova. Equivalenti vicini, non un downgrade
# estetico: una lanterna resta una lanterna, un cespuglio diventa la stessa
# chioma gia' usata per l'albero nello stesso template.
_BLOCCHI_RECENTI = {
    "waxed_oxidized_copper_lantern": lambda prop: ("lantern", dict(prop)),
    "bush": lambda prop: ("leaves", {"material": "dark_oak", "distance": "1",
                                     "persistent": "true", "check_decay": "false"}),
}


def _ripara_blocchi_recenti(tavolozza: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    fuori = []
    for nome, prop in tavolozza:
        sost = _BLOCCHI_RECENTI.get(nome)
        fuori.append(sost(prop) if sost else (nome, prop))
    return fuori


def sostituisci_blocchi_recenti(tavolozza: list[tuple[str, dict]]
                                ) -> list[tuple[str, dict]]:
    """Wrapper pubblico di `_ripara_blocchi_recenti` - serve al
    visualizzatore dei template (`vista_template.py`), che deve mostrare
    esattamente cosa finirebbe nel mondo senza duplicare `_BLOCCHI_RECENTI`."""
    return _ripara_blocchi_recenti(tavolozza)


def _carica_modelli_extra(cartelle: list[str], versione=(1, 21, 4)) -> list:
    """`template.carica_cartelle` piu' la sostituzione dei blocchi non ancora
    tradotti (vedi `_BLOCCHI_RECENTI`) - condivisa da cimitero e portale,
    entrambi caricati con lo stesso meccanismo delle case."""
    modelli = TM.carica_cartelle(cartelle, versione)
    for m in modelli:
        m.tavolozza = _ripara_blocchi_recenti(m.tavolozza)
    return modelli


def carica_cimiteri(cartelle: list[str], versione=(1, 21, 4)) -> list:
    """Modelli di cimitero da `.nbt`, uno o piu' file dentro le cartelle
    date - stessa idea di `template.carica_cartelle` per le case: si butta
    il file dentro la cartella e funziona, scelto a caso da `pianifica` se
    ce n'e' piu' di uno. Cartelle vuote o assenti -> lista vuota, e
    `pianifica`/`posa` ripiegano sul cimitero procedurale (vedi il
    commento in cima al modulo)."""
    return _carica_modelli_extra(cartelle, versione)


def carica_portali(cartelle: list[str], versione=(1, 21, 4)) -> list:
    """Modelli di portale da `.nbt` - stessa idea di `carica_cimiteri`, ma
    senza ripiego: cartelle vuote o assenti -> lista vuota, e `pianifica`
    semplicemente non piazza nessun portale (vedi il commento in cima al
    modulo)."""
    return _carica_modelli_extra(cartelle, versione)


# --------------------------------------------------------------------------
# Posa dentro il chunk
# --------------------------------------------------------------------------

class Tavolozza:
    """Id dei blocchi di campi e cimiteri, risolti sul livello aperto.

    Deliberatamente SENZA torce ne' lanterne: vedi il modulo. Il tronco delle
    panche e' un log NON scortecciato (`stripped="false"`) apposta, per
    distinguerlo a vista dai pali scortecciati della testa di pozzo delle
    miniere (`miniere.Tavolozza.palo`) - un campo e' rimediato, una miniera e'
    costruita.
    """

    def __init__(self, scrittore):
        s = scrittore
        self.falo = s.blocco("campfire", facing="north", lit="true",
                             signal_fire="false", waterlogged="false")
        self.panca_x = s.blocco("log", axis="x", material="oak", stripped="false")
        self.panca_z = s.blocco("log", axis="z", material="oak", stripped="false")
        self.barile = s.blocco("barrel", facing="up", open="false")
        self.staccionata = s.blocco("fence", material="oak", north="false",
                                    south="false", east="false", west="false",
                                    waterlogged="false")
        self.muro = s.blocco("cobblestone")
        self.cancello = s.blocco("fence_gate", material="oak", facing="south",
                                 in_wall="false", open="false", powered="false")
        self.lapide = s.blocco("wall", material="cobblestone", up="true",
                               north="none", south="none", east="none",
                               west="none", waterlogged="false")
        self.cripta_a = s.blocco("stone_bricks", variant="cracked")
        self.cripta_b = s.blocco("stone_bricks", variant="mossy")
        self.ragnatela = s.blocco("cobweb")
        self.rovo = s.blocco("plant", plant_type="dead_bush")


def _carica_campo(out: np.ndarray, tav: Tavolozza, av: Avamposto, ox: int,
                  oz: int, y0: int, protetto_c=None) -> None:
    """Il campo: fuoco al centro, panche ai quattro lati, barili sui due
    angoli opposti, staccionata rada intorno con un varco a sud.

    Gira su TUTTO il chunk come `miniere._carica_ingresso`, non solo sulla
    colonna del centro: il campo e' 7x7, quindi puo' stare a cavallo di piu'
    di un chunk. Funziona solo se `indice_per_chunk` registra l'avamposto in
    ogni chunk toccato - vedi il commento li'.
    """
    H = out.shape[1]
    r = RAGGIO_CAMPO
    base = av.y - y0
    for lx in range(16):
        wx = ox + lx
        dx = wx - av.x
        if abs(dx) > r:
            continue
        for lz in range(16):
            wz = oz + lz
            dz = wz - av.z
            if abs(dz) > r:
                continue
            if protetto_c is not None and protetto_c[lx, lz]:
                continue
            if not (0 <= base < H):
                continue
            if dx == 0 and dz == 0:
                out[lx, base, lz] = tav.falo
                continue
            if (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                out[lx, base, lz] = tav.panca_x if dx == 0 else tav.panca_z
                continue
            if (abs(dx), abs(dz)) == (r - 1, r - 1):
                out[lx, base, lz] = tav.barile
                continue
            if max(abs(dx), abs(dz)) == r:
                if dz == r and abs(dx) <= 1:
                    continue          # il varco d'ingresso, niente staccionata
                if (dx + dz) % 2 == 0:
                    out[lx, base, lz] = tav.staccionata


def _carica_cimitero_procedurale(out: np.ndarray, tav: Tavolozza, av: Avamposto,
                                 ox: int, oz: int, y0: int, protetto_c=None) -> None:
    """Il cimitero RIPIEGO, disegnato interamente da codice: cripta al
    centro (3x3, tetto compreso, varco a sud), lapidi in una griglia rada,
    muro di cinta col cancello sullo stesso asse del varco della cripta - un
    vialetto dritto dal cancello alla cripta.

    E' il cimitero che c'era prima del template - vedi il commento in cima
    al modulo: resta solo per quando nessun template e' disponibile
    (`carica_cimiteri` non trova nulla).

    Stesso principio di `_carica_campo`: gira su tutto il chunk, mai solo
    sulla colonna centrale.
    """
    H = out.shape[1]
    r = RAGGIO_CIMITERO
    base = av.y - y0
    for lx in range(16):
        wx = ox + lx
        dx = wx - av.x
        if abs(dx) > r:
            continue
        for lz in range(16):
            wz = oz + lz
            dz = wz - av.z
            if abs(dz) > r:
                continue
            if protetto_c is not None and protetto_c[lx, lz]:
                continue
            if not (0 <= base < H):
                continue

            if abs(dx) <= 1 and abs(dz) <= 1:
                if dx == 0 and dz == 1:
                    continue          # il varco della cripta, resta vuoto
                if dx != 0 or dz != 0:
                    cripta = tav.cripta_a if (dx + dz) % 2 == 0 else tav.cripta_b
                    for ly in range(base, base + 3):
                        if 0 <= ly < H:
                            out[lx, ly, lz] = cripta
                    tetto = base + 3
                    if 0 <= tetto < H:
                        out[lx, tetto, lz] = tav.cripta_a
                else:
                    out[lx, base, lz] = tav.ragnatela   # il centro della cripta
                    tetto = base + 3
                    if 0 <= tetto < H:
                        out[lx, tetto, lz] = tav.cripta_a
                continue

            if max(abs(dx), abs(dz)) == r:
                if dz == r and dx == 0:
                    out[lx, base, lz] = tav.cancello
                else:
                    for ly in range(base, base + 2):
                        if 0 <= ly < H:
                            out[lx, ly, lz] = tav.muro
                continue

            if (abs(dx), abs(dz)) == (3, 3):
                out[lx, base, lz] = tav.rovo
                continue

            # il vialetto (dx=0) resta sempre libero, dal cancello alla
            # cripta: nessuna lapide sulla colonna centrale.
            if dx != 0 and dx % 2 == 0 and dz % 2 == 0:
                out[lx, base, lz] = tav.lapide


def _carica_struttura_template(out: np.ndarray, cat: "TM.Catalogo", av: Avamposto,
                               ox: int, oz: int, y0: int, aria: int,
                               basamento: int, protetto_c=None) -> None:
    """Un avamposto da template (cimitero o portale): si posa come una casa
    (`template.fondazione`/`costruisci`), centrato sul punto scelto da
    `pianifica`. Quarti fisso a 0: a differenza delle case, non c'e' una
    strada a cui girarsi - e' una struttura sola, si posa cosi' come l'ha
    disegnata chi l'ha fatta."""
    k = av.modello
    m = cat.modelli[k]
    x0 = av.x - m.dx // 2
    z0 = av.z - m.dz // 2
    base = av.y
    TM.fondazione(out, y0, ox, oz, cat, k, 0, x0, z0, base,
                 basamento, aria, vietato=protetto_c)
    TM.costruisci(out, y0, ox, oz, cat, k, 0, x0, z0, base, vietato=protetto_c)


def posa(out: np.ndarray, tav: Tavolozza, ox: int, oz: int, y0: int,
        avamposti: list[Avamposto], quali, cat_cimitero: "TM.Catalogo | None" = None,
        aria_cimitero: int = 0, basamento_cimitero: int = 0,
        cat_portale: "TM.Catalogo | None" = None, aria_portale: int = 0,
        basamento_portale: int = 0, protetto_c=None) -> None:
    """Campi, cimiteri e portali dentro il chunk (16, H, 16).

    `cat_cimitero`/`cat_portale` (opzionali): i cataloghi dei modelli da
    template, costruiti da chi chiama con `carica_cimiteri`/`carica_portali`
    + un `template.Catalogo`. Per un cimitero, senza catalogo o per un
    avamposto senza modello assegnato (`av.modello == -1`, il ripiego
    procedurale quando nessun template era disponibile in `pianifica`), si
    disegna il cimitero da codice. Per un portale invece non c'e' ripiego:
    senza catalogo o modello semplicemente non si disegna nulla (non
    dovrebbe capitare - `pianifica` non piazza portali senza modelli - ma
    non e' un errore se succede)."""
    for i in quali:
        av = avamposti[i]
        if av.tipo == "campo":
            _carica_campo(out, tav, av, ox, oz, y0, protetto_c)
        elif av.tipo == "cimitero":
            if cat_cimitero is not None and av.modello >= 0:
                _carica_struttura_template(out, cat_cimitero, av, ox, oz, y0,
                                           aria_cimitero, basamento_cimitero,
                                           protetto_c)
            else:
                _carica_cimitero_procedurale(out, tav, av, ox, oz, y0, protetto_c)
        elif av.tipo == "portale" and cat_portale is not None and av.modello >= 0:
            _carica_struttura_template(out, cat_portale, av, ox, oz, y0,
                                       aria_portale, basamento_portale,
                                       protetto_c)
