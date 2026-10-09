"""Il bottino dentro i forzieri delle case.

Le case sono tutte da template (vedi `template.py`): un file `.nbt` di
blocco struttura che puo' gia' contenere un forziere - molti dei modelli in
`templates/strutture` ne hanno uno in dispensa o in camera. Il guaio e' che
`template.carica()` legge solo id di blocco e proprieta' (`voce["state"]`),
mai il tag `nbt` opzionale che il formato blocco struttura porta per ogni
cella: e' cosi' che un forziere del template arriva in gioco vuoto, anche
quando chi ha disegnato la casa ce l'aveva riempito.

Rifare `template.carica()` per portarsi dietro quel tag varrebbe solo per i
forzieri (e' l'unico blocco dei template che ha davvero bisogno di un
inventario), quindi la strada piu' corta e' un'altra: dopo che un edificio e'
stato piazzato nel chunk, si cerca l'id di un forziere singolo di legno
nell'array dei blocchi gia' scritto (l'id e' lo stesso, sia che l'abbia
messo il template sia che lo metta chiunque altro: `ScrittoreMondo.blocco()`
deduplica per nome e proprieta') e li' si genera un contenuto a caso.

Solo il forziere SINGOLO (`connection="none"`): un baule doppio ha
l'inventario condiviso fra le due meta' (54 slot, non 27+27 indipendenti), e
riempirne solo una meta' lascerebbe l'altra vuota in un modo che si nota. Un
forziere a doppio battente resta vuoto come prima - meglio vuoto che a meta'.
"""

from __future__ import annotations

import numpy as np

# Roba plausibile nella dispensa o nel bauletto di chi vive in una casa
# qualunque: cibo, materiali di base, qualche attrezzo. Niente di eccezionale
# - non e' un tesoro di dungeon, e' quello che si trova in una cucina.
OGGETTI: tuple[tuple[str, int, int], ...] = (
    ("bread", 1, 4),
    ("wheat", 1, 6),
    ("apple", 1, 4),
    ("carrot", 1, 6),
    ("potato", 1, 6),
    ("cooked_beef", 1, 3),
    ("cooked_chicken", 1, 3),
    ("coal", 1, 8),
    ("iron_ingot", 1, 4),
    ("stick", 2, 8),
    ("string", 1, 6),
    ("leather", 1, 4),
    ("emerald", 1, 2),
    ("torch", 2, 8),
    ("arrow", 4, 12),
    ("flint", 1, 6),
    ("white_wool", 1, 4),
    ("paper", 1, 6),
)


# Il bottino di un accampamento: sempre almeno un diamante e un'arma, poi
# provviste e un po' di refurtiva.
ARMI: tuple[str, ...] = ("iron_sword", "iron_axe", "bow", "crossbow", "stone_sword",
                         "stone_axe", "golden_sword", "iron_pickaxe")
EXTRA_CAMPO: tuple[tuple[str, int, int], ...] = (
    ("arrow", 8, 24), ("cooked_beef", 1, 4), ("bread", 1, 4), ("emerald", 1, 3),
    ("gold_ingot", 1, 5), ("iron_ingot", 1, 5), ("golden_apple", 1, 1),
    ("leather", 1, 4), ("string", 1, 6), ("flint", 1, 6),
)
# Il bottino di una miniera: minerale grezzo, gemme e qualche attrezzo.
BOTTINO_MINIERA: tuple[tuple[str, int, int], ...] = (
    ("raw_iron", 2, 10), ("raw_copper", 3, 12), ("coal", 4, 16), ("raw_gold", 1, 6),
    ("lapis_lazuli", 2, 9), ("redstone", 3, 12), ("amethyst_shard", 1, 5),
    ("emerald", 1, 3), ("diamond", 1, 3), ("iron_pickaxe", 1, 1),
    ("stone_pickaxe", 1, 1), ("torch", 4, 12), ("bread", 1, 3), ("iron_ingot", 1, 5),
)


class Tavolozza:
    """Gli id nel palette del livello per i forzieri che sappiamo riempire:
    solo `chest` (mai `trapped_chest`, `ender_chest` o le varianti di rame -
    quelle non sono la dispensa di una casa) con `connection="none"`, nelle
    quattro orientazioni possibili dopo la rotazione del template."""

    def __init__(self, scrittore):
        self.singolo = frozenset(
            scrittore.blocco("chest", facing=f, connection="none", material="wood")
            for f in ("north", "south", "east", "west"))


def _oggetti(rng: np.random.Generator) -> list[tuple[int, str, int]]:
    """2-5 pile scelte a caso, in slot diversi di un inventario da 27."""
    n = int(rng.integers(2, 6))
    slot = rng.choice(27, size=n, replace=False)
    fuori = []
    for s in slot:
        nome, minimo, massimo = OGGETTI[int(rng.integers(0, len(OGGETTI)))]
        fuori.append((int(s), nome, int(rng.integers(minimo, massimo + 1))))
    return fuori


def _oggetti_campo(rng: np.random.Generator) -> list[tuple[int, str, int]]:
    """Un diamante (1-3), un'arma e 3-5 cose in piu', in slot diversi."""
    n = int(rng.integers(5, 8))
    slot = [int(v) for v in rng.choice(27, size=n, replace=False)]
    fuori = [(slot[0], "diamond", int(rng.integers(1, 4))),
             (slot[1], ARMI[int(rng.integers(0, len(ARMI)))], 1)]
    for s in slot[2:]:
        nome, minimo, massimo = EXTRA_CAMPO[int(rng.integers(0, len(EXTRA_CAMPO)))]
        fuori.append((s, nome, int(rng.integers(minimo, massimo + 1))))
    return fuori


def _oggetti_miniera(rng: np.random.Generator) -> list[tuple[int, str, int]]:
    """4-7 pile di minerale, gemme e attrezzi, tutte di tipo diverso."""
    n = int(rng.integers(4, 8))
    slot = [int(v) for v in rng.choice(27, size=n, replace=False)]
    tipi = rng.choice(len(BOTTINO_MINIERA), size=n, replace=False)
    return [(s, BOTTINO_MINIERA[int(t)][0],
             int(rng.integers(BOTTINO_MINIERA[int(t)][1], BOTTINO_MINIERA[int(t)][2] + 1)))
            for s, t in zip(slot, tipi)]


def _seme(x: int, y: int, z: int, seed: int) -> int:
    """Deterministico sulla posizione: lo stesso forziere ha sempre lo
    stesso contenuto, anche rigenerando lo stesso lotto. Mascherato per le
    stesse ragioni di `sottosuolo._seme` - meta' del mondo ha coordinate
    negative."""
    return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791) ^
            (seed * 999999937)) & 0x7FFFFFFF


def trova(blocchi: np.ndarray, tav: Tavolozza, ox: int, oz: int, y0: int,
         seed: int = 0, campi: list | None = None, minerali: set | None = None) -> list:
    """Cerca i forzieri singoli gia' scritti in un chunk (16, H, 16) e
    ritorna la lista di `amulet.api.block_entity.BlockEntity` col bottino.

    Si cerca il BLOCCO, non si segue l'elenco degli edifici: a questo punto
    il template e' gia' stato ruotato e piazzato, e l'id nel palette e'
    l'unica cosa rimasta che dice "qui c'e' un forziere, riempilo" - la
    stessa idea di `miniere.Tavolozza.scavabile` per il sottosuolo.
    """
    from amulet.api.block_entity import BlockEntity
    from amulet_nbt import ByteTag, CompoundTag, IntTag, ListTag, NamedTag, StringTag

    fuori = []
    if not tav.singolo:
        return fuori
    posizioni = np.transpose(np.nonzero(np.isin(blocchi, list(tav.singolo))))
    for lx, ly, lz in posizioni:
        gx, gy, gz = ox + int(lx), y0 + int(ly), oz + int(lz)
        rng = np.random.default_rng(_seme(gx, gy, gz, seed))
        # il bottino dipende da DOVE sta il forziere: in una miniera minerale e
        # gemme, in un accampamento un diamante e un'arma, in casa la dispensa
        if minerali and (gx, gy, gz) in minerali:
            contenuto = _oggetti_miniera(rng)
        elif campi and any((gx - cx) ** 2 + (gz - cz) ** 2 <= r * r for cx, cz, r in campi):
            contenuto = _oggetti_campo(rng)
        else:
            contenuto = _oggetti(rng)
        items = ListTag([
            CompoundTag({"Slot": ByteTag(s), "id": StringTag(f"minecraft:{nome}"),
                        "count": IntTag(quanti)})
            for s, nome, quanti in contenuto
        ])
        # Namespace "minecraft" (raw) con NBT piatto si perde - verificato
        # con uno spike dedicato - non appena la sessione di scrittura tocca
        # PIU' di un chunk (con un solo chunk sopravvive, che e' perche' il
        # difetto e' passato inosservato finche' non si e' provato su una
        # citta' vera). Il namespace "universal_minecraft" con l'NBT
        # incapsulato in "utags" e' la stessa forma che amulet produce da
        # solo quando genera un block-entity dal proprio traduttore
        # universale, ed e' l'unica delle due che sopravvive al giro
        # salva/rileggi su un mondo multi-chunk.
        utags = CompoundTag({
            "isMovable": ByteTag(1),
            "Findable": ByteTag(0),
            "Items": items,
            "Lock": StringTag(""),
        })
        nbt = NamedTag(CompoundTag({"utags": utags}))
        fuori.append(BlockEntity("universal_minecraft", "chest", gx, gy, gz, nbt))
    return fuori
