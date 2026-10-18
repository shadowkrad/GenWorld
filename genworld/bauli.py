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

Singoli, doppi e botti. Ogni meta' di un forziere doppio ha il SUO blocco con
27 slot (la vista da 54 e' solo del gioco), quindi si riempiono entrambe; le
botti hanno 27 slot come un forziere. In piu' i forni, le fornaci e gli
affumicatoi ricevono il combustibile (carbone), cosi' si accendono subito.
"""

from __future__ import annotations

import numpy as np

# Roba plausibile nella dispensa o nel bauletto di chi vive in una casa
# qualunque: cibo, materiali di base, qualche attrezzo. Niente di eccezionale
# - non e' un tesoro di dungeon, e' quello che si trova in una cucina.
OGGETTI: tuple[tuple[str, int, int], ...] = (
    # attrezzi e materiali di base, in una casa se ne trova sempre
    ("stone_pickaxe", 1, 1), ("stone_axe", 1, 1), ("stone_shovel", 1, 1), ("stone_hoe", 1, 1),
    ("wooden_pickaxe", 1, 1), ("shears", 1, 1), ("bucket", 1, 1), ("fishing_rod", 1, 1),
    ("flint_and_steel", 1, 1), ("oak_planks", 4, 16), ("oak_log", 2, 8),
    ("cobblestone", 6, 24), ("copper_ingot", 2, 8), ("glass_bottle", 2, 6),
    ("clay_ball", 4, 12), ("brick", 3, 10), ("redstone", 2, 8), ("charcoal", 3, 10),
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
    ("book", 1, 3),
    ("candle", 1, 4),
    ("clock", 1, 1),
    ("compass", 1, 1),
    ("honey_bottle", 1, 2),
    ("cookie", 2, 8),
    ("pumpkin_pie", 1, 2),
    ("bone_meal", 1, 6),
    ("music_disc_13", 1, 1),
    ("spyglass", 1, 1),
)


# Il bottino di un accampamento: sempre almeno un diamante e un'arma, poi
# provviste e un po' di refurtiva.
ARMI: tuple[str, ...] = ("iron_sword", "iron_axe", "bow", "crossbow", "stone_sword",
                         "stone_axe", "golden_sword", "iron_pickaxe", "diamond_sword",
                         "iron_chestplate", "iron_helmet", "iron_leggings",
                         "iron_boots", "shield", "chainmail_chestplate")
EXTRA_CAMPO: tuple[tuple[str, int, int], ...] = (
    ("arrow", 8, 24), ("cooked_beef", 1, 4), ("bread", 1, 4), ("emerald", 1, 3),
    ("gold_ingot", 1, 5), ("iron_ingot", 1, 5), ("golden_apple", 1, 1),
    ("leather", 1, 4), ("string", 1, 6), ("flint", 1, 6),
    ("saddle", 1, 1), ("ender_pearl", 1, 3), ("spyglass", 1, 1), ("compass", 1, 1),
    ("name_tag", 1, 2), ("lead", 1, 2), ("cooked_porkchop", 1, 4),
)
# Il bottino di una cripta (cimitero): ossa, refurtiva dei morti, qualche
# oggetto raro.
BOTTINO_CRIPTA: tuple[tuple[str, int, int], ...] = (
    ("bone", 2, 12), ("rotten_flesh", 2, 10), ("gold_ingot", 1, 5),
    ("gold_nugget", 3, 14), ("emerald", 1, 3), ("diamond", 1, 2),
    ("iron_sword", 1, 1), ("golden_apple", 1, 1), ("name_tag", 1, 1),
    ("saddle", 1, 1), ("music_disc_cat", 1, 1), ("book", 1, 3),
    ("candle", 2, 6), ("amethyst_shard", 1, 4), ("ender_pearl", 1, 2),
    ("spider_eye", 1, 4), ("skeleton_skull", 1, 1),
)
# Il bottino di una miniera: minerale grezzo, gemme e qualche attrezzo.
BOTTINO_MINIERA: tuple[tuple[str, int, int], ...] = (
    ("raw_iron", 2, 10), ("raw_copper", 3, 12), ("coal", 4, 16), ("raw_gold", 1, 6),
    ("lapis_lazuli", 2, 9), ("redstone", 3, 12), ("amethyst_shard", 1, 5),
    ("emerald", 1, 3), ("diamond", 1, 3), ("iron_pickaxe", 1, 1),
    ("stone_pickaxe", 1, 1), ("torch", 4, 12), ("bread", 1, 3), ("iron_ingot", 1, 5),
    ("diamond_pickaxe", 1, 1), ("lantern", 1, 2), ("bucket", 1, 1), ("rail", 4, 16),
    ("powered_rail", 2, 6), ("tnt", 1, 3), ("golden_apple", 1, 1),
    ("glow_berries", 2, 8), ("raw_copper", 2, 8), ("music_disc_5", 1, 1),
)


# Tesori incantati: lontano dalle citta' un forziere di miniera, di campo o di
# cripta puo' avere uno di questi. (oggetto, componenti nel formato 1.21: gli
# incantesimi sono una mappa nome -> livello)
DISTANZA_TESORI = 250
TESORI: tuple[tuple[str, dict], ...] = (
    ("diamond_sword", {"enchantments": {"minecraft:sharpness": 3, "minecraft:unbreaking": 2}}),
    ("iron_pickaxe", {"enchantments": {"minecraft:efficiency": 3, "minecraft:fortune": 2}}),
    ("diamond_pickaxe", {"enchantments": {"minecraft:efficiency": 4, "minecraft:unbreaking": 3}}),
    ("bow", {"enchantments": {"minecraft:power": 3, "minecraft:infinity": 1}}),
    ("crossbow", {"enchantments": {"minecraft:quick_charge": 2, "minecraft:piercing": 3}}),
    ("iron_chestplate", {"enchantments": {"minecraft:protection": 3}}),
    ("diamond_helmet", {"enchantments": {"minecraft:respiration": 2, "minecraft:protection": 2}}),
    ("iron_boots", {"enchantments": {"minecraft:feather_falling": 3}}),
    ("enchanted_book", {"stored_enchantments": {"minecraft:mending": 1}}),
    ("enchanted_book", {"stored_enchantments": {"minecraft:sharpness": 4}}),
    ("enchanted_book", {"stored_enchantments": {"minecraft:fortune": 3}}),
    ("potion", {"potion_contents": {"potion": "minecraft:healing"}}),
    ("potion", {"potion_contents": {"potion": "minecraft:swiftness"}}),
    ("potion", {"potion_contents": {"potion": "minecraft:night_vision"}}),
)


def _tag(v):
    """Un valore python (dict, int, str) come tag NBT."""
    from amulet_nbt import CompoundTag, IntTag, ListTag, StringTag
    if isinstance(v, dict):
        return CompoundTag({k: _tag(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        return ListTag([_tag(x) for x in v])
    if isinstance(v, int):
        return IntTag(v)
    return StringTag(str(v))


def _aggiungi_tesori(contenuto: list, rng: np.random.Generator, quanti: int = 0) -> list:
    """Sostituisce una o due pile (o `quanti`) con tesori incantati (stesso slot)."""
    contenuto = list(contenuto)
    quanti = min(len(contenuto), quanti or int(rng.integers(1, 3)))
    for i in rng.choice(len(contenuto), size=quanti, replace=False):
        nome, comp = TESORI[int(rng.integers(0, len(TESORI)))]
        contenuto[int(i)] = (contenuto[int(i)][0], nome, 1, comp)
    return contenuto


class Tavolozza:
    """Gli id nel palette del livello per i forzieri che sappiamo riempire:
    solo `chest` (mai `trapped_chest`, `ender_chest` o le varianti di rame -
    quelle non sono la dispensa di una casa) con `connection="none"`, nelle
    quattro orientazioni possibili dopo la rotazione del template."""

    def __init__(self, scrittore):
        facce = ("north", "south", "east", "west")
        self.singolo = frozenset(
            scrittore.blocco("chest", facing=f, connection="none", material="wood")
            for f in facce)
        # le due meta' di un forziere doppio: ognuna ha il suo inventario da 27
        self.doppio = frozenset(
            scrittore.blocco("chest", facing=f, connection=c, material="wood")
            for f in facce for c in ("left", "right"))
        self.botti = frozenset(
            scrittore.blocco("barrel", facing=f, open=o)
            for f in facce + ("up", "down") for o in ("false", "true"))
        # forni: il combustibile nello slot 1 (carbone)
        self.forni = {
            nome: frozenset(scrittore.blocco(nome, facing=f, lit=lit)
                            for f in facce for lit in ("false", "true"))
            for nome in ("furnace", "blast_furnace", "smoker")}


def _oggetti(rng: np.random.Generator) -> list[tuple[int, str, int]]:
    """3-7 pile scelte a caso, in slot diversi di un inventario da 27."""
    n = int(rng.integers(3, 8))
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


# Il bottino di una cripta con rovina (vedi `dungeon.py`): la sala dello spawner
# ha provviste e minerali, il tesoro finale le cose migliori.
BOTTINO_DUNGEON: tuple[tuple[str, int, int], ...] = (
    ("iron_ingot", 2, 8), ("gold_ingot", 2, 8), ("diamond", 1, 4), ("emerald", 2, 6),
    ("golden_apple", 1, 2), ("ender_pearl", 1, 3), ("bone", 4, 16), ("arrow", 8, 24),
    ("saddle", 1, 1), ("name_tag", 1, 2), ("music_disc_cat", 1, 1),
    ("music_disc_13", 1, 1), ("book", 2, 6), ("experience_bottle", 2, 8),
    ("obsidian", 2, 6), ("redstone", 4, 14), ("gunpowder", 3, 10),
)


# Il bottino di un mago: pozioni, libri incantati e roba da alchimista.
POZIONI = ("healing", "swiftness", "night_vision", "water_breathing", "fire_resistance",
           "strength", "leaping", "regeneration")
BOTTINO_MAGO: tuple[tuple[str, int, int], ...] = (
    ("glass_bottle", 3, 9), ("redstone", 4, 14), ("glowstone_dust", 3, 10), ("lapis_lazuli", 4, 16),
    ("spider_eye", 2, 6), ("fermented_spider_eye", 1, 3), ("gunpowder", 3, 8), ("sugar", 3, 9),
    ("nether_wart", 3, 10), ("blaze_powder", 2, 6), ("experience_bottle", 3, 10),
    ("book", 3, 8), ("paper", 4, 10), ("amethyst_shard", 2, 6), ("ender_pearl", 1, 3),
    ("golden_carrot", 2, 6), ("candle", 3, 8), ("totem_of_undying", 1, 1),
)
LIBRI_MAGO = ("sharpness", "protection", "unbreaking", "efficiency", "fortune", "looting",
              "power", "feather_falling", "respiration", "aqua_affinity", "mending", "thorns")


def _oggetti_mago(rng: np.random.Generator, tipo: str, lontano: bool) -> list:
    """Il forziere di un mago: roba da alchimista, una o due pozioni e libri incantati;
    il tesoro in cima ne ha di piu' (e Riparazione se si e' lontani dalle citta')."""
    tesoro = tipo.endswith("tesoro")
    n = int(rng.integers(6, 9)) if tesoro else int(rng.integers(4, 7))
    slot = [int(v) for v in rng.choice(27, size=n + 5, replace=False)]
    tipi = rng.choice(len(BOTTINO_MAGO), size=n, replace=False)
    fuori = [(slot[k], BOTTINO_MAGO[int(t)][0],
              int(rng.integers(BOTTINO_MAGO[int(t)][1], BOTTINO_MAGO[int(t)][2] + 1)))
             for k, t in enumerate(tipi)]
    for k in range(int(rng.integers(1, 3 if tesoro else 2))):
        fuori.append((slot[n + k], "potion", 1,
                      {"potion_contents": {"potion": "minecraft:" + POZIONI[int(rng.integers(0, len(POZIONI)))]}}))
    libri = int(rng.integers(2, 4)) if tesoro else 1
    for k in range(libri):
        nome = LIBRI_MAGO[int(rng.integers(0, len(LIBRI_MAGO)))]
        if tesoro and lontano and k == 0:
            nome = "mending"
        livello = 1 if nome == "mending" else int(rng.integers(1, 4))
        fuori.append((slot[(n + 2 + k) % len(slot)], "enchanted_book", 1,
                      {"stored_enchantments": {f"minecraft:{nome}": livello}}))
    # una pila per slot: se un indice si e' ripetuto si tiene l'ultima
    per_slot = {}
    for voce in fuori:
        per_slot[voce[0]] = voce
    return list(per_slot.values())


# Il bottino del Nether e dell'End (vedi `nether.py`, `end.py`). (nome, minimo, massimo)
BOTTINO_DIMENSIONI: dict[str, tuple] = {
    "nether_fortezza": (("blaze_rod", 2, 6), ("nether_wart", 4, 12), ("gold_ingot", 3, 10),
                        ("golden_apple", 1, 2), ("diamond", 1, 3), ("saddle", 1, 1),
                        ("obsidian", 2, 6), ("flint_and_steel", 1, 1), ("glowstone_dust", 4, 12),
                        ("quartz", 4, 14), ("magma_cream", 2, 5), ("iron_horse_armor", 1, 1),
                        ("coal", 6, 16), ("ghast_tear", 1, 2)),
    "nether_tesoro": (("diamond", 2, 5), ("netherite_scrap", 1, 3), ("ancient_debris", 1, 3),
                      ("enchanted_golden_apple", 1, 1), ("obsidian", 4, 10),
                      ("gold_block", 1, 2), ("diamond_horse_armor", 1, 1), ("blaze_rod", 4, 8),
                      ("ender_pearl", 2, 5), ("experience_bottle", 6, 16)),
    "nether_bastione": (("gold_ingot", 6, 20), ("gold_block", 1, 3), ("golden_sword", 1, 1),
                        ("golden_axe", 1, 1), ("golden_pickaxe", 1, 1), ("crossbow", 1, 1),
                        ("arrow", 12, 32), ("spectral_arrow", 6, 16), ("iron_nugget", 8, 24),
                        ("crying_obsidian", 3, 8), ("golden_carrot", 4, 12), ("string", 6, 16),
                        ("ancient_debris", 1, 2), ("netherite_scrap", 1, 1), ("gilded_blackstone", 2, 6),
                        ("music_disc_pigstep", 1, 1), ("diamond", 1, 3)),
    "end_citta": (("ender_pearl", 2, 6), ("end_rod", 4, 12), ("chorus_fruit", 4, 12),
                  ("purpur_block", 8, 24), ("iron_ingot", 4, 12), ("gold_ingot", 3, 9),
                  ("diamond", 1, 3), ("experience_bottle", 4, 12), ("obsidian", 3, 8),
                  ("book", 2, 5), ("shulker_shell", 1, 2), ("golden_apple", 1, 2)),
    "end_tesoro": (("elytra", 1, 1), ("shulker_shell", 2, 4), ("diamond", 3, 6),
                   ("enchanted_golden_apple", 1, 2), ("ender_pearl", 4, 8), ("dragon_breath", 1, 2),
                   ("experience_bottle", 8, 20), ("netherite_scrap", 1, 2), ("end_rod", 8, 16)),
}


def _oggetti_dimensione(rng: np.random.Generator, tipo: str, lontano: bool) -> list:
    voci = BOTTINO_DIMENSIONI[tipo]
    n = min(len(voci), int(rng.integers(5, 9)))
    slot = [int(v) for v in rng.choice(27, size=n, replace=False)]
    scelti = rng.choice(len(voci), size=n, replace=False)
    # il tesoro di fondo ha sempre le cose rare (le prime della lista)
    if tipo in ("end_tesoro", "nether_tesoro") and 0 not in scelti:
        scelti[0] = 0
    contenuto = [(s, voci[int(t)][0], int(rng.integers(voci[int(t)][1], voci[int(t)][2] + 1)))
                 for s, t in zip(slot, scelti)]
    if tipo.endswith("tesoro"):
        contenuto = [c for c in contenuto if c[1] != "elytra"] + [
            c for c in contenuto if c[1] == "elytra"]
        contenuto = _aggiungi_tesori(contenuto, rng, 2) if "elytra" not in [c[1] for c in contenuto]             else contenuto
    return contenuto


def _oggetti_dungeon(rng: np.random.Generator, tipo: str, lontano: bool) -> list:
    """La sala dello spawner: 4-6 pile. Il tesoro: 6-9 pile e tesori incantati,
    uno vicino alle citta', tre lontano."""
    if tipo.startswith("mago"):
        return _oggetti_mago(rng, tipo, lontano)
    if tipo in BOTTINO_DIMENSIONI:
        return _oggetti_dimensione(rng, tipo, lontano)
    n = int(rng.integers(6, 10)) if tipo == "tesoro" else int(rng.integers(4, 7))
    slot = [int(v) for v in rng.choice(27, size=n, replace=False)]
    tipi = rng.choice(len(BOTTINO_DUNGEON), size=n, replace=False)
    contenuto = [(s, BOTTINO_DUNGEON[int(t)][0],
                  int(rng.integers(BOTTINO_DUNGEON[int(t)][1], BOTTINO_DUNGEON[int(t)][2] + 1)))
                 for s, t in zip(slot, tipi)]
    if tipo == "tesoro":
        contenuto = _aggiungi_tesori(contenuto, rng, 3 if lontano else 1)
    elif lontano:
        contenuto = _aggiungi_tesori(contenuto, rng, 1)
    return contenuto


# Il bottino delle strutture da template che hanno un loro carattere: la nave
# porta oro e provviste di bordo, il faro il necessario di chi veglia, il
# castello armi, libri e oggetti da signore. (nome, minimo, massimo)
BOTTINO_ZONA: dict[str, tuple] = {
    "nave": (("gold_ingot", 2, 8), ("emerald", 1, 5), ("iron_ingot", 3, 9),
             ("compass", 1, 1), ("spyglass", 1, 1), ("bread", 2, 6),
             ("cooked_cod", 3, 8), ("lead", 1, 3), ("name_tag", 1, 1),
             ("rotten_flesh", 2, 6), ("barrel", 1, 1), ("string", 4, 12)),
    "faro": (("coal", 4, 12), ("torch", 8, 24), ("cooked_salmon", 2, 6),
             ("fishing_rod", 1, 1), ("spyglass", 1, 1), ("lantern", 1, 3),
             ("iron_ingot", 2, 5), ("emerald", 1, 3), ("glass_bottle", 2, 6),
             ("bread", 2, 5), ("paper", 3, 8)),
    # il tesoro di un'isola in mare aperto: oro, gemme e roba da marinaio
    "isola": (("gold_ingot", 6, 18), ("emerald", 3, 9), ("diamond", 2, 5),
              ("golden_apple", 1, 2), ("compass", 1, 1), ("spyglass", 1, 1),
              ("name_tag", 1, 2), ("ender_pearl", 1, 3), ("heart_of_the_sea", 1, 1),
              ("music_disc_wait", 1, 1), ("experience_bottle", 4, 12), ("nautilus_shell", 1, 2),
              ("saddle", 1, 1), ("gold_block", 1, 2), ("iron_ingot", 4, 12)),
    # le botteghe del paese: il bottino segue il mestiere di chi ci abita
    "fabbro": (("iron_ingot", 3, 10), ("coal", 4, 16), ("iron_pickaxe", 1, 1),
               ("iron_axe", 1, 1), ("iron_shovel", 1, 1), ("iron_hoe", 1, 1),
               ("stone_pickaxe", 1, 1), ("bucket", 1, 1), ("shears", 1, 1),
               ("flint_and_steel", 1, 1), ("copper_ingot", 2, 8), ("redstone", 2, 8)),
    "armaiolo": (("iron_sword", 1, 1), ("stone_sword", 1, 1), ("bow", 1, 1), ("arrow", 8, 24),
                 ("shield", 1, 1), ("iron_ingot", 2, 8), ("flint", 3, 9), ("coal", 3, 10),
                 ("leather", 2, 6), ("crossbow", 1, 1)),
    "corazzaio": (("iron_helmet", 1, 1), ("iron_chestplate", 1, 1), ("iron_leggings", 1, 1),
                  ("iron_boots", 1, 1), ("iron_ingot", 3, 10), ("coal", 4, 14),
                  ("shield", 1, 1), ("leather", 3, 8), ("chainmail_helmet", 1, 1)),
    "falegname": (("oak_planks", 8, 24), ("oak_log", 4, 12), ("stick", 6, 20),
                  ("stone_axe", 1, 1), ("wooden_axe", 1, 1), ("arrow", 8, 24),
                  ("feather", 3, 9), ("flint", 3, 9), ("bow", 1, 1), ("ladder", 4, 12),
                  ("chest", 1, 2), ("barrel", 1, 1), ("spruce_planks", 6, 16)),
    "scalpellino": (("cobblestone", 8, 32), ("stone", 8, 24), ("stone_bricks", 6, 20),
                    ("stone_pickaxe", 1, 1), ("iron_pickaxe", 1, 1), ("coal", 3, 10),
                    ("brick", 4, 12), ("andesite", 8, 16), ("sand", 6, 18)),
    "macellaio": (("cooked_beef", 3, 8), ("cooked_porkchop", 3, 8), ("cooked_chicken", 2, 6),
                  ("beef", 3, 8), ("leather", 2, 6), ("bone", 2, 8), ("coal", 4, 12),
                  ("mutton", 2, 6), ("rabbit", 1, 4)),
    "pescatore": (("cod", 3, 8), ("salmon", 2, 6), ("cooked_cod", 3, 8), ("fishing_rod", 1, 1),
                  ("string", 2, 8), ("bucket", 1, 1), ("lily_pad", 2, 6), ("kelp", 3, 9),
                  ("ink_sac", 1, 4), ("water_bucket", 1, 1)),
    "fruttivendolo": (("wheat", 6, 20), ("carrot", 4, 12), ("potato", 4, 12), ("apple", 3, 8),
                      ("beetroot", 3, 8), ("melon_slice", 4, 12), ("pumpkin", 1, 3),
                      ("bread", 2, 6), ("stone_hoe", 1, 1), ("wheat_seeds", 4, 16),
                      ("bone_meal", 3, 10)),
    "pastore": (("white_wool", 4, 16), ("shears", 1, 1), ("lead", 1, 3), ("leather", 2, 6),
                ("mutton", 2, 6), ("wheat", 3, 8), ("string", 2, 8), ("red_dye", 2, 6),
                ("blue_dye", 2, 6), ("hay_block", 1, 3)),
    "libraio": (("book", 3, 8), ("paper", 4, 12), ("ink_sac", 1, 4), ("compass", 1, 1),
                ("clock", 1, 1), ("writable_book", 1, 1), ("candle", 2, 6),
                ("feather", 2, 6), ("bookshelf", 1, 2)),
    "cartografo": (("paper", 6, 16), ("map", 1, 2), ("compass", 1, 1), ("spyglass", 1, 1),
                   ("ink_sac", 1, 4), ("string", 2, 6), ("feather", 2, 6),
                   ("glass_pane", 4, 10), ("clock", 1, 1)),
    "conciatore": (("leather", 4, 16), ("rabbit_hide", 2, 8), ("leather_helmet", 1, 1),
                   ("leather_chestplate", 1, 1), ("leather_boots", 1, 1), ("string", 2, 8),
                   ("bone", 2, 6), ("red_dye", 2, 6), ("brown_dye", 2, 6)),
    "speziale": (("glass_bottle", 3, 8), ("redstone", 3, 9), ("glowstone_dust", 2, 6),
                 ("sugar", 3, 9), ("spider_eye", 1, 4), ("gunpowder", 2, 6),
                 ("dandelion", 2, 6), ("brown_mushroom", 2, 6), ("honey_bottle", 1, 3),
                 ("fermented_spider_eye", 1, 2)),
    "castello": (("iron_sword", 1, 1), ("shield", 1, 1), ("golden_apple", 1, 2),
                 ("diamond", 1, 3), ("emerald", 2, 6), ("book", 2, 5),
                 ("arrow", 12, 32), ("bow", 1, 1), ("iron_chestplate", 1, 1),
                 ("experience_bottle", 2, 6), ("saddle", 1, 1), ("ender_pearl", 1, 3),
                 ("gold_ingot", 3, 9), ("cooked_beef", 4, 10)),
}


def _oggetti_zona(rng: np.random.Generator, tipo: str, lontano: bool) -> list:
    """4-7 pile dal bottino del luogo, in slot diversi; lontano dalle citta' anche
    un tesoro incantato (come per le cripte)."""
    tabella = BOTTINO_ZONA[tipo]
    n = int(rng.integers(4, 8))
    slot = [int(v) for v in rng.choice(27, size=n, replace=False)]
    tipi = rng.choice(len(tabella), size=min(n, len(tabella)), replace=False)
    contenuto = [(s, tabella[int(t)][0],
                  int(rng.integers(tabella[int(t)][1], tabella[int(t)][2] + 1)))
                 for s, t in zip(slot, tipi)]
    if tipo == "isola":
        # un tesoro vero: sempre oro e almeno una gemma, qualunque sia il resto
        contenuto[0] = (contenuto[0][0], "gold_ingot", int(rng.integers(6, 19)))
        if len(contenuto) > 1:
            contenuto[1] = (contenuto[1][0], "diamond" if rng.random() < 0.5 else "emerald",
                            int(rng.integers(2, 6)))
    if lontano:
        contenuto = _aggiungi_tesori(contenuto, rng, 1)
    return contenuto


def _oggetti_cripta(rng: np.random.Generator) -> list[tuple[int, str, int]]:
    """3-6 pile di ossa, refurtiva e ogni tanto un oggetto raro."""
    n = int(rng.integers(3, 7))
    slot = [int(v) for v in rng.choice(27, size=n, replace=False)]
    tipi = rng.choice(len(BOTTINO_CRIPTA), size=n, replace=False)
    return [(s, BOTTINO_CRIPTA[int(t)][0],
             int(rng.integers(BOTTINO_CRIPTA[int(t)][1], BOTTINO_CRIPTA[int(t)][2] + 1)))
            for s, t in zip(slot, tipi)]


def componenti_oggetto(spec: dict) -> dict:
    """I componenti (formato 1.21) di un oggetto d'incarico: nome, descrizione e
    un'etichetta `custom_data` con cui un villager lo riconosce in uno scambio.
    `spec` = {"id", "nome", "lore": [..], "uid"}."""
    comp = {"custom_name": spec["nome"], "lore": list(spec.get("lore", [])),
            "custom_data": {"incarico": spec["uid"]}}
    if "map_id" in spec:                  # una mappa: il numero del suo file (vedi `mappe.py`)
        comp["map_id"] = int(spec["map_id"])
    return comp


TITOLO_LIBRO_MAX = 32          # oltre, il gioco scarta TUTTA la pila ('too long: 36, expected [0-32]')


def limita_libro(comp: dict | None) -> dict | None:
    """Una copia dei componenti con il titolo di un libro scritto al massimo di 32
    caratteri. Chi lo supera fa fallire la lettura dell'inventario, e il libro sparisce."""
    if not comp or "written_book_content" not in comp:
        return comp
    c = dict(comp)
    wb = dict(c["written_book_content"])
    t = wb.get("title")
    if isinstance(t, dict) and isinstance(t.get("raw"), str):
        wb["title"] = {**t, "raw": t["raw"][:TITOLO_LIBRO_MAX]}
    elif isinstance(t, str):
        wb["title"] = t[:TITOLO_LIBRO_MAX]
    c["written_book_content"] = wb
    return c


def _voce(v):
    """Una pila come CompoundTag; il quarto elemento, se c'e', sono i componenti."""
    from amulet_nbt import ByteTag, CompoundTag, IntTag, StringTag
    s, nome, quanti = v[0], v[1], v[2]
    d = {"Slot": ByteTag(s), "id": StringTag(f"minecraft:{nome}"),
         "count": IntTag(quanti)}
    if len(v) > 3 and v[3]:
        d["components"] = _tag({f"minecraft:{k}": x for k, x in limita_libro(v[3]).items()})
    return CompoundTag(d)


def _e_dispensa(gx, gy, gz, minerali, campi, cripte) -> bool:
    """Vero se il forziere NON e' di una miniera, di un campo o di una cripta."""
    if minerali and (gx, gy, gz) in minerali:
        return False
    if cripte and any((gx - cx) ** 2 + (gz - cz) ** 2 <= r * r for cx, cz, r in cripte):
        return False
    if campi and any((gx - cx) ** 2 + (gz - cz) ** 2 <= r * r for cx, cz, r in campi):
        return False
    return True


def _seme(x: int, y: int, z: int, seed: int) -> int:
    """Deterministico sulla posizione: lo stesso forziere ha sempre lo
    stesso contenuto, anche rigenerando lo stesso lotto. Mascherato per le
    stesse ragioni di `sottosuolo._seme` - meta' del mondo ha coordinate
    negative."""
    return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791) ^
            (seed * 999999937)) & 0x7FFFFFFF


def trova(blocchi: np.ndarray, tav: Tavolozza, ox: int, oz: int, y0: int,
         seed: int = 0, campi: list | None = None, minerali: set | None = None,
         cripte: list | None = None, centri: list | None = None,
         dungeon: dict | None = None, libri: dict | None = None,
         libro_citta: tuple | None = None, oggetti: dict | None = None,
         zone: list | None = None) -> list:
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
    contenitori = tav.singolo | tav.doppio | tav.botti
    if not contenitori:
        return fuori
    botti = np.isin(blocchi, list(tav.botti))
    posizioni = np.transpose(np.nonzero(np.isin(blocchi, list(contenitori))))
    for lx, ly, lz in posizioni:
        gx, gy, gz = ox + int(lx), y0 + int(ly), oz + int(lz)
        rng = np.random.default_rng(_seme(gx, gy, gz, seed))
        # il bottino dipende da DOVE sta il forziere: in una miniera minerale e
        # gemme, in un accampamento un diamante e un'arma, in casa la dispensa
        lontano = bool(centri) and min((gx - cx) ** 2 + (gz - cz) ** 2
                                       for cx, cz in centri) > DISTANZA_TESORI ** 2
        zona = next((t for x0, z0, x1, z1, t in (zone or ())
                     if x0 <= gx < x1 and z0 <= gz < z1), None)
        if dungeon and (gx, gy, gz) in dungeon:
            contenuto = _oggetti_dungeon(rng, dungeon[(gx, gy, gz)], lontano)
        elif zona is not None:
            contenuto = _oggetti_zona(rng, zona, lontano)
        elif minerali and (gx, gy, gz) in minerali:
            contenuto = _oggetti_miniera(rng)
        elif cripte and any((gx - cx) ** 2 + (gz - cz) ** 2 <= r * r for cx, cz, r in cripte):
            contenuto = _oggetti_cripta(rng)
        elif campi and any((gx - cx) ** 2 + (gz - cz) ** 2 <= r * r for cx, cz, r in campi):
            contenuto = _oggetti_campo(rng)
        else:
            contenuto = _oggetti(rng)
        # lontano da ogni citta' il bottino di miniere, campi e cripte ha
        # qualche tesoro incantato (le case, per definizione, sono vicine)
        if (centri and not (dungeon and (gx, gy, gz) in dungeon)
                and not _e_dispensa(gx, gy, gz, minerali, campi, cripte)):
            if min((gx - cx) ** 2 + (gz - cz) ** 2 for cx, cz in centri)                     > DISTANZA_TESORI ** 2:
                contenuto = _aggiungi_tesori(contenuto, rng)
        libro = None
        if libri and (gx, gy, gz) in libri:
            libro = libri[(gx, gy, gz)]
        elif libro_citta is not None and not (
                dungeon and (gx, gy, gz) in dungeon) and _e_dispensa(
                    gx, gy, gz, minerali, campi, cripte):
            cx, cz, raggio, spec = libro_citta
            if ((gx - cx) ** 2 + (gz - cz) ** 2 <= (raggio * 0.8) ** 2
                    and _seme(gx, gy, gz, 5) % 5 == 0):
                libro = spec
        if libro is not None:
            liberi = [s for s in range(27) if s not in {c[0] for c in contenuto}]
            if liberi:
                contenuto = list(contenuto) + [(liberi[0], "written_book", 1, {
                    "written_book_content": {
                        "title": {"raw": libro.titolo}, "author": libro.autore,
                        "pages": [{"raw": p} for p in libro.pagine]}})]
        if oggetti and (gx, gy, gz) in oggetti:
            for spec in oggetti[(gx, gy, gz)]:
                liberi = [s for s in range(27) if s not in {c[0] for c in contenuto}]
                if liberi:
                    contenuto = list(contenuto) + [(liberi[0], spec["id"], 1,
                                                    componenti_oggetto(spec))]
        items = ListTag([_voce(v) for v in contenuto])
        # Namespace "minecraft" (raw) con NBT piatto si perde - verificato
        # con uno spike dedicato - non appena la sessione di scrittura tocca
        # PIU' di un chunk (con un solo chunk sopravvive, che e' perche' il
        # difetto e' passato inosservato finche' non si e' provato su una
        # citta' vera). Il namespace "universal_minecraft" con l'NBT
        # incapsulato in "utags" e' la stessa forma che amulet produce da
        # solo quando genera un block-entity dal proprio traduttore
        # universale, ed e' l'unica delle due che sopravvive al giro
        # salva/rileggi su un mondo multi-chunk.
        e_botte = bool(botti[lx, ly, lz])
        utags = CompoundTag({
            "isMovable": ByteTag(1),
            **({} if e_botte else {"Findable": ByteTag(0)}),
            "Items": items,
            "Lock": StringTag(""),
        })
        nbt = NamedTag(CompoundTag({"utags": utags}))
        fuori.append(BlockEntity("universal_minecraft", "barrel" if e_botte else "chest",
                                 gx, gy, gz, nbt))
    fuori.extend(_forni(blocchi, tav, ox, oz, y0, seed))
    return fuori


def _forni(blocchi: np.ndarray, tav: Tavolozza, ox: int, oz: int, y0: int, seed: int) -> list:
    """I forni, le fornaci e gli affumicatoi gia' scritti nel chunk, col carbone
    nello slot del combustibile (1): da vuoti non servono a niente."""
    from amulet.api.block_entity import BlockEntity
    from amulet_nbt import ByteTag, CompoundTag, IntTag, ListTag, NamedTag, StringTag

    fuori = []
    for nome, ids in tav.forni.items():
        if not ids:
            continue
        for lx, ly, lz in np.transpose(np.nonzero(np.isin(blocchi, list(ids)))):
            gx, gy, gz = ox + int(lx), y0 + int(ly), oz + int(lz)
            rng = np.random.default_rng(_seme(gx, gy, gz, seed + 31))
            carbone = CompoundTag({"Slot": ByteTag(1), "id": StringTag("minecraft:coal"),
                                   "count": IntTag(int(rng.integers(8, 25)))})
            utags = CompoundTag({"isMovable": ByteTag(1), "Items": ListTag([carbone]),
                                 "Lock": StringTag("")})
            fuori.append(BlockEntity("universal_minecraft", nome, gx, gy, gz,
                                     NamedTag(CompoundTag({"utags": utags}))))
    return fuori
