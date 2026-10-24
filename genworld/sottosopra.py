"""Il SOTTOSOPRA: il Nether come specchio dell'overworld.

Dove sopra c'e' una citta', un fiume, una miniera, sotto c'e' la stessa cosa fatta con i
materiali del Nether: pietra diventa netherrack, erba nylium, legno funghi cremisi o distorti,
mattoni pietra nera e acqua diventa LAVA. Non e' una dimensione disegnata a parte: si prendono
gli STESSI chunk dell'overworld, gia' costruiti, e se ne cambiano i blocchi con una tabella.
Cosi' tutto quello che c'e' sopra c'e' anche sotto, alle stesse coordinate.

Perche' le coordinate coincidano il Nether deve avere scala 1:1 (non 1:8) e la stessa altezza
dell'overworld (da -64 a 320): lo fa un datapack che sovrascrive il tipo di dimensione
`minecraft:the_nether` (vedi `scrivi_datapack`). Senza datapack il Nether resta quello di prima.

La tabella e' per NOME di blocco (formato universale di amulet) e per MATERIALE: un `stairs`
di `oak` diventa un `stairs` di `crimson`. Quello che non e' in tabella resta com'e': i blocchi
sconosciuti passano senza danni.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

# --------------------------------------------------------------------------
# Il datapack: Nether a scala 1:1 e alto come l'overworld
# --------------------------------------------------------------------------

PACK_MCMETA = {"pack": {"description": "Sottosopra: il Nether come specchio dell'overworld",
                        "min_format": [107, 0], "max_format": 107}}

# copia del tipo di dimensione del gioco 26.2 (`data/minecraft/dimension_type/the_nether.json`),
# con scala 1, altezza da -64 a 320 e senza il soffitto
DIMENSIONE_NETHER = {
    "ambient_light": 0.1,
    "attributes": {
        "minecraft:gameplay/bed_rule": {"can_set_spawn": "never", "can_sleep": "never", "explodes": True},
        "minecraft:gameplay/can_start_raid": False,
        "minecraft:gameplay/fast_lava": True,
        "minecraft:gameplay/piglins_zombify": False,
        "minecraft:gameplay/respawn_anchor_works": True,
        "minecraft:gameplay/sky_light_level": 4.0,
        "minecraft:gameplay/snow_golem_melts": True,
        "minecraft:gameplay/water_evaporates": True,
        "minecraft:visual/ambient_light_color": "#3a2018",
        "minecraft:visual/default_dripstone_particle": {"type": "minecraft:dripping_dripstone_lava"},
        "minecraft:visual/fog_end_distance": 128.0,
        "minecraft:visual/fog_start_distance": 8.0,
        "minecraft:visual/sky_light_color": "#7a7aff",
        "minecraft:visual/sky_light_factor": 0.0,
    },
    "cardinal_light": "nether",
    "coordinate_scale": 1.0,
    "has_ceiling": False,
    "has_ender_dragon_fight": False,
    "has_fixed_time": True,
    "has_skylight": False,
    "height": 384,
    "infiniburn": "#minecraft:infiniburn_nether",
    "logical_height": 384,
    "min_y": -64,
    "monster_spawn_block_light_limit": 15,
    "monster_spawn_light_level": 7,
    "skybox": "none",
    "timelines": "#minecraft:in_nether",
}


def scrivi_datapack(percorso_mondo: str) -> str:
    """Crea `<mondo>/datapacks/sottosopra/`. Ritorna la cartella. Idempotente."""
    base = os.path.join(percorso_mondo, "datapacks", "sottosopra")
    tipo = os.path.join(base, "data", "minecraft", "dimension_type")
    os.makedirs(tipo, exist_ok=True)
    with open(os.path.join(base, "pack.mcmeta"), "w", encoding="utf-8") as f:
        json.dump(PACK_MCMETA, f, indent=2)
    with open(os.path.join(tipo, "the_nether.json"), "w", encoding="utf-8") as f:
        json.dump(DIMENSIONE_NETHER, f, indent=2)
    return base


# --------------------------------------------------------------------------
# La tabella dei materiali
# --------------------------------------------------------------------------

LEGNI = {"oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "bamboo",
         "bamboo_mosaic", "pale_oak", "poplar"}
MATERIALI_PIETRA = {
    "cobblestone": "blackstone", "mossy_cobblestone": "blackstone", "stone": "blackstone",
    "stone_brick": "polished_blackstone_brick", "mossy_stone_brick": "polished_blackstone_brick",
    "andesite": "blackstone", "diorite": "polished_blackstone", "granite": "blackstone",
    "polished_andesite": "polished_blackstone", "polished_diorite": "polished_blackstone",
    "polished_granite": "polished_blackstone", "brick": "nether_brick", "mud_brick": "nether_brick",
    "sandstone": "red_nether_brick", "red_sandstone": "red_nether_brick",
    "smooth_sandstone": "red_nether_brick", "smooth_red_sandstone": "red_nether_brick",
    "cobbled_deepslate": "blackstone", "polished_deepslate": "polished_blackstone",
    "deepslate_tile": "polished_blackstone_brick", "deepslate_brick": "polished_blackstone_brick",
    "tuff": "blackstone", "tuff_brick": "polished_blackstone_brick", "polished_tuff": "polished_blackstone",
    "end_stone_brick": "nether_brick", "prismarine": "polished_blackstone",
    "prismarine_brick": "polished_blackstone_brick", "dark_prismarine": "polished_blackstone",
}
# blocchi a due proprieta' di stato che hanno una versione col MATERIALE
FAMIGLIE_MATERIALE = {"stairs", "slab", "fence", "fence_gate", "door", "trapdoor", "wall", "planks",
                      "log", "wood", "button", "pressure_plate", "sign", "wall_sign", "hanging_sign",
                      "wall_hanging_sign", "standing_sign"}

# nome -> nome (stesse proprieta' valide, le altre si scartano)
SEMPLICI = {
    "stone": "netherrack", "dirt": "netherrack", "coarse_dirt": "netherrack", "podzol": "netherrack",
    "mycelium": "netherrack", "rooted_dirt": "netherrack", "mud": "soul_soil", "clay": "netherrack",
    "dirt_path": "soul_soil", "grass_path": "soul_soil", "farmland": "soul_soil",
    "sand": "soul_sand", "red_sand": "soul_sand", "sandstone": "soul_soil", "red_sandstone": "soul_soil",
    "deepslate": "blackstone", "cobbled_deepslate": "blackstone", "tuff": "smooth_basalt",
    "calcite": "smooth_basalt", "andesite": "smooth_basalt", "diorite": "smooth_basalt",
    "granite": "netherrack", "cobblestone": "blackstone", "mossy_cobblestone": "blackstone",
    "stone_bricks": "polished_blackstone_bricks", "bricks": "nether_bricks", "brick_block": "nether_bricks",
    "snow_block": "calcite", "snow": "air", "powder_snow": "air", "ice": "smooth_basalt",
    "packed_ice": "smooth_basalt", "blue_ice": "smooth_basalt", "frosted_ice": "smooth_basalt",
    "kelp": "air", "kelp_plant": "air", "seagrass": "air", "tall_seagrass": "air", "sea_pickle": "air",
    "bubble_column": "air", "lily_pad": "air", "vine": "air", "glow_lichen": "air", "moss_carpet": "air",
    "big_dripleaf": "air", "small_dripleaf": "air", "azalea": "air", "flowering_azalea": "air",
    "double_plant": "air", "tall_grass": "air", "short_grass": "nether_sprouts", "fern": "nether_sprouts",
    "grass": "nether_sprouts", "dead_bush": "nether_sprouts", "sweet_berry_bush": "air",
    "sugar_cane": "air", "cactus": "air", "bamboo": "air", "bamboo_sapling": "air",
    "wheat": "nether_wart", "carrots": "nether_wart", "potatoes": "nether_wart", "beetroots": "nether_wart",
    "melon_stem": "nether_wart", "pumpkin_stem": "nether_wart", "attached_melon_stem": "nether_wart",
    "attached_pumpkin_stem": "nether_wart",
    "pumpkin": "shroomlight", "carved_pumpkin": "shroomlight", "melon_block": "shroomlight",
    "melon": "shroomlight", "hay_block": "nether_wart_block", "moss_block": "warped_wart_block",
    "coal_ore": "nether_quartz_ore", "deepslate_coal_ore": "nether_quartz_ore",
    "iron_ore": "nether_gold_ore", "deepslate_iron_ore": "nether_gold_ore",
    "copper_ore": "nether_gold_ore", "deepslate_copper_ore": "nether_gold_ore",
    "gold_ore": "nether_gold_ore", "deepslate_gold_ore": "nether_gold_ore",
    "lapis_ore": "nether_quartz_ore", "deepslate_lapis_ore": "nether_quartz_ore",
    "redstone_ore": "nether_quartz_ore", "deepslate_redstone_ore": "nether_quartz_ore",
    "diamond_ore": "ancient_debris", "deepslate_diamond_ore": "ancient_debris",
    "emerald_ore": "ancient_debris", "deepslate_emerald_ore": "ancient_debris",
    "torch": "soul_torch", "wall_torch": "soul_torch", "lantern": "soul_lantern", "campfire": "soul_campfire",
    "fire": "soul_fire", "water": "lava", "flowing_water": "lava",
}
FIORI = ("flower", "tulip", "orchid", "allium", "azure", "lily", "rose", "peony", "lilac", "poppy",
         "dandelion", "bluet", "cornflower", "sunflower", "petals", "torchflower", "wildflower",
         "spore", "cherry_leaves_litter")
ACQUATICI = ("coral", "kelp", "seagrass", "bubble")


def _spec(nome: str):
    return _TRADUTTORE().universal_format.block.get_specification("universal_minecraft", nome)


_TM = []


def _TRADUTTORE():
    if not _TM:
        import PyMCTranslate
        _TM.append(PyMCTranslate.new_translation_manager())
    return _TM[0]


_SPEC_CACHE: dict = {}


def _proprieta_valide(nome: str, props: dict) -> dict | None:
    """Tiene solo le proprieta' che `nome` conosce, con valori ammessi; None se il blocco non esiste."""
    if nome not in _SPEC_CACHE:
        try:
            sp = _spec(nome)
        except Exception:
            sp = None
        _SPEC_CACHE[nome] = sp
    sp = _SPEC_CACHE[nome]
    if sp is None:
        return None
    ammessi = sp.get("properties", {})
    fuori = {}
    for k, v in props.items():
        if k in ammessi and f'"{v}"' in ammessi[k]:
            fuori[k] = v
    return fuori


def traduci(nome: str, props: dict, variante: int = 0) -> tuple[str, dict]:
    """(nome, proprieta') dell'overworld -> quelli del Sottosopra. `variante`: 0 cremisi, 1 distorto.
    `air` come nome vuol dire "togli il blocco"."""
    legno = ("crimson", "warped")[variante & 1]
    nylium = ("crimson_nylium", "warped_nylium")[variante & 1]
    radici = ("crimson_roots", "warped_roots")[variante & 1]
    parete = ("nether_wart_block", "warped_wart_block")[variante & 1]
    p = {k: str(v) for k, v in props.items()}
    p.pop("waterlogged", None)                  # dove c'era acqua ora c'e' lava: niente blocchi "bagnati"

    if nome == "air" or nome.endswith("_air"):
        return nome, {}
    if nome == "grass_block":
        return nylium, {}
    if nome == "leaves":
        return parete, {}
    if nome == "plant":                                   # fiori e arbusti: radici del Nether
        pt = p.get("plant_type", "")
        return (radici, {}) if pt in ("dandelion", "poppy", "fern", "grass") else ("nether_sprouts", {})
    if nome == "sapling":
        return ("crimson_fungus", "warped_fungus")[variante & 1], {}
    if nome == "water" or nome == "flowing_water":
        return "lava", {"level": p.get("level", "0")}
    if nome in FAMIGLIE_MATERIALE:
        m = p.get("material")
        if m in LEGNI:
            p["material"] = legno
        elif m in MATERIALI_PIETRA:
            p["material"] = MATERIALI_PIETRA[m]
        return nome, p
    if nome == "stained_glass" or nome == "glass" or nome == "glass_pane":
        return nome, p
    if nome in ("pointed_dripstone",):
        return nome, p
    if nome.startswith(ACQUATICI) or any(k in nome for k in ("coral", "seagrass", "kelp")):
        return "air", {}
    if any(k in nome for k in FIORI) and nome not in ("flower_pot",):
        return "air", {}
    if nome == "lantern":
        return "soul_lantern", {"hanging": p.get("hanging", "false")}
    if nome in SEMPLICI:
        nuovo = SEMPLICI[nome]
        if nuovo == "nether_wart":
            return nuovo, {"age": "3"}
        return nuovo, {}
    return nome, p


# --------------------------------------------------------------------------
# Il decadimento
# --------------------------------------------------------------------------

# Cio' che ha costruito una persona e quindi, di la', e' in rovina. Si decide dal blocco di
# partenza: un tronco di bosco o una roccia non sono costruzioni, un muro si'.
COSTRUZIONI = {"stone_bricks", "bricks", "brick_block", "planks", "stairs", "slab", "fence",
               "fence_gate", "wall", "door", "trapdoor", "glass", "glass_pane", "stained_glass",
               "stained_glass_pane", "bars", "ladder", "carpet", "wool", "bed", "bookshelf",
               "lantern", "torch", "wall_torch", "hay_block", "wall_banner", "banner"}
# quanto di tutto cio' si perde al massimo, con decadimento 1.0
PERDITA_MASSIMA = 0.55


def _macchie(x: np.ndarray, z: np.ndarray, seme: int) -> np.ndarray:
    """Un campo continuo in 0..1 con macchie larghe una trentina di blocchi: dove e' alto il
    decadimento e' forte, dove e' basso un edificio resta quasi intero."""
    ph = (seme * 0.001) % 6.283
    n = (np.sin(x * 0.045 + ph) * np.cos(z * 0.052 - ph) + np.sin((x + z) * 0.031 + 2 * ph)
         + 0.6 * np.sin(x * 0.11 - z * 0.09 + ph * 3)) / 2.6
    return np.clip(0.5 + 0.5 * n, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Lo specchio dei blocchi di un mondo
# --------------------------------------------------------------------------

class Specchio:
    """Tabella id-di-palette -> id-di-palette per un `ScrittoreMondo`. Si aggiorna da sola quando
    compaiono blocchi nuovi nel palette."""

    VARIANTI = 2

    def __init__(self, scrittore, decadimento: float = 0.0, seme: int = 0) -> None:
        self.s = scrittore
        self.decadimento = float(min(max(decadimento, 0.0), 1.0))
        self.seme = int(seme)
        self._visti = 0
        self._fatti: set = set()
        self._lut = [np.zeros(0, np.uint32) for _ in range(self.VARIANTI)]
        self._costr = np.zeros(0, bool)            # per id di partenza: e' una costruzione?
        self._biomi: dict = {}
        self._vite = {}

    def _aggiorna(self) -> None:
        cache = self.s._cache_blocchi
        if len(cache) == self._visti:
            return
        # la traduzione puo' creare id nuovi (le destinazioni): si ripete finche' la cache si ferma
        while len(cache) != self._visti:
            self._visti = len(cache)
            elementi = [(k, v) for k, v in list(cache.items()) if v not in self._fatti]
            if not elementi:
                break
            massimo = max(v for _, v in list(cache.items())) + 1
            for var in range(self.VARIANTI):
                if len(self._lut[var]) < massimo:
                    lut = np.arange(massimo, dtype=np.uint32)
                    lut[:len(self._lut[var])] = self._lut[var]
                    self._lut[var] = lut
            if len(self._costr) < massimo:
                nuova = np.zeros(massimo, bool)
                nuova[:len(self._costr)] = self._costr
                self._costr = nuova
            for (nome, props), idx in elementi:
                self._fatti.add(idx)
                self._costr[idx] = nome in COSTRUZIONI
                for var in range(self.VARIANTI):
                    self._lut[var][idx] = self._id_traduzione(nome, dict(props), var)
            # id creati dalle traduzioni: la tabella deve poterli indicizzare
            nuovo_max = max(v for _, v in list(cache.items())) + 1
            for var in range(self.VARIANTI):
                if len(self._lut[var]) < nuovo_max:
                    lut = np.arange(nuovo_max, dtype=np.uint32)
                    lut[:len(self._lut[var])] = self._lut[var]
                    self._lut[var] = lut

    def _id_traduzione(self, nome: str, props: dict, var: int) -> int:
        n2, p2 = traduci(nome, props, var)
        if n2 == "air":
            return self.s.id_aria
        valide = _proprieta_valide(n2, p2)
        if valide is None:                      # il nome nuovo non esiste: si lascia com'era
            return self.s.blocco(nome, **{k: str(v) for k, v in props.items()})
        return self.s.blocco(n2, **valide)

    def specchia(self, blocchi: np.ndarray, variante: int = 0, cx: int = 0, cz: int = 0) -> np.ndarray:
        """Il chunk specchiato: stessi indici, blocchi del Sottosopra. `cx, cz`: il chunk, per il
        decadimento (che dipende dal posto, non dall'ordine in cui si scrivono i chunk)."""
        self._aggiorna()
        lut = self._lut[variante & 1]
        massimo = int(blocchi.max()) + 1
        if massimo > len(lut):                  # palette cresciuta dopo l'ultimo aggiornamento
            self._visti = -1
            self._aggiorna()
            lut = self._lut[variante & 1]
        fuori = lut[blocchi]
        if self.decadimento > 0:
            fuori = self._decadi(blocchi, fuori, cx, cz)
        return fuori

    def _decadi(self, sorgente: np.ndarray, fuori: np.ndarray, cx: int, cz: int) -> np.ndarray:
        """Le costruzioni perdono pezzi: di piu' dove il campo a macchie e' alto e sui blocchi
        scoperti (aria sopra); sotto i pezzi rimasti pendono a volte liane del Nether."""
        costr = self._costr[sorgente]
        if not costr.any():
            return fuori
        rng = np.random.default_rng((self.seme * 1000003 + cx * 7919 + cz * 104729) & 0x7FFFFFFF)
        xs = (np.arange(16) + cx * 16)[:, None, None]
        zs = (np.arange(16) + cz * 16)[None, None, :]
        zona = _macchie(xs, zs, self.seme)                       # (16, 1, 16)
        p = PERDITA_MASSIMA * self.decadimento * (0.15 + 0.85 * zona ** 1.5)
        aria = self.s.id_aria
        scoperto = np.zeros(fuori.shape, bool)
        scoperto[:, :-1, :] = fuori[:, 1:, :] == aria
        scoperto[:, -1, :] = True
        p = np.broadcast_to(p, fuori.shape) * np.where(scoperto, 1.6, 0.55)
        via = costr & (rng.random(fuori.shape, dtype=np.float32) < p)
        fuori = fuori.copy()
        fuori[via] = aria
        # liane: sotto un pezzo di costruzione rimasto, con l'aria sotto
        resta = costr & ~via
        sotto_aria = np.zeros(fuori.shape, bool)
        sotto_aria[:, 1:, :] = fuori[:, :-1, :] == aria
        pendono = resta & sotto_aria & (rng.random(fuori.shape, dtype=np.float32) < 0.05 * self.decadimento)
        if pendono.any():
            liana = self.s.blocco("weeping_vines", age="3")
            pos = np.nonzero(pendono)
            fuori[pos[0], pos[1] - 1, pos[2]] = liana
        return fuori

    def biomi(self, id_overworld: list, nomi_overworld: tuple) -> np.ndarray:
        """Tabella indice-di-bioma (in `biomi.BIOMI`) -> id di palette del bioma del Sottosopra."""
        chiave = tuple(nomi_overworld)
        if chiave not in self._biomi:
            self._biomi[chiave] = np.array(
                [self.s.bioma(bioma_specchio(n)) for n in nomi_overworld], np.uint32)
        return self._biomi[chiave]


def bioma_specchio(nome: str) -> str:
    if any(k in nome for k in ("forest", "jungle")) and "birch" not in nome:
        return "crimson_forest"
    if "birch" in nome or "mangrove" in nome or "swamp" in nome:
        return "warped_forest"
    if any(k in nome for k in ("snowy", "tundra", "taiga", "frozen", "ice", "beach")):
        return "soul_sand_valley"
    if any(k in nome for k in ("desert", "badlands", "mountain", "peak", "slope", "savanna")):
        return "basalt_deltas"
    return "nether"


def variante_chunk(cx: int, cz: int) -> int:
    """Quale foresta del Nether (0 cremisi, 1 distorta): a macchie di 8 x 8 chunk."""
    h = ((cx // 8) * 73856093) ^ ((cz // 8) * 19349663)
    return (h >> 5) & 1


# --------------------------------------------------------------------------
# Le creature
# --------------------------------------------------------------------------

AQUATICHE = {"merluzzo", "salmone", "pesce_tropicale", "calamaro", "delfino", "pesce_palla",
             "calamaro_luminoso", "tartaruga", "rana", "ape"}
GREGGE = {"mucca", "pecora", "gallina", "maiale", "capra", "lama", "asino", "cavallo", "cammello",
          "coniglio", "volpe", "lupo", "gatto", "ocelotto", "panda", "armadillo", "orso_polare",
          "lama_mercante", "mercante", "pappagallo"}


def specchia_creature(creature: list) -> list:
    """Le creature del Sottosopra: gli abitanti e gli animali diventano piglin e hoglin; i
    mostri restano. Chi viveva in acqua sparisce (li' c'e' lava)."""
    from .entita import Abitante
    from .fauna import Animale
    from .torri import Combattente
    fuori = []
    for k, c in enumerate(creature):
        sp = getattr(c, "specie", None)
        if isinstance(c, Abitante):
            fuori.append(Combattente(x=c.x, y=c.y, z=c.z, specie="piglin",
                                     seme=(c.seme * 31 + 7) & 0x7FFFFFFF))
        elif isinstance(c, Animale):
            if sp in AQUATICHE:
                continue
            if sp in GREGGE:
                fuori.append(Combattente(x=c.x, y=c.y, z=c.z,
                                         specie="hoglin" if sp in ("mucca", "maiale", "capra", "lama") else "piglin",
                                         seme=(c.seme * 31 + 7) & 0x7FFFFFFF))
            else:
                fuori.append(copy.copy(c))
        else:
            fuori.append(copy.copy(c))
    return fuori
