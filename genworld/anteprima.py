"""Renderer top-down del mondo generato.

Lo strumento di verifica piu' utile del progetto: produce un PNG di quello che
c'e' davvero nei file region, cosi' si controlla una generazione in due secondi
invece di avviare Minecraft. Rilegge il mondo salvato, non gli array in
memoria: se la scrittura e' sbagliata, si vede.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import amulet

# Colori per blocco di superficie (RGB).
COLORI = {
    # arredi: si vedono solo dall'alto attraverso un buco nel tetto, ma
    # senza un colore il renderer li segnala come blocchi ignoti
    "bed": (168, 60, 60), "bookshelf": (140, 104, 66), "chest": (150, 110, 60),
    "crafting_table": (146, 108, 72), "furnace": (110, 110, 110),
    "ladder": (150, 122, 80),
    # botteghe e mercato
    "anvil": (72, 72, 78), "barrel": (140, 106, 62), "brewing_stand": (120, 110, 130),
    "campfire": (190, 110, 50), "cartography_table": (150, 126, 94),
    "cauldron": (70, 70, 74), "composter": (132, 106, 60),
    "fletching_table": (186, 164, 110), "grindstone": (120, 118, 116),
    "hay_block": (206, 176, 62), "lectern": (160, 130, 80),
    "loom": (176, 152, 104), "melon": (110, 158, 60), "pumpkin": (214, 132, 40),
    "smoker": (110, 100, 96), "stonecutter": (122, 120, 118),
    "wool": (216, 216, 216), "blast_furnace": (96, 96, 100),
    "smithing_table": (70, 68, 78), "lantern": (236, 200, 120),
    "farmland": (110, 76, 50), "wheat": (198, 176, 90),
    "carrots": (196, 132, 52), "potatoes": (150, 168, 84),
    "beetroots": (150, 82, 70), "sweet_berry_bush": (96, 120, 62),
    # strati di roccia e sottosuolo: si vedono sulle pareti e nei tagli
    "andesite": (136, 136, 136), "diorite": (188, 188, 190),
    "granite": (152, 106, 88), "tuff": (108, 109, 99),
    "calcite": (223, 226, 217), "deepslate": (77, 77, 80),
    "smooth_basalt": (72, 72, 78),
    "coal_ore": (112, 112, 112), "iron_ore": (170, 140, 112),
    "copper_ore": (150, 130, 100), "gold_ore": (180, 158, 80),
    "redstone_ore": (150, 100, 100), "lapis_ore": (100, 120, 170),
    "diamond_ore": (120, 180, 186), "emerald_ore": (100, 170, 120),
    "deepslate_coal_ore": (70, 70, 72), "deepslate_iron_ore": (110, 96, 84),
    "deepslate_copper_ore": (100, 92, 76), "deepslate_gold_ore": (120, 106, 60),
    "deepslate_redstone_ore": (104, 70, 70), "deepslate_lapis_ore": (70, 84, 118),
    "deepslate_diamond_ore": (84, 124, 128), "deepslate_emerald_ore": (70, 118, 84),
    "grass_block": (106, 153, 78),
    "dirt": (134, 96, 67),
    "sand": (219, 207, 163),
    "stone": (128, 128, 128),
    "mossy_cobblestone": (104, 118, 96),
    "bedrock": (60, 60, 60),
    "water": (58, 104, 176),
    "sandstone": (205, 190, 140),
    "snow_block": (238, 242, 248),
    "powder_snow": (232, 238, 246),
    "packed_ice": (170, 205, 230),
    "clay": (160, 160, 175),
    "coarse_dirt": (120, 88, 62),
    "leaves": (44, 86, 40),
    "log": (86, 66, 44),
    "cactus": (72, 118, 58),
    "plant": (118, 152, 86),
    "planks": (176, 140, 92),
    "stairs": (150, 118, 76),
    "cobblestone": (132, 132, 132),
    "stone_bricks": (142, 142, 140),
    "sandstone_wall": (208, 196, 152),
    "door": (120, 88, 56),
    "glass_pane": (196, 220, 228),
    "glass": (196, 220, 228),
    "torch": (240, 208, 120),
    "fence": (150, 118, 76),
    "slab": (162, 130, 86),
    "grass_path": (172, 142, 92),
    "basalt": (58, 54, 58),
    "blackstone": (42, 36, 42),
    "magma_block": (176, 78, 30),
    "lava": (232, 120, 30),
    "obsidian": (28, 20, 40),
    "gravel": (150, 145, 140),
    "snow": (245, 245, 250),
    "air": (24, 26, 30),
}
COLORE_IGNOTO = (255, 0, 255)


def rendi(
    percorso_mondo: str,
    percorso_png: str,
    y0: int = -64,
    y1: int = 200,
    ombreggiatura: float = 0.5,
) -> dict:
    """Renderizza il mondo e ritorna qualche statistica di verifica."""
    livello = amulet.load_level(percorso_mondo)
    dim = "minecraft:overworld"
    coords = list(livello.all_chunk_coords(dim))
    if not coords:
        livello.close()
        raise RuntimeError("il mondo non contiene chunk")

    cxs = [c[0] for c in coords]
    czs = [c[1] for c in coords]
    cx_min, cx_max = min(cxs), max(cxs)
    cz_min, cz_max = min(czs), max(czs)
    larg = (cx_max - cx_min + 1) * 16
    alt = (cz_max - cz_min + 1) * 16

    altezze = np.zeros((alt, larg), dtype=np.float32)
    rgb = np.zeros((alt, larg, 3), dtype=np.uint8)

    # ATTENZIONE: il palette del livello e' VUOTO appena caricato e si popola
    # man mano che si leggono i chunk. Va quindi risolto pigramente, non in
    # anticipo: costruirlo prima del ciclo produce una mappa tutta ignota.
    palette = livello.block_palette
    colore_per_id: dict[int, tuple[int, int, int]] = {}
    aria_per_id: dict[int, bool] = {}
    ignoti: set[str] = set()

    def risolvi(bid: int) -> None:
        if bid in colore_per_id:
            return
        nome = palette[bid].base_name
        colore_per_id[bid] = COLORI.get(nome, COLORE_IGNOTO)
        aria_per_id[bid] = nome == "air"
        if nome not in COLORI:
            ignoti.add(nome)

    for cx, cz in coords:
        chunk = livello.get_chunk(cx, cz, dim)
        blocchi = np.asarray(chunk.blocks[:, y0:y1, :])      # (16, H, 16)

        for bid in np.unique(blocchi):
            risolvi(int(bid))
        ids_aria = [b for b, e in aria_per_id.items() if e]
        non_aria = ~np.isin(blocchi, ids_aria) if ids_aria else np.ones_like(blocchi, bool)
        # indice del blocco piu' alto non-aria, per colonna
        rovescio = non_aria[:, ::-1, :]
        ha = rovescio.any(axis=1)
        idx_dal_alto = rovescio.argmax(axis=1)
        y_cima = (blocchi.shape[1] - 1 - idx_dal_alto)
        ids_cima = np.take_along_axis(blocchi, y_cima[:, None, :], axis=1)[:, 0, :]

        ox = (cx - cx_min) * 16
        oz = (cz - cz_min) * 16
        for lx in range(16):
            for lz in range(16):
                if not ha[lx, lz]:
                    continue
                bid = int(ids_cima[lx, lz])
                rgb[oz + lz, ox + lx] = colore_per_id[bid]
                altezze[oz + lz, ox + lx] = y_cima[lx, lz] + y0

    # Due contributi, perche' la sola pendenza non basta a far vedere un
    # rilievo dolce: l'ombreggiatura da' la forma locale, la tinta altimetrica
    # da' la quota assoluta.
    if ombreggiatura > 0:
        gz, gx = np.gradient(altezze)
        luce = np.clip(1.0 + ombreggiatura * (gx + gz) / 3.0, 0.45, 1.55)

        valide = altezze > 0
        if valide.any():
            hmin, hmax = altezze[valide].min(), altezze.max()
            norm = np.clip((altezze - hmin) / max(1.0, hmax - hmin), 0, 1)
        else:
            norm = np.zeros_like(altezze)
        quota = 0.72 + 0.56 * norm

        rgb = np.clip(
            rgb.astype(np.float32) * (luce * quota)[:, :, None], 0, 255
        ).astype(np.uint8)

    Image.fromarray(rgb, "RGB").save(percorso_png)
    livello.close()

    return {
        "png": percorso_png,
        "larghezza": larg,
        "altezza": alt,
        "chunk": len(coords),
        "quota_min": float(altezze[altezze > 0].min()) if (altezze > 0).any() else 0.0,
        "quota_max": float(altezze.max()),
        "blocchi_ignoti": sorted(ignoti),
        # altezze[z, x] = y del blocco piu' alto non-aria; serve a verificare
        # numericamente che sia stato scritto quello che si voleva scrivere.
        "altezze": altezze,
        "origine": (cx_min * 16, cz_min * 16),
    }
