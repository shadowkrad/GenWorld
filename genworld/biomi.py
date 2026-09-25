"""Biomi: il clima del mondo.

Fino a qui il mondo aveva i blocchi giusti e nessun clima: ogni chunk usciva
`plains`, cioe' erba dello stesso verde dal deserto alla tundra, acqua di un
colore solo, niente neve che si posa, mob sbagliati. I biomi sono l'unica
cosa che Minecraft NON deduce dai blocchi: il colore dell'erba e delle foglie,
il fogliame, la pioggia, la neve e chi compare di notte vengono tutti da li'.

Due regole che vengono dal resto del progetto:

1. **Un nome moderno non e' un nome universale.** `universal_minecraft:snowy_plains`
   non esiste: l'universale e' `snowy_tundra`, e `from_universal` su un nome
   sconosciuto lo restituisce tale e quale, senza errore. E' la stessa
   trappola di `universal_minecraft:oak_log`, la seconda volta. Qui c'e' una
   funzione che la chiude una volta per tutte: `verifica_traduzioni()`.
2. **Le soglie nette si vedono.** Un confine di bioma che segue una curva di
   livello e' riconoscibile a colpo d'occhio come generato. Il campo di
   freddo e' sporcato con un fBm prima di essere tagliato.
"""

from __future__ import annotations

import numpy as np

from .mappa import (CRATERE, DESERTO, FIUME, FORESTA, MARE, MONTAGNA, NEVE,
                    OCEANO, PIANURA, PRATERIA, SPIAGGIA, VULCANO)
from .rumore import fbm

# Nomi UNIVERSALI (non quelli di Minecraft): sono quelli che amulet vuole, e
# la traduzione alla versione di gioco la fa PyMCTranslate.
BIOMI = (
    "deep_frozen_ocean", "deep_cold_ocean", "deep_ocean", "deep_warm_ocean",
    "frozen_ocean", "cold_ocean", "ocean", "warm_ocean",
    "frozen_river", "river",
    "snowy_beach", "beach",
    "desert", "savanna", "plains", "sunflower_plains", "meadow", "snowy_tundra",
    "forest", "birch_forest", "dark_forest", "taiga", "snowy_taiga",
    "giant_spruce_taiga",
    "mountains", "gravelly_mountains", "snowy_slopes", "frozen_peaks",
    "stony_peaks", "badlands",
)
INDICE = {n: i for i, n in enumerate(BIOMI)}


def verifica_traduzioni(versione=(1, 21, 4)) -> list[str]:
    """Ritorna i nomi di BIOMI che NON traducono alla versione di gioco.

    Da chiamare quando si tocca la tavolozza. Un nome che non traduce si
    scrive nei file senza un lamento e in gioco semplicemente non c'e'.
    """
    import PyMCTranslate
    tr = PyMCTranslate.new_translation_manager()
    v = tr.get_version("java", versione)
    universali = set(tr.universal_format.biome.biome_ids)
    fuori = []
    for n in BIOMI:
        k = f"universal_minecraft:{n}"
        if k not in universali or not v.biome.from_universal(k).startswith("minecraft:"):
            fuori.append(n)
    return fuori


# --------------------------------------------------------------------------
# Clima
# --------------------------------------------------------------------------

def latitudine(cls: np.ndarray) -> np.ndarray | None:
    """Gradiente nord-sud dedotto da DOVE sta la neve, o None.

    Una mappa disegnata non dice dove sia il nord. Ma se la classe NEVE e'
    concentrata da una parte, quella parte e' il freddo: invece di inventare
    un asse, si misura quello che il disegno gia' mostra. Se la neve e'
    sparsa o non c'e', non si inventa nessun gradiente.
    """
    nevoso = cls == NEVE
    if nevoso.sum() < cls.size * 0.002:
        return None
    H = cls.shape[0]
    zc = float(np.nonzero(nevoso)[0].mean()) / max(H - 1, 1)
    if abs(zc - 0.5) < 0.08:
        return None          # neve al centro: e' quota, non latitudine
    z = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    # 1 = polo (dove sta la neve), 0 = parte opposta
    return (1.0 - z) if zc < 0.5 else z


BASE_TEMPERATA = 0.34     # clima al livello del mare, a meta' strada dal polo
PESO_QUOTA = 0.58
PESO_LATITUDINE = 0.62


def freddo(cls: np.ndarray, altezze: np.ndarray, livello_mare: int = 62,
           peso_latitudine: float = PESO_LATITUDINE, seed: int = 0) -> np.ndarray:
    """Campo 0 (caldo) .. 1 (gelido).

    La quota e' il contributo certo - l'aria si raffredda salendo, e la
    heightmap ce l'abbiamo. La latitudine si aggiunge solo se il disegno la
    suggerisce (vedi `latitudine`), e si aggiunge **centrata**: e' uno
    scostamento da un clima temperato, non una scala da 0 a 1.

    Prima versione sbagliata: latitudine presa cosi' com'e', da 0 a 1. Meta'
    mappa finiva sotto la soglia del caldo e l'oceano tropicale diventava il
    bioma piu' esteso del mondo - 154.615 celle su Arda, piu' dell'oceano
    normale. Un emisfero intero non puo' essere "l'estremo caldo".
    """
    h = altezze.astype(np.float32)
    per_quota = np.clip((h - (livello_mare + 38)) / 78.0, 0.0, 1.0)

    f = BASE_TEMPERATA + PESO_QUOTA * per_quota
    lat = latitudine(cls)
    if lat is not None:
        f = f + peso_latitudine * (lat - 0.5)

    # Le soglie nette diventano curve di livello, che si riconoscono subito
    # come generate: un fBm largo le fa serpeggiare.
    lato = max(cls.shape)
    n = fbm(lato, ottave=4, celle_base=max(4, lato // 90), persistenza=0.55,
            seed=seed) - 0.5
    f = f + n[:cls.shape[0], :cls.shape[1]] * 0.20

    # il deserto e' caldo per definizione: e' scritto nel disegno
    f = np.where(cls == DESERTO, np.minimum(f, 0.15), f)
    return np.clip(f, 0.0, 1.0)


def _macchie(forma, seed: int, celle: int = 12) -> np.ndarray:
    """Campo 0..1 a chiazze larghe, per variare il tipo dentro una classe.

    Un continente coperto da un unico `forest` e' esattamente il difetto che
    si ripete in tutto il progetto: l'uniformita' si legge come finta.
    """
    lato = max(forma)
    n = fbm(lato, ottave=3, celle_base=max(3, lato // celle), persistenza=0.5,
            seed=seed)
    return n[:forma[0], :forma[1]]


# --------------------------------------------------------------------------
# Assegnazione
# --------------------------------------------------------------------------

def assegna(cls: np.ndarray, altezze: np.ndarray, livello_mare: int = 62,
            seed: int = 5) -> np.ndarray:
    """Mappa [z, x] di indici in `BIOMI`."""
    f = freddo(cls, altezze, livello_mare, seed=seed)
    h = altezze.astype(np.float32)
    var = _macchie(cls.shape, seed + 1)

    gelido = f > 0.72
    fresco = (f > 0.48) & ~gelido
    caldo = f < 0.15

    out = np.full(cls.shape, INDICE["plains"], np.uint8)

    def posa(maschera, nome):
        out[maschera] = INDICE[nome]

    # -- acqua ----------------------------------------------------------
    profondo = cls == OCEANO
    posa(profondo, "deep_ocean")
    posa(profondo & fresco, "deep_cold_ocean")
    posa(profondo & gelido, "deep_frozen_ocean")
    posa(profondo & caldo, "deep_warm_ocean")

    basso = cls == MARE
    posa(basso, "ocean")
    posa(basso & fresco, "cold_ocean")
    posa(basso & gelido, "frozen_ocean")
    posa(basso & caldo, "warm_ocean")

    posa(cls == FIUME, "river")
    posa((cls == FIUME) & gelido, "frozen_river")

    posa(cls == SPIAGGIA, "beach")
    posa((cls == SPIAGGIA) & gelido, "snowy_beach")

    # -- terreno aperto -------------------------------------------------
    posa(cls == DESERTO, "desert")

    aperto = np.isin(cls, (PIANURA, PRATERIA))
    posa(aperto, "plains")
    posa(aperto & (var > 0.62), "sunflower_plains")
    posa(aperto & caldo, "savanna")
    posa(aperto & fresco, "meadow")
    posa(aperto & gelido, "snowy_tundra")

    # -- bosco ----------------------------------------------------------
    bosco = cls == FORESTA
    posa(bosco, "forest")
    posa(bosco & (var > 0.58), "birch_forest")
    posa(bosco & (var < 0.34), "dark_forest")
    posa(bosco & fresco, "taiga")
    posa(bosco & fresco & (var > 0.62), "giant_spruce_taiga")
    posa(bosco & gelido, "snowy_taiga")

    # -- roccia ---------------------------------------------------------
    monte = cls == MONTAGNA
    posa(monte, "mountains")
    posa(monte & (var < 0.38), "gravelly_mountains")
    posa(monte & gelido, "snowy_slopes")

    # La classe NEVE e' superficie innevata: sopra una certa quota diventa
    # cima gelata, sotto resta tundra. Senza questa distinzione una vetta e
    # una pianura artica avrebbero lo stesso cielo.
    neve = cls == NEVE
    alta = h > livello_mare + 95
    posa(neve, "snowy_tundra")
    posa(neve & ~alta & fresco, "snowy_slopes")
    posa(neve & alta, "frozen_peaks")
    posa(monte & alta & ~gelido, "stony_peaks")

    # -- vulcano --------------------------------------------------------
    # `badlands` e non `basalt_deltas`: il secondo e' un bioma del Nether e
    # nell'overworld si porterebbe dietro la sua nebbia e i suoi mob (ghast
    # sopra il lago di lava). Badlands da' quello che serve davvero - niente
    # pioggia, niente neve sulla cima, terreno arido - senza sorprese.
    posa(np.isin(cls, (VULCANO, CRATERE)), "badlands")
    return out


def statistiche(biomi: np.ndarray) -> dict[str, int]:
    v, n = np.unique(biomi, return_counts=True)
    ordine = np.argsort(-n)
    return {BIOMI[int(v[i])]: int(n[i]) for i in ordine}


# Colori per l'anteprima: non sono quelli di Minecraft, sono scelti per
# leggersi a colpo d'occhio - blu per l'acqua (piu' scuro = piu' profondo,
# piu' freddo = piu' violetto), verdi per il bosco, gialli per l'aperto,
# bianchi per il gelo.
COLORI = {
    "deep_frozen_ocean": (36, 48, 92), "deep_cold_ocean": (24, 40, 96),
    "deep_ocean": (20, 44, 116), "deep_warm_ocean": (18, 64, 130),
    "frozen_ocean": (120, 152, 196), "cold_ocean": (58, 100, 164),
    "ocean": (48, 96, 176), "warm_ocean": (40, 132, 190),
    "frozen_river": (150, 188, 220), "river": (80, 140, 210),
    "snowy_beach": (232, 232, 222), "beach": (226, 212, 160),
    "desert": (232, 210, 136), "savanna": (196, 190, 104),
    "plains": (150, 192, 104), "sunflower_plains": (176, 204, 96),
    "meadow": (128, 180, 128), "snowy_tundra": (238, 242, 246),
    "forest": (44, 118, 56), "birch_forest": (108, 156, 88),
    "dark_forest": (28, 78, 44), "taiga": (52, 104, 92),
    "snowy_taiga": (154, 186, 182), "giant_spruce_taiga": (38, 84, 76),
    "mountains": (132, 126, 118), "gravelly_mountains": (108, 104, 100),
    "snowy_slopes": (206, 218, 228), "frozen_peaks": (182, 214, 236),
    "stony_peaks": (158, 150, 142), "badlands": (58, 48, 46),
}


def anteprima(biomi: np.ndarray):
    """PNG di controllo: un colore per bioma."""
    from PIL import Image
    tinte = np.array([COLORI[n] for n in BIOMI], np.uint8)
    return Image.fromarray(tinte[biomi], "RGB")
