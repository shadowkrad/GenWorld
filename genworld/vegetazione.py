"""Alberi e copertura vegetale.

Senza questo, una foresta e' un prato: c'e' il blocco d'erba e basta. E' la
differenza piu' visibile appena si entra in gioco.

Due parti indipendenti:

  * SEMINA - dove vanno gli alberi. Vettoriale, calcolata una volta sulla mappa
    intera: griglia sfalsata con probabilita' per classe, piu' i filtri
    (pendenza, vicinanza all'acqua, quota).
  * DISEGNO - com'e' fatto un albero, e come si stampa dentro un chunk.

La separazione conta perche' un albero a cavallo del bordo di un chunk deve
comparire in ENTRAMBI: si tiene l'elenco in coordinate globali e ogni chunk
disegna tutti gli alberi il cui ingombro lo tocca, anche quelli piantati nel
chunk accanto. Filtrando per centro invece che per ingombro si ottengono
alberi tagliati a meta' lungo tutte le linee di chunk.
"""

from __future__ import annotations

import numpy as np

from .biomi import INDICE as INDICE_BIOMI
from .mappa import (ACQUA, DESERTO, FORESTA, MONTAGNA, NEVE, PIANURA,
                    PRATERIA, SPIAGGIA)
from .rumore import fbm

# specie
(QUERCIA, BETULLA, ABETE, CACTUS, ARBUSTO, CESPUGLIO, MELO, CILIEGIO, BACCHE,
 DENTE_DI_LEONE, PAPAVERO, TULIPANO, FIORDALISO, ALLIUM, GIGLIO_VALLE,
 FUNGO_MARRONE, FUNGO_ROSSO, CANNA, ACACIA, ROVERE_SCURO, GIUNGLA,
 ABETE_ALTO, GIRASOLE, FELCE_GRANDE, LILLA, ROSAIO, PEONIA, FUNGO_GIGANTE_M,
 FUNGO_GIGANTE_R, TORCHFLOWER, MANGROVIA, AZALEA, BAMBU, SASSO,
 TRONCO_CADUTO, SECCA, SECCA_ALTA, LETTIERA, BROCCA) = range(39)

# piante alte due blocchi (meta' bassa e meta' alta)
DOPPI = {GIRASOLE: "sunflower", FELCE_GRANDE: "large_fern", LILLA: "lilac",
         ROSAIO: "rose_bush", PEONIA: "peony"}
FUNGHI_GIGANTI = (FUNGO_GIGANTE_M, FUNGO_GIGANTE_R)

# i fiori a un blocco solo: stesso trattamento di arbusto/cespuglio nel
# disegno, distinti solo per varieta' e per come si mescolano per classe
FIORI = (DENTE_DI_LEONE, PAPAVERO, TULIPANO, FIORDALISO, ALLIUM, GIGLIO_VALLE)
FUNGHI = (FUNGO_MARRONE, FUNGO_ROSSO)

NOMI_SPECIE = {
    QUERCIA: "quercia", BETULLA: "betulla", ABETE: "abete",
    CACTUS: "cactus", ARBUSTO: "arbusto", CESPUGLIO: "cespuglio",
    # Il melo non esiste in Minecraft: e' la quercia, che le mele le fa
    # cadere davvero. Il ciliegio invece c'e' dalla 1.20 ed e' l'unico
    # albero che sembra un frutteto anche quando non lo e'.
    MELO: "melo", CILIEGIO: "ciliegio", BACCHE: "bacche",
    DENTE_DI_LEONE: "dente_di_leone", PAPAVERO: "papavero",
    TULIPANO: "tulipano", FIORDALISO: "fiordaliso", ALLIUM: "allium",
    GIGLIO_VALLE: "giglio_valle", FUNGO_MARRONE: "fungo_marrone",
    FUNGO_ROSSO: "fungo_rosso", CANNA: "canna",
    ACACIA: "acacia", ROVERE_SCURO: "rovere_scuro", GIUNGLA: "giungla",
    ABETE_ALTO: "abete_alto", GIRASOLE: "girasole", FELCE_GRANDE: "felce_grande",
    LILLA: "lilla", ROSAIO: "rosaio", PEONIA: "peonia",
    FUNGO_GIGANTE_M: "fungo_gigante_marrone", FUNGO_GIGANTE_R: "fungo_gigante_rosso",
    TORCHFLOWER: "torchflower", MANGROVIA: "mangrovia", AZALEA: "azalea",
    BAMBU: "bambu", SASSO: "sasso", TRONCO_CADUTO: "tronco_caduto",
    SECCA: "erba_secca", SECCA_ALTA: "erba_secca_alta", LETTIERA: "lettiera",
    BROCCA: "pianta_brocca",
}

# Alberi per blocco quadrato. Una foresta fitta di Minecraft sta intorno a
# 0,08-0,12; una prateria ha qualche albero isolato.
DENSITA = {
    FORESTA: 0.11,
    PRATERIA: 0.020,
    PIANURA: 0.007,
    MONTAGNA: 0.024,
    NEVE: 0.003,
    DESERTO: 0.005,
    SPIAGGIA: 0.0008,
}

# Specie per classe, con i rispettivi pesi. I fiori entrano nel PRATO e nella
# RADURA di bosco con colori diversi apposta - un prato viola di allium
# dappertutto si legge finto quanto un bosco tutto quercia - e restano fuori
# da montagna, neve e deserto, dove in Minecraft non nascono naturalmente.
MISCELA = {
    FORESTA:  [(QUERCIA, 0.40), (AZALEA, 0.03), (BETULLA, 0.19), (ABETE, 0.15),
               (GIGLIO_VALLE, 0.06), (FUNGO_MARRONE, 0.03), (FUNGO_ROSSO, 0.01),
               (LETTIERA, 0.13)],
    PRATERIA: [(QUERCIA, 0.55), (BETULLA, 0.15), (CESPUGLIO, 0.08),
               (DENTE_DI_LEONE, 0.08), (PAPAVERO, 0.06), (TULIPANO, 0.04),
               (FIORDALISO, 0.02), (ALLIUM, 0.02)],
    PIANURA:  [(QUERCIA, 0.55), (CESPUGLIO, 0.25), (DENTE_DI_LEONE, 0.10),
               (PAPAVERO, 0.10)],
    MONTAGNA: [(ABETE, 0.85), (CESPUGLIO, 0.15)],
    NEVE:     [(ABETE, 1.0)],
    DESERTO:  [(CACTUS, 0.40), (ARBUSTO, 0.30), (SECCA, 0.20), (SECCA_ALTA, 0.10)],
    SPIAGGIA: [(ARBUSTO, 1.0)],
}

MATERIALE = {QUERCIA: "oak", BETULLA: "birch", ABETE: "spruce",
             MELO: "oak", CILIEGIO: "cherry", ACACIA: "acacia",
             ROVERE_SCURO: "dark_oak", GIUNGLA: "jungle", ABETE_ALTO: "spruce",
             MANGROVIA: "mangrove"}
ALTEZZA = {QUERCIA: (4, 7), BETULLA: (5, 8), ABETE: (6, 12),
           CACTUS: (2, 5), ARBUSTO: (1, 2), CESPUGLIO: (1, 2),
           MELO: (4, 6), CILIEGIO: (5, 7), BACCHE: (1, 2),
           DENTE_DI_LEONE: (1, 2), PAPAVERO: (1, 2), TULIPANO: (1, 2),
           FIORDALISO: (1, 2), ALLIUM: (1, 2), GIGLIO_VALLE: (1, 2),
           FUNGO_MARRONE: (1, 2), FUNGO_ROSSO: (1, 2), CANNA: (1, 4),
           ACACIA: (5, 8), ROVERE_SCURO: (6, 9), GIUNGLA: (11, 18),
           ABETE_ALTO: (14, 23), GIRASOLE: (2, 3), FELCE_GRANDE: (2, 3),
           LILLA: (2, 3), ROSAIO: (2, 3), PEONIA: (2, 3),
           FUNGO_GIGANTE_M: (4, 7), FUNGO_GIGANTE_R: (4, 7), TORCHFLOWER: (1, 2),
           MANGROVIA: (5, 8), AZALEA: (1, 2), BAMBU: (5, 12),
           SASSO: (1, 4), TRONCO_CADUTO: (3, 6),
           SECCA: (1, 2), SECCA_ALTA: (1, 2), LETTIERA: (1, 2), BROCCA: (2, 3)}

# Rocce e tronchi caduti: una quota delle piante dei terreni boscosi e rocciosi
# diventa un sasso (anche in montagna e sulla neve), e nei boschi un tronco a
# terra. Non sono alberi: stanno sul terreno, a coppie e a gruppi sparsi come il
# resto.
QUOTA_SASSI = 0.045
QUOTA_TRONCHI = 0.035

# Le specie per BIOMA, dove il bioma ha un suo bosco: sostituiscono la miscela
# della classe. Il bioma lo decide `biomi.assegna` (caldo, freddo, macchie):
# la savana ha l'acacia piatta, il bosco scuro il rovere scuro a tronco doppio,
# la giungla alberi altissimi, il prato fresco i ciliegi, la taiga gigante gli
# abeti piu' alti.
MISCELA_BIOMA = {
    "savanna": [(ACACIA, 0.46), (CESPUGLIO, 0.14), (ARBUSTO, 0.08), (SECCA, 0.12),
                (SECCA_ALTA, 0.08), (QUERCIA, 0.06), (TORCHFLOWER, 0.06)],
    "dark_forest": [(ROVERE_SCURO, 0.45), (LETTIERA, 0.10), (QUERCIA, 0.14), (FUNGO_GIGANTE_M, 0.07),
                    (FUNGO_GIGANTE_R, 0.06), (FUNGO_MARRONE, 0.10),
                    (FUNGO_ROSSO, 0.08)],
    "sunflower_plains": [(GIRASOLE, 0.38), (QUERCIA, 0.10), (CESPUGLIO, 0.20),
                         (DENTE_DI_LEONE, 0.17), (TULIPANO, 0.10), (PAPAVERO, 0.05)],
    "taiga": [(ABETE, 0.62), (FELCE_GRANDE, 0.14), (CESPUGLIO, 0.12),
              (FUNGO_MARRONE, 0.06), (BACCHE, 0.06)],
    "snowy_taiga": [(ABETE, 0.80), (CESPUGLIO, 0.12), (BACCHE, 0.08)],
    "mangrove_swamp": [(MANGROVIA, 0.55), (CESPUGLIO, 0.20), (FUNGO_MARRONE, 0.05),
                       (AZALEA, 0.10), (CANNA, 0.10)],
    "jungle": [(GIUNGLA, 0.50), (BAMBU, 0.20), (CESPUGLIO, 0.20), (QUERCIA, 0.06),
               (BROCCA, 0.04)],
    "meadow": [(CILIEGIO, 0.20), (QUERCIA, 0.24), (DENTE_DI_LEONE, 0.12),
               (TULIPANO, 0.08), (LILLA, 0.08), (PEONIA, 0.08), (ROSAIO, 0.06),
               (CESPUGLIO, 0.14)],
    "giant_spruce_taiga": [(ABETE_ALTO, 0.55), (ABETE, 0.35), (CESPUGLIO, 0.10)],
}

# moltiplicatore della densita' di piante per bioma, dove la classe e' troppo rada
DENSITA_BIOMA = {"mangrove_swamp": 5.0, "jungle": 1.5, "savanna": 1.5}

# Dove puo' crescere la canna da zucchero: SOLO sul bordo dell'acqua, come in
# Minecraft davvero - non e' una questione di classe di terreno ma di essere
# a un blocco esatto da un fiume, un lago o il mare.
CANNA_CLASSI = (SPIAGGIA, PIANURA, PRATERIA, FORESTA)

# Ingombro orizzontale massimo, in blocchi dal centro. Serve a sapere quali
# alberi di chunk vicini possono sporgere dentro questo.
RAGGIO_MAX = 4


# --------------------------------------------------------------------------
# Semina
# --------------------------------------------------------------------------

def semina(
    cls: np.ndarray,
    altezze: np.ndarray,
    livello_mare: int = 62,
    passo: int = 3,
    pendenza_massima: float = 2.5,
    scala_densita: float = 1.0,
    grumi: float = 0.85,
    seed: int = 0,
    biomi: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Dove piantare. Con `biomi` (indici di `biomi.BIOMI`) le specie dei biomi in
    `MISCELA_BIOMA` prendono il posto della miscela della classe. Ritorna array paralleli in coordinate della mappa.

    La griglia sfalsata e' il modo economico di ottenere rumore blu: un
    candidato per cella, spostato a caso dentro la cella. Cosi' due alberi non
    possono mai stare a meno di un passo l'uno dall'altro, e non si formano
    ne' grumi ne' file regolari. Un campionamento puramente casuale darebbe
    grappoli e vuoti.
    """
    rng = np.random.default_rng(seed)
    H, W = cls.shape

    gx, gz = np.meshgrid(np.arange(0, W - 1, passo),
                         np.arange(0, H - 1, passo), indexing="xy")
    gx = gx.ravel(); gz = gz.ravel()
    n = gx.size
    x = np.clip(gx + rng.integers(0, passo, n), 0, W - 1)
    z = np.clip(gz + rng.integers(0, passo, n), 0, H - 1)

    c = cls[z, x]
    h = altezze[z, x]

    # probabilita' per classe
    prob = np.zeros(n, np.float32)
    for classe, d in DENSITA.items():
        prob[c == classe] = min(1.0, d * passo * passo * scala_densita)

    # Modulazione a bassa frequenza. Una densita' costante per classe da'
    # alberi a pois, sparsi uniformemente come su una tappezzeria. Un bosco
    # vero si addensa in macchie e lascia radure: basta moltiplicare la
    # probabilita' per un campo frattale a grandi celle. Stesso principio del
    # rumore di dettaglio mascherato dalla semantica: il caso uniforme e'
    # sempre l'ingrediente che fa sembrare tutto finto.
    if grumi > 0:
        lato_n = max(cls.shape)
        campo = fbm(lato_n, ottave=3, celle_base=6, persistenza=0.55, seed=seed + 77)
        campo = campo[:cls.shape[0], :cls.shape[1]]
        lo, hi = float(campo.min()), float(campo.max())
        if hi > lo:
            campo = (campo - lo) / (hi - lo)
        fattore = (1.0 - grumi) + 2.0 * grumi * campo[z, x]
        prob = np.clip(prob * fattore, 0.0, 1.0)

    # i biomi fitti hanno piu' alberi della loro classe: una palude di mangrovie
    # o una giungla con la densita' di una pianura sembrano un prato
    if biomi is not None:
        for nome, k in DENSITA_BIOMA.items():
            prob = np.where(biomi[z, x] == INDICE_BIOMI[nome], np.clip(prob * k, 0, 1), prob)

    tieni = rng.random(n) < prob

    # niente alberi nell'acqua, sulle pendenze impossibili, o a filo d'acqua
    pend = np.hypot(*np.gradient(altezze.astype(np.float32)))
    tieni &= h > livello_mare
    tieni &= pend[z, x] < pendenza_massima

    x, z, h, c = x[tieni], z[tieni], h[tieni], c[tieni]
    if x.size == 0:
        vuoto = np.zeros(0, np.int32)
        return {"x": vuoto, "z": vuoto, "y": vuoto,
                "specie": vuoto, "altezza": vuoto, "seme": vuoto}

    # scelta della specie
    r = rng.random(x.size)
    specie = np.full(x.size, QUERCIA, np.int32)
    for classe, miscela in MISCELA.items():
        m = c == classe
        if not m.any():
            continue
        soglia = 0.0
        for sp, peso in miscela:
            dentro = m & (r >= soglia) & (r < soglia + peso)
            specie[dentro] = sp
            soglia += peso
        specie[m & (r >= soglia)] = miscela[-1][0]

    if biomi is not None:
        b = biomi[z, x]
        r2 = rng.random(x.size)
        for nome, miscela in MISCELA_BIOMA.items():
            m = b == INDICE_BIOMI[nome]
            if not m.any():
                continue
            soglia = 0.0
            for sp, peso in miscela:
                dentro = m & (r2 >= soglia) & (r2 < soglia + peso)
                specie[dentro] = sp
                soglia += peso
            specie[m & (r2 >= soglia)] = miscela[-1][0]

    # sassi e tronchi caduti, al posto di una parte delle piante
    dado = rng.random(x.size)
    sasso = np.isin(c, (FORESTA, MONTAGNA, PRATERIA, NEVE)) & (dado < QUOTA_SASSI)
    tronco = (c == FORESTA) & (dado >= QUOTA_SASSI) & (dado < QUOTA_SASSI + QUOTA_TRONCHI)
    specie[sasso] = SASSO
    specie[tronco] = TRONCO_CADUTO

    altezza = np.zeros(x.size, np.int32)
    for sp, (lo, hi) in ALTEZZA.items():
        m = specie == sp
        if m.any():
            altezza[m] = rng.integers(lo, hi, int(m.sum()))

    return {
        "x": x.astype(np.int32), "z": z.astype(np.int32),
        "y": h.astype(np.int32), "specie": specie,
        "altezza": altezza,
        "seme": rng.integers(0, 2 ** 31 - 1, x.size).astype(np.int64),
    }


def semina_canna(
    cls: np.ndarray,
    altezze: np.ndarray,
    livello_mare: int = 62,
    passo: int = 2,
    probabilita: float = 0.35,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Canna da zucchero: solo sul bordo dell'acqua, un blocco esatto di
    distanza, come cresce davvero in Minecraft. Non e' una questione di
    densita' per classe come gli alberi - e' una questione di ADIACENZA, e
    per questo e' una semina a parte invece di un'altra riga in `MISCELA`.
    """
    rng = np.random.default_rng(seed + 991)
    H, W = cls.shape
    acqua = np.isin(cls, ACQUA)
    # bordo = terra affacciata sull'acqua, spostando la maschera d'acqua di
    # un blocco nelle quattro direzioni e guardando dove tocca terra
    bordo = np.zeros_like(acqua)
    bordo[1:, :] |= acqua[:-1, :]
    bordo[:-1, :] |= acqua[1:, :]
    bordo[:, 1:] |= acqua[:, :-1]
    bordo[:, :-1] |= acqua[:, 1:]
    bordo &= ~acqua & np.isin(cls, CANNA_CLASSI)

    gx, gz = np.meshgrid(np.arange(0, W - 1, passo),
                         np.arange(0, H - 1, passo), indexing="xy")
    gx = gx.ravel(); gz = gz.ravel()
    n = gx.size
    x = np.clip(gx + rng.integers(0, passo, n), 0, W - 1)
    z = np.clip(gz + rng.integers(0, passo, n), 0, H - 1)

    tieni = bordo[z, x] & (altezze[z, x] > livello_mare)
    tieni &= rng.random(n) < probabilita
    x, z = x[tieni], z[tieni]
    if x.size == 0:
        vuoto = np.zeros(0, np.int32)
        return {"x": vuoto, "z": vuoto, "y": vuoto, "specie": vuoto,
                "altezza": vuoto, "seme": vuoto}

    return {
        "x": x.astype(np.int32), "z": z.astype(np.int32),
        "y": altezze[z, x].astype(np.int32),
        "specie": np.full(x.size, CANNA, np.int32),
        "altezza": rng.integers(1, 4, x.size).astype(np.int32),
        "seme": rng.integers(0, 2 ** 31 - 1, x.size).astype(np.int64),
    }


def unisci(*gruppi: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Concatena piu' risultati di semina (alberi + canna) in un elenco
    solo, cosi' l'indice per chunk e il disegno lavorano su una lista sola
    invece di doverne conoscere il numero."""
    chiavi = ("x", "z", "y", "specie", "altezza", "seme")
    if not gruppi:
        vuoto = np.zeros(0, np.int32)
        return {k: vuoto for k in chiavi}
    return {k: np.concatenate([g[k] for g in gruppi]) for k in chiavi}


def indice_per_chunk(alberi: dict, lato: int, passo_chunk: int = 16) -> dict:
    """Raggruppa gli alberi per chunk, una volta sola.

    Senza indice, ogni chunk dovrebbe scorrere l'elenco intero: su 2704 chunk
    e decine di migliaia di alberi sono centinaia di milioni di confronti.
    """
    n = alberi["x"].size
    fuori: dict[tuple[int, int], list[int]] = {}
    if n == 0:
        return fuori
    cx = alberi["x"] // passo_chunk
    cz = alberi["z"] // passo_chunk
    for i in range(n):
        fuori.setdefault((int(cx[i]), int(cz[i])), []).append(i)
    return fuori


def alberi_vicini(indice: dict, cx: int, cz: int, margine: int = 1) -> list[int]:
    """Indici degli alberi che possono sporgere dentro il chunk (cx, cz)."""
    fuori: list[int] = []
    for dz in range(-margine, margine + 1):
        for dx in range(-margine, margine + 1):
            fuori.extend(indice.get((cx + dx, cz + dz), ()))
    return fuori


# --------------------------------------------------------------------------
# Disegno
# --------------------------------------------------------------------------

class Tavolozza:
    """Id dei blocchi vegetali, risolti una volta sola sul livello aperto."""

    def __init__(self, scrittore):
        s = scrittore
        self.tronco = {
            sp: s.blocco("log", axis="y", material=MATERIALE[sp], stripped="false")
            for sp in ALBERI
        }
        self.foglie = {
            sp: s.blocco("leaves", check_decay="false", distance="7",
                         material=MATERIALE[sp], persistent="true")
            for sp in ALBERI
        }
        self.bacche = s.blocco("sweet_berry_bush", age="3")
        self.cactus = s.blocco("cactus", age="0")
        self.arbusto = s.blocco("plant", plant_type="dead_bush")
        self.erba_alta = s.blocco("plant", plant_type="grass")
        self.felce = s.blocco("plant", plant_type="fern")
        self.canna = s.blocco("sugar_cane", age="0")
        # erba secca (savana, deserto, badlands) e lettiera di foglie (boschi)
        self.brocca = (s.blocco("pitcher_plant", half="lower"),
                       s.blocco("pitcher_plant", half="upper"))
        self.secca = s.blocco("plant", plant_type="short_dry_grass")
        self.secca_alta = s.blocco("plant", plant_type="tall_dry_grass")
        self.lettiera = {
            (f, n): s.blocco("leaf_litter", facing=f, segment_amount=str(n))
            for f in ("north", "south", "east", "west") for n in (1, 2, 3, 4)
        }
        self.fiore = {
            DENTE_DI_LEONE: s.blocco("plant", plant_type="dandelion"),
            PAPAVERO: s.blocco("plant", plant_type="poppy"),
            TULIPANO: s.blocco("plant", plant_type="red_tulip"),
            FIORDALISO: s.blocco("plant", plant_type="cornflower"),
            ALLIUM: s.blocco("plant", plant_type="allium"),
            GIGLIO_VALLE: s.blocco("plant", plant_type="lily_of_the_valley"),
        }
        self.fungo = {
            FUNGO_MARRONE: s.blocco("brown_mushroom"),
            FUNGO_ROSSO: s.blocco("red_mushroom"),
        }
        self.torchflower = s.blocco("torchflower")
        self.liana = {
            "north": s.blocco("vine", north="true", south="false", east="false",
                              west="false", up="false"),
            "south": s.blocco("vine", north="false", south="true", east="false",
                              west="false", up="false"),
            "east": s.blocco("vine", north="false", south="false", east="true",
                             west="false", up="false"),
            "west": s.blocco("vine", north="false", south="false", east="false",
                             west="true", up="false"),
        }
        self.bambu = {
            livello: s.blocco("bamboo", stage="0", leaves=livello, thickness="thick")
            for livello in ("none", "small", "large")
        }
        self.sasso = (s.blocco("mossy_cobblestone"), s.blocco("cobblestone"),
                      s.blocco("stone"), s.blocco("andesite", polished="false"))
        self.tronco_a_terra = {
            (mat, asse): s.blocco("log", axis=asse, material=mat, stripped="false")
            for mat in ("oak", "birch", "spruce") for asse in ("x", "z")
        }
        self.tappeto_muschio = s.blocco("moss_carpet")
        self.azalea = s.blocco("azalea")
        self.azalea_fiorita = s.blocco("flowering_azalea")
        self.radici = s.blocco("mangrove_roots")
        self.doppio = {
            sp: (s.blocco("double_plant", plant_type=n, half="lower"),
                 s.blocco("double_plant", plant_type=n, half="upper"))
            for sp, n in DOPPI.items()
        }
        lati = dict(north="true", south="true", east="true", west="true")
        self.gambo = s.blocco("mushroom_stem", up="false", down="false",
                              material="red", **lati)
        self.cappello = {
            FUNGO_GIGANTE_M: s.blocco("brown_mushroom_block", up="true", down="false",
                                      **lati),
            FUNGO_GIGANTE_R: s.blocco("red_mushroom_block", up="true", down="false",
                                      **lati),
        }
        self.aria = s.id_aria


# le specie che hanno un tronco e delle foglie (le altre sono piante a un blocco)
ALBERI = (QUERCIA, BETULLA, ABETE, MELO, CILIEGIO, ACACIA, ROVERE_SCURO, GIUNGLA,
          ABETE_ALTO, MANGROVIA)

# profilo dell'abete dall'alto verso il basso: raggio per livello
PROFILO_ABETE = (0, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2)


def _disco(posa, cx: float, cz: float, y: int, raggio2: float, foglie: int,
           rng=None, mangia: float = 0.0) -> None:
    """Un disco di foglie centrato in (cx, cz): le celle a distanza al quadrato
    <= raggio2. `mangia` e' la probabilita' di togliere una cella del bordo,
    perche' una chioma tonda perfetta si legge finta."""
    r = int(raggio2 ** 0.5) + 1
    for dx in range(-r, r + 2):
        for dz in range(-r, r + 2):
            gx, gz = int(np.floor(cx)) + dx, int(np.floor(cz)) + dz
            d2 = (gx - cx) ** 2 + (gz - cz) ** 2
            if d2 > raggio2:
                continue
            if rng is not None and d2 > raggio2 * 0.55 and rng.random() < mangia:
                continue
            posa(gx, y, gz, foglie)


def _disegna_acacia(posa, x, z, base, alt, rng, tronco, foglie) -> None:
    """L'acacia della savana: un tronco che si piega di lato e una chioma
    PIATTA, larga e bassa, come un ombrello."""
    dritto = max(3, alt - 3)
    dx, dz = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(0, 4))]
    for k in range(dritto):
        posa(x, base + k, z, tronco, solo_aria=False)
    # il tronco scavalca di un blocco e sale ancora di due
    px, pz = x + dx, z + dz
    for k in range(dritto, dritto + 3):
        posa(px, base + k, pz, tronco, solo_aria=False)
    cima = base + dritto + 2
    _disco(posa, px, pz, cima, 7.0, foglie, rng, 0.25)
    _disco(posa, px, pz, cima + 1, 2.2, foglie)
    # un ramo piu' basso dall'altra parte, con una chiomina
    ox_, oz_ = x - dx, z - dz
    posa(ox_, base + dritto - 1, oz_, tronco, solo_aria=False)
    _disco(posa, ox_, oz_, base + dritto, 3.2, foglie, rng, 0.2)


def _disegna_rovere_scuro(posa, x, z, base, alt, tronco, foglie) -> None:
    """Il rovere scuro: tronco a 2x2 e una chioma larghissima e fitta."""
    cx, cz = x + 0.5, z + 0.5
    cima = base + alt
    for y, r2 in ((cima - 3, 6.5), (cima - 2, 10.5), (cima - 1, 10.5),
                  (cima, 6.5), (cima + 1, 2.5)):
        _disco(posa, cx, cz, y, r2, foglie)
    for k in range(alt):
        for ddx, ddz in ((0, 0), (1, 0), (0, 1), (1, 1)):
            posa(x + ddx, base + k, z + ddz, tronco, solo_aria=False)


def _disegna_mangrovia(posa, x, z, base, alt, rng, tronco, foglie, radici) -> None:
    """La mangrovia: radici a trampolo attorno al piede, un tronco sottile e una
    chioma larga e irregolare."""
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if rng.random() < 0.8:
            posa(x + dx, base, z + dz, radici, solo_aria=False)
            if rng.random() < 0.5:
                posa(x + dx, base + 1, z + dz, radici, solo_aria=False)
    cima = base + alt
    for y, r2 in ((cima - 2, 9.0), (cima - 1, 9.0), (cima, 5.0), (cima + 1, 1.5)):
        _disco(posa, x, z, y, r2, foglie, rng, 0.2)
    for k in range(alt):
        posa(x, base + k, z, tronco, solo_aria=False)


def _disegna_giungla(posa, x, z, base, alt, tronco, foglie) -> None:
    """La giungla: un tronco altissimo, una chioma in cima e una corona piu'
    bassa a meta' altezza, cosi' si legge a piani come nei boschi tropicali."""
    cima = base + alt
    for y, r2 in ((cima - 2, 7.0), (cima - 1, 9.0), (cima, 5.0), (cima + 1, 1.5)):
        _disco(posa, x, z, y, r2, foglie)
    meta = base + alt // 2
    _disco(posa, x, z, meta, 4.5, foglie)
    _disco(posa, x, z, meta + 1, 2.0, foglie)
    for k in range(alt):
        posa(x, base + k, z, tronco, solo_aria=False)


def disegna(
    out: np.ndarray,
    y0: int,
    ox: int,
    oz: int,
    alberi: dict,
    quali,
    tav: Tavolozza,
) -> int:
    """Stampa gli alberi dentro l'array di chunk (16, H, 16). Ritorna quanti."""
    H = out.shape[1]
    messi = 0

    def posa(gx: int, gy: int, gz: int, blocco: int, solo_aria: bool = True) -> None:
        lx = gx - ox
        lz = gz - oz
        ly = gy - y0
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 <= ly < H:
            if not solo_aria or out[lx, ly, lz] == tav.aria:
                out[lx, ly, lz] = blocco

    for i in quali:
        x = int(alberi["x"][i]); z = int(alberi["z"][i])
        base = int(alberi["y"][i]); sp = int(alberi["specie"][i])
        alt = int(alberi["altezza"][i])
        rng = np.random.default_rng(int(alberi["seme"][i]))

        if sp == CACTUS:
            for k in range(alt):
                posa(x, base + k, z, tav.cactus)
            messi += 1
            continue
        if sp == ARBUSTO:
            posa(x, base, z, tav.arbusto)
            messi += 1
            continue
        if sp == CESPUGLIO:
            posa(x, base, z, tav.erba_alta if rng.random() < 0.7 else tav.felce)
            messi += 1
            continue
        if sp == BROCCA:                       # la pianta carnivora, alta due blocchi
            posa(x, base, z, tav.brocca[0])
            posa(x, base + 1, z, tav.brocca[1])
            messi += 1
            continue
        if sp in (SECCA, SECCA_ALTA):
            posa(x, base, z, tav.secca if sp == SECCA else tav.secca_alta)
            messi += 1
            continue
        if sp == LETTIERA:
            # un mucchietto di foglie secche: la cella e un paio di vicine
            for dx, dz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                if (dx, dz) != (0, 0) and rng.random() < 0.6:
                    continue
                posa(x + dx, base, z + dz,
                     tav.lettiera[(("north", "south", "east", "west")[int(rng.integers(0, 4))],
                                   int(rng.integers(1, 5)))])
            messi += 1
            continue
        if sp == BACCHE:
            posa(x, base, z, tav.bacche)
            messi += 1
            continue
        if sp in FIORI:
            posa(x, base, z, tav.fiore[sp])
            messi += 1
            continue
        if sp == SASSO:
            # un grumo di sassi: il centro, qualche vicino e, se grande, una cima
            blocchi = tav.sasso
            posa(x, base, z, blocchi[int(rng.integers(0, 2))])
            posa(x, base - 1, z, blocchi[1])
            vicini = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
            for k in range(min(alt - 1, len(vicini))):
                dx, dz = vicini[int(rng.integers(0, len(vicini)))]
                posa(x + dx, base, z + dz, blocchi[int(rng.integers(0, len(blocchi)))])
                posa(x + dx, base - 1, z + dz, blocchi[1])
            if alt >= 3:
                posa(x, base + 1, z, blocchi[int(rng.integers(0, 3))])
            messi += 1
            continue
        if sp == TRONCO_CADUTO:
            asse = "x" if rng.random() < 0.5 else "z"
            mat = ("oak", "birch", "spruce")[int(rng.integers(0, 3))]
            blocco = tav.tronco_a_terra[(mat, asse)]
            for k in range(alt):
                px, pz = (x + k, z) if asse == "x" else (x, z + k)
                posa(px, base, pz, blocco)
                posa(px, base - 1, pz, tav.sasso[1])
                r_ = rng.random()
                if r_ < 0.35:
                    posa(px, base + 1, pz, tav.tappeto_muschio)
                elif r_ < 0.45:
                    posa(px, base + 1, pz, tav.fungo[FUNGO_MARRONE])
            messi += 1
            continue
        if sp == BAMBU:
            # un gruppetto di canne vicine, di altezze diverse
            for dx, dz in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)):
                if (dx, dz) != (0, 0) and rng.random() < 0.55:
                    continue
                altezza = max(3, alt - int(rng.integers(0, 4)))
                for k in range(altezza):
                    liv = ("large" if k >= altezza - 2 else
                           "small" if k == altezza - 3 else "none")
                    posa(x + dx, base + k, z + dz, tav.bambu[liv])
            messi += 1
            continue
        if sp == AZALEA:
            posa(x, base, z, tav.azalea_fiorita if rng.random() < 0.4 else tav.azalea)
            messi += 1
            continue
        if sp == MANGROVIA:
            _disegna_mangrovia(posa, x, z, base, alt, rng, tav.tronco[sp], tav.foglie[sp],
                              tav.radici)
            messi += 1
            continue
        if sp == TORCHFLOWER:
            posa(x, base, z, tav.torchflower)
            messi += 1
            continue
        if sp in DOPPI:
            basso, alto = tav.doppio[sp]
            posa(x, base, z, basso)
            posa(x, base + 1, z, alto)
            messi += 1
            continue
        if sp in FUNGHI_GIGANTI:
            for k in range(alt - 1):
                posa(x, base + k, z, tav.gambo, solo_aria=False)
            _disco(posa, x, z, base + alt - 2, 5.0, tav.cappello[sp])
            _disco(posa, x, z, base + alt - 1, 2.5, tav.cappello[sp])
            messi += 1
            continue
        if sp in FUNGHI:
            posa(x, base, z, tav.fungo[sp])
            messi += 1
            continue
        if sp == CANNA:
            for k in range(alt):
                posa(x, base + k, z, tav.canna)
            messi += 1
            continue

        tronco = tav.tronco[sp]
        foglie = tav.foglie[sp]

        if sp == ACACIA:
            _disegna_acacia(posa, x, z, base, alt, rng, tronco, foglie)
            messi += 1
            continue
        if sp == ROVERE_SCURO:
            _disegna_rovere_scuro(posa, x, z, base, alt, tronco, foglie)
            messi += 1
            continue
        if sp == GIUNGLA:
            _disegna_giungla(posa, x, z, base, alt, tronco, foglie)
            # liane lungo il tronco: attaccate al tronco, sul lato opposto
            for k in range(2, alt - 2):
                if rng.random() < 0.28:
                    lati = (((1, 0), "west"), ((-1, 0), "east"),
                            ((0, 1), "north"), ((0, -1), "south"))
                    (dx, dz), faccia = lati[int(rng.integers(0, 4))]
                    posa(x + dx, base + k, z + dz, tav.liana[faccia])
            messi += 1
            continue

        if sp in (ABETE, ABETE_ALTO):
            profilo = PROFILO_ABETE if sp == ABETE else PROFILO_ABETE * 3
            livelli = min(len(profilo), max(3, alt - 1))
            for k in range(livelli):
                y = base + alt - 1 - k
                r = profilo[k]
                for dx in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if abs(dx) + abs(dz) > r + (1 if r > 1 else 0):
                            continue
                        posa(x + dx, y, z + dz, foglie)
        else:
            # latifoglia: due corone larghe, poi una stretta, poi la cima
            for y in (base + alt - 3, base + alt - 2):
                for dx in range(-2, 3):
                    for dz in range(-2, 3):
                        if abs(dx) == 2 and abs(dz) == 2 and rng.random() < 0.75:
                            continue
                        posa(x + dx, y, z + dz, foglie)
            for dx in range(-1, 2):
                for dz in range(-1, 2):
                    posa(x + dx, base + alt - 1, z + dz, foglie)
            for dx, dz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                posa(x + dx, base + alt, z + dz, foglie)

        # il tronco per ultimo: deve vincere sulle foglie. Ma non fino
        # all'apice: nell'abete la sagoma finisce con UNA foglia a
        # `base + alt - 1`, e un tronco alto `alt` la sostituiva - in gioco la
        # cima di certi abeti era un blocco di legno nudo. Nelle latifoglie
        # il tronco arriva a `alt - 1` e la cima e' un blocco piu' su.
        alt_tronco = alt - 1 if sp in (ABETE, ABETE_ALTO) else alt
        for k in range(alt_tronco):
            posa(x, base + k, z, tronco, solo_aria=False)
        messi += 1

    return messi


def conteggio(alberi: dict) -> dict[str, int]:
    fuori: dict[str, int] = {}
    for sp, nome in NOMI_SPECIE.items():
        n = int((alberi["specie"] == sp).sum())
        if n:
            fuori[nome] = n
    return fuori
