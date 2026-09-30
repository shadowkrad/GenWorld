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

from .mappa import (ACQUA, DESERTO, FORESTA, MONTAGNA, NEVE, PIANURA,
                    PRATERIA, SPIAGGIA)
from .rumore import fbm

# specie
(QUERCIA, BETULLA, ABETE, CACTUS, ARBUSTO, CESPUGLIO, MELO, CILIEGIO, BACCHE,
 DENTE_DI_LEONE, PAPAVERO, TULIPANO, FIORDALISO, ALLIUM, GIGLIO_VALLE,
 FUNGO_MARRONE, FUNGO_ROSSO, CANNA) = range(18)

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
    FORESTA:  [(QUERCIA, 0.50), (BETULLA, 0.22), (ABETE, 0.18),
               (GIGLIO_VALLE, 0.06), (FUNGO_MARRONE, 0.03), (FUNGO_ROSSO, 0.01)],
    PRATERIA: [(QUERCIA, 0.55), (BETULLA, 0.15), (CESPUGLIO, 0.08),
               (DENTE_DI_LEONE, 0.08), (PAPAVERO, 0.06), (TULIPANO, 0.04),
               (FIORDALISO, 0.02), (ALLIUM, 0.02)],
    PIANURA:  [(QUERCIA, 0.55), (CESPUGLIO, 0.25), (DENTE_DI_LEONE, 0.10),
               (PAPAVERO, 0.10)],
    MONTAGNA: [(ABETE, 0.85), (CESPUGLIO, 0.15)],
    NEVE:     [(ABETE, 1.0)],
    DESERTO:  [(CACTUS, 0.55), (ARBUSTO, 0.45)],
    SPIAGGIA: [(ARBUSTO, 1.0)],
}

MATERIALE = {QUERCIA: "oak", BETULLA: "birch", ABETE: "spruce",
             MELO: "oak", CILIEGIO: "cherry"}
ALTEZZA = {QUERCIA: (4, 7), BETULLA: (5, 8), ABETE: (6, 12),
           CACTUS: (2, 5), ARBUSTO: (1, 2), CESPUGLIO: (1, 2),
           MELO: (4, 6), CILIEGIO: (5, 7), BACCHE: (1, 2),
           DENTE_DI_LEONE: (1, 2), PAPAVERO: (1, 2), TULIPANO: (1, 2),
           FIORDALISO: (1, 2), ALLIUM: (1, 2), GIGLIO_VALLE: (1, 2),
           FUNGO_MARRONE: (1, 2), FUNGO_ROSSO: (1, 2), CANNA: (1, 4)}

# Dove puo' crescere la canna da zucchero: SOLO sul bordo dell'acqua, come in
# Minecraft davvero - non e' una questione di classe di terreno ma di essere
# a un blocco esatto da un fiume, un lago o il mare.
CANNA_CLASSI = (SPIAGGIA, PIANURA, PRATERIA, FORESTA)

# Ingombro orizzontale massimo, in blocchi dal centro. Serve a sapere quali
# alberi di chunk vicini possono sporgere dentro questo.
RAGGIO_MAX = 3


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
) -> dict[str, np.ndarray]:
    """Dove piantare. Ritorna array paralleli in coordinate della mappa.

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
            for sp in (QUERCIA, BETULLA, ABETE, MELO, CILIEGIO)
        }
        self.foglie = {
            sp: s.blocco("leaves", check_decay="false", distance="7",
                         material=MATERIALE[sp], persistent="true")
            for sp in (QUERCIA, BETULLA, ABETE, MELO, CILIEGIO)
        }
        self.bacche = s.blocco("sweet_berry_bush", age="3")
        self.cactus = s.blocco("cactus", age="0")
        self.arbusto = s.blocco("plant", plant_type="dead_bush")
        self.erba_alta = s.blocco("plant", plant_type="grass")
        self.felce = s.blocco("plant", plant_type="fern")
        self.canna = s.blocco("sugar_cane", age="0")
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
        self.aria = s.id_aria


# profilo dell'abete dall'alto verso il basso: raggio per livello
PROFILO_ABETE = (0, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2)


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
        if sp == BACCHE:
            posa(x, base, z, tav.bacche)
            messi += 1
            continue
        if sp in FIORI:
            posa(x, base, z, tav.fiore[sp])
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

        if sp == ABETE:
            livelli = min(len(PROFILO_ABETE), max(3, alt - 1))
            for k in range(livelli):
                y = base + alt - 1 - k
                r = PROFILO_ABETE[k]
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
        alt_tronco = alt - 1 if sp == ABETE else alt
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
