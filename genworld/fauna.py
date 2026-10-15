"""Animali, piazzati a mano come gli abitanti.

Minecraft farebbe comunque spawnare mob naturali col tempo, in base al
bioma - che e' gia' corretto, vedi `biomi.py`. Ma "col tempo" non e' "appena
apri il mondo", ed e' quello che questo modulo aggiunge: gli stessi file
`entities/*.mca` che scrive `entita.py`, con lo stesso meccanismo (un
`Animale` sa produrre il proprio tag NBT, esattamente come un `Abitante`),
cosi' i due si scrivono in un'unica passata invece di rischiare che il
secondo scavalchi il file region del primo.

Due famiglie, due ragioni diverse di esistere:

* **da cortile** - mucche, pecore, galline, maiali - dentro i poderi: sono
  la controparte animale degli abitanti nelle botteghe, la prova che quella
  campagna e' coltivata da qualcuno.
* **selvatica** - per classe di terreno, con la stessa logica di
  `vegetazione.MISCELA`: un lupo nel bosco, un cammello nel deserto, un
  orso polare sulla neve. Mai un branco finto: la densita' resta bassa
  apposta, questi sono animali grossi che si notano uno per uno.

A differenza degli alberi, un animale non e' un blocco: non serve
ritagliarlo per chunk ne' disegnarlo - una posizione e una specie bastano,
esattamente come per un `Abitante`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import biomi as B
from .mappa import (DESERTO, FORESTA, MARE, MONTAGNA, NEVE, OCEANO, PIANURA,
                    PRATERIA, SPIAGGIA)

# nome nostro -> id di gioco
SPECIE = {
    "mucca": "cow", "pecora": "sheep", "gallina": "chicken", "maiale": "pig",
    "lupo": "wolf", "coniglio": "rabbit", "volpe": "fox", "capra": "goat",
    "cammello": "camel", "cavallo": "horse", "orso_polare": "polar_bear", "tartaruga": "turtle",
    "lama": "llama", "asino": "donkey", "armadillo": "armadillo", "panda": "panda",
    "pappagallo": "parrot", "ocelotto": "ocelot", "ape": "bee", "gatto": "cat",
    "rana": "frog", "merluzzo": "cod", "salmone": "salmon",
    "pesce_tropicale": "tropical_fish", "calamaro": "squid", "delfino": "dolphin",
    "pesce_palla": "pufferfish", "calamaro_luminoso": "glow_squid",
}

# chi vive in acqua: nessun "OnGround", si piazza dentro l'acqua e non sul fondo
ACQUATICI = ("merluzzo", "salmone", "pesce_tropicale", "calamaro", "delfino",
             "pesce_palla", "calamaro_luminoso")

# vita massima vera di ciascuno: mettere 10 dappertutto per comodita' vorrebbe
# dire scrivere una gallina con piu' salute di un lupo.
SALUTE = {
    "mucca": 10.0, "pecora": 8.0, "gallina": 4.0, "maiale": 10.0,
    "lupo": 8.0, "coniglio": 3.0, "volpe": 10.0, "capra": 10.0,
    "cammello": 32.0, "cavallo": 22.0, "orso_polare": 30.0, "tartaruga": 30.0,
    "lama": 15.0, "asino": 15.0, "armadillo": 12.0, "panda": 20.0,
    "pappagallo": 6.0, "ocelotto": 10.0, "ape": 10.0, "gatto": 10.0, "rana": 10.0,
    "merluzzo": 3.0, "salmone": 3.0, "pesce_tropicale": 3.0, "calamaro": 10.0,
    "delfino": 10.0, "pesce_palla": 3.0, "calamaro_luminoso": 10.0,
}

CORTILE = ("mucca", "pecora", "gallina", "maiale")

# specie selvatiche per classe di terreno, con i rispettivi pesi - stesso
# principio di vegetazione.MISCELA: niente in acqua, niente in montagna dove
# in bosco non c'entra, niente animale da bioma caldo sulla neve.
MISCELA = {
    FORESTA:  [("lupo", 0.5), ("coniglio", 0.5)],
    PRATERIA: [("coniglio", 0.6), ("volpe", 0.4)],
    PIANURA:  [("coniglio", 1.0)],
    MONTAGNA: [("capra", 0.7), ("volpe", 0.3)],
    NEVE:     [("orso_polare", 0.3), ("volpe", 0.4), ("coniglio", 0.3)],
    DESERTO:  [("cammello", 1.0)],
    SPIAGGIA: [("tartaruga", 1.0)],
}

# Dove il bioma ha una sua fauna, prende il posto della miscela della classe
# (stessa idea di `vegetazione.MISCELA_BIOMA`): la savana ha lama, asini e
# armadilli, la giungla panda, pappagalli e ocelotti, i prati fioriti le api.
MISCELA_BIOMA = {
    "savanna": [("lama", 0.3), ("asino", 0.2), ("armadillo", 0.2), ("cavallo", 0.2),
                ("coniglio", 0.1)],
    "jungle": [("panda", 0.3), ("pappagallo", 0.4), ("ocelotto", 0.3)],
    "sunflower_plains": [("ape", 0.5), ("cavallo", 0.3), ("coniglio", 0.2)],
    "meadow": [("ape", 0.4), ("coniglio", 0.3), ("volpe", 0.15), ("cavallo", 0.15)],
    "plains": [("cavallo", 0.3), ("asino", 0.1), ("coniglio", 0.4), ("ape", 0.2)],
    "desert": [("cammello", 0.7), ("armadillo", 0.3)],
    "forest": [("lupo", 0.4), ("coniglio", 0.3), ("ape", 0.15), ("volpe", 0.15)],
    "taiga": [("lupo", 0.5), ("volpe", 0.3), ("coniglio", 0.2)],
    "mangrove_swamp": [("rana", 0.7), ("coniglio", 0.3)],
    "snowy_taiga": [("lupo", 0.4), ("volpe", 0.4), ("coniglio", 0.2)],
}

# animali per blocco quadrato: molto piu' radi degli alberi apposta, sono
# bestie grosse che si notano una per una, non un tappeto.
DENSITA = {
    FORESTA: 0.0009, PRATERIA: 0.0007, PIANURA: 0.0004, MONTAGNA: 0.0006,
    NEVE: 0.0004, DESERTO: 0.0003, SPIAGGIA: 0.0006,
}

# Varianti di colore per bioma - solo dove il gioco stesso le differenzia,
# verificate sulla wiki di Minecraft (voci "Fox" e "Rabbit", sezioni
# "Entity data" e "Spawning") e non indovinate, con lo stesso criterio con
# cui si verificano i nomi dei blocchi con PyMCTranslate:
#
# * la volpe ha un solo tag, "Type" (stringa "red"/"snow"). In gioco spawna
#   rossa nelle taighe e nei boschi, bianca ("snow") in grove e taiga
#   innevata - qui semplificato a "bianca sulla neve, rossa altrove", visto
#   che la nostra classe NEVE non distingue le due.
# * il coniglio ha "RabbitType" (intero): 0 marrone, 1 bianco, 2 nero,
#   3 bianco a chiazze, 4 dorato, 5 sale e pepe, 99 il coniglio killer (mai
#   usato qui, non e' un animale che si piazza a mano).
CONIGLIO_MARRONE, CONIGLIO_BIANCO, CONIGLIO_NERO = 0, 1, 2
CONIGLIO_CHIAZZE, CONIGLIO_DORATO, CONIGLIO_SALE = 3, 4, 5

# classi non elencate -> nessuna voce, la volpe resta "red" (il colore
# base del gioco, quello che compare senza il tag Type).
VOLPE_NEVOSA = (NEVE,)

# classi non elencate -> il misto "temperato" di CONIGLIO_TIPO_DEFAULT,
# lo stesso mix che spawnerebbe naturalmente il gioco in bosco/prateria.
CONIGLIO_TIPO = {
    NEVE: [(CONIGLIO_BIANCO, 0.8), (CONIGLIO_CHIAZZE, 0.2)],
}
CONIGLIO_TIPO_DEFAULT = [(CONIGLIO_MARRONE, 0.5), (CONIGLIO_SALE, 0.4),
                         (CONIGLIO_NERO, 0.1)]


def _pesca(pesi, r: np.ndarray) -> np.ndarray:
    """Sceglie un valore pesato per ciascun elemento di `r` (0..1) - stessa
    logica a soglie cumulative usata per le specie, isolata qui perche' la
    si usa due volte (volpe e coniglio)."""
    scelta = np.full(r.size, pesi[-1][0], dtype=object)
    soglia = 0.0
    for val, peso in pesi:
        scelta[(r >= soglia) & (r < soglia + peso)] = val
        soglia += peso
    return scelta


@dataclass
class Animale:
    """Un animale da scrivere nei file. Coordinate in blocchi del mondo."""
    x: float
    y: float
    z: float
    specie: str = "mucca"
    seme: int = 0
    # Colore/variante per bioma: stringa per la volpe ("red"/"snow"), intero
    # per il coniglio (vedi CONIGLIO_* sopra). None = il colore base del
    # gioco, e per ogni altra specie il campo non serve a nulla.
    variante: object = None

    def nbt_tag(self):
        from amulet_nbt import (ByteTag, CompoundTag, DoubleTag, FloatTag,
                                IntTag, ListTag, StringTag)

        from .entita import _uuid
        nome_gioco = SPECIE.get(self.specie, "cow")
        salute = SALUTE.get(self.specie, 10.0)
        base = {
            "id": StringTag(f"minecraft:{nome_gioco}"),
            "Pos": ListTag([DoubleTag(self.x), DoubleTag(self.y), DoubleTag(self.z)]),
            "Motion": ListTag([DoubleTag(0.0), DoubleTag(0.0), DoubleTag(0.0)]),
            "Rotation": ListTag([FloatTag(0.0), FloatTag(0.0)]),
            "UUID": _uuid(self.seme),
            "Health": FloatTag(salute),
            "Air": IntTag(300),
            "Fire": IntTag(-1),
            "FallDistance": FloatTag(0.0),
            "Invulnerable": ByteTag(0),
            "OnGround": ByteTag(0 if self.specie in ACQUATICI else 1),
            # senza questo un animale lontano dal giocatore puo' sparire nel
            # giro di pochi minuti - stessa ragione degli abitanti
            "PersistenceRequired": ByteTag(1),
        }
        if self.specie == "volpe":
            base["Type"] = StringTag(self.variante if self.variante else "red")
        elif self.specie == "coniglio":
            v = self.variante if self.variante is not None else CONIGLIO_MARRONE
            base["RabbitType"] = IntTag(int(v))
        return CompoundTag(base)


def semina_selvatica(
    cls: np.ndarray,
    altezze: np.ndarray,
    livello_mare: int = 62,
    passo: int = 24,
    scala_densita: float = 1.0,
    seed: int = 0,
    biomi: np.ndarray | None = None,
) -> list[Animale]:
    """Fauna selvatica sparsa per classe di terreno (e, con `biomi`, per bioma).

    Stessa griglia sfalsata degli alberi (un candidato per cella, spostato a
    caso dentro la cella) per lo stesso motivo: rumore blu a costo zero,
    niente branchi ne' vuoti. Il passo e' molto piu' largo di quello degli
    alberi apposta - questi sono animali radi, non una foresta.
    """
    rng = np.random.default_rng(seed + 5253)
    H, W = cls.shape
    gx, gz = np.meshgrid(np.arange(0, W - 1, passo),
                         np.arange(0, H - 1, passo), indexing="xy")
    gx = gx.ravel(); gz = gz.ravel()
    n = gx.size
    if n == 0:
        return []
    x = np.clip(gx + rng.integers(0, passo, n), 0, W - 1)
    z = np.clip(gz + rng.integers(0, passo, n), 0, H - 1)

    c = cls[z, x]
    h = altezze[z, x]

    prob = np.zeros(n, np.float32)
    for classe, d in DENSITA.items():
        prob[c == classe] = min(1.0, d * passo * passo * scala_densita)
    tieni = (rng.random(n) < prob) & (h > livello_mare)

    x, z, h, c = x[tieni], z[tieni], h[tieni], c[tieni]
    if x.size == 0:
        return []

    r = rng.random(x.size)
    specie = np.full(x.size, "", dtype=object)
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
        r3 = rng.random(x.size)
        for nome, miscela in MISCELA_BIOMA.items():
            m = b == B.INDICE[nome]
            if not m.any():
                continue
            soglia = 0.0
            for sp, peso in miscela:
                dentro = m & (r3 >= soglia) & (r3 < soglia + peso)
                specie[dentro] = sp
                soglia += peso
            specie[m & (r3 >= soglia)] = miscela[-1][0]
    tieni2 = specie != ""
    x, z, h, c, specie = x[tieni2], z[tieni2], h[tieni2], c[tieni2], specie[tieni2]

    # Variante di colore per bioma - un'estrazione indipendente da quella
    # della specie: riusare `r` qui sarebbe scorretto, perche' dentro un
    # sottoinsieme gia' filtrato (es. "solo i conigli") i valori di `r` non
    # sono piu' distribuiti uniformemente su 0..1.
    r2 = rng.random(x.size)
    variante = np.full(x.size, None, dtype=object)

    m_volpe = specie == "volpe"
    if m_volpe.any():
        variante[m_volpe] = "red"
        m_neve = m_volpe & np.isin(c, VOLPE_NEVOSA)
        variante[m_neve] = "snow"

    m_con = specie == "coniglio"
    if m_con.any():
        fatto = np.zeros(x.size, bool)
        for classe, pesi in CONIGLIO_TIPO.items():
            m = m_con & (c == classe)
            if m.any():
                variante[m] = _pesca(pesi, r2[m])
                fatto |= m
        resto = m_con & ~fatto
        if resto.any():
            variante[resto] = _pesca(CONIGLIO_TIPO_DEFAULT, r2[resto])

    semi = rng.integers(0, 2 ** 31 - 1, x.size)
    return [Animale(x=float(xi) + 0.5, y=float(hi), z=float(zi) + 0.5,
                    specie=str(spi), seme=int(si), variante=vi)
            for xi, zi, hi, spi, si, vi in zip(x, z, h, specie, semi, variante)]


# Fauna del mare, per bioma dell'acqua. (specie, peso, quanti nel branco)
FAUNA_MARINA = {
    "warm_ocean": [("pesce_tropicale", 0.5, 4), ("pesce_palla", 0.2, 1),
                   ("calamaro", 0.15, 1), ("delfino", 0.15, 2)],
    "deep_warm_ocean": [("pesce_tropicale", 0.4, 4), ("pesce_palla", 0.2, 1),
                        ("calamaro", 0.2, 1), ("delfino", 0.2, 2)],
    "ocean": [("merluzzo", 0.45, 4), ("salmone", 0.1, 3), ("calamaro", 0.2, 1),
              ("delfino", 0.25, 2)],
    "deep_ocean": [("merluzzo", 0.3, 3), ("calamaro", 0.3, 1), ("delfino", 0.25, 2),
                   ("calamaro_luminoso", 0.15, 1)],
    "cold_ocean": [("salmone", 0.45, 3), ("merluzzo", 0.4, 4), ("calamaro", 0.15, 1)],
    "deep_cold_ocean": [("salmone", 0.4, 3), ("merluzzo", 0.3, 3), ("calamaro", 0.15, 1),
                        ("calamaro_luminoso", 0.15, 1)],
}
PROFONDITA_MARINA = 4          # sotto questo non c'e' acqua per nuotare


def semina_marina(cls: np.ndarray, altezze: np.ndarray, biomi: np.ndarray,
                  livello_mare: int = 62, passo: int = 30, scala_densita: float = 1.0,
                  seed: int = 0) -> list[Animale]:
    """Pesci, calamari e delfini nel mare, per bioma dell'acqua.

    Si piazzano a mezz'acqua, tra il fondo e il pelo; in un branco i pesci sono
    vicini ma non nella stessa cella. Il mare gelato non ha fauna.
    """
    rng = np.random.default_rng(seed + 7475)
    H, W = cls.shape
    gx, gz = np.meshgrid(np.arange(0, W - 1, passo), np.arange(0, H - 1, passo),
                         indexing="xy")
    gx, gz = gx.ravel(), gz.ravel()
    if gx.size == 0:
        return []
    x = np.clip(gx + rng.integers(0, passo, gx.size), 0, W - 1)
    z = np.clip(gz + rng.integers(0, passo, gz.size), 0, H - 1)
    fuori: list[Animale] = []
    for xi, zi in zip(x.tolist(), z.tolist()):
        if int(cls[zi, xi]) not in (MARE, OCEANO):
            continue
        prof = livello_mare - int(altezze[zi, xi])
        if prof < PROFONDITA_MARINA or rng.random() > 0.55 * scala_densita:
            continue
        miscela = FAUNA_MARINA.get(B.BIOMI[int(biomi[zi, xi])])
        if not miscela:
            continue
        r = rng.random()
        soglia = 0.0
        scelta = miscela[-1]
        for voce in miscela:
            if r < soglia + voce[1]:
                scelta = voce
                break
            soglia += voce[1]
        sp, _, branco = scelta
        for _ in range(branco):
            px = xi + 0.5 + float(rng.uniform(-2, 2))
            pz = zi + 0.5 + float(rng.uniform(-2, 2))
            ix, iz = int(px), int(pz)
            if not (0 <= ix < W and 0 <= iz < H) or int(cls[iz, ix]) not in (MARE, OCEANO):
                continue
            fondo = int(altezze[iz, ix])
            if livello_mare - fondo < PROFONDITA_MARINA:
                continue
            y = float(rng.uniform(fondo + 2, livello_mare - 1))
            fuori.append(Animale(x=px, y=y, z=pz, specie=sp,
                                 seme=int(rng.integers(0, 2 ** 31 - 1))))
    return fuori


def gatti(citta: list, tipo_strada, altezze: np.ndarray, per_citta: tuple[int, int] = (1, 4),
          seed: int = 0) -> list[Animale]:
    """Qualche gatto per citta', sulle strade attorno alla piazza.

    Si scelgono celle di strada entro 14 blocchi dalla piazza: un gatto in
    mezzo a una via non finisce mai dentro un muro o una casa.
    """
    rng = np.random.default_rng(seed + 8586)
    fuori: list[Animale] = []
    if tipo_strada is None:
        return fuori
    H, W = tipo_strada.shape
    for c in citta:
        pz, px = c.piazza
        z0, z1 = max(0, pz - 14), min(H, pz + 15)
        x0, x1 = max(0, px - 14), min(W, px + 15)
        zz, xx = np.nonzero(tipo_strada[z0:z1, x0:x1] > 0)
        if zz.size == 0:
            continue
        n = int(rng.integers(per_citta[0], per_citta[1] + 1))
        for i in rng.choice(zz.size, size=min(n, zz.size), replace=False):
            z, x = int(zz[i]) + z0, int(xx[i]) + x0
            fuori.append(Animale(x=x + 0.5, y=float(altezze[z, x]), z=z + 0.5,
                                 specie="gatto",
                                 seme=int(rng.integers(0, 2 ** 31 - 1))))
    return fuori


def per_poderi(poderi: list, quanti: tuple[int, int] = (1, 3),
              seed: int = 0) -> list[Animale]:
    """Animali da cortile dentro i campi coltivati (non nei frutteti: li'
    ci stanno gli alberi da frutto, non le bestie)."""
    rng = np.random.default_rng(seed + 6364)
    fuori: list[Animale] = []
    for p in poderi:
        if getattr(p, "frutteto", False):
            continue
        n = int(rng.integers(quanti[0], quanti[1] + 1))
        for _ in range(n):
            sp = CORTILE[int(rng.integers(0, len(CORTILE)))]
            margine = max(1.0, p.lato * 0.15)
            px = p.x + margine + rng.random() * (p.lato - 2 * margine)
            pz = p.z + margine + rng.random() * (p.lato - 2 * margine)
            fuori.append(Animale(x=float(px), y=float(p.base), z=float(pz),
                                 specie=sp, seme=int(rng.integers(0, 2 ** 31 - 1))))
    return fuori


def conteggio(animali: list[Animale]) -> dict[str, int]:
    fuori: dict[str, int] = {}
    for a in animali:
        fuori[a.specie] = fuori.get(a.specie, 0) + 1
    return fuori
