"""Strati di roccia, roccia esposta e linea delle nevi.

Guardando le montagne di Arda in gioco si vedevano due cose: erano un blocco
solo dalla base alla cima - `stone` e basta - e la loro superficie era a righe
verticali, come un gelato alla crema. Le righe erano il dithering della
quantizzazione applicato dove non serviva (corretto in `rumore.quantizza`); il
blocco solo e' questo modulo.

Una montagna vera e' fatta a **strati**. Gli strati sono orizzontali, spessi
metri, e si vedono soprattutto dove la roccia e' scoperta, cioe' sulle pareti.
Due proprieta' contano piu' di tutte:

* **sono globali.** Lo stesso banco di arenaria si ritrova su due versanti
  opposti della stessa valle, alla stessa quota. E' questo che fa leggere
  un paesaggio come un paesaggio e non come del rumore colorato. Quindi la
  tavola degli strati e' UNA per tutta la mappa, indicizzata dalla quota.
* **non sono piatti.** Gli strati si piegano. Si aggiunge quindi uno
  scostamento per colonna, preso da un rumore largo: lo strato ondula di
  qualche blocco su scala di centinaia, e nelle sezioni si vede la piega.

Poi due regole di superficie:

* dove il pendio e' ripido non cresce niente: niente erba, niente neve, solo
  roccia nuda e ghiaione. E' la regola che da' le pareti.
* la neve non comincia a una quota netta. Fra il limite inferiore e quello
  superiore c'e' una fascia in cui la neve e' a chiazze, con lo strato sottile
  (`snow` posato sopra la roccia) invece del blocco pieno.
"""

from __future__ import annotations

import numpy as np

from .rumore import fbm

# Gli strati si applicano solo in alto: sotto ci pensa `sottosuolo`, che
# lavora sulla pietra liscia e che i banchi di roccia confonderebbero.
QUOTA_STRATI = 40

# La tavola copre tutta la colonna di mondo utile.
Y_TAVOLA_MIN = 0
Y_TAVOLA_MAX = 320

# Scostamento massimo, in blocchi, della piega degli strati.
PIEGA = 7

# Repertorio: nome universale, proprieta', peso, spessore (min, max).
# La pietra pesa piu' di tutto il resto messo insieme, perche' una montagna
# a bande tutte diverse sembra un campionario di minerali, non una montagna.
BANCHI = (
    ("stone", {}, 9.0, (6, 22)),
    ("andesite", {"polished": "false"}, 2.2, (3, 9)),
    ("diorite", {"polished": "false"}, 1.2, (2, 6)),
    ("granite", {"polished": "false"}, 1.4, (3, 8)),
    ("tuff", {}, 1.0, (2, 7)),
    ("calcite", {}, 0.30, (1, 2)),
    ("gravel", {}, 0.35, (1, 2)),
)

# Neve: sotto il primo limite mai, sopra il secondo sempre, in mezzo a chiazze.
NEVE_BASSA = 150
NEVE_ALTA = 185

# Pendenza (blocchi di quota per cella) oltre la quale la roccia resta nuda.
PENDENZA_PARETE = 1.6
PENDENZA_GHIAIONE = 1.0


def tavola(seed: int = 0) -> list[tuple[str, dict]]:
    """La successione degli strati, una per tutta la mappa.

    Indicizzata da `y - Y_TAVOLA_MIN`. Si costruisce una volta e si legge per
    colonna: e' quello che rende gli strati riconoscibili da un versante
    all'altro.
    """
    rng = np.random.default_rng(seed * 7919 + 13)
    pesi = np.array([b[2] for b in BANCHI], np.float64)
    pesi = pesi / pesi.sum()

    fuori: list[tuple[str, dict]] = []
    while len(fuori) < Y_TAVOLA_MAX - Y_TAVOLA_MIN:
        k = int(rng.choice(len(BANCHI), p=pesi))
        nome, prop, _, (s0, s1) = BANCHI[k]
        for _ in range(int(rng.integers(s0, s1 + 1))):
            fuori.append((nome, prop))
    return fuori[:Y_TAVOLA_MAX - Y_TAVOLA_MIN]


def piega(lato: int, seed: int = 0) -> np.ndarray:
    """Scostamento in blocchi degli strati, per cella di mappa.

    Due ottave larghe: gli strati si piegano su scala di centinaia di celle,
    non di due. Un rumore fine qui darebbe strati frastagliati, che in natura
    non esistono.
    """
    n = fbm(lato, ottave=3, celle_base=max(2, lato // 220),
            persistenza=0.55, seed=seed + 4409) - 0.5
    return np.round(n * 2.0 * PIEGA).astype(np.int32)


class TavolozzaRoccia:
    """Id di palette degli strati, risolti sul livello aperto."""

    def __init__(self, scrittore, seed: int = 0):
        self.tavola = tavola(seed)
        cache: dict[tuple, int] = {}
        ids = []
        for nome, prop in self.tavola:
            chiave = (nome, tuple(sorted(prop.items())))
            if chiave not in cache:
                cache[chiave] = scrittore.blocco(nome, **prop)
            ids.append(cache[chiave])
        self.per_quota = np.array(ids, np.uint32)
        self.ids = tuple(sorted(set(ids)))
        self.neve = scrittore.blocco("snow_block")
        self.manto = scrittore.blocco("snow", layers="1")
        self.ghiaione = scrittore.blocco("gravel")
        self.pietra = scrittore.blocco("stone")

    def colonna(self, ys: np.ndarray, scarto: np.ndarray) -> np.ndarray:
        """Blocco di strato per ogni (colonna, quota).

        `ys` ha forma (1, H, 1), `scarto` (16, 1, 16): il prodotto cartesiano
        viene da solo, senza cicli.
        """
        i = np.clip(ys + scarto - Y_TAVOLA_MIN, 0, len(self.tavola) - 1)
        return self.per_quota[i]


def applica(out: np.ndarray, tav: TavolozzaRoccia, ys: np.ndarray,
            solido: np.ndarray, scarto: np.ndarray) -> None:
    """Sostituisce la pietra con lo strato della sua quota, sopra QUOTA_STRATI.

    Si tocca SOLO quello che e' pietra: la superficie (erba, sabbia, basalto)
    l'ha gia' decisa chi ci ha costruito sopra, e un banco di tufo che si
    mangia un prato non e' una stratificazione, e' un bug.
    """
    alto = solido & (ys >= QUOTA_STRATI) & (out == tav.pietra)
    if not alto.any():
        return
    strati = tav.colonna(ys, scarto)
    out[alto] = np.broadcast_to(strati, out.shape)[alto]


def superficie_montana(h: np.ndarray, pendenza: np.ndarray, seed: int = 0,
                       freddo: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Decide, per cella, se la roccia e' nuda e quanta neve c'e'.

    Ritorna (nuda, neve) dove `neve` vale 0 = niente, 1 = manto sottile,
    2 = blocco pieno.

    `freddo` e' il campo climatico di `biomi`: senza, la linea delle nevi
    sarebbe la stessa all'equatore e al polo. Con un limite solo altimetrico
    la prima prova su Arda ha imbiancato tutte le montagne della mappa,
    comprese quelle in mezzo al deserto.
    """
    nuda = pendenza >= PENDENZA_PARETE

    lato = max(h.shape)
    chiazze = fbm(lato, ottave=3, celle_base=max(4, lato // 60),
                  persistenza=0.55, seed=seed + 77)[:h.shape[0], :h.shape[1]]
    basso, alto = NEVE_BASSA, NEVE_ALTA
    if freddo is not None:
        # il clima sposta la linea delle nevi in su o in giu' di trenta
        # blocchi: sul versante freddo comincia presto, su quello caldo tardi
        scostamento = (0.5 - np.clip(freddo, 0.0, 1.0)) * 70.0
        basso = NEVE_BASSA + scostamento
        alto = NEVE_ALTA + scostamento
    t = np.clip((h - basso) / np.maximum(alto - basso, 1.0), 0.0, 1.0)
    # La chiazza decide dove la neve attacca per prima: la soglia e' la quota
    # relativa, cosi' salendo la neve non compare tutta insieme.
    attacca = t > (chiazze * 0.85 + 0.08)
    neve = np.where(attacca, 1, 0).astype(np.uint8)
    neve = np.where(t >= 0.999, 2, neve).astype(np.uint8)
    # su una parete la neve non tiene
    neve = np.where(pendenza >= PENDENZA_PARETE, 0, neve).astype(np.uint8)
    return nuda, neve


def statistiche(h: np.ndarray, pendenza: np.ndarray, seed: int = 0) -> dict:
    nuda, neve = superficie_montana(h, pendenza, seed)
    n = h.size
    return {
        "roccia_nuda": float(nuda.sum() / n),
        "manto": float((neve == 1).sum() / n),
        "neve_piena": float((neve == 2).sum() / n),
        "banchi_distinti": len({b for b, _ in tavola(seed)}),
    }


def verifica_traduzioni(versione=(1, 21, 4)) -> list[str]:
    """Controlla che ogni roccia arrivi davvero in gioco.

    E' la guardia contro la trappola di `oak_log`: un nome che non esiste nel
    namespace universale non da' errore, si scrive e si rilegge identico dal
    file region, e semplicemente in gioco non compare. L'abbiamo pagata due
    volte - con i tronchi e con i biomi - e questa e' la terza famiglia di
    nomi nuovi.
    """
    import PyMCTranslate
    from amulet.api.block import Block
    from amulet_nbt import StringTag

    ver = PyMCTranslate.new_translation_manager().get_version("java", versione)
    guai = []
    nomi = [(n, p) for n, p, _, _ in BANCHI] + [
        ("snow_block", {}), ("snow", {"layers": "1"}), ("gravel", {}),
        ("stone", {}),
    ]
    for nome, prop in nomi:
        b = Block("universal_minecraft", nome,
                  {k: StringTag(v) for k, v in prop.items()})
        fuori = ver.block.from_universal(b)[0]
        if fuori.namespace != "minecraft":
            guai.append(f"{nome} -> {fuori.namespaced_name}")
    return guai
