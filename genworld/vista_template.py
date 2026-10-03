"""Il visualizzatore dei template: elenca gli `.nbt` di `templates/` per
cartella/scopo e ne disegna tre proiezioni ortogonali (sopra, fronte,
fianco), cosi' si vede A COLPO D'OCCHIO cos'e' davvero un file dal nome
criptico ("cementerio-grav-9ggj8p63.nbt", "portal-del-neth-54ojw1ng.nbt")
senza dover aprire Minecraft.

Deliberatamente senza Qt: la GUI (`gui.py`) chiama solo queste funzioni e le
mette a schermo, cosi' il rendering si puo' provare (e testare) anche senza
una finestra. Riusa gli stessi colori per blocco di `anteprima.py`, che gia'
serve esattamente a "vedere in due secondi cosa c'e' davvero" - qui per un
singolo modello invece che per un mondo intero.

Una proiezione ortogonale non e' un rendering 3D: e' il primo blocco pieno
incontrato scandendo lungo un asse, come un'ombra proiettata da quel lato.
Basta per lo scopo dichiarato - "verificare l'effettivo scopo" di un file -
ed e' molto piu' semplice di un motore 3D vero. Tre viste (non una sola)
perche' una struttura verticale come un portale, vista solo dall'alto, e'
quasi tutta invisibile: si vede la cornice superiore e basta.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .anteprima import COLORE_IGNOTO, COLORI
from . import avamposti as AV
from . import template as TM

# Le sottocartelle il cui contenuto passa per la sostituzione dei blocchi non
# ancora tradotti (vedi `avamposti.sostituisci_blocchi_recenti`) PRIMA di
# essere piazzato nel mondo - la stessa scelta di `motore.py`/`avamposti.py`:
# solo cimitero e portale, le case restano cosi' come sono (vedi
# `avamposti.py`, "sostituzione condivisa da cimitero e portale"). Qui serve
# a mostrare esattamente cio' che finirebbe nel mondo, non i nomi grezzi del
# file - un blocco non tradotto altrimenti comparirebbe "sbagliato" nella
# preview anche se la generazione vera lo aggiusta.
_CARTELLE_CON_SOSTITUZIONE = {"cimiteri", "portali"}

# Le cartelle "di scopo" dentro templates/, con un'etichetta leggibile.
# L'ordine e' quello con cui compaiono nell'elenco - dal piu' usato al meno.
CARTELLE = (
    ("Case (templates/strutture)", "strutture"),
    ("Cimiteri (templates/cimiteri)", "cimiteri"),
    ("Portali (templates/portali)", "portali"),
    ("Case legacy (templates/case)", "case"),
    ("Altro / non ancora usati (templates/altro)", "altro"),
)

COLORE_VUOTO = COLORI["air"]   # sfondo delle proiezioni: stesso grigio scuro
                               # usato da anteprima.py per l'aria


# Colori di riserva per i blocchi che `anteprima.COLORI` non elenca: nei
# template ce ne sono a decine (botole, cancelli, ceramiche, tinture...) e
# lasciarli magenta rendeva illeggibile proprio la casa che si voleva
# controllare. Il legno prende il colore dal `material` del blocco, la
# tintura dal `color`; il resto e' una tabella di sottostringhe. Solo un
# blocco che nessuna regola riconosce resta magenta.
_LEGNO = {
    "oak": (162, 130, 78), "spruce": (104, 78, 46), "birch": (196, 176, 118),
    "jungle": (160, 115, 80), "acacia": (168, 90, 50), "dark_oak": (66, 43, 20),
    "mangrove": (117, 54, 48), "cherry": (226, 178, 172), "bamboo": (193, 173, 80),
    "crimson": (101, 48, 70), "warped": (43, 104, 99), "pale_oak": (227, 214, 204),
    "poplar": (170, 140, 90),
}
_TINTE = {
    "white": (233, 236, 236), "orange": (240, 118, 19), "magenta": (189, 68, 179),
    "light_blue": (58, 175, 217), "yellow": (248, 197, 39), "lime": (112, 185, 25),
    "pink": (237, 141, 172), "gray": (62, 68, 71), "light_gray": (142, 142, 134),
    "cyan": (21, 137, 145), "purple": (121, 42, 172), "blue": (53, 57, 157),
    "brown": (114, 71, 40), "green": (84, 109, 27), "red": (161, 39, 34),
    "black": (20, 21, 25),
}
_REGOLE = (
    # (sottostringa nel nome, colore): la prima che combacia vince
    ("deepslate", (77, 77, 82)),
    ("brick_block", (150, 97, 83)), ("mud_brick", (137, 104, 79)),
    ("packed_mud", (142, 107, 80)), ("quartz", (235, 229, 222)),
    ("smooth_stone", (158, 158, 158)),
    ("tuff", (108, 109, 99)), ("cinnabar", (150, 60, 55)),
    ("sulfur", (200, 190, 70)), ("blackstone", (44, 40, 46)),
    ("prismarine", (99, 156, 151)), ("red_sandstone", (186, 99, 29)),
    ("dripstone", (134, 107, 92)), ("rooted_dirt", (144, 103, 76)),
    ("podzol", (91, 63, 24)), ("terracotta", (152, 94, 68)),
    ("concrete", (207, 213, 214)), ("coral", (200, 90, 130)),
    ("bone_block", (229, 225, 207)), ("andesite", (136, 136, 136)),
    ("copper", (192, 107, 79)), ("chain", (60, 62, 70)),
    ("lightning_rod", (192, 107, 79)),
    ("moss", (89, 109, 45)), ("azalea", (101, 124, 47)),
    ("bush", (60, 100, 40)), ("firefly", (60, 100, 40)),
    ("leaf_litter", (150, 110, 50)), ("sapling", (90, 130, 50)),
    ("spore_blossom", (206, 96, 157)), ("wildflowers", (220, 180, 90)),
    ("cactus_flower", (230, 120, 160)), ("vines", (60, 110, 40)),
    ("seagrass", (50, 130, 90)), ("sugar_cane", (148, 192, 101)),
    ("bamboo", (193, 173, 80)), ("grass", (100, 150, 70)),
    ("flower_pot", (150, 80, 60)), ("decorated_pot", (150, 80, 60)),
    ("candle", (230, 220, 190)), ("bell", (220, 190, 60)),
    ("hopper", (70, 70, 74)), ("lever", (110, 110, 110)),
    ("target", (200, 180, 160)), ("beehive", (196, 160, 70)),
    ("scaffolding", (170, 140, 80)), ("carved_pumpkin", (214, 132, 40)),
    ("shroomlight", (240, 150, 70)), ("tinted_glass", (60, 40, 70)),
    ("glass", (170, 210, 225)), ("banner", (180, 60, 60)),
    ("end_rod", (240, 235, 220)), ("cuckoo_clock", (150, 110, 60)),
    ("tripwire", (200, 200, 200)), ("nether_wart", (110, 20, 25)),
    ("weeping", (150, 30, 30)), ("chest", (150, 110, 60)),
    ("frogport", (150, 130, 100)),
    ("stone", (128, 128, 128)),
)
_NOMI_DI_LEGNO = ("trapdoor", "fence_gate", "button", "pressure_plate", "sign",
                  "shelf", "wood", "planks", "door", "ladder", "fence", "stairs",
                  "slab", "log", "lectern", "barrel")


def colore_blocco(nome: str, prop: dict | None = None):
    """RGB per un blocco della tavolozza di un modello, o None se nessuna
    regola lo riconosce (il chiamante lo disegna magenta e lo segnala)."""
    prop = prop or {}
    c = COLORI.get(nome)
    if c is not None:
        return c
    if nome in ("stained_glass", "stained_glass_pane", "stained_terracotta",
                "carpet", "concrete", "concrete_powder", "wool", "candle",
                "banner", "wall_banner", "bed", "shulker_box"):
        t = prop.get("color")
        if t in _TINTE:
            return _TINTE[t]
    mat = prop.get("material") or prop.get("wood_type")
    if mat is None:
        for k in _LEGNO:                        # poplar_trapdoor, pale_oak_shelf...
            if nome.startswith(k + "_"):
                mat = k
                break
    if mat in _LEGNO and any(w in nome for w in _NOMI_DI_LEGNO):
        return _LEGNO[mat]
    for parola, colore in _REGOLE:
        if parola in nome:
            return colore
    if any(w in nome for w in _NOMI_DI_LEGNO):
        return _LEGNO["oak"]
    return None


@dataclass
class VoceTemplate:
    """Un file `.nbt` trovato in una cartella di scopo, non ancora caricato -
    l'elenco dev'essere veloce anche con centinaia di file, quindi qui c'e'
    solo quello che si legge dal nome/percorso, non dal contenuto."""
    nome: str            # nome file senza estensione, es. "casa_bosco"
    percorso: str
    cartella_scopo: str  # l'etichetta di CARTELLE, es. "Case (templates/strutture)"
    escluso: bool = False  # vedi template.Modello.stili: "_escluso" qui


@dataclass
class InfoTemplate:
    """Un modello caricato, pronto da mostrare: dimensioni, stili, porta e
    le tre proiezioni gia' renderizzate."""
    nome: str
    dx: int
    dy: int
    dz: int
    stili: frozenset[str]
    ha_porta: bool
    blocchi: list[tuple[str, int]]   # (nome blocco, quante celle) per la legenda
    ignoti: list[str]                # blocchi senza colore in COLORI
    sopra: "Image.Image" = field(repr=False)
    fronte: "Image.Image" = field(repr=False)
    fianco: "Image.Image" = field(repr=False)
    modello: object = field(default=None, repr=False)   # `Modello`, per il 3D


def elenca(radice: str) -> list[VoceTemplate]:
    """Tutti gli `.nbt` delle cartelle di scopo sotto `radice` (la cartella
    `templates/`), con lo stile letto da `stili.json` solo per sapere se un
    file e' `_escluso` (vedi `template.carica_stili`) - non serve altro per
    l'elenco, il resto si carica solo quando l'utente lo seleziona."""
    fuori: list[VoceTemplate] = []
    for etichetta, sottocartella in CARTELLE:
        cartella = os.path.join(radice, sottocartella)
        if not os.path.isdir(cartella):
            continue
        stili = TM.carica_stili(cartella)
        for nome_file in sorted(os.listdir(cartella)):
            if not nome_file.lower().endswith(".nbt"):
                continue
            nome = os.path.splitext(nome_file)[0]
            fuori.append(VoceTemplate(
                nome=nome, percorso=os.path.join(cartella, nome_file),
                cartella_scopo=etichetta,
                escluso="_escluso" in stili.get(nome, frozenset())))
    return fuori


def _proietta(celle: np.ndarray, asse: int, dal_basso: bool) -> np.ndarray:
    """Il primo indice di tavolozza non vuoto (`!= -1`) scandendo lungo
    `asse` (0=x, 1=y, 2=z), a partire da un capo o dall'altro. Ritorna una
    mappa 2D di indici (-1 dove non c'e' niente lungo tutto l'asse) sulle
    due dimensioni rimaste, nello stesso ordine in cui compaiono in `celle`
    tolto `asse`."""
    c = np.moveaxis(celle, asse, 0)
    if not dal_basso:
        c = c[::-1]
    pieno = c != -1
    ha = pieno.any(axis=0)
    idx = np.argmax(pieno, axis=0)
    valori = np.take_along_axis(c, idx[None, ...], axis=0)[0]
    return np.where(ha, valori, -1)


def _colora(mappa_indici: np.ndarray, tavolozza: list[tuple[str, dict]]
           ) -> tuple[np.ndarray, list[str]]:
    """Mappa di indici di tavolozza -> immagine RGB, coi colori di
    `anteprima.COLORI`. Ritorna anche i nomi dei blocchi senza colore (utile
    solo come diagnostica, non blocca il rendering: meglio un rettangolo
    magenta visibile che un modello che non si apre)."""
    rgb = np.zeros(mappa_indici.shape + (3,), np.uint8)
    rgb[:, :] = COLORE_VUOTO
    ignoti: set[str] = set()
    for i in np.unique(mappa_indici):
        if i < 0:
            continue
        nome, prop = tavolozza[int(i)]
        colore = colore_blocco(nome, prop)
        if colore is None:
            colore = COLORE_IGNOTO
            ignoti.add(nome)
        rgb[mappa_indici == i] = colore
    return rgb, sorted(ignoti)


def _immagine(rgb: np.ndarray, scala: int) -> "Image.Image":
    im = Image.fromarray(rgb, "RGB")
    if scala > 1:
        im = im.resize((im.width * scala, im.height * scala), Image.NEAREST)
    return im


def carica_modello(percorso: str, versione=(1, 21, 4)):
    """Il `Modello` come finirebbe nel mondo: per cimiteri e portali con la
    sostituzione dei blocchi recenti gia' applicata."""
    m = TM.carica(percorso, versione)
    cartella = os.path.basename(os.path.dirname(percorso))
    if cartella in _CARTELLE_CON_SOSTITUZIONE:
        m.tavolozza = AV.sostituisci_blocchi_recenti(m.tavolozza)
    return m


_ARIA = {"air", "cave_air", "void_air"}
# Luce per faccia, fissa nel mondo (non nello schermo): girando il modello
# le facce cambiano tono, ed e' questo che fa leggere il volume.
_LUCE = {(0, 1, 0): 1.00, (0, -1, 0): 0.50, (1, 0, 0): 0.82, (-1, 0, 0): 0.68,
         (0, 0, 1): 0.90, (0, 0, -1): 0.60}
_ANGOLI = {
    (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (-1, 0, 0): [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
    (0, 1, 0): [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (0, 0, -1): [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
}


def render_3d(m, yaw: float = 35.0, pitch: float = 30.0, taglio: int | None = None,
              lato: int = 560) -> "Image.Image":
    """Il modello come volume, in proiezione ortogonale.

    `yaw` gira attorno alla verticale (gradi), `pitch` e' l'inclinazione
    della vista (0 = di taglio, 90 = dall'alto). `taglio` tiene solo gli
    strati sotto quella quota: e' la "sezione" per guardare dentro la casa un
    piano alla volta. Non e' un motore 3D: si disegnano solo le facce che
    affacciano sul vuoto e visibili dalla vista, dalla piu' lontana alla piu'
    vicina (algoritmo del pittore), e per case da poche decine di migliaia di
    facce basta a una frazione di secondo.
    """
    celle = m.celle
    if taglio is not None:
        celle = celle.copy()
        celle[:, max(0, taglio):, :] = -1
    dx, dy, dz = celle.shape

    pal = np.zeros((len(m.tavolozza), 3), np.float32)
    solido_pal = np.zeros(len(m.tavolozza), bool)
    for i, (nome, prop) in enumerate(m.tavolozza):
        if nome in _ARIA:
            continue
        solido_pal[i] = True
        c = colore_blocco(nome, prop)
        pal[i] = c if c is not None else COLORE_IGNOTO
    pieno = (celle >= 0) & solido_pal[np.clip(celle, 0, None)]

    th, ph = np.radians(yaw), np.radians(pitch)
    cth, sth, cph, sph = np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)

    def ruota(x, z):           # attorno al centro della pianta
        x, z = x - dx / 2.0, z - dz / 2.0
        return x * cth - z * sth, x * sth + z * cth

    vista = np.array([0.0, sph, -cph])         # dalla scena verso la camera

    facce = []
    for asse, segno in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        n = [0, 0, 0]
        n[asse] = segno
        nx, ny, nz = n
        rx, rz = nx * cth - nz * sth, nx * sth + nz * cth
        if rx * vista[0] + ny * vista[1] + rz * vista[2] <= 1e-6:
            continue                            # guarda dall'altra parte
        vicino = np.zeros_like(pieno)
        dst = [slice(None)] * 3
        src = [slice(None)] * 3
        if segno > 0:
            dst[asse], src[asse] = slice(0, -1), slice(1, None)
        else:
            dst[asse], src[asse] = slice(1, None), slice(0, -1)
        vicino[tuple(dst)] = pieno[tuple(src)]
        xs, ys, zs = np.nonzero(pieno & ~vicino)
        if len(xs):
            facce.append((n, xs, ys, zs, celle[xs, ys, zs]))
    if not facce:
        return Image.new("RGB", (lato, lato), COLORE_VUOTO)

    poligoni, profondita, colori = [], [], []
    for n, xs, ys, zs, idx in facce:
        cs = np.array(_ANGOLI[tuple(n)], np.float32)                # (4, 3)
        X = xs[:, None] + cs[None, :, 0]
        Y = ys[:, None] + cs[None, :, 1]
        Z = zs[:, None] + cs[None, :, 2]
        rx, rz = ruota(X, Z)
        poligoni.append(np.stack([rx, Y * cph + rz * sph], axis=-1))   # (N, 4, 2)
        _, cz = ruota(xs + 0.5 + n[0] * 0.5, zs + 0.5 + n[2] * 0.5)
        profondita.append((ys + 0.5 + n[1] * 0.5) * sph - cz * cph)
        colori.append(np.clip(pal[idx] * _LUCE[tuple(n)], 0, 255).astype(np.uint8))
    poligoni = np.concatenate(poligoni)
    profondita = np.concatenate(profondita)
    colori = np.concatenate(colori)

    minimo = poligoni.reshape(-1, 2).min(axis=0)
    massimo = poligoni.reshape(-1, 2).max(axis=0)
    ext = massimo - minimo
    S, M = 2, 12                                # sovracampionamento, margine
    px = (lato - 2 * M) * S / float(max(ext.max(), 1e-6))
    W = int(ext[0] * px + 2 * M * S)
    H = int(ext[1] * px + 2 * M * S)
    im = Image.new("RGB", (max(W, 1), max(H, 1)), COLORE_VUOTO)
    d = ImageDraw.Draw(im)
    for k in np.argsort(profondita, kind="stable"):
        pts = [((x - minimo[0]) * px + M * S, (massimo[1] - y) * px + M * S)
               for x, y in poligoni[k]]
        c = tuple(int(v) for v in colori[k])
        d.polygon(pts, fill=c, outline=c)
    return im.resize((max(W // S, 1), max(H // S, 1)), Image.LANCZOS)


def carica_info(percorso: str, versione=(1, 21, 4), scala: int = 16) -> InfoTemplate:
    """Carica un `.nbt` e prepara tutto quello che serve a mostrarlo: le tre
    proiezioni (sopra/fronte/fianco) gia' colorate e ingrandite di `scala`,
    piu' le dimensioni, gli stili, se ha una porta e un piccolo riepilogo
    della tavolozza per la legenda. Puo' sollevare un'eccezione (file rotto,
    traduttore mancante) - chi chiama (la GUI) la mostra come messaggio,
    non fa crashare la finestra: vedi il commento in cima al modulo."""
    m = carica_modello(percorso, versione)

    sopra = _proietta(m.celle, asse=1, dal_basso=False)          # vista dall'alto
    fronte = np.flipud(_proietta(m.celle, asse=2, dal_basso=True).T)   # da sud
    fianco = np.flipud(_proietta(m.celle, asse=0, dal_basso=True))     # da ovest

    ignoti: set[str] = set()
    immagini = []
    for mappa in (sopra, fronte, fianco):
        rgb, ign = _colora(mappa, m.tavolozza)
        ignoti.update(ign)
        immagini.append(_immagine(rgb, scala))

    conteggio: dict[str, int] = {}
    valori, quante = np.unique(m.celle, return_counts=True)
    for v, q in zip(valori.tolist(), quante.tolist()):
        if v < 0:
            continue
        nome = m.tavolozza[v][0]
        conteggio[nome] = conteggio.get(nome, 0) + q
    blocchi = sorted(conteggio.items(), key=lambda kv: -kv[1])

    return InfoTemplate(
        nome=m.nome, dx=m.dx, dy=m.dy, dz=m.dz, stili=m.stili,
        ha_porta=m.porta is not None, blocchi=blocchi, ignoti=sorted(ignoti),
        sopra=immagini[0], fronte=immagini[1], fianco=immagini[2], modello=m)
