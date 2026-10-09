"""Il castello (uno per mappa) e le navi (ogni tanto, in mare).

Sono due strutture da schema (`.schem` di WorldEdit, `.litematic`, `.nbt`) che
non sono case: troppo grandi per un lotto, e con un posto loro. Il castello va
su un terreno quasi piano lontano da tutto, e il terreno si adatta a lui con
una scarpata irregolare (come per le citta': vedi `citta._spiana`). La nave va
su mare profondo, con la linea di galleggiamento del modello alla quota del
mare.

Si pianificano come `CasaIsolata`: stessa posa, stesso indice per chunk, stessa
protezione del sottosuolo. Qui c'e' solo cio' che le distingue: caricare i
modelli e scegliere il posto.

La nave dello schema porta con se' il proprio mare (un parallelepipedo
d'acqua attorno allo scafo): si toglie, cosi' non rimane un blocco d'acqua
sospeso se il mare e' piu' basso, e si tiene invece l'aria dentro lo scafo, che
altrimenti si allagherebbe.
"""

from __future__ import annotations

import os

import numpy as np

from . import template as TM
from .isolate import CasaIsolata

ESTENSIONI = (".nbt", ".schem", ".litematic", ".schematic")
SOLIDI_MINIMI_NAVE = 450       # sotto questa misura un pezzo di un pacchetto non e' una nave

# quanto il terreno intorno al castello si adatta (celle) e con che pendenza
FASCIA_CASTELLO = 14
PENDENZA_CASTELLO = 0.5
DISLIVELLO_CASTELLO = 16        # tra il 5 e il 95 percentile, sul sedime: poi lo si appiana
MARGINE_CASTELLO = 8            # dal bordo della mappa
LIBERO_CASTELLO = 3             # di spazio senza strade, case, acqua attorno al sedime
DISTANZA_NAVI = 260             # fra due navi
MARGINE_NAVE = 4                # di mare libero attorno allo scafo
CELLE_PER_NAVE = 500_000        # una nave ogni tanto: mappa 1000x1000 -> 2


def _cartelle(testo: str) -> list[str]:
    return [c.strip() for c in testo.split(";") if c.strip()]


def _nave(percorso: str, versione) -> TM.Modello:
    """Una nave: l'acqua dello schema via, l'aria dello scafo tenuta."""
    voci, celle = TM._leggi_schem(percorso, tieni_aria=True)
    nomi = [n for n, _ in voci]
    acqua = [i for i, n in enumerate(nomi) if n.endswith("water")]
    aria = [i for i, n in enumerate(nomi) if n.endswith("air")]
    e_acqua = np.isin(celle, acqua)
    # il mare: gli strati in cui l'acqua e' la gran parte delle celle
    pesanti = [y for y in range(celle.shape[1])
               if e_acqua[:, y, :].sum() > 0.2 * celle.shape[0] * celle.shape[2]]
    if not pesanti:
        raise ValueError(f"{percorso}: nessuna acqua, non e' una nave da mare")
    cima = max(pesanti)
    celle = np.where(e_acqua, -1, celle)
    # sopra la linea di galleggiamento l'aria non scava nulla; sotto, e' lo scafo
    celle[:, cima + 1:, :] = np.where(np.isin(celle[:, cima + 1:, :], aria), -1,
                                      celle[:, cima + 1:, :])
    piene = celle >= 0
    y0 = int(np.nonzero(piene.any(axis=(0, 2)))[0][0])
    celle = TM._ritaglia(celle)
    m = TM._da_voci(percorso, voci, celle, versione)
    m.linea_acqua = cima - y0
    return m


def _traducibile(ver, nome: str, prop: dict) -> bool:
    from amulet.api.block import Block
    from amulet_nbt import StringTag
    b = Block("universal_minecraft", nome, {k: StringTag(v) for k, v in prop.items()})
    try:
        out, _, _ = ver.block.from_universal(b)
    except Exception:                                    # noqa: BLE001
        return False
    return out.namespace != "universal_minecraft"


def _flotta(percorso: str, versione) -> list[TM.Modello]:
    """Un file `.schematic` (vecchio formato, id numerici) con PIU' imbarcazioni
    dentro: ognuna diventa un modello a se'.

    Si legge il volume con il lettore di Amulet (che traduce gli id numerici in
    blocchi universali), si separano le barche per componenti connesse (con un
    giro di margine, cosi' vele e sartie restano attaccate allo scafo), e di
    ognuna si ritaglia la scatola. Il file non ha acqua: la linea di
    galleggiamento si stima dallo scafo (un pescaggio di poche celle, in
    proporzione all'altezza), e l'aria dentro lo scafo sotto quella linea si
    tiene, cosi' non si allaga.
    """
    import amulet
    from scipy.ndimage import binary_dilation, binary_fill_holes, label

    lev = amulet.load_level(percorso)
    try:
        box = lev.bounds("main")
        W, H, L = (int(box.max[i] - box.min[i]) for i in range(3))
        celle = np.full((W, H, L), -1, np.int32)
        voci: list[tuple[str, dict]] = []
        indice: dict = {}
        for cx, cz in lev.all_chunk_coords("main"):
            ch = lev.get_chunk(cx, cz, "main")
            pal = ch.block_palette
            lut = np.zeros(len(pal), np.int32)
            for i in range(len(pal)):
                b = pal[i]
                chiave = (b.base_name, tuple(sorted((k, str(v)) for k, v in b.properties.items())))
                if chiave not in indice:
                    indice[chiave] = len(voci)
                    voci.append((chiave[0], dict(chiave[1])))
                lut[i] = indice[chiave]
            for sy in ch.blocks.sub_chunks:
                sub = np.asarray(ch.blocks.get_sub_chunk(sy))
                x0, y0, z0 = cx * 16, sy * 16, cz * 16
                xs, ys, zs = (min(16, W - x0), min(16, H - y0), min(16, L - z0))
                if xs <= 0 or ys <= 0 or zs <= 0 or x0 < 0 or y0 < 0 or z0 < 0:
                    continue
                celle[x0:x0 + xs, y0:y0 + ys, z0:z0 + zs] = lut[sub[:xs, :ys, :zs]]
    finally:
        lev.close()

    nomi = [n for n, _ in voci]
    aria = [i for i, n in enumerate(nomi) if n == "air"]
    acqua = [i for i, n in enumerate(nomi) if n in ("water", "flowing_water")]
    # solo i blocchi che il traduttore sa portare in gioco: gli altri sono buchi
    # (un blocco universale che il traduttore non porta in gioco resterebbe un
    # nome senza blocco: meglio un buco dichiarato)
    ver = TM.traduttore(versione)
    ok = np.array([_traducibile(ver, n, p) for n, p in voci])
    celle = np.where((celle >= 0) & ~ok[np.clip(celle, 0, None)], -1, celle)
    solido = (celle >= 0) & ~np.isin(celle, aria + acqua)
    lab, quante = label(binary_dilation(solido, structure=np.ones((3, 3, 3))),
                        structure=np.ones((3, 3, 3)))
    fuori: list[TM.Modello] = []
    voce_aria = len(voci)
    tavolozza = voci + [("air", {})]
    for i in range(1, quante + 1):
        regione = (lab == i) & solido
        if int(regione.sum()) < SOLIDI_MINIMI_NAVE:
            continue
        xs, ys, zs = np.nonzero(regione)
        sl = (slice(xs.min(), xs.max() + 1), slice(ys.min(), ys.max() + 1),
              slice(zs.min(), zs.max() + 1))
        c = np.where(regione[sl], celle[sl], -1)
        alt = c.shape[1]
        pescaggio = int(np.clip(round(0.08 * alt) + 1, 2, 5))
        linea = pescaggio - 1                     # ultimo strato sott'acqua
        # l'aria racchiusa dentro lo scafo, sotto la linea d'acqua, resta aria
        for y in range(0, linea + 1):
            piena = c[:, y, :] >= 0
            dentro = binary_fill_holes(piena) & ~piena
            c[:, y, :][dentro] = voce_aria
        m = TM.Modello(nome=f"{os.path.splitext(os.path.basename(percorso))[0]}-{len(fuori) + 1:02d}",
                       celle=np.ascontiguousarray(c), tavolozza=list(tavolozza))
        m.linea_acqua = linea
        fuori.append(m)
    return fuori


def carica(castelli: str, navi: str, versione=(1, 21, 4)) -> tuple[list, list]:
    """(castelli, navi): i modelli, con `_escluso` fra gli stili cosi' che
    nessun lotto e nessuna casa isolata li scelga."""
    fuori_c: list[TM.Modello] = []
    for cartella in _cartelle(castelli):
        for m in TM.carica_cartella(cartella, versione, ESTENSIONI):
            m.stili = frozenset({"_escluso"})
            fuori_c.append(m)
    fuori_n: list[TM.Modello] = []
    for cartella in _cartelle(navi):
        if not os.path.isdir(cartella):
            continue
        for nome in sorted(os.listdir(cartella)):
            if not nome.lower().endswith(ESTENSIONI):
                continue
            percorso = os.path.join(cartella, nome)
            try:
                if nome.lower().endswith(".schematic"):
                    for m in _flotta(percorso, versione):
                        m.stili = frozenset({"_escluso"})
                        fuori_n.append(m)
                    continue
                if nome.lower().endswith(".schem"):
                    m = _nave(percorso, versione)
                else:
                    m = TM.carica(percorso, versione)
                    m.linea_acqua = 0
                m.stili = frozenset({"_escluso"})
                fuori_n.append(m)
            except Exception as guaio:                     # noqa: BLE001
                print(f"  nave saltata: {nome} ({guaio})")
    return fuori_c, fuori_n


# --------------------------------------------------------------------------
# Il castello
# --------------------------------------------------------------------------

def appiana(h: np.ndarray, x0: int, z0: int, ix: int, iz: int, base: int,
            seme: int, fascia: int = FASCIA_CASTELLO) -> None:
    """Il terreno si adatta al rettangolo (x0, z0, ix, iz): piano a `base` dentro,
    poi una scarpata a pendenza costante e irregolare, come per le citta'."""
    H, W = h.shape
    za, zb = max(0, z0 - fascia), min(H, z0 + iz + fascia)
    xa, xb = max(0, x0 - fascia), min(W, x0 + ix + fascia)
    zz, xx = np.mgrid[za:zb, xa:xb]
    dz = np.maximum(np.maximum(z0 - zz, zz - (z0 + iz - 1)), 0)
    dx = np.maximum(np.maximum(x0 - xx, xx - (x0 + ix - 1)), 0)
    d = np.hypot(dz, dx)
    rng = np.random.default_rng(seme)
    f = rng.uniform(0.05, 0.12, 4)
    ph = rng.uniform(0, 2 * np.pi, 4)
    n1 = 0.5 * (np.sin(zz * f[0] + ph[0]) + np.sin(xx * f[1] + ph[1]))
    n2 = 0.5 * (np.sin((zz + xx) * f[2] + ph[2]) + np.sin((zz - xx) * f[3] + ph[3]))
    limite = d * PENDENZA_CASTELLO * (1.0 + 0.45 * n1) + 1.6 * np.abs(n2) * np.minimum(d, 6) / 6
    q = h[za:zb, xa:xb].astype(np.float32)
    nuova = np.clip(q, base - limite, base + limite)
    nuova = np.where(d == 0, base, nuova)
    zona = d <= fascia
    h[za:zb, xa:xb] = np.where(zona, np.round(nuova), q).astype(h.dtype)


def pianifica_castello(modelli: list, primo: int, h: np.ndarray, cls: np.ndarray,
                       evita: np.ndarray, terreni, livello_mare: int,
                       seed: int = 0, tentativi: int = 6000) -> CasaIsolata | None:
    """Il posto del castello: un solo modello, il sedime piu' piano trovato.

    `primo` e' l'indice del primo castello nel catalogo di chi scrive. Si
    sceglie a caso fra i modelli, poi fra le posizioni a rilievo minimo.
    """
    if not modelli:
        return None
    H, W = h.shape
    rng = np.random.default_rng(seed * 7919 + 5)
    k = int(rng.integers(0, len(modelli)))
    m = modelli[k]
    meglio = None
    scarti = {"occupato": 0, "terreno": 0, "rilievo": 0, "quota": 0}
    pianifica_castello.scarti = scarti
    for _ in range(tentativi):
        q = int(rng.integers(0, 4))
        ix, iz = m.ingombro(q)
        marg = MARGINE_CASTELLO
        if ix + 2 * marg >= W or iz + 2 * marg >= H:
            return None
        x0 = int(rng.integers(marg, W - ix - marg))
        z0 = int(rng.integers(marg, H - iz - marg))
        lib = LIBERO_CASTELLO
        if evita[z0 - lib:z0 + iz + lib, x0 - lib:x0 + ix + lib].any():
            scarti["occupato"] += 1
            continue
        if not np.isin(cls[z0 - 3:z0 + iz + 3, x0 - 3:x0 + ix + 3], terreni).all():
            scarti["terreno"] += 1
            continue
        fp = h[z0:z0 + iz, x0:x0 + ix]
        lo, hi = np.percentile(fp, (5, 95))
        rilievo = float(hi - lo)
        base = int(np.median(fp))
        if rilievo > DISLIVELLO_CASTELLO or base <= livello_mare + 2:
            scarti["rilievo" if rilievo > DISLIVELLO_CASTELLO else "quota"] += 1
            continue
        if meglio is None or rilievo < meglio[0]:
            meglio = (rilievo, x0, z0, ix, iz, q, base)
    if meglio is None:
        return None
    _, x0, z0, ix, iz, q, base = meglio
    appiana(h, x0, z0, ix, iz, base, seme=seed * 31 + base)
    return CasaIsolata(x=x0, z=z0, larghezza=ix, profondita=iz, base=base,
                       modello=primo + k, quarti=q, stile="prato")


# --------------------------------------------------------------------------
# Le navi
# --------------------------------------------------------------------------

def _impronta_scafo(m: TM.Modello, quarti: int) -> np.ndarray:
    """Dove sta lo scafo (gli strati fino alla linea di galleggiamento) visto
    dall'alto, girato di `quarti`: (x, z) booleano."""
    c = m.celle[:, :m.linea_acqua + 2, :] >= 0
    imp = c.any(axis=1)
    for _ in range(quarti % 4):
        imp = np.rot90(imp, k=1, axes=(0, 1))     # come `Catalogo.celle`
    return imp


def pianifica_navi(modelli: list, primo: int, h: np.ndarray, cls: np.ndarray,
                   mare, livello_mare: int, densita: float = 1.0,
                   seed: int = 0, tentativi: int = 300) -> list[CasaIsolata]:
    """Qualche nave in mare aperto: scafo su fondale profondo e solo acqua
    attorno. `mare` sono le classi di mare vero (non fiumi e laghi)."""
    if not modelli or densita <= 0:
        return []
    H, W = h.shape
    rng = np.random.default_rng(seed * 6007 + 11)
    quante = max(1, int(round(H * W / CELLE_PER_NAVE * densita)))
    e_mare = np.isin(cls, mare)
    fuori: list[CasaIsolata] = []
    for n in range(quante):
        for _ in range(tentativi):
            # un modello a caso a ogni prova: un pacchetto ha navi di ogni taglia
            # e una grande puo' non trovare posto dove una piccola ci sta
            k = int(rng.integers(0, len(modelli)))
            m = modelli[k]
            q = int(rng.integers(0, 4))
            ix, iz = m.ingombro(q)
            if ix >= W or iz >= H:
                break
            x0 = int(rng.integers(0, W - ix))
            z0 = int(rng.integers(0, H - iz))
            imp = _impronta_scafo(m, q)
            xs, zs = np.nonzero(imp)                   # imp e' (x, z)
            if len(zs) == 0:
                break
            # lo scafo e il suo margine devono stare tutti su mare vero
            za, zb = max(0, z0 + zs.min() - MARGINE_NAVE), min(H, z0 + zs.max() + 1 + MARGINE_NAVE)
            xa, xb = max(0, x0 + xs.min() - MARGINE_NAVE), min(W, x0 + xs.max() + 1 + MARGINE_NAVE)
            if not e_mare[za:zb, xa:xb].all():
                continue
            base = livello_mare - m.linea_acqua
            if (h[za:zb, xa:xb] > base - 1).any():
                continue                                   # lo scafo toccherebbe il fondo
            cx, cz = x0 + ix // 2, z0 + iz // 2
            if any((cx - c.x - c.larghezza // 2) ** 2 + (cz - c.z - c.profondita // 2) ** 2
                   < DISTANZA_NAVI ** 2 for c in fuori):
                continue
            fuori.append(CasaIsolata(x=x0, z=z0, larghezza=ix, profondita=iz,
                                     base=base, modello=primo + k, quarti=q,
                                     stile="prato"))
            break
    return fuori
