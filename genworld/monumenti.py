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

import json
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


def _volume(percorso: str, regione: tuple | None = None):
    """Legge un volume con Amulet (schema o mondo): (celle (x, y, z) di indici
    nella palette, palette di (nome universale, proprieta')).

    `regione` = (x0, y0, z0, x1, y1, z1), estremo alto escluso, in coordinate del
    livello; senza, l'intero schema. I chunk che non esistono restano non
    specificati (-1).
    """
    import amulet
    from amulet.api.errors import ChunkDoesNotExist

    lev = amulet.load_level(percorso)
    try:
        dim = lev.dimensions[0]
        if regione is None:
            box = lev.bounds(dim)
            x0, y0, z0 = (int(v) for v in box.min)
            x1, y1, z1 = (int(v) for v in box.max)
        else:
            x0, y0, z0, x1, y1, z1 = regione
        W, H, L = x1 - x0, y1 - y0, z1 - z0
        celle = np.full((W, H, L), -1, np.int32)
        voci: list[tuple[str, dict]] = []
        indice: dict = {}
        for cx in range(x0 // 16, (x1 - 1) // 16 + 1):
            for cz in range(z0 // 16, (z1 - 1) // 16 + 1):
                try:
                    ch = lev.get_chunk(cx, cz, dim)
                except ChunkDoesNotExist:
                    continue
                pal = ch.block_palette
                lut = np.zeros(len(pal), np.int32)
                for i in range(len(pal)):
                    b = pal[i]
                    chiave = (b.base_name, tuple(sorted((k, str(v)) for k, v in b.properties.items())))
                    if chiave not in indice:
                        indice[chiave] = len(voci)
                        voci.append((chiave[0], dict(chiave[1])))
                    lut[i] = indice[chiave]
                gx0, gx1 = max(x0, cx * 16), min(x1, cx * 16 + 16)
                gz0, gz1 = max(z0, cz * 16), min(z1, cz * 16 + 16)
                if gx0 >= gx1 or gz0 >= gz1:
                    continue
                for sy in ch.blocks.sub_chunks:
                    gy0, gy1 = max(y0, sy * 16), min(y1, sy * 16 + 16)
                    if gy0 >= gy1:
                        continue
                    sub = np.asarray(ch.blocks.get_sub_chunk(sy))
                    celle[gx0 - x0:gx1 - x0, gy0 - y0:gy1 - y0, gz0 - z0:gz1 - z0] = lut[
                        sub[gx0 - cx * 16:gx1 - cx * 16, gy0 - sy * 16:gy1 - sy * 16,
                            gz0 - cz * 16:gz1 - cz * 16]]
    finally:
        lev.close()
    return celle, voci


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
    from scipy.ndimage import binary_dilation, binary_fill_holes, label

    celle, voci = _volume(percorso)

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


def _meta(cartella: str) -> dict:
    """`meta.json` accanto agli schemi: {nome del file senza estensione: {...}}."""
    percorso = os.path.join(cartella, "meta.json")
    if not os.path.isfile(percorso):
        return {}
    try:
        with open(percorso, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _bbox(piena: np.ndarray):
    xs, ys, zs = np.nonzero(piena)
    return (slice(xs.min(), xs.max() + 1), slice(ys.min(), ys.max() + 1),
            slice(zs.min(), zs.max() + 1))


def _molo(percorso: str, meta: dict, versione) -> TM.Modello:
    """Un molo da schema (`.schematic` vecchio): un tratto di riva, la banchina di
    tavole su pali e una casetta sopra. Parametri di posa da `meta`:
    `terra` (il verso, nel modello, in cui si va verso terra: "+x"), `costa_x`
    e `centro_z` (dove, nel modello, finisce la terra e dov'e' il mezzo del
    fronte) e `linea_acqua` (l'ultimo strato sott'acqua).
    """
    celle, voci = _volume(percorso)
    ver = TM.traduttore(versione)
    ok = np.array([_traducibile(ver, n, p) for n, p in voci] + [False])
    aria = np.array([n == "air" for n, _ in voci] + [True])
    celle = np.where((celle >= 0) & ok[celle] & ~aria[celle], celle, -1)
    sl = _bbox(celle >= 0)
    celle = np.ascontiguousarray(celle[sl])
    meta = dict(meta)
    meta["costa_x"] = int(meta.get("costa_x", 0)) - sl[0].start
    meta["centro_z"] = int(meta.get("centro_z", celle.shape[2] // 2)) - sl[2].start
    meta["linea_acqua"] = int(meta.get("linea_acqua", 1)) - sl[1].start
    m = TM.Modello(nome=os.path.splitext(os.path.basename(percorso))[0],
                   celle=celle, tavolozza=list(voci))
    m.linea_acqua = meta["linea_acqua"]
    m.meta = meta
    return m


def _faro(percorso: str, meta: dict, versione) -> TM.Modello:
    """Un faro preso da un MONDO (una cartella con `level.dat`): si legge la
    regione data in `meta["regione"]` (x0, y0, z0, x1, y1, z1), `meta["terreno_y"]`
    e' la quota dell'ultimo strato di terra del mondo piatto. Restano le
    costruzioni e i due strati di terra sotto la loro impronta (cosi' il modello si
    interra come una casa: vedi `Modello.affondo`); l'aria racchiusa dentro resta
    aria, il resto dello scavo no.
    """
    from scipy.ndimage import binary_fill_holes

    regione = tuple(meta["regione"])
    celle, voci = _volume(percorso, regione)
    ver = TM.traduttore(versione)
    ok = np.array([_traducibile(ver, n, p) for n, p in voci] + [False])
    e_aria = np.array([n == "air" for n, _ in voci] + [True])
    celle = np.where((celle >= 0) & ok[celle], celle, -1)
    gy = int(meta["terreno_y"]) - regione[1]            # indice dell'ultimo strato di terra
    solido = (celle >= 0) & ~e_aria[celle]
    sopra = solido[:, gy + 1:, :]
    impronta = sopra.any(axis=1)
    if not impronta.any():
        raise ValueError(f"{percorso}: niente costruzione sopra la quota {meta['terreno_y']}")
    voce_aria = len(voci)
    tavolozza = list(voci) + [("air", {})]
    fuori = np.full(celle.shape, -1, np.int32)
    # costruzione: tutto cio' che non e' aria
    fuori[:, gy + 1:, :] = np.where(solido[:, gy + 1:, :], celle[:, gy + 1:, :], -1)
    # aria racchiusa (stanze, scale): per strato, i buchi dell'impronta piena
    for y in range(gy + 1, celle.shape[1]):
        piena = fuori[:, y, :] >= 0
        if piena.any():
            dentro = binary_fill_holes(piena) & ~piena
            fuori[:, y, :][dentro] = voce_aria
    # i due strati di terra, solo sotto l'impronta
    for y in (gy - 1, gy):
        if 0 <= y < celle.shape[1]:
            terra = (celle[:, y, :] >= 0) & ~e_aria[celle[:, y, :]] & impronta
            fuori[:, y, :] = np.where(terra, celle[:, y, :], -1)
    fuori = _piede_a_gradoni(fuori, gy, tavolozza, int(meta.get("piede", PIEDE_FARO)),
                             int(meta.get("altura", ALTURA_FARO)))
    sl = _bbox(fuori >= 0)
    fuori = np.ascontiguousarray(fuori[sl][:, max(gy - 1, 0) - sl[1].start:, :]
                                 if sl[1].start <= max(gy - 1, 0) else fuori[sl])
    m = TM.Modello(nome=str(meta.get("nome", os.path.splitext(os.path.basename(percorso))[0])),
                   celle=fuori, tavolozza=tavolozza)
    m.meta = dict(meta)
    return m


PIEDE_FARO = 6              # larghezza del piede a gradoni attorno all'altura del faro
ALTURA_FARO = 7             # strati sopra il terreno che formano l'altura di pietra


def _piede_a_gradoni(celle: np.ndarray, gy: int, tavolozza: list, piede: int,
                     altura: int) -> np.ndarray:
    """Un raccordo di pietra e terra attorno all'altura di un faro.

    L'altura del modello e' un parallelepipedo con le pareti a picco: sul terreno
    vero sembra un blocco appoggiato. Qui le si aggiunge un piede a gradoni: a
    ogni strato (dal basso in alto) la sagoma si restringe, con il bordo mangiato
    a caso - una scarpata franata, non una piramide - di pietra in basso, terra
    poi, erba all'ultimo strato. Il piede si estende anche sotto, sugli strati di
    terra. Il modello cresce di `piede` celle per lato.
    """
    from scipy.ndimage import distance_transform_edt

    nomi = [n for n, _ in tavolozza]

    def voce(nome):
        return nomi.index(nome) if nome in nomi else None

    pietra, terra, erba = voce("stone"), voce("dirt"), voce("grass_block")
    if pietra is None or piede <= 0:
        return celle
    dx, dy, dz = celle.shape
    m = piede
    ampio = np.full((dx + 2 * m, dy, dz + 2 * m), -1, np.int32)
    ampio[m:m + dx, :, m:m + dz] = celle
    base = ampio[:, gy + 1, :] >= 0                       # la sagoma dell'altura
    if not base.any():
        return celle
    dist = distance_transform_edt(~base)
    zz, xx = np.mgrid[:ampio.shape[0], :ampio.shape[2]]
    rumore = ((xx * 73856093) ^ (zz * 19349663)) % 1000 / 1000.0
    for k in range(min(altura, dy - gy - 1)):
        y = gy + 1 + k
        raggio = piede - (k * piede) / max(altura - 1, 1)     # si restringe salendo
        entro = dist <= raggio * (0.55 + 0.45 * rumore)       # il bordo mangiato
        vuote = entro & (ampio[:, y, :] < 0)
        if k >= altura - 2 and erba is not None:
            blocco = erba
        elif k >= altura // 2 and terra is not None:
            blocco = terra
        else:
            blocco = pietra
        ampio[:, y, :][vuote] = blocco
    for y in (gy - 1, gy):                                    # la terra sotto
        if 0 <= y < dy and terra is not None:
            vuote = (dist <= piede * (0.55 + 0.45 * rumore)) & (ampio[:, y, :] < 0)
            ampio[:, y, :][vuote] = terra
    return ampio


def carica_porti(moli: str, fari: str, versione=(1, 21, 4)) -> tuple[list, list]:
    """(moli, fari) da schema: i moli da `.schematic`/`.schem`/`.nbt`, i fari da
    schemi o da mondi (cartelle con `level.dat`). Un file rotto non ferma gli altri."""
    fuori_m: list[TM.Modello] = []
    for cartella in _cartelle(moli):
        if not os.path.isdir(cartella):
            continue
        meta = _meta(cartella)
        for nome in sorted(os.listdir(cartella)):
            base, est = os.path.splitext(nome)
            if est.lower() not in (".schematic", ".schem", ".nbt", ".litematic"):
                continue
            try:
                m = _molo(os.path.join(cartella, nome), meta.get(base, {}), versione)
                m.stili = frozenset({"_escluso"})
                fuori_m.append(m)
            except Exception as guaio:                    # noqa: BLE001
                print(f"  molo saltato: {nome} ({guaio})")
    fuori_f: list[TM.Modello] = []
    for cartella in _cartelle(fari):
        if not os.path.isdir(cartella):
            continue
        meta = _meta(cartella)
        for nome in sorted(os.listdir(cartella)):
            percorso = os.path.join(cartella, nome)
            try:
                if os.path.isdir(percorso) and os.path.isfile(os.path.join(percorso, "level.dat")):
                    if nome not in meta:
                        print(f"  faro saltato: {nome} (manca la regione in meta.json)")
                        continue
                    m = _faro(percorso, meta[nome], versione)
                elif nome.lower().endswith((".schem", ".schematic", ".nbt", ".litematic")):
                    m = TM.carica(percorso, versione)
                else:
                    continue
                m.stili = frozenset({"_escluso"})
                fuori_f.append(m)
            except Exception as guaio:                    # noqa: BLE001
                print(f"  faro saltato: {nome} ({guaio})")
    return fuori_m, fuori_f


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
            seme: int, fascia: int = FASCIA_CASTELLO,
            escludi: np.ndarray | None = None) -> None:
    """Il terreno si adatta al rettangolo (x0, z0, ix, iz): piano a `base` dentro,
    poi una scarpata a pendenza costante e irregolare, come per le citta'.

    Va usata per OGNI struttura posata fuori dai paesi (case isolate, faro,
    castello, cimiteri, portali): senza, la costruzione sta su una zolla
    rettangolare con la parete a picco sul terreno piu' basso intorno - troppo
    squadrata per essere vera. `escludi` (maschera) marca le celle da non toccare,
    per esempio il mare."""
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
    if escludi is not None:
        zona = zona & ~escludi[za:zb, xa:xb]
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
