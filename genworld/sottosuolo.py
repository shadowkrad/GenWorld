"""Il sottosuolo: caverne, minerali, ardesia.

Censito sul mondo generato prima di questo modulo: il sottosuolo era **62%
pietra e niente altro**. Niente caverne, niente minerali, niente ardesia.
Si scavava e si trovava pietra fino alla bedrock, per sempre. Per un mondo da
guardare dall'alto non cambiava niente; per un mondo da giocare era il difetto
piu' grave rimasto, perche' toglieva meta' del gioco.

Tre cose, con tre meccaniche diverse:

* **le caverne** sono cunicoli, non bolle. Si scavano con dei "vermi": un
  cammino casuale in tre dimensioni con inerzia, raggio variabile, che si
  indicizza per chunk come gli alberi e gli edifici. Un rumore 3D darebbe
  caverne a groviera, tonde e tutte uguali; un cunicolo si percorre.
* **i minerali** sono grumi piccoli a fasce di quota, e si generano per
  chunk in modo deterministico - compresi quelli dei chunk vicini, che
  sporgono qui. E' la stessa regola degli alberi: filtrare per centro invece
  che per ingombro taglia a meta' tutto quello che sta sul confine.
* **l'ardesia** e' una fascia, non una riga: sotto y=0 la pietra diventa
  deepslate, con una zona di mescolanza di una decina di blocchi, altrimenti
  si vede il piano di taglio.

La regola che tiene insieme tutto: **non si scava mai sopra `terreno - 5`**.
Cinque blocchi di cappello garantiscono che una caverna non sfondi il prato e,
soprattutto, che non apra un buco sotto il mare - che allagherebbe tutto senza
che nessuno se ne accorga finche' non ci nuota dentro.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Quota sotto la quale la pietra diventa ardesia, e spessore della mescolanza.
QUOTA_ARDESIA = 0
FASCIA_ARDESIA = 10

# Cappello minimo fra la cima di una caverna e la superficie.
CAPPELLO = 5


@dataclass(frozen=True)
class Minerale:
    nome: str            # nome universale in pietra
    nome_ardesia: str    # nome universale sotto la fascia
    y_min: int
    y_max: int
    per_chunk: float     # grumi attesi per chunk
    grumo: tuple         # (min, max) blocchi per grumo
    proprieta: dict = None


# Distribuzione ispirata alla 1.18: il carbone sta in alto, il diamante in
# fondo, il lapis a meta'. Le quantita' sono un po' piu' generose del gioco,
# perche' un mondo generato si esplora una volta sola.
MINERALI = (
    Minerale("coal_ore", "deepslate_coal_ore", 5, 120, 3.2, (6, 18)),
    Minerale("copper_ore", "deepslate_copper_ore", -16, 96, 1.6, (5, 14)),
    Minerale("iron_ore", "deepslate_iron_ore", -32, 64, 2.4, (4, 12)),
    Minerale("gold_ore", "deepslate_gold_ore", -60, 30, 0.8, (3, 8)),
    Minerale("redstone_ore", "deepslate_redstone_ore", -60, 14, 1.0, (4, 10),
             {"lit": "false"}),
    Minerale("lapis_ore", "deepslate_lapis_ore", -48, 40, 0.5, (3, 7)),
    Minerale("diamond_ore", "deepslate_diamond_ore", -60, 14, 0.35, (2, 6)),
    Minerale("emerald_ore", "deepslate_emerald_ore", 90, 200, 0.5, (1, 3)),
)

# Grumi di roccia diversa: non sono minerali, ma senza di loro il sottosuolo
# e' un blocco solo dall'inizio alla fine.
ROCCE = (
    ("granite", 1.6, (20, 60), {"polished": "false"}),
    ("diorite", 1.6, (20, 60), {"polished": "false"}),
    ("andesite", 1.6, (20, 60), {"polished": "false"}),
    ("tuff", 0.8, (15, 40), {}),
    ("dirt", 1.2, (15, 40), {}),
    ("gravel", 1.0, (15, 40), {}),
)


# --------------------------------------------------------------------------
# Caverne
# --------------------------------------------------------------------------

def scava_caverne(
    altezze: np.ndarray,
    livello_mare: int = 62,
    densita: float = 1.0,
    seed: int = 0,
) -> dict:
    """Pianifica i cunicoli. Ritorna un dizionario di sfere (x, y, z, r).

    Un verme parte da un punto sottoterra e cammina: ogni passo gira un po'
    rispetto al precedente invece di scegliere una direzione nuova, ed e'
    l'inerzia che trasforma un cammino casuale in un cunicolo percorribile.
    Senza, si ottiene un gomitolo che non porta da nessuna parte.
    """
    H, W = altezze.shape
    rng = np.random.default_rng(seed)
    quanti = int(max(1, H * W / 4200 * densita))

    xs, ys, zs, rs = [], [], [], []
    for _ in range(quanti):
        z = int(rng.integers(8, H - 8))
        x = int(rng.integers(8, W - 8))
        tetto = int(altezze[z, x]) - CAPPELLO
        if tetto < -50:
            continue
        y = float(rng.integers(-55, max(-54, min(tetto, 50))))
        passi = int(rng.integers(140, 520))
        # direzione iniziale
        ang = float(rng.random() * 2 * np.pi)
        sal = float(rng.normal(0, 0.25))
        raggio = float(rng.uniform(1.8, 3.6))

        for _ in range(passi):
            ang += float(rng.normal(0, 0.22))
            sal = sal * 0.88 + float(rng.normal(0, 0.09))
            sal = float(np.clip(sal, -0.7, 0.7))
            x += np.cos(ang)
            z += np.sin(ang)
            y += sal
            if not (4 <= x < W - 4 and 4 <= z < H - 4):
                break
            tetto = int(altezze[int(z), int(x)]) - CAPPELLO
            if y > tetto:
                y = float(tetto)
                sal = -abs(sal)
            if y < -58:
                y = -58.0
                sal = abs(sal)
            # il raggio respira: un cunicolo di sezione costante e' un tubo
            raggio = float(np.clip(raggio + rng.normal(0, 0.12), 1.4, 4.4))
            xs.append(int(x)); ys.append(int(y)); zs.append(int(z))
            rs.append(raggio)

    return {"x": np.array(xs, np.int32), "y": np.array(ys, np.int32),
            "z": np.array(zs, np.int32), "r": np.array(rs, np.float32)}


def indice_per_chunk(caverne: dict, passo: int = 16) -> dict:
    """Ogni sfera va registrata in tutti i chunk che tocca."""
    fuori: dict[tuple[int, int], list[int]] = {}
    if caverne["x"].size == 0:
        return fuori
    for i in range(caverne["x"].size):
        x, z, r = int(caverne["x"][i]), int(caverne["z"][i]), caverne["r"][i]
        m = int(np.ceil(r)) + 1
        for cz in range((z - m) // passo, (z + m) // passo + 1):
            for cx in range((x - m) // passo, (x + m) // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def statistiche(caverne: dict) -> dict:
    if caverne["x"].size == 0:
        return {"sfere": 0}
    return {
        "sfere": int(caverne["x"].size),
        "raggio_medio": float(caverne["r"].mean()),
        "quota_min": int(caverne["y"].min()),
        "quota_max": int(caverne["y"].max()),
    }


# --------------------------------------------------------------------------
# Posa dentro il chunk
# --------------------------------------------------------------------------

class Tavolozza:
    """Id dei blocchi del sottosuolo, risolti sul livello aperto."""

    def __init__(self, scrittore):
        s = scrittore
        self.aria = s.id_aria
        self.pietra = s.blocco("stone")
        self.ardesia = s.blocco("deepslate", axis="y")
        self.minerale = {}
        for m in MINERALI:
            p = m.proprieta or {}
            self.minerale[(m.nome, False)] = s.blocco(m.nome, **p)
            self.minerale[(m.nome, True)] = s.blocco(m.nome_ardesia, **p)
        self.roccia = {n: s.blocco(n, **p) for n, _, _, p in ROCCE}
        # Tutto cio' che vale "pietra" per chi scava e per chi mette i filoni.
        self.scavabile = (self.pietra, self.ardesia)
        self.deco = _Decorazioni(s)

    def aggiungi_rocce(self, ids) -> None:
        """Gli strati di `stratigrafia` sono roccia a tutti gli effetti.

        Senza questo, un filone di ferro che capita dentro un banco di
        andesite non si posa - `solo_pietra` non lo riconosce - e le vene si
        interrompono esattamente dove passa uno strato. Che e' il contrario
        di quello che fa una vena vera.
        """
        self.scavabile = tuple(dict.fromkeys(self.scavabile + tuple(ids)))


class _Decorazioni:
    """Blocchi di caverne e geodi: gocciolatoi, grotta lussureggiante, ametista,
    sculk. Stanno in un attributo a parte perche' i test della posa usano una
    tavolozza finta senza: senza `deco` la decorazione semplicemente non c'e'."""

    def __init__(self, s):
        self.dripstone = s.blocco("dripstone_block")
        self.punta_giu = s.blocco("pointed_dripstone", thickness="tip",
                                  vertical_direction="down")
        self.punta_su = s.blocco("pointed_dripstone", thickness="tip",
                                 vertical_direction="up")
        self.muschio = s.blocco("moss_block")
        self.tappeto = s.blocco("moss_carpet")
        self.radici = s.blocco("hanging_roots")
        self.liana = s.blocco("cave_vines", age="5", berries="true")
        self.liana_nuda = s.blocco("cave_vines", age="5", berries="false")
        self.foglia = s.blocco("big_dripleaf", facing="north", tilt="none")
        self.azalea = s.blocco("azalea")
        self.azalea_fiorita = s.blocco("flowering_azalea")
        self.sculk = s.blocco("sculk")
        self.sensore = s.blocco("sculk_sensor", power="0",
                                sculk_sensor_phase="inactive")
        self.ametista = s.blocco("amethyst_block")
        self.ametista_viva = s.blocco("budding_amethyst")
        self.calcite = s.blocco("calcite")
        self.basalto = s.blocco("smooth_basalt")
        self.cristallo_su = s.blocco("amethyst_cluster", facing="up")
        self.cristallo_giu = s.blocco("amethyst_cluster", facing="down")
        self.acqua = s.blocco("water")


def grumo(out, gx, gz, gy, y0, h_c, blocco, quanti, solo_pietra,
         protetto=None) -> None:
    """Un grumo di blocchi centrato su (gx, gz, gy), dentro la pietra.

    `gx`/`gz` sono coordinate LOCALI di chunk e possono uscire da `0..16`:
    i blocchi che cadono fuori si scartano, ed e' cosi' che un grumo nato a
    un passo dal confine (o al centro di una caverna in un chunk vicino)
    sporge qui dentro invece di essere tagliato di netto. `gy` e' la quota
    assoluta, `y0` l'origine verticale di `out`.

    `solo_pietra` e' la garanzia che non si buchino case, caverne o terreno
    di superficie: si sostituisce soltanto quello che e' pietra o ardesia.

    `protetto` (16x16, opzionale) e' la stessa maschera colonnare usata
    altrove (`protetto_c`): quando data, un grumo non scrive nelle colonne
    protette - serve a `miniere._carica_giacimento`, dove tutta la colonna
    (scavo compreso) deve restare intatta sotto una struttura gia' disegnata
    in superficie, la stessa regola che `_carica_segmento` applica gia' al
    filone lungo il tunnel.

    Pubblica apposta: la usano anche `posa()` per i filoni lungo le caverne
    e `miniere.posa()` per i giacimenti lungo i condotti artificiali - un
    grumo e' un grumo, che nasca da solo o accanto a un cunicolo.
    """
    H = out.shape[1]
    raggio = max(1.0, (quanti / 4.2) ** (1 / 3) + 0.4)
    r = int(np.ceil(raggio))
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx * dx + dy * dy + dz * dz > raggio * raggio:
                    continue
                lx, lz = gx + dx, gz + dz
                if not (0 <= lx < 16 and 0 <= lz < 16):
                    continue
                if protetto is not None and protetto[lx, lz]:
                    continue
                ly = gy + dy - y0
                if not (1 <= ly < H):
                    continue
                if gy + dy > int(h_c[lx, lz]) - 2:
                    continue
                if out[lx, ly, lz] not in solo_pietra:
                    continue
                out[lx, ly, lz] = blocco


def _grumo(out, rng, off_x, off_z, y0, h_c, blocco, quanti, y_min, y_max,
           solo_pietra):
    """Sceglie un centro a caso dentro (o a cavallo di) il chunk e chiama
    `grumo()` - il campo ambientale di minerali e rocce, non legato a
    nessuna caverna."""
    gx = int(rng.integers(0, 16)) + off_x
    gz = int(rng.integers(0, 16)) + off_z
    gy = int(rng.integers(y_min, y_max + 1))
    grumo(out, gx, gz, gy, y0, h_c, blocco, quanti, solo_pietra)


def posa(out: np.ndarray, tav: Tavolozza, h_c: np.ndarray, ox: int, oz: int,
         y0: int, caverne: dict, quali, seed: int = 0,
         protetto_c: np.ndarray | None = None) -> None:
    """Ardesia, minerali e caverne dentro l'array di chunk (16, H, 16).

    `protetto_c` (16x16, opzionale) segna le colonne dove sopra c'e' gia'
    una struttura disegnata - mura, strade, campi, case (vedi `protetto` in
    `motore.pianifica()`). Le caverne si pianificano su tutta la mappa senza
    sapere dove sorgera' un insediamento, e si scavano DOPO che quella
    struttura e' gia' stata scritta nel chunk: senza questa maschera un
    cunicolo che passa sotto una cinta muraria toglie la roccia di
    fondazione e lascia il muro appeso sul vuoto.
    """
    H = out.shape[1]
    ys = np.arange(y0, y0 + H, dtype=np.int32)[None, :, None]

    # --- ardesia ---------------------------------------------------------
    # Fascia, non riga: la probabilita' di ardesia sale da 0 a 1 lungo dieci
    # blocchi, altrimenti si vede il piano di taglio a y=0.
    pietra = out == tav.pietra
    if pietra.any():
        t = np.clip((QUOTA_ARDESIA - ys) / FASCIA_ARDESIA, 0.0, 1.0)
        rng = np.random.default_rng(_seme(ox, oz, seed))
        dado = rng.random((16, H, 16))
        out[pietra & (dado < t)] = tav.ardesia

    solo_pietra = tav.scavabile

    # --- rocce e minerali ------------------------------------------------
    # Si generano anche i grumi dei chunk vicini: uno che nasce a un blocco
    # dal confine sporge qui, e filtrando per centro si otterrebbero vene
    # tagliate di netto lungo ogni linea di chunk.
    for dcx in (-1, 0, 1):
        for dcz in (-1, 0, 1):
            vx, vz = ox + dcx * 16, oz + dcz * 16
            rng = np.random.default_rng(_seme(vx, vz, seed))
            for nome, quanti_attesi, (q0, q1), _ in ROCCE:
                for _ in range(_quanti(rng, quanti_attesi)):
                    _grumo(out, rng, dcx * 16, dcz * 16, y0, h_c,
                           tav.roccia[nome], int(rng.integers(q0, q1 + 1)),
                           -58, 70, solo_pietra)
            for m in MINERALI:
                for _ in range(_quanti(rng, m.per_chunk)):
                    y = int(rng.integers(m.y_min, m.y_max + 1))
                    profondo = y < QUOTA_ARDESIA - FASCIA_ARDESIA // 2
                    _grumo(out, rng, dcx * 16, dcz * 16, y0, h_c,
                           tav.minerale[(m.nome, profondo)],
                           int(rng.integers(m.grumo[0], m.grumo[1] + 1)),
                           y, y, solo_pietra)

    # --- geodi di ametista ------------------------------------------------
    deco = getattr(tav, "deco", None)
    if deco is not None:
        for dcx in (-1, 0, 1):
            for dcz in (-1, 0, 1):
                vx, vz = ox + dcx * 16, oz + dcz * 16
                rng = np.random.default_rng(_seme(vx, vz, seed + 77))
                if rng.random() < GEODI_PER_CHUNK:
                    geode(out, deco, tav.aria, int(rng.integers(0, 16)) + dcx * 16,
                          int(rng.integers(0, 16)) + dcz * 16,
                          int(rng.integers(-50, -8)), y0, h_c,
                          float(rng.uniform(3.6, 5.4)), solo_pietra, protetto_c)

    # --- caverne ---------------------------------------------------------
    scavato = np.zeros(out.shape, bool)
    for i in quali:
        cx, cy, cz = (int(caverne["x"][i]), int(caverne["y"][i]),
                      int(caverne["z"][i]))
        r = float(caverne["r"][i])
        ir = int(np.ceil(r))
        for dx in range(-ir, ir + 1):
            lx = cx + dx - ox
            if not (0 <= lx < 16):
                continue
            for dz in range(-ir, ir + 1):
                lz = cz + dz - oz
                if not (0 <= lz < 16):
                    continue
                if protetto_c is not None and protetto_c[lx, lz]:
                    # qui sopra c'e' gia' una struttura: niente scavo in
                    # questa colonna, a nessuna quota.
                    continue
                tetto = int(h_c[lx, lz]) - CAPPELLO
                for dy in range(-ir, ir + 1):
                    if dx * dx + dy * dy + dz * dz > r * r:
                        continue
                    gy = cy + dy
                    # `>=`, non `>`: con il maggiore stretto il cappello
                    # restava di quattro blocchi invece che di cinque, e sul
                    # fondale quattro blocchi di sabbia sopra una caverna sono
                    # un allagamento che aspetta il primo giocatore.
                    if gy >= tetto or gy <= y0:
                        continue
                    ly = gy - y0
                    if 1 <= ly < H:
                        out[lx, ly, lz] = tav.aria
                        scavato[lx, ly, lz] = True

        # --- minerali lungo la caverna ------------------------------------
        # Senza questo, il campo ambientale di minerali (sopra) e i cunicoli
        # sono due sistemi che non si parlano: capita un lungo cunicolo
        # vuoto e un filone sepolto dall'altra parte della mappa. "caverne
        # dove ci sono minerali da scavare" vuol dire che scavare nella
        # parete di un cunicolo deve avere senso piu' spesso che altrove -
        # non sempre, altrimenti ogni caverna diventa una miniera a cielo
        # aperto e il filone smette di essere un ritrovamento.
        rng_f = np.random.default_rng(_seme(cx, cz, seed) ^ (cy & 0xFFFF))
        if rng_f.random() < 0.06:
            candidati = [m for m in MINERALI if m.y_min <= cy <= m.y_max]
            if candidati:
                pesi = np.array([m.per_chunk for m in candidati], np.float64)
                scelto = candidati[int(rng_f.choice(len(candidati),
                                                     p=pesi / pesi.sum()))]
                ang = float(rng_f.random() * 2 * np.pi)
                dist = r + float(rng_f.uniform(1.5, 3.5))
                gx = int(round(cx - ox + np.cos(ang) * dist))
                gz = int(round(cz - oz + np.sin(ang) * dist))
                gy = int(round(cy + rng_f.normal(0, 1.0)))
                profondo = gy < QUOTA_ARDESIA - FASCIA_ARDESIA // 2
                quanti = int(rng_f.integers(scelto.grumo[0], scelto.grumo[1] + 1))
                grumo(out, gx, gz, gy, y0, h_c,
                     tav.minerale[(scelto.nome, profondo)], quanti, solo_pietra)


    if deco is not None and scavato.any():
        decora_caverne(out, tav, scavato, ox, oz, y0, seed)


GEODI_PER_CHUNK = 0.10


def geode(out, deco, aria, gx, gz, gy, y0, h_c, r, solo_pietra,
          protetto=None) -> None:
    """Un geode: guscio di basalto liscio, calcite, ametista, e dentro un
    vuoto con ametista viva e i cristalli che ne crescono. Come `grumo`,
    gx/gz sono locali al chunk e possono uscire da 0..16."""
    H = out.shape[1]
    ir = int(np.ceil(r)) + 1
    dentro = []
    for dx in range(-ir, ir + 1):
        for dz in range(-ir, ir + 1):
            lx, lz = gx + dx, gz + dz
            if not (0 <= lx < 16 and 0 <= lz < 16):
                continue
            if protetto is not None and protetto[lx, lz]:
                continue
            for dy in range(-ir, ir + 1):
                ly = gy + dy - y0
                if not (1 <= ly < H) or gy + dy > int(h_c[lx, lz]) - 8:
                    continue
                d = (dx * dx + dy * dy * 1.15 + dz * dz) ** 0.5
                if d > r + 1.0 or out[lx, ly, lz] not in solo_pietra:
                    continue
                if d > r:
                    out[lx, ly, lz] = deco.basalto
                elif d > r - 1.0:
                    out[lx, ly, lz] = deco.calcite
                elif d > r - 2.2:
                    viva = (lx * 7 + ly * 13 + lz * 5) % 6 == 0
                    out[lx, ly, lz] = deco.ametista_viva if viva else deco.ametista
                else:
                    out[lx, ly, lz] = aria
                    dentro.append((lx, ly, lz))
    for lx, ly, lz in dentro:
        if out[lx, ly - 1, lz] == deco.ametista_viva and (lx + lz + ly) % 3 == 0:
            out[lx, ly, lz] = deco.cristallo_su
        elif (ly + 1 < H and out[lx, ly + 1, lz] == deco.ametista_viva
              and (lx * 3 + lz) % 4 == 0):
            out[lx, ly, lz] = deco.cristallo_giu


def decora_caverne(out, tav, scavato, ox, oz, y0, seed) -> int:
    """Gocciolatoi, grotta lussureggiante e sculk SOLO nelle caverne naturali
    (`scavato`): le gallerie delle miniere sono aria anche loro, ma non devono
    riempirsi di muschio. Ritorna quante celle ha cambiato."""
    d = tav.deco
    H = out.shape[1]
    aria = tav.aria
    solido = np.isin(out, tav.scavabile)
    libera = scavato & (out == aria)
    pavimento = np.zeros_like(libera)     # cella libera con roccia sotto
    soffitto = np.zeros_like(libera)      # cella libera con roccia sopra
    pavimento[:, 1:, :] = libera[:, 1:, :] & solido[:, :-1, :]
    soffitto[:, :-1, :] = libera[:, :-1, :] & solido[:, 1:, :]
    if not (pavimento.any() or soffitto.any()):
        return 0

    gx = ox + np.arange(16)[:, None]
    gz = oz + np.arange(16)[None, :]
    ys = (y0 + np.arange(H))[None, :, None]

    def campo(sale, lato):
        # chiazze larghe `lato` celle, uguali in ogni chunk (coordinate di mondo)
        x, z = gx // lato, gz // lato
        h = (x.astype(np.int64) * 73856093) ^ (z.astype(np.int64) * 19349663) \
            ^ (np.int64(sale) * 83492791)
        h = (h ^ (h >> 13)) * 1274126177
        return (((h ^ (h >> 16)) & 0xFFFF) / 65535.0)[:, None, :]

    gocciolato = campo(seed + 1, 22) > 0.60
    lussureggiante = (campo(seed + 2, 26) > 0.62) & (ys > -24)
    sculk = (campo(seed + 3, 20) > 0.80) & (ys < -34)
    dado = np.random.default_rng(_seme(ox, oz, seed + 5)).random(out.shape)
    cambiate = 0

    def metti(maschera, blocco):
        nonlocal cambiate
        out[maschera] = blocco
        cambiate += int(maschera.sum())

    def roccia_sotto(m):   # la roccia che fa da pavimento alle celle di `m`
        r = np.zeros_like(m)
        r[:, :-1, :] = m[:, 1:, :]
        return r

    def roccia_sopra(m):   # la roccia che fa da soffitto alle celle di `m`
        r = np.zeros_like(m)
        r[:, 1:, :] = m[:, :-1, :]
        return r

    # `pavimento` e `soffitto` sono celle LIBERE: la pianta o la punta va in
    # quella cella, la roccia che cambia e' il blocco sotto o sopra.

    # -- gocciolatoi: la roccia diventa dripstone, poi le punte ------------
    pav_g, sof_g = pavimento & gocciolato, soffitto & gocciolato
    metti(roccia_sotto(pav_g & (dado < 0.55)), d.dripstone)
    metti(roccia_sopra(sof_g & (dado < 0.55)), d.dripstone)
    metti(pav_g & (dado > 0.78), d.punta_su)
    metti(sof_g & (dado < 0.22), d.punta_giu)

    # -- grotta lussureggiante ------------------------------------------------
    pav_l, sof_l = pavimento & lussureggiante, soffitto & lussureggiante
    metti(roccia_sotto(pav_l & (dado < 0.70)), d.muschio)
    metti(roccia_sopra(sof_l & (dado < 0.45)), d.muschio)
    metti(pav_l & (dado > 0.50) & (dado < 0.85), d.tappeto)
    metti(pav_l & (dado >= 0.93) & (dado < 0.96), d.azalea)
    metti(pav_l & (dado >= 0.96) & (dado < 0.985), d.azalea_fiorita)
    metti(pav_l & (dado >= 0.985), d.foglia)
    metti(sof_l & (dado > 0.80) & (dado < 0.90), d.radici)
    metti(sof_l & (dado >= 0.90) & (dado < 0.95), d.liana)
    metti(sof_l & (dado >= 0.95), d.liana_nuda)

    # -- pozze nelle grotte lussureggianti ------------------------------------
    # Una cella di pavimento diventa acqua ferma solo se e' chiusa da roccia su
    # quattro lati e sotto: niente flussi, niente cascate che svuotano la grotta.
    # Si guarda `solido` com'era prima del muschio. Fuori dal bordo del chunk
    # non si sa cosa c'e' accanto, quindi le pozze stanno nei 14 blocchi interni.
    interno = np.zeros(out.shape, bool)
    interno[1:15, :, 1:15] = True
    roccia_p = roccia_sotto(pav_l & (dado > 0.965) & interno)
    chiusa = roccia_p.copy()
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        vicina = np.roll(solido, (dx, dz), axis=(0, 2))
        chiusa &= vicina
    sotto_p = np.zeros_like(solido)
    sotto_p[:, 1:, :] = solido[:, :-1, :]
    chiusa &= sotto_p
    metti(chiusa, d.acqua)

    # -- ruscelli: stesso criterio delle pozze, ma lungo una fascia sinuosa -----
    # Un ruscello e' acqua FERMA in un solco di una cella: ogni cella e' chiusa
    # da roccia sui quattro lati e sotto, quindi non scorre e non si svuota. La
    # fascia usa le coordinate di mondo, cosi' prosegue da un chunk al vicino
    # (dove il bordo interrompe i controlli resta un varco di roccia). Dove il
    # pavimento fa un gradino la cella non e' piu' chiusa e il ruscello si ferma.
    ondulazione = np.sin(gx * 0.21 + 1.7 * np.sin(gz * 0.11 + seed * 0.37) + seed)
    banda = (np.abs(ondulazione) < 0.14)[:, None, :]
    chiusa_r = roccia_sotto(pav_l & banda & interno)
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        chiusa_r &= np.roll(solido, (dx, dz), axis=(0, 2))
    chiusa_r &= sotto_p
    metti(chiusa_r, d.acqua)

    # -- sculk nelle caverne profonde ----------------------------------------
    pav_s, sof_s = pavimento & sculk, soffitto & sculk
    metti(roccia_sotto(pav_s & (dado < 0.80)), d.sculk)
    metti(pav_s & (dado > 0.97), d.sensore)
    metti(roccia_sopra(sof_s & (dado < 0.30)), d.sculk)
    return cambiate


def _seme(x: int, z: int, seed: int) -> int:
    """Seme deterministico per un chunk.

    La maschera non e' un dettaglio: meta' del mondo ha coordinate negative,
    lo XOR di due negativi e' negativo, e `default_rng` di un numero negativo
    solleva un'eccezione. Il mondo si fermava al primo chunk a ovest.
    """
    return ((x * 73856093) ^ (z * 19349663) ^ (seed * 83492791)) & 0x7FFFFFFF


def _quanti(rng, atteso: float) -> int:
    """Numero di grumi: parte intera piu' un dado sulla frazione."""
    n = int(atteso)
    return n + (1 if rng.random() < atteso - n else 0)
