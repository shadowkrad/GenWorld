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

    def aggiungi_rocce(self, ids) -> None:
        """Gli strati di `stratigrafia` sono roccia a tutti gli effetti.

        Senza questo, un filone di ferro che capita dentro un banco di
        andesite non si posa - `solo_pietra` non lo riconosce - e le vene si
        interrompono esattamente dove passa uno strato. Che e' il contrario
        di quello che fa una vena vera.
        """
        self.scavabile = tuple(dict.fromkeys(self.scavabile + tuple(ids)))


def _grumo(out, rng, off_x, off_z, y0, h_c, blocco, quanti, y_min, y_max,
           solo_pietra):
    """Un grumo di blocchi attorno a un punto, dentro la pietra.

    `off_x`/`off_z` spostano il centro nel sistema del chunk vicino che lo ha
    generato: cosi' un grumo nato a un blocco dal confine sporge qui dentro
    invece di essere tagliato di netto.

    `solo_pietra` e' la garanzia che non si buchino case, caverne o terreno di
    superficie: si sostituisce soltanto quello che e' pietra o ardesia.
    """
    H = out.shape[1]
    gx = int(rng.integers(0, 16)) + off_x
    gz = int(rng.integers(0, 16)) + off_z
    gy = int(rng.integers(y_min, y_max + 1))
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
                ly = gy + dy - y0
                if not (1 <= ly < H):
                    continue
                if gy + dy > int(h_c[lx, lz]) - 2:
                    continue
                if out[lx, ly, lz] not in solo_pietra:
                    continue
                out[lx, ly, lz] = blocco


def posa(out: np.ndarray, tav: Tavolozza, h_c: np.ndarray, ox: int, oz: int,
         y0: int, caverne: dict, quali, seed: int = 0) -> None:
    """Ardesia, minerali e caverne dentro l'array di chunk (16, H, 16)."""
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

    # --- caverne ---------------------------------------------------------
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
