"""Vulcani: cono, cratere, lago di lava e colate.

Un vulcano non e' una montagna colorata di nero. Ha una forma riconoscibile e
precisa: il profilo e' concavo verso l'alto - dolce alla base e sempre piu'
ripido verso la cima - il cratere e' un invaso, e le colate seguono la massima
pendenza fin dove si raffreddano.

E' l'unico elemento del progetto che si SOVRAPPONE al terreno invece di
dedurlo: il cono si somma alla heightmap, non la sostituisce. Cosi' un vulcano
piazzato su una collina ne eredita la base e non sembra appoggiato su un piano.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from .mappa import ACQUA, CRATERE, VULCANO
from .rumore import fbm


@dataclass
class Vulcano:
    z: int
    x: int
    raggio: int
    altezza: float
    raggio_cratere: int
    profondita_cratere: float
    quota_base: float
    seme: int = 0

    @property
    def quota_orlo(self) -> float:
        """Quota del bordo del cratere."""
        t = self.raggio_cratere / max(self.raggio, 1)
        return self.quota_base + self.altezza * (1.0 - t ** 1.5)

    @property
    def quota_lava(self) -> float:
        return self.quota_orlo - self.profondita_cratere * 0.45


def scegli_siti(
    cls: np.ndarray,
    altezze: np.ndarray,
    quanti: int = 3,
    separazione: int = 180,
    livello_mare: int = 62,
    seed: int = 0,
) -> list[Vulcano]:
    """Siti su terra alta e lontani fra loro.

    Un vulcano vuole spazio: il cono e le colate occupano un raggio ampio, e
    due coni sovrapposti diventano una collina informe.
    """
    if quanti <= 0:
        return []
    rng = np.random.default_rng(seed)
    terra = ~np.isin(cls, ACQUA)
    if not terra.any():
        return []

    # lontano dalla costa: un cono a filo d'acqua viene tagliato a meta'
    dist_costa = distance_transform_edt(terra)
    punteggio = np.where(terra & (altezze > livello_mare + 4),
                         altezze.astype(np.float32) + dist_costa * 1.5, -1.0)
    punteggio = punteggio + rng.random(cls.shape) * 6.0

    fuori: list[Vulcano] = []
    p = punteggio.copy()
    zz, xx = np.ogrid[:cls.shape[0], :cls.shape[1]]
    # un tentativo scartato (troppo vicino al bordo) non deve consumare un
    # vulcano: si cerca finche' non se ne trovano `quanti` o finisce la terra
    tentativi = 0
    while len(fuori) < quanti and tentativi < quanti * 40:
        tentativi += 1
        i = int(np.argmax(p))
        z, x = divmod(i, cls.shape[1])
        if p[z, x] <= 0:
            break
        raggio = int(rng.integers(30, 52))
        if min(z, x, cls.shape[0] - z, cls.shape[1] - x) < raggio + 4:
            p[max(0, z - 20):z + 20, max(0, x - 20):x + 20] = -1
            continue
        # la base e' la quota del terreno attorno, non quella del centro
        anello = ((zz - z) ** 2 + (xx - x) ** 2 >= (raggio - 4) ** 2) & \
                 ((zz - z) ** 2 + (xx - x) ** 2 <= (raggio + 3) ** 2)
        base = float(np.median(altezze[anello])) if anello.any() else float(altezze[z, x])
        rc = max(4, raggio // 5)
        fuori.append(Vulcano(
            z=z, x=x, raggio=raggio,
            altezza=float(rng.integers(42, 78)),
            raggio_cratere=rc,
            profondita_cratere=float(rng.integers(8, 15)),
            quota_base=base,
            seme=int(rng.integers(0, 2 ** 31 - 1)),
        ))
        p[(zz - z) ** 2 + (xx - x) ** 2 < separazione ** 2] = -1
    return fuori


def modella(
    cls: np.ndarray,
    altezze: np.ndarray,
    vulcani: list[Vulcano],
    livello: np.ndarray | None = None,
    rugosita: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scolpisce i coni. Ritorna (cls, altezze, lava, quota_lava)."""
    h = altezze.astype(np.float32).copy()
    c = cls.copy()
    H, W = h.shape
    lava = np.zeros((H, W), bool)
    q_lava = np.zeros((H, W), np.float32)
    if not vulcani:
        return c, np.round(h).astype(np.int32), lava, q_lava

    zz, xx = np.ogrid[:H, :W]
    for v in vulcani:
        rng_v = np.random.default_rng(v.seme)
        d = np.sqrt((zz - v.z) ** 2 + (xx - v.x) ** 2)
        ang = np.arctan2(zz - v.z, xx - v.x)

        # La base non e' un cerchio di compasso: tre onde lente deformano il
        # raggio del 10% circa. Senza, il vulcano sembra timbrato sulla mappa.
        if rugosita > 0:
            deforma = sum(
                a * np.sin(f * ang + float(rng_v.random() * 6.28))
                for f, a in ((2, 0.06), (3, 0.04), (5, 0.025))
            )
        else:
            deforma = 0.0
        raggio = v.raggio * (1.0 + deforma)
        dentro = d <= raggio
        if not np.any(dentro):
            continue

        # Profilo concavo verso l'alto: l'esponente sopra 1 rende il cono
        # dolce alla base e ripido alla cima. Un cono lineare sembra un
        # mucchio di terra, non un vulcano.
        t = np.clip(d / raggio, 0.0, 1.0)
        cono = v.quota_base + v.altezza * (1.0 - t ** 1.5)

        # VALLONI. Un cono matematicamente perfetto ha il gradiente esattamente
        # radiale, quindi le colate scendono in linea retta dal cratere e il
        # risultato sembra una ruota con i raggi. Un vulcano vero e' solcato da
        # valloni che partono dall'orlo e si allargano scendendo, e la lava li
        # segue invece di andare dritta.
        #
        # Il rumore deve essere ANGOLARE, non planare: un fbm con celle piu'
        # larghe del cono lo inclina soltanto, non lo solca. Qui la quota
        # dipende dall'angolo attorno al cratere, cosi' le pieghe corrono
        # naturalmente lungo la linea di massima pendenza.
        if rugosita > 0:
            n_valloni = int(rng_v.integers(9, 16))
            g = np.zeros_like(t, np.float32)
            peso = 0.0
            for freq, a in ((n_valloni, 0.60),
                            (n_valloni * 2, 0.25),
                            (max(3, n_valloni // 2), 0.15)):
                fase = float(rng_v.random() * 2 * np.pi)
                # i valloni non sono rettilinei: si torcono scendendo
                torsione = float(rng_v.normal(0.0, 0.7))
                g += a * np.sin(freq * (ang + torsione * t) + fase)
                peso += a
            g /= peso
            # I valloni non sono equispaziati: un'onda lenta apre alcuni
            # settori e ne appiattisce altri, altrimenti il cono sembra una
            # zucca scanalata.
            settori = 1.0 + 0.55 * np.sin(3 * ang + float(rng_v.random() * 6.28))
            g = g * settori

            # e non sono lisci: un fbm fine, con celle piu' piccole del cono,
            # sporca i crinali quel tanto che basta
            lato = max(cls.shape)
            fine = fbm(lato, ottave=3, celle_base=max(8, lato // 14),
                       persistenza=0.5, seed=v.seme % (2 ** 31)) - 0.5
            g = g + 0.7 * fine[:cls.shape[0], :cls.shape[1]]

            # ampiezza nulla sull'orlo (che resta un anello netto) e nulla
            # sul bordo esterno (altrimenti la base diventa una stella)
            u = (d - v.raggio_cratere) / np.maximum(raggio - v.raggio_cratere, 1)
            rampa = np.clip(u / 0.35, 0, 1) * np.clip((1.0 - u) / 0.30, 0, 1)
            cono = cono + g * v.altezza * rugosita * rampa

        # si SOMMA al terreno: un vulcano su una collina eredita la collina
        nuovo = np.maximum(h, cono)
        h = np.where(dentro, nuovo, h)

        # cratere: invaso con il fondo piatto
        in_cratere = d <= v.raggio_cratere
        if in_cratere.any():
            tc = np.clip(d / max(v.raggio_cratere, 1), 0.0, 1.0)
            fondo = v.quota_orlo - v.profondita_cratere * (1.0 - tc ** 2)
            h = np.where(in_cratere, fondo, h)
            lago = in_cratere & (fondo < v.quota_lava)
            lava |= lago
            q_lava = np.where(lago, np.maximum(q_lava, v.quota_lava), q_lava)

        c = np.where(dentro & ~np.isin(c, ACQUA), np.uint8(VULCANO), c)
        c = np.where(in_cratere, np.uint8(CRATERE), c)

    h_int = np.round(h).astype(np.int32)
    return c, h_int, lava, q_lava


# Le tre fasi di una colata, lette sull'avanzamento.
SOGLIA_LIQUIDA = 0.30     # fin qui e' lava viva
SOGLIA_CROSTA = 0.65      # fin qui e' magma incandescente sotto la crosta
                          # oltre, basalto freddo


def manto(cls: np.ndarray, seed: int = 0) -> np.ndarray:
    """Variante di roccia per ogni cella del cono: 0 basalto, 1 basalto
    levigato, 2 blackstone, 3 tufo.

    Un vulcano tutto di basalto identico e' una montagna grigia. Le colate
    vecchie, la cenere compattata e le fratture non hanno lo stesso colore, e
    sono loro a far leggere il cono come una cosa costruita a strati.
    """
    from .rumore import fbm
    lato = max(cls.shape)
    n = fbm(lato, ottave=3, celle_base=max(4, lato // 90), persistenza=0.55,
            seed=seed + 5501)[:cls.shape[0], :cls.shape[1]]
    lo, hi = float(n.min()), float(n.max())
    n = (n - lo) / max(hi - lo, 1e-6)
    fuori = np.zeros(cls.shape, np.uint8)
    fuori[n > 0.42] = 1
    fuori[n > 0.62] = 2
    fuori[n > 0.82] = 3
    return fuori


def _bocche(liscio: np.ndarray, v: "Vulcano", quante: int,
            rng: np.random.Generator) -> list[float]:
    """Gli angoli da cui esce la lava: le SELLE dell'orlo, non angoli a caso.

    Con le partenze estratte a sorte le colate uscivano a raggiera e poi
    tagliavano i valloni di traverso, perche' sul fianco di un cono la
    pendenza generale e' molto piu' forte di quella del vallone. Dall'alto
    veniva fuori una ruota di carro.

    L'orlo di un cratere vero non e' una circonferenza: ha delle selle, e la
    lava esce da li'. Partendo dal punto basso la colata e' gia' dentro il
    vallone, e l'inerzia ce la tiene.
    """
    H, W = liscio.shape
    n = 240
    angoli = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = v.raggio_cratere + 1
    zz = np.clip((v.z + r * np.sin(angoli)).round().astype(int), 0, H - 1)
    xx = np.clip((v.x + r * np.cos(angoli)).round().astype(int), 0, W - 1)
    orlo = liscio[zz, xx]

    # minimi locali sull'anello, in ordine di quota
    prec, succ = np.roll(orlo, 1), np.roll(orlo, -1)
    selle = np.flatnonzero((orlo <= prec) & (orlo <= succ))
    if selle.size < quante:
        return [float(a) for a in rng.choice(angoli, size=quante, replace=False)]
    selle = selle[np.argsort(orlo[selle])]

    # prese le piu' basse, ma non due attaccate: sarebbe una colata sola
    scelte: list[int] = []
    for i in selle:
        if all(min(abs(i - j), n - abs(i - j)) > n // (quante * 3)
               for j in scelte):
            scelte.append(int(i))
        if len(scelte) == quante:
            break
    return [float(angoli[i]) for i in scelte]


def colate(
    altezze: np.ndarray,
    cls: np.ndarray,
    vulcani: list[Vulcano],
    per_vulcano: int = 7,
    lunghezza: float = 2.4,
    larghezza: int = 1,
    pendenza_minima: float = 0.22,
) -> np.ndarray:
    """Colate di lava dall'orlo del cratere lungo la massima pendenza.

    Non e' un fiume: la lava scende dritta e si ferma quando si raffredda, non
    quando arriva al mare. Si arresta dove il pendio si spiana - senza questa
    soglia l'inerzia la porta a tirare righe rette per mezza mappa, perche' in
    pianura il gradiente e' nullo e nulla la frena.

    Ritorna l'AVANZAMENTO, non una maschera: 0 dove colata non ce n'e',
    poi da poco sopra zero all'orlo del cratere fino a 1 alla punta. Serve
    perche' una colata non e' fatta di una materia sola. Vicino alla bocca e'
    liquida e si vede la lava; a meta' ha gia' la crosta e si vede il magma;
    in punta e' basalto freddo. La prima versione posava mattoncini di magma
    dall'inizio alla fine, ed e' il difetto che si vede nello screenshot: una
    striscia di mattonelle arancioni sempre uguali, dall'orlo alla base.
    """
    H, W = altezze.shape
    fuori = np.zeros((H, W), np.float32)
    if not vulcani:
        return fuori

    liscio = gaussian_filter(altezze.astype(np.float32), 1.0)
    gz, gx = np.gradient(liscio)
    pendenza = np.hypot(gz, gx)
    acqua = np.isin(cls, ACQUA)

    for v in vulcani:
        rng = np.random.default_rng(v.seme + 7)
        for ang in _bocche(liscio, v, per_vulcano, rng):
            z = v.z + (v.raggio_cratere + 1) * np.sin(ang)
            x = v.x + (v.raggio_cratere + 1) * np.cos(ang)
            passi = int(v.raggio * lunghezza)
            dz = dx = 0.0
            for passo in range(passi):
                iz, ix = int(round(z)), int(round(x))
                if not (0 <= iz < H and 0 <= ix < W) or acqua[iz, ix]:
                    break
                # una colata non torna dentro il cratere: col filo di
                # casualita' nella direzione, ogni tanto ci rientrava
                if cls[iz, ix] == CRATERE:
                    break
                # l'avanzamento vale per la colata piu' "giovane" che passa di
                # qui: due colate che si incrociano non si raffreddano a vicenda
                t = (passo + 1) / max(passi, 1)
                fuori[iz, ix] = (t if fuori[iz, ix] == 0
                                 else min(fuori[iz, ix], t))
                if pendenza[iz, ix] < pendenza_minima:
                    break  # raffreddamento: il pendio non basta piu'
                # inerzia: senza, la colata zigzaga a ogni increspatura;
                # troppa, e scavalca i valloni invece di infilarcisi
                # un filo di casualita' nella direzione: senza, quattro
                # colate su un cono liscio vengono quattro raggi dritti, e
                # dall'alto il vulcano sembra una ruota di carro
                dz = 0.40 * dz - 0.60 * gz[iz, ix] + float(rng.normal(0, 0.10))
                dx = 0.40 * dx - 0.60 * gx[iz, ix] + float(rng.normal(0, 0.10))
                n = float(np.hypot(dz, dx))
                if n < 1e-4:
                    break
                z += dz / n
                x += dx / n
    if larghezza > 0:
        # si allarga l'avanzamento, non una maschera: il bordo della colata
        # eredita la temperatura del suo asse. Un massimo locale va bene
        # perche' l'avanzamento cresce scendendo: allargando prendiamo il
        # valore piu' avanzato, cioe' la crosta, che sul bordo e' giusta.
        from scipy.ndimage import grey_dilation
        fuori = grey_dilation(fuori, size=2 * larghezza + 1)
    fuori = np.where(acqua, 0.0, fuori).astype(np.float32)
    return fuori


def statistiche(vulcani: list[Vulcano], cls: np.ndarray,
                lava: np.ndarray, colata: np.ndarray) -> dict:
    return {
        "vulcani": len(vulcani),
        "cono": int((cls == VULCANO).sum()),
        "cratere": int((cls == CRATERE).sum()),
        "lago_di_lava": int(lava.sum()),
        "colate": int((colata > 0).sum()),
        "colate_incandescenti": int(((colata > 0) & (colata <= SOGLIA_LIQUIDA)).sum()),
        "quota_massima": int(max((v.quota_base + v.altezza for v in vulcani), default=0)),
    }
