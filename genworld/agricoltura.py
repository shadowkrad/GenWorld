"""Campi, poderi e frutteti.

Un paese senza campi e' un fondale. Le botteghe vendono frutta e carne, ma
attorno alle mura c'era il bosco fino al fossato: la campagna coltivata e' la
prova che ci abita qualcuno, ed e' anche la cosa che dall'alto si legge
prima di tutte, perche' e' fatta di forme regolari in mezzo a forme che non
lo sono.

Tre elementi, ognuno con la sua ragione:

* **il podere** - un rettangolo arato con un canale d'acqua nel mezzo. Il
  canale non e' decorazione: in Minecraft la terra arata resta bagnata solo
  entro quattro blocchi dall'acqua, altrimenti si secca e torna terra. La
  forma del podere DISCENDE da quella regola, come i ponti discendevano dalla
  quota della strada.
* **il frutteto** - alberi a filari su un reticolo con un po' di disordine.
  Il melo in Minecraft non esiste: e' la quercia, che le mele le fa cadere
  davvero. Il ciliegio invece c'e', ed e' l'unico albero che sembra un
  frutteto anche visto da lontano.
* **la siepe** - cespugli di bacche lungo il recinto, che chiudono il campo e
  danno qualcosa da raccogliere.

I poderi si SPIANANO, come i lotti delle case: un campo in pendenza perde
l'acqua e si vede.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from . import vegetazione as V
from .mappa import ACQUA, FORESTA, PIANURA, PRATERIA, SPIAGGIA

# Codici della mappa dei campi. L'arato non e' un codice solo: ce n'e' uno
# per coltura, perche' la coltura appartiene al PODERE e il chunk, quando
# scrive i blocchi, non sa di quale podere sia la cella che ha davanti.
# Prima la sceglievo dalla posizione della cella e veniva fuori una
# scacchiera di grano, carote e patate dentro lo stesso campo: bella da
# vedere e agricolturalmente assurda.
NIENTE, CANALE, SENTIERO, RECINTO, PRATO = range(5)
ARATO = 8                      # ARATO + indice della coltura

COLTURE = ("wheat", "carrots", "potatoes", "beetroots")

# dove si puo' coltivare
COLTIVABILI = (PIANURA, PRATERIA, FORESTA, SPIAGGIA)

LATO_PODERE = 9        # piu' il recinto: 11 in tutto
LATO_FRUTTETO = 17


@dataclass
class Podere:
    x: int
    z: int
    lato: int
    base: int
    coltura: str = "wheat"
    frutteto: bool = False
    specie: int = V.MELO
    seme: int = 0

    @property
    def x1(self) -> int: return self.x + self.lato
    @property
    def z1(self) -> int: return self.z + self.lato


def _spiana(h: np.ndarray, x0: int, z0: int, lato: int, quota: int,
            raccordo: int = 2) -> None:
    """Come `_terrazza` per le case: il campo e' piano e i bordi raccordano.

    Senza raccordo il podere diventa un francobollo sollevato in mezzo al
    pendio - lo stesso difetto che aveva il lotto edilizio prima di avere il
    suo raccordo.
    """
    H, W = h.shape
    a0, b0 = max(0, x0 - raccordo), max(0, z0 - raccordo)
    a1, b1 = min(W, x0 + lato + raccordo), min(H, z0 + lato + raccordo)
    zz, xx = np.ogrid[b0:b1, a0:a1]
    dx = np.maximum(np.maximum(x0 - xx, xx - (x0 + lato - 1)), 0)
    dz = np.maximum(np.maximum(z0 - zz, zz - (z0 + lato - 1)), 0)
    d = np.maximum(dx, dz).astype(np.float32)
    peso = np.clip(1.0 - d / (raccordo + 1.0), 0.0, 1.0)
    porzione = h[b0:b1, a0:a1].astype(np.float32)
    h[b0:b1, a0:a1] = np.round(porzione * (1 - peso) + quota * peso).astype(np.int32)


def pianifica(
    cls: np.ndarray,
    altezze: np.ndarray,
    occupato: np.ndarray,
    citta: list,
    livello_mare: int = 62,
    scala: float = 1.0,
    seed: int = 0,
) -> tuple[list[Podere], np.ndarray, np.ndarray, dict]:
    """Sistema i poderi attorno agli insediamenti.

    `occupato` marca tutto quello che c'e' gia': strade, case, mura, banchi.
    Ritorna (poderi, altezze spianate, mappa dei campi, alberi da frutto).
    """
    h = altezze.astype(np.int32).copy()
    H, W = cls.shape
    campi = np.zeros((H, W), np.uint8)
    poderi: list[Podere] = []
    if scala <= 0 or not citta:
        return poderi, h, campi, _alberi_vuoti()

    pend = np.hypot(*np.gradient(gaussian_filter(h.astype(np.float32), 1.5)))
    buono = (np.isin(cls, COLTIVABILI) & (h > livello_mare + 1) & (pend < 0.9))
    preso = occupato.copy()

    alberi_x: list[int] = []
    alberi_z: list[int] = []
    alberi_y: list[int] = []
    alberi_sp: list[int] = []
    alberi_alt: list[int] = []
    alberi_seme: list[int] = []

    for n, c in enumerate(citta):
        rng = np.random.default_rng(seed * 7919 + n)
        # quanti poderi: uno ogni tre case, che e' gia' una campagna magra
        quanti = int(max(2, round(c.edifici / 3.0 * scala)))
        raggio = int(c.raggio * 1.9)
        messi = 0
        for _ in range(quanti * 60):
            if messi >= quanti:
                break
            frutteto = rng.random() < 0.33
            lato = (LATO_FRUTTETO if frutteto else LATO_PODERE) + 2
            ang = rng.random() * 2 * np.pi
            d = c.raggio * 0.75 + rng.random() * (raggio - c.raggio * 0.75)
            z0 = int(round(c.z + d * np.sin(ang))) - lato // 2
            x0 = int(round(c.x + d * np.cos(ang))) - lato // 2
            if x0 < 2 or z0 < 2 or x0 + lato >= W - 2 or z0 + lato >= H - 2:
                continue
            fetta_b = buono[z0:z0 + lato, x0:x0 + lato]
            if not fetta_b.all() or preso[z0 - 1:z0 + lato + 1,
                                         x0 - 1:x0 + lato + 1].any():
                continue
            quote = h[z0:z0 + lato, x0:x0 + lato]
            if int(np.percentile(quote, 90) - np.percentile(quote, 10)) > 4:
                continue

            base = int(np.median(quote))
            _spiana(h, x0, z0, lato, base)
            p = Podere(x=x0, z=z0, lato=lato, base=base,
                       coltura=COLTURE[int(rng.integers(0, len(COLTURE)))],
                       frutteto=frutteto,
                       specie=V.CILIEGIO if rng.random() < 0.35 else V.MELO,
                       seme=int(rng.integers(0, 2 ** 31 - 1)))
            _disegna_podere(campi, p)
            if frutteto:
                _pianta_filari(p, h, rng, alberi_x, alberi_z, alberi_y,
                               alberi_sp, alberi_alt, alberi_seme)
            _siepe(p, h, rng, alberi_x, alberi_z, alberi_y,
                   alberi_sp, alberi_alt, alberi_seme)
            preso[z0 - 1:z0 + lato + 1, x0 - 1:x0 + lato + 1] = True
            poderi.append(p)
            messi += 1

    alberi = {
        "x": np.array(alberi_x, np.int32), "z": np.array(alberi_z, np.int32),
        "y": np.array(alberi_y, np.int32),
        "specie": np.array(alberi_sp, np.int32),
        "altezza": np.array(alberi_alt, np.int32),
        "seme": np.array(alberi_seme, np.int64),
    }
    return poderi, h, campi, alberi


def _alberi_vuoti() -> dict:
    v = np.zeros(0, np.int32)
    return {"x": v, "z": v, "y": v, "specie": v, "altezza": v,
            "seme": np.zeros(0, np.int64)}


def _disegna_podere(campi: np.ndarray, p: Podere) -> None:
    """Recinto sul perimetro, dentro il campo o il prato del frutteto."""
    x0, z0, l = p.x, p.z, p.lato
    campi[z0:z0 + l, x0:x0 + l] = RECINTO
    campi[z0 + 1:z0 + l - 1, x0 + 1:x0 + l - 1] = (
        PRATO if p.frutteto else ARATO + COLTURE.index(p.coltura))
    if not p.frutteto:
        # Il canale nel mezzo. E' QUI che sta la regola: la terra arata resta
        # bagnata entro quattro blocchi dall'acqua, quindi un campo largo
        # nove con il canale in mezzo e' bagnato tutto, uno largo undici no.
        mz = z0 + l // 2
        campi[mz, x0 + 1:x0 + l - 1] = CANALE
    # un varco nel recinto, per entrarci
    campi[z0 + l // 2, x0] = SENTIERO


def _pianta_filari(p: Podere, h: np.ndarray, rng, ax, az, ay, asp, aalt,
                   aseme) -> None:
    """Alberi a filari: reticolo di quattro, con un po' di disordine.

    Un frutteto e' regolare - e' la sua firma vista dall'alto - ma non e' una
    scacchiera perfetta: quella si legge come generata, esattamente come il
    cono del vulcano senza valloni.
    """
    lo, hi = V.ALTEZZA[p.specie]
    # passo 5, non 4: la chioma di una latifoglia e' larga cinque blocchi, e
    # a quattro di distanza i filari si saldano in un bosco unico
    for gz in range(p.z + 3, p.z + p.lato - 2, 5):
        for gx in range(p.x + 3, p.x + p.lato - 2, 5):
            z = gz + int(rng.integers(-1, 2))
            x = gx + int(rng.integers(-1, 2))
            if not (p.z + 1 < z < p.z + p.lato - 2
                    and p.x + 1 < x < p.x + p.lato - 2):
                continue
            ax.append(x); az.append(z); ay.append(int(h[z, x]))
            asp.append(p.specie)
            aalt.append(int(rng.integers(lo, hi + 1)))
            aseme.append(int(rng.integers(0, 2 ** 31 - 1)))


def _siepe(p: Podere, h: np.ndarray, rng, ax, az, ay, asp, aalt, aseme) -> None:
    """Cespugli di bacche appoggiati al recinto, qua e la'."""
    l = p.lato
    bordi = ([(p.z - 1, x) for x in range(p.x, p.x + l)]
             + [(p.z + l, x) for x in range(p.x, p.x + l)]
             + [(z, p.x - 1) for z in range(p.z, p.z + l)]
             + [(z, p.x + l) for z in range(p.z, p.z + l)])
    for z, x in bordi:
        if rng.random() > 0.22:
            continue
        if not (0 <= z < h.shape[0] and 0 <= x < h.shape[1]):
            continue
        ax.append(x); az.append(z); ay.append(int(h[z, x]))
        asp.append(V.BACCHE); aalt.append(1)
        aseme.append(int(rng.integers(0, 2 ** 31 - 1)))


def statistiche(poderi: list[Podere], campi: np.ndarray,
                alberi: dict) -> dict:
    da_frutto = int(np.isin(alberi["specie"], (V.MELO, V.CILIEGIO)).sum()) \
        if alberi["specie"].size else 0
    return {
        "poderi": sum(1 for p in poderi if not p.frutteto),
        "frutteti": sum(1 for p in poderi if p.frutteto),
        "arato": int((campi >= ARATO).sum()),
        "canali": int((campi == CANALE).sum()),
        "recinto": int((campi == RECINTO).sum()),
        "alberi_da_frutto": da_frutto,
        "bacche": int((alberi["specie"] == V.BACCHE).sum())
        if alberi["specie"].size else 0,
    }
