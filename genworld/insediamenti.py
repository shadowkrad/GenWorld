"""Villaggi: scelta dei siti, lotti, terrazzamento.

La pianificazione avviene UNA VOLTA sulla mappa intera, prima di scrivere
qualunque chunk, perche' deve poter modificare il terreno: un edificio su un
pendio va messo su un lotto spianato, altrimenti esce mezzo sepolto da un lato
e sospeso dall'altro. Spianare mentre si scrivono i chunk e' impossibile - il
chunk accanto e' gia' stato chiuso.

Le palafitte non hanno codice proprio. Un lotto che cade sull'acqua produce un
edificio col modificatore `palafitta`, e il generatore di edifici fa il resto.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from . import edifici as E
from .mappa import (ACQUA, FIUME, FORESTA, MARE, MONTAGNA, OCEANO, PIANURA,
                    PRATERIA, SPIAGGIA)

# classi su cui si puo' fondare un villaggio
ABITABILI = (PIANURA, PRATERIA, FORESTA, SPIAGGIA)


def scegli_siti(
    cls: np.ndarray,
    altezze: np.ndarray,
    livello_mare: int = 62,
    separazione: int = 130,
    celle_per_villaggio: int = 45_000,
    massimo: int = 14,
    raggio_min: int = 18,
    raggio_max: int = 52,
    seed: int = 0,
) -> list[tuple[int, int, int]]:
    """Ritorna [(z, x, raggio)]. I siti stanno lontani fra loro e vicino all'acqua.

    Il criterio "vicino all'acqua" non e' folklore: un insediamento nasce dove
    c'e' da bere, e visivamente un villaggio sul nulla in mezzo alla pianura
    sembra piazzato a caso, mentre uno su un'ansa sembra scelto.
    """
    rng = np.random.default_rng(seed)
    terra = np.isin(cls, ABITABILI)
    if not terra.any():
        return []

    pend = np.hypot(*np.gradient(gaussian_filter(altezze.astype(np.float32), 2)))
    acqua = np.isin(cls, ACQUA)
    dist_acqua = distance_transform_edt(~acqua) if acqua.any() else np.full(cls.shape, 999.0)

    buono = terra & (pend < 1.1) & (altezze > livello_mare + 1)
    if not buono.any():
        return []

    # punteggio: piatto e vicino all'acqua, ma non a filo d'acqua
    punteggio = np.where(buono,
                         1.0 / (1.0 + pend) * np.exp(-np.abs(dist_acqua - 8.0) / 22.0),
                         -1.0)
    punteggio += rng.random(cls.shape) * 0.12      # rompe i pareggi

    quanti = int(np.clip(terra.sum() // celle_per_villaggio, 1, massimo))
    siti: list[tuple[int, int, int]] = []
    p = punteggio.copy()
    for k in range(quanti):
        i = int(np.argmax(p))
        z, x = divmod(i, cls.shape[1])
        if p[z, x] <= 0:
            break
        # Il sito migliore diventa il piu' grande. Non e' un vezzo: una
        # gerarchia urbana esiste perche' il posto buono attira, e una mappa
        # con cinque insediamenti tutti uguali sembra un campeggio.
        frazione = k / max(quanti - 1, 1)
        raggio = int(round(raggio_max - frazione * (raggio_max - raggio_min)))
        raggio += int(rng.integers(-3, 4))
        raggio = int(np.clip(raggio, raggio_min, raggio_max))
        siti.append((z, x, raggio))
        zz, xx = np.ogrid[:cls.shape[0], :cls.shape[1]]
        p[(zz - z) ** 2 + (xx - x) ** 2 < separazione ** 2] = -1.0
    return siti


def pianifica(
    cls: np.ndarray,
    altezze: np.ndarray,
    livello: np.ndarray,
    livello_mare: int = 62,
    passo_lotti: int = 12,
    scala: float = 1.0,
    seed: int = 0,
) -> tuple[list[E.Edificio], np.ndarray, int]:
    """Progetta i villaggi. Ritorna (edifici, altezze spianate, n. villaggi)."""
    if scala <= 0:
        return [], altezze, 0

    h = altezze.astype(np.int32).copy()
    H, W = cls.shape
    rng = np.random.default_rng(seed)
    siti = scegli_siti(cls, h, livello_mare,
                       celle_per_villaggio=int(45_000 / max(scala, 1e-3)),
                       seed=seed)
    fuori: list[E.Edificio] = []

    for nz, nx, raggio in siti:
        # griglia di lotti dentro il raggio, sfalsata
        for lz in range(nz - raggio, nz + raggio + 1, passo_lotti):
            for lx in range(nx - raggio, nx + raggio + 1, passo_lotti):
                cz = lz + int(rng.integers(0, 3))
                cx = lx + int(rng.integers(0, 3))
                if (cz - nz) ** 2 + (cx - nx) ** 2 > raggio ** 2:
                    continue
                if (abs(cz - nz) < 5 and abs(cx - nx) < 5):
                    continue                      # piazza libera al centro
                ed = _progetta_lotto(cls, h, livello, cz, cx, livello_mare, rng)
                if ed is not None:
                    _terrazza(h, ed)
                    fuori.append(ed)
    return fuori, h, len(siti)


def _progetta_lotto(cls, h, livello, cz, cx, livello_mare, rng) -> E.Edificio | None:
    H, W = cls.shape
    larg = int(rng.integers(7, 11))
    prof = int(rng.integers(7, 11))
    x0, z0 = cx - larg // 2, cz - prof // 2
    if x0 < 2 or z0 < 2 or x0 + larg >= W - 2 or z0 + prof >= H - 2:
        return None

    fetta_c = cls[z0:z0 + prof, x0:x0 + larg]
    fetta_h = h[z0:z0 + prof, x0:x0 + larg]
    if fetta_c.size == 0:
        return None

    acqua = np.isin(fetta_c, ACQUA)
    quota_acqua = float(np.max(livello[z0:z0 + prof, x0:x0 + larg]))

    if acqua.mean() > 0.6:
        # sedime sull'acqua: palafitta. Il generatore di edifici non sapra'
        # nemmeno di essere sull'acqua, monta sulla piattaforma e basta.
        if int(np.max(fetta_c[acqua], initial=OCEANO)) == OCEANO:
            return None                            # non in mare aperto
        base = int(round(quota_acqua)) + 2
        fondale = int(np.min(fetta_h))
        palafitta = True
        stile = "prato"
    elif acqua.any():
        return None                                # mezzo dentro e mezzo fuori
    else:
        dislivello = int(fetta_h.max() - fetta_h.min())
        if dislivello > 6:
            return None                            # pendio troppo ripido
        base = int(np.median(fetta_h))
        if base <= livello_mare:
            return None
        fondale = base
        palafitta = False
        classe = int(np.bincount(fetta_c.ravel()).argmax())
        if classe == MONTAGNA and dislivello > 4:
            return None
        stile = E.PALETTE_PER_CLASSE.get(classe, "prato")

    return E.Edificio(
        x=x0, z=z0, larghezza=larg, profondita=prof, base=base,
        piani=int(rng.integers(1, 3)), porta=int(rng.integers(0, 4)),
        stile=stile, palafitta=palafitta, fondale=fondale,
        seme=int(rng.integers(0, 2 ** 31 - 1)),
    )


def _terrazza(h: np.ndarray, ed: E.Edificio, raccordo: int = 5,
              intoccabile: np.ndarray | None = None) -> None:
    """Spiana il lotto e raccorda i bordi.

    Senza raccordo il lotto spianato diventa un piedistallo squadrato in mezzo
    al pendio, che e' peggio della casa storta.

    Il raccordo e' passato da 3 a 5 celle dopo aver guardato una citta' di
    Arda dall'alto: fra due case a quote diverse restava un salto di cinque o
    sei blocchi smaltito in tre celle, cioe' una parete. Cinquanta case cosi'
    non sono un paese in pendenza, sono un paese di trincee.

    `intoccabile` e' la maschera di cio' che il raccordo non deve toccare -
    in pratica l'acqua e le sue sponde. Un raccordo che abbassa una riva
    lascia il fiume sospeso in aria: e' successo, si vede nello screenshot.
    """
    if ed.palafitta:
        return
    H, W = h.shape
    x0, z0 = max(0, ed.x - raccordo), max(0, ed.z - raccordo)
    x1, z1 = min(W, ed.x1 + raccordo), min(H, ed.z1 + raccordo)
    zz, xx = np.ogrid[z0:z1, x0:x1]
    # distanza dal sedime, 0 dentro e crescente fuori
    dx = np.maximum(np.maximum(ed.x - xx, xx - (ed.x1 - 1)), 0)
    dz = np.maximum(np.maximum(ed.z - zz, zz - (ed.z1 - 1)), 0)
    d = np.maximum(dx, dz).astype(np.float32)
    # curva a S invece che rampa lineare: il raccordo deve attaccarsi al
    # terreno con pendenza nulla, altrimenti il gradino si sposta solo in fuori
    t = np.clip(1.0 - d / (raccordo + 1.0), 0.0, 1.0)
    peso = t * t * (3.0 - 2.0 * t)
    if intoccabile is not None:
        peso = np.where(intoccabile[z0:z1, x0:x1], 0.0, peso)
    porzione = h[z0:z1, x0:x1].astype(np.float32)
    h[z0:z1, x0:x1] = np.round(porzione * (1 - peso) + ed.base * peso).astype(np.int32)


# --------------------------------------------------------------------------
# Indice per chunk
# --------------------------------------------------------------------------

def indice_per_chunk(edifici: list[E.Edificio], passo: int = 16) -> dict:
    """Ogni edificio va registrato in TUTTI i chunk che il suo ingombro tocca.

    Un edificio e' largo piu' di un chunk: registrarlo solo dove sta il suo
    angolo lo farebbe comparire tagliato.
    """
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, ed in enumerate(edifici):
        x0, z0, x1, z1 = E.ingombro(ed)
        for cz in range(z0 // passo, z1 // passo + 1):
            for cx in range(x0 // passo, x1 // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def estendi_indice_per_modello(indice: dict, i: int, px: int, pz: int,
                               ix: int, iz: int, passo: int = 16) -> None:
    """Aggiunge l'edificio `i` ai chunk che il modello REALE tocca.

    `indice_per_chunk` registra ogni edificio sui chunk del suo lotto
    (`edifici.ingombro`, margine fisso), calcolato in fase di pianificazione -
    prima che si sappia quale modello da `templates/` finira' su quel lotto.
    L'ultimo ripiego di `template.scegli()` puo' pero' restituire un modello
    piu' grande del lotto, di quanto capita: se l'eccedenza supera il
    margine gia' incluso in quell'indice, i chunk oltre il bordo del lotto
    non sanno di dover disegnare questa casa, e in gioco si vede tagliata di
    netto al confine del chunk. Va chiamata dopo la scelta del modello, con
    `px, pz` l'angolo minimo dove verra' posato e `ix, iz` il suo ingombro
    reale (girato)."""
    for cz in range(pz // passo, (pz + iz - 1) // passo + 1):
        for cx in range(px // passo, (px + ix - 1) // passo + 1):
            cella = indice.setdefault((cx, cz), [])
            if i not in cella:
                cella.append(i)


def conteggio(edifici: list[E.Edificio]) -> dict[str, int]:
    fuori: dict[str, int] = {}
    for ed in edifici:
        chiave = "palafitta" if ed.palafitta else ed.stile
        fuori[chiave] = fuori.get(chiave, 0) + 1
    return fuori
