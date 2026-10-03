"""Laghi e laghetti: bacini chiusi che una mappa disegnata non segnala.

Un fiume ha bisogno di un pendio che porti al mare: `fiumi.da_terreno` lo
trova seguendo il deflusso. Un bacino CHIUSO - una conca senza sbocco - non
ha un fiume, ha un lago. Prima di questo modulo quelle conche restavano
terreno asciutto: la mappa disegnata non dice mai "qui c'e' un lago", quindi
senza calcolarle dal rilievo non comparivano mai.

La stessa `fiumi.riempi_depressioni()` che serve al deflusso dei fiumi (un
priority-flood: alza ogni conca fino al punto di sfioro) e' esattamente lo
strumento giusto qui: dove il terreno riempito sta SOPRA il terreno vero, la'
c'e' una conca, e l'altezza del riempimento e' il pelo naturale dell'acqua se
quella conca fosse piena.

Un lago non e' un fiume - non scorre, non ha una foce - ma per il resto del
motore (biomi, vegetazione, insediamenti, disegno dei blocchi) e' la stessa
cosa: acqua dolce in quota, con un pelo suo che non e' il livello del mare.
Per questo si classifica come `mappa.FIUME`: tutto quello che sa gia' trattare
un fiume (niente alberi sopra, niente case, il blocco giusto in
`motore.SUPERFICIE`) tratta gia' correttamente un lago, senza toccare altri
moduli.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from .fiumi import riempi_depressioni


def trova(
    altezze: np.ndarray,
    mare: np.ndarray,
    esclusi: np.ndarray,
    profondita_minima: float = 1.5,
    area_minima: int = 12,
    area_massima: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Bacini chiusi abbastanza profondi da meritare un lago.

    Ritorna (maschera, livello): `livello` vale il pelo dell'acqua dentro la
    maschera, NaN altrove. `esclusi` toglie dalla ricerca quello che ha gia'
    un'acqua sua (fiumi) o una quota decisa altrimenti (vulcani/crateri) -
    senza, un lago potrebbe formarsi dentro un cratere o a cavallo di un
    fiume che in quel punto scorre piu' in basso del terreno intorno.

    `area_massima` e' un freno, non un capriccio: una grande vallata bassa
    SENZA sbocco e' rara ma possibile, e allagarla tutta produrrebbe un mare
    interno che sulla mappa non c'era. Un bacino piu' grande resta terreno
    asciutto - meglio un lago mancato che un lago assurdo.
    """
    riempito = riempi_depressioni(altezze, mare)
    profondita = riempito - altezze
    bacino = (profondita >= profondita_minima) & ~mare & ~esclusi
    vuoto_m = np.zeros_like(bacino)
    vuoto_l = np.full(altezze.shape, np.nan, np.float32)
    if not bacino.any():
        return vuoto_m, vuoto_l

    cc, n = label(bacino, structure=np.ones((3, 3), bool))
    if n == 0:
        return vuoto_m, vuoto_l
    dim = np.bincount(cc.ravel())
    dim[0] = 0
    buoni = np.flatnonzero((dim >= area_minima) & (dim <= area_massima))
    if buoni.size == 0:
        return vuoto_m, vuoto_l

    maschera = np.isin(cc, buoni)
    livello = np.full(altezze.shape, np.nan, np.float32)
    # Il pelo di un bacino e' UNIFORME (un lago e' piatto): si prende il
    # valore massimo di riempimento dentro ogni componente, non quello
    # cella per cella, che vicino al bordo puo' scendere sotto il vero
    # punto di sfioro per via di come cammina il priority-flood.
    for k in buoni:
        m = cc == k
        livello[m] = float(riempito[m].max())
    return maschera, livello


def scava(altezze: np.ndarray, maschera: np.ndarray, livello: np.ndarray,
          profondita_massima: float = 5.0) -> np.ndarray:
    """Il fondo del lago.

    Il terreno vero dentro un bacino chiuso e' gia' sotto il pelo per
    definizione (e' cosi' che lo si e' trovato) - non serve scavare, solo
    tenere un fondo ragionevole: non piu' in basso di `profondita_massima`
    sotto il pelo (un lago non e' un pozzo), e comunque abbastanza sotto il
    pelo perche' ci sia almeno un blocco d'acqua vera anche sul bordo.
    """
    if not maschera.any():
        return altezze
    fondo = np.minimum(altezze.astype(np.float32), livello - 1.0)
    fondo = np.maximum(fondo, livello - profondita_massima)
    return np.where(maschera, fondo, altezze)


def statistiche(maschera: np.ndarray, livello: np.ndarray) -> dict:
    if not maschera.any():
        return {"celle": 0, "bacini": 0, "quota_media": 0.0}
    cc, n = label(maschera, structure=np.ones((3, 3), bool))
    return {
        "celle": int(maschera.sum()),
        "bacini": int(n),
        "quota_media": float(np.nanmean(livello[maschera])),
    }
