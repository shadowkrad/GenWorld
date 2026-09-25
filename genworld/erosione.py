"""Erosione idraulica per gocce.

E' il passaggio che separa "terreno generato" da "terreno credibile". Una
heightmap interpolata da etichette e' fatta di dossi gonfi e versanti lisci:
non ha valli a V, ne' creste affilate, ne' conoidi di deposito a valle. Un
occhio lo legge come artificiale anche senza saper dire perche'.

Il modello e' quello a gocce: una particella d'acqua parte in un punto a caso,
scende seguendo la pendenza, erode dove la corrente e' veloce e deposita dove
rallenta. Migliaia di gocce scavano i solchi che poi diventano valli.

L'implementazione e' VETTORIALE: invece di simulare una goccia alla volta -
in Python sarebbero milioni di iterazioni e minuti di attesa - si fanno
avanzare tutte le gocce insieme, un passo per volta, come array numpy. Gli
scatter su celle ripetute si fanno con np.add.at, che somma invece di
sovrascrivere: due gocce che passano sulla stessa cella devono sommare i loro
effetti, non annullarsi.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def erodi(
    altezze: np.ndarray,
    gocce_per_cella: float = 0.3,
    n_gocce: int | None = None,
    passi: int = 36,
    inerzia: float = 0.05,
    capacita: float = 2.0,
    erosione: float = 0.30,
    deposito: float = 0.30,
    evaporazione: float = 0.05,
    gravita: float = 4.0,
    velocita_massima: float = 5.0,
    sedimento_massimo: float = 2.0,
    dislivello_massimo: float = 3.0,
    scavo_per_passo: float = 0.4,
    pendenza_minima: float = 0.02,
    dislivello_minimo: float = 0.01,
    raggio_pennello: int = 2,
    ammorbidisci: float = 0.6,
    quota_minima: float | None = None,
    maschera: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Applica l'erosione a una heightmap float e ne ritorna una nuova.

    `maschera` (True dove erodere) serve a tenere fuori l'acqua: un fondale
    marino eroso produce secche e scogli che non c'entrano niente.
    """
    base = altezze.astype(np.float32)
    h = base.copy()
    H, W = h.shape
    if H < 8 or W < 8:
        return h

    # La densita' di gocce va scalata sull'area, non fissata a un numero.
    # Misurato: 0,3 gocce per cella scolpisce un reticolo dendritico pulito;
    # a 1 goccia per cella la pianura si riempie dei cumuli lasciati dalle
    # gocce che si fermano, e il terreno peggiora invece di migliorare.
    rng = np.random.default_rng(seed)
    n = int(n_gocce) if n_gocce is not None else int(H * W * gocce_per_cella)
    if n <= 0:
        return h

    # Pennello di erosione. Concentrare lo scavo sui quattro vertici della
    # cella innesca un anello di retroazione: la cella si abbassa, la goccia
    # successiva ci cade dentro con un dislivello maggiore, scava di piu', e
    # in poche migliaia di gocce si apre un pozzo che tira giu' tutto. A 4.000
    # gocce non si vede, a 20.000 la simulazione diverge.
    # Spargendo lo scavo su un disco il problema sparisce, perche' nessuna
    # cella riceve mai l'intero contributo di una goccia.
    off_x, off_y, pesi = _pennello(raggio_pennello)

    # posizione in coordinate continue, direzione, velocita', acqua, sedimento
    px = rng.uniform(1.0, W - 2.001, n).astype(np.float32)
    py = rng.uniform(1.0, H - 2.001, n).astype(np.float32)
    if maschera is not None:
        # le gocce nascono solo dove si puo' erodere
        validi = maschera[py.astype(np.int32), px.astype(np.int32)]
        if validi.any():
            ys, xs = np.nonzero(maschera)
            k = rng.integers(0, ys.size, n)
            # Le celle valide possono stare sul bordo: senza clip la goccia
            # nasce fuori e l'interpolazione bilineare sfora l'array.
            px = np.clip(np.where(validi, px, xs[k].astype(np.float32) + 0.5),
                         1.0, W - 2.001).astype(np.float32)
            py = np.clip(np.where(validi, py, ys[k].astype(np.float32) + 0.5),
                         1.0, H - 2.001).astype(np.float32)

    dx = np.zeros(n, np.float32)
    dy = np.zeros(n, np.float32)
    vel = np.ones(n, np.float32)
    acqua = np.ones(n, np.float32)
    sed = np.zeros(n, np.float32)
    viva = np.ones(n, bool)

    delta = np.zeros(H * W, np.float32)   # variazione accumulata, piatta

    for _ in range(passi):
        if not viva.any():
            break

        x0 = px.astype(np.int32)
        y0 = py.astype(np.int32)
        fx = px - x0
        fy = py - y0

        i00 = y0 * W + x0
        i10 = i00 + 1
        i01 = i00 + W
        i11 = i01 + 1

        hp = h.ravel()
        h00, h10, h01, h11 = hp[i00], hp[i10], hp[i01], hp[i11]

        # gradiente bilineare e quota interpolata
        gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
        gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
        alt = (h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy)
               + h01 * (1 - fx) * fy + h11 * fx * fy)

        # l'inerzia impedisce alla goccia di seguire ogni increspatura:
        # senza, i solchi diventano nervosi e poco naturali
        dx = dx * inerzia - gx * (1 - inerzia)
        dy = dy * inerzia - gy * (1 - inerzia)
        # Dove non c'e' pendenza la goccia si FERMA: l'acqua ristagna, non
        # vaga a caso. Dandole una direzione casuale invece, come fa
        # l'implementazione ingenua, le gocce girovagano per la pianura
        # scavando e depositando a vuoto, e il risultato e' una piana
        # butterata di crateri. Si vede a occhio appena si alza il numero di
        # gocce: il reticolo di valli resta corretto e tutto il resto marcisce.
        ln = np.hypot(dx, dy)
        ferme = viva & (ln < pendenza_minima)
        if ferme.any():
            resto = np.where(ferme, sed, 0.0).astype(np.float32)
            if maschera is not None:
                resto = np.where(maschera.ravel()[i00], resto, 0.0).astype(np.float32)
            xf = px.astype(np.int32); yf = py.astype(np.int32)
            for ox, oy, pesoi in zip(off_x, off_y, pesi):
                np.add.at(delta,
                          np.clip(yf + oy, 0, H - 1) * W + np.clip(xf + ox, 0, W - 1),
                          resto * pesoi)
            sed = np.where(ferme, 0.0, sed).astype(np.float32)
            viva &= ~ferme
        ln = np.maximum(ln, 1e-6)
        dx /= ln
        dy /= ln

        nx = px + dx
        ny = py + dy

        # Una goccia che esce dalla mappa si porta via il sedimento che ha
        # in corpo. Su centinaia di migliaia di gocce e' una falla di massa
        # enorme: il terreno sprofonda uniformemente senza che nulla si
        # depositi da nessuna parte. Prima di ucciderla, le si fa scaricare
        # tutto dove si trova.
        fuori = (nx < 1) | (nx >= W - 2) | (ny < 1) | (ny >= H - 2)
        muore = fuori & viva
        if muore.any():
            resto = np.where(muore, sed, 0.0).astype(np.float32)
            if maschera is not None:
                resto = np.where(maschera.ravel()[i00], resto, 0.0).astype(np.float32)
            for ox, oy, pesoi in zip(off_x, off_y, pesi):
                bx = np.clip(x0 + ox, 0, W - 1)
                by = np.clip(y0 + oy, 0, H - 1)
                np.add.at(delta, by * W + bx, resto * pesoi)
            sed = np.where(muore, 0.0, sed).astype(np.float32)
        viva &= ~fuori
        nx = np.clip(nx, 1, W - 2.001)
        ny = np.clip(ny, 1, H - 2.001)

        # quota nella nuova posizione
        ax = nx.astype(np.int32); ay = ny.astype(np.int32)
        ux = nx - ax; uy = ny - ay
        j00 = ay * W + ax
        nuova = (hp[j00] * (1 - ux) * (1 - uy) + hp[j00 + 1] * ux * (1 - uy)
                 + hp[j00 + W] * (1 - ux) * uy + hp[j00 + W + 1] * ux * uy)
        # Il dislivello visto dalla goccia va limitato. Una scarpata di
        # cinquanta blocchi non fa scendere l'acqua cinquanta volte piu'
        # forte: satura. Senza questo tetto un pozzo accidentale autorizza
        # uno scavo proporzionale alla propria profondita', ed e' esattamente
        # l'anello che fa divergere la simulazione.
        dh = np.clip(nuova - alt, -dislivello_massimo, dislivello_massimo)

        # capacita' di trasporto: alta in discesa ripida e veloce
        cap = np.maximum(-dh * vel * acqua * capacita, dislivello_minimo)

        # In salita o sovraccarica: deposita (quanto > 0, il terreno sale).
        # Altrimenti erode (quanto < 0). Il tetto -dh sull'erosione e' quello
        # che evita i pozzi: una goccia non puo' scavare piu' del dislivello
        # che sta scendendo, altrimenti si fora il terreno sotto di se'.
        scarica = (sed > cap) | (dh > 0)
        deposita = np.where(dh > 0, np.minimum(dh, sed), (sed - cap) * deposito)
        erode = -np.minimum((cap - sed) * erosione, np.maximum(-dh, 0.0))
        quanto = np.where(scarica, deposita, erode).astype(np.float32)
        # Una goccia sola non sposta un masso: tetto assoluto per passo.
        np.clip(quanto, -scavo_per_passo, scavo_per_passo, out=quanto)
        quanto = np.where(viva, quanto, 0.0).astype(np.float32)
        if maschera is not None:
            quanto = np.where(maschera.ravel()[i00], quanto, 0.0).astype(np.float32)

        sed -= quanto

        # scatter ai quattro vertici, pesato: np.add.at somma le celle ripetute
        # Il deposito e' locale: il sedimento cade dove la goccia rallenta,
        # e va messo sui quattro vertici con i pesi bilineari.
        giu = np.maximum(quanto, 0.0)
        if giu.any():
            w00 = (1 - fx) * (1 - fy); w10 = fx * (1 - fy)
            w01 = (1 - fx) * fy;       w11 = fx * fy
            np.add.at(delta, i00, giu * w00)
            np.add.at(delta, i10, giu * w10)
            np.add.at(delta, i01, giu * w01)
            np.add.at(delta, i11, giu * w11)

        # Lo scavo si sparge sul pennello.
        su = np.minimum(quanto, 0.0)
        if su.any():
            for ox, oy, pw in zip(off_x, off_y, pesi):
                bx = np.clip(x0 + ox, 0, W - 1)
                by = np.clip(y0 + oy, 0, H - 1)
                np.add.at(delta, by * W + bx, su * pw)

        np.copyto(h, base + delta.reshape(H, W))

        # VELOCITA' TERMINALE. Senza questo tetto la simulazione diverge, e ci
        # mette una ventina di passi a farlo: la velocita' cresce a ogni
        # discesa, la capacita' di trasporto cresce con lei, e la goccia non
        # raggiunge mai la saturazione che la farebbe depositare. Resta
        # un'escavatrice affamata all'infinito, finche' un pozzo diventa
        # abbastanza profondo da innescare la valanga.
        # Una goccia d'acqua vera raggiunge la velocita' terminale in fretta.
        vel = np.sqrt(np.maximum(vel * vel + (-dh) * gravita, 1e-6)).astype(np.float32)
        np.minimum(vel, velocita_massima, out=vel)
        np.minimum(sed, sedimento_massimo, out=sed)
        acqua *= (1.0 - evaporazione)
        px, py = nx, ny

    # Anche a fine simulazione il sedimento ancora in sospensione va reso al
    # terreno, altrimenti il bilancio di massa resta in perdita.
    if viva.any() and sed.any():
        x0 = px.astype(np.int32); y0 = py.astype(np.int32)
        fx = px - x0; fy = py - y0
        i00 = y0 * W + x0
        resto = np.where(viva, sed, 0.0).astype(np.float32)
        if maschera is not None:
            resto = np.where(maschera[y0, x0], resto, 0.0).astype(np.float32)
        for ox, oy, pesoi in zip(off_x, off_y, pesi):
            bx = np.clip(x0 + ox, 0, W - 1)
            by = np.clip(y0 + oy, 0, H - 1)
            np.add.at(delta, by * W + bx, resto * pesoi)

    if maschera is not None:
        # Il pennello sparge su un disco: qualche peso cade inevitabilmente
        # fuori dalla maschera anche quando la goccia e' dentro. Si azzera in
        # blocco alla fine, cosi' "fuori dalla maschera non si tocca niente"
        # e' una garanzia e non un'approssimazione.
        delta *= maschera.ravel().astype(np.float32)

    def _chiudi(x):
        # Un pavimento e' indispensabile: senza, l'erosione scava pozzi sotto
        # il livello del mare in mezzo al continente, che alla generazione
        # diventano laghetti spuri sparsi ovunque.
        return x if quota_minima is None else np.maximum(x, quota_minima)

    if ammorbidisci > 0:
        # I solchi scavati da una singola goccia sono larghi un pixel. Si
        # sfoca LA VARIAZIONE, non il terreno: cosi' i canali si allargano
        # senza che la forma generale del rilievo venga spianata.
        return _chiudi(base + gaussian_filter(delta.reshape(H, W), ammorbidisci)).astype(np.float32)
    return _chiudi(base + delta.reshape(H, W)).astype(np.float32)


def _pennello(raggio: int):
    """Offset e pesi di un disco, normalizzati a somma 1."""
    raggio = max(0, int(raggio))
    ox, oy, pw = [], [], []
    for dy in range(-raggio, raggio + 1):
        for dx in range(-raggio, raggio + 1):
            d = np.hypot(dx, dy)
            if d > raggio:
                continue
            ox.append(dx); oy.append(dy); pw.append(1.0 - d / (raggio + 1.0))
    pw = np.asarray(pw, np.float32)
    return (np.asarray(ox, np.int32), np.asarray(oy, np.int32), pw / pw.sum())


def statistiche(prima: np.ndarray, dopo: np.ndarray,
                maschera: np.ndarray | None = None) -> dict:
    """Numeri per capire se l'erosione ha fatto qualcosa di sensato."""
    d = (dopo - prima).astype(np.float32)
    m = maschera if maschera is not None else np.ones(d.shape, bool)
    gp = np.hypot(*np.gradient(prima.astype(np.float32)))
    gd = np.hypot(*np.gradient(dopo.astype(np.float32)))
    return {
        "scavo_massimo": float(-d[m].min()) if m.any() else 0.0,
        "deposito_massimo": float(d[m].max()) if m.any() else 0.0,
        "spostamento_medio": float(np.abs(d[m]).mean()) if m.any() else 0.0,
        "pendenza_media_prima": float(gp[m].mean()) if m.any() else 0.0,
        "pendenza_media_dopo": float(gd[m].mean()) if m.any() else 0.0,
        "volume_netto": float(d[m].sum()) if m.any() else 0.0,
    }
