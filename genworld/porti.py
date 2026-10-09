"""Porti: un molo, qualche barca ormeggiata e un faro per gli abitati sulla costa.

Per ogni abitato con il mare a portata (`DISTANZA_MARE` celle dal bordo della
piana) si cerca il punto di costa piu' vicino e si costruisce, da li' verso il
largo, un molo di legno su pali: tre celle di larghezza, tanto lungo quanto il
fondale basso lo permette. Ai suoi fianchi si ormeggiano barche prese dal
pacchetto delle navi (le piu' piccole), col fianco lungo parallelo al molo e il
galleggiamento alla quota del mare. Le citta' murate hanno anche un faro, sulla
riva a fianco del molo.

Molo e faro sono disegni (`arredi.Disegno`) e si posano come le case isolate
(`isolate.CasaIsolata`): stessa posa, stesso indice per chunk, stessa protezione
del sottosuolo. Qui c'e' solo la scelta del posto. Le barche sono modelli gia'
nel catalogo delle navi: si registrano solo gli indici.
"""

from __future__ import annotations

import numpy as np

from .arredi import Disegno
from .isolate import CasaIsolata

DISTANZA_MARE = 40          # dal bordo della piana al mare, perche' sia un porto
MARE_MINIMO = 1500          # celle di una distesa: sotto e' un laghetto, non un mare
PIANO_MOLO = 63             # quota del tavolato: un blocco sopra il pelo dell'acqua
PALI = 5                    # altezza dei pali sotto il tavolato
LUNGHEZZA_MOLO = (8, 18)    # minima e massima
BARCHE_PER_MOLO = 3
LATO_BARCA_MAX = (26, 12)   # lunghezza e larghezza massime di una barca ormeggiata
LATO_FARO = 7

# quarti di rotazione che portano l'asse +x del molo verso (dz, dx)
_QUARTI = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}


def molo(lunghezza: int) -> Disegno:
    """Un molo lungo `lunghezza` (asse x, il lato di terra e' x=0) e largo 3.

    Pali di tronco ogni quattro celle sui due lati, tavolato di assi, una
    ringhiera di staccionata (con l'imbocco libero) e una lanterna su ogni palo.
    """
    d = Disegno("molo", lunghezza, PALI + 3, 3)
    for x in range(lunghezza):
        for z in range(3):
            d.metti(x, PALI, z, "spruce_planks")
        palo = x % 4 == 0 or x == lunghezza - 1
        if palo:
            for z in (0, 2):
                for y in range(PALI):
                    d.metti(x, y, z, "spruce_log", axis="y")
        if x >= 2:
            for z in (0, 2):
                d.metti(x, PALI + 1, z, "spruce_fence")
        if palo and x >= 2:
            for z in (0, 2):
                d.metti(x, PALI + 2, z, "lantern", hanging="false", waterlogged="false")
    return d


def faro() -> Disegno:
    """Un faro a fasce rosse e bianche, cavo, con una scaletta interna, una
    galleria con ringhiera, la lanterna di vetro con le lanterne marine e un
    tetto di mattoni. 7 x 19 x 7; l'apertura d'ingresso e' a sud (z alto)."""
    n = LATO_FARO
    c = n // 2
    d = Disegno("faro", n, 19, n)

    def r(x: int, z: int) -> float:
        return float(np.hypot(x - c, z - c))

    for y in range(14):
        raggio = 3.2 if y < 6 else 2.2
        colore = "white_concrete" if (y // 3) % 2 == 0 else "red_concrete"
        for x in range(n):
            for z in range(n):
                dist = r(x, z)
                if raggio - 1.1 < dist <= raggio:
                    # un varco d'ingresso a sud, alto due
                    if x == c and z == n - 1 and y in (0, 1):
                        continue
                    d.metti(x, y, z, colore)
    # la scaletta, addossata al muro nord: sale fino al foro della galleria
    for y in range(1, 14):
        d.metti(c, y, c - 1, "ladder", facing="south")
    # la galleria: pavimento a mezzo blocco con il foro sopra la scala, ringhiera
    for x in range(n):
        for z in range(n):
            dist = r(x, z)
            if dist <= 3.2 and (x, z) != (c, c - 1):
                d.metti(x, 14, z, "stone_brick_slab", type="bottom", waterlogged="false")
            if 2.4 < dist <= 3.2:
                d.metti(x, 15, z, "dark_oak_fence")
    # la lanterna: vetro tutto attorno, due lanterne marine al centro
    for x in range(n):
        for z in range(n):
            if 1.0 < r(x, z) <= 1.6:
                for y in (15, 16):
                    d.metti(x, y, z, "glass")
    for y in (15, 16):
        d.metti(c, y, c, "sea_lantern")
    # il tetto
    for x in range(n):
        for z in range(n):
            dist = r(x, z)
            if dist <= 2.3:
                d.metti(x, 17, z, "stone_brick_slab", type="bottom", waterlogged="false")
    d.metti(c, 18, c, "lightning_rod", facing="up", powered="false", waterlogged="false")
    return d


# --------------------------------------------------------------------------
# Dove
# --------------------------------------------------------------------------

def _mari(cls: np.ndarray, marino) -> np.ndarray:
    """Le distese di mare vero: le componenti connesse grandi."""
    from scipy.ndimage import label
    e_mare = np.isin(cls, marino)
    lab, n = label(e_mare)
    if n == 0:
        return e_mare
    grandezze = np.bincount(lab.ravel())
    grandezze[0] = 0
    return np.isin(lab, np.nonzero(grandezze >= MARE_MINIMO)[0])


def _punto_di_costa(mare: np.ndarray, z: int, x: int, raggio: int):
    """(punto di terra sulla costa, direzione verso il largo come (dz, dx)) del
    mare piu' vicino all'abitato, o None se e' oltre `DISTANZA_MARE`."""
    H, W = mare.shape
    r = raggio + DISTANZA_MARE
    za, zb, xa, xb = max(0, z - r), min(H, z + r + 1), max(0, x - r), min(W, x + r + 1)
    zz, xx = np.nonzero(mare[za:zb, xa:xb])
    if len(zz) == 0:
        return None
    zz, xx = zz + za, xx + xa
    d = np.hypot(zz - z, xx - x)
    k = int(np.argmin(d))
    if d[k] - raggio > DISTANZA_MARE:
        return None
    mz, mx = int(zz[k]), int(xx[k])
    vz, vx = mz - z, mx - x
    if abs(vx) >= abs(vz):
        dz, dx = 0, (1 if vx > 0 else -1)
    else:
        dz, dx = (1 if vz > 0 else -1), 0
    # si torna indietro verso terra finche' non si esce dal mare
    pz, px = mz, mx
    for _ in range(80):
        if not (0 <= pz - dz < H and 0 <= px - dx < W):
            return None
        pz, px = pz - dz, px - dx
        if not mare[pz, px]:
            return (pz, px), (dz, dx)
    return None


def _lunghezza_molo(h: np.ndarray, mare: np.ndarray, inizio: tuple[int, int],
                    verso: tuple[int, int], laterale: tuple[int, int]) -> int:
    """Quanto puo' essere lungo il molo, contando da `inizio` (due celle dentro
    terra): finche' il fondale e' alto abbastanza perche' i pali lo tocchino, e
    le tre celle di larghezza sono tutte mare (dopo la riva)."""
    H, W = h.shape
    base = PIANO_MOLO - PALI
    lung = 0
    in_mare = 0
    for i in range(LUNGHEZZA_MOLO[1]):
        celle = []
        for k in (-1, 0, 1):
            z = inizio[0] + verso[0] * i + laterale[0] * k
            x = inizio[1] + verso[1] * i + laterale[1] * k
            if not (0 <= z < H and 0 <= x < W):
                return lung if in_mare >= 4 else 0
            celle.append((z, x))
        centro = celle[1]
        if i >= 3:
            # conta la linea di mezzo: i lati possono essere ancora riva, ma dove
            # sono acqua il fondale non puo' essere piu' profondo dei pali
            if not mare[centro]:
                break
            if int(h[centro]) < base + 1:
                break
            if any(mare[z, x] and int(h[z, x]) < base + 1 for z, x in celle):
                break
            in_mare += 1
        elif any(int(h[z, x]) > PIANO_MOLO + 2 for z, x in celle):
            return 0                                           # la riva e' una scogliera
        lung = i + 1
    return lung if in_mare >= 4 else 0


def _ruota_punto(x: int, z: int, dx: int, dz: int, quarti: int) -> tuple[int, int]:
    """Dove va a finire la cella (x, z) di un modello dx x dz dopo `quarti` rotazioni
    (la stessa di `Catalogo.celle`: (x, z) -> (dz - 1 - z, x))."""
    for _ in range(quarti % 4):
        x, z = dz - 1 - z, x
        dx, dz = dz, dx
    return x, z


def _punti_di_costa(mare: np.ndarray, centro: tuple[int, int], raggio: int, passo: int = 3):
    """[(z, x, (dz, dx))]: celle di TERRA con il mare subito oltre, nelle quattro
    direzioni, entro `raggio` da `centro`, una ogni `passo`."""
    H, W = mare.shape
    cz, cx = centro
    za, zb = max(2, cz - raggio), min(H - 2, cz + raggio)
    xa, xb = max(2, cx - raggio), min(W - 2, cx + raggio)
    fuori = []
    for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        dz, dx = d
        sotto = ~mare[za:zb, xa:xb]
        oltre = mare[za + dz:zb + dz, xa + dx:xb + dx]
        zz, xx = np.nonzero(sotto & oltre)
        for z, x in zip(zz.tolist(), xx.tolist()):
            if (z + x) % passo == 0:
                fuori.append((za + z, xa + x, d))
    return fuori


def _molo_da_modello(m, indice: int, c, h, mare, occ, livello_mare: int):
    """Il posto migliore per un molo da schema vicino all'abitato `c`: la riva con
    il mare davanti, il lato di terra del modello girato verso terra. Ritorna
    None se nessuna riva ci sta."""
    H, W = h.shape
    costa_x, centro_z = int(m.meta["costa_x"]), int(m.meta["centro_z"])
    base_c = livello_mare - m.linea_acqua
    piena = (m.celle >= 0).any(axis=1)                            # (x, z)
    colonna = np.broadcast_to(np.arange(m.dx)[:, None], (m.dx, m.dz))
    meglio = None
    for (pz, px, (dz, dx)) in _punti_di_costa(mare, (c.z, c.x), c.raggio + DISTANZA_MARE):
        q = _QUARTI[(-dz, -dx)]
        ix, iz = m.ingombro(q)
        ax, az = _ruota_punto(costa_x, centro_z, m.dx, m.dz, q)
        x0, z0 = px - ax, pz - az
        if x0 < 1 or z0 < 1 or x0 + ix >= W or z0 + iz >= H:
            continue
        if occ[z0:z0 + iz, x0:x0 + ix].any():
            continue
        mm, pp = piena & (colonna < costa_x), piena & (colonna >= costa_x)
        for _ in range(q % 4):
            mm, pp = np.rot90(mm, k=-1), np.rot90(pp, k=-1)    # come `Catalogo.celle`
        mm, pp = mm.T, pp.T                                    # (x, z) -> (z, x), come le mappe
        if not mm.any() or not pp.any():
            continue
        reg_mare = mare[z0:z0 + iz, x0:x0 + ix]
        reg_h = h[z0:z0 + iz, x0:x0 + ix]
        sul_mare = mm & reg_mare
        frazione_mare = float(reg_mare[mm].mean())
        frazione_terra = float((~reg_mare)[pp].mean())
        fondo_ok = float((reg_h[sul_mare] >= base_c - 4).mean()) if sul_mare.any() else 0.0
        if frazione_mare < 0.8 or frazione_terra < 0.5 or fondo_ok < 0.85:
            continue
        punteggio = (frazione_mare + frazione_terra + fondo_ok
                     - 0.002 * float(np.hypot(pz - c.z, px - c.x)))
        if meglio is None or punteggio > meglio[0]:
            meglio = (punteggio, x0, z0, ix, iz, q, px, pz, dz, dx)
    if meglio is None:
        return None
    _, x0, z0, ix, iz, q, px, pz, dz, dx = meglio
    return (CasaIsolata(x=x0, z=z0, larghezza=ix, profondita=iz, base=base_c + m.affondo,
                        modello=indice, quarti=q, stile="prato"), (pz, px, (dz, dx)))


def _barche_al_largo(costa, m, barche: list, primo_barche: int, piccole: list,
                     mare: np.ndarray, h: np.ndarray, occ: np.ndarray, livello_mare: int,
                     rng, quante: int = BARCHE_PER_MOLO) -> list[CasaIsolata]:
    """Barche ormeggiate sul lato del largo di un molo da schema, col lato lungo
    parallelo alla riva."""
    H, W = h.shape
    if not piccole:
        return []
    pz, px, (dz, dx) = costa
    lungo_riva = (1, 0) if dz == 0 else (0, 1)                # (dz, dx) lungo la riva
    fuori: list[CasaIsolata] = []
    s = int(m.meta["costa_x"]) + 3                            # dal punto di costa al largo
    for lato in (0, 1, -1):
        if len(fuori) >= quante:
            break
        k = piccole[int(rng.integers(0, len(piccole)))]
        b = barche[k]
        cand = [q for q in range(4) if (b.ingombro(q)[0] >= b.ingombro(q)[1]) == (dz != 0)]
        q = cand[int(rng.integers(0, len(cand)))]
        bx, bz = b.ingombro(q)
        verso_mare = bx if dz == 0 else bz                    # estensione lungo il mare
        for extra in (0, 3, 6):
            centro_z = pz + dz * (s + verso_mare // 2 + extra) + lungo_riva[0] * lato * 12
            centro_x = px + dx * (s + verso_mare // 2 + extra) + lungo_riva[1] * lato * 12
            bx0, bz0 = centro_x - bx // 2, centro_z - bz // 2
            if bx0 < 1 or bz0 < 1 or bx0 + bx >= W or bz0 + bz >= H:
                continue
            base_b = livello_mare - b.linea_acqua
            sub = mare[bz0 - 1:bz0 + bz + 1, bx0 - 1:bx0 + bx + 1]
            prof = h[bz0 - 1:bz0 + bz + 1, bx0 - 1:bx0 + bx + 1]
            if not sub.all() or (prof > base_b).any():
                continue
            if occ[bz0 - 1:bz0 + bz + 1, bx0 - 1:bx0 + bx + 1].any():
                continue
            fuori.append(CasaIsolata(x=bx0, z=bz0, larghezza=bx, profondita=bz, base=base_b,
                                     modello=primo_barche + k, quarti=q, stile="prato"))
            occ[bz0 - 1:bz0 + bz + 1, bx0 - 1:bx0 + bx + 1] = True
            break
    return fuori


def _faro_da_modello(m, indice: int, citta: list, h, mare, occ, livello_mare: int):
    """Un faro da schema sulla riva piu' piana vicino a un abitato: tutta
    l'impronta su terra, poco rilievo (l'altura e' del modello), il mare entro
    una ventina di celle."""
    from scipy.ndimage import distance_transform_edt
    H, W = h.shape
    dist_mare = distance_transform_edt(~mare)
    meglio = None
    for c in citta:
        for (pz, px, (dz, dx)) in _punti_di_costa(mare, (c.z, c.x), c.raggio + DISTANZA_MARE, passo=7):
            for q in range(4):
                ix, iz = m.ingombro(q)
                cz_, cx_ = pz - dz * (iz // 2 + 4), px - dx * (ix // 2 + 4)
                x0, z0 = cx_ - ix // 2, cz_ - iz // 2
                if x0 < 1 or z0 < 1 or x0 + ix >= W or z0 + iz >= H:
                    continue
                if mare[z0:z0 + iz, x0:x0 + ix].any() or occ[z0 - 1:z0 + iz + 1, x0 - 1:x0 + ix + 1].any():
                    continue
                fp = h[z0:z0 + iz, x0:x0 + ix]
                lo, hi = np.percentile(fp, (5, 95))
                base = int(np.median(fp))
                if hi - lo > 5 or base <= livello_mare:
                    continue
                vicino = float(dist_mare[z0:z0 + iz, x0:x0 + ix].min())
                if vicino > 20:
                    continue
                punteggio = -float(hi - lo) - 0.05 * vicino
                if meglio is None or punteggio > meglio[0]:
                    meglio = (punteggio, x0, z0, ix, iz, q, base)
    if meglio is None:
        return None
    _, x0, z0, ix, iz, q, base = meglio
    return CasaIsolata(x=x0, z=z0, larghezza=ix, profondita=iz, base=base,
                       modello=indice, quarti=q, stile="prato")


def pianifica(citta: list, cls: np.ndarray, h: np.ndarray, marino, livello_mare: int,
              evita: np.ndarray, barche: list, primo_barche: int, primo_extra: int,
              ver, seed: int = 0, con_faro: bool = True, *, moli: list = (),
              primo_molo: int = 0, fari: list = (), primo_faro: int = 0
              ) -> tuple[list[CasaIsolata], list]:
    """Molo, barche e faro per gli abitati sulla costa.

    Ritorna (pezzi da posare, modelli nuovi). I modelli nuovi (il molo, il faro)
    vanno in coda al catalogo, dopo le navi: `primo_extra` e' il loro primo
    indice; le barche sono modelli gia' presenti, `barche` con indici da
    `primo_barche`.
    """
    H, W = h.shape
    scarti = pianifica.scarti = {"bordo": 0, "mare": 0, "fondo": 0, "occupato": 0}
    rng = np.random.default_rng(seed * 4217 + 7)
    mare = _mari(cls, marino)
    occ = evita.copy()
    pezzi: list[CasaIsolata] = []
    extra: list = []
    indice_faro: int | None = None
    piccole = [k for k, m in enumerate(barche)
               if max(m.dx, m.dz) <= LATO_BARCA_MAX[0] and min(m.dx, m.dz) <= LATO_BARCA_MAX[1]
               and m.linea_acqua >= 1]
    if moli or fari:
        # molo e faro da schema (vedi `monumenti.carica_porti`): il posto lo
        # sceglie la forma della costa, il disegno e' quello dell'autore
        r = rng
        for c in citta:
            if not moli:
                break
            k = int(r.integers(0, len(moli)))
            trovato = _molo_da_modello(moli[k], primo_molo + k, c, h, mare, occ, livello_mare)
            if trovato is None:
                continue
            pezzo_molo, costa = trovato
            pezzi.append(pezzo_molo)
            occ[pezzo_molo.z - 1:pezzo_molo.z + pezzo_molo.profondita + 1,
                pezzo_molo.x - 1:pezzo_molo.x + pezzo_molo.larghezza + 1] = True
            pezzi.extend(_barche_al_largo(costa, moli[k], barche, primo_barche, piccole, mare,
                                          h, occ, livello_mare, r))
        if fari:
            k = int(r.integers(0, len(fari)))
            faro_t = _faro_da_modello(fari[k], primo_faro + k, citta, h, mare, occ, livello_mare)
            if faro_t is not None:
                pezzi.append(faro_t)
                occ[faro_t.z - 1:faro_t.z + faro_t.profondita + 1,
                    faro_t.x - 1:faro_t.x + faro_t.larghezza + 1] = True
        if moli and fari:
            return pezzi, extra
        con_faro = con_faro and not fari
    for c in citta:
        if moli:
            break                                  # i moli sono gia' fatti sopra
        costa = _punto_di_costa(mare, c.z, c.x, c.raggio)
        if costa is None:
            continue
        (cz, cx), (dz, dx) = costa
        laterale = (abs(dx), abs(dz))                          # (dz, dx) del lato, perpendicolare al molo
        inizio = (cz - 2 * dz, cx - 2 * dx)
        lung = _lunghezza_molo(h, mare, inizio, (dz, dx), laterale)
        if lung < LUNGHEZZA_MOLO[0]:
            continue
        # impronta del molo, girata come il modello
        quarti = _QUARTI[(dz, dx)]
        if dx != 0:
            ix, iz = lung, 3
            z0 = inizio[0] - 1
            x0 = inizio[1] if dx > 0 else inizio[1] - (lung - 1)
        else:
            ix, iz = 3, lung
            x0 = inizio[1] - 1
            z0 = inizio[0] if dz > 0 else inizio[0] - (lung - 1)
        if x0 < 1 or z0 < 1 or x0 + ix >= W or z0 + iz >= H:
            continue
        if occ[z0 - 1:z0 + iz + 1, x0 - 1:x0 + ix + 1].any():
            continue
        modello_molo = molo(lung).modello(ver)
        extra.append(modello_molo)
        pezzi.append(CasaIsolata(x=x0, z=z0, larghezza=ix, profondita=iz,
                                 base=PIANO_MOLO - PALI, modello=primo_extra + len(extra) - 1,
                                 quarti=quarti, stile="prato"))
        occ[z0 - 1:z0 + iz + 1, x0 - 1:x0 + ix + 1] = True
        # le barche: ai due lati, col lato lungo lungo il molo
        if piccole:
            ormeggiate = 0
            for lato in (1, -1):
                k = piccole[int(rng.integers(0, len(piccole)))]
                m = barche[k]
                candidati = [q for q in range(4)
                             if (m.ingombro(q)[0] >= m.ingombro(q)[1]) == (dz == 0)]
                q = candidati[int(rng.integers(0, len(candidati)))]
                bx, bz = m.ingombro(q)
                lungo_barca = bx if dz == 0 else bz
                # la barca sta dove c'e' mare e fondale: si prova lungo il molo,
                # a partire da dopo la riva, e oltre la punta se serve
                for lungo_molo in range(3 + lungo_barca // 2 + 1, lung + lungo_barca // 2 + 6, 2):
                    if ormeggiate >= BARCHE_PER_MOLO:
                        break
                    # il centro della barca sta a `lungo_molo` lungo l'asse del molo,
                    # a due celle dal bordo (tre di molo + una di intervallo)
                    centro_z = inizio[0] + dz * lungo_molo + laterale[0] * lato * (3 + (bz if dz == 0 else bx) // 2 + 1)
                    centro_x = inizio[1] + dx * lungo_molo + laterale[1] * lato * (3 + (bx if dx == 0 else bz) // 2 + 1)
                    bx0, bz0 = centro_x - bx // 2, centro_z - bz // 2
                    if bx0 < 1 or bz0 < 1 or bx0 + bx >= W or bz0 + bz >= H:
                        scarti["bordo"] += 1
                        continue
                    base_b = livello_mare - m.linea_acqua
                    sub = mare[bz0 - 1:bz0 + bz + 1, bx0 - 1:bx0 + bx + 1]
                    prof = h[bz0 - 1:bz0 + bz + 1, bx0 - 1:bx0 + bx + 1]
                    if not sub.all():
                        scarti["mare"] += 1
                        continue
                    if (prof > base_b).any():
                        scarti["fondo"] += 1
                        continue
                    if occ[bz0 - 1:bz0 + bz + 1, bx0 - 1:bx0 + bx + 1].any():
                        scarti["occupato"] += 1
                        continue
                    pezzi.append(CasaIsolata(x=bx0, z=bz0, larghezza=bx, profondita=bz,
                                             base=base_b, modello=primo_barche + k,
                                             quarti=q, stile="prato"))
                    occ[bz0 - 1:bz0 + bz + 1, bx0 - 1:bx0 + bx + 1] = True
                    ormeggiate += 1
                    break
        # il faro: sulla riva a fianco del molo, nelle citta' murate
        if con_faro and c.murata:
            posto = [(lato, dist) for dist in (11, 15, 20, 26) for lato in (1, -1)]
            for lato, dist in posto:
                # a fianco del molo e un po' dentro terra: il piede nel mare no
                fz = cz + laterale[0] * lato * dist - dz * 5 - LATO_FARO // 2
                fx = cx + laterale[1] * lato * dist - dx * 5 - LATO_FARO // 2
                if fx < 1 or fz < 1 or fx + LATO_FARO >= W or fz + LATO_FARO >= H:
                    continue
                area = (slice(fz, fz + LATO_FARO), slice(fx, fx + LATO_FARO))
                if mare[area].any() or occ[fz - 1:fz + LATO_FARO + 1, fx - 1:fx + LATO_FARO + 1].any():
                    continue
                quote = h[area]
                if int(quote.max() - quote.min()) > 3 or int(np.median(quote)) <= livello_mare:
                    continue
                if indice_faro is None:
                    extra.append(faro().modello(ver))
                    indice_faro = primo_extra + len(extra) - 1
                pezzi.append(CasaIsolata(x=fx, z=fz, larghezza=LATO_FARO, profondita=LATO_FARO,
                                         base=int(np.median(quote)), modello=indice_faro,
                                         quarti=0, stile="prato"))
                occ[fz - 1:fz + LATO_FARO + 1, fx - 1:fx + LATO_FARO + 1] = True
                break
    return pezzi, extra
