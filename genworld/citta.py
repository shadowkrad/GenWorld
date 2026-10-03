"""Citta': pianta regolare, mura, fosso, isolati, lotti sulla strada.

Una citta' e' il contrario di un mucchio di case: prima esiste il **vuoto** -
piazza, assi, circonvallazione - e le case vengono dopo, appoggiate a quel
vuoto, ognuna con la facciata verso la strada che la serve.

La pianta e' **regolare**: un quadrato con due assi che si incrociano nella
piazza e, nelle citta', una cinta con quattro porte, il fosso e un ponte per
ogni porta. Una versione precedente, a maglia organica (punti sparsi,
triangolati), dava isolati di ogni forma ma anche citta' "incasinate e non
ordinate" - cosi' le definiva chi le guardava in gioco - con mura che
seguivano un contorno a caso.

E soprattutto **il terreno si adatta alla citta', non il contrario**: dove la
pianta passa sopra un fiume o una collina il terreno viene spianato e il fiume
sparisce; la citta' non si deforma per rispettare l'ambiente. Le copie di
`cls` e `livello` che `pianifica` restituisce sono la mappa dopo questo
cambiamento.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import edifici as E
from .mappa import (ACQUA, CRATERE, FIUME, FORESTA, MARINO, MONTAGNA, NEVE,
                    PIANURA, PRATERIA, SPIAGGIA, VULCANO)

ABITABILI = (PIANURA, PRATERIA, FORESTA, SPIAGGIA)

# larghezza della carreggiata per rango
VICOLO, SECONDARIA, ASSE = 1, 2, 3
LARGHEZZA = {VICOLO: 1, SECONDARIA: 2, ASSE: 3}

# oltre questo raggio un insediamento e' una citta': ha mura e piazza vera
RAGGIO_CITTA = 30

# Chi sta dove. Una citta' non e' un dormitorio: le botteghe stanno sulla
# piazza e sulle vie principali, gli artigiani rumorosi e puzzolenti un po'
# piu' in la', e chi lavora la terra o il pesce sta al bordo, vicino a quello
# che lavora. E' la ragione per cui i mestieri hanno dato i nomi alle vie di
# mezza Europa.
BOTTEGHE_CENTRO = ("libraio", "cartografo", "speziale", "fruttivendolo",
                   "macellaio", "fabbro")
BOTTEGHE_MEDIO = ("armaiolo", "corazzaio", "falegname", "scalpellino",
                  "conciatore", "pastore")
BOTTEGHE_BORDO = ("fruttivendolo", "pastore", "falegname")


def _mestiere(t: float, sull_acqua: bool, rng: np.random.Generator) -> str:
    """Il mestiere di una casa, dedotto da dove sta.

    Ritorna "" per una semplice abitazione: una citta' fatta di sole botteghe
    e' un centro commerciale, non una citta'.
    """
    if sull_acqua:
        return "pescatore"
    if t < 0.3:
        quota, elenco = 0.55, BOTTEGHE_CENTRO
    elif t < 0.65:
        quota, elenco = 0.35, BOTTEGHE_MEDIO
    else:
        quota, elenco = 0.20, BOTTEGHE_BORDO
    if rng.random() > quota:
        return ""
    return elenco[int(rng.integers(0, len(elenco)))]


@dataclass
class Citta:
    z: int
    x: int
    raggio: int
    piazza: tuple[int, int]
    porte: list[tuple[int, int]] = field(default_factory=list)
    murata: bool = False
    edifici: int = 0

    @property
    def e_citta(self) -> bool:
        return self.murata


# --------------------------------------------------------------------------
# Pianta regolare
# --------------------------------------------------------------------------
#
# Una citta' si progetta PRIMA e il mondo si adatta, non il contrario: dove
# la pianta passa sopra un fiume, una collina o un laghetto, il terreno viene
# spianato e il fiume sparisce sotto la citta'. E' il verso giusto - chi
# costruisce una citta' non la deforma per rispettare un ruscello - e toglie
# alla radice tutta la famiglia di difetti nata dal verso opposto (case che
# scartano lotti in riva all'acqua, cinte a pezzi, sponde da ripuntellare).
#
# La pianta e' un QUADRATO con lo stesso schema ovunque:
#
#   * due assi larghi tre celle che si incrociano nella piazza (11x11);
#   * se ci sono le mura, una via di circonvallazione a ridosso della cinta;
#   * i lotti riempiono gli isolati che i due assi e la circonvallazione
#     lasciano liberi, con la facciata verso la strada.
#
# Una citta' ha in piu' la cinta (tre celle di spessore, torri agli angoli, ai
# lati delle porte e a meta' dei lati), il fosso e il ponte: gli assi escono
# dalle quattro porte, attraversano il fosso e li' diventano ponte. Un
# villaggio ha la stessa pianta, piu' piccola, senza cinta.

SPESSORE_MURA = 3
LARGHEZZA_FOSSO = 3
PROFONDITA_FOSSO = 3          # di quanto il fondo sta sotto la piana
MARGINE_RACCORDO = 10         # fascia in cui la piana sfuma nel terreno vero
SALTO_MASSIMO = 20            # dislivello (5-95 percentile) oltre cui il sito e' troppo aspro
MARINO_MASSIMO = 0.12         # quota di mare ammessa dentro la piana
SPOSTAMENTO_MASSIMO = 60      # di quanto un sito puo' spostarsi per trovare posto
PIAZZA = 5                    # mezzo lato della piazza
MIN_LOTTI_VILLAGGIO = 4       # sulle mappe grandi i lotti sono 12 e 16: vedi `lotti`
MIN_LOTTI_CITTA = 4


LATI_VILLAGGIO = (38, 30, 24, 20)
LATI_CITTA = (46, 38, 30)


def piante_possibili(raggio: int) -> list[tuple[int, bool]]:
    """(mezzo lato dell'abitato, murata) che un sito di raggio dato puo'
    avere, dalla migliore. Un sito da citta' prova le misure da citta' e,
    se la mappa non ha posto, ripiega su un villaggio: meno case e niente
    mura, ma un abitato. Lo stesso vale, in piccolo, per un sito da villaggio.
    """
    villaggio = [(lato, False) for lato in LATI_VILLAGGIO]
    if raggio >= RAGGIO_CITTA:
        return [(lato, True) for lato in LATI_CITTA] + villaggio
    return villaggio


def _distanza(forma: tuple[int, int], z: int, x: int):
    """Distanza di Chebyshev dal centro (i quadrati sono le sue curve di livello)
    e le due coordinate relative."""
    zz, xx = np.ogrid[:forma[0], :forma[1]]
    dz = np.broadcast_to(zz - z, forma)
    dx = np.broadcast_to(xx - x, forma)
    return np.maximum(np.abs(dz), np.abs(dx)), dz, dx


def _raggio_piana(lato: int, murata: bool) -> int:
    """Fin dove arriva la piana: con le mura, cinta + berma + fosso + argine."""
    if murata:
        return lato + SPESSORE_MURA + 1 + LARGHEZZA_FOSSO + 1
    return lato + 3


def _valuta(cls, h, urbano, z, x, raggio_piana, livello_mare) -> tuple[int, float] | None:
    """(quota di spianamento, costo) di una piana centrata in (z, x), o None.

    Il terreno si piega alla citta', ma non senza limiti: dentro il quadrato
    non ci puo' essere troppo mare (non si interra un golfo), niente vulcano,
    ne' un dislivello da montagna, ne' un'altra citta'. Il costo e' quanto
    mare e quanto dislivello bisogna sistemare: a parita' di sito, meno e'
    meglio. Si lavora su un ritaglio, perche' se ne valutano centinaia.
    """
    H, W = cls.shape
    s = raggio_piana + MARGINE_RACCORDO + 2
    if z - s < 0 or x - s < 0 or z + s >= H or x + s >= W:
        return None
    fetta = (slice(z - s, z + s + 1), slice(x - s, x + s + 1))
    c, q, u = cls[fetta], h[fetta], urbano[fetta]
    D = _distanza(c.shape, s, s)[0]
    zona = D <= raggio_piana
    if np.isin(c, (VULCANO, CRATERE))[D <= raggio_piana + MARGINE_RACCORDO].any():
        return None
    if u[D <= raggio_piana + MARGINE_RACCORDO].any():
        return None
    mare = float(np.isin(c, MARINO)[zona].mean())
    if mare > MARINO_MASSIMO:
        return None
    terra = zona & ~np.isin(c, ACQUA)
    if terra.sum() < zona.sum() * 0.5:
        return None
    quote = q[terra]
    lo, hi = np.percentile(quote, (5, 95))
    if hi - lo > SALTO_MASSIMO:
        return None
    base = max(int(np.median(quote)), livello_mare + PROFONDITA_FOSSO + 2)
    return base, mare * 200.0 + float(hi - lo)


def _cerca_sito(cls, h, urbano, z, x, raggio_piana, livello_mare):
    """Il posto migliore per la piana vicino al sito scelto: (z, x, quota).

    I siti sono scelti vicino all'acqua, quindi spesso a ridosso di una costa
    o di un lago. Invece di scartarli si prova a spostarli: la citta' cerca il
    suo posto, poi lo spiana.
    """
    meglio = None
    for oz in range(-SPOSTAMENTO_MASSIMO, SPOSTAMENTO_MASSIMO + 1, 4):
        for ox in range(-SPOSTAMENTO_MASSIMO, SPOSTAMENTO_MASSIMO + 1, 4):
            v = _valuta(cls, h, urbano, z + oz, x + ox, raggio_piana, livello_mare)
            if v is None:
                continue
            costo = v[1] + 0.15 * float(np.hypot(oz, ox))
            if meglio is None or costo < meglio[0]:
                meglio = (costo, z + oz, x + ox, v[0])
    return None if meglio is None else meglio[1:]


def _spiana(h, cls, livello, cls_originale, D, lato, murata, base, livello_mare):
    """Spiana la piana a `base`, scava il fosso, sfuma il raccordo.

    Opera sulle copie di lavoro `h`, `cls`, `livello`. L'acqua dentro la piana
    sparisce (diventa pianura): e' il terreno che si adatta alla citta'.
    """
    r = _raggio_piana(lato, murata)
    dentro = D <= r
    h[dentro] = base
    cls[dentro] = PIANURA
    livello[dentro] = livello_mare

    # il raccordo: la piana sfuma nel terreno vero senza gradini. L'acqua che
    # sta fuori resta com'e' (il suo letto non si alza).
    fascia = (D > r) & (D <= r + MARGINE_RACCORDO) & ~np.isin(cls_originale, ACQUA)
    t = np.clip(1.0 - (D[fascia] - r) / MARGINE_RACCORDO, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    h[fascia] = np.round(h[fascia] * (1.0 - t) + base * t).astype(h.dtype)

    fosso = np.zeros_like(dentro)
    if murata:
        primo = lato + SPESSORE_MURA + 2
        fosso = (D >= primo) & (D < primo + LARGHEZZA_FOSSO)
        h[fosso] = base - PROFONDITA_FOSSO
        cls[fosso] = FIUME
        livello[fosso] = base - 2
    return fosso


def _strade(D, dz, dx, lato, murata) -> tuple[np.ndarray, np.ndarray]:
    """Le vie dell'abitato (rango per cella) e il prolungamento degli assi fuori
    dalla cinta, che passa il fosso e arriva alla campagna."""
    rango = np.zeros(D.shape, np.uint8)
    dentro = D <= lato - 1
    asse = (np.abs(dz) <= 1) | (np.abs(dx) <= 1)
    rango[asse & dentro] = ASSE
    if murata:
        giro = ((D == lato - 3) | (D == lato - 4)) & (rango == 0)
        rango[giro] = SECONDARIA
    rango[D <= PIAZZA] = ASSE
    fuori = lato + (10 if murata else 2)
    esterno = asse & (D > lato - 1) & (D <= fuori)
    return rango, esterno


def _mura(D, dz, dx, lato) -> np.ndarray:
    """Cinta quadrata: 1 muro, 2 porta, 3 torre."""
    anello = (D >= lato + 1) & (D <= lato + SPESSORE_MURA)
    muro = np.zeros(D.shape, np.uint8)
    muro[anello] = 1
    porta = anello & ((np.abs(dz) <= 1) | (np.abs(dx) <= 1))
    lungo = np.minimum(np.abs(dz), np.abs(dx))        # posizione lungo il lato
    torre = (np.abs(dz) >= lato - 1) & (np.abs(dx) >= lato - 1)        # angoli
    torre |= (lungo >= 2) & (lungo <= 4)                                # accanto alle porte
    torre |= np.abs(lungo - lato // 2) <= 1                             # a meta' lato
    muro[anello & torre] = 3
    muro[porta] = 2
    return muro


def _porte_esterne(z: int, x: int, lato: int, murata: bool,
                   forma: tuple[int, int]) -> list[tuple[int, int]]:
    """I quattro punti, in campagna, da cui partono le strade fra gli abitati."""
    d = lato + (10 if murata else 2)
    punti = [(z - d, x), (z + d, x), (z, x - d), (z, x + d)]
    return [(pz, px) for pz, px in punti if 0 <= pz < forma[0] and 0 <= px < forma[1]]


# --------------------------------------------------------------------------
# Lotti
# --------------------------------------------------------------------------

# Un isolato e' il quadrato che gli assi e la circonvallazione lasciano libero
# in ciascun quadrante. Lo si divide in quattro lotti, separati da due celle
# (le gronde di due case vicine non si toccano), e ogni lotto guarda la strada
# a cui sta appoggiato. Lasciare che la misura dei lotti la decida l'isolato,
# come faceva la pianta organica, dava case di ogni taglia ma quasi mai della
# taglia dei modelli scaricati (misurato: con lotti da 16x16 e gronda entrano
# 20 modelli diversi su 40, con 13x13 solo 10), quindi qui e' la pianta a
# partire dalla misura del lotto, non il contrario.
MARCIAPIEDE = 4        # dal centro dell'asse al primo lotto
STACCO = 2             # fra due lotti
SGOMBRO_PIAZZA = 7     # il lotto d'angolo parte oltre la piazza
LOTTO_MINIMO = 11      # sotto questa misura l'isolato non si divide


def lotti(cls: np.ndarray, h: np.ndarray, livello: np.ndarray,
          centro: tuple[int, int], lato: int, murata: bool, livello_mare: int,
          rng: np.random.Generator) -> list[E.Edificio]:
    """Quattro lotti per quadrante (tre in un villaggio: l'ultimo non
    toccherebbe nessuna strada), con la facciata verso la strada vicina."""
    cz, cx = centro
    w = (lato - 9 - STACCO) // 2
    # un isolato troppo piccolo per quattro case ne regge una sola
    if w < LOTTO_MINIMO:
        w, divisioni = lato - 9, (0,)
    else:
        divisioni = (0, 1)
    fuori: list[E.Edificio] = []
    for sv in (1, -1):
        for su in (1, -1):
            for b in divisioni:
                for a in divisioni:
                    if a == 1 and b == 1 and not murata:
                        continue
                    u0 = MARCIAPIEDE + a * (w + STACCO)
                    v0 = MARCIAPIEDE + b * (w + STACCO)
                    ul = vl = w
                    if a == 0 and b == 0:
                        u0, ul = SGOMBRO_PIAZZA, w - (SGOMBRO_PIAZZA - MARCIAPIEDE)
                    if a == 1 and b == 1:
                        verso = E.EST if su > 0 else E.OVEST
                    elif b == 0:
                        verso = E.NORD if sv > 0 else E.SUD
                    else:
                        verso = E.OVEST if su > 0 else E.EST
                    x0 = cx + u0 if su > 0 else cx - u0 - ul + 1
                    z0 = cz + v0 if sv > 0 else cz - v0 - vl + 1
                    t = float(np.hypot(u0 + ul / 2, v0 + vl / 2)) / (lato * 1.25)
                    ed = _fabbrica(cls, h, livello, x0, z0, ul, vl, verso, t,
                                   livello_mare, rng, 1, _mestiere(t, False, rng))
                    if ed is not None:
                        fuori.append(ed)
    return fuori


def _fabbrica(cls, h, livello, x0, z0, larg, prof, verso, t,
              livello_mare, rng, gronda: int = 1,
              mestiere: str = "") -> E.Edificio | None:
    fetta_c = cls[z0:z0 + prof, x0:x0 + larg]
    fetta_h = h[z0:z0 + prof, x0:x0 + larg]
    if fetta_c.size == 0 or np.isin(fetta_c, ACQUA).any():
        return None
    # ...e nemmeno proprio sul filo dell'acqua. Una cella di rispetto basta:
    # il raccordo del lotto arriva piu' in la', ma le sponde sono nella
    # maschera `intoccabile` e il raccordo non le tocca. Pretenderne tre
    # faceva sparire un terzo delle case di una citta' di fiume.
    if np.isin(cls[max(0, z0 - 1):z0 + prof + 1,
                   max(0, x0 - 1):x0 + larg + 1], ACQUA).any():
        return None
    # Dislivello ROBUSTO, non massimo meno minimo: il terreno porta addosso
    # il dettaglio frattale, e un solo pixel fuori posto bocciava il lotto.
    # Il sedime viene spianato comunque; quello che va evitato e' la casa sul
    # dirupo, non la casa sul dosso.
    lo, hi = np.percentile(fetta_h, (10, 90))
    dislivello = float(hi - lo)
    if dislivello > 7:
        return None
    base = int(np.median(fetta_h))
    if base <= livello_mare:
        return None
    classe = int(np.bincount(fetta_c.ravel()).argmax())
    if classe in (MONTAGNA, NEVE) and dislivello > 5:
        return None
    # In centro si costruisce alto, in periferia no. Il terzo piano e' raro:
    # in una citta' medievale e' un'eccezione, e messo a meta' delle case del
    # centro trasformava i vicoli in trincee.
    piani = 3 if t < 0.18 and rng.random() < 0.3 else (2 if t < 0.55 else 1)
    return E.Edificio(
        x=x0, z=z0, larghezza=larg, profondita=prof, base=base,
        piani=piani, porta=verso,
        stile=E.PALETTE_PER_CLASSE.get(classe, "prato"),
        palafitta=False, fondale=base, gronda=gronda, mestiere=mestiere,
        seme=int(rng.integers(0, 2 ** 31 - 1)),
    )


# --------------------------------------------------------------------------
# Mercato
# --------------------------------------------------------------------------

MERCI = ("frutta", "carne", "pesce", "verdura")


def mercato(centro: tuple[int, int], lato: int, murata: bool, h: np.ndarray,
            rng: np.random.Generator) -> list[E.Banco]:
    """Banchi sulla piazza, agli angoli, rivolti verso il centro.

    Il banco sta SUL selciato (la piazza e' il posto del mercato) e agli
    angoli, fuori dagli assi, che devono restare liberi. Una citta' ne ha
    quattro, un villaggio due (sulla diagonale), un borgo piccolo nessuno.
    """
    if lato < 30:
        return []
    cz, cx = centro
    angoli = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
    quanti = 4 if murata else 2
    fuori: list[E.Banco] = []
    for k, (sv, su) in enumerate(angoli[:quanti]):
        bz, bx = cz + sv * 3 - 1, cx + su * 3 - 1            # 3x3 centrato a (+-3, +-3)
        verso = E.OVEST if su > 0 else E.EST
        fuori.append(E.Banco(x=bx, z=bz, base=int(h[bz:bz + 3, bx:bx + 3].min()),
                             verso=verso, merce=MERCI[k % len(MERCI)],
                             seme=int(rng.integers(0, 2 ** 31 - 1))))
    return fuori


def scarpate(h: np.ndarray, urbano: np.ndarray, salto: int = 2) -> np.ndarray:
    """Le celle che formano la fronte di un gradone.

    Servono al motore per vestirle di pietra invece che di erba: un muro di
    sostegno si vede, un taglio di terra nuda sembra un difetto.
    """
    hh = h.astype(np.int32)
    piu_basso = np.minimum.reduce([
        np.roll(hh, 1, 0), np.roll(hh, -1, 0),
        np.roll(hh, 1, 1), np.roll(hh, -1, 1),
    ])
    return urbano & ((hh - piu_basso) >= salto)


def pianifica(
    cls: np.ndarray,
    altezze: np.ndarray,
    livello: np.ndarray,
    siti: list[tuple[int, int, int]],
    livello_mare: int = 62,
    seed: int = 0,
) -> tuple[list[E.Edificio], np.ndarray, np.ndarray, np.ndarray,
           list[Citta], list[E.Banco], np.ndarray, np.ndarray, np.ndarray]:
    """Progetta gli insediamenti.

    Ritorna (edifici, altezze spianate, vie, mura, citta', banchi, urbano,
    classi, livello dell'acqua). Le ultime due sono le COPIE della mappa dopo
    che le citta' l'hanno cambiata: dove una pianta passa sopra un fiume il
    fiume non c'e' piu', e dove c'e' il fosso c'e' acqua che sulla mappa non
    c'era. Chi disegna il mondo deve leggere quelle, non le originali.

    Un sito che non si presta (troppo mare, vulcano, terreno da montagna,
    un'altra citta' accanto) o che non produce abbastanza lotti viene
    scartato per intero: un abitato di due case non e' un abitato.
    """
    h = altezze.astype(np.int32).copy()
    cls = cls.copy()
    livello = livello.astype(np.float32).copy()
    H, W = cls.shape
    vie = np.zeros((H, W), np.uint8)
    muro = np.zeros((H, W), np.uint8)
    urbano = np.zeros((H, W), bool)
    fuori: list[E.Edificio] = []
    citta: list[Citta] = []
    banchi: list[E.Banco] = []

    for n, (sz, sx, raggio) in enumerate(siti):
        rng = np.random.default_rng(seed * 977 + n)
        for lato, murata in piante_possibili(raggio):
            posto = _cerca_sito(cls, h, urbano, sz, sx, _raggio_piana(lato, murata),
                                livello_mare)
            if posto is not None:
                break
        else:
            continue
        sz, sx, base = posto
        D, dz, dx = _distanza((H, W), sz, sx)

        # tutto si prova su copie: se il sito non da' abbastanza lotti non
        # deve lasciare traccia (una piana spianata e un fosso senza citta')
        h_t, cls_t, liv_t = h.copy(), cls.copy(), livello.copy()
        _spiana(h_t, cls_t, liv_t, cls, D, lato, murata, base, livello_mare)
        rango, esterno = _strade(D, dz, dx, lato, murata)
        ed = lotti(cls_t, h_t, liv_t, (sz, sx), lato, murata, livello_mare, rng)
        if len(ed) < (MIN_LOTTI_CITTA if murata else MIN_LOTTI_VILLAGGIO):
            continue
        for e in ed:
            e.villaggio = n

        b = mercato((sz, sx), lato, murata, h_t, rng)

        h, cls, livello = h_t, cls_t, liv_t
        vie = np.maximum(vie, rango)
        vie[esterno] = ASSE
        urbano |= D <= lato
        fuori.extend(ed)
        banchi.extend(b)
        if murata:
            m = _mura(D, dz, dx, lato)
            muro[m > 0] = m[m > 0]
        citta.append(Citta(
            z=sz, x=sx, raggio=_raggio_piana(lato, murata), piazza=(sz, sx),
            porte=_porte_esterne(sz, sx, lato, murata, (H, W)),
            murata=murata, edifici=len(ed)))

    return fuori, h, vie, muro, citta, banchi, urbano, cls, livello


def statistiche(edifici: list[E.Edificio], vie: np.ndarray, muro: np.ndarray,
                citta: list[Citta], banchi: list | None = None) -> dict:
    return {
        "insediamenti": len(citta),
        "citta": sum(1 for c in citta if c.e_citta),
        "edifici": len(edifici),
        "assi": int((vie == ASSE).sum()),
        "secondarie": int((vie == SECONDARIA).sum()),
        "vicoli": int((vie == VICOLO).sum()),
        "mura": int((muro == 1).sum()),
        "porte": int((muro == 2).sum()),
        "torri": int((muro == 3).sum()),
        "botteghe": sum(1 for e in edifici if e.mestiere),
        "banchi": len(banchi or []),
    }


def conteggio_mestieri(edifici: list[E.Edificio]) -> dict[str, int]:
    fuori: dict[str, int] = {}
    for e in edifici:
        if e.mestiere:
            fuori[e.mestiere] = fuori.get(e.mestiere, 0) + 1
    return dict(sorted(fuori.items(), key=lambda t: -t[1]))
