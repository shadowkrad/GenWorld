"""Fortezze isolate (#6): una cinta di pietra con torri agli angoli, un portale, un
cortile allestito e un maschio al centro che e' una torre curata (`torri.py`).

Una fortezza sta su un pianoro asciutto e lontano da tutto. Dentro ci vivono dei
soldati e, nel maschio, il capo con la sua armatura e un nome. Piu' e' lontana
dalle citta' piu' il capo e' forte (come per le torri).

Coordinate locali (u, v) dal centro: u va verso il portale (la fronte), v e'
la perpendicolare. La cinta e' un quadrato di mezzo lato `F`; il maschio e'
spostato verso il fondo cosi' il cortile, fra portale e maschio, resta aperto.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import torri as TR

F = 14                       # mezzo lato della cinta (29 x 29)
ALTEZZA_MURA = 7             # la cinta sale di sette blocchi: il cammino sta in cima
ALTEZZA_TORRE_ANGOLO = 12    # le torri d'angolo, due piani e il tetto
CENTRO_MASCHIO = -5          # il maschio sta qui, sull'asse (v = 0)
RILIEVO_MASSIMO = 4
DISTANZA_FRA_FORTEZZE = 320
CORTILE = 12                 # mezzo lato del cortile interno

FACCIA = TR.FACCIA
SPECIE = TR.SPECIE
NOMI_FORTEZZA = ("Barone Rotger", "Dama Isolda", "Conte Valdemaro", "Contessa Aurelia",
                 "Signore di Roccanera", "Castellano Olmo", "Maresciallo Brenno",
                 "Baronessa Livia")
CAPO_PER_ZONA = ("vindicatore", "scheletro_nero", "zombie", "scheletro")


@dataclass
class Fortezza:
    x: int                      # centro della cinta, coordinate di mappa
    z: int
    ys: int                     # quota del cortile
    dx: int                     # il portale guarda da questa parte
    dz: int
    specie: str = "vindicatore"
    nome: str = "Barone Rotger"
    seme: int = 0

    def punto(self, u: int, v: int) -> tuple[int, int]:
        return (self.x + self.dx * u - self.dz * v, self.z + self.dz * u + self.dx * v)

    def maschio(self) -> TR.Torre:
        """La torre del maschio, con gli stessi seme e capo della fortezza."""
        mx, mz = self.punto(CENTRO_MASCHIO, 0)
        return TR.Torre(x=mx, z=mz, ys=self.ys, dx=self.dx, dz=self.dz,
                        specie=self.specie, nome=self.nome, seme=self.seme)

    def bbox(self) -> tuple[int, int, int, int]:
        r = F + 3
        pts = [self.punto(u, v) for u in (-r, r) for v in (-r, r)]
        return (min(p[0] for p in pts), min(p[1] for p in pts),
                max(p[0] for p in pts), max(p[1] for p in pts))

    def forzieri(self) -> list[tuple[int, int, int, str]]:
        """(x, y, z, tipo): quelli del maschio piu' una cassa nella fucina."""
        fuori = list(self.maschio().forzieri())
        x, z = self.punto(5, -F + 2)
        fuori.append((x, self.ys, z, "stanza"))
        return fuori


# --------------------------------------------------------------------------
# Pianificazione
# --------------------------------------------------------------------------

def pianifica(altezze: np.ndarray, acqua: np.ndarray, evita: np.ndarray | None = None,
              densita: float = 1.0, seed: int = 0, celle_per: int = 500_000) -> list[Fortezza]:
    """Siti su un pianoro asciutto (rilievo <= 4 su 35 x 35), lontani da tutto."""
    H, W = altezze.shape
    rng = np.random.default_rng(seed + 505)
    quanti = int(max(0, round(H * W / celle_per * densita)))
    fuori: list[Fortezza] = []
    liberi = list(NOMI_FORTEZZA)
    tentativi, limite = 0, quanti * 120 + 120
    r = F + 3
    while len(fuori) < quanti and tentativi < limite:
        tentativi += 1
        x = int(rng.integers(r + 6, W - r - 6))
        z = int(rng.integers(r + 6, H - r - 6))
        if any((f.x - x) ** 2 + (f.z - z) ** 2 < DISTANZA_FRA_FORTEZZE ** 2 for f in fuori):
            continue
        quadro = altezze[z - r:z + r + 1, x - r:x + r + 1]
        if int(quadro.max() - quadro.min()) > RILIEVO_MASSIMO:
            continue
        ys = int(quadro.max())
        if ys < 64 or ys + 24 > 250:
            continue
        if acqua[z - r:z + r + 1, x - r:x + r + 1].any():
            continue
        if evita is not None and evita[z - r:z + r + 1, x - r:x + r + 1].any():
            continue
        dx, dz = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(0, 4))]
        nome = liberi.pop(int(rng.integers(0, len(liberi)))) if liberi else \
            f"{NOMI_FORTEZZA[int(rng.integers(0, len(NOMI_FORTEZZA)))]} il {int(rng.integers(2, 9))}°"
        fuori.append(Fortezza(x=x, z=z, ys=ys, dx=dx, dz=dz,
                              specie=CAPO_PER_ZONA[int(rng.integers(0, len(CAPO_PER_ZONA)))],
                              nome=nome, seme=int(rng.integers(0, 2 ** 31 - 1))))
    return fuori


def maschera(fortezze: list[Fortezza], forma: tuple[int, int], margine: int = 0) -> np.ndarray:
    occ = np.zeros(forma, bool)
    H, W = forma
    r = F + 2 + margine
    for f in fortezze:
        for u in range(-r, r + 1):
            for v in range(-r, r + 1):
                x, z = f.punto(u, v)
                if 0 <= x < W and 0 <= z < H:
                    occ[z, x] = True
    return occ


def indice_per_chunk(fortezze: list[Fortezza], passo: int = 16) -> dict:
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, f in enumerate(fortezze):
        x0, z0, x1, z1 = f.bbox()
        for cz in range(z0 // passo, z1 // passo + 1):
            for cx in range(x0 // passo, x1 // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def combattenti(fortezze: list[Fortezza], centri: list | None = None,
                seed: int = 0, trofei: dict | None = None) -> list[TR.Combattente]:
    """Il capo e due guardie nel maschio (come una torre), quattro soldati nel
    cortile e un arciere sul tetto di ogni torre d'angolo."""
    fuori: list[TR.Combattente] = []
    for i, f in enumerate(fortezze):
        fuori.extend(TR.combattenti([f.maschio()], centri, seed=seed + i, trofei=trofei))
        rng = np.random.default_rng(f.seme + seed + 7)
        for k, (u, v) in enumerate(((6, -3), (6, 3), (-2, 8), (-2, -8))):
            x, z = f.punto(u, v)
            sp = ("vindicatore", "zombie", "scheletro", "zombie")[k]
            fuori.append(TR.Combattente(x=x + 0.5, y=float(f.ys), z=z + 0.5, specie=sp,
                                        seme=f.seme + 101 * (k + 1) + i))
        for k, (su, sv) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            x, z = f.punto(su * (F - 2), sv * (F - 2))
            fuori.append(TR.Combattente(x=x + 0.5, y=float(f.ys + ALTEZZA_TORRE_ANGOLO), z=z + 0.5,
                                        specie="scheletro", seme=f.seme + 211 * (k + 1) + i))
        del rng
    return fuori


def statistiche(fortezze: list[Fortezza]) -> dict:
    return {"fortezze": len(fortezze)}


# --------------------------------------------------------------------------
# Posa
# --------------------------------------------------------------------------

class Tavolozza:
    """Tutti i blocchi della torre piu' quelli del cortile."""

    def __init__(self, scrittore):
        s = scrittore
        self.torre = TR.Tavolozza(s)
        self.aria = s.id_aria
        self.ciottoli = s.blocco("cobblestone")
        self.ciottoli_muschio = s.blocco("mossy_cobblestone")
        self.ghiaia = s.blocco("gravel")
        self.muretto_ciottoli = s.blocco("wall", material="cobblestone", up="true", north="none",
                                         south="none", east="none", west="none")
        self.acqua = s.blocco("water")
        self.pioli = {f: s.blocco("ladder", facing=f) for f in FACCIA.values()}
        self.fieno = s.blocco("hay_block", axis="y")
        self.recinto = s.blocco("fence", material="spruce", north="false", south="false",
                                east="false", west="false")
        self.falo = s.blocco("campfire", facing="north", lit="true", signal_fire="false",
                             waterlogged="false")
        self.scala_ciottoli = {f: s.blocco("stairs", material="cobblestone", facing=f,
                                           half="bottom", shape="straight")
                               for f in FACCIA.values()}
        self.bauli: list[tuple[int, int, int, str]] = []


def _hash(a: int, b: int, c: int) -> int:
    return ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) & 0x7FFFFFFF


def posa(out: np.ndarray, tav: Tavolozza, h_c: np.ndarray, ox: int, oz: int, y0: int,
         fortezze: list[Fortezza], quali, meta: int = 0) -> None:
    """Disegna le fortezze nel chunk (16, H, 16): `ox, oz` sono di GIOCO, le fortezze
    di MAPPA (`meta` e' lo scarto)."""
    H = out.shape[1]
    tt = tav.torre

    def metti(x: int, y: int, z: int, blocco: int) -> None:
        lx, lz, ly = x - meta - ox, z - meta - oz, y - y0
        if 0 <= lx < 16 and 0 <= lz < 16 and 1 <= ly < H:
            out[lx, ly, lz] = blocco

    def nel_chunk(x: int, z: int) -> bool:
        return 0 <= x - meta - ox < 16 and 0 <= z - meta - oz < 16

    for i in quali:
        f = fortezze[i]
        ys = f.ys
        colore = TR.STENDARDI[f.seme % len(TR.STENDARDI)]

        def P(u: int, v: int) -> tuple[int, int]:
            return f.punto(u, v)

        def faccia(du: int, dv: int) -> str:
            return FACCIA[(f.dx * du - f.dz * dv, f.dz * du + f.dx * dv)]

        def alza(u: int, v: int, y: int, blocco: int) -> None:
            x, z = P(u, v)
            metti(x, y, z, blocco)

        fuori = faccia(1, 0)

        def mattone(u: int, v: int, y: int) -> int:
            x, z = P(u, v)
            return TR._mattone(tt, _hash(x, y, z))

        # ---- fondazione e pavimento del cortile -----------------------------------
        for u in range(-F, F + 1):
            for v in range(-F, F + 1):
                x, z = P(u, v)
                lx, lz = x - meta - ox, z - meta - oz
                if 0 <= lx < 16 and 0 <= lz < 16:
                    for y in range(int(h_c[lx, lz]) - 1, ys - 1):
                        metti(x, y, z, tav.ciottoli)
                r = _hash(x, 5, z) % 100
                suolo = (tav.ciottoli_muschio if r < 12 else tav.ghiaia if r < 28
                         else tav.ciottoli)
                if abs(v) <= 1 and u >= -1:
                    suolo = tav.ghiaia                              # il viale dal portale al maschio
                alza(u, v, ys - 1, suolo)

        # ---- la cinta ---------------------------------------------------------------
        for u in range(-F, F + 1):
            for v in range(-F, F + 1):
                d = max(abs(u), abs(v))
                if d < F - 1:
                    continue
                for y in range(ys, ys + ALTEZZA_MURA):
                    alza(u, v, y, mattone(u, v, y))
                if d == F:                                            # il parapetto, sul filo esterno
                    alza(u, v, ys + ALTEZZA_MURA, mattone(u, v, ys))
                    if (u + v) % 2 == 0:
                        alza(u, v, ys + ALTEZZA_MURA + 1, mattone(u, v, ys + 1))
                else:                                                  # il camminamento
                    for y in range(ys + ALTEZZA_MURA, ys + ALTEZZA_MURA + 3):
                        alza(u, v, y, tt.aria)
        # le fasce scolpite a meta' altezza
        for u in range(-F, F + 1):
            for v in range(-F, F + 1):
                if max(abs(u), abs(v)) == F and (abs(u) < F - 4 or abs(v) < F - 4):
                    alza(u, v, ys + 3, tt.scolpiti)

        # ---- torri d'angolo (5 x 5) -------------------------------------------------
        for su in (1, -1):
            for sv in (1, -1):
                for a in range(F - 4, F + 1):
                    for b in range(F - 4, F + 1):
                        u, v = su * a, sv * b
                        bordo = a in (F - 4, F) or b in (F - 4, F)
                        # al piano terra il pavimento e' quello del cortile
                        for y in range(ys, ys + ALTEZZA_TORRE_ANGOLO):
                            alza(u, v, y, mattone(u, v, y) if bordo else tt.aria)
                        # solaio del camminamento e tetto
                        for y in (ys + ALTEZZA_MURA - 1, ys + ALTEZZA_TORRE_ANGOLO - 1):
                            if not bordo:
                                alza(u, v, y, tt.assi if y == ys + ALTEZZA_MURA - 1
                                     else mattone(u, v, y))
                        if bordo:
                            for y in (ys + ALTEZZA_TORRE_ANGOLO, ys + ALTEZZA_TORRE_ANGOLO + 1):
                                if (a + b) % 2 == 0 or y == ys + ALTEZZA_TORRE_ANGOLO:
                                    alza(u, v, y, mattone(u, v, y))
                # lanterna sul pilastro d'angolo e feritoie
                alza(su * F, sv * F, ys + ALTEZZA_TORRE_ANGOLO + 2, tt.muretto)
                alza(su * F, sv * F, ys + ALTEZZA_TORRE_ANGOLO + 3, tt.lanterna_su)
                for a in (F - 2,):
                    for (u, v) in ((su * a, sv * F), (su * F, sv * a)):
                        for y in (ys + 2, ys + ALTEZZA_MURA + 1):
                            alza(u, v, y, tt.sbarre)
                # porta dal cortile verso il piano terra della torre
                alza(su * (F - 4), sv * (F - 2), ys, tt.aria)
                alza(su * (F - 4), sv * (F - 2), ys + 1, tt.aria)
                # accesso dal camminamento al secondo piano della torre
                for (u, v) in ((su * (F - 4), sv * (F - 1)), (su * (F - 1), sv * (F - 4))):
                    alza(u, v, ys + ALTEZZA_MURA, tt.aria)
                    alza(u, v, ys + ALTEZZA_MURA + 1, tt.aria)
                # scala a pioli dal secondo piano al tetto
                # (attaccata al muro esterno d'angolo; il solaio del tetto e' bucato li')
                lx_, lz_ = su * (F - 1), sv * (F - 1)
                for y in range(ys + ALTEZZA_MURA, ys + ALTEZZA_TORRE_ANGOLO):
                    alza(lx_, lz_, y, tav.pioli[faccia(-su, 0)])

        # ---- il portale ---------------------------------------------------------------
        for v in range(-1, 2):
            for y in range(ys, ys + 4):
                alza(F, v, y, tt.aria)
                alza(F - 1, v, y, tt.aria)
        for v in (-1, 0, 1):
            alza(F, v, ys + 3, tt.sbarre)                              # la saracinesca alzata
        for v in (-2, 2):
            for y in range(ys, ys + 5):
                alza(F, v, y, tt.scolpiti)
            alza(F + 1, v, ys + 3, tt.torcia[fuori])
        for v in (-4, 4):
            alza(F + 1, v, ys + 4, tt.stendardo[(colore, fuori)])
        for v in range(-4, 5):
            for u in (F - 1, F):
                for y in range(ys + ALTEZZA_MURA + 2, ys + ALTEZZA_MURA + 4):
                    if abs(v) >= 2:
                        alza(u, v, y, mattone(u, v, y))

        # ---- scale d'accesso al camminamento (sul lato di fondo) ---------------------
        for sv in (1, -1):
            for j in range(7):
                v = sv * (3 + j)
                for y in range(ys, ys + j):
                    alza(-F + 2, v, y, tt.mattoni)
                alza(-F + 2, v, ys + j, tt.scala_pietra[faccia(0, sv)])

        # ---- il maschio: una torre curata -------------------------------------------------
        torre = f.maschio()
        TR.posa(out, tav.torre, h_c, ox, oz, y0, [torre], [0], meta=meta)

        # ---- il cortile ---------------------------------------------------------------
        # il pozzo
        for u in range(5, 8):
            for v in range(5, 8):
                if (u, v) == (6, 6):
                    alza(u, v, ys - 1, tav.acqua)
                else:
                    alza(u, v, ys, tav.muretto_ciottoli)
        for u, v in ((5, 5), (5, 7), (7, 5), (7, 7)):
            for y in range(ys + 1, ys + 3):
                alza(u, v, y, tav.recinto)
        for u in range(5, 8):
            for v in range(5, 8):
                alza(u, v, ys + 3, tt.lastra_legno)
        # la fucina: tettoia contro la cinta, a destra del portale
        for u in range(2, 9):
            for v in (-F + 2, -F + 3, -F + 4):
                alza(u, v, ys + 3, tt.lastra_legno)
        for u in (2, 8):
            for y in range(ys, ys + 3):
                alza(u, -F + 4, y, tav.recinto)
        for u, blocco in ((3, tt.fucina), (4, tt.incudine[faccia(0, 1)]),
                          (6, tt.forno[faccia(0, 1)]), (7, tt.mola[faccia(0, 1)])):
            alza(u, -F + 2, ys, blocco)
        alza(4, -F + 2, ys, tt.incudine[faccia(0, 1)])
        x, z = P(5, -F + 2)
        metti(x, ys, z, tt.forziere[faccia(0, 1)])
        if nel_chunk(x, z):
            tav.bauli.append((x, ys, z, "stanza"))
        alza(8, -F + 2, ys, tt.barile)
        alza(5, -F + 3, ys + 2, tt.lanterna)
        # la stalla: recinto con fieno contro la cinta, a sinistra del portale
        for u in range(2, 9):
            alza(u, F - 4, ys, tav.recinto)
        for v in (F - 4, F - 3, F - 2):
            alza(2, v, ys, tav.recinto)
            alza(8, v, ys, tav.recinto)
        for u in (3, 4, 7):
            alza(u, F - 2, ys, tav.fieno)
        # il campo d'armi: balle di fieno
        alza(9, -6, ys, tav.fieno)
        alza(10, -6, ys, tav.fieno)
        alza(9, -6, ys + 1, tav.fieno)
        # bracieri lungo il viale e lampioni sui pali
        for u in (3, 9):
            for v in (-3, 3):
                alza(u, v, ys, tt.mattoni)
                alza(u, v, ys + 1, tav.falo)
        for u in (6, 12):
            for v in (-2, 2):
                for y in range(ys, ys + 3):
                    alza(u, v, y, tav.recinto)
                alza(u, v, ys + 3, tt.lanterna_su)
        # stendardi sulla faccia interna della cinta
        for v in (-8, 8):
            alza(F - 2, v, ys + 4, tt.stendardo[(colore, faccia(-1, 0))])
        for u in (-8, -2, 4):
            alza(u, F - 2, ys + 4, tt.stendardo[(colore, faccia(0, -1))])
            alza(u, -F + 2, ys + 4, tt.stendardo[(colore, faccia(0, 1))])
