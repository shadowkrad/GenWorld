"""Test del fondale.

Questi test nascono da una misura, non da un'idea: prima di `batimetria.py` il
53% di tutta l'acqua di Arda stava a esattamente y=54 - un piano unico - e
c'erano 2.836 celle con un salto di piu' di otto blocchi in una cella sola,
con punte di 67. Il fondale era due terrazze e un muro.

Quindi qui si misura proprio quello: che la profondita' cresca allontanandosi
da riva, che nessuna quota domini, e che al largo non ci siano muri.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import batimetria as B  # noqa: E402
from genworld.mappa import FIUME, MARE, OCEANO, PIANURA  # noqa: E402

MARE_Y = 62


def isola(lato=400, raggio=70):
    """Un'isola tonda in mezzo al mare: il caso piu' semplice in cui la
    distanza dalla costa e' nota e verificabile."""
    zz, xx = np.ogrid[:lato, :lato]
    d = np.hypot(zz - lato / 2, xx - lato / 2)
    cls = np.where(d <= raggio, PIANURA, MARE).astype(np.uint8)
    cls = np.where(d > raggio * 2.4, OCEANO, cls).astype(np.uint8)
    h = np.where(d <= raggio, 80, MARE_Y - 3).astype(np.int32)
    return cls, h


class TestProfondita(unittest.TestCase):

    def setUp(self):
        self.cls, self.h = isola()
        self.p = B.profondita(self.cls, MARE_Y, seed=3)
        self.marino = np.isin(self.cls, (MARE, OCEANO))
        self.d = distance_transform_edt(self.marino)

    def test_sulla_terra_e_zero(self):
        self.assertTrue((self.p[~self.marino] == 0).all())

    def test_in_acqua_e_sempre_almeno_un_blocco(self):
        """Profondita' zero vuol dire terra asciutta in mezzo al mare."""
        self.assertGreaterEqual(float(self.p[self.marino].min()), 1.0)

    def test_cresce_allontanandosi_da_riva(self):
        """La profondita' discende dalla distanza dalla costa: e' questo il
        cambiamento. Prima discendeva dalla classe, e da li' veniva il muro."""
        medie = []
        for lo, hi in ((1, 4), (4, 12), (12, 30), (30, 60), (60, 120)):
            m = self.marino & (self.d >= lo) & (self.d < hi)
            if m.sum() > 100:
                medie.append(float(self.p[m].mean()))
        self.assertGreater(len(medie), 3)
        self.assertEqual(medie, sorted(medie), f"non monotona: {medie}")

    def test_i_tre_regimi_esistono(self):
        """Piattaforma dolce, scarpata ripida, piana quasi piatta: se il
        gradiente fosse uguale ovunque sarebbe un cono, non un fondale."""
        def pendenza(lo, hi):
            m = self.marino & (self.d >= lo) & (self.d < hi)
            gz, gx = np.gradient(self.p)
            return float(np.hypot(gz, gx)[m].mean())
        piattaforma = pendenza(4, B.FINE_PIATTAFORMA - 4)
        scarpata = pendenza(B.FINE_PIATTAFORMA + 4, B.FINE_SCARPATA - 4)
        self.assertGreater(scarpata, piattaforma * 1.5,
                           "la scarpata non e' piu' ripida della piattaforma")

    def test_senza_rilievo_e_liscia(self):
        p = B.profondita(self.cls, MARE_Y, rilievo=0.0, seed=3)
        gz, gx = np.gradient(p)
        lontano = binary_erosion(self.marino, iterations=3, border_value=0)
        self.assertLess(float(np.hypot(gz, gx)[lontano].max()), 2.0)

    def test_col_rilievo_il_fondo_non_e_un_piano(self):
        """Si misura lungo un'ISOBATA: fra tutte le celle alla stessa
        distanza dalla costa, quanto varia la profondita'.

        Senza rilievo variano di zero - il fondale e' una superficie di
        rotazione attorno all'isola, cioe' un imbuto liscio. Col rilievo
        devono variare, perche' ci sono secche e fosse. La deviazione
        complessiva non serve: e' dominata dall'andamento riva-largo, che
        c'e' in tutti e due i casi e nasconde la differenza (misurato: 13,7
        contro 12,6, cioe' niente).
        """
        anello = self.marino & (self.d >= 40) & (self.d < 42)
        self.assertGreater(int(anello.sum()), 200)
        liscio = B.profondita(self.cls, MARE_Y, rilievo=0.0, seed=3)
        mosso = B.profondita(self.cls, MARE_Y, rilievo=1.0, seed=3)
        self.assertLess(float(liscio[anello].std()), 0.5, "imbuto non liscio")
        self.assertGreater(float(mosso[anello].std()), 2.0,
                           "il fondale e' ancora una superficie di rotazione")

class TestApplica(unittest.TestCase):

    def setUp(self):
        self.cls, self.h = isola()
        self.nuovo = B.applica(self.h.copy(), self.cls, MARE_Y, seed=3)
        self.marino = np.isin(self.cls, (MARE, OCEANO))

    def test_la_terra_non_si_tocca(self):
        self.assertTrue((self.nuovo[~self.marino] == self.h[~self.marino]).all())

    def test_il_fondo_sta_sotto_il_pelo_dell_acqua(self):
        self.assertLess(int(self.nuovo[self.marino].max()), MARE_Y)

    def test_nessuna_quota_domina(self):
        """Il difetto di partenza in una riga: il 53% dell'acqua a y=54."""
        q, n = np.unique(self.nuovo[self.marino], return_counts=True)
        self.assertLess(n.max() / self.marino.sum(), 0.30,
                        f"una sola quota copre il {n.max()/self.marino.sum():.0%}")
        self.assertGreater(len(q), 15, "troppe poche quote distinte")

    def test_niente_muri_al_largo(self):
        """2.836 celle con un salto oltre otto blocchi, punte di 67."""
        gz, gx = np.gradient(self.nuovo.astype(float))
        pend = np.hypot(gz, gx)
        lontano = binary_erosion(self.marino, iterations=3, border_value=0)
        self.assertEqual(int((pend[lontano] > 8).sum()), 0)
        self.assertLess(float(pend[lontano].max()), 6.0)

    def test_i_fiumi_non_sono_mare(self):
        """Un fiume scorre in quota: se lo trattassimo da mare finirebbe sul
        fondale, cioe' sottoterra."""
        cls = self.cls.copy()
        cls[200, :] = FIUME
        h = self.h.copy()
        h[200, :] = 120
        nuovo = B.applica(h, cls, MARE_Y, seed=3)
        self.assertTrue((nuovo[200, :] == 120).all())

    def test_senza_mare_non_cambia_niente(self):
        cls = np.full((50, 50), PIANURA, np.uint8)
        h = np.full((50, 50), 80, np.int32)
        self.assertTrue((B.applica(h, cls, MARE_Y) == h).all())


if __name__ == "__main__":
    unittest.main()
