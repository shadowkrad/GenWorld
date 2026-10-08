"""Verifica di strade e ponti.

Il punto del modulo: i ponti non si piazzano, emergono. I test devono
verificare l'emergere, non la presenza di una funzione "metti un ponte".
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import edifici as E
from genworld import mappa as M
from genworld import strade as S


class TestCosto(unittest.TestCase):

    def test_acqua_costa_piu_della_terra(self):
        cls = np.full((40, 40), M.PRATERIA, np.uint8)
        cls[:, 20:] = M.MARE
        h = np.full((40, 40), 80, np.int32)
        c = S.griglia_costo(cls, h)
        self.assertGreater(float(c[20, 30]), float(c[20, 5]) + 5)

    def test_pendenza_costa(self):
        cls = np.full((40, 40), M.PRATERIA, np.uint8)
        z, x = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
        h = np.where(x > 20, 80 + (x - 20) * 4, 80).astype(np.int32)
        c = S.griglia_costo(cls, h)
        self.assertGreater(float(c[20, 30]), float(c[20, 5]))


class TestGeometria(unittest.TestCase):

    def test_segmento_tocca_gli_estremi(self):
        s = S._segmento((2, 3), (9, 14))
        self.assertEqual(s[0], (2, 3))
        self.assertEqual(s[-1], (9, 14))

    def test_segmento_continuo(self):
        s = S._segmento((0, 0), (11, 5))
        for a, b in zip(s, s[1:]):
            self.assertLessEqual(max(abs(a[0] - b[0]), abs(a[1] - b[1])), 1)

    def test_astar_trova_e_gira_attorno(self):
        """Con un muro costoso in mezzo, il cammino deve aggirarlo."""
        c = np.ones((30, 30), np.float32)
        c[10:20, 5:25] = 500.0            # barriera
        cam = S._astar(c, (2, 15), (27, 15))
        self.assertGreater(len(cam), 0, "nessun cammino trovato")
        dentro = sum(1 for z, x in cam if 10 <= z < 20 and 5 <= x < 25)
        self.assertLess(dentro, 6, "il cammino attraversa la barriera invece di girarla")

    def test_astar_senza_uscita(self):
        c = np.ones((10, 10), np.float32)
        self.assertEqual(S._astar(c, (0, 0), (50, 50)), [])


class TestPianificazione(unittest.TestCase):

    def scenario(self, n=200):
        """Due rive separate da un fiume verticale."""
        cls = np.full((n, n), M.PRATERIA, np.uint8)
        cls[:, 95:105] = M.FIUME
        h = np.full((n, n), 80, np.int32)
        h[:, 95:105] = 58
        liv = np.full((n, n), 62.0, np.float32)
        liv[:, 95:105] = 66.0
        siti = [(60, 40, 20), (60, 160, 20)]
        return cls, h, liv, siti

    def test_il_ponte_emerge_sull_acqua(self):
        """Nessuno ha chiesto un ponte: c'e' perche' la strada passa sul fiume."""
        cls, h, liv, siti = self.scenario()
        tipo, quota, h2 = S.pianifica(cls, h, liv, siti, [], seed=1)
        ponte = (tipo == S.PONTE) | (tipo == S.PARAPETTO)
        self.assertTrue(ponte.any(), "nessun ponte dove la strada attraversa il fiume")
        # e sta proprio sul fiume
        self.assertTrue(np.isin(cls[ponte], list(M.ACQUA)).all()
                        | (h[ponte] < quota[ponte] - 3).all())

    def test_il_ponte_sta_sopra_il_pelo_dell_acqua(self):
        cls, h, liv, siti = self.scenario()
        tipo, quota, _ = S.pianifica(cls, h, liv, siti, [], seed=2)
        ponte = (tipo == S.PONTE) | (tipo == S.PARAPETTO)
        if ponte.any():
            self.assertTrue((quota[ponte] > liv[ponte]).all(),
                            "impalcato sommerso")

    def test_la_sede_e_spianata_ma_il_ponte_no(self):
        """Sotto il ponte il terreno resta: e' li' che poggiano i pilastri."""
        cls, h, liv, siti = self.scenario()
        tipo, quota, h2 = S.pianifica(cls, h, liv, siti, [], seed=3)
        ponte = (tipo == S.PONTE) | (tipo == S.PARAPETTO)
        if ponte.any():
            self.assertTrue(np.array_equal(h2[ponte], h[ponte]),
                            "il terreno sotto il ponte e' stato spianato")

    def test_senza_villaggi_nessuna_strada(self):
        cls, h, liv, _ = self.scenario()
        tipo, _, h2 = S.pianifica(cls, h, liv, [], [], seed=4)
        self.assertEqual(int((tipo > 0).sum()), 0)
        self.assertTrue(np.array_equal(h2, h))

    def test_parapetto_sui_bordi(self):
        cls, h, liv, siti = self.scenario()
        tipo, _, _ = S.pianifica(cls, h, liv, siti, [], seed=5)
        if (tipo == S.PARAPETTO).any():
            self.assertGreater(int((tipo == S.PARAPETTO).sum()), 0)

    def test_i_vicoli_raggiungono_le_case(self):
        cls, h, liv, siti = self.scenario()
        ed = [E.Edificio(x=48, z=64, larghezza=8, profondita=8, base=80)]
        tipo, _, _ = S.pianifica(cls, h, liv, siti, ed, seed=6)
        vicino = tipo[60:72, 44:58]
        self.assertGreater(int((vicino > 0).sum()), 0,
                           "nessun vicolo verso la casa")


if __name__ == "__main__":
    unittest.main()

class TestStradaVicinoAllAcqua(unittest.TestCase):

    def test_un_fosso_non_abbassa_la_strada_che_lo_costeggia(self):
        """Il fosso di una citta' sta tre blocchi sotto la piana: la media del
        terreno sulla strada lo contava, e la strada scendeva di uno vicino
        alla riva, un blocco sotto l'impalcato del ponte."""
        cls = np.full((60, 60), M.PIANURA, np.uint8)
        h = np.full((60, 60), 75, np.int32)
        cls[:, 28:31] = M.FIUME
        h[:, 28:31] = 72
        liv = np.full((60, 60), 62.0, np.float32)
        vie = np.zeros((60, 60), np.uint8)
        vie[30, :] = 2                       # una strada che attraversa il fosso
        _, quota, _ = S.pianifica(cls, h, liv, [(30, 10, 20), (30, 50, 20)], [],
                                  livello_mare=62, vie=vie)
        sponda = quota[30, 26]
        self.assertEqual(int(sponda), 75, "la strada sulla riva deve stare a quota piana")
        self.assertEqual(int(quota[30, 29]), int(sponda), "l'impalcato a quota della strada")
