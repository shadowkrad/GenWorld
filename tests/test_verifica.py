"""Verifica automatica del mondo: i rilevatori su blocchi finti."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import verifica as V  # noqa: E402


def finestra():
    s = np.zeros((20, 30, 20), bool)
    s[:, 0] = True               # il suolo
    return s


class TestSospesi(unittest.TestCase):

    def test_un_pezzo_attaccato_al_suolo_non_e_sospeso(self):
        s = finestra()
        s[8:11, 1:5, 8:11] = True
        self.assertEqual(V.pezzi_sospesi(s), [])

    def test_un_pezzo_in_aria_e_sospeso(self):
        s = finestra()
        s[8:11, 10:12, 8:11] = True
        r = V.pezzi_sospesi(s)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][3], 18)
        self.assertEqual(r[0][1], 10)

    def test_un_pezzo_che_tocca_il_bordo_continua_nella_finestra_accanto(self):
        s = finestra()
        s[0:3, 10:12, 8:11] = True
        self.assertEqual(V.pezzi_sospesi(s), [])

    def test_un_pezzo_appeso_a_una_colonna_e_attaccato(self):
        s = finestra()
        s[8:11, 1:10, 8] = True      # un pilastro
        s[5:15, 10, 5:12] = True     # il piano sopra
        self.assertEqual(V.pezzi_sospesi(s), [])

    def test_uno_o_due_blocchi_isolati_sono_minerali_di_caverna(self):
        s = finestra()
        s[8, 10, 8] = True
        s[12:14, 10, 12] = True
        self.assertEqual(V.pezzi_sospesi(s), [])

    def test_un_isola_enorme_non_e_un_detrito(self):
        s = finestra()
        s[2:18, 10:14, 2:18] = True
        self.assertEqual(V.pezzi_sospesi(s, massimo=100), [])


class TestAcqua(unittest.TestCase):

    def test_acqua_contenuta_non_trabocca(self):
        acqua = np.zeros((10, 10, 10), bool)
        aria = np.ones((10, 10, 10), bool)
        acqua[3:6, 2, 3:6] = True
        aria[3:6, 2, 3:6] = False
        pieno = ~aria & ~acqua
        pieno[2:7, 2, 2:7] = True
        pieno[3:6, 2, 3:6] = False
        aria &= ~pieno
        self.assertEqual(V.acqua_che_trabocca(acqua, aria), [])

    def test_acqua_con_aria_accanto_trabocca(self):
        acqua = np.zeros((10, 10, 10), bool)
        aria = np.ones((10, 10, 10), bool)
        acqua[4, 2, 4] = True
        aria[4, 2, 4] = False
        self.assertEqual(V.acqua_che_trabocca(acqua, aria), [(4, 2, 4)])


class TestRapporto(unittest.TestCase):

    def test_pulito_solo_se_completo_e_senza_difetti(self):
        r = V.Rapporto(chunk_attesi=4, chunk_trovati=4)
        self.assertTrue(r.pulito)
        r.sospesi.append({"x": 0, "y": 0, "z": 0, "blocchi": 3})
        self.assertFalse(r.pulito)
        self.assertFalse(V.Rapporto(chunk_attesi=4, chunk_trovati=3).pulito)


if __name__ == "__main__":
    unittest.main()
