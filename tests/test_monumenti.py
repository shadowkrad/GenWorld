"""Il castello (uno per mappa) e le navi: lettura degli schemi, posto, terreno."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import monumenti as MN  # noqa: E402
from genworld import motore as MO  # noqa: E402
from genworld import template as TM  # noqa: E402
from genworld.mappa import MARE, PIANURA  # noqa: E402

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAVE = os.path.join(RADICE, "templates", "navi")
CASTELLO = os.path.join(RADICE, "templates", "castelli")
HA_FILE = os.path.isdir(NAVE) and os.path.isdir(CASTELLO)


class Finto:
    id_aria = 0

    def __init__(self):
        self._id = {}

    def blocco(self, nome, **prop):
        return self._id.setdefault((nome, tuple(sorted(prop.items()))), len(self._id) + 1)


@unittest.skipUnless(HA_FILE, "mancano i file di nave e castello")
class TestLettura(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.castelli, cls.navi = MN.carica(CASTELLO, NAVE, (1, 21, 4))

    def test_si_leggono_entrambi(self):
        self.assertTrue(self.castelli and self.navi)

    def test_il_pacchetto_di_navi_da_piu_modelli(self):
        self.assertGreaterEqual(len(self.navi), 20)
        nomi = {m.nome for m in self.navi}
        self.assertEqual(len(nomi), len(self.navi), "nomi ripetuti")

    def test_le_navi_non_portano_con_se_il_mare(self):
        for m in self.navi:
            nomi = [n for n, _ in m.tavolozza]
            acqua = [i for i, n in enumerate(nomi) if n in ("water", "flowing_water")]
            self.assertFalse(np.isin(m.celle, acqua).any(), m.nome)
            self.assertGreaterEqual(m.linea_acqua, 1)
            self.assertLess(m.linea_acqua, m.dy - 3, "scafo tutto sott'acqua")

    def test_ogni_nave_e_ritagliata_e_ha_la_chiglia_in_basso(self):
        for m in self.navi:
            self.assertTrue((m.celle[:, 0, :] >= 0).any(), f"{m.nome}: strato vuoto in fondo")
            self.assertGreaterEqual(int((m.celle >= 0).sum()), MN.SOLIDI_MINIMI_NAVE)

    def test_una_nave_e_una_sola(self):
        """Componenti separate: una nave non deve contenere due scafi lontani."""
        from scipy.ndimage import label
        for m in self.navi:
            _, n = label(m.celle >= 0, structure=np.ones((3, 3, 3)))
            self.assertLessEqual(n, 3, f"{m.nome}: {n} pezzi")

    def test_i_modelli_non_vanno_ai_lotti(self):
        for m in self.castelli + self.navi:
            self.assertIn("_escluso", m.stili)


class TestPosto(unittest.TestCase):

    def test_il_castello_sta_su_terra_e_il_terreno_si_adatta(self):
        n = 300
        cls = np.full((n, n), PIANURA, np.uint8)
        h = np.full((n, n), 80, np.int32)
        h[:, 150:] += np.arange(n - 150)[None, :] // 8          # un pendio
        m = TM.Modello(nome="c", celle=np.zeros((30, 10, 30), np.int32),
                       tavolozza=[("stone", {})])
        evita = np.zeros((n, n), bool)
        c = MN.pianifica_castello([m], 7, h, cls, evita, (PIANURA,), 62, seed=1)
        self.assertIsNotNone(c)
        self.assertEqual(c.modello, 7)
        piano = h[c.z:c.z + c.profondita, c.x:c.x + c.larghezza]
        self.assertEqual(len(np.unique(piano)), 1, "il sedime non e' piano")

    def test_le_navi_stanno_solo_in_mare_profondo(self):
        n = 400
        cls = np.full((n, n), MARE, np.uint8)
        cls[:, :60] = PIANURA
        h = np.full((n, n), 30, np.int32)
        celle = np.full((10, 12, 30), -1, np.int32)
        celle[:, :6, :] = 0
        m = TM.Modello(nome="n", celle=celle, tavolozza=[("planks", {})])
        m.linea_acqua = 3
        navi = MN.pianifica_navi([m], 5, h, cls, (MARE,), 62, seed=2)
        self.assertGreaterEqual(len(navi), 1)
        for c in navi:
            self.assertGreaterEqual(c.x - MN.MARGINE_NAVE, 60, "la nave tocca la terra")
            self.assertEqual(c.base, 62 - 3, "linea di galleggiamento sbagliata")

    def test_senza_mare_profondo_niente_navi(self):
        n = 200
        cls = np.full((n, n), MARE, np.uint8)
        h = np.full((n, n), 61, np.int32)             # fondale a pelo d'acqua
        m = TM.Modello(nome="n", celle=np.zeros((10, 12, 30), np.int32),
                       tavolozza=[("planks", {})])
        m.linea_acqua = 8
        self.assertEqual(MN.pianifica_navi([m], 0, h, cls, (MARE,), 62, seed=3), [])


class TestPiazza(unittest.TestCase):

    def test_il_lastricato_e_tondo(self):
        s = Finto()
        out = np.zeros((16, 100, 16), np.uint32)
        h_c = np.full((16, 16), 70, np.int32)
        MO._posa_piazze(out, s, [(8, 8, 4)], h_c, 0, 0, 0)
        y = 69
        piena = out[:, y, :] != 0
        self.assertTrue(piena[8, 8])
        self.assertTrue(piena[8, 12] or piena[8, 11])
        self.assertFalse(piena[0, 0], "fuori dal cerchio")
        self.assertFalse(piena[8, 14], "oltre il raggio")
        self.assertFalse(piena[12, 12], "l'angolo del quadrato non e' nel cerchio")
        self.assertTrue(piena[12, 8])


if __name__ == "__main__":
    unittest.main()
