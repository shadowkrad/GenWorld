"""Test dei laghi: bacini chiusi trovati dal rilievo, non dalla mappa."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import laghi as L  # noqa: E402


class TestTrova(unittest.TestCase):

    def conca(self, n=60, profondita=6.0):
        """Un altopiano piatto con una conca chiusa in mezzo, mare lontano
        e irraggiungibile dalla conca (niente sbocco)."""
        h = np.full((n, n), 100.0, np.float32)
        zz, xx = np.mgrid[:n, :n]
        d = np.hypot(zz - n / 2, xx - n / 2)
        h[d < 8] = 100.0 - profondita
        mare = np.zeros((n, n), bool)
        mare[:3, :] = True                  # un mare in un angolo, lontano
        return h, mare

    def test_trova_una_conca_chiusa(self):
        h, mare = self.conca()
        esclusi = np.zeros(h.shape, bool)
        maschera, livello = L.trova(h, mare, esclusi)
        self.assertTrue(maschera[30, 30])
        self.assertAlmostEqual(float(livello[30, 30]), 100.0, delta=0.51)

    def test_niente_lago_senza_conca(self):
        h = np.full((40, 40), 80.0, np.float32)
        mare = np.zeros((40, 40), bool)
        mare[:3, :] = True
        esclusi = np.zeros(h.shape, bool)
        maschera, livello = L.trova(h, mare, esclusi)
        self.assertFalse(maschera.any())

    def test_conca_troppo_piccola_scartata(self):
        h, mare = self.conca(profondita=6.0)
        esclusi = np.zeros(h.shape, bool)
        maschera, _ = L.trova(h, mare, esclusi, area_minima=10_000)
        self.assertFalse(maschera.any())

    def test_conca_troppo_grande_scartata(self):
        h, mare = self.conca(profondita=6.0)
        esclusi = np.zeros(h.shape, bool)
        maschera, _ = L.trova(h, mare, esclusi, area_massima=5)
        self.assertFalse(maschera.any())

    def test_conca_poco_profonda_scartata(self):
        h, mare = self.conca(profondita=0.3)
        esclusi = np.zeros(h.shape, bool)
        maschera, _ = L.trova(h, mare, esclusi, profondita_minima=1.5)
        self.assertFalse(maschera.any())

    def test_esclusi_toglie_la_conca(self):
        h, mare = self.conca()
        esclusi = np.ones(h.shape, bool)
        maschera, _ = L.trova(h, mare, esclusi)
        self.assertFalse(maschera.any())

    def test_il_pelo_e_piatto(self):
        """Un lago non ha increspature: il pelo dentro un bacino e' un
        unico valore, non uno diverso per ogni cella."""
        h, mare = self.conca()
        esclusi = np.zeros(h.shape, bool)
        maschera, livello = L.trova(h, mare, esclusi)
        valori = np.unique(livello[maschera])
        self.assertEqual(valori.size, 1)


class TestScava(unittest.TestCase):

    def test_il_fondo_resta_sotto_il_pelo(self):
        h = np.full((20, 20), 90.0, np.float32)
        h[8:12, 8:12] = 85.0
        maschera = np.zeros((20, 20), bool)
        maschera[8:12, 8:12] = True
        livello = np.full((20, 20), np.nan, np.float32)
        livello[maschera] = 90.0
        fondo = L.scava(h, maschera, livello)
        self.assertTrue((fondo[maschera] <= 89.0).all())

    def test_non_scava_piu_del_limite(self):
        h = np.full((20, 20), 90.0, np.float32)
        h[8:12, 8:12] = 10.0            # una voragine, non un lago
        maschera = np.zeros((20, 20), bool)
        maschera[8:12, 8:12] = True
        livello = np.full((20, 20), np.nan, np.float32)
        livello[maschera] = 90.0
        fondo = L.scava(h, maschera, livello, profondita_massima=5.0)
        self.assertTrue((fondo[maschera] >= 85.0).all())

    def test_fuori_dalla_maschera_non_cambia(self):
        h = np.full((20, 20), 90.0, np.float32)
        maschera = np.zeros((20, 20), bool)
        livello = np.full((20, 20), np.nan, np.float32)
        fondo = L.scava(h, maschera, livello)
        self.assertTrue(np.array_equal(fondo, h))


class TestStatistiche(unittest.TestCase):

    def test_conta_bacini_e_celle(self):
        maschera = np.zeros((30, 30), bool)
        maschera[2:5, 2:5] = True        # bacino 1, 9 celle
        maschera[20:23, 20:24] = True    # bacino 2, 12 celle
        livello = np.full((30, 30), np.nan, np.float32)
        livello[2:5, 2:5] = 70.0
        livello[20:23, 20:24] = 90.0
        st = L.statistiche(maschera, livello)
        self.assertEqual(st["celle"], 21)
        self.assertEqual(st["bacini"], 2)
        self.assertAlmostEqual(st["quota_media"], (70.0 * 9 + 90.0 * 12) / 21,
                               places=2)

    def test_senza_laghi(self):
        maschera = np.zeros((10, 10), bool)
        livello = np.full((10, 10), np.nan, np.float32)
        st = L.statistiche(maschera, livello)
        self.assertEqual(st, {"celle": 0, "bacini": 0, "quota_media": 0.0})


if __name__ == "__main__":
    unittest.main()
