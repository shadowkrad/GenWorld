"""Verifica dell'erosione: stabilita', bilancio di massa, limiti."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld.erosione import _pennello, erodi, statistiche


def collina(n=96):
    x = np.linspace(-3, 3, n)
    X, Z = np.meshgrid(x, x, indexing="ij")
    return (80 + 60 * np.exp(-(X ** 2 + Z ** 2) / 3)
            + 6 * np.sin(X * 3) * np.cos(Z * 2)).astype(np.float32)


class TestPennello(unittest.TestCase):

    def test_pesi_normalizzati(self):
        for r in (0, 1, 2, 4):
            _, _, w = _pennello(r)
            self.assertAlmostEqual(float(w.sum()), 1.0, places=5)

    def test_disco(self):
        ox, oy, _ = _pennello(2)
        self.assertTrue(np.all(np.hypot(ox, oy) <= 2 + 1e-9))


class TestErosione(unittest.TestCase):

    def test_non_diverge(self):
        """La regressione che e' costata piu' tempo: divergenza dopo ~17 passi."""
        h = collina()
        for passi in (16, 32, 64):
            e = erodi(h, gocce_per_cella=0.5, passi=passi, seed=1)
            self.assertTrue(np.isfinite(e).all(), f"non finito a {passi} passi")
            self.assertLess(float(np.abs(e - h).max()), 100.0,
                            f"variazione fuori scala a {passi} passi")

    def test_massa_quasi_conservata(self):
        h = collina()
        e = erodi(h, gocce_per_cella=0.3, passi=32, seed=2)
        perdita = abs(float((e - h).sum()) / float(h.sum()))
        self.assertLess(perdita, 0.02)

    def test_scolpisce_qualcosa(self):
        """L'erosione deve scavare e depositare in modo misurabile.

        NON si controlla la pendenza media: su una griglia piccola l'erosione
        smussa piu' di quanto incida, e la pendenza media CALA. Sale solo su
        griglie grandi, dove i canali hanno spazio per formarsi. Asserire il
        contrario significava scrivere un test che descrive un'aspettativa
        invece del comportamento reale.
        """
        h = collina(192)
        e = erodi(h, gocce_per_cella=0.4, passi=40, seed=3)
        s = statistiche(h, e)
        self.assertGreater(s["scavo_massimo"], 1.0)
        self.assertGreater(s["deposito_massimo"], 0.5)
        self.assertGreater(s["spostamento_medio"], 0.01)

    def test_pavimento_rispettato(self):
        h = collina()
        e = erodi(h, gocce_per_cella=0.6, passi=40, quota_minima=78.0, seed=4)
        self.assertGreaterEqual(float(e.min()), 78.0 - 1e-4)

    def test_maschera_protegge(self):
        """Fuori dalla maschera il terreno non deve muoversi."""
        h = collina()
        m = np.zeros(h.shape, bool)
        m[:48, :] = True
        e = erodi(h, gocce_per_cella=0.5, passi=32, maschera=m, seed=5)
        intatto = np.abs(e[60:, :] - h[60:, :]).max()
        self.assertLess(float(intatto), 0.6)

    def test_deterministico(self):
        h = collina()
        a = erodi(h, gocce_per_cella=0.2, passi=24, seed=7)
        b = erodi(h, gocce_per_cella=0.2, passi=24, seed=7)
        self.assertTrue(np.array_equal(a, b))

    def test_zero_gocce_non_cambia_nulla(self):
        h = collina()
        self.assertTrue(np.array_equal(erodi(h, n_gocce=0), h))


if __name__ == "__main__":
    unittest.main()
