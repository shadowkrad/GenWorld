"""Mondi da quote reali (`dem.py`)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import dem as DEM
from genworld import mappa as M
from genworld import vulcani as V


def griglia(ny=120, nx=100):
    e = np.full((ny, nx), -500.0, np.float32)                 # mare profondo
    e[:, 40:] = 30.0                                           # pianura costiera
    e[:, 60:] = 400.0
    e[20:60, 80:95] = 3200.0                                   # un massiccio
    lago = np.zeros((ny, nx), bool)
    lago[90:100, 45:55] = True
    e[lago] = 65.0
    return {"elev": e, "lago": lago, "x0": 4_000_000.0, "y1": 2_500_000.0, "passo": 300.0}


class TestAMondo(unittest.TestCase):
    def test_forme_e_centramento(self):
        r = DEM.a_mondo(griglia(), 200)
        self.assertEqual(r["cls"].shape, (200, 200))
        self.assertEqual(r["h"].shape, (200, 200))
        self.assertEqual(r["oz"], 0)                           # il lato lungo e' z
        self.assertGreater(r["ox"], 0)
        self.assertAlmostEqual(r["passo"], 300.0 * 120 / 200, places=3)

    def test_classi_per_quota(self):
        r = DEM.a_mondo(griglia(), 120)
        c, oz, ox = r["cls"], r["oz"], r["ox"]
        self.assertEqual(c[60, ox + 5], M.OCEANO)
        self.assertEqual(c[60, ox + 50], M.PIANURA)
        self.assertEqual(c[100, ox + 80], M.PRATERIA)
        self.assertIn(c[40, ox + 88], (M.NEVE, M.MONTAGNA))
        self.assertEqual(c[40, ox + 88], M.NEVE)

    def test_quote_in_blocchi(self):
        r = DEM.a_mondo(griglia(), 120)
        h, ox = r["h"], r["ox"]
        self.assertGreater(int(h[60, ox + 50]), DEM.LIVELLO_MARE)           # terra sopra il mare
        self.assertLess(int(h[60, ox + 5]), DEM.LIVELLO_MARE)               # mare sotto
        self.assertAlmostEqual(float(h[40, ox + 88]), DEM.LIVELLO_MARE + 3200 / DEM.METRI_PER_BLOCCO, delta=2)
        self.assertGreaterEqual(int(h.min()), DEM.LIVELLO_MARE - DEM.PROFONDITA_MASSIMA - 2)

    def test_il_fondale_e_compresso_non_lineare(self):
        e = np.full((60, 60), -100.0, np.float32)
        e[:, :30] = -3000.0
        e[:, 50:] = 10.0
        r = DEM.a_mondo({"elev": e, "lago": np.zeros(e.shape, bool), "x0": 0.0, "y1": 0.0,
                         "passo": 300.0}, 60)
        h = r["h"]
        prof_bassa = DEM.LIVELLO_MARE - int(h[30, 40])
        prof_alta = DEM.LIVELLO_MARE - int(h[30, 5])
        self.assertGreater(prof_alta, prof_bassa)
        self.assertLess(prof_alta, 3000 / DEM.METRI_PER_BLOCCO)             # molto meno che lineare


class TestLaghi(unittest.TestCase):
    def test_lago_con_pelo_costante_e_fondo(self):
        g = griglia()
        r = DEM.a_mondo(g, 120)
        livello = np.full(r["cls"].shape, 62.0, np.float32)
        cls, h, liv = DEM.applica_laghi(r["cls"], r["h"], livello, r)
        m = r["lago"]
        self.assertTrue(m.any())
        self.assertTrue((cls[m] == M.FIUME).all())
        self.assertEqual(len(np.unique(liv[m])), 1)
        self.assertAlmostEqual(float(liv[m][0]), DEM.LIVELLO_MARE + 65 / DEM.METRI_PER_BLOCCO, delta=0.6)
        self.assertTrue((h[m] < liv[m]).all())
        self.assertTrue((h[~m] == r["h"][~m]).all())


class TestVulcanoReale(unittest.TestCase):
    def test_non_aggiunge_un_cono_e_scava_il_cratere(self):
        h = np.full((200, 200), 70, np.int32)
        h[90:110, 90:110] = 180                                   # la cima c'e' gia'
        cls = np.full(h.shape, M.PRATERIA, np.uint8)
        v = V.scegli_siti(cls, h, posizioni=[(100, 100, 40, 5, 4)])
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].reale, 180.0)
        c2, h2, lava, q = V.modella(cls, h, v)
        # lontano dal cratere il terreno resta com'era (niente altopiano)
        self.assertEqual(int(h2[100, 130]), 70)
        self.assertEqual(int(h2[100, 60]), 70)
        # il cratere e' piu' basso della cima e ha lava
        self.assertLess(int(h2[100, 100]), 180)
        self.assertTrue(lava.any())
        self.assertEqual(c2[100, 120], M.VULCANO)

    def test_senza_extra_il_vulcano_resta_quello_di_prima(self):
        cls = np.full((300, 300), M.PIANURA, np.uint8)
        h = np.full(cls.shape, 80, np.int32)
        v = V.scegli_siti(cls, h, posizioni=[(150, 150)])
        self.assertIsNone(v[0].reale)


class TestVulcanoSpento(unittest.TestCase):
    def terreno(self):
        h = np.full((200, 200), 70, np.int32)
        h[90:110, 90:110] = 150
        cls = np.full(h.shape, M.FORESTA, np.uint8)
        return cls, h

    def test_niente_lava_ne_classe_vulcano_ma_il_cratere_c_e(self):
        cls, h = self.terreno()
        v = V.scegli_siti(cls, h, posizioni=[(100, 100, 30, 4, 4, 1, 0)])
        self.assertTrue(v[0].spento)
        self.assertFalse(v[0].lago)
        c2, h2, lava, q = V.modella(cls, h, v)
        self.assertFalse(lava.any())
        self.assertFalse((c2 == M.VULCANO).any())
        self.assertTrue((c2 == M.FORESTA).all())                # il bosco resta
        self.assertLess(int(h2[100, 100]), 150)                  # il cratere e' un invaso
        self.assertEqual(int(h2[100, 125]), 70)

    def test_il_vulcano_attivo_resta_attivo(self):
        cls, h = self.terreno()
        v = V.scegli_siti(cls, h, posizioni=[(100, 100, 30, 4, 4, 0, 0)])
        self.assertFalse(v[0].spento)
        c2, h2, lava, q = V.modella(cls, h, v)
        self.assertTrue(lava.any())
        self.assertTrue((c2 == M.VULCANO).any())

    def test_lago_nel_cratere(self):
        cls, h = self.terreno()
        v = V.scegli_siti(cls, h, posizioni=[(100, 100, 30, 4, 5, 1, 1)])
        c2, h2, lava, q = V.modella(cls, h, v)
        livello = np.full(h.shape, 62.0, np.float32)
        c3, h3, liv3 = V.laghi_dei_crateri(c2, h2, livello, v)
        lago = c3 == M.FIUME
        self.assertTrue(lago.any())
        self.assertTrue(lago[100, 100])
        pelo = np.unique(liv3[lago])
        self.assertEqual(len(pelo), 1)
        self.assertLess(float(pelo[0]), 150)                     # sotto l'orlo
        self.assertTrue((h3[lago] < liv3[lago]).all())           # il fondo sta sotto il pelo
        # un vulcano spento senza lago non cambia niente
        v2 = V.scegli_siti(cls, h, posizioni=[(100, 100, 30, 4, 5, 1, 0)])
        c4, h4, l4 = V.laghi_dei_crateri(c2, h2, livello, v2)
        self.assertTrue((c4 == c2).all())


class TestLatitudine(unittest.TestCase):
    def test_il_limite_delle_nevi_scende_andando_a_nord(self):
        e = np.full((40, 40), 1500.0, np.float32)
        e[:, :5] = -100.0
        lat = np.tile(np.linspace(40, 70, 40, dtype=np.float32)[:, None], (1, 40))
        g = {"elev": e, "lago": np.zeros(e.shape, bool), "x0": 0.0, "y1": 0.0, "passo": 1000.0,
             "lat": lat[::2, ::2][:20, :20].copy(), "lat_sub": 2}
        r = DEM.a_mondo(g, 40)
        c = r["cls"]
        self.assertNotEqual(c[2, 20], M.NEVE)                   # a 40 N 1500 m non e' neve
        self.assertEqual(c[38, 20], M.NEVE)                     # a 70 N si'


class TestLonLat(unittest.TestCase):
    def test_frazioni_da_lonlat(self):
        try:
            import rasterio  # noqa: F401
        except ImportError:
            self.skipTest("rasterio non installato")
        from rasterio.warp import transform
        x, y = transform("EPSG:4326", "EPSG:3035", [12.5], [41.9])
        g = griglia(100, 80)
        g["x0"], g["y1"], g["passo"] = x[0] - 40 * 300, y[0] + 50 * 300, 300.0
        f = DEM.frazioni_da_lonlat(g, 100, [(12.5, 41.9)])
        self.assertAlmostEqual(f[0][0], (40 + (100 - 80) / 2) / 100, places=2)
        self.assertAlmostEqual(f[0][1], 0.5, places=2)


class TestDemInOpzioni(unittest.TestCase):
    def test_il_contenuto_del_dem_cambia_la_chiave_di_cache(self):
        from genworld.motore import Opzioni
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.npz")
            open(p, "wb").write(b"a")
            a = Opzioni(immagine="", uscita="u", dem=p).firma_analisi()
            open(p, "wb").write(b"bb")
            b = Opzioni(immagine="", uscita="u", dem=p).firma_analisi()
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
