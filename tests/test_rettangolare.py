"""Mondi rettangolari (`lato` x `altezza`): per le carte 2:1 come la Terra."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import dem as DEM
from genworld import parallelo as PA
from genworld.motore import Opzioni


class TestGeometria(unittest.TestCase):
    def test_coordinate_chunk_quadrate_come_prima(self):
        c = PA.coordinate_chunk(64)
        self.assertEqual(len(c), 16)
        self.assertEqual(min(c), (-2, -2))
        self.assertEqual(max(c), (1, 1))

    def test_coordinate_chunk_rettangolari(self):
        c = PA.coordinate_chunk(128, 64)                   # 8 x 4 chunk
        self.assertEqual(len(c), 32)
        self.assertEqual(min(c), (-4, -4))
        self.assertEqual(max(c), (3, -1))                  # in z da -lato/2: l'origine resta in x

    def test_opzioni_alt_e_spawn(self):
        q = Opzioni(immagine="", uscita="u", lato=1000)
        self.assertEqual((q.alt, q.spawn), (1000, (0, 150, 0)))
        r = Opzioni(immagine="", uscita="u", lato=1000, altezza=500)
        self.assertEqual(r.alt, 500)
        self.assertEqual(r.spawn, (0, 150, -250))          # il centro del mondo, non l'origine
        self.assertNotEqual(q.firma_analisi(), r.firma_analisi())

    def test_la_grafica_dello_scrittore_segue_altezza(self):
        from genworld.livello_dat import ImpostazioniMondo
        from genworld.mondo import ScrittoreMondo
        with tempfile.TemporaryDirectory() as d:
            with ScrittoreMondo(os.path.join(d, "w"), ImpostazioniMondo(versione=(26, 2, 0)),
                                lato_blocchi=128, altezza_blocchi=64, crea=True) as m:
                c = list(m.chunk_coords())
            self.assertEqual(c, PA.coordinate_chunk(128, 64))


class TestDemRettangolare(unittest.TestCase):
    def griglia(self):
        e = np.full((100, 200), 300.0, np.float32)         # una carta 2:1
        e[:, :20] = -400.0
        return {"elev": e, "lago": np.zeros(e.shape, bool), "x0": -1000.0, "y1": 500.0,
                "passo": 1000.0, "crs": "EPSG:3857", "metri_per_blocco": 36.0}

    def test_la_carta_2_a_1_riempie_un_mondo_2_a_1_senza_margini(self):
        r = DEM.a_mondo(self.griglia(), 400, 200)
        self.assertEqual(r["cls"].shape, (200, 400))
        self.assertEqual((r["oz"], r["ox"]), (0, 0))
        self.assertAlmostEqual(r["fattore"], 2.0)

    def test_in_un_mondo_piu_alto_si_centra(self):
        r = DEM.a_mondo(self.griglia(), 200, 200)
        self.assertEqual(r["cls"].shape, (200, 200))
        self.assertEqual(r["oz"], 50)

    def test_frazioni_lonlat_di_un_mondo_rettangolare(self):
        g = self.griglia()
        g["crs"] = "EPSG:4326"
        g["x0"], g["y1"], g["passo"] = -100.0, 50.0, 1.0           # -100..100 E, 50..-50 N
        f = DEM.frazioni_da_lonlat(g, 400, [(0.0, 0.0), (-100.0, 50.0), (99.0, -49.0)], 200)
        self.assertAlmostEqual(f[0][0], 0.5)
        self.assertAlmostEqual(f[0][1], 0.5)
        self.assertAlmostEqual(f[1][0], 0.0)
        self.assertAlmostEqual(f[1][1], 0.0)
        self.assertGreater(f[2][0], 0.95)
        self.assertGreater(f[2][1], 0.95)


class TestMondoCompleto(unittest.TestCase):
    def test_un_mondo_rettangolare_si_genera_e_ha_i_chunk_giusti(self):
        import amulet
        from genworld.motore import genera
        with tempfile.TemporaryDirectory() as d:
            dem = os.path.join(d, "dem.npz")
            e = np.full((96, 160), 120.0, np.float32)
            e[:, :30] = -300.0
            np.savez_compressed(dem, elev=e, lago=np.zeros(e.shape, bool), x0=0.0, y1=0.0, passo=1000.0,
                                nx=160, ny=96, crs="EPSG:3035")
            op = Opzioni(immagine="", uscita=os.path.join(d, "w"), nome="T", lato=160, altezza=96, dem=dem,
                         versione=(26, 2, 0), vulcani=0, villaggi=0, strade=False, miniere=0, dungeon=0,
                         torri=0, fortezze=0, castelli_rovina=0, isole=0, siti=0, accampamenti=0,
                         cimiteri=0, portali=0, incarichi=False, alberi=0, caverne=0, laghi=0,
                         fiumi=0, arredi=0, isolate=0, fauna=0, dimensioni=False, fuori_mappa="vuoto")
            st = genera(op)
            self.assertEqual(st["chunk"], 10 * 6)
            lv = amulet.load_level(op.uscita)
            try:
                for cx, cz in ((-5, -5), (4, 0), (0, -3)):
                    ch = lv.get_chunk(cx, cz, "minecraft:overworld")
                    self.assertGreater(int((np.asarray(ch.blocks) != 0).sum()), 0)
                with self.assertRaises(Exception):
                    lv.get_chunk(0, 1, "minecraft:overworld")         # oltre l'altezza: non c'e'
            finally:
                lv.close()


if __name__ == "__main__":
    unittest.main()
