"""Anteprima (#18): ombreggiatura e render di un mondo piccolo."""
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genworld import anteprima as AN  # noqa: E402


class TestOmbreggia(unittest.TestCase):
    def test_forza_zero_non_cambia_niente(self):
        rgb = np.full((8, 8, 3), 100, np.uint8)
        alt = np.tile(np.arange(8, dtype=np.float32), (8, 1)) + 60
        self.assertTrue(np.array_equal(AN.ombreggia(rgb, alt, 0.0), rgb))

    def test_piatto_a_quota_uniforme_resta_uniforme(self):
        rgb = np.full((8, 8, 3), 100, np.uint8)
        alt = np.full((8, 8), 70, np.float32)
        out = AN.ombreggia(rgb, alt, 0.5)
        self.assertEqual(len(np.unique(out)), 1)
        self.assertEqual(out.dtype, np.uint8)

    def test_il_pendio_cambia_la_luce_e_la_quota_la_tinta(self):
        rgb = np.full((8, 8, 3), 100, np.uint8)
        alt = np.tile(np.arange(8, dtype=np.float32) * 4, (8, 1)) + 60
        out = AN.ombreggia(rgb, alt, 0.5)
        self.assertGreater(len(np.unique(out)), 1)
        # piu' in alto = piu' chiaro (a pari pendenza)
        self.assertGreater(int(out[4, 6, 0]), int(out[4, 1, 0]))

    def test_senza_terra_non_esplode(self):
        rgb = np.zeros((4, 4, 3), np.uint8)
        out = AN.ombreggia(rgb, np.zeros((4, 4), np.float32), 1.0)
        self.assertEqual(out.shape, (4, 4, 3))

    def test_i_colori_sono_rgb_validi(self):
        for nome, c in AN.COLORI.items():
            self.assertEqual(len(c), 3, nome)
            self.assertTrue(all(0 <= v <= 255 for v in c), nome)


class TestRendi(unittest.TestCase):
    def test_un_mondo_piccolo(self):
        from genworld.motore import Opzioni, genera
        from test_motore import mappa_finta
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "i.png"), 128)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=64, vulcani=0,
                         alberi=0.0, villaggi=0.0, strade=False)
            genera(op)
            png = os.path.join(d, "a.png")
            st = AN.rendi(op.uscita, png, ombreggiatura=0.0)
            self.assertEqual((st["larghezza"], st["altezza"]), (64, 64))
            self.assertEqual(st["chunk"], 16)
            self.assertEqual(st["origine"], (-32, -32))
            self.assertEqual(Image.open(png).size, (64, 64))
            self.assertEqual(st["altezze"].shape, (64, 64))
            self.assertGreater(st["quota_max"], st["quota_min"])
            self.assertEqual(st["blocchi_ignoti"], [])
            # senza ombreggiatura i pixel sono i colori della tabella: l'oceano
            # (al centro non c'e') o la terra; ai bordi la mappa e' mare
            px = tuple(Image.open(png).getpixel((0, 0)))
            self.assertIn(px, set(AN.COLORI.values()))

    def test_mondo_vuoto_da_errore(self):
        import amulet
        from genworld.livello_dat import ImpostazioniMondo
        from genworld.mondo import ScrittoreMondo
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "w")
            with ScrittoreMondo(w, ImpostazioniMondo(), lato_blocchi=32, crea=True):
                pass
            with self.assertRaises(RuntimeError):
                AN.rendi(w, os.path.join(d, "a.png"))
            _ = amulet  # il livello si chiude da solo in rendi()


if __name__ == "__main__":
    unittest.main()
