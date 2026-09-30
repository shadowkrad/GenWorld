"""Un mondo vero, dall'immagine ai file, riletto da disco.

I test dei singoli moduli controllano i pezzi; questo controlla che la
pipeline intera li metta d'accordo: una mappa finta con un villaggio produce un
mondo in cui ci sono le case, una campana, dei lampioni, e che l'anteprima sa
disegnare.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld.motore import Opzioni, genera  # noqa: E402

try:
    import amulet  # noqa: F401
    from genworld.anteprima import rendi
    AMULET = True
except ImportError:
    AMULET = False

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRUTTURE = os.path.join(RADICE, "templates", "strutture")
HA_CASE = os.path.isdir(STRUTTURE) and any(f.endswith(".nbt") for f in os.listdir(STRUTTURE))


def mappa_finta(percorso: str, lato: int = 192) -> str:
    zz, xx = np.mgrid[:lato, :lato]
    d = np.hypot(zz - lato / 2, xx - lato / 2)
    rgb = np.zeros((lato, lato, 3), np.uint8)
    rgb[...] = (40, 80, 160)
    rgb[d < lato * 0.40] = (90, 150, 80)
    rgb[d < lato * 0.15] = (170, 170, 165)
    Image.fromarray(rgb).save(percorso)
    return percorso


@unittest.skipUnless(AMULET and HA_CASE, "amulet o templates/strutture non disponibili")
class TestMondoIntero(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        img = mappa_finta(os.path.join(cls.dir.name, "isola.png"))
        cls.op = Opzioni(immagine=img, uscita=os.path.join(cls.dir.name, "mondo"),
                         lato=192, vulcani=0, fiumi=80.0, villaggi=1.0,
                         templates=STRUTTURE, seed=11)
        cls.st = genera(cls.op)
        cls.nomi = cls._conta_blocchi(cls.op.uscita)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    @staticmethod
    def _conta_blocchi(percorso: str) -> dict[str, int]:
        lv = amulet.load_level(percorso)
        conteggio: dict[str, int] = {}
        try:
            pal = lv.block_palette
            for cx, cz in lv.all_chunk_coords("minecraft:overworld"):
                ch = lv.get_chunk(cx, cz, "minecraft:overworld")
                valori, quanti = np.unique(np.asarray(ch.blocks[:, 40:140, :]),
                                           return_counts=True)
                for v, q in zip(valori.tolist(), quanti.tolist()):
                    n = pal[int(v)].base_name
                    conteggio[n] = conteggio.get(n, 0) + q
        finally:
            lv.close()
        return conteggio

    def test_il_mondo_e_completo(self):
        self.assertEqual(self.st["restano"], 0)
        self.assertGreater(self.st["chunk"], 100)

    def test_ci_sono_le_case(self):
        costruito = sum(self.nomi.get(n, 0) for n in ("planks", "stairs", "log", "wood"))
        self.assertGreater(costruito, 200)

    def test_ogni_insediamento_ha_una_campana(self):
        self.assertGreaterEqual(self.nomi.get("bell", 0), 1)

    def test_ci_sono_lampioni_o_arredi(self):
        self.assertGreater(self.nomi.get("lantern", 0) + self.nomi.get("fence", 0), 0)

    def test_senza_arredi_non_c_e_la_campana(self):
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"))
            op = Opzioni(immagine=img, uscita=os.path.join(d, "mondo"), lato=192,
                         vulcani=0, fiumi=80.0, villaggi=1.0, templates=STRUTTURE,
                         seed=11, arredi=0.0)
            genera(op)
            self.assertEqual(self._conta_blocchi(op.uscita).get("bell", 0), 0)

    def test_l_anteprima_disegna_il_mondo(self):
        with tempfile.TemporaryDirectory() as d:
            png = os.path.join(d, "anteprima.png")
            st = rendi(self.op.uscita, png, y0=-64, y1=220)
            self.assertTrue(os.path.isfile(png))
            im = Image.open(png)
            self.assertEqual(im.size, (st["larghezza"], st["altezza"]))
            self.assertEqual(st["chunk"], self.st["totale"])
            self.assertGreater(len(set(im.convert("RGB").getdata())), 5)


if __name__ == "__main__":
    unittest.main()
