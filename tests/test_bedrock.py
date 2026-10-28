"""Esportazione in Bedrock: un mondo minuscolo diventa un .mcworld che amulet rilegge."""
import os
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import bedrock
from genworld.livello_dat import ImpostazioniMondo
from genworld.mondo import ScrittoreMondo
from genworld.motore import blocchi_chunk


class TestBedrock(unittest.TestCase):
    def test_un_mondo_java_diventa_un_mcworld_leggibile(self):
        import amulet
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "j")
            with ScrittoreMondo(w, ImpostazioniMondo(versione=(26, 2, 0)), lato_blocchi=64, crea=True) as m:
                for cx, cz in m.chunk_coords():
                    h = np.full((16, 16), 70, np.int32)
                    cls = np.full((16, 16), 1, np.uint8)
                    m.scrivi_chunk(cx, cz, blocchi_chunk(m, h, cls, ox=cx * 16, oz=cz * 16), y0=-64)
            out = os.path.join(d, "prova.mcworld")
            r = bedrock.esporta(w, out, nome="Prova")
            self.assertEqual(r["minecraft:overworld"], 16)
            with zipfile.ZipFile(out) as z:
                nomi = set(z.namelist())
            self.assertIn("level.dat", nomi)
            self.assertTrue(any(n.startswith("db/") for n in nomi))
            dest = os.path.join(d, "estratto")
            with zipfile.ZipFile(out) as z:
                z.extractall(dest)
            lv = amulet.load_level(dest)
            try:
                ch = lv.get_chunk(0, 0, "minecraft:overworld")
                self.assertEqual(ch.block_palette[int(ch.blocks[0, 69, 0])].base_name, "sand")
                self.assertEqual(ch.block_palette[int(ch.blocks[0, 75, 0])].base_name, "air")
            finally:
                lv.close()


if __name__ == "__main__":
    unittest.main()
