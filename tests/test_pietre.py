"""Pietre per bioma (#30): la mesa a fasce e il vulcano lucido."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import biomi as B
from genworld import mappa as M
from genworld import pietre as PI

Y0, ALT = 0, 200


class Finto:
    def __init__(self):
        self.ids = {("air", ()): 0}
        self.id_aria = 0

    def blocco(self, nome, **pr):
        return self.ids.setdefault((nome, tuple(sorted(pr.items()))), len(self.ids))

    def nome(self, i):
        for (n, p), v in self.ids.items():
            if v == i:
                return n + ("" if not p else str(dict(p)))
        return "?"


def colonne(s, quota=100):
    out = np.zeros((16, ALT, 16), np.uint32)
    out[:, :quota, :] = s.blocco("stone")
    return out, np.full((16, 16), quota, np.int32)


class TestMesa(unittest.TestCase):
    def test_sabbia_rossa_e_fasce_di_terracotta(self):
        s = Finto()
        out, h = colonne(s)
        cls = np.full((16, 16), M.DESERTO, np.uint8)
        bio = np.full((16, 16), B.INDICE["badlands"], np.uint8)
        self.assertEqual(PI.mesa(out, s, h, cls, bio, None, Y0), 256)
        self.assertEqual(s.nome(int(out[3, 99, 3])), "red_sand")
        nomi = {s.nome(int(i)) for i in out[3, 80:98, 3]}
        self.assertGreaterEqual(len(nomi), 4, nomi)
        self.assertTrue(all("terracotta" in n or "sandstone" in n for n in nomi), nomi)

    def test_le_fasce_dipendono_solo_dalla_quota(self):
        s = Finto()
        out, h = colonne(s)
        cls = np.full((16, 16), M.DESERTO, np.uint8)
        bio = np.full((16, 16), B.INDICE["badlands"], np.uint8)
        PI.mesa(out, s, h, cls, bio, None, Y0)
        self.assertTrue((out[0, 70:96, 0] == out[9, 70:96, 12]).all())

    def test_il_vulcano_e_il_deserto_comune_non_sono_mesa(self):
        s = Finto()
        for classe, bioma in ((M.VULCANO, "badlands"), (M.DESERTO, "desert")):
            out, h = colonne(s)
            prima = out.copy()
            cls = np.full((16, 16), classe, np.uint8)
            bio = np.full((16, 16), B.INDICE[bioma], np.uint8)
            self.assertEqual(PI.mesa(out, s, h, cls, bio, None, Y0), 0)
            self.assertTrue((out == prima).all())

    def test_una_parete_nuda_mostra_le_fasce_anche_in_cima(self):
        s = Finto()
        out, h = colonne(s)
        cls = np.full((16, 16), M.DESERTO, np.uint8)
        bio = np.full((16, 16), B.INDICE["badlands"], np.uint8)
        PI.mesa(out, s, h, cls, bio, np.ones((16, 16), bool), Y0)
        self.assertNotEqual(s.nome(int(out[3, 99, 3])), "red_sand")


class TestVulcano(unittest.TestCase):
    def test_pietra_lucida_e_obsidiana(self):
        s = Finto()
        trovati = set()
        for ox in range(0, 800, 16):
            out, h = colonne(s)
            cls = np.full((16, 16), M.VULCANO, np.uint8)
            cls[:, 8:] = M.CRATERE
            PI.vulcano(out, s, h, cls, None, ox, 0, Y0)
            trovati |= {s.nome(int(i)) for i in np.unique(out)}
        for atteso in ("polished_blackstone", "obsidian", "crying_obsidian"):
            self.assertIn(atteso, trovati)

    def test_la_lava_non_si_tocca(self):
        s = Finto()
        out, h = colonne(s)
        prima = out.copy()
        cls = np.full((16, 16), M.CRATERE, np.uint8)
        PI.vulcano(out, s, h, cls, np.ones((16, 16), bool), 0, 0, Y0)
        self.assertTrue((out == prima).all())

    def test_fuori_dal_vulcano_niente(self):
        s = Finto()
        out, h = colonne(s)
        prima = out.copy()
        PI.vulcano(out, s, h, np.full((16, 16), M.PIANURA, np.uint8), None, 0, 0, Y0)
        self.assertTrue((out == prima).all())


class TestPalude(unittest.TestCase):
    def test_fango_e_argilla_solo_nella_palude(self):
        s = Finto()
        trovati = set()
        for ox in range(0, 800, 16):
            out, h = colonne(s)
            cls = np.full((16, 16), M.PIANURA, np.uint8)
            bio = np.full((16, 16), B.INDICE["mangrove_swamp"], np.uint8)
            PI.palude(out, s, h, cls, bio, ox, 0, Y0)
            trovati |= {s.nome(int(i)) for i in np.unique(out)}
        self.assertTrue({"mud", "clay"} <= trovati, trovati)
        out, h = colonne(s)
        prima = out.copy()
        PI.palude(out, s, h, np.full((16, 16), M.PIANURA, np.uint8),
                  np.full((16, 16), B.INDICE["plains"], np.uint8), 0, 0, Y0)
        self.assertTrue((out == prima).all())


class TestTraduzioni(unittest.TestCase):
    def test_i_blocchi_si_traducono(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        for nome, pr, java in (("terracotta", {}, "terracotta"),
                               ("stained_terracotta", {"color": "orange"}, "orange_terracotta"),
                               ("stained_terracotta", {"color": "light_gray"}, "light_gray_terracotta"),
                               ("red_sand", {}, "red_sand"),
                               ("red_sandstone", {"variant": "normal"}, "red_sandstone"),
                               ("polished_blackstone", {}, "polished_blackstone"),
                               ("gilded_blackstone", {}, "gilded_blackstone"),
                               ("obsidian", {}, "obsidian"), ("crying_obsidian", {}, "crying_obsidian")):
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in pr.items()})
            o, _, _ = v.block.from_universal(b)
            self.assertEqual(o.base_name, java)


if __name__ == "__main__":
    unittest.main()
