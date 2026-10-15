"""Neve sottile sui tetti e sulle chiome (#28)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import biomi as B
from genworld import neve as NV


class Finto:
    def __init__(self):
        self.ids = {("air", ()): 0}
        self.id_aria = 0

    def blocco(self, nome, **pr):
        return self.ids.setdefault((nome, tuple(sorted(pr.items()))), len(self.ids))

    def nome_blocco(self, i):
        return next(n for (n, _), v in self.ids.items() if v == i)


def chunk(s, nome):
    out = np.zeros((16, 80, 16), np.uint32)
    out[:, :40, :] = s.blocco("stone")
    out[4:8, 40:46, 4:8] = s.blocco(nome)
    return out


class TestNeve(unittest.TestCase):
    def test_la_neve_si_posa_sulle_foglie_nei_biomi_nevosi(self):
        s = Finto()
        out = chunk(s, "leaves")
        bio = np.full((16, 16), B.INDICE["snowy_taiga"], np.uint8)
        n = NV.neve_sui_tetti(out, s, bio, 0, 0)
        self.assertGreater(n, 10)
        neve = s.blocco("snow", layers="1")
        self.assertTrue((out[4:8, 46, 4:8] == neve).sum() >= 10)

    def test_non_nevica_fuori_dai_biomi_nevosi(self):
        s = Finto()
        out = chunk(s, "leaves")
        prima = out.copy()
        bio = np.full((16, 16), B.INDICE["plains"], np.uint8)
        self.assertEqual(NV.neve_sui_tetti(out, s, bio, 0, 0), 0)
        self.assertTrue((out == prima).all())

    def test_niente_neve_su_blocchi_non_pieni_o_su_fiori(self):
        s = Finto()
        out = chunk(s, "oak_stairs")
        prima = out.copy()
        bio = np.full((16, 16), B.INDICE["snowy_tundra"], np.uint8)
        NV.neve_sui_tetti(out, s, bio, 0, 0)
        self.assertTrue((out[4:8, 40:46, 4:8] == prima[4:8, 40:46, 4:8]).all())
        self.assertTrue((out[4:8, 46, 4:8] == 0).all())

    def test_il_manto_non_copre_la_neve_gia_posata_ne_sporge_in_aria_piena(self):
        s = Finto()
        out = chunk(s, "leaves")
        out[4:8, 46, 4:8] = s.blocco("snow", layers="1")      # gia' innevato
        prima = out.copy()
        bio = np.full((16, 16), B.INDICE["snowy_taiga"], np.uint8)
        NV.neve_sui_tetti(out, s, bio, 0, 0)
        self.assertTrue((out[4:8, 46:48, 4:8] == prima[4:8, 46:48, 4:8]).all())

    def test_e_deterministico_e_dipende_dalle_coordinate_di_mondo(self):
        a, b, c = Finto(), Finto(), Finto()
        bio = np.full((16, 16), B.INDICE["snowy_taiga"], np.uint8)
        oa, ob, oc = chunk(a, "leaves"), chunk(b, "leaves"), chunk(c, "leaves")
        NV.neve_sui_tetti(oa, a, bio, 32, 48)
        NV.neve_sui_tetti(ob, b, bio, 32, 48)
        NV.neve_sui_tetti(oc, c, bio, 320, 480)
        self.assertTrue((oa == ob).all())
        self.assertFalse((oa == oc).all())


if __name__ == "__main__":
    unittest.main()
