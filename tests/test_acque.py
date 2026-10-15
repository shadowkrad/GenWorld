"""Vita dell'acqua (#28): ghiaccio, coralli, alghe, ninfee, argilla."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import acque as AC
from genworld import biomi as B
from genworld import mappa as M

Y0, ALT = 0, 100


class Finto:
    """Id finti, uno per (nome, proprieta'): niente livello aperto."""
    def __init__(self):
        self.ids = {}
        self.id_aria = 0
        self.ids[("air", ())] = 0

    def blocco(self, nome, **pr):
        k = (nome, tuple(sorted(pr.items())))
        if k not in self.ids:
            self.ids[k] = len(self.ids) + 1
        return self.ids[k]

    def nome(self, i):
        for (n, _), v in self.ids.items():
            if v == i:
                return n
        return "?"


def vasca(s, bioma, fondale=40, mare=62, cls=M.MARE, lato=16):
    """Un chunk: pietra fino a `fondale`, acqua fino a `mare`, sabbia sul fondo."""
    out = np.zeros((16, ALT, 16), np.uint32)
    out[:, :fondale - 1, :] = s.blocco("stone")
    out[:, fondale - 1, :] = s.blocco("sand")
    out[:, fondale:mare + 1, :] = s.blocco("water")
    h = np.full((16, 16), fondale, np.int32)
    liv = np.full((16, 16), mare, np.int32)
    c = np.full((16, 16), cls, np.uint8)
    b = np.full((16, 16), B.INDICE[bioma], np.uint8)
    return out, h, liv, c, b


def nomi_in(s, out):
    return {s.nome(int(i)) for i in np.unique(out)}


class TestAcque(unittest.TestCase):
    def test_il_mare_gelato_ha_il_ghiaccio_in_superficie(self):
        s = Finto()
        out, h, liv, c, b = vasca(s, "frozen_ocean", fondale=61)
        AC.decora(out, s, h, liv, c, b, 0, 0, Y0)
        self.assertEqual({s.nome(int(i)) for i in np.unique(out[:, 62, :])},
                         {"ice", "packed_ice"})
        self.assertEqual(nomi_in(s, out[:, 61:62, :]), {"water"})

    def test_i_mari_caldi_hanno_coralli(self):
        s = Finto()
        trovati = set()
        for x0 in range(0, 400, 16):
            out, h, liv, c, b = vasca(s, "warm_ocean", fondale=55)
            AC.decora(out, s, h, liv, c, b, x0, 0, Y0)
            trovati |= nomi_in(s, out)
        self.assertIn("coral_block", trovati)
        self.assertTrue({"coral", "coral_fan"} & trovati)

    def test_il_mare_normale_non_ha_coralli_ne_ghiaccio(self):
        s = Finto()
        out, h, liv, c, b = vasca(s, "ocean")
        AC.decora(out, s, h, liv, c, b, 0, 0, Y0)
        n = nomi_in(s, out)
        self.assertFalse(n & {"coral_block", "ice", "packed_ice"})

    def test_alghe_e_erba_di_mare_nel_mare_aperto(self):
        s = Finto()
        trovati = set()
        for x0 in range(0, 800, 16):
            out, h, liv, c, b = vasca(s, "ocean", fondale=45)
            AC.decora(out, s, h, liv, c, b, x0, 0, Y0)
            trovati |= nomi_in(s, out)
        self.assertIn("kelp", trovati)
        self.assertIn("seagrass", trovati)

    def test_le_alghe_stanno_nell_acqua_e_sul_fondo(self):
        s = Finto()
        for x0 in range(0, 800, 16):
            out, h, liv, c, b = vasca(s, "ocean", fondale=45)
            AC.decora(out, s, h, liv, c, b, x0, 0, Y0)
            for lx in range(16):
                for lz in range(16):
                    col = out[lx, :, lz]
                    for y in np.nonzero(col)[0]:
                        if s.nome(int(col[y])) in ("kelp", "kelp_plant"):
                            sotto = s.nome(int(col[y - 1]))
                            self.assertIn(sotto, ("sand", "gravel", "clay", "kelp_plant"))
                            self.assertLess(y, 63)

    def test_le_ninfee_stanno_sul_pelo_dell_acqua_dei_fiumi(self):
        s = Finto()
        for x0 in range(0, 1600, 16):
            out, h, liv, c, b = vasca(s, "river", fondale=60, cls=M.FIUME)
            AC.decora(out, s, h, liv, c, b, x0, 0, Y0)
            for lx, lz in zip(*np.nonzero(out[:, 63, :] != 0)):
                self.assertEqual(s.nome(int(out[lx, 63, lz])), "lily_pad")
                self.assertEqual(s.nome(int(out[lx, 62, lz])), "water")

    def test_i_fiumi_hanno_argilla_o_fango(self):
        s = Finto()
        trovati = set()
        for x0 in range(0, 1600, 16):
            out, h, liv, c, b = vasca(s, "river", fondale=60, cls=M.FIUME)
            AC.decora(out, s, h, liv, c, b, x0, 0, Y0)
            trovati |= nomi_in(s, out)
        self.assertTrue({"clay", "mud"} & trovati)

    def test_il_vulcano_e_la_terraferma_non_si_toccano(self):
        s = Finto()
        out, h, liv, c, b = vasca(s, "warm_ocean", cls=M.CRATERE)
        prima = out.copy()
        self.assertEqual(AC.decora(out, s, h, liv, c, b, 0, 0, Y0), 0)
        self.assertTrue((out == prima).all())
        out = np.zeros((16, ALT, 16), np.uint32)
        out[:, :70, :] = 1
        h = np.full((16, 16), 70, np.int32)
        self.assertEqual(AC.decora(out, s, h, np.full((16, 16), 62), c, b, 0, 0, Y0), 0)

    def test_e_deterministico(self):
        s = Finto()
        a = vasca(s, "warm_ocean"); b_ = vasca(s, "warm_ocean")
        AC.decora(a[0], s, *a[1:], 32, 48, Y0); AC.decora(b_[0], s, *b_[1:], 32, 48, Y0)
        self.assertTrue((a[0] == b_[0]).all())


class TestTraduzioni(unittest.TestCase):
    def test_i_blocchi_nuovi_si_traducono(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        for nome, pr, java in (
                ("seagrass", {}, "seagrass"), ("kelp_plant", {"age": "25"}, "kelp_plant"),
                ("kelp", {"age": "3"}, "kelp"), ("lily_pad", {}, "lily_pad"),
                ("ice", {}, "ice"), ("packed_ice", {}, "packed_ice"),
                ("clay", {}, "clay"), ("mud", {}, "mud"),
                ("tall_seagrass", {"half": "lower"}, "tall_seagrass"),
                ("coral_block", {"coral_type": "fire", "dead": "false"}, "fire_coral_block"),
                ("coral", {"coral_type": "tube", "dead": "false"}, "tube_coral"),
                ("coral_fan", {"coral_type": "brain", "facing": "up", "dead": "false"},
                 "brain_coral_fan"),
                ("sea_pickle", {"pickles": "2", "dead": "false"}, "sea_pickle")):
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in pr.items()})
            o, _, _ = v.block.from_universal(b)
            self.assertEqual(o.base_name, java, nome)


if __name__ == "__main__":
    unittest.main()
