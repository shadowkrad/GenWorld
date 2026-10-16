"""Sottosuolo (#29): gocciolatoi, grotta lussureggiante, sculk, geodi."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import sottosuolo as SS

Y0, ALT = -64, 320


class Finto:
    def __init__(self):
        self.ids = {("air", ()): 0}
        self.id_aria = 0

    def blocco(self, nome, **pr):
        k = (nome, tuple(sorted(pr.items())))
        return self.ids.setdefault(k, len(self.ids))

    def nome(self, i):
        return next((n for (n, _), v in self.ids.items() if v == i), "?")


def mondo(quota=120):
    s = Finto()
    tav = SS.Tavolozza(s)
    out = np.zeros((16, ALT, 16), np.uint32)
    ys = np.arange(Y0, Y0 + ALT)[None, :, None]
    out[np.broadcast_to(ys < quota, out.shape)] = tav.pietra
    h = np.full((16, 16), quota, np.int32)
    return s, tav, out, h


def caverna(r=6, y=-10):
    return {"x": np.array([8], np.int32), "y": np.array([y], np.int32),
            "z": np.array([8], np.int32), "r": np.array([r], np.float32)}


def nomi(s, out):
    return {s.nome(int(i)) for i in np.unique(out)}


class TestCaverne(unittest.TestCase):
    def trovati(self, y, seed_da=0, seed_a=60):
        s, tav, out, h = mondo()
        tot = set()
        for seed in range(seed_da, seed_a):
            s, tav, out, h = mondo()
            for ox in (0, 400, 800, 1200):
                o = out.copy()
                SS.posa(o, tav, h, ox, 0, Y0, caverna(6, y), [0], seed=seed)
                tot |= nomi(s, o)
        return tot

    def test_le_caverne_a_media_quota_hanno_gocciolatoi_o_muschio(self):
        t = self.trovati(-10)
        self.assertTrue(t & {"dripstone_block", "pointed_dripstone", "moss_block"}, t)

    def test_lo_sculk_solo_in_profondita(self):
        self.assertFalse(self.trovati(30) & {"sculk", "sculk_sensor"})
        self.assertTrue(self.trovati(-50) & {"sculk"})

    def test_il_muschio_solo_dove_non_e_troppo_profondo(self):
        self.assertFalse(self.trovati(-50) & {"moss_block", "cave_vines"})

    def test_senza_deco_la_posa_di_prima_non_cambia(self):
        s, tav, out, h = mondo()
        del tav.deco
        SS.posa(out, tav, h, 0, 0, Y0, caverna(), [0], seed=1)
        self.assertFalse(nomi(s, out) & {"dripstone_block", "moss_block", "sculk"})

    def test_le_punte_hanno_roccia_o_dripstone_da_cui_pendono(self):
        for seed in range(30):
            s, tav, out, h = mondo()
            SS.posa(out, tav, h, 0, 0, Y0, caverna(6, -10), [0], seed=seed)
            giu = s.blocco("pointed_dripstone", thickness="tip", vertical_direction="down")
            su = s.blocco("pointed_dripstone", thickness="tip", vertical_direction="up")
            for lx, ly, lz in zip(*np.nonzero(out == giu)):
                self.assertNotEqual(out[lx, ly + 1, lz], 0)
            for lx, ly, lz in zip(*np.nonzero(out == su)):
                self.assertNotEqual(out[lx, ly - 1, lz], 0)

    def test_non_decora_l_aria_scavata_da_altro(self):
        """Un tunnel di miniera (aria non scavata dalle caverne) resta com'e'."""
        s, tav, out, h = mondo()
        out[2:5, 100:104, :] = 0         # galleria artificiale
        prima = out.copy()
        SS.posa(out, tav, h, 0, 0, Y0, {"x": np.array([], np.int32), "y": np.array([], np.int32),
                                          "z": np.array([], np.int32), "r": np.array([], np.float32)},
                [], seed=1)
        gall = out[2:5, 100:104, :]
        self.assertTrue((gall == 0).all())
        self.assertEqual(prima.shape, out.shape)


class TestPozze(unittest.TestCase):
    def test_le_pozze_sono_chiuse_da_roccia_e_non_fluiscono(self):
        trovate = 0
        for seed in range(60):
            s, tav, out, h = mondo()
            SS.posa(out, tav, h, 0, 0, Y0, caverna(6, -10), [0], seed=seed)
            acqua = s.blocco("water")
            for lx, ly, lz in zip(*np.nonzero(out == acqua)):
                trovate += 1
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    self.assertNotEqual(out[lx + dx, ly, lz + dz], 0, "acqua con aria accanto")
                self.assertNotIn(int(out[lx, ly - 1, lz]), (0, int(acqua)), "acqua sul vuoto")
        self.assertGreater(trovate, 0, "mai una pozza in 60 caverne")


class TestRuscelli(unittest.TestCase):
    def acqua_nei_chunk(self, ox, oz, seed):
        s, tav, out, h = mondo()
        SS.posa(out, tav, h, ox, oz, Y0, caverna(6, -10), [0], seed=seed)
        return s, out, s.blocco("water")

    def test_i_ruscelli_sono_acqua_ferma_chiusa_da_roccia(self):
        celle = 0
        for seed in range(40):
            s, out, acqua = self.acqua_nei_chunk(0, 0, seed)
            for lx, ly, lz in zip(*np.nonzero(out == acqua)):
                celle += 1
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    self.assertNotEqual(out[lx + dx, ly, lz + dz], 0, "acqua con aria accanto")
                self.assertNotIn(int(out[lx, ly - 1, lz]), (0, int(acqua)), "acqua sul vuoto")
        self.assertGreater(celle, 0)

    def test_ci_sono_tratti_lunghi_di_ruscello(self):
        """Piu' celle d'acqua in fila (4 vicine fra loro) di quanto faccia una pozza."""
        lunghi = 0
        for seed in range(60):
            s, out, acqua = self.acqua_nei_chunk(0, 0, seed)
            m = out == acqua
            for ly in np.unique(np.nonzero(m)[1]):
                piano = m[:, ly, :]
                zz, xx = np.nonzero(piano)
                if len(zz) >= 6 and (np.ptp(zz) >= 5 or np.ptp(xx) >= 5):
                    lunghi += 1
        self.assertGreater(lunghi, 0, "mai un ruscello lungo in 60 caverne")

    def test_la_fascia_e_la_stessa_attraverso_i_chunk(self):
        """Stessa caverna sull'orlo fra due chunk: le coordinate di mondo decidono la fascia,
        non il chunk (la funzione e' deterministica sul seme e sulla posizione)."""
        a = self.acqua_nei_chunk(0, 0, 5)[1]
        b = self.acqua_nei_chunk(0, 0, 5)[1]
        self.assertTrue(np.array_equal(a, b))


class TestGeodi(unittest.TestCase):
    def test_il_geode_ha_i_gusci_e_il_vuoto(self):
        s, tav, out, h = mondo()
        SS.geode(out, tav.deco, 0, 8, 8, -30, Y0, h, 4.5, tav.scavabile)
        n = nomi(s, out)
        for atteso in ("smooth_basalt", "calcite", "amethyst_block", "air"):
            self.assertIn(atteso, n)
        # il vuoto interno e' circondato: nessuna aria a contatto con la pietra
        vuoto = np.nonzero(out[:, 30:35, :] == 0)
        self.assertGreater(len(vuoto[0]), 5)

    def test_il_geode_non_buca_il_terreno_ne_le_strutture(self):
        s, tav, out, h = mondo(quota=-22)
        prima = out.copy()
        SS.geode(out, tav.deco, 0, 8, 8, -28, Y0, h, 5.0, tav.scavabile)
        sopra = (np.arange(Y0, Y0 + ALT)[None, :, None] >= -29)
        self.assertTrue((out[np.broadcast_to(sopra, out.shape)]
                         == prima[np.broadcast_to(sopra, out.shape)]).all())
        prot = np.ones((16, 16), bool)
        s, tav, out, h = mondo()
        prima = out.copy()
        SS.geode(out, tav.deco, 0, 8, 8, -30, Y0, h, 5.0, tav.scavabile, prot)
        self.assertTrue((out == prima).all())

    def test_alcuni_chunk_hanno_geodi(self):
        trovati = 0
        for cx in range(0, 1200, 16):
            s, tav, out, h = mondo()
            SS.posa(out, tav, h, cx, 0, Y0, caverna(1, 100), [], seed=3)
            trovati += "amethyst_block" in nomi(s, out)
        self.assertGreater(trovati, 5)


class TestTraduzioni(unittest.TestCase):
    def test_i_blocchi_nuovi_si_traducono(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        casi = (("dripstone_block", {}), ("moss_block", {}), ("moss_carpet", {}),
                ("hanging_roots", {}), ("amethyst_block", {}), ("budding_amethyst", {}),
                ("calcite", {}), ("smooth_basalt", {}), ("sculk", {}),
                ("azalea", {}), ("flowering_azalea", {}),
                ("pointed_dripstone", {"thickness": "tip", "vertical_direction": "up"}),
                ("cave_vines", {"age": "5", "berries": "true"}),
                ("big_dripleaf", {"facing": "north", "tilt": "none"}),
                ("amethyst_cluster", {"facing": "down"}),
                ("sculk_sensor", {"power": "0", "sculk_sensor_phase": "inactive"}))
        for nome, pr in casi:
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in pr.items()})
            o, _, _ = v.block.from_universal(b)
            self.assertEqual(o.base_name, nome, nome)


if __name__ == "__main__":
    unittest.main()
