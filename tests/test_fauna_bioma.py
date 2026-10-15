"""Fauna per bioma (#34): savana, giungla, prati, e la vita del mare."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import biomi as B
from genworld import fauna as FA
from genworld import mappa as M


def terra(bioma, lato=400):
    cls = np.full((lato, lato), M.PRATERIA, np.uint8)
    h = np.full((lato, lato), 80, np.int32)
    return cls, h, np.full((lato, lato), B.INDICE[bioma], np.uint8)


class TestTerra(unittest.TestCase):
    def specie(self, bioma):
        cls, h, bio = terra(bioma)
        return {a.specie for a in FA.semina_selvatica(cls, h, seed=3, biomi=bio,
                                                     scala_densita=3.0)}

    def test_la_savana_ha_lama_e_asini(self):
        self.assertTrue(self.specie("savanna") & {"lama", "asino", "armadillo"})

    def test_la_giungla_ha_panda_e_pappagalli(self):
        s = self.specie("jungle")
        self.assertTrue(s & {"panda", "pappagallo", "ocelotto"})
        self.assertFalse(s & {"lupo", "cammello", "orso_polare"})

    def test_i_prati_hanno_le_api(self):
        self.assertIn("ape", self.specie("meadow"))

    def test_senza_biomi_la_fauna_e_quella_di_prima(self):
        cls, h, _ = terra("plains")
        s = {a.specie for a in FA.semina_selvatica(cls, h, seed=3, scala_densita=3.0)}
        self.assertLessEqual(s, {"coniglio", "volpe"})

    def test_ogni_specie_ha_id_e_salute(self):
        for miscela in FA.MISCELA_BIOMA.values():
            for sp, _ in miscela:
                self.assertIn(sp, FA.SPECIE)
                self.assertIn(sp, FA.SALUTE)
        for miscela in FA.FAUNA_MARINA.values():
            for sp, _, _ in miscela:
                self.assertIn(sp, FA.SPECIE)
                self.assertIn(sp, FA.ACQUATICI)

    def test_i_biomi_della_fauna_esistono(self):
        for nome in list(FA.MISCELA_BIOMA) + list(FA.FAUNA_MARINA):
            self.assertIn(nome, B.BIOMI)


class TestMare(unittest.TestCase):
    def mare(self, bioma, fondale=40, lato=400):
        cls = np.full((lato, lato), M.OCEANO, np.uint8)
        h = np.full((lato, lato), fondale, np.int32)
        return cls, h, np.full((lato, lato), B.INDICE[bioma], np.uint8)

    def test_nel_mare_caldo_ci_sono_pesci_tropicali(self):
        cls, h, bio = self.mare("warm_ocean")
        sp = {a.specie for a in FA.semina_marina(cls, h, bio, seed=1)}
        self.assertIn("pesce_tropicale", sp)

    def test_i_pesci_stanno_dentro_l_acqua(self):
        cls, h, bio = self.mare("ocean", fondale=45)
        pesci = FA.semina_marina(cls, h, bio, seed=2)
        self.assertGreater(len(pesci), 5)
        for a in pesci:
            self.assertGreater(a.y, 45)
            self.assertLess(a.y, 62)

    def test_non_ci_sono_pesci_dove_l_acqua_e_bassa_o_sulla_terra(self):
        cls, h, bio = self.mare("ocean", fondale=60)
        self.assertEqual(FA.semina_marina(cls, h, bio, seed=2), [])
        cls, h, bio = terra("plains")
        self.assertEqual(FA.semina_marina(cls, h, bio, seed=2), [])

    def test_il_mare_gelato_non_ha_fauna(self):
        cls, h, bio = self.mare("frozen_ocean")
        self.assertEqual(FA.semina_marina(cls, h, bio, seed=2), [])

    def test_e_deterministica(self):
        cls, h, bio = self.mare("ocean")
        a = [(x.x, x.y, x.z, x.specie) for x in FA.semina_marina(cls, h, bio, seed=4)]
        b = [(x.x, x.y, x.z, x.specie) for x in FA.semina_marina(cls, h, bio, seed=4)]
        self.assertEqual(a, b)


class TestNbt(unittest.TestCase):
    def test_un_pesce_non_e_a_terra(self):
        from amulet_nbt import StringTag  # noqa: F401
        t = FA.Animale(x=1.5, y=50.0, z=1.5, specie="merluzzo", seme=3).nbt_tag()
        self.assertEqual(str(t["id"]), "minecraft:cod")
        self.assertEqual(int(t["OnGround"]), 0)

    def test_un_lama_si_scrive(self):
        t = FA.Animale(x=1.5, y=70.0, z=1.5, specie="lama", seme=3).nbt_tag()
        self.assertEqual(str(t["id"]), "minecraft:llama")
        self.assertEqual(int(t["OnGround"]), 1)



class TestGatti(unittest.TestCase):
    def test_i_gatti_stanno_sulle_strade_vicino_alla_piazza(self):
        from genworld.citta import Citta
        strada = np.zeros((100, 100), np.uint8)
        strada[50, 20:80] = 1
        strada[20:80, 50] = 1
        h = np.full((100, 100), 70, np.int32)
        c = Citta(z=50, x=50, raggio=30, piazza=(50, 50))
        gatti = FA.gatti([c], strada, h, seed=1)
        self.assertGreaterEqual(len(gatti), 1)
        for g in gatti:
            self.assertEqual(g.specie, "gatto")
            self.assertGreater(strada[int(g.z), int(g.x)], 0)
            self.assertLessEqual(abs(g.x - 50.5), 15)
            self.assertEqual(g.y, 70.0)

    def test_senza_strade_niente_gatti(self):
        from genworld.citta import Citta
        c = Citta(z=50, x=50, raggio=30, piazza=(50, 50))
        self.assertEqual(FA.gatti([c], None, np.zeros((100, 100), np.int32)), [])
        self.assertEqual(FA.gatti([c], np.zeros((100, 100), np.uint8),
                                  np.zeros((100, 100), np.int32)), [])


if __name__ == "__main__":
    unittest.main()
