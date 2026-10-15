"""Alberi per bioma (#27): acacia, rovere scuro, giungla, abeti alti, ciliegi."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import biomi as B
from genworld import mappa as M
from genworld import vegetazione as V


class Finta:
    def __init__(self):
        self.tronco = {sp: 100 + sp for sp in V.ALBERI}
        self.foglie = {sp: 200 + sp for sp in V.ALBERI}
        self.cactus, self.arbusto = 30, 31
        self.erba_alta, self.felce = 32, 33
        self.torchflower = 34
        self.doppio = {sp: (300 + sp, 400 + sp) for sp in V.DOPPI}
        self.gambo = 500
        self.cappello = {sp: 600 + sp for sp in V.FUNGHI_GIGANTI}
        self.aria = 0


def albero(sp, alt, x=8, z=8):
    return {"x": np.array([x]), "z": np.array([z]), "y": np.array([70]),
            "specie": np.array([sp]), "altezza": np.array([alt]),
            "seme": np.array([7])}


def disegnato(sp, alt):
    out = np.zeros((16, 140, 16), np.uint32)
    V.disegna(out, 0, 0, 0, albero(sp, alt), [0], Finta())
    return out


class TestSemina(unittest.TestCase):
    def test_ogni_bioma_semina_le_sue_specie(self):
        lato = 160
        cls = np.full((lato, lato), M.FORESTA, np.uint8)
        h = np.full((lato, lato), 80, np.int32)
        for nome, attese in (("savanna", V.ACACIA), ("dark_forest", V.ROVERE_SCURO),
                             ("jungle", V.GIUNGLA), ("meadow", V.CILIEGIO),
                             ("giant_spruce_taiga", V.ABETE_ALTO)):
            biomi = np.full((lato, lato), B.INDICE[nome], np.uint8)
            a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi)
            self.assertIn(attese, set(a["specie"].tolist()), nome)

    def test_senza_biomi_nessuna_specie_nuova(self):
        cls = np.full((120, 120), M.FORESTA, np.uint8)
        h = np.full((120, 120), 80, np.int32)
        a = V.semina(cls, h, livello_mare=62, seed=3)
        nuove = {V.ACACIA, V.ROVERE_SCURO, V.GIUNGLA, V.ABETE_ALTO, V.CILIEGIO}
        self.assertFalse(nuove & set(a["specie"].tolist()))

    def test_altezze_nei_limiti(self):
        cls = np.full((160, 160), M.FORESTA, np.uint8)
        h = np.full((160, 160), 80, np.int32)
        biomi = np.full((160, 160), B.INDICE["jungle"], np.uint8)
        a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi)
        g = a["specie"] == V.GIUNGLA
        lo, hi = V.ALTEZZA[V.GIUNGLA]
        self.assertTrue(((a["altezza"][g] >= lo) & (a["altezza"][g] <= hi)).all())


class TestDisegno(unittest.TestCase):
    def test_acacia_ha_chioma_piatta_e_larga(self):
        out = disegnato(V.ACACIA, 7)
        f = 200 + V.ACACIA
        strati = [(out[:, y, :] == f).sum() for y in range(70, 90)]
        self.assertGreaterEqual(max(strati), 15)
        self.assertLessEqual(sum(1 for s in strati if s), 4)

    def test_rovere_scuro_ha_tronco_2x2(self):
        out = disegnato(V.ROVERE_SCURO, 7)
        self.assertEqual(int((out[:, 71, :] == 100 + V.ROVERE_SCURO).sum()), 4)

    def test_giungla_e_alta_e_a_piu_livelli(self):
        out = disegnato(V.GIUNGLA, 14)
        f = 200 + V.GIUNGLA
        con_foglie = [y for y in range(70, 100) if (out[:, y, :] == f).any()]
        self.assertGreater(max(con_foglie) - min(con_foglie), 6)
        self.assertGreater(len(con_foglie), 5)
        self.assertTrue((out[8, 70:84, 8] == 100 + V.GIUNGLA).all())

    def test_abete_alto_supera_l_abete(self):
        a = disegnato(V.ABETE_ALTO, 20)
        self.assertTrue((a[8, 70:89, 8] == 100 + V.ABETE_ALTO).all())

    def test_nessun_disegno_esce_dal_raggio(self):
        for sp, alt in ((V.ACACIA, 7), (V.ROVERE_SCURO, 8), (V.GIUNGLA, 14)):
            out = disegnato(sp, alt)
            xs, _, zs = np.nonzero(out)
            self.assertLessEqual(max(abs(xs - 8).max(), abs(zs - 8).max()), V.RAGGIO_MAX, sp)


class TestPianteNuove(unittest.TestCase):
    def test_le_piante_alte_hanno_due_meta(self):
        for sp in V.DOPPI:
            out = disegnato(sp, 2)
            self.assertEqual(int(out[8, 70, 8]), 300 + sp)
            self.assertEqual(int(out[8, 71, 8]), 400 + sp)

    def test_il_fungo_gigante_ha_gambo_e_cappello_largo(self):
        for sp in V.FUNGHI_GIGANTI:
            out = disegnato(sp, 6)
            self.assertTrue((out[8, 70:75, 8] == 500).all())
            cap = (out == 600 + sp)
            self.assertGreaterEqual(int(cap.sum()), 14)
            xs, _, zs = np.nonzero(cap)
            self.assertGreaterEqual(xs.max() - xs.min(), 3)

    def test_ogni_bioma_nuovo_semina_le_sue_piante(self):
        cls = np.full((200, 200), M.PRATERIA, np.uint8)
        h = np.full((200, 200), 80, np.int32)
        for nome, attese in (("sunflower_plains", V.GIRASOLE), ("taiga", V.FELCE_GRANDE),
                             ("meadow", V.LILLA), ("dark_forest", V.FUNGO_GIGANTE_M),
                             ("savanna", V.TORCHFLOWER)):
            biomi = np.full((200, 200), B.INDICE[nome], np.uint8)
            a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi, scala_densita=3.0)
            self.assertIn(attese, set(a["specie"].tolist()), nome)

    def test_ogni_specie_ha_altezza_e_nome(self):
        for sp in range(30):
            self.assertIn(sp, V.ALTEZZA)
            self.assertIn(sp, V.NOMI_SPECIE)


class TestTraduzioni(unittest.TestCase):
    def test_i_blocchi_nuovi_si_traducono(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        lati = {"north": "true", "south": "true", "east": "true", "west": "true"}
        for nome, pr, java in (
                ("double_plant", {"plant_type": "sunflower", "half": "upper"}, "sunflower"),
                ("double_plant", {"plant_type": "large_fern", "half": "lower"}, "large_fern"),
                ("double_plant", {"plant_type": "lilac", "half": "lower"}, "lilac"),
                ("double_plant", {"plant_type": "rose_bush", "half": "lower"}, "rose_bush"),
                ("double_plant", {"plant_type": "peony", "half": "lower"}, "peony"),
                ("torchflower", {}, "torchflower"),
                ("brown_mushroom_block", dict(lati, up="true", down="false"),
                 "brown_mushroom_block"),
                ("red_mushroom_block", dict(lati, up="true", down="false"),
                 "red_mushroom_block"),
                ("mushroom_stem", dict(lati, up="false", down="false", material="red"),
                 "mushroom_stem")):
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in pr.items()})
            o, _, _ = v.block.from_universal(b)
            self.assertEqual(o.base_name, java, nome)

    def test_il_bioma_giungla_traduce(self):
        self.assertNotIn("jungle", B.verifica_traduzioni((26, 2, 0)))


if __name__ == "__main__":
    unittest.main()
