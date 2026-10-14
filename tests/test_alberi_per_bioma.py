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


class TestTraduzioni(unittest.TestCase):
    def test_il_bioma_giungla_traduce(self):
        self.assertNotIn("jungle", B.verifica_traduzioni((26, 2, 0)))


if __name__ == "__main__":
    unittest.main()
