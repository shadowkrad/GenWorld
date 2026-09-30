"""Verifica della semina e del disegno degli alberi."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import mappa as M
from genworld import vegetazione as V


def scenario(lato=128, mare=62):
    """Meta' prateria a quota 80, meta' oceano a quota 40."""
    cls = np.full((lato, lato), M.OCEANO, np.uint8)
    cls[:, lato // 2:] = M.PRATERIA
    h = np.full((lato, lato), 40, np.int32)
    h[:, lato // 2:] = 80
    return cls, h


class FintaTavolozza:
    """Id finti: il disegno non deve dipendere da un livello aperto."""
    def __init__(self):
        self.tronco = {V.QUERCIA: 10, V.BETULLA: 11, V.ABETE: 12}
        self.foglie = {V.QUERCIA: 20, V.BETULLA: 21, V.ABETE: 22}
        self.cactus, self.arbusto = 30, 31
        self.erba_alta, self.felce = 32, 33
        self.aria = 0


class TestSemina(unittest.TestCase):

    def test_mai_nell_acqua(self):
        cls, h = scenario()
        a = V.semina(cls, h, livello_mare=62, seed=1)
        self.assertGreater(a["x"].size, 0)
        self.assertTrue((a["y"] > 62).all())
        self.assertTrue((cls[a["z"], a["x"]] != M.OCEANO).all())

    def test_densita_scala(self):
        cls, h = scenario()
        poco = V.semina(cls, h, scala_densita=0.25, seed=2)["x"].size
        molto = V.semina(cls, h, scala_densita=3.0, seed=2)["x"].size
        self.assertGreater(molto, poco * 2)

    def test_zero_densita(self):
        cls, h = scenario()
        self.assertEqual(V.semina(cls, h, scala_densita=0.0, seed=3)["x"].size, 0)

    def test_deterministica(self):
        cls, h = scenario()
        a = V.semina(cls, h, seed=5); b = V.semina(cls, h, seed=5)
        self.assertTrue(np.array_equal(a["x"], b["x"]))
        self.assertTrue(np.array_equal(a["specie"], b["specie"]))

    def test_distanza_minima(self):
        """La griglia sfalsata garantisce che due alberi non si accavallino."""
        cls, h = scenario()
        a = V.semina(cls, h, passo=4, scala_densita=5.0, seed=6)
        p = np.stack([a["x"], a["z"]], 1)
        if len(p) > 1:
            # nessuna coppia identica
            self.assertEqual(len(np.unique(p, axis=0)), len(p))

    def test_tutta_acqua_non_semina(self):
        cls = np.full((64, 64), M.OCEANO, np.uint8)
        h = np.full((64, 64), 40, np.int32)
        self.assertEqual(V.semina(cls, h, seed=7)["x"].size, 0)


class TestIndice(unittest.TestCase):

    def test_indice_copre_tutto(self):
        cls, h = scenario()
        a = V.semina(cls, h, seed=8)
        idx = V.indice_per_chunk(a, 128)
        self.assertEqual(sum(len(v) for v in idx.values()), a["x"].size)

    def test_vicini_includono_il_centro(self):
        idx = {(3, 4): [0, 1], (2, 4): [2], (9, 9): [3]}
        v = V.alberi_vicini(idx, 3, 4)
        self.assertIn(0, v); self.assertIn(2, v); self.assertNotIn(3, v)


class TestDisegno(unittest.TestCase):

    def albero(self, x, z, sp=V.QUERCIA, alt=5):
        return {"x": np.array([x]), "z": np.array([z]), "y": np.array([70]),
                "specie": np.array([sp]), "altezza": np.array([alt]),
                "seme": np.array([12345])}

    def test_tronco_vince_sulle_foglie(self):
        out = np.zeros((16, 120, 16), np.uint32)
        V.disegna(out, 0, 0, 0, self.albero(8, 8), [0], FintaTavolozza())
        colonna = out[8, 70:75, 8]
        self.assertTrue((colonna == 10).all(), f"tronco interrotto: {colonna}")

    def test_foglie_presenti(self):
        out = np.zeros((16, 120, 16), np.uint32)
        V.disegna(out, 0, 0, 0, self.albero(8, 8), [0], FintaTavolozza())
        self.assertGreater(int((out == 20).sum()), 12)

    def test_albero_a_cavallo_compare_in_entrambi_i_chunk(self):
        """L'invariante che rende sensato l'indice per chunk.

        Un albero piantato a un blocco dal bordo sporge nel chunk accanto. Se
        ogni chunk disegnasse solo gli alberi col centro dentro di se', lungo
        ogni linea di chunk si vedrebbero alberi tagliati a meta'.
        """
        tav = FintaTavolozza()
        alb = self.albero(15, 8)          # ultimo blocco del chunk (0,0)
        sinistra = np.zeros((16, 120, 16), np.uint32)
        destra = np.zeros((16, 120, 16), np.uint32)
        V.disegna(sinistra, 0, 0, 0, alb, [0], tav)
        V.disegna(destra, 0, 16, 0, alb, [0], tav)
        self.assertGreater(int((sinistra != 0).sum()), 0, "niente nel chunk di casa")
        self.assertGreater(int((destra != 0).sum()), 0, "niente nel chunk accanto")

    def test_niente_fuori_dai_bordi(self):
        out = np.zeros((16, 120, 16), np.uint32)
        prima = out.shape
        V.disegna(out, 0, 0, 0, self.albero(0, 0), [0], FintaTavolozza())
        self.assertEqual(out.shape, prima)   # nessun errore di indice

    def test_cactus_e_arbusto(self):
        tav = FintaTavolozza()
        out = np.zeros((16, 120, 16), np.uint32)
        V.disegna(out, 0, 0, 0, self.albero(8, 8, V.CACTUS, 3), [0], tav)
        self.assertEqual(int((out == 30).sum()), 3)
        out2 = np.zeros((16, 120, 16), np.uint32)
        V.disegna(out2, 0, 0, 0, self.albero(8, 8, V.ARBUSTO, 1), [0], tav)
        self.assertEqual(int((out2 == 31).sum()), 1)

    def test_la_cima_e_di_foglie_non_di_tronco(self):
        """Difetto visto in gioco: su alcuni alberi il tronco spuntava dalla
        cima della chioma. Nell'abete il tronco arrivava fino all'apice, dove
        la sagoma ha una sola foglia, e siccome si disegna per ultimo vinceva
        su di essa. Il blocco piu' alto della colonna centrale deve essere
        una foglia, per ogni specie e ogni altezza."""
        tav = FintaTavolozza()
        for sp, foglia in ((V.QUERCIA, 20), (V.BETULLA, 21), (V.ABETE, 22)):
            lo, hi = V.ALTEZZA[sp]
            for alt in range(lo, hi + 1):
                out = np.zeros((16, 160, 16), np.uint32)
                V.disegna(out, 0, 0, 0, self.albero(8, 8, sp, alt), [0], tav)
                colonna = out[8, :, 8]
                cima = int(np.nonzero(colonna)[0].max())
                self.assertEqual(int(colonna[cima]), foglia,
                                 f"specie {sp} alt {alt}: in cima c'e' {colonna[cima]}")

    def test_il_tronco_dell_abete_arriva_sotto_l_apice(self):
        out = np.zeros((16, 160, 16), np.uint32)
        V.disegna(out, 0, 0, 0, self.albero(8, 8, V.ABETE, 10), [0], FintaTavolozza())
        self.assertTrue((out[8, 70:79, 8] == 12).all(), "tronco interrotto")

    def test_abete_conico(self):
        """Dall'alto verso il basso la chioma deve allargarsi, non restringersi."""
        out = np.zeros((16, 160, 16), np.uint32)
        V.disegna(out, 0, 0, 0, self.albero(8, 8, V.ABETE, 10), [0], FintaTavolozza())
        larghezze = [(out[:, y, :] == 22).sum() for y in range(70, 81)]
        self.assertGreater(sum(larghezze[:4]), 0)
        self.assertGreaterEqual(max(larghezze[:6]), max(larghezze[7:]))


if __name__ == "__main__":
    unittest.main()


class TestInvariantiPipeline(unittest.TestCase):
    """Regressioni viste solo guardando il mondo finito."""

    def test_nessun_albero_sopra_l_acqua(self):
        """Boschetti in mezzo all'oceano: il rumore di dettaglio rialzava
        celle di mare sopra il livello dell'acqua dopo che il tetto era gia'
        stato imposto, creando isolotti da un blocco su cui seminare."""
        lato = 128
        cls = np.full((lato, lato), M.OCEANO, np.uint8)
        cls[:, 90:] = M.PRATERIA
        h = np.full((lato, lato), 50, np.int32)
        h[:, 90:] = 80
        # una manciata di celle d'acqua rialzate per sbaglio
        h[10:14, 10:14] = 66
        a = V.semina(cls, h, livello_mare=62, seed=11)
        classi = cls[a["z"], a["x"]]
        self.assertFalse(np.isin(classi, list(M.ACQUA)).any(),
                         "seminato su una cella classificata come acqua")
