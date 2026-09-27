"""Test degli strati di roccia.

Nascono da uno screenshot di montagne: erano un blocco solo dalla base alla
cima, e la superficie era a righe verticali come un gelato alla crema. Le
righe erano il dithering applicato dove non serviva; il blocco solo e'
`stratigrafia`.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import stratigrafia as SG  # noqa: E402
from genworld.rumore import _peso_dither, quantizza  # noqa: E402


class TestTavola(unittest.TestCase):

    def test_copre_tutta_la_colonna(self):
        t = SG.tavola(seed=3)
        self.assertEqual(len(t), SG.Y_TAVOLA_MAX - SG.Y_TAVOLA_MIN)

    def test_e_la_stessa_per_tutta_la_mappa(self):
        """Uno strato si deve ritrovare uguale sui due versanti di una valle:
        e' questo che fa leggere un paesaggio come un paesaggio."""
        self.assertEqual(SG.tavola(seed=3), SG.tavola(seed=3))
        self.assertNotEqual(SG.tavola(seed=3), SG.tavola(seed=4))

    def test_piu_rocce_ma_la_pietra_domina(self):
        t = SG.tavola(seed=7)
        nomi = [n for n, _ in t]
        distinti = set(nomi)
        self.assertGreaterEqual(len(distinti), 4)
        quota_pietra = nomi.count("stone") / len(nomi)
        self.assertGreater(quota_pietra, 0.45,
                           "una montagna a bande tutte diverse e' un campionario")

    def test_i_banchi_sono_spessi_non_alternati(self):
        """Uno strato alto un blocco non e' uno strato, e' rumore."""
        t = [n for n, _ in SG.tavola(seed=11)]
        tratti = []
        for n in t:
            if tratti and tratti[-1][0] == n:
                tratti[-1][1] += 1
            else:
                tratti.append([n, 1])
        medio = sum(c for _, c in tratti) / len(tratti)
        self.assertGreater(medio, 4.0, f"spessore medio {medio:.1f}")


class TestPiega(unittest.TestCase):

    def test_gli_strati_ondulano(self):
        p = SG.piega(300, seed=2)
        self.assertGreater(int(np.ptp(p)), 2, "strati perfettamente piatti")
        self.assertLessEqual(int(np.abs(p).max()), SG.PIEGA + 1)

    def test_ondulano_su_scala_larga(self):
        """Uno strato frastagliato da una cella all'altra non esiste in
        natura: la piega e' regionale."""
        p = SG.piega(400, seed=2).astype(np.float32)
        salto = np.abs(np.diff(p, axis=1)).mean()
        self.assertLess(salto, 0.35, f"salto medio fra celle vicine {salto:.2f}")


class TestSuperficie(unittest.TestCase):

    def pendio(self, pendenza):
        h = np.zeros((80, 80), np.float32)
        h += np.arange(80, dtype=np.float32)[None, :] * pendenza
        return h

    def test_la_parete_resta_nuda(self):
        h = self.pendio(3.0) + 100
        pend = np.full(h.shape, 3.0, np.float32)
        nuda, _ = SG.superficie_montana(h, pend, seed=1)
        self.assertTrue(nuda.all())

    def test_il_prato_non_e_nudo(self):
        h = self.pendio(0.1) + 100
        pend = np.full(h.shape, 0.1, np.float32)
        nuda, _ = SG.superficie_montana(h, pend, seed=1)
        self.assertFalse(nuda.any())

    def test_la_neve_non_comincia_col_righello(self):
        """Fra il limite basso e quello alto la neve deve essere a chiazze:
        una linea delle nevi netta si riconosce da chilometri."""
        zz, xx = np.mgrid[:120, :120]
        h = (SG.NEVE_BASSA + (SG.NEVE_ALTA - SG.NEVE_BASSA)
             * xx / 119.0).astype(np.float32)
        pend = np.full(h.shape, 0.2, np.float32)
        _, neve = SG.superficie_montana(h, pend, seed=5)
        fascia = neve[:, 30:90]
        self.assertTrue((fascia == 1).any(), "nessun manto sottile")
        self.assertTrue((fascia == 0).any(), "la fascia e' tutta innevata")
        # e in cima la neve c'e' sempre
        self.assertTrue((neve[:, -1] == 2).all())

    def test_sulla_parete_la_neve_non_tiene(self):
        h = np.full((40, 40), SG.NEVE_ALTA + 30, np.float32)
        pend = np.full(h.shape, SG.PENDENZA_PARETE + 1, np.float32)
        _, neve = SG.superficie_montana(h, pend, seed=5)
        self.assertTrue((neve == 0).all())


class TestDitherSelettivo(unittest.TestCase):
    """Il dithering cura le terrazze dei pendii dolci. Su una parete non c'e'
    niente da curare e quello che aggiunge sono le righe verticali."""

    def test_sul_pendio_dolce_il_dither_lavora(self):
        h = np.arange(200, dtype=np.float32)[None, :] * 0.1 + np.zeros((200, 1), np.float32)
        self.assertGreater(float(_peso_dither(h).mean()), 0.9)

    def test_sulla_parete_il_dither_si_spegne(self):
        h = np.arange(200, dtype=np.float32)[None, :] * 3.0 + np.zeros((200, 1), np.float32)
        self.assertLess(float(_peso_dither(h).max()), 0.01)

    def test_la_parete_resta_liscia_dopo_la_quantizzazione(self):
        """La misura del difetto: quanto si discosta una colonna dalla media
        dei suoi vicini. Su una rampa esatta deve essere zero."""
        h = (np.arange(200, dtype=np.float32)[None, :] * 3.0
             + np.zeros((200, 1), np.float32))
        q = quantizza(h, forza=1.0, seed=4).astype(np.float32)
        residuo = q - h
        # la rampa e' esatta: senza dither l'arrotondamento e' costante lungo z
        self.assertLess(float(residuo.std(axis=0).max()), 1e-6,
                        "la parete porta ancora il disturbo colonna per colonna")

    def test_il_pendio_dolce_invece_si_frastaglia(self):
        h = (np.arange(200, dtype=np.float32)[None, :] * 0.1
             + np.zeros((200, 1), np.float32))
        q = quantizza(h, forza=1.0, seed=4).astype(np.float32)
        self.assertGreater(float((q - h).std(axis=0).max()), 0.1)


if __name__ == "__main__":
    unittest.main()


class TestTraduzioni(unittest.TestCase):
    """La trappola di `oak_log`, terza puntata: un nome fuori dal namespace
    universale si scrive e si rilegge identico, e in gioco non c'e'."""

    def test_tutte_le_rocce_arrivano_in_gioco(self):
        try:
            import PyMCTranslate  # noqa: F401
        except ImportError:
            self.skipTest("PyMCTranslate non installato")
        self.assertEqual(SG.verifica_traduzioni(), [])

    def test_un_nome_moderno_invece_fallisce(self):
        """Il test vale solo se sa anche bocciare: `deepslate_bricks` non e'
        un nome universale."""
        try:
            import PyMCTranslate
            from amulet.api.block import Block
        except ImportError:
            self.skipTest("PyMCTranslate non installato")
        ver = PyMCTranslate.new_translation_manager().get_version("java", (1, 21, 4))
        fuori = ver.block.from_universal(
            Block("universal_minecraft", "questo_blocco_non_esiste"))[0]
        self.assertNotEqual(fuori.namespace, "minecraft")
