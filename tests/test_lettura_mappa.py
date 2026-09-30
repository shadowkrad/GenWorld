"""Test della lettura di una mappa disegnata: spazio colore, cornice,
bilanciamento, famiglia, classificazione automatica e guidata.

E' la porta d'ingresso di ogni mondo: se la mappa si legge male, tutto quello
che segue (quote, fiumi, villaggi) e' costruito sul niente. Le immagini qui
sono sintetiche e di poche decine di pixel: si controllano le PROPRIETA' che
devono valere per qualunque mappa (l'acqua e' blu, una cornice si ritaglia,
un campione decide la sua classe), non numeri tarati su un'immagine.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import mappa as M  # noqa: E402
from genworld import classi_auto as CA  # noqa: E402
from genworld import classi_guidate as CG  # noqa: E402
from genworld import normalizza as N  # noqa: E402

MARE_RGB = (0.16, 0.36, 0.72)
ERBA_RGB = (0.35, 0.60, 0.28)
SABBIA_RGB = (0.85, 0.78, 0.50)
NEVE_RGB = (0.96, 0.97, 0.98)


def immagine(lato=64, sfondo=ERBA_RGB) -> np.ndarray:
    im = np.empty((lato, lato, 3), np.float64)
    im[:] = sfondo
    return im


def isola(lato=96) -> np.ndarray:
    """Mare a sinistra, erba al centro, sabbia a destra."""
    im = immagine(lato)
    im[:, : lato // 3] = MARE_RGB
    im[:, 2 * lato // 3:] = SABBIA_RGB
    return im


class TestLab(unittest.TestCase):

    def test_bianco_e_nero(self):
        lab = N.a_lab(np.array([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]))
        self.assertAlmostEqual(float(lab[0, 0, 0]), 100.0, delta=0.1)
        self.assertAlmostEqual(float(lab[0, 1, 0]), 0.0, delta=0.1)
        self.assertLess(abs(float(lab[0, 0, 1])), 0.5)          # senza tinta
        self.assertLess(abs(float(lab[0, 0, 2])), 0.5)

    def test_il_blu_ha_b_negativo_il_giallo_positivo(self):
        blu = N.a_lab(np.array([[MARE_RGB]]))[0, 0]
        giallo = N.a_lab(np.array([[SABBIA_RGB]]))[0, 0]
        self.assertLess(float(blu[2]), -10)
        self.assertGreater(float(giallo[2]), 10)

    def test_il_verde_ha_a_negativo_il_rosso_positivo(self):
        self.assertLess(float(N.a_lab(np.array([[ERBA_RGB]]))[0, 0, 1]), -10)
        self.assertGreater(float(N.a_lab(np.array([[(0.8, 0.2, 0.2)]]))[0, 0, 1]), 20)

    def test_conserva_la_forma(self):
        self.assertEqual(N.a_lab(np.zeros((5, 7, 3))).shape, (5, 7, 3))


class TestCornice(unittest.TestCase):

    def con_cornice(self, alto=6, basso=6, sinistra=6, destra=6):
        im = immagine(80)
        im[:alto] = 1.0
        if basso:
            im[-basso:] = 1.0
        im[:, :sinistra] = 1.0
        if destra:
            im[:, -destra:] = 1.0
        return im

    def test_trova_la_cornice_su_ogni_lato(self):
        a, b, s, d = N.trova_cornice(self.con_cornice())
        for lato in (a, b, s, d):
            self.assertGreaterEqual(lato, 5)
            self.assertLessEqual(lato, 7)

    def test_senza_cornice_non_ritaglia(self):
        self.assertEqual(N.trova_cornice(immagine(80)), (0, 0, 0, 0))

    def test_una_cornice_solo_in_alto_ritaglia_solo_in_alto(self):
        """Il titolo sta spesso in un lato solo: ritagliare in modo uniforme
        butterebbe via mappa buona sugli altri tre."""
        a, b, s, d = N.trova_cornice(self.con_cornice(alto=8, basso=0, sinistra=0, destra=0))
        self.assertGreaterEqual(a, 7)
        self.assertEqual((b, s, d), (0, 0, 0))

    def test_il_ritaglio_riduce_l_immagine(self):
        im = self.con_cornice()
        out = N.ritaglia_cornice(im)
        self.assertLess(out.shape[0], im.shape[0])
        self.assertLess(out.shape[1], im.shape[1])
        self.assertTrue((out < 0.99).all(), "e' rimasto del bianco di cornice")


class TestBilancia(unittest.TestCase):

    def test_stende_un_istogramma_stretto(self):
        rng = np.random.default_rng(0)
        im = 0.4 + 0.1 * rng.random((32, 32, 3))
        out = N.bilancia(im, forza=1.0)
        self.assertGreater(float(out.max() - out.min()), float(im.max() - im.min()) * 2)

    def test_forza_zero_non_cambia_niente(self):
        rng = np.random.default_rng(1)
        im = rng.random((16, 16, 3))
        self.assertTrue(np.allclose(N.bilancia(im, forza=0.0), im))

    def test_un_canale_costante_resta_com_e(self):
        im = immagine(16)
        self.assertTrue(np.allclose(N.bilancia(im), im))

    def test_resta_fra_zero_e_uno(self):
        rng = np.random.default_rng(2)
        out = N.bilancia(rng.random((16, 16, 3)))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


class TestFamiglia(unittest.TestCase):

    def test_una_mappa_senza_colore_e_un_disegno_a_grigi(self):
        rng = np.random.default_rng(0)
        g = rng.random((64, 64, 1)).repeat(3, axis=2)
        self.assertEqual(N.famiglia(g)[0], "grigi")

    def test_pochi_colori_piatti(self):
        self.assertEqual(N.famiglia(isola())[0], "colori_piatti")

    def test_una_dipinta_ha_migliaia_di_colori_ma_poca_grana(self):
        zz, xx = np.mgrid[:96, :96]
        im = np.stack([0.2 + 0.6 * xx / 96, 0.3 + 0.5 * zz / 96,
                       0.5 + 0.2 * np.sin(xx / 7.0)], axis=-1)
        self.assertEqual(N.famiglia(np.clip(im, 0, 1))[0], "dipinta")

    def test_una_fotografica_ha_molta_grana(self):
        rng = np.random.default_rng(3)
        im = np.clip(0.5 + rng.normal(0, 0.25, (96, 96, 3)), 0, 1)
        im[..., 1] += 0.15
        self.assertEqual(N.famiglia(np.clip(im, 0, 1))[0], "fotografica")

    def test_restituisce_le_misure_che_ha_usato(self):
        _, info = N.famiglia(isola())
        self.assertEqual(set(info), {"colori_significativi", "croma_media", "grana"})


class TestRaggruppa(unittest.TestCase):

    def test_due_colori_diventano_due_gruppi(self):
        gruppi, centri = CA.raggruppa(isola(), n=3, seed=0)
        self.assertEqual(gruppi.shape, (96, 96))
        self.assertEqual(centri.shape, (3, 3))
        # pixel dello stesso colore -> stesso gruppo
        self.assertEqual(len(set(gruppi[:, :10].ravel().tolist())), 1)
        self.assertEqual(len(set(gruppi[:, 40:50].ravel().tolist())), 1)
        self.assertNotEqual(int(gruppi[0, 0]), int(gruppi[0, 45]))

    def test_e_deterministico(self):
        a, _ = CA.raggruppa(isola(), n=4, seed=5)
        b, _ = CA.raggruppa(isola(), n=4, seed=5)
        self.assertTrue((a == b).all())


def gruppo(i, L, a, b, croma=None):
    return {"id": i, "L": L, "a": a, "b": b,
            "croma": croma if croma is not None else float(np.hypot(a, b)),
            "area": 0.2, "bordo": 0.0, "grana": 1.0}


class TestAssegna(unittest.TestCase):
    """Criteri RELATIVI: l'acqua e' il gruppo piu' blu, non un colore fisso."""

    def test_il_gruppo_blu_e_acqua(self):
        m = CA.assegna([gruppo(0, 40, -5, -40), gruppo(1, 60, -30, 30)])
        self.assertIn(m[0], (M.MARE, M.OCEANO))
        self.assertNotIn(m[1], (M.MARE, M.OCEANO))

    def test_due_gruppi_blu_il_piu_scuro_e_oceano(self):
        m = CA.assegna([gruppo(0, 25, -5, -40), gruppo(1, 55, -5, -40), gruppo(2, 60, -30, 30)])
        self.assertEqual(m[0], M.OCEANO)
        self.assertEqual(m[1], M.MARE)

    def test_senza_blu_si_prende_il_piu_freddo(self):
        m = CA.assegna([gruppo(0, 50, -20, 5), gruppo(1, 60, -10, 30), gruppo(2, 55, -5, 40)])
        self.assertIn(m[0], (M.MARE, M.OCEANO))

    def test_l_estremo_luminoso_e_smorto_e_neve(self):
        stat = [gruppo(0, 40, -5, -40), gruppo(1, 97, 0, 1), gruppo(2, 60, -30, 30),
                gruppo(3, 55, -25, 35)]
        m = CA.assegna(stat)
        self.assertEqual(m[1], M.NEVE)

    def test_il_giallo_e_deserto_il_verde_scuro_foresta(self):
        stat = [gruppo(0, 40, -5, -40),
                gruppo(1, 80, -2, 45),              # sabbia
                gruppo(2, 35, -35, 20),             # bosco
                gruppo(3, 65, -25, 25)]             # prato
        m = CA.assegna(stat)
        self.assertEqual(m[1], M.DESERTO)
        self.assertEqual(m[2], M.FORESTA)
        self.assertIn(m[3], (M.PRATERIA, M.PIANURA))

    def test_ogni_gruppo_riceve_una_classe(self):
        stat = [gruppo(i, 20 + 5 * i, -30 + 6 * i, -30 + 9 * i) for i in range(10)]
        m = CA.assegna(stat)
        self.assertEqual(set(m), set(range(10)))

    def test_solo_acqua_non_esplode(self):
        self.assertEqual(len(CA.assegna([gruppo(0, 40, -5, -40)])), 1)


class TestClassificaAdattiva(unittest.TestCase):

    def test_isola_mare_a_sinistra_terra_a_destra(self):
        cls, stat = CA.classifica_adattiva(isola(), rug=np.zeros((96, 96)), n_gruppi=4)
        self.assertEqual(cls.shape, (96, 96))
        self.assertTrue(np.isin(cls[:, :20], M.ACQUA).all(), "il mare non e' acqua")
        self.assertFalse(np.isin(cls[:, 40:], M.ACQUA).any(), "la terra e' acqua")
        self.assertTrue(all("classe" in s for s in stat))

    def test_la_costa_diventa_spiaggia(self):
        cls, _ = CA.classifica_adattiva(isola(), rug=np.zeros((96, 96)), n_gruppi=4)
        self.assertTrue((cls[:, 33:35] == M.SPIAGGIA).any())

    def test_il_rilievo_alto_diventa_montagna(self):
        rug = np.zeros((96, 96))
        rug[:, 45:60] = 1.0
        cls, _ = CA.classifica_adattiva(isola(), rug=rug, n_gruppi=4)
        self.assertTrue((cls[10:80, 47:58] == M.MONTAGNA).all())


class TestGuidata(unittest.TestCase):

    def campioni(self):
        im = isola()
        punti = {M.MARE: [(0.1, 0.5), (0.15, 0.3)], M.PRATERIA: [(0.5, 0.5), (0.45, 0.2)],
                 M.DESERTO: [(0.9, 0.5)]}
        return im, CG.campiona(im, punti)

    def test_campiona_da_un_campione_per_ogni_clic(self):
        _, camp = self.campioni()
        self.assertEqual(len(camp), 5)
        self.assertEqual({k for _, k in camp}, {M.MARE, M.PRATERIA, M.DESERTO})
        self.assertEqual(camp[0][0].shape, (3,))

    def test_i_clic_fuori_dall_immagine_non_esplodono(self):
        camp = CG.campiona(isola(), {M.MARE: [(1.5, 1.5), (0.5, 0.5)]})
        self.assertGreaterEqual(len(camp), 1)

    def test_ogni_pixel_prende_la_classe_del_campione_piu_vicino(self):
        im, camp = self.campioni()
        cls = CG.classifica_guidata(im, camp, rug=np.zeros((96, 96)))
        self.assertTrue((cls[:, :20] == M.MARE).all())
        self.assertTrue((cls[:, 40:55] == M.PRATERIA).all())
        self.assertTrue((cls[:, 75:] == M.DESERTO).all())

    def test_senza_campioni_da_errore(self):
        with self.assertRaises(ValueError):
            CG.classifica_guidata(isola(), [])

    def test_il_decoro_non_diventa_spiaggia_ne_montagna(self):
        """Cornici e legende non sono terreno: la costa e la montagna
        geometriche non devono colorarle."""
        im = isola()
        im[:10, :] = (0.9, 0.1, 0.1)                        # una fascia di cornice rossa
        punti = {M.MARE: [(0.1, 0.5)], M.PRATERIA: [(0.5, 0.5)], CG.DECORO: [(0.5, 0.02)]}
        camp = CG.campiona(im, punti)
        rug = np.ones((96, 96))
        cls = CG.classifica_guidata(im, camp, rug=rug)
        self.assertTrue((cls[:8, 45:60] == CG.DECORO).all())

    def test_la_pulizia_toglie_il_sale_e_pepe(self):
        im, camp = self.campioni()
        im[40, 50] = MARE_RGB                               # un pixel di rumore nell'erba
        cls = CG.classifica_guidata(im, camp, rug=np.zeros((96, 96)), pulisci=5)
        self.assertEqual(int(cls[40, 50]), M.PRATERIA)


if __name__ == "__main__":
    unittest.main()
