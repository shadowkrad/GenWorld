"""Forme della pianta: quadrato, rettangolo, ovale e tondo."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import citta as C
from genworld import mappa as M

C_ = (110, 110)


def piana(lato=46, forma=None, forme="quadrato", seed=1, murata_raggio=40):
    H = W = 240
    cls = np.full((H, W), M.PIANURA, np.uint8)
    h = np.full((H, W), 70, np.int32)
    liv = np.full((H, W), 62.0, np.float32)
    return C.pianifica(cls, h, liv, [(120, 120, murata_raggio)], seed=seed, forme=forme)


class TestMetrica(unittest.TestCase):
    def griglia(self, n=200, c=100):
        z, x = np.ogrid[:n, :n]
        return np.broadcast_to(z - c, (n, n)), np.broadcast_to(x - c, (n, n))

    def test_il_quadrato_coincide_con_la_distanza_di_chebyshev_di_prima(self):
        dz, dx = self.griglia()
        D = C._metrica(C.forma_quadrata(46), 46, dz, dx)
        self.assertTrue((D == np.maximum(np.abs(dz), np.abs(dx))).all())

    def test_il_tondo_e_la_distanza_euclidea_e_l_ovale_con_assi_uguali_gli_somiglia(self):
        dz, dx = self.griglia()
        D = C._metrica(C.Forma("tondo", 46, 46), 46, dz, dx)
        self.assertTrue((D == np.rint(np.hypot(dz, dx)).astype(int)).all())
        O = C._metrica(C.Forma("ovale", 46, 46), 46, dz, dx)
        self.assertLessEqual(int(np.abs(O - D).max()), 1)

    def test_il_contorno_sta_dove_deve(self):
        """D == lato lungo gli assi, alla semi-estensione di ogni forma."""
        z0 = np.array([0]);
        for forma, atteso_z, atteso_x in (
                (C.Forma("rettangolo", 34, 46), 34, 46),
                (C.Forma("ovale", 34, 46), 34, 46),
                (C.Forma("tondo", 46, 46), 46, 46)):
            self.assertEqual(int(C._metrica(forma, 46, np.array([atteso_z]), z0)[0]), 46, forma)
            self.assertEqual(int(C._metrica(forma, 46, z0, np.array([atteso_x]))[0]), 46, forma)

    def test_la_distanza_cresce_lungo_ogni_raggio_su_ogni_forma(self):
        for tipo in C.FORME:
            f = C.Forma(tipo, 34, 46) if tipo in ("rettangolo", "ovale") else C.Forma(tipo, 46, 46)
            for ang in np.linspace(0, np.pi / 2, 13):
                r = np.arange(20, 120)
                dz = np.rint(r * np.sin(ang)).astype(int)
                dx = np.rint(r * np.cos(ang)).astype(int)
                D = C._metrica(f, 46, dz, dx)
                self.assertTrue((np.diff(D) >= -1).all(), (tipo, ang))
                self.assertGreater(int(D[-1]), int(D[0]))

    def test_il_centro_dell_ovale_non_da_errori(self):
        D = C._metrica(C.Forma("ovale", 34, 46), 46, np.array([0, 0]), np.array([0, 1]))
        self.assertTrue(np.isfinite(D).all())
        self.assertLess(int(D[0]), 46)


class TestSceltaForma(unittest.TestCase):
    def test_quadrato_e_i_villaggi_restano_quadrati(self):
        rng = np.random.default_rng(1)
        self.assertEqual(C.scegli_forma("quadrato", 46, True, rng), C.forma_quadrata(46))
        self.assertEqual(C.scegli_forma("miste", 46, False, rng), C.forma_quadrata(46))

    def test_ogni_forma_ha_le_misure_giuste(self):
        rng = np.random.default_rng(1)
        r = C.scegli_forma("rettangolo", 46, True, rng)
        self.assertEqual(r.tipo, "rettangolo")
        self.assertEqual(max(r.lz, r.lx), 46)
        self.assertLess(min(r.lz, r.lx), 46)
        self.assertGreaterEqual(min(r.lz, r.lx) / 46, 0.6)
        t = C.scegli_forma("tondo", 38, True, rng)
        self.assertEqual((t.tipo, t.lz, t.lx), ("tondo", 38, 38))

    def test_le_piccole_non_diventano_allungate(self):
        rng = np.random.default_rng(1)
        self.assertEqual(C.scegli_forma("rettangolo", 30, True, rng).tipo, "quadrato")
        self.assertEqual(C.scegli_forma("ovale", 30, True, rng).tipo, "tondo")

    def test_miste_le_da_tutte_e_la_forma_e_deterministica_sul_seme(self):
        viste = {C.scegli_forma("miste", 46, True, np.random.default_rng(s)).tipo
                 for s in range(80)}
        self.assertEqual(viste, set(C.FORME))
        a = C.scegli_forma("miste", 46, True, np.random.default_rng(7))
        b = C.scegli_forma("miste", 46, True, np.random.default_rng(7))
        self.assertEqual(a, b)

    def test_una_forma_sconosciuta_solleva(self):
        with self.assertRaises(ValueError):
            C.scegli_forma("triangolo", 46, True, np.random.default_rng(1))


class TestPianta(unittest.TestCase):
    def test_il_quadrato_e_identico_a_prima(self):
        ed, h, vie, muro, citta, *_ = piana(forme="quadrato")
        self.assertEqual(citta[0].forma, "quadrato")
        self.assertEqual((citta[0].lz, citta[0].lx), (citta[0].lato, citta[0].lato))
        self.assertEqual(len(ed), 16)

    def test_ogni_forma_produce_una_citta_murata_con_case_strade_e_porte(self):
        for forma in ("rettangolo", "ovale", "tondo"):
            ed, h, vie, muro, citta, banchi, urbano, *_ = piana(forme=forma)
            self.assertEqual(len(citta), 1, forma)
            c = citta[0]
            self.assertTrue(c.murata, forma)
            self.assertEqual(c.forma, forma)
            self.assertGreaterEqual(len(ed), 6, forma)
            self.assertEqual(int((muro == 2).sum() > 0), 1, f"{forma}: nessuna porta")
            self.assertGreater(int((vie > 0).sum()), 100, forma)
            self.assertEqual(len(c.porte), 4, forma)

    def test_le_quattro_porte_sono_sugli_assi_su_ogni_forma(self):
        for forma in C.FORME:
            ed, h, vie, muro, citta, *_ = piana(forme=forma)
            c = citta[0]
            zc, xc = c.z, c.x
            porta = muro == 2
            zz, xx = np.nonzero(porta)
            nord = (zz < zc - 5) & (np.abs(xx - xc) <= 2)
            sud = (zz > zc + 5) & (np.abs(xx - xc) <= 2)
            est = (xx > xc + 5) & (np.abs(zz - zc) <= 2)
            ovest = (xx < xc - 5) & (np.abs(zz - zc) <= 2)
            for nome, m in (("nord", nord), ("sud", sud), ("est", est), ("ovest", ovest)):
                self.assertTrue(m.any(), f"{forma}: manca la porta {nome}")

    def test_i_lotti_non_si_sovrappongono_ne_toccano_strade_e_mura(self):
        for forma in C.FORME:
            ed, h, vie, muro, citta, *_ = piana(forme=forma)
            occ = np.zeros(vie.shape, int)
            for e in ed:
                occ[e.z:e.z + e.profondita, e.x:e.x + e.larghezza] += 1
                self.assertFalse((vie[e.z:e.z + e.profondita, e.x:e.x + e.larghezza] > 0).any(),
                                 f"{forma}: casa sulla strada")
                self.assertFalse((muro[e.z:e.z + e.profondita, e.x:e.x + e.larghezza] > 0).any(),
                                 f"{forma}: casa sul muro")
            self.assertLessEqual(int(occ.max()), 1, f"{forma}: due case sulle stesse celle")

    def test_la_cinta_e_chiusa_su_ogni_forma(self):
        """Nessun varco fuori dalle quattro porte: la cinta divide dentro da fuori."""
        from scipy.ndimage import binary_fill_holes
        for forma in C.FORME:
            ed, h, vie, muro, citta, *_ = piana(forme=forma)
            c = citta[0]
            chiusa = binary_fill_holes(muro > 0)
            self.assertTrue(chiusa[c.z, c.x], forma)
            self.assertFalse(chiusa[c.z, c.x + c.lx + 12], forma)
            self.assertFalse(chiusa[c.z - c.lz - 12, c.x], forma)

    def test_le_case_stanno_dentro_la_forma(self):
        for forma in ("ovale", "tondo", "rettangolo"):
            ed, h, vie, muro, citta, *_ = piana(forme=forma)
            c = citta[0]
            f = C.Forma(c.forma, c.lz, c.lx)
            for e in ed:
                dz = np.array([e.z - c.z, e.z + e.profondita - 1 - c.z] * 2)
                dx = np.array([e.x - c.x, e.x - c.x, e.x + e.larghezza - 1 - c.x,
                               e.x + e.larghezza - 1 - c.x])
                self.assertTrue((C._metrica(f, c.lato, dz, dx) <= c.lato - 3).all(),
                                f"{forma}: casa fuori dalla circonvallazione")

    def test_il_fosso_circonda_la_cinta_su_ogni_forma(self):
        for forma in C.FORME:
            ed, h, vie, muro, citta, banchi, urbano, cls, liv = piana(forme=forma)
            c = citta[0]
            self.assertGreater(int((cls == M.FIUME).sum()), 60, forma)
            self.assertTrue((cls[c.z, c.x + c.lx + 5: c.x + c.lx + 7] == M.FIUME).any(), forma)

    def test_il_rettangolo_e_piu_stretto_da_un_lato(self):
        ed, h, vie, muro, citta, *_ = piana(forme="rettangolo")
        c = citta[0]
        self.assertNotEqual(c.lz, c.lx)
        zz, xx = np.nonzero(muro > 0)
        self.assertEqual(int(zz.max() - zz.min()) + 1 > int(xx.max() - xx.min()) + 1, c.lz > c.lx)

    def test_il_piano_e_deterministico(self):
        a = piana(forme="miste", seed=3)
        b = piana(forme="miste", seed=3)
        self.assertEqual((a[4][0].forma, a[4][0].lz, a[4][0].lx),
                         (b[4][0].forma, b[4][0].lz, b[4][0].lx))
        self.assertTrue((a[2] == b[2]).all())

    def test_senza_citta_murata_la_forma_resta_quadrata(self):
        ed, h, vie, muro, citta, *_ = piana(forme="tondo", murata_raggio=20)
        self.assertEqual(citta[0].forma, "quadrato")


class TestScaleELotti(unittest.TestCase):
    def test_le_citta_curve_non_hanno_scalinate_e_le_rettangolari_si(self):
        for forma, atteso in (("tondo", False), ("ovale", False), ("rettangolo", True),
                              ("quadrato", True)):
            ed, h, vie, muro, citta, *_ = piana(forme=forma)
            self.assertEqual(bool((muro >= C.SCALA_BASE).any()), atteso, forma)

    def test_le_scalinate_del_rettangolo_stanno_contro_la_cinta(self):
        ed, h, vie, muro, citta, *_ = piana(forme="rettangolo")
        c = citta[0]
        zz, xx = np.nonzero(muro >= C.SCALA_BASE)
        self.assertGreater(len(zz), 10)
        for z, x in zip(zz, xx):
            vicino = muro[max(0, z - 2):z + 3, max(0, x - 2):x + 3]
            self.assertTrue(((vicino == 1) | (vicino == 3)).any(), "scalinata lontana dalla cinta")


if __name__ == "__main__":
    unittest.main()
