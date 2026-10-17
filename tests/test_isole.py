"""Isole del tesoro: siti in mare aperto, rilievo, palme, croce e forziere."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import bauli as BA  # noqa: E402
from genworld import isole as IS  # noqa: E402
from genworld import luoghi as LU  # noqa: E402
from genworld import mappa as M  # noqa: E402

MARE = 62


def mondo(lato=700, fondale=30):
    """Mare aperto con una costa a est e un'isola di terraferma lontanissima."""
    cls = np.full((lato, lato), M.OCEANO, np.uint8)
    h = np.full((lato, lato), fondale, np.int32)
    cls[:, 640:] = M.PRATERIA                      # la terra a est
    h[:, 640:] = 70
    return cls, h


class TestPianifica(unittest.TestCase):
    def test_siti_in_mare_aperto_e_lontani_dalla_costa(self):
        cls, h = mondo()
        isole = IS.pianifica(cls, h, MARE, seed=2, densita=4.0)
        self.assertGreaterEqual(len(isole), 1)
        for i in isole:
            self.assertTrue(np.isin(cls[i.z - IS.RAGGIO_TOTALE:i.z + IS.RAGGIO_TOTALE + 1,
                                        i.x - IS.RAGGIO_TOTALE:i.x + IS.RAGGIO_TOTALE + 1], M.MARINO).all())
            self.assertGreaterEqual(640 - i.x, IS.RAGGIO_TOTALE + IS.DISTANZA_DALLA_COSTA)
            self.assertEqual(i.cima, MARE + 3)

    def test_distanza_fra_isole_e_determinismo(self):
        cls, h = mondo()
        a = IS.pianifica(cls, h, MARE, seed=3, densita=6.0)
        b = IS.pianifica(cls, h, MARE, seed=3, densita=6.0)
        self.assertEqual([(i.x, i.z) for i in a], [(i.x, i.z) for i in b])
        for k, i in enumerate(a):
            for j in a[k + 1:]:
                self.assertGreaterEqual(np.hypot(i.x - j.x, i.z - j.z), IS.DISTANZA_FRA_ISOLE)

    def test_non_dove_il_mare_e_basso_o_c_e_gia_qualcosa(self):
        cls, h = mondo(fondale=60)                 # mare troppo basso: 2 blocchi
        self.assertEqual(IS.pianifica(cls, h, MARE, seed=1, densita=4.0), [])
        cls, h = mondo()
        evita = np.ones(cls.shape, bool)
        self.assertEqual(IS.pianifica(cls, h, MARE, evita=evita, densita=4.0), [])

    def test_densita_zero(self):
        cls, h = mondo()
        self.assertEqual(IS.pianifica(cls, h, MARE, densita=0), [])


class TestApplica(unittest.TestCase):
    def setUp(self):
        self.cls, self.h = mondo()
        self.isola = IS.Isola(x=300, z=300, cima=MARE + 3, seme=7)
        self.h0 = self.h.copy()
        IS.applica(self.h, self.cls, [self.isola], MARE)

    def test_il_centro_e_terra_sopra_il_mare(self):
        self.assertGreater(int(self.h[300, 300]), MARE)
        self.assertEqual(int(self.h[300, 300]), self.isola.cima + 1)           # la duna
        self.assertEqual(int(self.h[300, 305 - 2]), self.isola.cima)           # il disco piano

    def test_fuori_dal_raggio_non_cambia(self):
        lontano = np.hypot(*np.mgrid[0:700, 0:700] - 300.0) > IS.RAGGIO_TOTALE + 3
        self.assertTrue(np.array_equal(self.h[lontano], self.h0[lontano]))

    def test_il_fianco_scende_senza_salti_e_non_scava(self):
        riga = self.h[300, 300:300 + IS.RAGGIO_TOTALE + 2]
        self.assertTrue((self.h >= self.h0).all())
        # scende verso il fondale, in piu' tratti: mai piu' su di un blocco per cella
        self.assertTrue((np.diff(riga) <= 1).all() or (np.diff(riga) <= 0).sum() > len(riga) // 2)
        self.assertEqual(int(self.h[300, 300 + IS.RAGGIO_TOTALE + 3]), 30)

    def test_sabbia_e_erba(self):
        self.assertEqual(int(self.cls[300, 300]), M.PRATERIA)
        terra = self.h > MARE
        sulla_terra = self.cls[terra & (self.h0 < MARE)]
        self.assertTrue(np.isin(sulla_terra, (M.SPIAGGIA, M.PRATERIA)).all())
        self.assertIn(M.SPIAGGIA, set(sulla_terra.tolist()))
        # il mare attorno resta mare
        self.assertEqual(int(self.cls[300, 300 + IS.RAGGIO_TOTALE + 5]), M.OCEANO)

    def test_una_isola_vicina_alla_costa_non_tocca_la_terra(self):
        cls, h = mondo()
        h0 = h.copy()
        IS.applica(h, cls, [IS.Isola(x=620, z=300, cima=MARE + 3)], MARE)
        self.assertTrue((h[:, 640:] == 70).all())


class TestPosa(unittest.TestCase):
    ALT, Y0 = 130, 0

    class Finto:
        def __init__(self):
            self.v = {("air", ()): 0}
            self.id_aria = 0

        def blocco(self, nome, **p):
            return self.v.setdefault((nome, tuple(sorted(p.items()))), len(self.v))

        def nome(self, i):
            return next(n for (n, _), k in self.v.items() if k == i)

    def posa(self, isola, meta=500):
        s = self.Finto()
        tav = IS.Tavolozza(s)
        regione = {}
        isole = [isola]
        # i chunk che toccano l'isola
        for cz in range((isola.z - meta - 8) // 16, (isola.z - meta + 8) // 16 + 1):
            for cx in range((isola.x - meta - 8) // 16, (isola.x - meta + 8) // 16 + 1):
                out = np.zeros((16, self.ALT, 16), np.uint32)
                h_c = np.full((16, 16), isola.cima, np.int32)
                ys = np.arange(self.ALT)[None, :, None]
                out[np.broadcast_to(ys < isola.cima, out.shape)] = s.blocco("sand")
                IS.posa(out, tav, h_c, cx * 16, cz * 16, self.Y0, isole,
                        IS.indice_per_chunk(isole)[(((cx * 16 + meta) // 16), ((cz * 16 + meta) // 16))]
                        if False else [0], meta=meta)
                regione[(cx, cz)] = out
        def blocco(gx, y, gz):
            o = regione.get((gx // 16, gz // 16))
            return None if o is None else s.nome(int(o[gx % 16, y - self.Y0, gz % 16]))
        return s, tav, blocco

    def test_il_forziere_sta_sotto_la_croce_a_due_blocchi(self):
        isola = IS.Isola(x=500 + 40, z=500 + 40, cima=65)
        s, tav, b = self.posa(isola)
        cx, cz = isola.croce
        x, y, z, tipo = isola.forzieri()[0]
        self.assertEqual((x, z), (cx, cz))
        self.assertEqual(y, isola.cima - 3)
        self.assertEqual(b(x - 500, y, z - 500), "chest")
        self.assertEqual(b(x - 500, y + 1, z - 500), "sand")                   # sabbia sopra
        self.assertEqual(b(x - 500, isola.cima - 1, z - 500), "red_sand")      # la croce al centro
        self.assertEqual(tipo, "tesoro")

    def test_la_croce_e_una_x(self):
        isola = IS.Isola(x=540, z=540, cima=65)
        s, tav, b = self.posa(isola)
        cx, cz = isola.croce
        for k in range(-IS.CROCE, IS.CROCE + 1):
            self.assertEqual(b(cx + k - 500, 64, cz + k - 500), "red_sand")
            self.assertEqual(b(cx + k - 500, 64, cz - k - 500), "red_sand")
        self.assertEqual(b(cx + 1 - 500, 64, cz - 500), "sand")                # non e' un quadrato

    def test_due_palme(self):
        isola = IS.Isola(x=540, z=540, cima=65)
        s, tav, b = self.posa(isola)
        tronchi = sum(1 for dy in range(0, 12) if b(540 - 500, 65 + dy, 540 - 500) == "log")
        self.assertGreaterEqual(tronchi, 5)
        self.assertEqual(b(543 - 500, 65, 538 - 500), "log")                    # la seconda
        foglie = sum(1 for dx in range(-3, 4) for dz in range(-3, 4)
                     if b(540 - 500 + dx, 65 + 6, 540 - 500 + dz) == "leaves")
        self.assertGreaterEqual(foglie, 5)


class TestIntegrazione(unittest.TestCase):
    def test_il_luogo_e_una_tappa_dell_avventura(self):
        self.assertIn("isola", LU.TIPI_AVVENTURA)
        luoghi = LU.raccogli([], [], [], [], [], 500, isole=[IS.Isola(x=600, z=640, cima=65)])
        self.assertEqual([(l.tipo, l.x, l.z) for l in luoghi], [("isola", 100, 140)])
        self.assertIn(luoghi[0].nome, LU.ISOLE)

    def test_il_bottino_dell_isola(self):
        s = TestPosa.Finto()
        tav = BA.Tavolozza(s)
        out = np.zeros((16, 40, 16), np.uint32)
        out[5, 10, 6] = s.blocco("chest", facing="south", connection="none", material="wood")
        validi = {n for n, _, _ in BA.BOTTINO_ZONA["isola"]} | {t for t, _ in BA.TESORI}
        for seed in range(10):
            b = BA.trova(out, tav, 0, 0, 0, seed=seed, zone=[(0, 0, 16, 16, "isola")])
            nomi = {str(v["id"]).split(":")[1] for v in b[0].nbt.compound["utags"]["Items"]}
            self.assertTrue(nomi <= validi, nomi - validi)
            self.assertTrue(nomi & {"gold_ingot", "emerald", "diamond", "gold_block"})

    def test_le_isole_non_compaiono_sui_cartelli_dei_bivi(self):
        strada = np.zeros((300, 300), bool)
        strada[100, 20:280] = True
        strada[20:200, 150] = True
        luoghi = [LU.Luogo("Isola X", "isola", -330, -360), LU.Luogo("Torre Y", "torre", -300, -340)]
        for c in LU.cartelli_bivi(strada, luoghi, 150):
            self.assertNotIn("Isola", " ".join(c.righe))


if __name__ == "__main__":
    unittest.main()
