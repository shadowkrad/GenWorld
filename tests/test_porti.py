"""Porti: molo, barche ormeggiate e faro per gli abitati sulla costa."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import porti as PO  # noqa: E402
from genworld import template as TM  # noqa: E402
from genworld.citta import Citta  # noqa: E402
from genworld.mappa import MARE, PIANURA  # noqa: E402

try:
    VER = TM.traduttore((1, 21, 4))
except Exception:                                        # noqa: BLE001
    VER = None


def mondo(n=300, costa_x=200, fondale=55):
    """Terra a sinistra, mare a destra; il fondale scende piano oltre la riva."""
    cls = np.full((n, n), PIANURA, np.uint8)
    h = np.full((n, n), 64, np.int32)
    cls[:, costa_x:] = MARE
    for x in range(costa_x, n):
        h[:, x] = max(fondale, 61 - (x - costa_x) // 3)
    return cls, h


@unittest.skipUnless(VER, "PyMCTranslate non disponibile")
class TestDisegni(unittest.TestCase):

    def test_il_molo_ha_pali_tavolato_e_lanterne(self):
        m = PO.molo(12).modello(VER)
        self.assertEqual((m.dx, m.dy, m.dz), (12, PO.PALI + 3, 3))
        nomi = {n for n, _ in m.tavolozza}
        self.assertTrue({"spruce_planks", "spruce_log", "lantern"} <= {
            n.replace("universal_minecraft:", "") for n in nomi} or len(nomi) >= 3)
        deck = m.celle[:, PO.PALI, :]
        self.assertTrue((deck >= 0).all(), "tavolato con buchi")
        self.assertTrue((m.celle[0, :PO.PALI, 0] >= 0).all(), "palo all'imbocco")

    def test_il_molo_ruotato_guarda_dalla_parte_giusta(self):
        m = PO.molo(10).modello(VER)
        cat = TM.Catalogo([m], None)
        for (dz, dx), q in PO._QUARTI.items():
            c = cat.celle(0, q)
            piena = (c[:, PO.PALI, :] >= 0)
            lungo_x = dz == 0
            self.assertEqual(piena.shape, (10, 3) if lungo_x else (3, 10))
            # l'imbocco (lato di terra) e' il primo strato lungo l'asse, nel verso opposto al mare
            if dx > 0:
                self.assertTrue(piena[0, :].all())
            if dx < 0:
                self.assertTrue(piena[-1, :].all())
            if dz > 0:
                self.assertTrue(piena[:, 0].all())
            if dz < 0:
                self.assertTrue(piena[:, -1].all())

    def test_il_faro_e_alto_e_ha_la_lanterna(self):
        m = PO.faro().modello(VER)
        self.assertEqual((m.dx, m.dz), (PO.LATO_FARO, PO.LATO_FARO))
        self.assertGreaterEqual(m.dy, 18)
        # cavo: il centro del piano terra e' libero
        self.assertEqual(int(m.celle[3, 3, 3]), -1)
        self.assertTrue((m.celle[3, 15:17, 3] >= 0).all(), "senza lanterna marina")


@unittest.skipUnless(VER, "PyMCTranslate non disponibile")
class TestPosto(unittest.TestCase):

    def barca(self):
        celle = np.full((16, 10, 6), -1, np.int32)
        celle[:, :4, :] = 0
        m = TM.Modello(nome="b", celle=celle, tavolozza=[("planks", {})])
        m.linea_acqua = 2
        return m

    def pianifica(self, cls, h, citta, barche=()):
        evita = np.zeros(cls.shape, bool)
        return PO.pianifica(citta, cls, h, (MARE,), 62, evita, list(barche), 100, 200, VER, seed=1)

    def test_un_abitato_sulla_costa_ha_un_molo_in_mare(self):
        cls, h = mondo()
        citta = [Citta(z=150, x=160, raggio=30, piazza=(150, 160), murata=False, lato=20)]
        pezzi, extra = self.pianifica(cls, h, citta)
        moli = [p for p in pezzi if p.modello >= 200]
        self.assertEqual(len(moli), 1)
        self.assertEqual(len(extra), 1)
        p = moli[0]
        self.assertEqual(p.base, PO.PIANO_MOLO - PO.PALI)
        # il molo si allunga dentro il mare, verso est
        self.assertGreater(p.x + p.larghezza, 200)
        self.assertLess(p.x, 200)

    def test_senza_mare_vicino_niente_porto(self):
        cls, h = mondo(costa_x=290)
        citta = [Citta(z=100, x=60, raggio=30, piazza=(100, 60), murata=False, lato=20)]
        pezzi, extra = self.pianifica(cls, h, citta)
        self.assertEqual((pezzi, extra), ([], []))

    def test_un_laghetto_non_e_un_mare(self):
        cls = np.full((300, 300), PIANURA, np.uint8)
        h = np.full((300, 300), 64, np.int32)
        cls[140:160, 190:210] = MARE
        h[140:160, 190:210] = 55
        citta = [Citta(z=150, x=160, raggio=20, piazza=(150, 160), murata=False, lato=15)]
        pezzi, _ = self.pianifica(cls, h, citta)
        self.assertEqual(pezzi, [])

    def test_le_barche_stanno_in_acqua_a_fianco_del_molo(self):
        cls, h = mondo(fondale=50)
        citta = [Citta(z=150, x=160, raggio=30, piazza=(150, 160), murata=False, lato=20)]
        pezzi, _ = self.pianifica(cls, h, citta, barche=[self.barca()])
        barche = [p for p in pezzi if 100 <= p.modello < 200]
        self.assertGreaterEqual(len(barche), 1)
        molo = [p for p in pezzi if p.modello >= 200][0]
        for b in barche:
            self.assertEqual(b.base, 62 - 2, "linea di galleggiamento sbagliata")
            self.assertTrue(cls[b.z:b.z + b.profondita, b.x:b.x + b.larghezza].tolist()
                            == [[MARE] * b.larghezza] * b.profondita, "barca sulla terra")
            # non si sovrappone al molo
            fuori = (b.x + b.larghezza <= molo.x or molo.x + molo.larghezza <= b.x
                     or b.z + b.profondita <= molo.z or molo.z + molo.profondita <= b.z)
            self.assertTrue(fuori)

    def test_il_faro_solo_nelle_citta_murate(self):
        cls, h = mondo()
        borgo = Citta(z=150, x=160, raggio=30, piazza=(150, 160), murata=False, lato=20)
        citta_m = Citta(z=150, x=160, raggio=30, piazza=(150, 160), murata=True, lato=45)
        _, extra_b = self.pianifica(cls, h, [borgo])
        _, extra_c = self.pianifica(cls, h, [citta_m])
        self.assertEqual(len(extra_b), 1)       # il solo molo
        self.assertEqual(len(extra_c), 2)       # molo e faro


if __name__ == "__main__":
    unittest.main()


class TestMoloDaSchema(unittest.TestCase):
    """Il molo da schema: allineato alla costa e girato verso il mare."""

    def modello(self):
        # 12 lungo x (terra a x >= 6), 3 alto, 10 lungo z; un marcatore (7) a (6, 1, 5)
        celle = np.full((12, 3, 10), -1, np.int32)
        celle[:, 1, :] = 0
        celle[6, 1, 5] = 1
        m = TM.Modello(nome="molo", celle=celle, tavolozza=[("planks", {}), ("log", {})])
        m.linea_acqua = 1
        m.meta = {"costa_x": 6, "centro_z": 5, "linea_acqua": 1, "affondo": 0}
        return m

    def test_ruotare_un_punto_segue_la_rotazione_del_catalogo(self):
        m = self.modello()
        cat = TM.Catalogo([m], None)
        for q in range(4):
            x, z = PO._ruota_punto(6, 5, m.dx, m.dz, q)
            ruotato = cat.celle(0, q)
            self.assertEqual(int(ruotato[x, 1, z]), 1, f"q={q}")

    def test_il_molo_si_posa_con_la_terra_verso_terra(self):
        from genworld.citta import Citta
        for costa_x, lato_mare in ((200, "est"), (100, "ovest")):
            n = 300
            cls = np.full((n, n), PIANURA, np.uint8)
            h = np.full((n, n), 64, np.int32)
            if lato_mare == "est":
                cls[:, costa_x:] = MARE
                h[:, costa_x:] = 58
                c = Citta(z=150, x=170, raggio=15, piazza=(150, 170), murata=False, lato=15)
            else:
                cls[:, :costa_x] = MARE
                h[:, :costa_x] = 58
                c = Citta(z=150, x=130, raggio=15, piazza=(150, 130), murata=False, lato=15)
            m = self.modello()
            occ = np.zeros((n, n), bool)
            res = PO._molo_da_modello(m, 50, c, h, cls == MARE, occ, 62)
            self.assertIsNotNone(res, lato_mare)
            pezzo, (pz, px, (dz, dx)) = res
            self.assertEqual((dz, dx), (0, 1) if lato_mare == "est" else (0, -1))
            self.assertEqual(pezzo.modello, 50)
            self.assertEqual(pezzo.base, 62 - 1 + 0)
            # la meta' di mare del modello sta sul mare, quella di terra sulla terra
            lungo = slice(pezzo.x, pezzo.x + pezzo.larghezza)
            fascia = cls[pezzo.z:pezzo.z + pezzo.profondita, lungo]
            self.assertTrue((fascia == MARE).any() and (fascia != MARE).any())

    def test_senza_riva_non_c_e_molo(self):
        from genworld.citta import Citta
        n = 200
        cls = np.full((n, n), PIANURA, np.uint8)
        h = np.full((n, n), 64, np.int32)
        c = Citta(z=100, x=100, raggio=15, piazza=(100, 100), murata=False, lato=15)
        res = PO._molo_da_modello(self.modello(), 0, c, h, cls == MARE,
                                  np.zeros((n, n), bool), 62)
        self.assertIsNone(res)
