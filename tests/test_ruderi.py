"""Megaliti e ruderi antichi."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import arredi as AR  # noqa: E402
from genworld import mappa as M  # noqa: E402
from genworld import ruderi as RU  # noqa: E402
from genworld import template as TM  # noqa: E402
from genworld.citta import Citta  # noqa: E402

VER = TM.traduttore((26, 2, 0))


def nome(m, x, y, z):
    i = int(m.celle[x, y, z])
    return None if i < 0 else m.tavolozza[i][0]


class TestDisegni(unittest.TestCase):
    def test_si_traducono_e_hanno_le_dimensioni(self):
        for tipo, fabbrica in RU.FABBRICHE.items():
            for seme in range(6):
                m = fabbrica(seme).modello(VER)
                self.assertEqual(m.celle.shape, RU.DIMENSIONI[tipo], (tipo, seme))
                self.assertGreater(int((m.celle >= 0).sum()), 30)

    def test_deterministici_e_diversi_fra_loro(self):
        for fabbrica in RU.FABBRICHE.values():
            a, b = fabbrica(5).modello(VER), fabbrica(5).modello(VER)
            self.assertTrue(np.array_equal(a.celle, b.celle))
            c = fabbrica(6).modello(VER)
            self.assertFalse(a.celle.shape == c.celle.shape and np.array_equal(a.celle, c.celle))

    def test_il_cromlech_ha_pietre_in_cerchio_e_il_trilite(self):
        m = RU.cromlech(1).modello(VER)
        # pietre in piedi: colonne alte almeno 3 sul bordo, dentro il raggio 7-8
        alte = [(x, z) for x in range(17) for z in range(17)
                if int(m.celle[x, 2, z]) >= 0 and int(np.hypot(x - 8, z - 8)) in (6, 7, 8)]
        self.assertGreaterEqual(len(alte), 5)
        # il trilite al centro: due pilastri e le lastre sopra
        self.assertEqual(nome(m, 8, 3, 7) in ("stone_bricks",), True)
        self.assertEqual(nome(m, 8, 4, 8), "slab")
        self.assertEqual(nome(m, 7, 1, 8), "candle")

    def test_il_tempio_ha_colonne_a_quote_diverse(self):
        for seme in range(8):
            m = RU.tempio(seme).modello(VER)
            alt = []
            for x in (1, 4, 7, 10):
                for z in (1, 5):
                    alt.append(sum(1 for y in range(1, 8) if int(m.celle[x, y, z]) >= 0))
            self.assertGreater(len(set(alt)), 1, seme)

    def test_il_dolmen_ha_la_lastra_sopra_i_piedritti(self):
        m = RU.dolmen(2).modello(VER)
        for x in (1, 2, 4, 5):
            for y in range(3):
                self.assertGreaterEqual(int(m.celle[x, y, 3]), 0)
        self.assertGreater(int((m.celle[:, 3, :] >= 0).sum()), 12)


def mondo(lato=700):
    cls = np.full((lato, lato), M.PRATERIA, np.uint8)
    h = np.full((lato, lato), 75, np.int32)
    strada = np.zeros((lato, lato), np.uint8)
    strada[300, :] = 1
    citta = [Citta(z=300, x=100, raggio=30, piazza=(300, 100), murata=True, lato=30)]
    return cls, h, strada, citta


class TestPianifica(unittest.TestCase):
    def esegui(self, seed=1, dens=4.0, **kw):
        cls, h, strada, citta = mondo()
        for k, v in kw.items():
            if k == "cls":
                cls = v
        ris = AR.Risultato()
        c = RU.pianifica(ris, cls, h, strada, citta, None, VER, seed=seed, densita=dens)
        return ris, c, strada, citta

    def test_ne_mette_di_tutti_i_tipi_e_riusa_il_risultato(self):
        ris, c, *_ = self.esegui(dens=12.0)
        self.assertGreaterEqual(sum(c.values()), 5)
        self.assertEqual(len(ris.arredi), sum(c.values()))
        self.assertEqual(len(ris.modelli), len(ris.arredi))
        self.assertTrue(set(a.tipo for a in ris.arredi) <= set(RU.TIPI))

    def test_deterministico(self):
        a = self.esegui(seed=3)[0]
        b = self.esegui(seed=3)[0]
        self.assertEqual([(x.tipo, x.x, x.z) for x in a.arredi], [(x.tipo, x.x, x.z) for x in b.arredi])

    def test_lontano_da_strade_paesi_e_fra_loro(self):
        ris, c, strada, citta = self.esegui(dens=10.0)
        for a in ris.arredi:
            self.assertFalse(strada[a.z - 3:a.z + a.profondita + 3, a.x - 3:a.x + a.larghezza + 3].any(), a)
            r = 30 + RU.DISTANZA_DAL_PAESE
            self.assertFalse(abs(a.x + a.larghezza // 2 - 100) <= r and abs(a.z + a.profondita // 2 - 300) <= r, a)
        for i, a in enumerate(ris.arredi):
            for b in ris.arredi[i + 1:]:
                self.assertGreaterEqual(np.hypot(a.x - b.x, a.z - b.z), RU.DISTANZA_FRA_RUDERI - 17)

    def test_solo_su_terra_asciutta_e_piana(self):
        cls, h, strada, citta = mondo()
        cls[:, 350:] = M.OCEANO
        zz, xx = np.mgrid[:700, :700]
        h = (75 + 6 * np.sin(xx / 3.0)).astype(np.int32)         # tutto troppo ondulato
        ris = AR.Risultato()
        RU.pianifica(ris, cls, h, strada, citta, None, VER, seed=2, densita=12.0)
        for a in ris.arredi:
            q = h[a.z:a.z + a.profondita, a.x:a.x + a.larghezza]
            self.assertLessEqual(int(q.max() - q.min()), 2)
            self.assertTrue((cls[a.z:a.z + a.profondita, a.x:a.x + a.larghezza] != M.OCEANO).all())

    def test_densita_zero_e_mare(self):
        cls, h, strada, citta = mondo()
        ris = AR.Risultato()
        self.assertEqual(sum(RU.pianifica(ris, cls, h, strada, citta, None, VER, densita=0).values()), 0)
        cls[:, :] = M.OCEANO
        self.assertEqual(sum(RU.pianifica(ris, cls, h, strada, citta, None, VER, densita=10).values()), 0)
        self.assertEqual(ris.arredi, [])


if __name__ == "__main__":
    unittest.main()
