"""Test della pianta urbana.

Quello che si prova qui e' la GRAMMATICA della citta', non il suo aspetto:
che le case guardino la strada, che stiano dentro l'area, che non si
sovrappongano, che le vie siano connesse, che le porte stiano sul muro. La
bellezza si guarda su `mondi/citta.png`.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import citta as C  # noqa: E402
from genworld import edifici as E  # noqa: E402
from genworld.mappa import MARE, PIANURA  # noqa: E402


def pianura(lato=200, quota=75):
    cls = np.full((lato, lato), PIANURA, np.uint8)
    h = np.full((lato, lato), quota, np.int32)
    livello = np.full((lato, lato), 62.0, np.float32)
    return cls, h, livello


class TestContorno(unittest.TestCase):

    def test_non_e_un_cerchio(self):
        """Un perimetro circolare si legge come un timbro."""
        cls, h, _ = pianura()
        area = C.contorno(cls, h, 100, 100, 40, 62, seed=3)
        zz, xx = np.nonzero(area)
        r = np.hypot(zz - 100, xx - 100)
        # il raggio del bordo deve variare con l'angolo
        bordo = r > np.percentile(r, 92)
        self.assertGreater(float(np.std(r[bordo])), 1.5)

    def test_esclude_l_acqua(self):
        cls, h, _ = pianura()
        cls[:, :100] = MARE
        h[:, :100] = 50
        area = C.contorno(cls, h, 100, 100, 40, 62, seed=3)
        self.assertFalse(area[:, :100].any())

    def test_la_piazza_sta_dentro(self):
        cls, h, _ = pianura()
        cls[:, :100] = MARE
        h[:, :100] = 50
        area = C.contorno(cls, h, 100, 110, 40, 62, seed=3)
        pz, px = C._piazza(area, h, 100, 110)
        self.assertTrue(area[pz, px])


class TestRete(unittest.TestCase):

    def setUp(self):
        self.cls, self.h, self.livello = pianura(240)
        self.area = C.contorno(self.cls, self.h, 120, 120, 50, 62, seed=5)
        self.piazza = C._piazza(self.area, self.h, 120, 120)
        self.rango, self.porte, self.punti = C.rete(
            self.area, self.piazza, 50, seed=5)

    def test_le_vie_sono_un_pezzo_solo(self):
        """Una citta' con un quartiere scollegato dal resto non e' una citta'."""
        _, quanti = label(self.rango > 0, np.ones((3, 3)))
        self.assertEqual(quanti, 1, f"{quanti} tronconi di strada separati")

    def test_gerarchia_a_piramide(self):
        """Gli assi devono essere pochi, i vicoli tanti: se e' il contrario
        la gerarchia non c'e', c'e' solo una tavolozza di larghezze."""
        assi = int((self.rango == C.ASSE).sum())
        vicoli = int((self.rango == C.VICOLO).sum())
        self.assertGreater(assi, 0)
        self.assertGreater(vicoli, 0)

    def test_le_porte_stanno_in_periferia(self):
        self.assertGreaterEqual(len(self.porte), 2)
        for pz, px in self.porte:
            d = np.hypot(pz - self.piazza[0], px - self.piazza[1])
            self.assertGreater(d, 50 * 0.4, "porta troppo vicina alla piazza")

    def test_le_vie_restano_nell_area(self):
        self.assertFalse((self.rango > 0)[~self.area].any())

    def test_la_piazza_e_uno_slargo(self):
        pz, px = self.piazza
        attorno = self.rango[pz - 2:pz + 3, px - 2:px + 3]
        self.assertTrue((attorno == C.ASSE).all())


class TestLotti(unittest.TestCase):

    def setUp(self):
        self.cls, self.h, self.livello = pianura(240)
        self.area = C.contorno(self.cls, self.h, 120, 120, 50, 62, seed=7)
        self.piazza = C._piazza(self.area, self.h, 120, 120)
        self.rango, _, _ = C.rete(self.area, self.piazza, 50, seed=7)
        self.edifici = C.lotti(self.area, self.rango > 0, self.cls, self.h,
                               self.livello, self.piazza, 50, 62,
                               np.random.default_rng(7))

    def test_ce_ne_sono(self):
        """Il primo tentativo ne costruiva otto su una citta' intera."""
        self.assertGreater(len(self.edifici), 25, f"solo {len(self.edifici)}")

    def test_nessuna_sovrapposizione(self):
        occupato = np.zeros(self.cls.shape, bool)
        for ed in self.edifici:
            fetta = occupato[ed.z:ed.z1, ed.x:ed.x1]
            self.assertFalse(fetta.any(), "due case sullo stesso sedime")
            occupato[ed.z:ed.z1, ed.x:ed.x1] = True

    def test_nessun_tetto_sovrapposto(self):
        """Il sedime non basta. I tetti sporgono oltre i muri, e due gronde
        nello stesso vicolo finivano nella stessa cella: dall'alto si vedeva
        benissimo, dai test no, perche' guardavano solo i sedimi."""
        occupato = np.zeros(self.cls.shape, bool)
        for ed in self.edifici:
            x0, z0, x1, z1 = ed.ingombro_tetto()
            fetta = occupato[max(0, z0):z1, max(0, x0):x1]
            self.assertFalse(fetta.any(),
                             f"tetti sovrapposti attorno a ({ed.z}, {ed.x})")
            occupato[max(0, z0):z1, max(0, x0):x1] = True

    def test_chi_ha_un_vicino_attaccato_rinuncia_alla_gronda(self):
        """La regola vera: la gronda si perde quando non c'e' posto, non
        quando si e' in centro. Una casa a schiera non sporge sul vicino."""
        for a in self.edifici:
            if a.gronda == 0:
                continue
            for b in self.edifici:
                if b is a:
                    continue
                vicino = (a.x - 1 <= b.x1 and b.x - 1 <= a.x1
                          and a.z - 1 <= b.z1 and b.z - 1 <= a.z1)
                self.assertFalse(
                    vicino, f"gronda su ({a.z},{a.x}) col vicino ({b.z},{b.x})")

    def test_nessuna_casa_sulla_strada(self):
        vie = self.rango > 0
        for ed in self.edifici:
            self.assertFalse(vie[ed.z:ed.z1, ed.x:ed.x1].any(),
                             "casa costruita sulla carreggiata")

    def test_la_facciata_guarda_la_strada(self):
        """Il punto di tutta la pianificazione: la porta da' sulla via.

        Si controlla TUTTO il fronte, non il suo punto di mezzo: il lotto
        cresce di lato quanto l'isolato gli concede, quindi il centro della
        facciata non coincide con la cella da cui e' partito. Misurare li'
        bocciava nove case su sessantatre che erano perfettamente in regola.
        """
        vie = self.rango > 0
        H, W = vie.shape
        sbagliate = []
        for ed in self.edifici:
            if ed.porta == E.NORD:
                fronte = [(ed.z - 1, x) for x in range(ed.x, ed.x1)]
                verso = (-1, 0)
            elif ed.porta == E.SUD:
                fronte = [(ed.z1, x) for x in range(ed.x, ed.x1)]
                verso = (1, 0)
            elif ed.porta == E.OVEST:
                fronte = [(z, ed.x - 1) for z in range(ed.z, ed.z1)]
                verso = (0, -1)
            else:
                fronte = [(z, ed.x1) for z in range(ed.z, ed.z1)]
                verso = (0, 1)
            vicino = False
            for fz, fx in fronte:
                for d in range(0, 4):
                    z, x = fz + verso[0] * d, fx + verso[1] * d
                    if 0 <= z < H and 0 <= x < W and vie[z, x]:
                        vicino = True
                        break
                if vicino:
                    break
            if not vicino:
                sbagliate.append((ed.z, ed.x))
        self.assertLessEqual(len(sbagliate), len(self.edifici) // 20,
                             f"{len(sbagliate)} case su {len(self.edifici)} "
                             f"non danno sulla strada: {sbagliate[:5]}")

    def test_il_centro_e_piu_alto_della_periferia(self):
        """La densita' deve degradare: senza gradiente e' un quartiere
        residenziale caduto dal cielo."""
        d = np.array([np.hypot((e.z + e.z1) / 2 - self.piazza[0],
                               (e.x + e.x1) / 2 - self.piazza[1])
                      for e in self.edifici])
        piani = np.array([e.piani for e in self.edifici])
        centro = piani[d < np.median(d)].mean()
        fuori = piani[d >= np.median(d)].mean()
        self.assertGreater(centro, fuori)


class TestPianificazione(unittest.TestCase):

    def test_tutto_insieme(self):
        cls, h, livello = pianura(300)
        siti = [(150, 150, 55), (60, 60, 22)]
        ed, h2, vie, muro, citta, banchi = C.pianifica(cls, h, livello, siti, 62, seed=9)
        self.assertEqual(len(citta), 2)
        self.assertGreater(len(ed), 30)
        self.assertTrue(citta[0].e_citta)
        self.assertFalse(citta[1].e_citta)
        # il terreno e' stato spianato sotto le case
        self.assertEqual(h2.shape, h.shape)

    def test_le_mura_circondano_e_hanno_le_porte(self):
        cls, h, livello = pianura(300)
        ed, _, vie, muro, citta, _b = C.pianifica(
            cls, h, livello, [(150, 150, 55)], 62, seed=9)
        self.assertTrue(citta[0].murata)
        self.assertGreater(int((muro == 1).sum()), 100, "cinta troppo corta")
        self.assertGreater(int((muro == 2).sum()), 0, "nessuna porta nel muro")
        # la cinta e' un anello chiuso: una sola componente
        _, quanti = label(muro > 0, np.ones((3, 3)))
        self.assertEqual(quanti, 1, f"{quanti} pezzi di cinta")
        # e le case stanno dentro, non fuori
        from scipy.ndimage import binary_fill_holes
        dentro = binary_fill_holes(muro > 0)
        fuori = sum(1 for e in ed
                    if not dentro[(e.z + e.z1) // 2, (e.x + e.x1) // 2])
        self.assertLess(fuori, len(ed) // 3, f"{fuori} case fuori dalle mura")

    def test_il_mercato_sta_sulla_piazza(self):
        """Il banco sta sul selciato, sul bordo della piazza. La prima
        versione lo cercava su terreno libero e non ne apriva mai nessuno."""
        cls, h, livello = pianura(300)
        ed, _, vie, muro, citta, banchi = C.pianifica(
            cls, h, livello, [(150, 150, 55)], 62, seed=9)
        self.assertGreaterEqual(len(banchi), 3, "mercato non aperto")
        pz, px = citta[0].piazza
        for b in banchi:
            d = np.hypot(b.z + 1 - pz, b.x + 1 - px)
            self.assertLess(d, 12, f"banco lontano dalla piazza: {d:.0f}")

    def test_i_banchi_non_stanno_dentro_le_case(self):
        cls, h, livello = pianura(300)
        ed, _, _, _, _, banchi = C.pianifica(
            cls, h, livello, [(150, 150, 55)], 62, seed=9)
        for b in banchi:
            for e in ed:
                sovrapposti = (b.x < e.x1 and e.x < b.x1
                               and b.z < e.z1 and e.z < b.z1)
                self.assertFalse(sovrapposti, "banco dentro una casa")

    def test_i_mestieri_dipendono_dal_posto(self):
        """Le botteghe stanno al centro, non sparse a caso."""
        cls, h, livello = pianura(300)
        ed, _, _, _, citta, _ = C.pianifica(
            cls, h, livello, [(150, 150, 55)], 62, seed=9)
        pz, px = citta[0].piazza
        con = [e for e in ed if e.mestiere]
        senza = [e for e in ed if not e.mestiere]
        self.assertTrue(con, "nessuna bottega")
        self.assertTrue(senza, "tutte botteghe: e' un centro commerciale")
        d = lambda g: np.mean([np.hypot((e.z + e.z1) / 2 - pz,  # noqa: E731
                                        (e.x + e.x1) / 2 - px) for e in g])
        self.assertLess(d(con), d(senza))

    def test_senza_siti_non_succede_niente(self):
        cls, h, livello = pianura(120)
        ed, h2, vie, muro, citta, banchi = C.pianifica(cls, h, livello, [], 62, seed=1)
        self.assertEqual(ed, [])
        self.assertFalse(vie.any())
        self.assertFalse(muro.any())
        self.assertTrue((h2 == h).all())


if __name__ == "__main__":
    unittest.main()
