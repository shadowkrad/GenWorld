"""Abitanti: dove stanno dentro una casa da template (non nel muro, non sotto il pavimento)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import template as TM  # noqa: E402
from genworld.edifici import Edificio  # noqa: E402

TAV = [("dirt", {}), ("planks", {}), ("stone_bricks", {}), ("crafting_table", {}),
       ("air", {}), ("carpet", {}), ("bed", {}), ("door", {})]
DIRT, ASSI, MURO, BANCO, ARIA, TAPPETO, LETTO, PORTA = range(8)


def casa(piano_pavimento=2, mobile_al_centro=True, tetto=True, lato=7):
    """Un modello lato x 8 x lato: due strati di terra, il pavimento, due strati di
    stanza con il muro intorno e il soffitto. `piano_pavimento` e' lo strato delle assi."""
    c = np.full((lato, 8, lato), -1, np.int16)
    c[:, :piano_pavimento, :] = DIRT
    c[:, piano_pavimento, :] = ASSI
    for y in (piano_pavimento + 1, piano_pavimento + 2):
        c[:, y, :] = ARIA
        c[0, y, :] = c[-1, y, :] = c[:, y, 0] = c[:, y, -1] = MURO
    if tetto:
        c[:, piano_pavimento + 3, :] = MURO
    m = int(lato // 2)
    if mobile_al_centro:
        c[m, piano_pavimento + 1, m] = BANCO
    return TM.Modello(nome="casa", celle=c, tavolozza=list(TAV))


class TestPosti(unittest.TestCase):
    def test_non_il_centro_se_e_un_mobile(self):
        m = casa()
        posti = TM.posti_in_piedi(m, 0)
        self.assertTrue(posti)
        self.assertNotEqual(posti[0][::2], (3, 3))
        # il primo e' vicino al centro
        x, y, z = posti[0]
        self.assertLessEqual(abs(x - 3) + abs(z - 3), 2)

    def test_sopra_il_pavimento_non_dentro(self):
        """Il pavimento e' allo strato `affondo` (2): si sta sullo strato dopo."""
        m = casa(piano_pavimento=2)
        self.assertEqual(m.affondo, 2)
        for x, y, z in TM.posti_in_piedi(m, 0):
            self.assertEqual(y, 3)
        m = casa(piano_pavimento=4)
        # con il pavimento piu' in alto lo strato terra e' 4: l'abitante sta a 5
        self.assertTrue(all(y == m.affondo + 1 for x, y, z in TM.posti_in_piedi(m, 0)))

    def test_mai_nel_muro_ne_su_un_mobile(self):
        m = casa()
        for q in range(4):
            c = np.rot90(m.celle, k=-q, axes=(2, 0)) if q else m.celle
            for x, y, z in TM.posti_in_piedi(m, q):
                self.assertEqual(c[x, y, z], ARIA, (q, x, y, z))
                self.assertEqual(c[x, y + 1, z], ARIA, (q, x, y, z))
                self.assertNotEqual(c[x, y - 1, z], ARIA)

    def test_al_coperto(self):
        # un tetto solo sopra meta' casa: i posti stanno sotto il tetto
        m = casa()
        m.celle[:, 5, 4:] = -1
        for x, y, z in TM.posti_in_piedi(m, 0):
            self.assertLess(z, 4)

    def test_senza_tetto_si_accetta_l_aperto(self):
        m = casa(tetto=False)
        self.assertTrue(TM.posti_in_piedi(m, 0))

    def test_pieno_senza_stanze_nessun_posto(self):
        c = np.full((5, 5, 5), MURO, np.int16)
        m = TM.Modello(nome="blocco", celle=c, tavolozza=list(TAV))
        self.assertEqual(TM.posti_in_piedi(m, 0), [])

    def test_il_tappeto_si_calpesta_la_porta_no(self):
        m = casa()
        m.celle[1:6, 3, 1:6] = TAPPETO                        # tutta la stanza e' tappeto
        posti = TM.posti_in_piedi(m, 0)
        self.assertTrue(posti)
        m2 = casa(mobile_al_centro=False)
        m2.celle[1:6, 3, 1:6] = PORTA                         # un campo di porte: non ci si sta
        self.assertEqual(TM.posti_in_piedi(m2, 0), [])

    def test_ruotato_il_posto_resta_dentro_la_stanza(self):
        m = casa(lato=7)
        for q in range(4):
            for x, y, z in TM.posti_in_piedi(m, q):
                self.assertTrue(1 <= x <= 5 and 1 <= z <= 5)

    def test_coordinate_assolute_nel_lotto(self):
        m = casa()
        ed = Edificio(x=100, z=200, larghezza=9, profondita=9, base=70)
        celle = TM.celle_di_casa(m, 0, ed)
        px = 100 + (9 - 7) // 2
        pz = 200 + (9 - 7) // 2
        x, y, z = celle[0]
        self.assertTrue(px < x < px + 6 and pz < z < pz + 6)
        # lo strato `affondo` (2) sta a quota ed.base (70): la cella a strato 3 e' a 71
        self.assertEqual(y, 70 - m.affondo + 3)


if __name__ == "__main__":
    unittest.main()
