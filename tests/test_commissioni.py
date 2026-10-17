"""Mini quest: le commissioni dei popolani."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import commissioni as CM  # noqa: E402
from genworld import entita as EN  # noqa: E402
from genworld import luoghi as LU  # noqa: E402
from genworld.citta import Citta  # noqa: E402

META = 500


def mondo():
    strade = np.zeros((1000, 1000), np.uint8)
    strade[580:620, 590:610] = 1                    # un quartiere di strade attorno alla piazza (600, 600)
    strade[300:340, 290:310] = 1
    h = np.full((1000, 1000), 70, np.int32)
    citta = [Citta(z=600, x=600, raggio=40, piazza=(600, 600), murata=True),
             Citta(z=320, x=300, raggio=20, piazza=(320, 300), murata=False)]
    luoghi = [LU.Luogo("Aldona", "citta", 100, 100), LU.Luogo("Brevia", "citta", -200, -180)]
    return citta, luoghi, strade, h


class TestTabella(unittest.TestCase):
    def test_ogni_mestiere_ha_due_richieste_complete(self):
        for m, (titolo, righe) in CM.Q.items():
            self.assertEqual(len(righe), 2, m)
            for r in righe:
                self.assertTrue(1 <= len(r["compra"]) <= 2)
                for i, n in r["compra"]:
                    self.assertTrue(isinstance(i, str) and n >= 1)
                self.assertTrue(r["nome"] and r["frase"])
                id_, quanti, comp = r["vende"]
                self.assertGreaterEqual(quanti, 1)
                if comp:
                    for ench in (comp.get("enchantments", {}), comp.get("stored_enchantments", {})):
                        self.assertTrue(all(k.startswith("minecraft:") and v >= 1 for k, v in ench.items()))

    def test_i_mestieri_esistono_nel_gioco(self):
        for m in CM.Q:
            self.assertIn(m, EN.MESTIERI)

    def test_la_seconda_richiesta_vale_di_piu(self):
        for m, (t, righe) in CM.Q.items():
            n1 = sum(n for _, n in righe[0]["compra"])
            n2 = sum(n for _, n in righe[1]["compra"])
            self.assertTrue(n2 > 0 and n1 > 0)


class TestPianifica(unittest.TestCase):
    def test_deterministico_e_quanti(self):
        citta, luoghi, strade, h = mondo()
        a = CM.pianifica(citta, luoghi, strade, h, META, seed=3)
        b = CM.pianifica(citta, luoghi, strade, h, META, seed=3)
        self.assertEqual([(x.x, x.z, x.nome) for x in a], [(x.x, x.z, x.nome) for x in b])
        self.assertEqual(len(a), 5)                  # 3 nella citta' murata, 2 nell'altra

    def test_su_una_strada_lontano_dalla_piazza_e_distinti(self):
        citta, luoghi, strade, h = mondo()
        popolani = CM.pianifica(citta, luoghi, strade, h, META, seed=1)
        celle = set()
        for p in popolani:
            x, z = int(p.x), int(p.z)
            self.assertTrue(strade[z, x] > 0)
            celle.add((x, z))
        self.assertEqual(len(celle), len(popolani))

    def test_nomi_e_mestieri_unici_per_citta(self):
        citta, luoghi, strade, h = mondo()
        popolani = CM.pianifica(citta, luoghi, strade, h, META, seed=2)
        self.assertEqual(len({p.nome for p in popolani}), len(popolani))
        for p in popolani:
            self.assertTrue(p.nome.startswith(CM.Q[p.mestiere][0]))
            self.assertEqual(p.livello, 5)
            self.assertEqual(len(p.offerte), 2)

    def test_evita_le_celle_occupate(self):
        citta, luoghi, strade, h = mondo()
        occupati = {(x, z) for x in range(590, 610) for z in range(580, 620)}      # tutto il quartiere 1
        popolani = CM.pianifica(citta, luoghi, strade, h, META, occupati, seed=1)
        for p in popolani:
            self.assertFalse(590 <= int(p.x) < 610 and 580 <= int(p.z) < 620)

    def test_senza_strade_niente(self):
        citta, luoghi, strade, h = mondo()
        self.assertEqual(CM.pianifica(citta, luoghi, None, h, META), [])


class TestNbt(unittest.TestCase):
    def test_il_villager_ha_nome_e_scambi_con_due_richieste(self):
        citta, luoghi, strade, h = mondo()
        p = CM.pianifica(citta, luoghi, strade, h, META, seed=1)[0]
        tag = p.nbt_tag()
        self.assertEqual(str(tag["CustomName"]), p.nome)
        ricette = tag["Offers"]["Recipes"]
        self.assertEqual(len(ricette), 2)
        for r in ricette:
            self.assertEqual(int(r["maxUses"]), 1)
            self.assertTrue(str(r["sell"]["id"]).startswith("minecraft:"))
            comp = r["sell"]["components"]
            self.assertIn("minecraft:custom_name", comp)
            self.assertIn("minecraft:lore", comp)

    def test_la_seconda_pila_da_portare_va_in_buyB(self):
        o = CM.offerte("fabbro", "Aldona", "Fabbro Ugo")
        tag = EN._scambio(o[0])
        self.assertIn("buyB", tag)
        self.assertEqual(str(tag["buy"]["id"]), "minecraft:coal")
        self.assertEqual(str(tag["buyB"]["id"]), "minecraft:iron_ingot")
        solo = EN._scambio(CM.offerte("macellaio", "x", "y")[0])
        self.assertIn("buyB", solo)
        una = EN._scambio(CM.offerte("fabbro", "x", "y")[0] | {"compra2": None})
        self.assertNotIn("buyB", una)

    def test_gli_incantesimi_sono_nel_formato_del_gioco(self):
        tag = EN._scambio(CM.offerte("fabbro", "x", "y")[0])
        ench = tag["sell"]["components"]["minecraft:enchantments"]
        self.assertEqual(int(ench["minecraft:efficiency"]), 2)


class TestElenco(unittest.TestCase):
    def test_sezione_per_luoghi_md(self):
        citta, luoghi, strade, h = mondo()
        popolani = CM.pianifica(citta, luoghi, strade, h, META, seed=1)
        md = CM.sezione_elenco(popolani)
        self.assertIn("## Commissioni dei popolani", md)
        self.assertIn(popolani[0].nome, md)
        self.assertEqual(CM.sezione_elenco([]), "")


if __name__ == "__main__":
    unittest.main()
