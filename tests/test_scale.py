"""Verifica dei conti di scala. Eseguire con: python3 -m unittest discover tests"""

import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld.fidelity import stima
from genworld.presets import GARDA, ITALIA, MONUMENTI, insediamenti_italia
from genworld.scale import (Scala, Territorio, mondo_per_errore, mondo_per_scala)


class TestScala(unittest.TestCase):

    def test_metri_per_blocco(self):
        s = Scala(1000, ITALIA)
        self.assertAlmostEqual(s.metri_per_blocco, 1290.0)

    def test_esagerazione_uguale_scala_se_oggetti_1a1(self):
        s = Scala(1000, ITALIA)
        self.assertAlmostEqual(s.fattore_esagerazione, s.metri_per_blocco)

    def test_budget_verticale(self):
        s = Scala(1000, ITALIA, livello_mare_y=62)
        self.assertEqual(s.blocchi_sopra_mare, 258)

    def test_esagerazione_ottimale_usa_tutto_il_budget(self):
        s = Scala(1000, ITALIA).con_esagerazione_ottimale()
        # la quota massima deve cadere esattamente sul tetto verticale
        self.assertAlmostEqual(
            s.quota_max_rappresentabile_m, ITALIA.quota_max_m, places=3
        )

    def test_torre_eiffel_non_ci_sta(self):
        s = Scala(1000, ITALIA)
        r = s.resa_landmark(MONUMENTI["torre_eiffel"])
        self.assertTrue(r["compresso"])
        self.assertEqual(r["blocchi"], 258)
        self.assertLess(r["fedelta"], 1.0)

    def test_colosseo_ci_sta(self):
        s = Scala(1000, ITALIA)
        r = s.resa_landmark(MONUMENTI["colosseo"])
        self.assertFalse(r["compresso"])
        self.assertEqual(r["blocchi"], 48)

    def test_regimi(self):
        self.assertEqual(Scala(1000, ITALIA).regime, "caricatura")
        self.assertEqual(Scala(20000, GARDA).regime, "riproduzione realistica")

    def test_tre_modi_sono_coerenti(self):
        lato = mondo_per_scala(ITALIA, 100.0)
        self.assertAlmostEqual(Scala(lato, ITALIA).metri_per_blocco, 100.0, places=2)
        lato2 = mondo_per_errore(ITALIA, 50.0)
        self.assertLessEqual(Scala(lato2, ITALIA).metri_per_blocco / 2, 50.0)


class TestStima(unittest.TestCase):

    def test_forma_dipende_solo_dalla_risoluzione(self):
        """Stesso lato in blocchi -> stessa fedelta' di forma, territori diversi."""
        a = stima(Scala(1024, ITALIA))
        b = stima(Scala(1024, GARDA))
        pa = {d.nome: d.punteggio for d in a.dimensioni}["Forma (coste, confini)"]
        pb = {d.nome: d.punteggio for d in b.dimensioni}["Forma (coste, confini)"]
        self.assertAlmostEqual(pa, pb)

    def test_piu_blocchi_piu_insediamenti(self):
        ins = insediamenti_italia()
        piccolo = stima(Scala(512, ITALIA), insediamenti=ins)
        grande = stima(Scala(8192, ITALIA), insediamenti=ins)
        self.assertLess(len(piccolo.insediamenti_tenuti), len(grande.insediamenti_tenuti))

    def test_avviso_di_taglio_del_rilievo(self):
        """Esagerando troppo in verticale, le Alpi sfondano il tetto."""
        s = Scala(1000, ITALIA, esagerazione_verticale=200.0)
        self.assertLess(s.quota_max_rappresentabile_m, ITALIA.quota_max_m)
        st = stima(s)
        self.assertTrue(any("tagliato" in a for a in st.avvisi))

    def test_avviso_budget_verticale_sottoutilizzato(self):
        """Senza esagerazione, su un territorio compresso il rilievo e' piatto."""
        st = stima(Scala(20000, ITALIA))
        self.assertTrue(any("sottoutilizzato" in a for a in st.avvisi))

    def test_indice_fra_zero_e_uno(self):
        st = stima(Scala(1024, GARDA), insediamenti=[], landmark=[MONUMENTI["colosseo"]])
        self.assertGreaterEqual(st.indice, 0.0)
        self.assertLessEqual(st.indice, 1.0)


if __name__ == "__main__":
    unittest.main()
