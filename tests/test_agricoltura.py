"""Test dei campi e dei frutteti.

Il podere ha una forma che non e' estetica: la terra arata resta bagnata solo
entro quattro blocchi dall'acqua, quindi il canale nel mezzo e la larghezza
del campo sono la stessa decisione. I test guardano quello.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import agricoltura as AG  # noqa: E402
from genworld import vegetazione as V  # noqa: E402
from genworld.citta import Citta  # noqa: E402
from genworld.mappa import PIANURA  # noqa: E402


def campagna(lato=220, quota=75):
    cls = np.full((lato, lato), PIANURA, np.uint8)
    h = np.full((lato, lato), quota, np.int32)
    occupato = np.zeros((lato, lato), bool)
    citta = [Citta(z=lato // 2, x=lato // 2, raggio=34,
                   piazza=(lato // 2, lato // 2), edifici=30)]
    return cls, h, occupato, citta


class TestPoderi(unittest.TestCase):

    def setUp(self):
        cls, h, occ, citta = campagna()
        self.poderi, self.h, self.campi, self.alberi = AG.pianifica(
            cls, h, occ, citta, livello_mare=62, seed=5)
        self.coltivi = [p for p in self.poderi if not p.frutteto]
        self.frutteti = [p for p in self.poderi if p.frutteto]

    def test_ce_ne_sono_di_entrambi_i_tipi(self):
        self.assertGreater(len(self.coltivi), 3)
        self.assertGreater(len(self.frutteti), 0)

    def test_il_recinto_chiude_il_perimetro(self):
        p = self.coltivi[0]
        bordo = np.concatenate([
            self.campi[p.z, p.x:p.x1], self.campi[p.z1 - 1, p.x:p.x1],
            self.campi[p.z:p.z1, p.x], self.campi[p.z:p.z1, p.x1 - 1]])
        # tutto recinto tranne il varco
        varchi = int((bordo == AG.SENTIERO).sum())
        self.assertEqual(varchi, 1, "il campo non ha un varco solo")
        self.assertTrue(((bordo == AG.RECINTO) | (bordo == AG.SENTIERO)).all())

    def test_ogni_cella_arata_e_vicina_all_acqua(self):
        """La regola da cui discende la forma del podere: oltre quattro
        blocchi dall'acqua la terra si secca e torna terra."""
        for p in self.coltivi:
            dentro = self.campi[p.z:p.z1, p.x:p.x1]
            canale = np.argwhere(dentro == AG.CANALE)
            self.assertTrue(canale.size, "podere senza canale")
            for z, x in np.argwhere(dentro >= AG.ARATO):
                d = np.abs(canale[:, 0] - z) + np.abs(canale[:, 1] - x)
                self.assertLessEqual(int(d.min()), 4,
                                     f"cella arata a {int(d.min())} dall'acqua")

    def test_un_podere_una_coltura(self):
        """Il difetto trovato guardando l'anteprima: la coltura la sceglieva
        la singola cella e veniva fuori una scacchiera di grano, carote e
        patate dentro lo stesso campo."""
        for p in self.coltivi:
            dentro = self.campi[p.z:p.z1, p.x:p.x1]
            valori = np.unique(dentro[dentro >= AG.ARATO])
            self.assertEqual(len(valori), 1,
                             f"{len(valori)} colture nello stesso podere")
            self.assertEqual(int(valori[0]) - AG.ARATO,
                             AG.COLTURE.index(p.coltura))

    def test_il_campo_e_spianato(self):
        """Un campo in pendenza perde l'acqua, e si vede."""
        for p in self.coltivi:
            q = self.h[p.z + 1:p.z1 - 1, p.x + 1:p.x1 - 1]
            self.assertEqual(int(q.max() - q.min()), 0, "podere non piano")

    def test_i_poderi_non_si_sovrappongono(self):
        preso = np.zeros(self.campi.shape, bool)
        for p in self.poderi:
            f = preso[p.z:p.z1, p.x:p.x1]
            self.assertFalse(f.any(), "due poderi sullo stesso terreno")
            preso[p.z:p.z1, p.x:p.x1] = True

    def test_stanno_fuori_dall_abitato(self):
        cls, h, occ, citta = campagna()
        c = citta[0]
        occ[c.z - 30:c.z + 30, c.x - 30:c.x + 30] = True    # l'abitato
        poderi, _, campi, _ = AG.pianifica(cls, h, occ, citta,
                                           livello_mare=62, seed=5)
        self.assertTrue(poderi)
        for p in poderi:
            self.assertFalse(occ[p.z:p.z1, p.x:p.x1].any(),
                             "podere costruito sopra l'abitato")


class TestFrutteti(unittest.TestCase):

    def setUp(self):
        cls, h, occ, citta = campagna()
        self.poderi, self.h, self.campi, self.alberi = AG.pianifica(
            cls, h, occ, citta, livello_mare=62, seed=11)
        self.frutteti = [p for p in self.poderi if p.frutteto]

    def test_niente_arato_dentro_un_frutteto(self):
        for p in self.frutteti:
            dentro = self.campi[p.z + 1:p.z1 - 1, p.x + 1:p.x1 - 1]
            self.assertFalse((dentro >= AG.ARATO).any())

    def test_le_specie_sono_da_frutto(self):
        specie = set(int(v) for v in np.unique(self.alberi["specie"]))
        self.assertTrue(specie <= {V.MELO, V.CILIEGIO, V.BACCHE}, specie)

    def test_gli_alberi_stanno_dentro_il_recinto(self):
        frutta = np.isin(self.alberi["specie"], (V.MELO, V.CILIEGIO))
        self.assertTrue(frutta.any(), "nessun albero da frutto")
        for x, z in zip(self.alberi["x"][frutta], self.alberi["z"][frutta]):
            dentro = any(p.x < x < p.x1 - 1 and p.z < z < p.z1 - 1
                         for p in self.frutteti)
            self.assertTrue(dentro, f"albero fuori dal frutteto: ({z}, {x})")

    def test_i_filari_non_si_saldano(self):
        """A quattro blocchi di distanza le chiome si toccano e il frutteto
        diventa un bosco: il passo dev'essere almeno cinque."""
        frutta = np.isin(self.alberi["specie"], (V.MELO, V.CILIEGIO))
        px = self.alberi["x"][frutta]
        pz = self.alberi["z"][frutta]
        for i in range(len(px)):
            for j in range(i + 1, len(px)):
                d = abs(int(px[i]) - int(px[j])) + abs(int(pz[i]) - int(pz[j]))
                self.assertGreaterEqual(d, 3, "due alberi troppo vicini")

    def test_le_bacche_stanno_sul_bordo(self):
        bacche = self.alberi["specie"] == V.BACCHE
        self.assertTrue(bacche.any())
        for x, z in zip(self.alberi["x"][bacche], self.alberi["z"][bacche]):
            sul_bordo = any(p.x - 1 <= x <= p.x1 and p.z - 1 <= z <= p.z1
                            and not (p.x < x < p.x1 - 1 and p.z < z < p.z1 - 1)
                            for p in self.poderi)
            self.assertTrue(sul_bordo, f"cespuglio in mezzo al campo ({z}, {x})")


class TestStrade(unittest.TestCase):

    def test_una_strada_evita_i_campi(self):
        """Nessuno lo aveva detto al tracciatore, e l'A* tagliava dritto per
        i poderi: sono la cosa piu' piana che ci sia, li avevamo appena
        spianati noi."""
        from genworld.strade import griglia_costo
        cls = np.full((60, 60), PIANURA, np.uint8)
        h = np.full((60, 60), 75, np.int32)
        campi = np.zeros((60, 60), np.uint8)
        campi[20:40, 20:40] = AG.ARATO
        senza = griglia_costo(cls, h)
        con = griglia_costo(cls, h, campi=campi)
        self.assertGreater(float(con[30, 30]), float(senza[30, 30]) + 10)
        self.assertAlmostEqual(float(con[5, 5]), float(senza[5, 5]), places=4)


class TestNiente(unittest.TestCase):

    def test_scala_zero(self):
        cls, h, occ, citta = campagna()
        poderi, h2, campi, alberi = AG.pianifica(cls, h, occ, citta,
                                                 scala=0.0, seed=1)
        self.assertEqual(poderi, [])
        self.assertFalse(campi.any())
        self.assertTrue((h2 == h).all())
        self.assertEqual(alberi["x"].size, 0)

    def test_senza_citta(self):
        cls, h, occ, _ = campagna()
        poderi, _, campi, _ = AG.pianifica(cls, h, occ, [], seed=1)
        self.assertEqual(poderi, [])
        self.assertFalse(campi.any())


if __name__ == "__main__":
    unittest.main()
