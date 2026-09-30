"""Test della fauna: semina selvatica, animali da cortile, e la scrittura
mista con gli abitanti nello stesso file region.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import fauna as FA
from genworld import mappa as M

try:
    from amulet_nbt import load as nbt_load  # noqa: F401
    NBT = True
except ImportError:
    NBT = False


def scenario(lato=256, mare=62):
    """Meta' foresta a quota 80, meta' oceano a quota 40."""
    cls = np.full((lato, lato), M.OCEANO, np.uint8)
    cls[:, lato // 2:] = M.FORESTA
    h = np.full((lato, lato), 40, np.int32)
    h[:, lato // 2:] = 80
    return cls, h


class TestSeminaSelvatica(unittest.TestCase):

    def test_mai_nell_acqua(self):
        cls, h = scenario()
        a = FA.semina_selvatica(cls, h, livello_mare=62, scala_densita=5.0, seed=1)
        self.assertGreater(len(a), 0)
        for an in a:
            self.assertNotEqual(cls[int(an.z), int(an.x)], M.OCEANO)

    def test_niente_sulla_mappa_tutta_acqua(self):
        cls = np.full((64, 64), M.OCEANO, np.uint8)
        h = np.full((64, 64), 40, np.int32)
        self.assertEqual(FA.semina_selvatica(cls, h, seed=2), [])

    def test_densita_scala(self):
        cls, h = scenario()
        poco = len(FA.semina_selvatica(cls, h, scala_densita=0.2, seed=3))
        molto = len(FA.semina_selvatica(cls, h, scala_densita=5.0, seed=3))
        self.assertGreater(molto, poco)

    def test_zero_densita(self):
        cls, h = scenario()
        self.assertEqual(FA.semina_selvatica(cls, h, scala_densita=0.0, seed=4), [])

    def test_deterministica(self):
        cls, h = scenario()
        a = FA.semina_selvatica(cls, h, seed=5)
        b = FA.semina_selvatica(cls, h, seed=5)
        self.assertEqual([(x.x, x.z, x.specie) for x in a],
                         [(x.x, x.z, x.specie) for x in b])

    def test_solo_le_specie_di_classe_giusta(self):
        """Niente cammello nel bosco, niente lupo nel deserto."""
        cls = np.full((256, 256), M.DESERTO, np.uint8)
        h = np.full((256, 256), 80, np.int32)
        a = FA.semina_selvatica(cls, h, scala_densita=5.0, seed=6)
        self.assertGreater(len(a), 0)
        specie_attese = {sp for sp, _ in FA.MISCELA[M.DESERTO]}
        self.assertTrue({an.specie for an in a} <= specie_attese)


class TestVarianti(unittest.TestCase):
    """Colore per bioma: volpe delle nevi, coniglio bianco - vedi fauna.py
    per dove sono verificati i nomi/valori dei tag NBT."""

    def scenario_neve(self, lato=256):
        cls = np.full((lato, lato), M.NEVE, np.uint8)
        h = np.full((lato, lato), 80, np.int32)
        return cls, h

    def test_volpe_bianca_sulla_neve(self):
        cls, h = self.scenario_neve()
        a = FA.semina_selvatica(cls, h, scala_densita=5.0, seed=10)
        volpi = [an for an in a if an.specie == "volpe"]
        self.assertGreater(len(volpi), 0)
        self.assertTrue(all(v.variante == "snow" for v in volpi))

    def test_volpe_rossa_altrove(self):
        cls = np.full((256, 256), M.PRATERIA, np.uint8)
        h = np.full((256, 256), 80, np.int32)
        a = FA.semina_selvatica(cls, h, scala_densita=5.0, seed=11)
        volpi = [an for an in a if an.specie == "volpe"]
        self.assertGreater(len(volpi), 0)
        self.assertTrue(all(v.variante == "red" for v in volpi))

    def test_coniglio_bianco_o_a_chiazze_sulla_neve(self):
        cls, h = self.scenario_neve()
        a = FA.semina_selvatica(cls, h, scala_densita=5.0, seed=12)
        conigli = [an for an in a if an.specie == "coniglio"]
        self.assertGreater(len(conigli), 0)
        attesi = {FA.CONIGLIO_BIANCO, FA.CONIGLIO_CHIAZZE}
        self.assertTrue(all(c.variante in attesi for c in conigli))

    def test_coniglio_misto_temperato_altrove(self):
        cls = np.full((256, 256), M.PIANURA, np.uint8)
        h = np.full((256, 256), 80, np.int32)
        a = FA.semina_selvatica(cls, h, scala_densita=5.0, seed=13)
        conigli = [an for an in a if an.specie == "coniglio"]
        self.assertGreater(len(conigli), 0)
        attesi = {FA.CONIGLIO_MARRONE, FA.CONIGLIO_SALE, FA.CONIGLIO_NERO}
        self.assertTrue(all(c.variante in attesi for c in conigli))

    def test_altre_specie_senza_variante(self):
        cls = np.full((256, 256), M.DESERTO, np.uint8)
        h = np.full((256, 256), 80, np.int32)
        a = FA.semina_selvatica(cls, h, scala_densita=5.0, seed=14)
        self.assertGreater(len(a), 0)
        self.assertTrue(all(an.variante is None for an in a))


@unittest.skipUnless(NBT, "amulet-nbt non installato in questo interprete")
class TestNbtVariante(unittest.TestCase):

    def test_volpe_delle_nevi_ha_type_snow(self):
        a = FA.Animale(x=0.0, y=70.0, z=0.0, specie="volpe", variante="snow")
        self.assertEqual(str(a.nbt_tag()["Type"]), "snow")

    def test_volpe_senza_variante_e_rossa(self):
        a = FA.Animale(x=0.0, y=70.0, z=0.0, specie="volpe")
        self.assertEqual(str(a.nbt_tag()["Type"]), "red")

    def test_coniglio_bianco_ha_rabbittype_1(self):
        a = FA.Animale(x=0.0, y=70.0, z=0.0, specie="coniglio",
                       variante=FA.CONIGLIO_BIANCO)
        self.assertEqual(int(a.nbt_tag()["RabbitType"]), 1)

    def test_coniglio_senza_variante_e_marrone(self):
        a = FA.Animale(x=0.0, y=70.0, z=0.0, specie="coniglio")
        self.assertEqual(int(a.nbt_tag()["RabbitType"]), FA.CONIGLIO_MARRONE)

    def test_una_mucca_non_ha_ne_type_ne_rabbittype(self):
        c = FA.Animale(x=0.0, y=70.0, z=0.0, specie="mucca").nbt_tag()
        self.assertNotIn("Type", c)
        self.assertNotIn("RabbitType", c)


class TestPerPoderi(unittest.TestCase):

    class FintoPodere:
        def __init__(self, x, z, lato, base, frutteto=False):
            self.x, self.z, self.lato, self.base = x, z, lato, base
            self.frutteto = frutteto

    def test_gli_animali_stanno_dentro_il_podere(self):
        p = self.FintoPodere(x=10, z=10, lato=11, base=70)
        animali = FA.per_poderi([p], quanti=(3, 3), seed=1)
        self.assertEqual(len(animali), 3)
        for a in animali:
            self.assertTrue(p.x <= a.x <= p.x + p.lato, a.x)
            self.assertTrue(p.z <= a.z <= p.z + p.lato, a.z)
            self.assertIn(a.specie, FA.CORTILE)

    def test_niente_bestie_nel_frutteto(self):
        p = self.FintoPodere(x=10, z=10, lato=17, base=70, frutteto=True)
        self.assertEqual(FA.per_poderi([p], seed=2), [])

    def test_nessun_podere_nessun_animale(self):
        self.assertEqual(FA.per_poderi([], seed=3), [])


@unittest.skipUnless(NBT, "amulet-nbt non installato in questo interprete")
class TestNbtAnimale(unittest.TestCase):

    def test_id_di_gioco_e_salute(self):
        a = FA.Animale(x=1.5, y=70.0, z=2.5, specie="cammello", seme=1)
        c = a.nbt_tag()
        self.assertEqual(str(c["id"]), "minecraft:camel")
        self.assertEqual(float(c["Health"]), 32.0)

    def test_non_sparisce(self):
        a = FA.Animale(x=0.0, y=70.0, z=0.0, specie="lupo")
        self.assertEqual(int(a.nbt_tag()["PersistenceRequired"]), 1)

    def test_specie_ignota_non_esplode(self):
        """Un nome di specie sbagliato non deve far fallire la scrittura del
        mondo: meglio una mucca di troppo che un chunk che non si carica."""
        a = FA.Animale(x=0.0, y=70.0, z=0.0, specie="boh")
        self.assertEqual(str(a.nbt_tag()["id"]), "minecraft:cow")

    def test_due_animali_hanno_uuid_diversi(self):
        a = FA.Animale(x=0.0, y=70.0, z=0.0, seme=1).nbt_tag()
        b = FA.Animale(x=0.0, y=70.0, z=0.0, seme=2).nbt_tag()
        self.assertNotEqual(list(a["UUID"]), list(b["UUID"]))


@unittest.skipUnless(NBT, "amulet-nbt non installato in questo interprete")
class TestScritturaMista(unittest.TestCase):
    """Il punto delicato: abitanti e animali nello STESSO file region."""

    def test_non_si_scavalcano(self):
        import tempfile

        from genworld.entita import Abitante, leggi_chunk, scrivi_regioni
        with tempfile.TemporaryDirectory() as d:
            misti = [Abitante(x=5.5, y=71.0, z=6.5, mestiere="fabbro", seme=1),
                     FA.Animale(x=5.5, y=71.0, z=6.5, specie="mucca", seme=2),
                     FA.Animale(x=20.5, y=71.0, z=20.5, specie="lupo", seme=3)]
            st = scrivi_regioni(d, misti, 4189)
            self.assertEqual(st["abitanti"], 3)
            c0 = leggi_chunk(os.path.join(d, "entities", "r.0.0.mca"), 0, 0)
            c1 = leggi_chunk(os.path.join(d, "entities", "r.0.0.mca"), 1, 1)
            self.assertEqual(len(c0["Entities"]), 2)
            self.assertEqual(len(c1["Entities"]), 1)
            ids0 = {str(e["id"]) for e in c0["Entities"]}
            self.assertEqual(ids0, {"minecraft:villager", "minecraft:cow"})


class TestConteggio(unittest.TestCase):

    def test_conta_per_specie(self):
        animali = [FA.Animale(specie="mucca", x=0, y=0, z=0),
                  FA.Animale(specie="mucca", x=1, y=0, z=0),
                  FA.Animale(specie="lupo", x=2, y=0, z=0)]
        self.assertEqual(FA.conteggio(animali), {"mucca": 2, "lupo": 1})


if __name__ == "__main__":
    unittest.main()
