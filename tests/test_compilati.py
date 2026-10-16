"""Template compilati: la cache su disco dei modelli gia' letti."""
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import compilati as CO  # noqa: E402
from genworld import template as TM  # noqa: E402


def modello(nome="casa"):
    return TM.Modello(nome=nome, celle=np.zeros((2, 2, 2), np.int16),
                      tavolozza=[("stone", {})])


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.tpl = os.path.join(self.d.name, "tpl")
        os.makedirs(self.tpl)
        with open(os.path.join(self.tpl, "a.nbt"), "wb") as f:
            f.write(b"uno")
        self.env = mock.patch.dict(os.environ, {"GENWORLD_COMPILATI": os.path.join(self.d.name, "comp")})
        self.env.start()
        os.environ.pop("GENWORLD_SENZA_COMPILATI", None)
        self.chiamate = 0

    def tearDown(self):
        self.env.stop()
        self.d.cleanup()

    def produci(self):
        self.chiamate += 1
        return [modello()]

    def giro(self, genere="t", cartelle=None):
        return CO.con_cache(genere, cartelle or [self.tpl], (1, 21, 4), self.produci)


class TestCache(Base):
    def test_la_seconda_volta_non_rilegge(self):
        a = self.giro()
        b = self.giro()
        self.assertEqual(self.chiamate, 1)
        self.assertEqual(a[0].nome, b[0].nome)
        self.assertTrue(np.array_equal(a[0].celle, b[0].celle))

    def test_ogni_chiamata_riceve_oggetti_nuovi(self):
        a = self.giro()
        b = self.giro()
        a[0].stili = frozenset({"x"})
        a[0].celle[0, 0, 0] = 5
        c = self.giro()
        self.assertNotEqual(b[0].celle[0, 0, 0], 5)
        self.assertEqual(c[0].stili, frozenset())

    def test_un_file_cambiato_o_nuovo_o_tolto_invalida(self):
        self.giro()
        with open(os.path.join(self.tpl, "a.nbt"), "wb") as f:
            f.write(b"due-piu-lungo")
        self.giro()
        self.assertEqual(self.chiamate, 2)
        with open(os.path.join(self.tpl, "b.nbt"), "wb") as f:
            f.write(b"x")
        self.giro()
        self.assertEqual(self.chiamate, 3)
        os.remove(os.path.join(self.tpl, "b.nbt"))
        self.giro()
        self.assertEqual(self.chiamate, 4)

    def test_un_file_dentro_una_sottocartella_conta(self):
        sub = os.path.join(self.tpl, "mondo", "region")
        os.makedirs(sub)
        self.giro()
        with open(os.path.join(sub, "r.0.0.mca"), "wb") as f:
            f.write(b"1")
        self.giro()
        self.assertEqual(self.chiamate, 2)

    def test_la_versione_di_gioco_distingue(self):
        CO.con_cache("t", [self.tpl], (1, 21, 4), self.produci)
        CO.con_cache("t", [self.tpl], (26, 2, 0), self.produci)
        self.assertEqual(self.chiamate, 2)

    def test_cartelle_diverse_non_si_pestano(self):
        altra = os.path.join(self.d.name, "altra")
        os.makedirs(altra)
        self.giro(cartelle=[self.tpl])
        self.giro(cartelle=[altra])
        self.giro(cartelle=[self.tpl])
        self.giro(cartelle=[altra])
        self.assertEqual(self.chiamate, 2)

    def test_il_codice_dei_lettori_cambiato_invalida(self):
        self.giro()
        with mock.patch.object(CO, "_codice", return_value="altro"):
            self.giro()
        self.assertEqual(self.chiamate, 2)

    def test_file_rovinato_si_rifa(self):
        self.giro()
        for f in os.listdir(CO.cartella()):
            with open(os.path.join(CO.cartella(), f), "wb") as fh:
                fh.write(b"non e' un pickle")
        self.giro()
        self.assertEqual(self.chiamate, 2)
        self.giro()
        self.assertEqual(self.chiamate, 2)

    def test_disattivata_da_ambiente(self):
        with mock.patch.dict(os.environ, {"GENWORLD_SENZA_COMPILATI": "1"}):
            self.giro()
            self.giro()
        self.assertEqual(self.chiamate, 2)
        self.assertFalse(os.path.isdir(CO.cartella()))

    def test_una_versione_vecchia_viene_tolta(self):
        self.giro()
        with open(os.path.join(self.tpl, "a.nbt"), "wb") as f:
            f.write(b"diverso")
        self.giro()
        self.assertEqual(len([f for f in os.listdir(CO.cartella()) if f.endswith(".pkl")]), 1)

    def test_svuota(self):
        self.giro()
        self.assertEqual(CO.svuota(), 1)
        self.giro()
        self.assertEqual(self.chiamate, 2)

    def test_i_miglioramenti_si_applicano_una_volta_e_restano(self):
        def segna(m):
            m.meta = {**m.meta, "migliorato": True}
            return m
        with mock.patch.object(CO, "MIGLIORAMENTI", [segna]):
            a = self.giro()
        self.assertTrue(a[0].meta["migliorato"])
        b = self.giro()                                   # senza il miglioramento attivo
        self.assertTrue(b[0].meta["migliorato"])
        self.assertEqual(self.chiamate, 1)

    def test_risultato_a_tupla_di_liste(self):
        def due():
            self.chiamate += 1
            return [modello("m")], [modello("f")]
        with mock.patch.object(CO, "MIGLIORAMENTI", [lambda m: m]):
            a = CO.con_cache("p", [self.tpl], (1, 21, 4), due)
        b = CO.con_cache("p", [self.tpl], (1, 21, 4), due)
        self.assertEqual((a[0][0].nome, a[1][0].nome), (b[0][0].nome, b[1][0].nome))
        self.assertEqual(self.chiamate, 1)


class TestArenaria(unittest.TestCase):
    def modello(self, base="sandstone", variant="normal"):
        celle = np.zeros((10, 10, 10), np.int16)
        return TM.Modello(nome="casa", celle=celle,
                          tavolozza=[(base, {"variant": variant}), ("stone", {})])

    def test_varia_l_arenaria_normale(self):
        m = CO.varia_arenaria(self.modello())
        nomi = {(n, p.get("variant")) for n, p in m.tavolozza}
        self.assertTrue({("sandstone", "cut"), ("sandstone", "smooth"),
                         ("sandstone", "chiseled")} <= nomi)
        usati = {m.tavolozza[int(i)][1].get("variant") for i in np.unique(m.celle)}
        self.assertIn("normal", usati)                       # la maggior parte resta com'e'
        self.assertIn("cut", usati)
        self.assertGreater(int((m.celle == 0).sum()), 500)   # >50% invariato
        self.assertEqual(m.celle.shape, (10, 10, 10))

    def test_non_tocca_altro(self):
        m = self.modello("stone")
        self.assertIs(CO.varia_arenaria(m), m)
        m2 = self.modello("sandstone", "smooth")
        self.assertIs(CO.varia_arenaria(m2), m2)

    def test_deterministica_e_arenaria_rossa(self):
        a = CO.varia_arenaria(self.modello("red_sandstone"))
        b = CO.varia_arenaria(self.modello("red_sandstone"))
        self.assertTrue(np.array_equal(a.celle, b.celle))
        self.assertIn(("red_sandstone", "cut"), {(n, p.get("variant")) for n, p in a.tavolozza})

    def test_e_fra_i_miglioramenti_di_default(self):
        self.assertIn(CO.varia_arenaria, CO.MIGLIORAMENTI)


class TestIntegrazione(Base):
    def test_carica_cartelle_usa_la_cache_sui_template_veri(self):
        reale = os.path.join(CO.RADICE, "templates", "strutture")
        if not any(f.endswith(".nbt") for f in (os.listdir(reale) if os.path.isdir(reale) else [])):
            self.skipTest("nessun template locale")
        t0 = time.time()
        a = TM.carica_cartelle([reale])
        primo = time.time() - t0
        TM._CACHE_MODELLI.clear()
        t0 = time.time()
        b = TM.carica_cartelle([reale])
        secondo = time.time() - t0
        self.assertEqual([m.nome for m in a], [m.nome for m in b])
        self.assertTrue(all(np.array_equal(x.celle, y.celle) for x, y in zip(a, b)))
        self.assertLess(secondo, primo)


if __name__ == "__main__":
    unittest.main()
