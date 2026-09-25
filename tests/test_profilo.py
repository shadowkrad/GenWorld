"""Verifica del profilo di lettura: serializzazione e maschera dipinta."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import mappa as M
from genworld.profilo import (Profilo, _decodifica_rle, codifica_rle,
                              maschera_decoro)
from genworld.classi_guidate import DECORO


class TestSerializzazione(unittest.TestCase):

    def profilo(self):
        return Profilo(
            nome="prova", roi=(0.1, 0.2, 0.9, 0.8),
            campioni={M.MARE: [(0.5, 0.5)], DECORO: [(0.02, 0.02)]},
            pulizia=7, soglia_montagna=0.4,
        )

    def test_andata_e_ritorno(self):
        p = self.profilo()
        q = Profilo.da_dizionario(json.loads(json.dumps(p.a_dizionario())))
        self.assertEqual(q.nome, p.nome)
        self.assertEqual(q.roi, p.roi)
        self.assertEqual(q.campioni, p.campioni)
        self.assertEqual(q.pulizia, p.pulizia)
        self.assertAlmostEqual(q.soglia_montagna, p.soglia_montagna)

    def test_salva_e_carica(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "p.json")
            self.profilo().salva(f)
            self.assertEqual(Profilo.carica(f).n_campioni(), 2)

    def test_classe_sconosciuta_rifiutata(self):
        with self.assertRaises(ValueError):
            Profilo.da_dizionario({"campioni": {"vulcano": [[0.5, 0.5]]}})

    def test_campioni_fuori_dal_rettangolo_scartati(self):
        """Un campione preso sulla cornice non deve rientrare dal ritaglio."""
        p = Profilo(roi=(0.25, 0.25, 0.75, 0.75),
                    campioni={M.MARE: [(0.5, 0.5), (0.05, 0.05)]})
        dentro = p._campioni_nel_ritaglio()[M.MARE]
        self.assertEqual(len(dentro), 1)
        self.assertAlmostEqual(dentro[0][0], 0.5)   # ricentrato nel ritaglio


class TestMaschera(unittest.TestCase):

    def maschera(self):
        m = np.zeros((64, 64), np.uint8)
        m[8:30, 6:40] = 1
        m[40:60, 20:55] = 2
        return m

    def test_rle_identico(self):
        m = self.maschera()
        self.assertTrue(np.array_equal(_decodifica_rle(codifica_rle(m)), m))

    def test_rle_compatto(self):
        """Una maschera a zone larghe deve comprimere parecchio."""
        d = codifica_rle(self.maschera())
        self.assertLess(len(d["rle"]), 64 * 64 / 4)

    def test_rle_tratti_lunghi(self):
        """Oltre 65535 celle uguali il formato spezza la corsa, senza perdere."""
        m = np.zeros((400, 400), np.uint8)
        self.assertTrue(np.array_equal(_decodifica_rle(codifica_rle(m)), m))

    def test_pennello_ha_ultima_parola(self):
        cls = np.full((64, 64), M.PIANURA, dtype=np.uint8)
        cls[:32, :] = M.OCEANO
        p = Profilo(maschera=codifica_rle(self.maschera()))
        fuori = p.applica_maschera(cls)
        m = p.maschera_dipinta(cls.shape)
        self.assertTrue(np.isin(fuori[m == 1], list(M.ACQUA)).all())
        self.assertFalse(np.isin(fuori[m == 2], list(M.ACQUA)).any())

    def test_senza_maschera_non_cambia_nulla(self):
        cls = np.full((16, 16), M.PRATERIA, dtype=np.uint8)
        self.assertTrue(np.array_equal(Profilo().applica_maschera(cls), cls))

    def test_maschera_decoro(self):
        cls = np.full((8, 8), M.PIANURA, dtype=np.uint8)
        cls[0, 0] = DECORO
        self.assertEqual(int(maschera_decoro(cls).sum()), 1)


if __name__ == "__main__":
    unittest.main()
