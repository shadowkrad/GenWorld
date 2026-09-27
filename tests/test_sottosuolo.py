"""Test del sottosuolo.

Il censimento del mondo prima di questo modulo diceva: 62% pietra e niente
altro, dalla superficie alla bedrock. Poi, scritto il modulo, il censimento
continuava a dire zero ardesia - e la colpa non era del modulo ma di `mondo`,
che apriva il livello con i limiti di prima della 1.18 (y da 0 a 256) e
buttava via in silenzio tutto cio' che stava sotto y=0. Qui si misura il
modulo; il limite del livello ha il suo test in `test_mondo_valido`.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import sottosuolo as SS  # noqa: E402

Y0 = -64
ALTEZZA = 284


class FintaTavolozza:
    """Id finti: la posa non deve dipendere da un livello aperto."""

    def __init__(self):
        self.aria = 0
        self.pietra = 1
        self.ardesia = 2
        self.minerale = {}
        for i, m in enumerate(SS.MINERALI):
            self.minerale[(m.nome, False)] = 100 + i
            self.minerale[(m.nome, True)] = 200 + i
        self.roccia = {n: 300 + i for i, (n, _, _, _) in enumerate(SS.ROCCE)}
        self.scavabile = (self.pietra, self.ardesia)

    aggiungi_rocce = SS.Tavolozza.aggiungi_rocce


def colonna_piena(quota_terreno=100):
    """Un chunk di pietra piena fino a `quota_terreno`."""
    out = np.zeros((16, ALTEZZA, 16), np.uint32)
    h_c = np.full((16, 16), quota_terreno, np.int32)
    ys = np.arange(Y0, Y0 + ALTEZZA)[None, :, None]
    out[np.broadcast_to(ys < quota_terreno, out.shape)] = 1
    return out, h_c


class TestCaverne(unittest.TestCase):

    def setUp(self):
        self.h = np.full((160, 160), 90, np.int32)
        self.c = SS.scava_caverne(self.h, densita=1.0, seed=3)

    def test_ci_sono_cunicoli(self):
        self.assertGreater(self.c["x"].size, 500)

    def test_stanno_sottoterra_col_cappello(self):
        self.assertLessEqual(int(self.c["y"].max()), 90 - SS.CAPPELLO)

    def test_non_sfondano_la_bedrock(self):
        self.assertGreaterEqual(int(self.c["y"].min()), -60)

    def test_sono_cunicoli_non_gomitoli(self):
        """Un verme con inerzia percorre distanza; un cammino casuale puro
        gira su se' stesso. Si misura lo spostamento fra sfere consecutive."""
        d = np.hypot(np.diff(self.c["x"].astype(float)),
                     np.diff(self.c["z"].astype(float)))
        vicine = d[d < 5]          # escluse le cuciture fra un verme e l'altro
        self.assertGreater(float(vicine.mean()), 0.7)

    def test_l_indice_registra_tutti_i_chunk_toccati(self):
        idx = SS.indice_per_chunk(self.c)
        i = 0
        x, z, r = int(self.c["x"][i]), int(self.c["z"][i]), self.c["r"][i]
        m = int(np.ceil(r)) + 1
        for cz in range((z - m) // 16, (z + m) // 16 + 1):
            for cx in range((x - m) // 16, (x + m) // 16 + 1):
                self.assertIn(i, idx.get((cx, cz), []))


class TestArdesia(unittest.TestCase):

    def test_sotto_zero_e_ardesia_sopra_e_pietra(self):
        out, h_c = colonna_piena()
        tav = FintaTavolozza()
        SS.posa(out, tav, h_c, 0, 0, Y0, {"x": np.zeros(0, np.int32)}, (), seed=1)
        ys = np.arange(Y0, Y0 + ALTEZZA)
        fondo = out[:, ys < -20, :]
        cima = out[:, (ys > 20) & (ys < 80), :]
        self.assertGreater(float((fondo == tav.ardesia).mean()), 0.85)
        self.assertEqual(int((cima == tav.ardesia).sum()), 0)

    def test_la_fascia_e_mescolata(self):
        out, h_c = colonna_piena()
        tav = FintaTavolozza()
        SS.posa(out, tav, h_c, 0, 0, Y0, {"x": np.zeros(0, np.int32)}, (), seed=1)
        ys = np.arange(Y0, Y0 + ALTEZZA)
        mezzo = out[:, (ys >= -SS.FASCIA_ARDESIA) & (ys < 0), :]
        frazione = float((mezzo == tav.ardesia).mean())
        self.assertGreater(frazione, 0.05)
        self.assertLess(frazione, 0.95,
                        "il piano di taglio a y=0 si vede a occhio")


class TestMinerali(unittest.TestCase):

    def setUp(self):
        self.out, self.h_c = colonna_piena()
        self.tav = FintaTavolozza()
        SS.posa(self.out, self.tav, self.h_c, 0, 0, Y0,
                {"x": np.zeros(0, np.int32)}, (), seed=5)

    def test_ci_sono_tutti(self):
        """Su venti chunk, non su uno: il diamante ha 0,35 grumi attesi per
        chunk, quindi in un chunk solo due volte su tre non c'e'."""
        visti = set()
        for k in range(20):
            out, h_c = colonna_piena()
            SS.posa(out, self.tav, h_c, k * 16, 0, Y0,
                    {"x": np.zeros(0, np.int32)}, (), seed=5)
            for i, m in enumerate(SS.MINERALI):
                if (out == 100 + i).any() or (out == 200 + i).any():
                    visti.add(m.nome)
        mancano = [m.nome for m in SS.MINERALI if m.nome not in visti]
        self.assertEqual(mancano, [], f"minerali assenti: {mancano}")

    def test_stanno_nella_loro_fascia(self):
        ys = np.arange(Y0, Y0 + ALTEZZA)
        for m in SS.MINERALI:
            for profondo in (False, True):
                dove = self.out == self.tav.minerale[(m.nome, profondo)]
                if not dove.any():
                    continue
                quote = ys[np.nonzero(dove.any(axis=(0, 2)))[0]]
                self.assertGreaterEqual(int(quote.min()), m.y_min - 3, m.nome)
                self.assertLessEqual(int(quote.max()), m.y_max + 3, m.nome)

    def test_non_bucano_la_superficie(self):
        """Un filone che spunta nel prato si vede, e non e' quello che deve
        fare un filone."""
        ys = np.arange(Y0, Y0 + ALTEZZA)
        alto = self.out[:, ys >= 100 - 2, :]
        for i in range(len(SS.MINERALI)):
            self.assertEqual(int((alto == 100 + i).sum()), 0)

    def test_le_rocce_degli_strati_si_possono_sostituire(self):
        """Senza `aggiungi_rocce` una vena si interrompe esattamente dove
        passa un banco di andesite, che e' il contrario di quel che fa una
        vena vera."""
        tav = FintaTavolozza()
        self.assertNotIn(777, tav.scavabile)
        tav.aggiungi_rocce((777, 778))
        self.assertIn(777, tav.scavabile)
        self.assertIn(tav.pietra, tav.scavabile)


class TestScavo(unittest.TestCase):

    def test_la_caverna_apre_il_vuoto_ma_non_il_tetto(self):
        out, h_c = colonna_piena(quota_terreno=100)
        tav = FintaTavolozza()
        caverne = {"x": np.array([8], np.int32), "y": np.array([40], np.int32),
                   "z": np.array([8], np.int32), "r": np.array([3.0], np.float32)}
        SS.posa(out, tav, h_c, 0, 0, Y0, caverne, [0], seed=1)
        ys = np.arange(Y0, Y0 + ALTEZZA)
        self.assertGreater(int((out[:, (ys > 36) & (ys < 44), :] == tav.aria).sum()), 20)
        # e niente buchi sotto la superficie
        vicino_al_cielo = out[:, (ys >= 100 - SS.CAPPELLO) & (ys < 100), :]
        self.assertEqual(int((vicino_al_cielo == tav.aria).sum()), 0)

    def test_niente_caverne_sotto_il_mare(self):
        """Una caverna che sfonda il fondale allaga tutto senza che nessuno
        se ne accorga finche' non ci nuota dentro."""
        out, h_c = colonna_piena(quota_terreno=58)   # fondale sotto il mare
        tav = FintaTavolozza()
        caverne = {"x": np.array([8], np.int32), "y": np.array([56], np.int32),
                   "z": np.array([8], np.int32), "r": np.array([4.0], np.float32)}
        SS.posa(out, tav, h_c, 0, 0, Y0, caverne, [0], seed=1)
        ys = np.arange(Y0, Y0 + ALTEZZA)
        sotto_il_fondale = out[:, (ys >= 58 - SS.CAPPELLO) & (ys < 58), :]
        self.assertEqual(int((sotto_il_fondale == tav.aria).sum()), 0)

    def test_il_seme_regge_le_coordinate_negative(self):
        """Meta' del mondo ha coordinate negative, e lo XOR di due negativi e'
        negativo: `default_rng` di un numero negativo solleva un'eccezione e
        il mondo si fermava al primo chunk a ovest."""
        for x, z in ((-1, -1), (-4096, -4096), (-1, 5), (7, -9)):
            self.assertGreaterEqual(SS._seme(x, z, 19), 0)


if __name__ == "__main__":
    unittest.main()
