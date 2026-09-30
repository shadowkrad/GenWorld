"""Verifica dei fiumi: separazione per forma, deflusso, livellamento."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import fiumi as F


class TestEstrazionePerForma(unittest.TestCase):

    def scena(self, n=80):
        """Un mare largo in basso e un canale sottile che lo raggiunge."""
        acqua = np.zeros((n, n), bool)
        acqua[n - 20:, :] = True          # mare
        acqua[10:n - 20, 38:41] = True    # canale largo 3
        return acqua

    def test_il_canale_e_fiume_il_mare_no(self):
        a = self.scena()
        f = F.estrai(a, larghezza_mare=5, lunghezza_minima=20)
        self.assertTrue(f[30, 39], "il canale non e' stato riconosciuto")
        self.assertFalse(f[75, 40], "il mare e' stato scambiato per fiume")

    def test_frammenti_scartati(self):
        a = np.zeros((60, 60), bool)
        a[10:12, 10:13] = True            # pozza di 6 celle
        self.assertFalse(F.estrai(a, lunghezza_minima=40).any())

    def test_senza_acqua(self):
        self.assertFalse(F.estrai(np.zeros((30, 30), bool)).any())


class TestDeflusso(unittest.TestCase):

    def conca(self, n=48):
        """Un piano inclinato con una buca in mezzo."""
        z, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        h = (100 - z * 0.8).astype(np.float32)
        h[20:26, 20:26] -= 15             # depressione chiusa
        return h

    def test_depressioni_riempite(self):
        h = self.conca()
        mare = np.zeros(h.shape, bool)
        mare[-1, :] = True
        q = F.riempi_depressioni(h, mare)
        self.assertGreaterEqual(float(q[22, 22]), float(h[22, 22]))
        # nessuna cella interna piu' bassa di TUTTI i suoi vicini
        interno = q[1:-1, 1:-1]
        piu_basso = np.ones_like(interno, bool)
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dx == 0:
                    continue
                piu_basso &= interno < q[1 + dz:q.shape[0] - 1 + dz,
                                         1 + dx:q.shape[1] - 1 + dx]
        self.assertEqual(int(piu_basso.sum()), 0, "restano conche chiuse")

    def test_accumulo_cresce_a_valle(self):
        h = self.conca()
        mare = np.zeros(h.shape, bool); mare[-1, :] = True
        acc = F.accumulo(h, mare)
        # su un piano inclinato l'accumulo deve crescere scendendo
        self.assertGreater(float(acc[40, 24]), float(acc[5, 24]))

    def test_accumulo_parte_da_uno(self):
        h = self.conca()
        mare = np.zeros(h.shape, bool); mare[-1, :] = True
        self.assertGreaterEqual(float(F.accumulo(h, mare).min()), 1.0)


class TestLivellamento(unittest.TestCase):

    def scena(self, n=60):
        z, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        h = (120 - z * 1.0).astype(np.float32)
        h[-12:, :] = 50                      # mare in fondo
        mare = np.zeros((n, n), bool); mare[-12:, :] = True
        fiumi = np.zeros((n, n), bool); fiumi[5:-12, 29:32] = True
        return h, mare, fiumi

    def test_il_fiume_scende_verso_la_foce(self):
        """L'invariante che rende il modulo utile: niente acqua in salita."""
        h, mare, f = self.scena()
        liv, _ = F.livella(f, h, mare, livello_mare=62)
        colonna = [float(liv[z, 30]) for z in range(5, h.shape[0] - 12)]
        for a, b in zip(colonna, colonna[1:]):
            self.assertGreaterEqual(a + 1e-4, b,
                                    "il pelo dell'acqua risale verso la foce")

    def test_la_foce_e_al_livello_del_mare(self):
        h, mare, f = self.scena()
        liv, _ = F.livella(f, h, mare, livello_mare=62)
        z_foce = h.shape[0] - 13
        self.assertAlmostEqual(float(liv[z_foce, 30]), 62.0, places=3)

    def test_fuori_dal_fiume_resta_il_livello_del_mare(self):
        h, mare, f = self.scena()
        liv, _ = F.livella(f, h, mare, livello_mare=62)
        self.assertAlmostEqual(float(liv[20, 5]), 62.0, places=3)

    def test_scavo_limitato(self):
        """Senza tetto allo scavo si aprono canyon da decine di blocchi."""
        h, mare, f = self.scena()
        h = h.copy(); h[30, 30] += 60          # un dosso sul corso
        _, h2 = F.livella(f, h, mare, livello_mare=62, scavo_massimo=8.0)
        self.assertGreaterEqual(float(h2[30, 30]), float(h[30, 30]) - 8.0 - 1e-4)

    def test_senza_fiumi_non_cambia_nulla(self):
        h, mare, _ = self.scena()
        vuoto = np.zeros(h.shape, bool)
        liv, h2 = F.livella(vuoto, h, mare, livello_mare=62)
        self.assertTrue(np.array_equal(h2, h.astype(np.float32)))
        self.assertTrue((liv == 62).all())


class TestDaTerreno(unittest.TestCase):

    def test_i_fiumi_non_invadono_il_mare(self):
        n = 64
        z, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        h = (120 - z).astype(np.float32)
        mare = np.zeros((n, n), bool); mare[-14:, :] = True
        f, acc = F.da_terreno(h, mare, soglia=40, lunghezza_minima=5)
        self.assertFalse((f & mare).any())

    def test_soglia_alta_meno_fiumi(self):
        n = 64
        z, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        h = (120 - z + 3 * np.sin(x / 3.0)).astype(np.float32)
        mare = np.zeros((n, n), bool); mare[-14:, :] = True
        bassa = F.da_terreno(h, mare, soglia=20, lunghezza_minima=3)[0].sum()
        alta = F.da_terreno(h, mare, soglia=200, lunghezza_minima=3)[0].sum()
        self.assertGreaterEqual(int(bassa), int(alta))

    def test_larghezza_base_allarga_il_fiume(self):
        """L'accumulo D8 e' un filo largo una cella per costruzione - senza
        allargamento un fiume qualunque, anche uno che porta molta acqua,
        resta largo un pixel. E' il "poco largo al cospetto della realta'"
        visto in gioco."""
        n = 64
        z, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        h = (120 - z + 3 * np.sin(x / 3.0)).astype(np.float32)
        mare = np.zeros((n, n), bool); mare[-14:, :] = True
        stretto = F.da_terreno(h, mare, soglia=40, lunghezza_minima=5,
                               larghezza_base=0)[0].sum()
        largo = F.da_terreno(h, mare, soglia=40, lunghezza_minima=5,
                             larghezza_base=1)[0].sum()
        self.assertGreater(int(largo), int(stretto))


if __name__ == "__main__":
    unittest.main()


class TestSpondeInQuota(unittest.TestCase):
    """Un fiume in quota non deve scavarsi una trincea fino al livello del mare.

    Questo test nasce da uno screenshot: in mezzo a una citta' di Arda si
    apriva un crepaccio vuoto con dentro, sospeso, un fiumiciattolo. La causa
    era in `livella`: lo scavo delle rive leggeva `livello` nella cella di
    riva, dove pero' vale il livello del mare, perche' quella cella non e'
    fiume. Un corso a quota 133 si vedeva scavare le sponde fino a 64.
    """

    def scenario(self):
        lato = 60
        h = np.full((lato, lato), 140, np.int32)
        h[:, 30] = 134                      # un solco in cima all'altopiano
        fiumi = np.zeros((lato, lato), bool)
        fiumi[:, 30] = True
        mare = np.zeros((lato, lato), bool)
        mare[:2, :] = True                  # una foce fittizia in alto
        return h, fiumi, mare

    def test_le_rive_non_scendono_al_livello_del_mare(self):
        h, fiumi, mare = self.scenario()
        livello, nuovo = F.livella(fiumi, h, mare, livello_mare=62)
        riva = nuovo[5:, 28]
        self.assertGreater(int(riva.min()), 120,
                           f"sponda scavata a {int(riva.min())} invece che in quota")

    def test_il_letto_non_si_alza_sopra_il_terreno(self):
        """Un alveo sopra il terreno intorno e' un argine; con l'acqua in
        cima e' un acquedotto."""
        h, fiumi, mare = self.scenario()
        _, nuovo = F.livella(fiumi, h, mare, livello_mare=62)
        self.assertTrue((nuovo[fiumi] <= h[fiumi]).all())

    def test_puntella_alza_solo_le_sponde_basse(self):
        h = np.full((40, 40), 100, np.int32)
        fiumi = np.zeros((40, 40), bool)
        fiumi[:, 20] = True
        livello = np.full((40, 40), 62.0, np.float32)
        livello[:, 20] = 99.0
        h[:, 18] = 80                        # una sponda sfondata
        nuovo = F.puntella(h, fiumi, livello, livello_mare=62)
        self.assertEqual(int(nuovo[10, 18]), 100)   # alzata a contenere
        self.assertEqual(int(nuovo[10, 30]), 100)   # lontano non si tocca
        self.assertEqual(int(nuovo[10, 20]), 100)   # il letto non si tocca


class TestFiumeALivelloDelTerreno(unittest.TestCase):
    """Nato da uno screenshot dopo il fix delle sponde di terra: il fiume
    aveva ancora l'aspetto di un fossato incassato, con le rive a un'unica
    quota piatta e uno scalino netto verso il terreno vero. Qui si verifica
    che il pelo resti vicino alla quota circostante e che la riva digradi
    invece di essere un terrazzo a un solo livello."""

    def scena(self, n=60):
        h = np.full((n, n), 100.0, np.float32)
        fiumi = np.zeros((n, n), bool)
        fiumi[:, 30] = True
        mare = np.zeros((n, n), bool)
        mare[:2, :] = True
        return h, fiumi, mare

    def test_il_pelo_resta_vicino_alla_quota_circostante(self):
        """Con l'incassamento ridotto il pelo non deve stare a due o piu'
        blocchi sotto il terreno intorno, come prima di questo fix."""
        h, fiumi, mare = self.scena()
        livello, _ = F.livella(fiumi, h, mare, livello_mare=62)
        centro = float(livello[30, 30])
        self.assertGreater(centro, float(h[30, 30]) - 1.5,
                           "il pelo dell'acqua e' ancora incassato di piu' "
                           "di un blocco e mezzo sotto il terreno intorno")

    def test_la_riva_digrada_invece_di_essere_un_terrazzo(self):
        """Le celle della fascia di riva a distanze diverse dal fiume non
        devono finire tutte alla stessa quota (il "fossato dai bordi
        verticali" dello screenshot): devono crescere via via che ci si
        allontana dal fiume."""
        h, fiumi, mare = self.scena()
        _, nuovo = F.livella(fiumi, h, mare, livello_mare=62, scavo_rive=3)
        riga = nuovo[30, 27:31]              # dal bordo della fascia al fiume
        # non tutte uguali: una vera china, non un gradino a un solo livello
        self.assertGreater(len(set(np.round(riga, 2).tolist())), 1)
        vicino_al_fiume, lontano = float(nuovo[30, 29]), float(nuovo[30, 27])
        self.assertLessEqual(vicino_al_fiume, lontano + 1e-6,
                             "la sponda vicino all'acqua deve stare piu' "
                             "bassa (o uguale) di quella lontana, non il "
                             "contrario")
