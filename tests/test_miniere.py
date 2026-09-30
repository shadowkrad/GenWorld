"""Test delle miniere artificiali: pozzo, gallerie, binari, travi, filoni."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import miniere as MI  # noqa: E402
from genworld import sottosuolo as SS  # noqa: E402


class FintoScrittore:
    """Assegna un id progressivo a ogni (nome, proprieta') - vedi lo stesso
    aiuto in test_motore.py."""
    id_aria = 0

    def __init__(self):
        self.voci: dict = {}

    def blocco(self, nome, **prop):
        chiave = (nome, tuple(sorted(prop.items())))
        return self.voci.setdefault(chiave, len(self.voci) + 1)


class TestSicuro(unittest.TestCase):

    def test_un_tratto_troppo_vicino_alla_superficie_non_e_sicuro(self):
        h = np.full((40, 40), 70.0, np.float32)
        acqua = np.zeros((40, 40), bool)
        self.assertFalse(MI._sicuro(h, acqua, 5, 5, 20, 5, 68, cappello=4))

    def test_un_tratto_abbastanza_sotto_e_sicuro(self):
        h = np.full((40, 40), 70.0, np.float32)
        acqua = np.zeros((40, 40), bool)
        self.assertTrue(MI._sicuro(h, acqua, 5, 5, 20, 5, 40, cappello=4))

    def test_un_tratto_sott_acqua_non_e_sicuro(self):
        h = np.full((40, 40), 70.0, np.float32)
        acqua = np.zeros((40, 40), bool)
        acqua[10:15, 5] = True          # acqua[z, x]: attraversa x=5
        self.assertFalse(MI._sicuro(h, acqua, 5, 5, 5, 20, 40, cappello=4))

    def test_fuori_mappa_non_e_sicuro(self):
        h = np.full((40, 40), 70.0, np.float32)
        acqua = np.zeros((40, 40), bool)
        self.assertFalse(MI._sicuro(h, acqua, 5, 5, -10, 5, 40, cappello=4))


class TestPianifica(unittest.TestCase):

    def scena(self, n=200):
        h = np.full((n, n), 90.0, np.float32)
        mare = np.zeros((n, n), bool)
        mare[:6, :] = True
        return h, mare

    def test_ne_pianifica_almeno_una(self):
        h, mare = self.scena()
        mn = MI.pianifica(h, mare, densita=2.0, seed=3)
        self.assertGreater(len(mn), 0)

    def test_niente_con_densita_zero(self):
        h, mare = self.scena()
        mn = MI.pianifica(h, mare, densita=0.0, seed=3)
        self.assertEqual(mn, [])

    def test_piu_densita_piu_miniere(self):
        h, mare = self.scena()
        poche = MI.pianifica(h, mare, densita=0.5, seed=7)
        tante = MI.pianifica(h, mare, densita=3.0, seed=7)
        self.assertGreaterEqual(len(tante), len(poche))

    def test_il_fondo_rispetta_i_limiti(self):
        h, mare = self.scena()
        for mn in MI.pianifica(h, mare, densita=3.0, seed=11):
            self.assertLessEqual(mn.y_fondo, mn.y_superficie - 16)
            self.assertGreaterEqual(mn.y_fondo, -58)

    def test_ogni_tratto_e_dritto_e_sicuro(self):
        h, mare = self.scena()
        acqua = mare
        for mn in MI.pianifica(h, mare, acqua=acqua, densita=3.0, seed=13):
            for s in mn.segmenti:
                self.assertTrue(s.x0 == s.x1 or s.z0 == s.z1,
                               "un tratto deve correre su un solo asse")
                self.assertTrue(MI._sicuro(h, acqua, s.x0, s.z0, s.x1, s.z1,
                                          s.y, MI.CAPPELLO))

    def test_niente_ingresso_nell_acqua(self):
        h, mare = self.scena()
        for mn in MI.pianifica(h, mare, densita=3.0, seed=17):
            self.assertFalse(mare[mn.z, mn.x])

    def test_evita_tiene_fuori_gli_ingressi(self):
        h, mare = self.scena()
        evita = np.ones(h.shape, bool)
        mn = MI.pianifica(h, mare, evita=evita, densita=3.0, seed=19)
        self.assertEqual(mn, [])

    def test_le_svolte_sono_ad_angolo_retto(self):
        """Ogni cambio di direzione dentro una galleria e' di 90 gradi, mai
        dritto e mai a marcia indietro."""
        h, mare = self.scena()
        for mn in MI.pianifica(h, mare, densita=3.0, seed=23):
            for s in mn.segmenti:
                if s.forma_fine is None:
                    continue
                self.assertIn(s.forma_fine, ("north_east", "north_west",
                                             "south_east", "south_west"))

    def test_ogni_giacimento_ha_un_minerale_valido_alla_sua_quota(self):
        """Il criterio dichiarato dall'utente: il minerale di un giacimento
        non e' scelto a caso, deve coprire davvero la quota del ramo."""
        h, mare = self.scena()
        n = 0
        for mn in MI.pianifica(h, mare, densita=3.0, seed=29):
            for g in mn.giacimenti:
                self.assertEqual(g.y, mn.y_fondo)
                m = SS.MINERALI[[m.nome for m in SS.MINERALI].index(g.minerale)]
                self.assertLessEqual(m.y_min, g.y)
                self.assertLessEqual(g.y, m.y_max)
                n += 1
        self.assertGreater(n, 0, "nessun giacimento pianificato: il test non prova nulla")

    def test_ogni_braccio_ha_almeno_un_giacimento_in_fondo(self):
        """Ogni miniera con almeno una galleria deve avere almeno un
        giacimento: un tunnel non finisce piu' nel nulla."""
        h, mare = self.scena()
        miniere = MI.pianifica(h, mare, densita=3.0, seed=31)
        self.assertGreater(len(miniere), 0)
        for mn in miniere:
            self.assertGreater(len(mn.giacimenti), 0,
                               "una miniera con gallerie ma senza giacimenti")

    def test_le_gallerie_si_diramano_davvero(self):
        """La richiesta era una SERIE di tunnel diramati, non solo i bracci
        dritti dal pozzo: su abbastanza semi, almeno qualche miniera deve
        avere piu' giacimenti che bracci diretti dal pozzo - la prova che
        una diramazione secondaria si e' davvero staccata a meta' di un
        braccio, non solo alla sua fine."""
        h, mare = self.scena()
        vista_una_diramazione = False
        for seed in range(40):
            for mn in MI.pianifica(h, mare, densita=4.0, seed=seed):
                # numero di bracci dal pozzo: quanti segmenti iniziano
                # esattamente sul pozzo (x0, z0) == (mn.x, mn.z)
                bracci = sum(1 for s in mn.segmenti
                            if (s.x0, s.z0) == (mn.x, mn.z))
                if len(mn.giacimenti) > bracci:
                    vista_una_diramazione = True
                    break
            if vista_una_diramazione:
                break
        self.assertTrue(vista_una_diramazione,
                        "mai una diramazione secondaria in 40 semi")


class TestIndicePerChunk(unittest.TestCase):

    def test_il_pozzo_e_nel_suo_chunk(self):
        mn = MI.Miniera(x=20, z=20, y_superficie=90, y_fondo=50, segmenti=[])
        idx = MI.indice_per_chunk([mn])
        self.assertIn(0, idx.get((1, 1), []))

    def test_una_galleria_lunga_tocca_piu_chunk(self):
        seg = MI.Segmento(x0=5, z0=5, x1=45, z1=5, y=50)
        mn = MI.Miniera(x=5, z=5, y_superficie=90, y_fondo=50, segmenti=[seg])
        idx = MI.indice_per_chunk([mn])
        # da x=5 a x=45 attraversa i chunk 0, 1 e 2
        for cx in (0, 1, 2):
            self.assertIn(0, idx.get((cx, 0), []), f"manca il chunk ({cx}, 0)")

    def test_un_giacimento_si_registra_anche_nel_chunk_vicino(self):
        """Stessa idea della capanna d'ingresso: un giacimento vicino al
        bordo di un chunk deve comparire registrato anche in quello
        accanto, altrimenti la saletta che vi sporge non verrebbe mai
        disegnata li'."""
        g = MI.Giacimento(x=17, z=20, y=50, minerale="iron_ore")
        mn = MI.Miniera(x=5, z=5, y_superficie=90, y_fondo=50, segmenti=[],
                        giacimenti=[g])
        idx = MI.indice_per_chunk([mn])
        self.assertIn(0, idx.get((1, 1), []))
        self.assertIn(0, idx.get((0, 1), []), "manca il chunk (0, 1)")


class TestCarrelli(unittest.TestCase):

    def test_un_carrello_sta_sul_binario(self):
        seg = MI.Segmento(x0=0, z0=0, x1=20, z1=0, y=50)
        mn = MI.Miniera(x=0, z=0, y_superficie=90, y_fondo=50, segmenti=[seg])
        # con probabilita' alta forzata dal seme, deve comparirne almeno uno
        trovato = any(MI.carrelli([mn], seed=s) for s in range(30))
        self.assertTrue(trovato)

    def test_niente_carrelli_senza_miniere(self):
        self.assertEqual(MI.carrelli([], seed=1), [])


class TestPosa(unittest.TestCase):

    Y0 = -64
    ALTEZZA = 284
    # un id "pietra piena" che non puo' scontrarsi con nessun id assegnato
    # dinamicamente da FintoScrittore ai blocchi di miniere.py (che parte
    # da 1 e non arriva mai a queste cifre).
    SOLIDO = 9999

    def colonna_piena(self, quota_terreno=100):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        h_c = np.full((16, 16), quota_terreno, np.int32)
        ys = np.arange(self.Y0, self.Y0 + self.ALTEZZA)[None, :, None]
        out[np.broadcast_to(ys < quota_terreno, out.shape)] = self.SOLIDO
        return out, h_c

    def test_il_pozzo_scava_dalla_superficie_al_fondo(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn = MI.Miniera(x=8, z=8, y_superficie=90, y_fondo=50, segmenti=[])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        ys = np.arange(self.Y0, self.Y0 + self.ALTEZZA)
        colonna = out[8, (ys >= 51) & (ys < 89), 8]
        # il pozzo e' scavato quasi ovunque: per lo piu' scalette (non
        # pietra piena), con qualche gradino d'aria pura ogni tanto.
        self.assertGreater(int((colonna != self.SOLIDO).sum()), 30)

    def test_la_galleria_ha_un_binario_al_centro(self):
        out, h_c = self.colonna_piena(quota_terreno=100)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        seg = MI.Segmento(x0=0, z0=8, x1=15, z1=8, y=50)
        mn = MI.Miniera(x=0, z=8, y_superficie=100, y_fondo=50, segmenti=[seg])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        y_binario = 50 + 1 - self.Y0
        binario_atteso = tav.rotaia["east_west"]
        presenti = out[1:15, y_binario, 8]
        self.assertTrue((presenti == binario_atteso).any())

    def test_la_galleria_e_larga_tre_celle(self):
        out, h_c = self.colonna_piena(quota_terreno=100)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        seg = MI.Segmento(x0=0, z0=8, x1=15, z1=8, y=50)
        mn = MI.Miniera(x=0, z=8, y_superficie=100, y_fondo=50, segmenti=[seg])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        y_a1 = 50 + 1 - self.Y0
        y_a2 = 50 + 2 - self.Y0
        # sul filo (a1), il centro (z=8) ha il binario, non aria pura -
        # ma ai lati (z=7, z=9) e sopra la testa (a2) e' tutto scavato
        self.assertNotEqual(int(out[8, y_a1, 8]), s.id_aria)
        self.assertTrue((out[8, y_a1, (7, 9)] == s.id_aria).all())
        self.assertTrue((out[8, y_a2, 7:10] == s.id_aria).all())
        # ma z=0 e' fuori dalla galleria e fuori dalla portata dei
        # filoni casuali lungo il tratto, resta pieno
        self.assertEqual(int(out[8, y_a1, 0]), self.SOLIDO)

    def test_ci_sono_travi_di_sostegno(self):
        out, h_c = self.colonna_piena(quota_terreno=100)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        seg = MI.Segmento(x0=0, z0=8, x1=15, z1=8, y=50)
        mn = MI.Miniera(x=0, z=8, y_superficie=100, y_fondo=50, segmenti=[seg])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        self.assertTrue((out == tav.palo).any())

    def test_niente_si_scava_fuori_dalla_galleria(self):
        """Lontano dal percorso, il chunk resta come prima - un test a
        difesa contro un bug che scava dappertutto invece che sul tratto."""
        out, h_c = self.colonna_piena(quota_terreno=100)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        seg = MI.Segmento(x0=0, z0=0, x1=3, z1=0, y=50)
        mn = MI.Miniera(x=0, z=0, y_superficie=100, y_fondo=50, segmenti=[seg])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        y_aria = 50 + 1 - self.Y0
        self.assertEqual(int(out[10, y_aria, 10]), self.SOLIDO)

    def test_protetto_c_blocca_la_galleria_sotto_una_struttura(self):
        """Una colonna "protetta" (mura, strade, campi, case: vedi
        `protetto` in `motore.pianifica`) non deve mai diventare aria,
        anche se il tratto ci passa proprio in mezzo - altrimenti la
        struttura disegnata sopra da `blocchi_chunk` resta appesa sul
        vuoto scavato dalla galleria."""
        out, h_c = self.colonna_piena(quota_terreno=100)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        seg = MI.Segmento(x0=0, z0=8, x1=15, z1=8, y=50)
        mn = MI.Miniera(x=0, z=8, y_superficie=100, y_fondo=50, segmenti=[seg])
        protetto_c = np.zeros((16, 16), bool)
        protetto_c[8, 8] = True
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,),
               seed=1, protetto_c=protetto_c)
        # sotto il livello del terreno la colonna protetta deve restare
        # tutta pietra piena - sopra (y >= 100) era gia' aria in partenza
        # (`colonna_piena` non riempie oltre `quota_terreno`), quindi il
        # confronto si limita al sottosuolo.
        colonna = out[8, :100 - self.Y0, 8]
        self.assertTrue((colonna == self.SOLIDO).all())
        # una colonna vicina, non protetta, resta invece scavata come prima
        y_a1 = 50 + 1 - self.Y0
        self.assertNotEqual(int(out[9, y_a1, 8]), self.SOLIDO)

    def test_protetto_c_blocca_il_pozzo(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn = MI.Miniera(x=8, z=8, y_superficie=90, y_fondo=50, segmenti=[])
        protetto_c = np.zeros((16, 16), bool)
        protetto_c[8, 8] = True
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,),
               seed=1, protetto_c=protetto_c)
        self.assertTrue((out[8, :90 - self.Y0, 8] == self.SOLIDO).all())


class TestIngresso(unittest.TestCase):
    """La capanna d'ingresso: pareti chiuse, una porta, una finestra, un
    tetto pieno e due lanterne - non piu' un castelletto aperto ("un gazebo
    con un buco nel mezzo", segnalato dall'utente in gioco)."""

    Y0 = -64
    ALTEZZA = 284
    SOLIDO = 9999

    def colonna_piena(self, quota_terreno=100):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        h_c = np.full((16, 16), quota_terreno, np.int32)
        ys = np.arange(self.Y0, self.Y0 + self.ALTEZZA)[None, :, None]
        out[np.broadcast_to(ys < quota_terreno, out.shape)] = self.SOLIDO
        return out, h_c

    def _mina_e_lati(self, x=8, z=8, quota_terreno=90):
        """Una miniera di prova piu' il verso della sua porta, la cella
        della porta e quella della finestra (sul lato opposto) - calcolati
        con la stessa funzione usata dal codice, cosi' il test resta valido
        qualunque lato scelga il verso deterministico."""
        mn = MI.Miniera(x=x, z=z, y_superficie=quota_terreno, y_fondo=50,
                        segmenti=[])
        verso = MI._verso_ingresso(mn)
        dxp, dzp = MI._DELTA_LATO[verso]
        r = MI.RAGGIO_INGRESSO
        porta = (x + dxp * r, z + dzp * r)
        finestra = (x - dxp * r, z - dzp * r)
        return mn, porta, finestra

    def test_quattro_piloni_ai_quattro_angoli(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn, _, _ = self._mina_e_lati()
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        base = 90 - self.Y0
        r = MI.RAGGIO_INGRESSO
        for dx, dz in ((-r, -r), (r, -r), (-r, r), (r, r)):
            colonna = out[8 + dx, base:base + MI.ALTEZZA_INGRESSO, 8 + dz]
            self.assertTrue((colonna == tav.palo).all(),
                            f"pilone mancante o incompleto in ({dx}, {dz})")

    def test_la_porta_e_un_vano_aperto_su_pavimento(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn, (px, pz), _ = self._mina_e_lati()
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        base = 90 - self.Y0
        self.assertEqual(int(out[px, base, pz]), tav.pavimento)
        self.assertEqual(int(out[px, base + 1, pz]), tav.aria)
        # base + 2, sulla soglia, e' dove pende la lanterna della porta
        # (vedi test_due_lanterne...) - comunque non un muro ne' un pilone.
        self.assertNotIn(int(out[px, base + 2, pz]), (tav.parete, tav.base_muro, tav.palo))

    def test_la_finestra_e_sul_lato_opposto_alla_porta(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn, _, (fx, fz) = self._mina_e_lati()
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        base = 90 - self.Y0
        self.assertEqual(int(out[fx, base, fz]), tav.base_muro)
        self.assertEqual(int(out[fx, base + 1, fz]), tav.vetro)
        self.assertEqual(int(out[fx, base + 2, fz]), tav.parete)

    def test_i_lati_senza_porta_ne_finestra_sono_pareti_piene(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn, (px, pz), (fx, fz) = self._mina_e_lati()
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        base = 90 - self.Y0
        r = MI.RAGGIO_INGRESSO
        # il lato perpendicolare a quello porta/finestra: ruota di 90 gradi
        dxp, dzp = (px - mn.x) // r, (pz - mn.z) // r
        pdx, pdz = -dzp, dxp
        wx, wz = mn.x + pdx * r, mn.z + pdz * r
        self.assertEqual(int(out[wx, base, wz]), tav.base_muro)
        self.assertEqual(int(out[wx, base + 1, wz]), tav.parete)
        self.assertEqual(int(out[wx, base + 2, wz]), tav.parete)

    def test_l_interno_e_vuoto_sopra_il_pavimento(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn, _, _ = self._mina_e_lati()
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        base = 90 - self.Y0
        self.assertEqual(int(out[9, base, 8]), tav.pavimento)
        self.assertEqual(int(out[9, base + 1, 8]), tav.aria)
        self.assertEqual(int(out[9, base + 2, 8]), tav.aria)

    def test_tetto_pieno_su_tutta_la_capanna(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn, (px, pz), _ = self._mina_e_lati()
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        cima = 90 - self.Y0 + MI.ALTEZZA_INGRESSO
        r = MI.RAGGIO_INGRESSO
        for dx, dz in ((0, 0), (r, r), (-r, -r), (px - 8, pz - 8)):
            self.assertEqual(int(out[8 + dx, cima, 8 + dz]), tav.tetto)

    def test_due_lanterne_una_sul_pozzo_una_sulla_porta(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn, (px, pz), _ = self._mina_e_lati()
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        cima = 90 - self.Y0 + MI.ALTEZZA_INGRESSO
        self.assertEqual(int(out[8, cima - 1, 8]), tav.lanterna)
        self.assertEqual(int(out[px, cima - 1, pz]), tav.lanterna)

    def test_protetto_c_blocca_la_capanna(self):
        """Stessa idea di `test_protetto_c_blocca_il_pozzo`: un angolo della
        capanna che cade su una colonna gia' occupata in superficie (mura,
        strade, case) non deve scriverci sopra."""
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn, _, _ = self._mina_e_lati()
        r = MI.RAGGIO_INGRESSO
        protetto_c = np.zeros((16, 16), bool)
        protetto_c[8 + r, 8 + r] = True          # uno dei quattro angoli
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,),
               seed=1, protetto_c=protetto_c)
        base = 90 - self.Y0
        colonna = out[8 + r, base:base + MI.ALTEZZA_INGRESSO, 8 + r]
        self.assertTrue((colonna != tav.palo).all())

    def test_la_capanna_si_registra_anche_nel_chunk_vicino(self):
        """Un pozzo a un blocco dal bordo del chunk deve comparire
        registrato anche nel chunk accanto, altrimenti la capanna che vi
        sporge non verrebbe mai disegnata li' - lo stesso difetto
        "smezzato" delle case da template senza margine nell'indice."""
        mn = MI.Miniera(x=15, z=8, y_superficie=90, y_fondo=50, segmenti=[])
        idx = MI.indice_per_chunk([mn])
        self.assertIn(0, idx.get((0, 0), []))
        self.assertIn(0, idx.get((1, 0), []), "manca il chunk (1, 0)")

    def test_la_capanna_attraversa_davvero_il_confine_del_chunk(self):
        """Non solo l'indice: il pilone che sporge nel chunk vicino deve
        arrivare completo anche disegnando quel secondo chunk da solo."""
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        mn = MI.Miniera(x=15, z=8, y_superficie=90, y_fondo=50, segmenti=[])
        out1, h_c = self.colonna_piena(quota_terreno=90)
        MI.posa(out1, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        out2, h_c = self.colonna_piena(quota_terreno=90)
        MI.posa(out2, tav, h_c, 16, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        base = 90 - self.Y0
        # il pilone a (17, 10) (mn.x+2, mn.z+2) sta nel SECONDO chunk
        # (locale (1, 10)), non nel primo
        self.assertTrue((out1[15, base:base + MI.ALTEZZA_INGRESSO, 10]
                         != tav.palo).all())
        self.assertTrue((out2[1, base:base + MI.ALTEZZA_INGRESSO, 10]
                         == tav.palo).all())


class TestGiacimenti(unittest.TestCase):
    """La saletta a fine diramazione: vuota al centro, con il minerale
    scelto incastonato nelle pareti appena scavate - il punto d'arrivo di
    un ramo di galleria, non un filoncino nascosto lungo il percorso."""

    Y0 = -64
    ALTEZZA = 284
    SOLIDO = 9999

    def colonna_piena(self, quota_terreno=100):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        h_c = np.full((16, 16), quota_terreno, np.int32)
        ys = np.arange(self.Y0, self.Y0 + self.ALTEZZA)[None, :, None]
        out[np.broadcast_to(ys < quota_terreno, out.shape)] = self.SOLIDO
        return out, h_c

    def test_il_centro_e_vuoto(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        g = MI.Giacimento(x=8, z=8, y=40, minerale="iron_ore")
        MI._carica_giacimento(out, tav, h_c, np.random.default_rng(1), g,
                              0, 0, self.Y0, (self.SOLIDO,))
        gy = g.y - self.Y0
        self.assertEqual(int(out[8, gy, 8]), tav.aria)

    def test_il_minerale_e_incastonato_nelle_pareti(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        g = MI.Giacimento(x=8, z=8, y=40, minerale="iron_ore")
        MI._carica_giacimento(out, tav, h_c, np.random.default_rng(1), g,
                              0, 0, self.Y0, (self.SOLIDO,))
        minerale = tav.minerale[("iron_ore", False)]
        self.assertGreater(int((out == minerale).sum()), 0,
                           "nessun blocco di minerale piazzato")
        # niente minerale al centro della saletta: e' vuoto, non incastonato
        gy = g.y - self.Y0
        self.assertNotEqual(int(out[8, gy, 8]), minerale)

    def test_ardesia_sotto_la_fascia(self):
        """Sotto la fascia dell'ardesia il minerale deve essere la sua
        variante deepslate, stessa regola di sottosuolo.posa()."""
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        g = MI.Giacimento(x=8, z=8, y=-40, minerale="diamond_ore")
        MI._carica_giacimento(out, tav, h_c, np.random.default_rng(1), g,
                              0, 0, self.Y0, (self.SOLIDO,))
        superficie = tav.minerale[("diamond_ore", False)]
        ardesia = tav.minerale[("diamond_ore", True)]
        self.assertEqual(int((out == superficie).sum()), 0)
        self.assertGreater(int((out == ardesia).sum()), 0)

    def test_protetto_c_blocca_la_saletta(self):
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        g = MI.Giacimento(x=8, z=8, y=40, minerale="iron_ore")
        protetto_c = np.ones((16, 16), bool)
        MI._carica_giacimento(out, tav, h_c, np.random.default_rng(1), g,
                              0, 0, self.Y0, (self.SOLIDO,),
                              protetto_c=protetto_c)
        gy = g.y - self.Y0
        self.assertEqual(int(out[8, gy, 8]), self.SOLIDO,
                         "protetto_c avrebbe dovuto impedire lo scavo")

    def test_minerale_sconosciuto_non_esplode(self):
        """Difesa: un nome di minerale che non esiste in MINERALI (non
        dovrebbe capitare, ma _carica_giacimento non deve fallire)."""
        out, h_c = self.colonna_piena(quota_terreno=90)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        g = MI.Giacimento(x=8, z=8, y=40, minerale="nome_a_caso")
        MI._carica_giacimento(out, tav, h_c, np.random.default_rng(1), g,
                              0, 0, self.Y0, (self.SOLIDO,))   # non deve sollevare

    def test_un_giacimento_a_cavallo_di_due_chunk_arriva_completo(self):
        """Stessa idea della capanna d'ingresso e del filone: un giacimento
        vicino al bordo di un chunk deve arrivare completo anche disegnando
        il secondo chunk da solo."""
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        g = MI.Giacimento(x=15, z=8, y=40, minerale="iron_ore")
        out1, h_c = self.colonna_piena(quota_terreno=90)
        MI._carica_giacimento(out1, tav, h_c, np.random.default_rng(1), g,
                              0, 0, self.Y0, (self.SOLIDO,))
        out2, h_c = self.colonna_piena(quota_terreno=90)
        MI._carica_giacimento(out2, tav, h_c, np.random.default_rng(1), g,
                              16, 0, self.Y0, (self.SOLIDO,))
        gy = g.y - self.Y0
        minerale = tav.minerale[("iron_ore", False)]
        # nel primo chunk (locale x=15..) e nel secondo (locale x=0..) deve
        # comparire il minerale da qualche parte lungo la sezione centrale
        trovato_1 = bool((out1[13:16, gy, 6:11] == minerale).any())
        trovato_2 = bool((out2[0:3, gy, 6:11] == minerale).any())
        self.assertTrue(trovato_1 and trovato_2,
                        "il giacimento non attraversa il confine di chunk")


if __name__ == "__main__":
    unittest.main()
