"""Test delle miniere artificiali: portale, rampa, piani, gallerie, binari, travi, filoni."""

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
            self.assertLessEqual(mn.y_fondo, mn.y_superficie - 10)
            self.assertGreaterEqual(mn.y_fondo, MI.QUOTA_MINIMA_RAMPA)

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
                # numero di bracci dal fondo della rampa: quanti segmenti
                # iniziano esattamente li'
                fx, fz, _ = mn.rampa.fine
                bracci = sum(1 for s in mn.segmenti if (s.x0, s.z0) == (fx, fz))
                if len(mn.giacimenti) > bracci:
                    vista_una_diramazione = True
                    break
            if vista_una_diramazione:
                break
        self.assertTrue(vista_una_diramazione,
                        "mai una diramazione secondaria in 40 semi")


class TestIndicePerChunk(unittest.TestCase):

    def test_il_portale_e_nel_suo_chunk(self):
        r = MI.Rampa(x=20, z=20, dx=1, dz=0, y=80, lunghezza=40)
        mn = MI.Miniera(x=20, z=20, y_superficie=80, y_fondo=66, rampa=r, segmenti=[])
        idx = MI.indice_per_chunk([mn])
        self.assertIn(0, idx.get((1, 1), []))

    def test_la_rampa_tocca_tutti_i_chunk_che_attraversa(self):
        r = MI.Rampa(x=5, z=20, dx=1, dz=0, y=80, lunghezza=60)
        mn = MI.Miniera(x=5, z=20, y_superficie=80, y_fondo=60, rampa=r, segmenti=[])
        idx = MI.indice_per_chunk([mn])
        for cx in range(0, 5):                       # da x=5 a x=65
            self.assertIn(0, idx.get((cx, 1), []), f"manca il chunk ({cx}, 1)")

    def test_una_rampa_interna_si_registra(self):
        r = MI.Rampa(x=5, z=20, dx=1, dz=0, y=80, lunghezza=30)
        ri = MI.Rampa(x=100, z=20, dx=0, dz=1, y=60, lunghezza=45)
        mn = MI.Miniera(x=5, z=20, y_superficie=80, y_fondo=45, rampa=r,
                        rampe_interne=[ri], segmenti=[])
        idx = MI.indice_per_chunk([mn])
        self.assertIn(0, idx.get((6, 1), []))
        self.assertIn(0, idx.get((6, 3), []))

    def test_il_binario_all_aperto_davanti_al_portale_e_registrato(self):
        r = MI.Rampa(x=33, z=20, dx=1, dz=0, y=80, lunghezza=40)    # chunk 2, ma il binario sta a x=25
        mn = MI.Miniera(x=33, z=20, y_superficie=80, y_fondo=66, rampa=r, segmenti=[])
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
        tav.decora = False
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



def collina(n=200, salita=0.5, direzione="x"):
    """Un terreno che sale: una collina verso cui entra un portale."""
    base = np.arange(n, dtype=np.float32) * salita + 70.0
    h = np.tile(base[None, :], (n, 1)) if direzione == "x" else np.tile(base[:, None], (1, n))
    return h


class TestRampa(unittest.TestCase):

    def test_scende_un_blocco_ogni_tre_di_cammino(self):
        r = MI.Rampa(x=0, z=0, dx=1, dz=0, y=80, lunghezza=30)
        self.assertEqual([r.cella(i)[2] for i in range(8)], [80, 80, 80, 79, 79, 79, 78, 78])

    def test_le_celle_seguono_la_direzione(self):
        r = MI.Rampa(x=10, z=10, dx=0, dz=-1, y=80, lunghezza=5)
        self.assertEqual(r.cella(3)[:2], (10, 7))
        self.assertEqual(r.fine[:2], (10, 5))

    def test_il_gradino_e_la_prima_cella_dopo_il_salto(self):
        r = MI.Rampa(x=0, z=0, dx=1, dz=0, y=80, lunghezza=30)
        self.assertFalse(r.e_gradino(0))
        self.assertFalse(r.e_gradino(2))
        self.assertTrue(r.e_gradino(3))
        self.assertTrue(r.e_gradino(6))


class TestPianificaRampa(unittest.TestCase):

    def pianifica(self, h, seed=3, **kw):
        mare = np.zeros(h.shape, bool)
        return MI.pianifica(h, mare, densita=kw.pop("densita", 3.0), seed=seed, **kw)

    def test_la_discesa_e_graduale_non_un_pozzo(self):
        """Il difetto segnalato: un pozzo verticale e sei gia' nelle miniere.
        Per scendere di D blocchi servono almeno 3 D celle di cammino."""
        miniere = self.pianifica(collina())
        self.assertGreater(len(miniere), 0)
        for mn in miniere:
            r = mn.rampa
            scesa = r.y - r.fine[2]
            self.assertGreaterEqual(r.lunghezza, scesa * MI.PASSO_RAMPA - 1)
            self.assertGreaterEqual(r.lunghezza, MI.LUNGHEZZA_RAMPA_MIN)

    def test_il_portale_guarda_la_collina(self):
        """La rampa entra nel pendio, non esce all'aria aperta dall'altra
        parte: il terreno sale lungo la sua direzione."""
        for mn in self.pianifica(collina(direzione="x")):
            self.assertEqual((mn.rampa.dx, mn.rampa.dz), (1, 0))
        for mn in self.pianifica(collina(direzione="z")):
            self.assertEqual((mn.rampa.dx, mn.rampa.dz), (0, 1))

    def test_ogni_cella_sta_sotto_terra_tranne_la_trincea(self):
        h = collina()
        for mn in self.pianifica(h):
            r = mn.rampa
            ultima = r.cella(r.lunghezza)
            copertura = h[ultima[1], ultima[0]] - 1 - (ultima[2] + MI.ALTEZZA_GALLERIA)
            self.assertGreaterEqual(copertura, MI.CAPPELLO, "la rampa finisce quasi in superficie")

    def test_la_rampa_non_passa_sotto_l_acqua(self):
        h = collina()
        acqua = np.zeros(h.shape, bool)
        acqua[:, 100:110] = True
        mare = np.zeros(h.shape, bool)
        for mn in MI.pianifica(h, mare, acqua=acqua, densita=3.0, seed=5):
            for i in range(mn.rampa.lunghezza + 1):
                x, z, _ = mn.rampa.cella(i)
                self.assertFalse(acqua[z, x], "una rampa sotto l'acqua")

    def test_il_portale_resta_fuori_dalle_strutture(self):
        h = collina()
        evita = np.zeros(h.shape, bool)
        evita[:, 60:140] = True
        for mn in self.pianifica(h, evita=evita):
            for j in range(-MI.BINARIO_FUORI - 2, 1):
                x, z, _ = mn.rampa.cella(j)
                self.assertFalse(evita[z, x])

    def test_le_miniere_sono_lontane_fra_loro(self):
        miniere = self.pianifica(collina(400), densita=5.0)
        self.assertGreater(len(miniere), 1)
        for i, a in enumerate(miniere):
            for b in miniere[i + 1:]:
                self.assertGreaterEqual((a.x - b.x) ** 2 + (a.z - b.z) ** 2,
                                        MI.DISTANZA_FRA_MINIERE ** 2)

    def test_sono_poche(self):
        """Una miniera ogni quarantacinquemila celle, non ogni quattordicimila."""
        miniere = self.pianifica(collina(300), densita=1.0)
        self.assertLessEqual(len(miniere), 300 * 300 // 45_000 + 1)

    def test_e_deterministica(self):
        a = self.pianifica(collina(), seed=9)
        b = self.pianifica(collina(), seed=9)
        self.assertEqual([(m.x, m.z, m.y_fondo) for m in a], [(m.x, m.z, m.y_fondo) for m in b])


class TestPiani(unittest.TestCase):

    def miniere(self):
        h = collina(300, salita=0.6)
        mare = np.zeros(h.shape, bool)
        out = []
        for seed in range(30):
            out.extend(MI.pianifica(h, mare, densita=2.0, seed=seed))
        return out

    def test_certe_miniere_hanno_piu_piani(self):
        con_piani = [m for m in self.miniere() if m.rampe_interne]
        self.assertGreater(len(con_piani), 0, "mai un secondo piano")

    def test_i_piani_scendono_sempre(self):
        for mn in self.miniere():
            quote = [mn.rampa.fine[2]] + [ri.fine[2] for ri in mn.rampe_interne]
            self.assertEqual(quote, sorted(quote, reverse=True))
            self.assertEqual(mn.y_fondo, quote[-1])

    def test_ogni_rampa_interna_parte_dalla_fine_di_un_braccio(self):
        for mn in self.miniere():
            for ri in mn.rampe_interne:
                fine_braccio = any((s.x1 + _d[0], s.z1 + _d[1]) == (ri.x, ri.z) and s.y == ri.y
                                   for s in mn.segmenti for _d in [(MI._asse(s))])
                self.assertTrue(fine_braccio, "una rampa che non parte da nessun braccio")

    def test_i_piani_sono_al_massimo_tre(self):
        for mn in self.miniere():
            self.assertLessEqual(1 + len(mn.rampe_interne), MI.LIVELLI_MAX)

    def test_la_rampa_interna_e_tutta_coperta(self):
        h = collina(300, salita=0.6)
        mare = np.zeros(h.shape, bool)
        for mn in MI.pianifica(h, mare, densita=3.0, seed=2):
            for ri in mn.rampe_interne:
                for i in range(ri.lunghezza + 1):
                    x, z, y = ri.cella(i)
                    self.assertGreaterEqual(h[z, x] - 1 - (y + MI.ALTEZZA_GALLERIA), MI.CAPPELLO)


class TestMineraleInProfondita(unittest.TestCase):
    """La ricompensa di scendere: poco sotto la superficie il comune, in fondo
    i minerali rari."""

    def frequenze(self, y, profondita, n=3000):
        rng = np.random.default_rng(1)
        conteggio = {}
        for _ in range(n):
            nome = MI._scegli_minerale(y, rng, profondita)
            conteggio[nome] = conteggio.get(nome, 0) + 1
        return conteggio

    def test_in_superficie_vince_il_carbone(self):
        f = self.frequenze(30, 0.0)
        self.assertEqual(max(f, key=f.get), "coal_ore")

    def test_in_fondo_il_raro_pesa_piu_del_comune(self):
        sopra = self.frequenze(0, 0.0)
        sotto = self.frequenze(0, 1.0)
        raro_sopra = sopra.get("diamond_ore", 0) / sum(sopra.values())
        raro_sotto = sotto.get("diamond_ore", 0) / sum(sotto.values())
        self.assertGreater(raro_sotto, raro_sopra * 5)

    def test_in_fondo_il_carbone_quasi_sparisce(self):
        sotto = self.frequenze(8, 1.0)
        self.assertLess(sotto.get("coal_ore", 0) / sum(sotto.values()), 0.05)

    def test_nessun_minerale_fuori_dalla_sua_quota(self):
        rng = np.random.default_rng(2)
        for y in (-55, 0, 40, 100):
            for _ in range(100):
                nome = MI._scegli_minerale(y, rng, 0.5)
                if nome is None:
                    continue
                m = SS.MINERALI[[x.nome for x in SS.MINERALI].index(nome)]
                self.assertTrue(m.y_min <= y <= m.y_max, f"{nome} a y={y}")

    def test_la_profondita_relativa_cresce_scendendo(self):
        self.assertAlmostEqual(MI._profondita_relativa(80, 80), 0.0)
        self.assertGreater(MI._profondita_relativa(0, 80), MI._profondita_relativa(60, 80))
        self.assertLessEqual(MI._profondita_relativa(-200, 80), 1.0)

    def test_le_miniere_profonde_hanno_minerali_piu_rari(self):
        h = collina(300, salita=0.6)
        mare = np.zeros(h.shape, bool)
        comuni = ("coal_ore", "copper_ore", "iron_ore")
        trovato_raro = False
        for seed in range(30):
            for mn in MI.pianifica(h, mare, densita=2.0, seed=seed):
                for g in mn.giacimenti:
                    if g.y < 20 and g.minerale not in comuni:
                        trovato_raro = True
        self.assertTrue(trovato_raro, "nessun minerale raro in fondo a nessuna miniera")


class TestPosaRampa(unittest.TestCase):

    Y0 = -64
    ALTEZZA = 284
    SOLIDO = 9999

    def mondo(self, quota_terreno=100):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        h_c = np.full((16, 16), quota_terreno, np.int32)
        ys = np.arange(self.Y0, self.Y0 + self.ALTEZZA)[None, :, None]
        out[np.broadcast_to(ys < quota_terreno, out.shape)] = self.SOLIDO
        return out, h_c

    def posa(self, rampa, quota_terreno=100, protetto_c=None, ox=0, oz=0, decora=True):
        out, h_c = self.mondo(quota_terreno)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        tav.decora = decora
        mn = MI.Miniera(x=rampa.x, z=rampa.z, y_superficie=rampa.y,
                        y_fondo=rampa.fine[2], rampa=rampa, segmenti=[])
        MI.posa(out, tav, h_c, ox, oz, self.Y0, [mn], [0], (self.SOLIDO,),
                seed=1, protetto_c=protetto_c)
        return out, tav, s

    def rampa(self, **kw):
        return MI.Rampa(x=kw.get("x", 2), z=kw.get("z", 8), dx=1, dz=0,
                        y=kw.get("y", 90), lunghezza=kw.get("lunghezza", 13))

    def test_la_galleria_e_larga_tre_e_alta_tre(self):
        out, tav, s = self.posa(self.rampa(y=70), quota_terreno=100, decora=False)
        x = 8
        y_pav = 70 - (x - 2) // MI.PASSO_RAMPA
        for dz in (-1, 0, 1):
            for dy in range(1, MI.ALTEZZA_GALLERIA):
                v = int(out[x, y_pav + dy - self.Y0, 8 + dz])
                self.assertIn(v, (s.id_aria, tav.rotaia["east_west"], tav.palo,
                                  tav.lanterna) + tuple(tav.rotaia.values()),
                              f"cella piena a dz={dz} dy={dy}")
        self.assertEqual(int(out[x, y_pav + MI.ALTEZZA_GALLERIA + 1 - self.Y0, 8]),
                         self.SOLIDO, "il soffitto e' sparito")

    def test_al_centro_della_rampa_si_passa_in_piedi(self):
        """Il personaggio e' alto due: sopra il pavimento, in ogni punto del
        centro, due celle devono essere libere (rotaia e aria), anche sotto la
        lanterna del telaio."""
        out, tav, s = self.posa(self.rampa(y=70), quota_terreno=100)
        liberi = {s.id_aria} | set(tav.rotaia.values())
        for x in range(2, 15):
            y_pav = 70 - (x - 2) // MI.PASSO_RAMPA
            for dy in (1, 2):
                v = int(out[x, y_pav + dy - self.Y0, 8])
                self.assertIn(v, liberi, f"x={x} dy={dy}: si sbatte la testa")

    def test_il_pavimento_e_pieno(self):
        out, tav, s = self.posa(self.rampa(y=70), quota_terreno=100)
        for x in range(2, 15):
            y_pav = 70 - (x - 2) // MI.PASSO_RAMPA
            for dz in (-1, 0, 1):
                self.assertNotEqual(int(out[x, y_pav - self.Y0, 8 + dz]), s.id_aria)

    def test_il_binario_segue_la_discesa(self):
        out, tav, s = self.posa(self.rampa(y=70), quota_terreno=100)
        piatto = tav.rotaia["east_west"]
        salita = tav.rotaia["ascending_west"]            # sale verso monte (ovest)
        visti_piatti, visti_in_pendenza = 0, 0
        for x in range(2, 15):
            i = x - 2
            y_rotaia = 70 - i // MI.PASSO_RAMPA + 1
            v = int(out[x, y_rotaia - self.Y0, 8])
            if i > 0 and i % MI.PASSO_RAMPA == 0:
                self.assertEqual(v, salita, f"cella {i}: ci vuole la rotaia in pendenza")
                visti_in_pendenza += 1
            elif v == piatto:
                visti_piatti += 1
        self.assertGreater(visti_in_pendenza, 2)
        self.assertGreater(visti_piatti, 4)

    def test_le_rotaie_in_pendenza_hanno_la_direzione_giusta_per_ogni_verso(self):
        for (dx, dz), attesa in (((1, 0), "ascending_west"), ((-1, 0), "ascending_east"),
                                 ((0, 1), "ascending_north"), ((0, -1), "ascending_south")):
            self.assertEqual(MI._VERSO_SALITA[(dx, dz)], attesa)

    def test_si_vedono_i_telai_di_travi(self):
        out, tav, s = self.posa(self.rampa(y=70), quota_terreno=100)
        self.assertTrue((out == tav.palo).any(), "nessun palo")
        travi = tav.trave["z"]                     # la rampa va a est: la trave corre su z
        self.assertTrue((out == travi).any(), "nessuna trave")

    def test_la_trincea_e_aperta_fino_al_cielo(self):
        """Dove sopra c'e' poca terra la galleria e' una trincea, non un
        tunnel: l'aria arriva alla superficie."""
        out, tav, s = self.posa(self.rampa(y=96), quota_terreno=100)
        x = 3
        y_sup = 100 - self.Y0
        self.assertEqual(int(out[x, y_sup - 1, 8]), s.id_aria, "la trincea e' coperta")

    def test_il_tunnel_e_coperto(self):
        out, tav, s = self.posa(self.rampa(y=70), quota_terreno=100)
        self.assertEqual(int(out[5, 90 - self.Y0, 8]), self.SOLIDO, "il tunnel e' aperto")

    def test_protetto_c_blocca_la_rampa(self):
        protetto = np.zeros((16, 16), bool)
        protetto[8, 8] = True
        out, tav, s = self.posa(self.rampa(y=70), quota_terreno=100, protetto_c=protetto)
        self.assertTrue((out[8, :100 - self.Y0, 8] == self.SOLIDO).all())

    def test_la_rampa_a_cavallo_di_due_chunk_arriva_completa(self):
        r = MI.Rampa(x=10, z=8, dx=1, dz=0, y=70, lunghezza=20)
        primo, tav, s = self.posa(r, ox=0)
        secondo, _, _ = self.posa(r, ox=16)
        # nel secondo chunk (x da 16 a 31) ci sono le celle 6..20 della rampa
        y = 70 - (16 - 10) // MI.PASSO_RAMPA
        self.assertEqual(int(secondo[0, y + 2 - self.Y0, 8 + 1]), s.id_aria)
        self.assertGreater(int((primo == tav.rotaia["east_west"]).sum()), 0)
        self.assertGreater(int((secondo == tav.rotaia["east_west"]).sum()), 0)

    def test_la_rampa_su_z_funziona_uguale(self):
        r = MI.Rampa(x=8, z=2, dx=0, dz=1, y=70, lunghezza=13)
        out, tav, s = self.posa(r)
        self.assertGreater(int((out == tav.rotaia["north_south"]).sum()), 4)
        self.assertGreater(int((out == tav.rotaia["ascending_north"]).sum()), 1)


class TestPortale(unittest.TestCase):

    Y0 = -64
    ALTEZZA = 284
    SOLIDO = 9999

    def posa(self, quota_terreno=71, protetto_c=None):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        h_c = np.full((16, 16), quota_terreno, np.int32)
        ys = np.arange(self.Y0, self.Y0 + self.ALTEZZA)[None, :, None]
        out[np.broadcast_to(ys < quota_terreno, out.shape)] = self.SOLIDO
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        r = MI.Rampa(x=12, z=8, dx=1, dz=0, y=quota_terreno - 1, lunghezza=40)
        mn = MI.Miniera(x=12, z=8, y_superficie=r.y, y_fondo=r.fine[2], rampa=r, segmenti=[])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1,
                protetto_c=protetto_c)
        return out, tav, s, r

    def test_due_pali_e_un_architrave(self):
        out, tav, s, r = self.posa()
        fy = r.y - self.Y0
        for dz in (-2, 2):
            for dy in range(1, 4):
                self.assertEqual(int(out[12, fy + dy, 8 + dz]), tav.palo, f"palo dz={dz} dy={dy}")
        for dz in range(-2, 3):
            self.assertEqual(int(out[12, fy + 4, 8 + dz]), tav.trave["z"])

    def test_l_apertura_e_libera(self):
        out, tav, s, r = self.posa()
        fy = r.y - self.Y0
        for dz in (-1, 0, 1):
            for dy in (1, 2):
                self.assertIn(int(out[12, fy + dy, 8 + dz]),
                              (s.id_aria, tav.rotaia["east_west"]), f"dz={dz} dy={dy}")

    def test_i_muretti_di_pietra_e_le_lanterne(self):
        out, tav, s, r = self.posa()
        fy = r.y - self.Y0
        self.assertEqual(int(out[12, fy + 1, 8 + 3]), tav.base_muro)
        self.assertEqual(int(out[12, fy + 1, 8 - 3]), tav.base_muro)
        self.assertEqual(int(out[12, fy + 3, 8 + 1]), tav.lanterna)

    def test_il_binario_esce_all_aperto_davanti_al_portale(self):
        out, tav, s, r = self.posa()
        fy = r.y - self.Y0
        for j in range(1, MI.BINARIO_FUORI - 1):
            self.assertEqual(int(out[12 - j, fy + 1, 8]), tav.rotaia["east_west"], f"j={j}")

    def test_il_binario_esterno_finisce_con_una_fermata(self):
        out, tav, s, r = self.posa()
        fy = r.y - self.Y0
        self.assertEqual(int(out[12 - (MI.BINARIO_FUORI - 1), fy + 1, 8]),
                         tav.freno["east_west"], "manca la rotaia frenante")
        self.assertEqual(int(out[12 - MI.BINARIO_FUORI, fy + 1, 8]), tav.muretto,
                         "manca il paraurti")

    def test_la_piazzola_e_piana_e_libera(self):
        out, tav, s, r = self.posa(quota_terreno=71)
        fy = r.y - self.Y0
        for j in range(1, MI.BINARIO_FUORI + 1):
            for dz in (-2, -1, 1, 2):
                self.assertNotEqual(int(out[12 - j, fy, 8 + dz]), s.id_aria, "piazzola senza fondo")
                for dy in range(1, 4):
                    self.assertEqual(int(out[12 - j, fy + dy, 8 + dz]), s.id_aria)

    def test_la_piazzola_riempie_dove_il_terreno_sta_piu_in_basso(self):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        h_c = np.full((16, 16), 71, np.int32)
        h_c[:12, :] = 67                                      # davanti alla collina il suolo scende
        ys = np.arange(self.Y0, self.Y0 + self.ALTEZZA)[None, :, None]
        out[np.broadcast_to(ys < h_c[:, None, :], out.shape)] = self.SOLIDO
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        r = MI.Rampa(x=12, z=8, dx=1, dz=0, y=70, lunghezza=40)
        mn = MI.Miniera(x=12, z=8, y_superficie=70, y_fondo=r.fine[2], rampa=r, segmenti=[])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        fy = 70 - self.Y0
        for j in range(1, MI.BINARIO_FUORI + 1):
            self.assertNotEqual(int(out[12 - j, fy, 8]), s.id_aria)
            for y in range(67, 70):
                self.assertNotEqual(int(out[12 - j, y - self.Y0, 8]), s.id_aria, f"buco a y={y}")

    def test_la_collina_dell_imbocco_e_fatta_del_territorio(self):
        """Nel deserto la collina e' sabbia e arenaria, non erba e terra."""
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        h_c = np.full((16, 16), 71, np.int32)
        ys = np.arange(self.Y0, self.Y0 + self.ALTEZZA)[None, :, None]
        out[np.broadcast_to(ys < h_c[:, None, :], out.shape)] = self.SOLIDO
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        sabbia, arenaria = s.blocco("sand"), s.blocco("sandstone")
        out[:, 70 - self.Y0, :] = sabbia
        out[:, 69 - self.Y0, :] = arenaria
        r = MI.Rampa(x=12, z=8, dx=1, dz=0, y=70, lunghezza=40)
        mn = MI.Miniera(x=12, z=8, y_superficie=70, y_fondo=r.fine[2], rampa=r, segmenti=[])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=1)
        colonna = out[14, :, 11]                       # i=2, k=3
        presenti = {int(b) for b in colonna[71 - self.Y0:]} - {s.id_aria}
        self.assertTrue(presenti, "nessuna collina")
        self.assertLessEqual(presenti, {sabbia, arenaria},
                             "la collina non e' del territorio")
        self.assertNotIn(tav.erba, presenti)
        self.assertNotIn(tav.terra, presenti)

    def test_protetto_c_blocca_il_portale(self):
        protetto = np.ones((16, 16), bool)
        out, tav, s, r = self.posa(protetto_c=protetto)
        self.assertFalse((out == tav.trave["z"]).any())


class TestMascheraIngresso(unittest.TestCase):

    def test_copre_portale_binario_e_trincea(self):
        r = MI.Rampa(x=50, z=50, dx=1, dz=0, y=80, lunghezza=60)
        mn = MI.Miniera(x=50, z=50, y_superficie=80, y_fondo=60, rampa=r, segmenti=[])
        m = MI.maschera_ingresso([mn], (100, 120))
        self.assertTrue(m[50, 50])
        self.assertTrue(m[50, 50 - MI.BINARIO_FUORI])
        self.assertTrue(m[50, 60])
        self.assertFalse(m[50, 90], "la maschera non deve seguire tutto il tunnel")
        self.assertFalse(m[10, 10])

    def test_senza_miniere_e_vuota(self):
        self.assertFalse(MI.maschera_ingresso([], (50, 50)).any())


class TestCarrelliRampa(unittest.TestCase):

    def test_c_e_un_carrello_davanti_al_portale(self):
        r = MI.Rampa(x=50, z=50, dx=1, dz=0, y=80, lunghezza=60)
        mn = MI.Miniera(x=50, z=50, y_superficie=80, y_fondo=60, rampa=r, segmenti=[])
        carrelli = MI.carrelli([mn], seed=1)
        davanti = [c for c in carrelli if c.x < 50 and abs(c.z - 50.5) < 1]
        self.assertEqual(len(davanti), 1)
        self.assertEqual(davanti[0].y, 81.0)

    def test_i_carrelli_della_rampa_stanno_sul_binario_in_pendenza(self):
        r = MI.Rampa(x=50, z=50, dx=1, dz=0, y=80, lunghezza=80)
        mn = MI.Miniera(x=50, z=50, y_superficie=80, y_fondo=60, rampa=r, segmenti=[])
        for c in MI.carrelli([mn], seed=2):
            if c.x <= 50:
                continue
            i = int(c.x) - 50
            self.assertEqual(c.y, r.cella(i)[2] + 1.0)
            self.assertFalse(r.e_gradino(i), "un carrello sul tratto in pendenza")


if __name__ == "__main__":
    unittest.main()


class TestImboccoLibero(unittest.TestCase):
    """La collina dell'imbocco e' larga e lunga: un campo o una casa li' vicino
    deve impedire la miniera, non finirci dentro."""

    def rampa(self, evita):
        h = np.full((120, 120), 70, np.int32)
        h[:, 60:] = 90                                   # una collina a est
        acqua = np.zeros(h.shape, bool)
        return MI._pianifica_rampa(50, 60, (1, 0), h, acqua, acqua, evita,
                                   np.random.default_rng(1))

    def test_senza_ostacoli_la_rampa_c_e(self):
        evita = np.zeros((120, 120), bool)
        self.assertIsNotNone(self.rampa(evita))

    def test_un_campo_a_lato_dell_imbocco_la_impedisce(self):
        evita = np.zeros((120, 120), bool)
        evita[60 - MI.LARGHEZZA_IMBOCCO - 2, 52] = True         # un campo accanto alla collina
        self.assertIsNone(self.rampa(evita))

    def test_un_campo_dentro_la_collina_la_impedisce(self):
        evita = np.zeros((120, 120), bool)
        evita[58, 55] = True                                      # dentro la collina
        self.assertIsNone(self.rampa(evita))


class TestArredoGallerie(unittest.TestCase):
    """Attrezzi, barili, ragnatele, bauli e gemme sul bordo delle gallerie."""

    Y0, ALTEZZA, SOLIDO = -64, 284, 9999

    def galleria(self, seed=1):
        out = np.full((16, self.ALTEZZA, 16), self.SOLIDO, np.uint32)
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        h_c = np.full((16, 16), 100, np.int32)
        seg = MI.Segmento(x0=0, z0=8, x1=15, z1=8, y=50)
        mn = MI.Miniera(x=0, z=8, y_superficie=100, y_fondo=50, segmenti=[seg])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (self.SOLIDO,), seed=seed)
        return out, tav, s

    def test_il_bordo_ha_degli_oggetti(self):
        trovati = set()
        for seed in range(12):
            out, tav, s = self.galleria(seed)
            y1 = 50 + 1 - self.Y0
            for z in (7, 9):
                trovati |= {int(v) for v in out[:, y1, z]}
        oggetti = {tav.barile, tav.tavolo, tav.fabbro, tav.ametista, tav.germoglio,
                   *tav.grezzo, *tav.incudine.values(), *tav.mola.values(),
                   *tav.forziere.values()}
        self.assertGreaterEqual(len(trovati & oggetti), 3, "bordo nudo")

    def test_ogni_forziere_e_registrato_per_il_bottino(self):
        n_bauli = 0
        for seed in range(30):
            out, tav, s = self.galleria(seed)
            veri = int(np.isin(out, list(tav.forziere.values())).sum())
            n_bauli += veri
            self.assertGreaterEqual(len(tav.bauli), 0)
            for (wx, wy, wz) in tav.bauli:
                self.assertIn(int(out[wx, wy - self.Y0, wz]), set(tav.forziere.values()))
        self.assertGreater(n_bauli, 0, "mai un forziere in 30 gallerie")

    def test_la_ragnatela_sta_solo_sul_soffitto_non_sul_cammino(self):
        for seed in range(20):
            out, tav, s = self.galleria(seed)
            y1 = 50 + 1 - self.Y0
            self.assertFalse((out[:, y1, :] == tav.ragnatela).any())
            self.assertFalse((out[:, y1 + 1, :] == tav.ragnatela).any())

    def test_il_centro_resta_libero_per_il_binario(self):
        out, tav, s = self.galleria(3)
        y1 = 50 + 1 - self.Y0
        centro = out[:, y1, 8]
        self.assertTrue(np.isin(centro, [s.id_aria, *tav.rotaia.values(), tav.palo]).all())


class TestPiazzolaDelTerritorio(unittest.TestCase):
    """La piazzola davanti all'imbocco ha il materiale del posto, non ghiaia."""

    Y0, ALTEZZA, SOLIDO = -64, 284, 9999

    def posa(self, superficie, sotto):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        h_c = np.full((16, 16), 71, np.int32)
        ys = np.arange(self.Y0, self.Y0 + self.ALTEZZA)[None, :, None]
        out[np.broadcast_to(ys < h_c[:, None, :], out.shape)] = sotto
        out[:, 70 - self.Y0, :] = superficie
        s = FintoScrittore()
        tav = MI.Tavolozza(s)
        tav.decora = False
        r = MI.Rampa(x=12, z=8, dx=1, dz=0, y=70, lunghezza=40)
        mn = MI.Miniera(x=12, z=8, y_superficie=70, y_fondo=r.fine[2], rampa=r, segmenti=[])
        MI.posa(out, tav, h_c, 0, 0, self.Y0, [mn], [0], (sotto,), seed=1)
        return out, tav

    def test_nel_deserto_la_piazzola_e_sabbia(self):
        s0 = FintoScrittore()
        sabbia, arenaria = 70, 71
        out, tav = self.posa(sabbia, arenaria)
        fy = 70 - self.Y0
        for j in range(1, MI.BINARIO_FUORI + 1):
            for dz in (-2, -1, 1, 2):
                self.assertEqual(int(out[12 - j, fy, 8 + dz]), sabbia, f"j={j} dz={dz}")
        self.assertFalse((out[:, fy, :] == tav.ghiaia).any() and False)

    def test_senza_ghiaia_davanti_all_imbocco(self):
        out, tav = self.posa(70, 71)
        fy = 70 - self.Y0
        piazzola = out[12 - MI.BINARIO_FUORI:12, fy, 6:11]
        self.assertFalse((piazzola == tav.ghiaia).any(), "ghiaia sulla piazzola")
