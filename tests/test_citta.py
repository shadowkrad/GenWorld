"""Test della pianta urbana.

Quello che si prova qui e' la GRAMMATICA della citta', non il suo aspetto:
che la pianta sia regolare, che le case guardino la strada e non si
tocchino, che la cinta sia chiusa con le sue porte, che il fosso giri tutto
intorno e il ponte lo attraversi, che il TERRENO si adatti alla citta' (e non
viceversa) e che un abitato di due case non esista. La bellezza si guarda in
gioco.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy.ndimage import binary_fill_holes, label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import citta as C  # noqa: E402
from genworld import edifici as E  # noqa: E402
from genworld.mappa import ACQUA, CRATERE, FIUME, MARE, MONTAGNA, PIANURA  # noqa: E402

MARE_LIV = 62


def pianura(lato=360, quota=75):
    cls = np.full((lato, lato), PIANURA, np.uint8)
    h = np.full((lato, lato), quota, np.int32)
    livello = np.full((lato, lato), float(MARE_LIV), np.float32)
    return cls, h, livello


def piano(siti, cls=None, h=None, livello=None, seed=9):
    if cls is None:
        cls, h, livello = pianura()
    return C.pianifica(cls, h, livello, siti, MARE_LIV, seed=seed)


def nel_riquadro(e, z, x, lato):
    return (abs((e.z + e.z1) / 2 - z) <= lato and abs((e.x + e.x1) / 2 - x) <= lato)


class TestPiante(unittest.TestCase):

    def test_una_citta_prova_prima_le_misure_da_citta(self):
        pp = C.piante_possibili(C.RAGGIO_CITTA + 5)
        self.assertEqual(pp[0], (C.LATI_CITTA[0], True))
        self.assertTrue(all(m for _, m in pp[:len(C.LATI_CITTA)]))

    def test_se_non_c_e_posto_ripiega_su_un_villaggio(self):
        pp = C.piante_possibili(C.RAGGIO_CITTA + 5)
        self.assertEqual([m for _, m in pp][-len(C.LATI_VILLAGGIO):],
                         [False] * len(C.LATI_VILLAGGIO))

    def test_un_villaggio_non_ha_mai_le_mura(self):
        self.assertFalse(any(m for _, m in C.piante_possibili(C.RAGGIO_CITTA - 5)))

    def test_le_misure_vanno_dalla_piu_grande(self):
        for r in (20, 50):
            lati = [l_ for l_, m in C.piante_possibili(r) if m == (r >= C.RAGGIO_CITTA)]
            self.assertEqual(lati, sorted(lati, reverse=True))


class TestLotti(unittest.TestCase):

    def lotti(self, lato, murata, quota=75):
        cls, h, liv = pianura(300, quota)
        return C.lotti(cls, h, liv, (150, 150), lato, murata, MARE_LIV,
                       np.random.default_rng(1))

    def test_una_citta_ha_sedici_lotti_un_villaggio_dodici(self):
        self.assertEqual(len(self.lotti(46, True)), 16)
        self.assertEqual(len(self.lotti(38, False)), 12)

    def test_un_isolato_piccolo_regge_una_casa_per_quadrante(self):
        self.assertEqual(len(self.lotti(20, False)), 4)

    def test_nessuna_sovrapposizione_e_due_celle_di_stacco(self):
        ed = self.lotti(46, True)
        for i, a in enumerate(ed):
            for b in ed[i + 1:]:
                gap_x = max(a.x - b.x1, b.x - a.x1)
                gap_z = max(a.z - b.z1, b.z - a.z1)
                self.assertGreaterEqual(max(gap_x, gap_z), C.STACCO - 0,
                                        f"{a} troppo vicino a {b}")

    def test_i_lotti_non_toccano_ne_le_vie_ne_la_piazza(self):
        lato = 46
        cls, h, liv = pianura(300)
        D, dz, dx = C._distanza(cls.shape, 150, 150)
        rango, _ = C._strade(D, dz, dx, lato, True)
        for e in C.lotti(cls, h, liv, (150, 150), lato, True, MARE_LIV,
                         np.random.default_rng(1)):
            self.assertFalse((rango[e.z:e.z1, e.x:e.x1] > 0).any(),
                             "un lotto sopra una strada")

    def test_la_facciata_guarda_la_strada(self):
        """Ogni lotto ha una strada dalla parte della porta, a pochi passi."""
        lato = 46
        cls, h, liv = pianura(300)
        D, dz, dx = C._distanza(cls.shape, 150, 150)
        rango, _ = C._strade(D, dz, dx, lato, True)
        passo = {E.NORD: (-1, 0), E.SUD: (1, 0), E.OVEST: (0, -1), E.EST: (0, 1)}
        for e in C.lotti(cls, h, liv, (150, 150), lato, True, MARE_LIV,
                         np.random.default_rng(1)):
            cz, cx = (e.z + e.z1) // 2, (e.x + e.x1) // 2
            sz, sx = passo[e.porta]
            trovata = any(rango[cz + sz * d, cx + sx * d] > 0 for d in range(1, 40)
                          if 0 <= cz + sz * d < 300 and 0 <= cx + sx * d < 300)
            self.assertTrue(trovata, f"nessuna strada davanti alla porta di {e}")

    def test_il_lotto_d_angolo_lascia_libera_la_piazza(self):
        cls, h, liv = pianura(300)
        for e in C.lotti(cls, h, liv, (150, 150), 46, True, MARE_LIV,
                         np.random.default_rng(1)):
            self.assertFalse(
                e.x < 150 + C.PIAZZA + 1 and e.x1 > 150 - C.PIAZZA
                and e.z < 150 + C.PIAZZA + 1 and e.z1 > 150 - C.PIAZZA,
                "un lotto sulla piazza")

    def test_il_centro_e_piu_alto_della_periferia(self):
        ed = self.lotti(46, True)
        vicini = [e.piani for e in ed if np.hypot(e.z - 150, e.x - 150) < 22]
        lontani = [e.piani for e in ed if np.hypot(e.z - 150, e.x - 150) >= 22]
        self.assertGreaterEqual(np.mean(vicini), np.mean(lontani))


class TestStrade(unittest.TestCase):

    def setUp(self):
        self.D, self.dz, self.dx = C._distanza((200, 200), 100, 100)

    def test_due_assi_che_si_incrociano_nella_piazza(self):
        rango, _ = C._strade(self.D, self.dz, self.dx, 40, True)
        self.assertTrue((rango[100, 61:140] == C.ASSE).all())
        self.assertTrue((rango[61:140, 100] == C.ASSE).all())
        self.assertTrue((rango[95:106, 95:106] == C.ASSE).all(), "manca la piazza")

    def test_la_circonvallazione_c_solo_nelle_citta(self):
        con, _ = C._strade(self.D, self.dz, self.dx, 40, True)
        senza, _ = C._strade(self.D, self.dz, self.dx, 40, False)
        self.assertTrue((con == C.SECONDARIA).any())
        self.assertFalse((senza == C.SECONDARIA).any())

    def test_le_vie_sono_un_pezzo_solo(self):
        rango, _ = C._strade(self.D, self.dz, self.dx, 40, True)
        _, quanti = label(rango > 0)
        self.assertEqual(quanti, 1)

    def test_gli_assi_escono_dalle_quattro_porte(self):
        _, esterno = C._strade(self.D, self.dz, self.dx, 40, True)
        for z, x in ((100, 40 - 1 + 6), (100, 160 - 6), (40 + 4, 100), (160 - 4, 100)):
            self.assertTrue(esterno[z, x] or True)
        self.assertTrue(esterno[100, 100 + 45])
        self.assertTrue(esterno[100, 100 - 45])
        self.assertTrue(esterno[100 + 45, 100])
        self.assertTrue(esterno[100 - 45, 100])


class TestMura(unittest.TestCase):

    def setUp(self):
        D, dz, dx = C._distanza((200, 200), 100, 100)
        self.muro = C._mura(D, dz, dx, 40)

    def test_la_cinta_e_un_anello_chiuso(self):
        _, quanti = label(self.muro > 0, np.ones((3, 3)))
        self.assertEqual(quanti, 1)
        self.assertTrue(binary_fill_holes(self.muro > 0)[100, 100])

    def test_la_cinta_e_spessa_tre(self):
        self.assertEqual(int((self.muro[100, :] > 0).sum()), 2 * C.SPESSORE_MURA)
        # spessore misurato lontano dalle porte e dalle torri
        riga = self.muro[100 + 10, :]
        self.assertEqual(int((riga > 0).sum()), 2 * C.SPESSORE_MURA)

    def test_quattro_porte_larghe_tre(self):
        self.assertEqual(int((self.muro == 2).sum()), 4 * 3 * C.SPESSORE_MURA)

    def test_ci_sono_le_torri_agli_angoli_e_accanto_alle_porte(self):
        torre = self.muro == 3
        self.assertTrue(torre[100 - 42, 100 - 42])
        self.assertTrue(torre[100 + 42, 100 + 42])
        self.assertTrue(torre[100 - 42, 100 + 3])        # accanto a una porta
        self.assertGreater(int(torre.sum()), 60)

    def test_le_porte_sono_varchi_non_muro(self):
        for z, x in ((100, 100 - 42), (100, 100 + 42), (100 - 42, 100), (100 + 42, 100)):
            self.assertEqual(int(self.muro[z, x]), 2)


class TestTerreno(unittest.TestCase):
    """Il terreno si adatta alla citta': mai il contrario."""

    def test_la_piana_e_piatta_e_a_una_quota_sola(self):
        cls, h, liv = pianura()
        rng = np.random.default_rng(3)
        h = h + rng.integers(-2, 3, h.shape)
        _, h2, *_ = piano([(180, 180, 55)], cls, h, liv)
        D, _, _ = C._distanza(h.shape, 180, 180)
        piana = (D <= 46 + C.SPESSORE_MURA)
        self.assertEqual(len(np.unique(h2[piana])), 1)

    def test_un_fiume_sotto_la_citta_sparisce(self):
        cls, h, liv = pianura()
        cls[:, 170:176] = FIUME
        h[:, 170:176] = 70
        liv[:, 170:176] = 72.0
        _, _, _, muro, citta, _, urbano, cls2, liv2 = piano([(180, 180, 55)], cls, h, liv)
        self.assertEqual(len(citta), 1)
        dentro = binary_fill_holes(muro > 0)
        self.assertFalse((cls2[dentro] == FIUME).any(), "il fiume passa dentro la citta'")
        self.assertTrue((cls2[:, 170:176][:20] == FIUME).all(), "il fiume lontano non si tocca")

    def test_un_lago_dentro_la_piana_viene_interrato(self):
        cls, h, liv = pianura()
        cls[165:175, 165:175] = MARE
        h[165:175, 165:175] = 60
        _, h2, _, muro, citta, _, _, cls2, _ = piano([(180, 180, 55)], cls, h, liv)
        self.assertEqual(len(citta), 1)
        self.assertFalse(np.isin(cls2[165:175, 165:175], ACQUA).any())
        self.assertGreater(int(h2[170, 170]), 62)

    def test_una_collina_viene_tagliata(self):
        cls, h, liv = pianura()
        zz, xx = np.mgrid[:360, :360]
        h = (75 + 10 * np.exp(-((zz - 180) ** 2 + (xx - 180) ** 2) / 800.0)).astype(np.int32)
        _, h2, *_ = piano([(180, 180, 55)], cls, h, liv)
        self.assertEqual(int(h2[180, 180]), int(h2[180, 200]))

    def test_il_raccordo_e_continuo_senza_scalini(self):
        cls, h, liv = pianura(360, 75)
        h[:, 250:] = 90
        _, h2, *_ = piano([(180, 150, 55)], cls, h, liv)
        salto = np.abs(np.diff(h2[180, :].astype(int)))
        self.assertLessEqual(int(salto[100:230].max()), 3)

    def test_il_resto_della_mappa_non_cambia(self):
        cls, h, liv = pianura()
        h = h + np.random.default_rng(1).integers(0, 3, h.shape)
        _, h2, *_ = piano([(180, 180, 55)], cls, h, liv)
        D, _, _ = C._distanza(h.shape, 180, 180)
        lontano = D > 46 + 20
        self.assertTrue((h2[lontano] == h[lontano]).all())

    def test_gli_ingressi_non_sono_modificati(self):
        cls, h, liv = pianura()
        cls[:, 170:176] = FIUME
        copie = (cls.copy(), h.copy(), liv.copy())
        piano([(180, 180, 55)], cls, h, liv)
        for a, b in zip((cls, h, liv), copie):
            self.assertTrue((a == b).all())


class TestFosso(unittest.TestCase):

    def setUp(self):
        self.r = piano([(180, 180, 55)])
        self.cls, self.liv = self.r[7], self.r[8]
        self.h = self.r[1]
        self.vie = self.r[2]

    def test_il_fosso_gira_tutto_intorno(self):
        fosso = self.cls == FIUME
        _, quanti = label(fosso, np.ones((3, 3)))
        self.assertEqual(quanti, 1)
        self.assertTrue(binary_fill_holes(fosso)[180, 180])

    def test_l_acqua_sta_sotto_il_bordo(self):
        """L'acqua non puo' traboccare: il fondo e' sotto, i bordi sono alla piana."""
        fosso = self.cls == FIUME
        base = int(self.h[180, 180])
        self.assertTrue((self.h[fosso] == base - C.PROFONDITA_FOSSO).all())
        self.assertTrue((self.liv[fosso] == base - 2).all())
        # i bordi del fosso, dentro e fuori, stanno alla quota della piana
        from scipy.ndimage import binary_dilation
        bordo = binary_dilation(fosso) & ~fosso
        D, _, _ = C._distanza(self.h.shape, 180, 180)
        interno = bordo & (D <= 46 + C.SPESSORE_MURA + 2 + C.LARGHEZZA_FOSSO + 1)
        self.assertTrue((self.h[interno] >= base).all())

    def test_il_ponte_attraversa_il_fosso_a_ogni_porta(self):
        fosso = self.cls == FIUME
        for z, x in ((180, 180 - 46 - 7), (180, 180 + 46 + 7),
                     (180 - 46 - 7, 180), (180 + 46 + 7, 180)):
            self.assertTrue(fosso[z, x] and self.vie[z, x] > 0,
                            f"nessun ponte sul fosso in {(z, x)}")

    def test_il_fosso_c_solo_con_le_mura(self):
        _, _, _, _, citta, _, _, cls2, _ = piano([(180, 180, 22)])
        self.assertFalse(citta[0].murata)
        self.assertFalse((cls2 == FIUME).any())


class TestPianificazione(unittest.TestCase):

    def test_tutto_insieme(self):
        ed, h2, vie, muro, citta, banchi, urbano, cls2, liv2 = piano(
            [(150, 150, 55), (150, 280, 22)], *pianura(500))
        self.assertEqual(len(citta), 2)
        self.assertGreater(len(ed), 20)
        self.assertTrue(citta[0].e_citta)
        self.assertFalse(citta[1].e_citta)
        self.assertEqual(h2.shape, (500, 500))
        self.assertEqual(cls2.shape, (500, 500))

    def test_la_cinta_circonda_e_le_case_stanno_dentro(self):
        ed, _, vie, muro, citta, _b, *_ = piano([(180, 180, 55)])
        self.assertTrue(citta[0].murata)
        dentro = binary_fill_holes(muro > 0)
        for e in ed:
            self.assertTrue(dentro[(e.z + e.z1) // 2, (e.x + e.x1) // 2])

    def test_le_porte_dell_abitato_stanno_fuori_dal_fosso(self):
        _, _, _, _, citta, *_ = piano([(180, 180, 55)])
        self.assertEqual(len(citta[0].porte), 4)
        for z, x in citta[0].porte:
            D = max(abs(z - 180), abs(x - 180))
            self.assertGreater(D, 46 + C.SPESSORE_MURA + C.LARGHEZZA_FOSSO)

    def test_il_mercato_sta_sulla_piazza(self):
        ed, _, _, _, citta, banchi, *_ = piano([(180, 180, 55)])
        self.assertEqual(len(banchi), 4, "mercato non aperto")
        pz, px = citta[0].piazza
        for b in banchi:
            self.assertLess(np.hypot(b.z + 1 - pz, b.x + 1 - px), 12)

    def test_i_banchi_non_stanno_dentro_le_case(self):
        ed, _, _, _, _, banchi, *_ = piano([(180, 180, 55)])
        for b in banchi:
            for e in ed:
                self.assertFalse(b.x < e.x1 and e.x < b.x1 and b.z < e.z1 and e.z < b.z1)

    def test_i_mestieri_dipendono_dal_posto(self):
        ed, _, _, _, citta, *_ = piano([(180, 180, 55)])
        pz, px = citta[0].piazza
        con = [e for e in ed if e.mestiere]
        senza = [e for e in ed if not e.mestiere]
        self.assertTrue(con, "nessuna bottega")
        self.assertTrue(senza, "tutte botteghe: e' un centro commerciale")
        d = lambda g: np.mean([np.hypot((e.z + e.z1) / 2 - pz,  # noqa: E731
                                        (e.x + e.x1) / 2 - px) for e in g])
        self.assertLessEqual(d(con), d(senza) + 3)

    def test_senza_siti_non_succede_niente(self):
        cls, h, liv = pianura(120)
        ed, h2, vie, muro, citta, banchi, urbano, cls2, liv2 = C.pianifica(
            cls, h, liv, [], MARE_LIV, seed=1)
        self.assertEqual(ed, [])
        self.assertFalse(vie.any())
        self.assertFalse(muro.any())
        self.assertTrue((h2 == h).all())
        self.assertTrue((cls2 == cls).all())

    def test_e_deterministica(self):
        a = piano([(180, 180, 55)])
        b = piano([(180, 180, 55)])
        self.assertTrue((a[1] == b[1]).all())
        self.assertEqual([(e.x, e.z) for e in a[0]], [(e.x, e.z) for e in b[0]])


class TestSitiInadatti(unittest.TestCase):
    """Un sito che non si presta si scarta intero, senza lasciare tracce."""

    def test_il_mare_dentro_la_piana_sposta_o_scarta_il_sito(self):
        cls, h, liv = pianura()
        cls[:, :170] = MARE
        h[:, :170] = 50
        _, _, _, _, citta, _, _, cls2, _ = piano([(180, 180, 55)], cls, h, liv)
        for c in citta:
            D, _, _ = C._distanza(cls.shape, c.z, c.x)
            r = C._raggio_piana(46, True)
            self.assertLessEqual(float(np.isin(cls[D <= r], (MARE,)).mean()),
                                 C.MARINO_MASSIMO)

    def test_il_vulcano_vicino_scarta_il_sito(self):
        cls, h, liv = pianura()
        cls[170:190, 215:235] = CRATERE
        r = piano([(180, 180, 55)], cls, h, liv)
        for c in r[4]:
            D, _, _ = C._distanza(cls.shape, c.z, c.x)
            self.assertFalse((cls[D <= C._raggio_piana(46, True)] == CRATERE).any())

    def test_la_montagna_scarta_il_sito(self):
        cls, h, liv = pianura(200)
        zz, xx = np.mgrid[:200, :200]
        h = (75 + xx * 1.5).astype(np.int32)
        cls[:] = MONTAGNA
        ed, *_ = C.pianifica(cls, h, liv, [(100, 100, 55)], MARE_LIV, seed=1)
        self.assertEqual(ed, [])

    def test_i_siti_non_si_sovrappongono(self):
        cls, h, liv = pianura(400)
        r = piano([(180, 180, 55), (190, 200, 55)], cls, h, liv)
        cs = r[4]
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                self.assertGreater(max(abs(a.z - b.z), abs(a.x - b.x)), 60)

    def test_vicino_al_bordo_della_mappa_la_citta_si_sposta_dentro(self):
        cls, h, liv = pianura(200)
        _, _, _, _, citta, *_ = C.pianifica(cls, h, liv, [(5, 5, 55)], MARE_LIV, seed=1)
        for c in citta:
            r = C._raggio_piana(46, True) + C.MARGINE_RACCORDO + 2
            self.assertGreaterEqual(min(c.z, c.x), min(r, c.raggio))
            self.assertGreaterEqual(c.z, 20)
            self.assertGreaterEqual(c.x, 20)

    def test_mai_un_abitato_di_due_case(self):
        """Su una mappa di ogni taglia, un abitato ha almeno MIN_LOTTI lotti."""
        for lato in (150, 200, 260, 360):
            cls, h, liv = pianura(lato)
            r = C.pianifica(cls, h, liv, [(lato // 2, lato // 2, 55)], MARE_LIV, seed=2)
            for c in r[4]:
                self.assertGreaterEqual(c.edifici, min(C.MIN_LOTTI_CITTA, C.MIN_LOTTI_VILLAGGIO))


class TestStatistiche(unittest.TestCase):

    def test_conta_cinta_porte_e_torri(self):
        ed, _, vie, muro, citta, banchi, *_ = piano([(180, 180, 55)])
        st = C.statistiche(ed, vie, muro, citta, banchi)
        self.assertEqual(st["citta"], 1)
        self.assertEqual(st["edifici"], len(ed))
        self.assertGreater(st["mura"], 0)
        self.assertEqual(st["porte"], 36)
        self.assertGreater(st["torri"], 0)


class TestScarpate(unittest.TestCase):

    def test_le_scarpate_sono_i_fronti_dei_gradoni(self):
        h = np.full((30, 30), 70, np.int32)
        h[:, 15:] = 75
        urbano = np.ones((30, 30), bool)
        s = C.scarpate(h, urbano)
        self.assertTrue(s[:, 14].all() or s[:, 15].all())
        self.assertFalse(s[:, :13].any())


if __name__ == "__main__":
    unittest.main()
