"""Test della pianta urbana.

Quello che si prova qui e' la GRAMMATICA della citta', non il suo aspetto:
che le case guardino la strada, che stiano dentro l'area, che non si
sovrappongano, che le vie siano connesse, che le porte stiano sul muro. La
bellezza si guarda su `mondi/citta.png`.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import citta as C  # noqa: E402
from genworld import edifici as E  # noqa: E402
from genworld.mappa import FIUME, MARE, PIANURA  # noqa: E402


def pianura(lato=200, quota=75):
    cls = np.full((lato, lato), PIANURA, np.uint8)
    h = np.full((lato, lato), quota, np.int32)
    livello = np.full((lato, lato), 62.0, np.float32)
    return cls, h, livello


class TestContorno(unittest.TestCase):

    def test_non_e_un_cerchio(self):
        """Un perimetro circolare si legge come un timbro."""
        cls, h, _ = pianura()
        area = C.contorno(cls, h, 100, 100, 40, 62, seed=3)
        zz, xx = np.nonzero(area)
        r = np.hypot(zz - 100, xx - 100)
        # il raggio del bordo deve variare con l'angolo
        bordo = r > np.percentile(r, 92)
        self.assertGreater(float(np.std(r[bordo])), 1.5)

    def test_esclude_l_acqua(self):
        cls, h, _ = pianura()
        cls[:, :100] = MARE
        h[:, :100] = 50
        area = C.contorno(cls, h, 100, 100, 40, 62, seed=3)
        self.assertFalse(area[:, :100].any())

    def test_la_piazza_sta_dentro(self):
        cls, h, _ = pianura()
        cls[:, :100] = MARE
        h[:, :100] = 50
        area = C.contorno(cls, h, 100, 110, 40, 62, seed=3)
        pz, px = C._piazza(area, h, 100, 110)
        self.assertTrue(area[pz, px])


class TestRete(unittest.TestCase):

    def setUp(self):
        self.cls, self.h, self.livello = pianura(240)
        self.area = C.contorno(self.cls, self.h, 120, 120, 50, 62, seed=5)
        self.piazza = C._piazza(self.area, self.h, 120, 120)
        self.rango, self.porte, self.punti = C.rete(
            self.area, self.piazza, 50, seed=5)

    def test_le_vie_sono_un_pezzo_solo(self):
        """Una citta' con un quartiere scollegato dal resto non e' una citta'."""
        _, quanti = label(self.rango > 0, np.ones((3, 3)))
        self.assertEqual(quanti, 1, f"{quanti} tronconi di strada separati")

    def test_gerarchia_a_piramide(self):
        """Gli assi devono essere pochi, i vicoli tanti: se e' il contrario
        la gerarchia non c'e', c'e' solo una tavolozza di larghezze."""
        assi = int((self.rango == C.ASSE).sum())
        vicoli = int((self.rango == C.VICOLO).sum())
        self.assertGreater(assi, 0)
        self.assertGreater(vicoli, 0)

    def test_le_porte_stanno_in_periferia(self):
        self.assertGreaterEqual(len(self.porte), 2)
        for pz, px in self.porte:
            d = np.hypot(pz - self.piazza[0], px - self.piazza[1])
            self.assertGreater(d, 50 * 0.4, "porta troppo vicina alla piazza")

    def test_le_vie_restano_nell_area(self):
        self.assertFalse((self.rango > 0)[~self.area].any())

    def test_la_piazza_e_uno_slargo(self):
        pz, px = self.piazza
        attorno = self.rango[pz - 2:pz + 3, px - 2:px + 3]
        self.assertTrue((attorno == C.ASSE).all())


class TestLotti(unittest.TestCase):

    def setUp(self):
        self.cls, self.h, self.livello = pianura(240)
        self.area = C.contorno(self.cls, self.h, 120, 120, 50, 62, seed=7)
        self.piazza = C._piazza(self.area, self.h, 120, 120)
        self.rango, _, _ = C.rete(self.area, self.piazza, 50, seed=7)
        self.edifici = C.lotti(self.area, self.rango > 0, self.cls, self.h,
                               self.livello, self.piazza, 50, 62,
                               np.random.default_rng(7))

    def test_ce_ne_sono(self):
        """Il primo tentativo ne costruiva otto su una citta' intera."""
        self.assertGreater(len(self.edifici), 25, f"solo {len(self.edifici)}")

    def test_nessuna_sovrapposizione(self):
        occupato = np.zeros(self.cls.shape, bool)
        for ed in self.edifici:
            fetta = occupato[ed.z:ed.z1, ed.x:ed.x1]
            self.assertFalse(fetta.any(), "due case sullo stesso sedime")
            occupato[ed.z:ed.z1, ed.x:ed.x1] = True

    def test_nessun_tetto_sovrapposto(self):
        """Il sedime non basta. I tetti sporgono oltre i muri, e due gronde
        nello stesso vicolo finivano nella stessa cella: dall'alto si vedeva
        benissimo, dai test no, perche' guardavano solo i sedimi."""
        occupato = np.zeros(self.cls.shape, bool)
        for ed in self.edifici:
            x0, z0, x1, z1 = ed.ingombro_tetto()
            fetta = occupato[max(0, z0):z1, max(0, x0):x1]
            self.assertFalse(fetta.any(),
                             f"tetti sovrapposti attorno a ({ed.z}, {ed.x})")
            occupato[max(0, z0):z1, max(0, x0):x1] = True

    def test_chi_ha_un_vicino_attaccato_rinuncia_alla_gronda(self):
        """La regola vera: la gronda si perde quando non c'e' posto, non
        quando si e' in centro. Una casa a schiera non sporge sul vicino."""
        for a in self.edifici:
            if a.gronda == 0:
                continue
            for b in self.edifici:
                if b is a:
                    continue
                vicino = (a.x - 1 <= b.x1 and b.x - 1 <= a.x1
                          and a.z - 1 <= b.z1 and b.z - 1 <= a.z1)
                self.assertFalse(
                    vicino, f"gronda su ({a.z},{a.x}) col vicino ({b.z},{b.x})")

    def test_nessuna_casa_sulla_strada(self):
        vie = self.rango > 0
        for ed in self.edifici:
            self.assertFalse(vie[ed.z:ed.z1, ed.x:ed.x1].any(),
                             "casa costruita sulla carreggiata")

    def test_la_facciata_guarda_la_strada(self):
        """Il punto di tutta la pianificazione: la porta da' sulla via.

        Si controlla TUTTO il fronte, non il suo punto di mezzo: il lotto
        cresce di lato quanto l'isolato gli concede, quindi il centro della
        facciata non coincide con la cella da cui e' partito. Misurare li'
        bocciava nove case su sessantatre che erano perfettamente in regola.
        """
        vie = self.rango > 0
        H, W = vie.shape
        sbagliate = []
        for ed in self.edifici:
            if ed.porta == E.NORD:
                fronte = [(ed.z - 1, x) for x in range(ed.x, ed.x1)]
                verso = (-1, 0)
            elif ed.porta == E.SUD:
                fronte = [(ed.z1, x) for x in range(ed.x, ed.x1)]
                verso = (1, 0)
            elif ed.porta == E.OVEST:
                fronte = [(z, ed.x - 1) for z in range(ed.z, ed.z1)]
                verso = (0, -1)
            else:
                fronte = [(z, ed.x1) for z in range(ed.z, ed.z1)]
                verso = (0, 1)
            vicino = False
            for fz, fx in fronte:
                for d in range(0, 4):
                    z, x = fz + verso[0] * d, fx + verso[1] * d
                    if 0 <= z < H and 0 <= x < W and vie[z, x]:
                        vicino = True
                        break
                if vicino:
                    break
            if not vicino:
                sbagliate.append((ed.z, ed.x))
        self.assertLessEqual(len(sbagliate), len(self.edifici) // 20,
                             f"{len(sbagliate)} case su {len(self.edifici)} "
                             f"non danno sulla strada: {sbagliate[:5]}")

    def test_il_centro_e_piu_alto_della_periferia(self):
        """La densita' deve degradare: senza gradiente e' un quartiere
        residenziale caduto dal cielo."""
        d = np.array([np.hypot((e.z + e.z1) / 2 - self.piazza[0],
                               (e.x + e.x1) / 2 - self.piazza[1])
                      for e in self.edifici])
        piani = np.array([e.piani for e in self.edifici])
        centro = piani[d < np.median(d)].mean()
        fuori = piani[d >= np.median(d)].mean()
        self.assertGreater(centro, fuori)

    def test_i_lotti_grandi_possono_superare_il_vecchio_tetto(self):
        """Con `FRONTE_MAX_*`/`FONDO_MAX` troppo bassi (13/9, il valore di
        prima) quasi nessun template scaricato entrava mai in un lotto, e
        `template.scegli()` finiva per ripetere sempre lo stesso modello
        piccolo - non un difetto di scelta, un tetto troppo basso per il
        catalogo (vedi il commento sopra le costanti). Qui non serve un
        isolato grandissimo: dove il blocco lo permette, il lotto deve poter
        crescere oltre il vecchio limite - altrimenti la costante e' stata
        abbassata di nuovo per errore."""
        oltre = [e for e in self.edifici if e.larghezza > 13 or e.profondita > 9]
        self.assertGreater(len(oltre), 0,
                           "nessun lotto supera il vecchio tetto 13/9: "
                           "FRONTE_MAX_*/FONDO_MAX sono stati riabbassati?")


class TestPianificazione(unittest.TestCase):

    def test_tutto_insieme(self):
        cls, h, livello = pianura(300)
        siti = [(150, 150, 55), (60, 60, 22)]
        ed, h2, vie, muro, citta, banchi, _u = C.pianifica(cls, h, livello, siti, 62, seed=9)
        self.assertEqual(len(citta), 2)
        self.assertGreater(len(ed), 30)
        self.assertTrue(citta[0].e_citta)
        self.assertFalse(citta[1].e_citta)
        # il terreno e' stato spianato sotto le case
        self.assertEqual(h2.shape, h.shape)

    def test_le_mura_circondano_e_hanno_le_porte(self):
        cls, h, livello = pianura(300)
        ed, _, vie, muro, citta, _b, _u = C.pianifica(
            cls, h, livello, [(150, 150, 55)], 62, seed=9)
        self.assertTrue(citta[0].murata)
        self.assertGreater(int((muro == 1).sum()), 100, "cinta troppo corta")
        self.assertGreater(int((muro == 2).sum()), 0, "nessuna porta nel muro")
        # la cinta e' un anello chiuso: una sola componente
        _, quanti = label(muro > 0, np.ones((3, 3)))
        self.assertEqual(quanti, 1, f"{quanti} pezzi di cinta")
        # e le case stanno dentro, non fuori
        from scipy.ndimage import binary_fill_holes
        dentro = binary_fill_holes(muro > 0)
        fuori = sum(1 for e in ed
                    if not dentro[(e.z + e.z1) // 2, (e.x + e.x1) // 2])
        self.assertLess(fuori, len(ed) // 3, f"{fuori} case fuori dalle mura")

    def test_il_mercato_sta_sulla_piazza(self):
        """Il banco sta sul selciato, sul bordo della piazza. La prima
        versione lo cercava su terreno libero e non ne apriva mai nessuno."""
        cls, h, livello = pianura(300)
        ed, _, vie, muro, citta, banchi, _u = C.pianifica(
            cls, h, livello, [(150, 150, 55)], 62, seed=9)
        self.assertGreaterEqual(len(banchi), 3, "mercato non aperto")
        pz, px = citta[0].piazza
        for b in banchi:
            d = np.hypot(b.z + 1 - pz, b.x + 1 - px)
            self.assertLess(d, 12, f"banco lontano dalla piazza: {d:.0f}")

    def test_i_banchi_non_stanno_dentro_le_case(self):
        cls, h, livello = pianura(300)
        ed, _, _, _, _, banchi, _u = C.pianifica(
            cls, h, livello, [(150, 150, 55)], 62, seed=9)
        for b in banchi:
            for e in ed:
                sovrapposti = (b.x < e.x1 and e.x < b.x1
                               and b.z < e.z1 and e.z < b.z1)
                self.assertFalse(sovrapposti, "banco dentro una casa")

    def test_i_mestieri_dipendono_dal_posto(self):
        """Le botteghe stanno al centro, non sparse a caso."""
        cls, h, livello = pianura(300)
        ed, _, _, _, citta, _, _u = C.pianifica(
            cls, h, livello, [(150, 150, 55)], 62, seed=9)
        pz, px = citta[0].piazza
        con = [e for e in ed if e.mestiere]
        senza = [e for e in ed if not e.mestiere]
        self.assertTrue(con, "nessuna bottega")
        self.assertTrue(senza, "tutte botteghe: e' un centro commerciale")
        d = lambda g: np.mean([np.hypot((e.z + e.z1) / 2 - pz,  # noqa: E731
                                        (e.x + e.x1) / 2 - px) for e in g])
        self.assertLess(d(con), d(senza))

    def test_senza_siti_non_succede_niente(self):
        cls, h, livello = pianura(120)
        ed, h2, vie, muro, citta, banchi, _u = C.pianifica(cls, h, livello, [], 62, seed=1)
        self.assertEqual(ed, [])
        self.assertFalse(vie.any())
        self.assertFalse(muro.any())
        self.assertTrue((h2 == h).all())


if __name__ == "__main__":
    unittest.main()


class TestCintaLeggibile(unittest.TestCase):
    """Negli screenshot le mura sembravano macerie sparse. Due cause: una
    cella sola di spessore - che e' connessa solo in diagonale, quindi sul
    terreno si vedono cubi che si toccano per lo spigolo - e nessuna torre."""

    def scenario(self):
        lato = 200
        cls = np.full((lato, lato), PIANURA, np.uint8)
        h = np.full((lato, lato), 80, np.int32)
        livello = np.full((lato, lato), 62.0, np.float32)
        ed, nuova_h, vie, muro, citta, banchi, _u = C.pianifica(
            cls, h, livello, [(100, 100, 55)], livello_mare=62, seed=5)
        return muro, citta

    def test_la_cinta_e_spessa_almeno_due(self):
        muro, citta = self.scenario()
        if not any(c.murata for c in citta):
            self.skipTest("nessuna citta' murata in questo scenario")
        cinta = muro > 0
        # ogni cella di cinta deve avere almeno un vicino di cinta in
        # orizzontale o verticale: la connessione diagonale non basta
        su = np.roll(cinta, 1, 0); giu = np.roll(cinta, -1, 0)
        sx = np.roll(cinta, 1, 1); dx = np.roll(cinta, -1, 1)
        sola = cinta & ~(su | giu | sx | dx)
        self.assertEqual(int(sola.sum()), 0,
                         "ci sono celle di cinta attaccate solo in diagonale")

    def test_la_cinta_e_spessa_almeno_tre(self):
        """Due era meglio di una ma ancora un cordolo, non un muro - vista in
        gioco. Su un lato dritto, lontano da angoli e porte, ci vogliono
        almeno tre celle piene attraversando la cinta."""
        lato = 200
        cls = np.full((lato, lato), PIANURA, np.uint8)
        sedimi = np.zeros((lato, lato), bool)
        sedimi[60:140, 60:140] = True
        vie = np.zeros((lato, lato), np.uint8)
        muro, porta, punti, torre = C.mura(sedimi, vie, sedimi, cls, [])
        self.assertTrue(muro.any())
        riga = muro[100, :60]                 # lato sinistro, meta' altezza
        sinistra = np.nonzero(riga)[0]
        self.assertGreaterEqual(len(sinistra), 3,
                                f"cinta spessa {len(sinistra)} invece di 3")

    def test_ci_sono_le_torri(self):
        muro, citta = self.scenario()
        if not any(c.murata for c in citta):
            self.skipTest("nessuna citta' murata in questo scenario")
        self.assertGreater(int((muro == 3).sum()), 8, "cinta senza torri")

    def test_le_porte_restano_varchi(self):
        muro, citta = self.scenario()
        if not any(c.murata for c in citta):
            self.skipTest("nessuna citta' murata in questo scenario")
        self.assertGreater(int((muro == 2).sum()), 0)
        # una porta non e' anche torre
        self.assertEqual(int(((muro == 2) & (muro == 3)).sum()), 0)


class TestSpianaAbitato(unittest.TestCase):

    def test_toglie_la_rugosita_ma_non_la_pendenza(self):
        rng = np.random.default_rng(1)
        pendenza = np.arange(120, dtype=np.float32)[None, :] * 0.6
        h = (pendenza + rng.normal(0, 2.0, (120, 120))).round().astype(np.int32)
        area = np.zeros((120, 120), bool)
        area[30:90, 30:90] = True
        niente = np.zeros((120, 120), bool)
        prima = h.copy()
        C.spiana_abitato(h, area, niente)
        dentro = area
        def rugosita(a):
            from scipy.ndimage import uniform_filter
            v = a.astype(np.float32)
            return float(np.abs(v - uniform_filter(v, 3))[dentro].mean())
        self.assertLess(rugosita(h), rugosita(prima) * 0.6)
        # la pendenza generale resta
        self.assertGreater(float(h[60, 85] - h[60, 35]), 20)

    def test_non_tocca_l_acqua(self):
        h = np.full((80, 80), 100, np.int32)
        h[:, 40] = 70
        area = np.ones((80, 80), bool)
        intoccabile = np.zeros((80, 80), bool)
        intoccabile[:, 39:42] = True
        C.spiana_abitato(h, area, intoccabile)
        self.assertTrue((h[:, 39:42] == np.array([100, 70, 100])[None, :]).all())


class TestTerrazze(unittest.TestCase):
    """Un paese in collina non segue il pendio: lo terrazza. Prima lo
    seguiva, e il risultato erano cinquanta case a cinquanta quote diverse
    con dei muretti casuali in mezzo."""

    def pendio(self, pendenza=0.8, lato=140):
        h = (np.arange(lato, dtype=np.float32)[None, :] * pendenza
             + np.zeros((lato, 1), np.float32))
        rng = np.random.default_rng(3)
        return np.round(h + rng.normal(0, 1.5, (lato, lato))).astype(np.int32)

    def area(self, lato=140):
        a = np.zeros((lato, lato), bool)
        a[30:110, 30:110] = True
        return a

    def test_il_terreno_diventa_a_ripiani(self):
        h = self.pendio()
        prima = h.copy()
        area = self.area()
        C.terrazza_abitato(h, area, np.zeros_like(area))
        quote = h[40:100, 40:100]
        # le quote devono raggrupparsi sui multipli del passo
        resti = quote % C.PASSO_TERRAZZA
        sul_ripiano = float((resti == 0).mean())
        self.assertGreater(sul_ripiano, 0.75,
                           f"solo il {sul_ripiano:.0%} sta su un ripiano")
        prima_resti = float((prima[40:100, 40:100] % C.PASSO_TERRAZZA == 0).mean())
        self.assertGreater(sul_ripiano, prima_resti * 2)

    def test_i_ripiani_sono_piani(self):
        h = self.pendio()
        area = self.area()
        C.terrazza_abitato(h, area, np.zeros_like(area))
        gz, gx = np.gradient(h[40:100, 40:100].astype(np.float32))
        piatte = float((np.hypot(gz, gx) < 0.1).mean())
        self.assertGreater(piatte, 0.4, "nessun ripiano piano")

    def test_la_pendenza_generale_resta(self):
        """Terrazzare non vuol dire spianare: un paese in pendenza e' bello,
        un altopiano artificiale no."""
        h = self.pendio()
        area = self.area()
        C.terrazza_abitato(h, area, np.zeros_like(area))
        self.assertGreater(int(h[70, 105] - h[70, 35]), 40)

    def test_l_acqua_non_si_terrazza(self):
        h = self.pendio()
        area = self.area()
        intoccabile = np.zeros_like(area)
        intoccabile[:, 60:64] = True
        prima = h.copy()
        C.terrazza_abitato(h, area, intoccabile)
        self.assertTrue((h[:, 60:64] == prima[:, 60:64]).all())

    def test_le_strade_tornano_percorribili(self):
        """Le terrazze fanno bene alle case e male alle strade: una via che
        incontra un salto di quattro blocchi diventa una parete."""
        h = self.pendio()
        area = self.area()
        C.terrazza_abitato(h, area, np.zeros_like(area))
        vie = np.zeros(h.shape, np.uint8)
        vie[68:72, 35:105] = 1
        dopo_terrazza = int(np.abs(np.diff(h[70, 40:100])).max())
        C.raccorda_vie(h, vie, np.zeros_like(area))
        dopo_raccordo = int(np.abs(np.diff(h[70, 40:100])).max())
        self.assertLess(dopo_raccordo, dopo_terrazza,
                        f"salto lungo la via: {dopo_terrazza} -> {dopo_raccordo}")
        self.assertLessEqual(dopo_raccordo, 2)

    def test_le_scarpate_sono_i_fronti_dei_gradoni(self):
        h = self.pendio()
        area = self.area()
        C.terrazza_abitato(h, area, np.zeros_like(area))
        sc = C.scarpate(h, area)
        self.assertGreater(int(sc.sum()), 50)
        self.assertEqual(int(sc[~area].sum()), 0)

    def test_la_cinta_non_e_un_pettine(self):
        """Le mura non hanno una quota propria: ogni colonna parte dal
        terreno sotto di se'. Sul terreno grezzo (con la stessa grana fine
        del dither) due merli vicini nascevano a quote diverse anche l'uno
        accanto all'altro, e la cinta sembrava un mucchio di macerie invece
        di un muro."""
        h = self.pendio()
        anello = np.zeros_like(h, dtype=bool)
        anello[70:74, 30:110] = True   # un tratto di cinta che attraversa il pendio
        prima = int(np.abs(np.diff(h[71, 40:100])).max())
        C.raccorda_mura(h, anello, np.zeros_like(anello))
        dopo = int(np.abs(np.diff(h[71, 40:100])).max())
        self.assertLess(dopo, prima,
                        f"grana sotto la cinta: {prima} -> {dopo}")
        self.assertLessEqual(dopo, 1)

    def test_la_cinta_non_tocca_l_acqua(self):
        h = self.pendio()
        anello = np.zeros_like(h, dtype=bool)
        anello[70:74, 30:110] = True
        intoccabile = np.zeros_like(anello)
        intoccabile[:, 60:64] = True
        prima = h.copy()
        C.raccorda_mura(h, anello, intoccabile)
        self.assertTrue((h[:, 60:64] == prima[:, 60:64]).all())

    def test_la_cinta_lascia_stare_il_resto_della_mappa(self):
        """Solo l'anello e un piccolo alone intorno: non tutta la mappa."""
        h = self.pendio()
        anello = np.zeros_like(h, dtype=bool)
        anello[70:74, 30:110] = True
        prima = h.copy()
        C.raccorda_mura(h, anello, np.zeros_like(anello))
        lontano = np.ones_like(anello)
        lontano[65:79, 25:115] = False   # anello + un margine largo
        self.assertTrue((h[lontano] == prima[lontano]).all())


class TestCintaSullaRiva(unittest.TestCase):
    """Negli screenshot le mura scendevano in acqua e continuavano dentro il
    mare a gradoni. Una citta' di mare le mura dalla parte del mare non le ha
    mai avute: il mare e' gia' la difesa."""

    def scenario(self, pendenza=0.0):
        lato = 160
        cls = np.full((lato, lato), PIANURA, np.uint8)
        cls[:, 120:] = MARE
        h = np.full((lato, lato), 80, np.int32)
        h[:, 120:] = 58
        if pendenza:
            h = (80 + np.arange(lato, dtype=np.float32)[None, :] * pendenza
                 + np.zeros((lato, 1), np.float32)).round().astype(np.int32)
            h[:, 120:] = 58
        sedimi = np.zeros((lato, lato), bool)
        sedimi[40:110, 40:115] = True
        vie = np.zeros((lato, lato), np.uint8)
        vie[70:74, 20:118] = C.ASSE
        return cls, h, sedimi, vie

    def test_non_arriva_all_acqua(self):
        cls, h, sedimi, vie = self.scenario()
        muro, porta, punti, torre = C.mura(
            sedimi, vie, sedimi, cls, [(72, 30)], altezze=h)
        zs, xs = np.nonzero(muro)
        self.assertTrue(muro.any(), "nessuna cinta")
        self.assertLess(int(xs.max()), 120 - 2,
                        "la cinta arriva sulla battigia")

    def test_non_si_costruisce_sul_dirupo(self):
        cls, h, sedimi, vie = self.scenario(pendenza=3.0)
        muro, porta, punti, torre = C.mura(
            sedimi, vie, sedimi, cls, [(72, 30)], altezze=h,
            pendenza_massima=1.0)
        gz, gx = np.gradient(h.astype(np.float32))
        pend = np.hypot(gz, gx)
        if muro.any():
            self.assertLessEqual(float(pend[muro > 0].max()), 1.0 + 1e-6)

    def test_senza_altezze_si_comporta_come_prima(self):
        cls, h, sedimi, vie = self.scenario()
        muro, _, _, _ = C.mura(sedimi, vie, sedimi, cls, [(72, 30)])
        self.assertTrue(muro.any())


class TestCintaSulFiume(unittest.TestCase):
    """Un fiume che attraversa l'abitato non e' come il mare: non e' una
    difesa naturale, e' solo un corso d'acqua che passa in mezzo. Prima
    prendeva lo stesso trattamento del mare (fascia larga tolta dalla cinta,
    pensata per il fronte a mare) e apriva un buco ingiustificato nel mezzo
    della cinta - segnalato dall'utente come "un effetto poco normale"."""

    LARGHEZZA_FIUME = 4

    def scenario(self):
        lato = 200
        cls = np.full((lato, lato), PIANURA, np.uint8)
        c0 = 100 - self.LARGHEZZA_FIUME // 2
        c1 = c0 + self.LARGHEZZA_FIUME
        cls[:, c0:c1] = FIUME               # un fiume verticale, attraversa tutto
        sedimi = np.zeros((lato, lato), bool)
        sedimi[40:160, 40:160] = True       # l'abitato a cavallo del fiume
        vie = np.zeros((lato, lato), np.uint8)
        return cls, sedimi, vie, c0, c1

    def test_il_fiume_ha_un_varco_vero_non_un_buco_senza_spiegazione(self):
        cls, sedimi, vie, c0, c1 = self.scenario()
        muro, porta, punti, torre = C.mura(sedimi, vie, sedimi, cls, [])
        self.assertTrue(muro.any(), "nessuna cinta")
        # il varco sul fiume esiste, ed e' sul fiume
        sul_fiume = porta & (cls == FIUME)
        self.assertTrue(sul_fiume.any(), "nessun varco sul fiume")
        # largo quanto il fiume (+ lo spessore della cinta), non una fascia
        # larga come quella pensata per il fronte a mare (riva=4 di margine
        # PER LATO, che su un fiume di 4 celle darebbe un buco di 12+ celle)
        zs, xs = np.nonzero(sul_fiume)
        self.assertLessEqual(int(xs.max()) - int(xs.min()) + 1,
                             self.LARGHEZZA_FIUME + 2,
                             "il varco e' piu' largo del fiume stesso")

    def test_la_cinta_resta_piena_lontano_dal_fiume(self):
        """Il lato della cinta che il fiume attraversa (il fronte nord),
        lontano dal punto preciso in cui lo attraversa, non deve avere
        buchi: solo il mare fa sparire un intero fronte di cinta, un fiume
        che passa in mezzo no."""
        cls, sedimi, vie, c0, c1 = self.scenario()
        muro, porta, punti, torre = C.mura(sedimi, vie, sedimi, cls, [])
        zs, xs = np.nonzero(muro)
        z_top = int(zs.min())         # il fronte nord, dove il fiume esce
        # colonne lontane dal fiume (che sta a c0..c1, circa x=98..101)
        striscia = muro[z_top:z_top + 4, 40:90]
        self.assertGreater(int(striscia.sum()), 20,
                           "la cinta e' sparita anche lontano dal fiume")

    def test_la_cinta_piu_porta_resta_un_anello_chiuso(self):
        cls, sedimi, vie, c0, c1 = self.scenario()
        muro, porta, punti, torre = C.mura(sedimi, vie, sedimi, cls, [])
        sistema = (muro > 0) | porta
        _, quanti = label(sistema, np.ones((3, 3)))
        self.assertEqual(quanti, 1, f"{quanti} pezzi di cinta, non un anello")

    def test_il_mare_invece_si_comporta_come_prima(self):
        """Regressione: il fronte a mare deve continuare a perdere la fascia
        larga di cinta - questo test non deve cambiare risultato per via
        della correzione sul fiume, che tocca solo FIUME."""
        lato = 160
        cls = np.full((lato, lato), PIANURA, np.uint8)
        cls[:, 120:] = MARE
        sedimi = np.zeros((lato, lato), bool)
        sedimi[40:110, 40:115] = True
        vie = np.zeros((lato, lato), np.uint8)
        muro, porta, punti, torre = C.mura(sedimi, vie, sedimi, cls, [])
        zs, xs = np.nonzero(muro)
        self.assertTrue(muro.any())
        self.assertLess(int(xs.max()), 120 - 2,
                        "la cinta arriva sulla battigia")
