"""Verifica degli edifici e della pianificazione dei villaggi."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import edifici as E
from genworld import insediamenti as I
from genworld import mappa as M


class FintaTavolozza:
    """Id finti: la costruzione non deve dipendere da un livello aperto."""
    ARIA = 0

    def __init__(self):
        self.aria = self.ARIA
        self.blocco = {}
        for i, stile in enumerate(E.PALETTE):
            for j, ruolo in enumerate(("muro", "telaio", "pavimento",
                                       "basamento", "palo")):
                self.blocco[(stile, ruolo)] = 10 + i * 10 + j
        self.vetro, self.vetro_pieno, self.torcia = 90, 91, 92

    # gli arredi hanno id propri, cosi' un test puo' cercarli
    ARREDO = {"ladder": 60, "bed": 61, "furnace": 62, "crafting_table": 63,
              "chest": 64, "bookshelf": 65}

    def scala(self, materiale, verso, meta="bottom"): return 80
    def porta(self, materiale, verso, meta): return 70 if meta == "lower" else 71
    def staccionata(self, materiale): return 72
    def trave(self, materiale, asse): return 73 if asse == "x" else 74

    def b(self, nome, **proprieta): return self.ARREDO.get(nome, 69)


def casa(**kw):
    base = dict(x=4, z=4, larghezza=9, profondita=9, base=70, piani=1, seme=1)
    base.update(kw)
    return E.Edificio(**base)


class TestGeometria(unittest.TestCase):

    def test_estremi(self):
        ed = casa()
        self.assertEqual(ed.x1, 13)
        self.assertEqual(ed.z1, 13)

    def test_ingombro_copre_la_casa(self):
        ed = casa()
        x0, z0, x1, z1 = E.ingombro(ed, margine=2)
        self.assertLessEqual(x0, ed.x)
        self.assertGreaterEqual(x1, ed.x1)

    def test_colmo_sopra_i_muri(self):
        ed = casa(piani=2)
        self.assertGreater(ed.colmo, ed.base + 2 * ed.altezza_piano)


class TestCostruzione(unittest.TestCase):

    def volume(self, ed, lato_chunk=(0, 0)):
        out = np.zeros((16, 200, 16), np.uint32)
        E.costruisci(out, 0, lato_chunk[0], lato_chunk[1], ed, FintaTavolozza())
        return out

    def test_muri_presenti_e_interno_vuoto(self):
        ed = casa(x=2, z=2, larghezza=9, profondita=9, base=70)
        out = self.volume(ed)
        y = ed.base + 1
        self.assertNotEqual(int(out[ed.x, y, ed.z + 4]), 0, "manca il muro")
        self.assertEqual(int(out[ed.x + 4, y, ed.z + 4]), 0, "interno non vuoto")

    def test_porta_sul_lato_giusto(self):
        ed = casa(x=2, z=2, base=70, porta=E.NORD)
        out = self.volume(ed)
        cx = (ed.x + ed.x1 - 1) // 2
        self.assertEqual(int(out[cx, ed.base, ed.z]), 70)
        self.assertEqual(int(out[cx, ed.base + 1, ed.z]), 71)

    def test_l_arredo_sta_dentro_i_muri(self):
        """Un letto che spunta dal muro e' peggio di nessun letto."""
        ed = casa(x=2, z=2, larghezza=9, profondita=9, base=70, piani=2)
        out = self.volume(ed)
        arredi = set(FintaTavolozza.ARREDO.values())
        trovati = 0
        for lx in range(out.shape[0]):
            for ly in range(out.shape[1]):
                for lz in range(out.shape[2]):
                    if int(out[lx, ly, lz]) in arredi:
                        trovati += 1
                        self.assertTrue(ed.x < lx < ed.x1 - 1,
                                        f"arredo fuori in x: {lx}")
                        self.assertTrue(ed.z < lz < ed.z1 - 1,
                                        f"arredo fuori in z: {lz}")
        self.assertGreater(trovati, 4, "casa ancora vuota")

    def test_il_piano_terra_ha_il_focolare(self):
        ed = casa(x=2, z=2, larghezza=9, profondita=9, base=70, piani=1)
        out = self.volume(ed)
        y = ed.base
        presenti = {int(v) for v in out[:, y, :].ravel()}
        for nome in ("furnace", "crafting_table", "chest"):
            self.assertIn(FintaTavolozza.ARREDO[nome], presenti, f"manca {nome}")

    def test_due_piani_hanno_la_scala_e_il_buco(self):
        """Senza il buco nel solaio la scala porta contro un soffitto."""
        ed = casa(x=2, z=2, larghezza=9, profondita=9, base=70, piani=2)
        out = self.volume(ed)
        scala = FintaTavolozza.ARREDO["ladder"]
        colonna = out[ed.x + 1, ed.base:ed.base + ed.altezza_piano, ed.z + 1]
        self.assertTrue((colonna == scala).all(), "scala incompleta")
        solaio_y = ed.base + ed.altezza_piano - 1
        self.assertEqual(int(out[ed.x + 1, solaio_y, ed.z + 1]), scala,
                         "il solaio ha tappato il vano scala")

    def test_un_solo_piano_non_ha_scale(self):
        ed = casa(x=2, z=2, larghezza=9, profondita=9, base=70, piani=1)
        out = self.volume(ed)
        self.assertEqual(int((out == FintaTavolozza.ARREDO["ladder"]).sum()), 0)

    def test_tetto_sopra_i_muri(self):
        ed = casa(x=2, z=2, base=70, piani=1)
        out = self.volume(ed)
        cima = ed.base + ed.piani * ed.altezza_piano
        self.assertGreater(int((out[:, cima:cima + 6, :] == 80).sum()), 8,
                           "nessuna scala di tetto")

    def test_palafitta_pianta_i_pali(self):
        """I pali devono scendere fino al fondale, non fermarsi al pelo."""
        ed = casa(x=2, z=2, base=70, palafitta=True, fondale=58)
        out = self.volume(ed)
        tav = FintaTavolozza()
        palo = tav.blocco[(ed.stile, "palo")]
        colonna = out[ed.x, 58:70, ed.z]
        self.assertGreater(int((colonna == palo).sum()), 8,
                           f"pali mancanti sotto la piattaforma: {colonna}")

    def test_palafitta_senza_basamento_sommerso(self):
        ed = casa(base=70, palafitta=True, fondale=58)
        out = self.volume(ed)
        tav = FintaTavolozza()
        basamento = tav.blocco[(ed.stile, "basamento")]
        self.assertEqual(int((out[:, 58:69, :] == basamento).sum()), 0)

    def test_non_sfora_il_chunk(self):
        """Una casa a cavallo del bordo non deve dare errori di indice."""
        ed = casa(x=12, z=12, larghezza=9, profondita=9, base=70)
        out = self.volume(ed)          # chunk (0,0): la casa esce a destra
        self.assertEqual(out.shape, (16, 200, 16))
        self.assertGreater(int((out != 0).sum()), 0)

    def test_compare_nei_chunk_vicini(self):
        """L'invariante dell'indice: una casa larga tocca piu' chunk."""
        ed = casa(x=12, z=4, larghezza=9, profondita=9, base=70)
        sinistra = self.volume(ed, (0, 0))
        destra = self.volume(ed, (16, 0))
        self.assertGreater(int((sinistra != 0).sum()), 0)
        self.assertGreater(int((destra != 0).sum()), 0)


class TestIndice(unittest.TestCase):

    def test_registra_tutti_i_chunk_toccati(self):
        ed = casa(x=12, z=12, larghezza=10, profondita=10)
        idx = I.indice_per_chunk([ed])
        self.assertIn((0, 0), idx)
        self.assertIn((1, 1), idx)


class TestPianificazione(unittest.TestCase):

    def scenario(self, n=300):
        cls = np.full((n, n), M.PRATERIA, np.uint8)
        cls[:, :60] = M.OCEANO
        h = np.full((n, n), 80, np.int32)
        h[:, :60] = 45
        liv = np.full((n, n), 62.0, np.float32)
        return cls, h, liv

    def test_siti_separati(self):
        cls, h, liv = self.scenario()
        siti = I.scegli_siti(cls, h, separazione=80, celle_per_villaggio=8000, seed=2)
        self.assertGreater(len(siti), 1)
        for i, (z1, x1, _) in enumerate(siti):
            for z2, x2, _ in siti[i + 1:]:
                self.assertGreaterEqual((z1 - z2) ** 2 + (x1 - x2) ** 2, 80 ** 2 * 0.9)

    def test_siti_solo_su_terra(self):
        cls, h, liv = self.scenario()
        for z, x, _ in I.scegli_siti(cls, h, celle_per_villaggio=8000, seed=3):
            self.assertNotIn(int(cls[z, x]), list(M.ACQUA))

    def test_edifici_sopra_il_mare(self):
        cls, h, liv = self.scenario()
        ed, h2, n = I.pianifica(cls, h, liv, seed=4)
        self.assertGreater(len(ed), 0)
        for e in ed:
            self.assertGreater(e.base, 62)

    def test_terrazzamento_spiana(self):
        z, x = np.meshgrid(np.arange(80), np.arange(80), indexing="ij")
        h = (70 + x // 4).astype(np.int32)
        ed = casa(x=30, z=30, larghezza=9, profondita=9, base=int(h[34, 34]))
        prima = h[30:39, 30:39].copy()
        I._terrazza(h, ed)
        dopo = h[ed.z:ed.z1, ed.x:ed.x1]
        self.assertGreater(int(prima.max() - prima.min()), 0)
        self.assertEqual(int(dopo.max() - dopo.min()), 0, "il lotto non e' piano")

    def test_scala_zero_nessun_villaggio(self):
        cls, h, liv = self.scenario()
        ed, h2, n = I.pianifica(cls, h, liv, scala=0.0, seed=5)
        self.assertEqual(len(ed), 0)
        self.assertTrue(np.array_equal(h2, h))


if __name__ == "__main__":
    unittest.main()


class TestTettoADueFalde(unittest.TestCase):
    """Il tetto era a padiglione - una piramide - e sta bene su una villa,
    non su una casa di paese. Questi test tengono ferme le tre cose che
    rendono un tetto a due falde un tetto e non un mucchio di scale."""

    def costruisci(self, **kw):
        ed = casa(x=2, z=2, larghezza=11, profondita=8, base=8, piani=2,
                  gronda=1, **kw)
        out = np.zeros((16, 40, 16), np.int32)
        out[:, :8, :] = 1
        E.costruisci(out, 0, 0, 0, ed, FintaTavolozza())
        return ed, out

    def test_c_e_un_colmo_e_non_una_scanalatura(self):
        """Con la campata pari le due falde si incontrano su due file e in
        cima resta un solco lungo quanto la casa."""
        ed, out = self.costruisci()
        cima = ed.base + ed.piani * ed.altezza_piano
        colmo = cima - ed.gronda + ed.falde - 1
        riga = out[3:12, colmo, :]
        self.assertTrue((riga != 0).any(), "niente di niente sul colmo")
        # il colmo e' una trave, non due scale che si scontrano
        self.assertIn(73, set(out[:, colmo, :].ravel().tolist()))

    def test_niente_feritoia_fra_muro_e_falda(self):
        """La falda partiva sopra il filo di gronda e restava aperta una
        fessura lungo tutti e due i lati lunghi: da dentro si vedeva il
        cielo."""
        ed, out = self.costruisci()
        cima = ed.base + ed.piani * ed.altezza_piano
        # sul filo del muro lungo, appena sopra l'ultimo corso, ci deve
        # essere il tetto
        for gx in range(ed.x + 1, ed.x1 - 1):
            self.assertNotEqual(int(out[gx, cima, ed.z]), 0,
                                f"buco sopra il muro in x={gx}")

    def test_i_timpani_sono_murati(self):
        """Le testate sotto la falda sono muro, non aria: senza, la casa ha
        due pareti triangolari mancanti."""
        ed, out = self.costruisci()
        cima = ed.base + ed.piani * ed.altezza_piano
        muro = FintaTavolozza().blocco[(ed.stile, "muro")]
        testata = out[ed.x, cima:cima + ed.falde, ed.z + 1:ed.z1 - 1]
        self.assertGreater(int((testata == muro).sum()), 3)

    def test_la_campata_del_tetto_e_dispari(self):
        for larg, prof, g in ((11, 8, 1), (9, 9, 0), (8, 10, 1), (7, 7, 1)):
            ed = casa(larghezza=larg, profondita=prof, gronda=g)
            corto = min(larg, prof) + 2 * g
            atteso = (corto - 1 if corto % 2 == 0 else corto) // 2 + 1
            self.assertEqual(ed.falde, atteso, (larg, prof, g))


class TestGraticcioEFondazione(unittest.TestCase):

    def test_la_facciata_non_e_una_parete_liscia(self):
        ed = casa(x=2, z=2, larghezza=11, profondita=8, base=8, piani=1)
        out = np.zeros((16, 40, 16), np.int32)
        out[:, :8, :] = 1
        tav = FintaTavolozza()
        E.costruisci(out, 0, 0, 0, ed, tav)
        facciata = out[ed.x:ed.x1, ed.base:ed.base + ed.altezza_piano, ed.z]
        self.assertGreaterEqual(len(set(facciata.ravel().tolist())), 4,
                                "la facciata e' fatta di un materiale solo")

    def test_le_finestre_sono_alte_due(self):
        ed = casa(x=2, z=2, larghezza=11, profondita=8, base=8, piani=1)
        out = np.zeros((16, 40, 16), np.int32)
        out[:, :8, :] = 1
        tav = FintaTavolozza()
        E.costruisci(out, 0, 0, 0, ed, tav)
        colonne = [gx for gx in range(ed.x + 1, ed.x1 - 1)
                   if out[gx, ed.base + 1, ed.z] == tav.vetro
                   and out[gx, ed.base + 2, ed.z] == tav.vetro]
        self.assertGreaterEqual(len(colonne), 2)

    def test_la_casa_non_resta_appesa_sul_bordo(self):
        """Il raccordo del lotto lascia il bordo piu' basso: con un solo
        strato di basamento la casa appoggiava sull'aria."""
        ed = casa(x=2, z=2, larghezza=9, profondita=9, base=10, piani=1)
        out = np.zeros((16, 40, 16), np.int32)
        out[:, :7, :] = 1                  # terreno tre blocchi piu' in basso
        tav = FintaTavolozza()
        E.costruisci(out, 0, 0, 0, ed, tav)
        basamento = tav.blocco[(ed.stile, "basamento")]
        for gy in range(7, ed.base):
            self.assertEqual(int(out[ed.x + 4, gy, ed.z + 4]), basamento,
                             f"vuoto sotto la casa a y={gy}")
