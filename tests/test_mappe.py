"""Mappe del tesoro (#7): colori, file della mappa, oggetti nei forzieri."""
import gzip
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import bauli as BA  # noqa: E402
from genworld import luoghi as LU  # noqa: E402
from genworld import mappa as M  # noqa: E402
from genworld import mappe as MP  # noqa: E402

META = 100


def terreno(lato=200):
    cls = np.full((lato, lato), M.PRATERIA, np.uint8)
    cls[:, :40] = M.OCEANO
    cls[:, 150:] = M.MONTAGNA
    h = np.full((lato, lato), 70, np.int32)
    h[:, :40] = 40
    h[:, 150:] = np.arange(lato - 150) * 2 + 80
    liv = np.full((lato, lato), 62.0, np.float32)
    return cls, h, liv


def px(colori, i, j):
    return int(np.asarray(colori).view(np.uint8)[j * 128 + i])


class TestDisegna(unittest.TestCase):
    def setUp(self):
        self.cls, self.h, self.liv = terreno()

    def test_dimensione_e_tipo(self):
        c = MP.disegna(self.cls, self.h, self.liv, META, 0, 0)
        self.assertEqual(c.shape, (16384,))
        self.assertEqual(c.dtype, np.int8)

    def test_i_colori_stanno_nella_tavolozza(self):
        c = MP.disegna(self.cls, self.h, self.liv, META, 0, 0).view(np.uint8)
        self.assertLessEqual(int(c.max()), 61 * 4 + 3)

    def test_classi_e_acqua(self):
        # centro in mappa (100, 100): erba; a ovest 40 celle di oceano
        c = MP.disegna(self.cls, self.h, self.liv, META, 0, 0)
        self.assertEqual(px(c, 64, 64) // 4, MP.ERBA)
        # x = -90 (mappa 10) e' oceano: pixel i = 64 + (-90)/4 ~ 41
        self.assertEqual(px(c, 41, 64) // 4, MP.ACQUA)
        # x = 70 (mappa 170) e' montagna: i = 64 + 70/4 ~ 81
        self.assertEqual(px(c, 81, 64) // 4, MP.PIETRA)

    def test_fuori_dal_mondo_e_trasparente(self):
        c = MP.disegna(self.cls, self.h, self.liv, META, -200, 0)
        self.assertEqual(px(c, 0, 0), 0)

    def test_acqua_profonda_piu_scura(self):
        cls, h, liv = terreno()
        h[:, :20] = 20            # fondale profondo
        h[:, 20:40] = 60          # basso
        c = MP.disegna(cls, h, liv, META, -80, 0)
        # x = -100 -> mappa 0..19 profondo; x = -70 -> mappa 30 basso
        prof = px(c, 64 + (-95 + 80) // 4, 64) % 4
        basso = px(c, 64 + (-65 + 80) // 4, 64) % 4
        self.assertLess(prof, basso)

    def test_la_croce_segna_il_destino(self):
        c = MP.disegna(self.cls, self.h, self.liv, META, 10, 10, destino=(10, 10))
        croce = MP.colore(MP.FUOCO, 2)
        self.assertEqual(px(c, 64, 64), croce)
        self.assertEqual(px(c, 64 + 3, 64 + 3), croce)
        self.assertEqual(px(c, 64 - 3, 64 + 3), croce)
        # lontano dalla croce non c'e' rosso
        self.assertNotEqual(px(c, 64 + 20, 64), croce)

    def test_il_punto_di_partenza(self):
        c = MP.disegna(self.cls, self.h, self.liv, META, 0, 0, qui=(0, 0))
        self.assertEqual(px(c, 64, 64) // 4, MP.NEVE)
        self.assertEqual(px(c, 63, 64) // 4, MP.NERO)

    def test_case_e_strade(self):
        class Ed:
            x, z, larghezza, profondita = 98, 98, 8, 8
        strade = np.zeros((200, 200), np.uint8)
        strade[96:108, 120:] = 1
        c = MP.disegna(self.cls, self.h, self.liv, META, 0, 0, [Ed()], strade)
        self.assertEqual(px(c, 64, 64) // 4, MP.LEGNO)
        self.assertEqual(px(c, 64 + 8, 64) // 4, MP.TERRA)

    def test_deterministica(self):
        a = MP.disegna(self.cls, self.h, self.liv, META, 5, -5, destino=(5, -5))
        b = MP.disegna(self.cls, self.h, self.liv, META, 5, -5, destino=(5, -5))
        self.assertTrue(np.array_equal(a, b))


class TestFile(unittest.TestCase):
    def leggi(self, percorso):
        import amulet_nbt
        return amulet_nbt.load(percorso).compound

    def test_nuovo_formato(self):
        with tempfile.TemporaryDirectory() as d:
            colori = np.arange(16384, dtype=np.int64).astype(np.uint8).view(np.int8)
            p = MP.scrivi_file(d, 3, colori, 40, -60, 4800)
            self.assertTrue(p.endswith(os.path.join("data", "minecraft", "maps", "3.dat")))
            t = self.leggi(p)
            self.assertEqual(int(t["DataVersion"]), 4800)
            dati = t["data"]
            self.assertEqual(int(dati["scale"]), MP.SCALA)
            self.assertEqual(str(dati["dimension"]), "minecraft:overworld")
            self.assertEqual((int(dati["xCenter"]), int(dati["zCenter"])), (40, -60))
            self.assertEqual(int(dati["locked"]), 1)
            self.assertEqual(len(dati["colors"]), 16384)
            self.assertTrue(np.array_equal(np.asarray(dati["colors"]), colori))

    def test_vecchio_formato(self):
        with tempfile.TemporaryDirectory() as d:
            p = MP.scrivi_file(d, 2, np.zeros(16384, np.int8), 0, 0, 4000)
            self.assertTrue(p.endswith(os.path.join("data", "map_2.dat")))

    def test_contatore(self):
        with tempfile.TemporaryDirectory() as d:
            MP.scrivi_ultimo(d, 7, 4800)
            t = self.leggi(os.path.join(d, "data", "minecraft", "last_id.dat"))
            self.assertEqual(int(t["data"]["map"]), 7)
            self.assertTrue(os.path.exists(os.path.join(d, "data", "last_id.dat")))
        with tempfile.TemporaryDirectory() as d:
            MP.scrivi_ultimo(d, 4, 4000)
            t = self.leggi(os.path.join(d, "data", "idcounts.dat"))
            self.assertEqual(int(t["data"]["map"]), 4)

    def test_i_file_sono_gzip(self):
        with tempfile.TemporaryDirectory() as d:
            p = MP.scrivi_file(d, 1, np.zeros(16384, np.int8), 0, 0, 4800)
            with open(p, "rb") as f:
                self.assertEqual(f.read(2), b"\x1f\x8b")
                f.seek(0)
                gzip.decompress(f.read())


class TestPianifica(unittest.TestCase):
    def luoghi(self):
        return [LU.Luogo("Aldona", "citta", 0, 0),
                LU.Luogo("Torre A", "torre", 100, 0), LU.Luogo("Cripta B", "rovina", 250, 50),
                LU.Luogo("Rocca C", "castello", -300, 200)]

    def sedi(self):
        return {("torre", 100, 0): (101, 70, 2), ("rovina", 250, 50): (251, -20, 52),
                ("castello", -300, 200): (-299, 60, 201)}

    def test_una_mappa_per_tappa_con_successiva(self):
        m = MP.pianifica(self.luoghi(), self.sedi(), META)
        # tappe in ordine: Torre A, Cripta B, Rocca C -> due mappe (l'ultima non ha successiva)
        self.assertEqual(len(m), 2)
        self.assertEqual([x["numero"] for x in m], [1, 2])
        self.assertEqual(m[0]["chiave"], (101, 70, 2))
        self.assertEqual(m[0]["destino"], (250, 50))
        self.assertEqual(m[0]["luogo"], "Cripta B")
        self.assertEqual(m[1]["destino"], (-300, 200))
        self.assertEqual(m[0]["oggetto"]["id"], "filled_map")
        self.assertEqual(m[0]["oggetto"]["map_id"], 1)

    def test_senza_forziere_non_c_e_mappa(self):
        sedi = self.sedi()
        del sedi[("torre", 100, 0)]
        m = MP.pianifica(self.luoghi(), sedi, META)
        self.assertEqual([x["luogo"] for x in m], ["Rocca C"])

    def test_scrivi_tutte(self):
        cls, h, liv = terreno(1000)
        with tempfile.TemporaryDirectory() as d:
            m = MP.pianifica(self.luoghi(), self.sedi(), 500)
            n = MP.scrivi_tutte(d, m, cls, h, liv, 500, 4800)
            self.assertEqual(n, 2)
            for k in (1, 2):
                self.assertTrue(os.path.exists(os.path.join(d, "data", "minecraft", "maps", f"{k}.dat")))
            self.assertTrue(os.path.exists(os.path.join(d, "data", "minecraft", "last_id.dat")))
        self.assertEqual(MP.scrivi_tutte("/non/si/scrive", [], cls, h, liv, 500, 4800), 0)


class TestOggetto(unittest.TestCase):
    def test_componenti_con_il_numero_della_mappa(self):
        spec = MP.oggetto_mappa(5, "Cripta B", "mappa5", "a circa 150 passi a est")
        comp = BA.componenti_oggetto(spec)
        self.assertEqual(comp["map_id"], 5)
        self.assertEqual(comp["custom_name"], "Mappa: Cripta B")
        self.assertEqual(len(comp["lore"]), 2)

    def test_la_voce_del_forziere_ha_il_componente(self):
        spec = MP.oggetto_mappa(5, "Cripta B", "mappa5")
        voce = BA._voce((3, spec["id"], 1, BA.componenti_oggetto(spec)))
        self.assertEqual(str(voce["id"]), "minecraft:filled_map")
        self.assertEqual(int(voce["components"]["minecraft:map_id"]), 5)


if __name__ == "__main__":
    unittest.main()
