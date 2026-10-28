"""Ferrovie: tracciato sulle strade, forme dei binari, stazioni e locomotive."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import ferrovie as FE
from genworld import mappa as M
from genworld import siti as SI
from genworld import strade as ST
from genworld.citta import Citta
from genworld.motore import Analisi, Opzioni, pianifica


def citta(z, x, r=20):
    return Citta(z=z, x=x, raggio=r, piazza=(z, x), lato=r)


class TestPercorso(unittest.TestCase):
    def test_il_cammino_segue_le_celle_utili_e_non_taglia(self):
        utile = np.zeros((20, 40), bool)
        utile[10, :] = True
        q = np.full((20, 40), 70, np.int32)
        c = FE._percorso(utile, q, (10, 2), (10, 35))
        self.assertEqual(c[0], (10, 2))
        self.assertEqual(c[-1], (10, 35))
        self.assertEqual(len(c), 34)

    def test_nessun_cammino_se_manca_la_strada(self):
        utile = np.zeros((10, 10), bool)
        utile[5, :4] = True
        utile[5, 6:] = True
        self.assertIsNone(FE._percorso(utile, np.zeros((10, 10), np.int32), (5, 0), (5, 9)))

    def test_niente_curve_sulle_pendenze(self):
        utile = np.zeros((10, 10), bool)
        utile[5, :6] = True
        utile[:6, 5] = True
        q = np.full((10, 10), 70, np.int32)
        q[5, 4:] = 71                       # il gomito in (5, 5) sarebbe su una salita
        self.assertIsNone(FE._percorso(utile, q, (5, 0), (0, 5)))


class TestForme(unittest.TestCase):
    def test_rettilineo_curva_e_salita(self):
        q = np.full((10, 10), 70, np.int32)
        q[5, 3:] = 71
        cam = [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)]
        f = FE._forme(cam, q)
        self.assertEqual(f[0], "east_west")
        self.assertEqual(f[2], "ascending_east")
        self.assertEqual(f[3], "east_west")
        q2 = np.full((10, 10), 70, np.int32)
        f2 = FE._forme([(5, 5), (5, 6), (6, 6)], q2)
        self.assertEqual(f2[1], "south_west")           # collega ovest e sud


def scenario(lato=700):
    cls = np.full((lato, lato), M.PIANURA, np.uint8)
    h = np.full((lato, lato), 70, np.int32)
    return Analisi(cls=cls, h=h, livello=np.full((lato, lato), 62.0, np.float32),
                   lava=np.zeros((lato, lato), bool), colata=np.zeros((lato, lato), np.float32))


class TestPiano(unittest.TestCase):
    def piano(self, **kw):
        lato = 700
        reali = (("Alfa", 0.20, 0.30, True), ("Beta", 0.75, 0.35, True), ("Gamma", 0.50, 0.80, True))
        op = Opzioni(immagine="", uscita="mondi/_prova_ferrovie", lato=lato, citta_reali=reali,
                     solo_citta_reali=True, vulcani=0, miniere=0, dungeon=0, torri=0, fortezze=0,
                     castelli_rovina=0, isole=0, siti=0, accampamenti=0, cimiteri=0, portali=0,
                     incarichi=False, fauna=0, alberi=0, **kw)
        return pianifica(op, scenario(lato))

    def test_collega_le_citta_con_binari_e_stazioni(self):
        p = self.piano()
        self.assertIsNotNone(p.ferrovia)
        self.assertGreater(p.ferrovia.lunghezza, 100)
        staz = [s for s in p.siti if s.tipo == "stazione"]
        self.assertGreaterEqual(len(staz), 2)
        self.assertTrue(all(s.nome.startswith("Stazione di ") for s in staz))
        nomi = {s.nome for s in staz}
        self.assertTrue(nomi <= {"Stazione di Alfa", "Stazione di Beta", "Stazione di Gamma"}, nomi)

    def test_senza_ferrovie_non_c_e_niente(self):
        p = self.piano(ferrovie=0.0)
        self.assertIsNone(p.ferrovia)
        self.assertFalse([s for s in p.siti if s.tipo == "stazione"])

    def test_i_binari_stanno_sulla_sede_stradale(self):
        p = self.piano()
        meta = 350
        for (cx, cz), celle in p.ferrovia.rotaie.items():
            for lx, lz, y, forma, alim in celle[:20]:
                x, z = cx * 16 + lx + meta, cz * 16 + lz + meta
                self.assertIn(int(p.tipo_strada[z, x]), (ST.STRADA, ST.PONTE))
                self.assertEqual(y, int(p.quota_strada[z, x]))


class TestStazione(unittest.TestCase):
    def test_la_stazione_ha_locomotiva_carrelli_e_il_binario_morto(self):
        for asse in ("x", "z"):
            s = SI.Sito(tipo="stazione", x=100, z=100, y=70, seme=3, extra={"asse": asse})
            d = SI.disegna(s)
            chiavi = {p[7] for p in d.prim}
            self.assertIn("carbone", chiavi)
            self.assertIn("fuoco", chiavi)
            self.assertIn("ru" if False else ("rotaia:ew" if asse == "x" else "rotaia:ns"), chiavi)
            self.assertEqual(len(d.carrelli), 2)
            bb = s.bbox()
            for p in d.prim:                              # il disegno resta nel suo ingombro
                if p[0] == "B":
                    self.assertGreaterEqual(p[1], bb[0] - 1, (asse, p))
                    self.assertLessEqual(p[4], bb[2] + 1, (asse, p))
                    self.assertGreaterEqual(p[3], bb[1] - 1, (asse, p))
                    self.assertLessEqual(p[6], bb[3] + 1, (asse, p))
        self.assertEqual(len(SI.carrelli([s], 50)), 2)


class TestMondo(unittest.TestCase):
    def test_un_mondo_con_la_ferrovia_si_genera_e_ha_i_binari(self):
        import amulet
        from genworld.motore import genera
        with tempfile.TemporaryDirectory() as d:
            lato = 640
            reali = (("Alfa", 0.20, 0.30, True), ("Beta", 0.75, 0.35, True))
            e = np.full((lato, lato), 90.0, np.float32)
            dem = os.path.join(d, "dem.npz")
            np.savez_compressed(dem, elev=e, lago=np.zeros(e.shape, bool), x0=0.0, y1=0.0,
                                passo=1000.0, nx=lato, ny=lato, crs="EPSG:3035")
            op = Opzioni(immagine="", uscita=os.path.join(d, "w"), nome="T", lato=lato, dem=dem,
                         citta_reali=reali, solo_citta_reali=True, versione=(26, 2, 0), vulcani=0,
                         miniere=0, dungeon=0, torri=0, fortezze=0, castelli_rovina=0, isole=0,
                         siti=0, accampamenti=0, cimiteri=0, portali=0, incarichi=False, alberi=0,
                         caverne=0, laghi=0, fiumi=0, arredi=0, isolate=0, fauna=0,
                         dimensioni=False, fuori_mappa="vuoto")
            genera(op)
            lv = amulet.load_level(op.uscita)
            try:
                trovati = 0
                for cx in range(-20, 20):
                    for cz in range(-20, 20):
                        try:
                            ch = lv.get_chunk(cx, cz, "minecraft:overworld")
                        except Exception:
                            continue
                        for b in ch.block_palette:
                            if "rail" in str(b):
                                trovati += 1
                                break
                self.assertGreater(trovati, 5)
            finally:
                lv.close()


if __name__ == "__main__":
    unittest.main()
