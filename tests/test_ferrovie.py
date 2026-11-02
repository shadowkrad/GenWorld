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


def _arr(forma, riempi=True):
    return np.full(forma, riempi, bool)


class TestPercorso(unittest.TestCase):
    def cerca(self, passabile, q, a, b, strada=None):
        costo = np.ones(passabile.shape, np.float32)
        strada = np.zeros(passabile.shape, bool) if strada is None else strada
        return FE._percorso(passabile, costo, q, strada, a, b)

    def test_il_cammino_collega_i_due_punti(self):
        c = self.cerca(_arr((20, 40)), np.full((20, 40), 70, np.int32), (10, 2), (10, 35))
        self.assertEqual((c[0], c[-1]), ((10, 2), (10, 35)))
        self.assertEqual(len(c), 34)

    def test_aggira_gli_ostacoli(self):
        p = _arr((20, 40))
        p[:15, 20] = False                                  # un muro
        c = self.cerca(p, np.full((20, 40), 70, np.int32), (10, 2), (10, 35))
        self.assertTrue(all(p[z, x] for z, x in c))
        self.assertGreater(len(c), 34)

    def test_nessun_cammino_se_e_tutto_chiuso(self):
        p = _arr((10, 10))
        p[:, 5] = False
        self.assertIsNone(self.cerca(p, np.zeros((10, 10), np.int32), (5, 0), (5, 9)))

    def test_mai_un_dislivello_oltre_uno_e_niente_curve_in_pendenza(self):
        q = np.full((30, 30), 70, np.int32)
        q[:, 10:] = 71
        q[:, 20:] = 72
        c = self.cerca(_arr((30, 30)), q, (5, 2), (25, 28))
        self.assertIsNotNone(c)
        for (z1, x1), (z2, x2) in zip(c, c[1:]):
            self.assertLessEqual(abs(int(q[z2, x2]) - int(q[z1, x1])), 1)
        for i in range(1, len(c) - 1):
            d1 = (c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1])
            d2 = (c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1])
            if d1 != d2:                                    # una curva: piana prima e dopo
                self.assertEqual(q[c[i - 1]], q[c[i]])
                self.assertEqual(q[c[i + 1]], q[c[i]])

    def test_le_strade_si_attraversano_dritte(self):
        strada = np.zeros((20, 30), bool)
        strada[:, 14:17] = True
        c = self.cerca(_arr((20, 30)), np.full((20, 30), 70, np.int32), (3, 2), (15, 27), strada)
        for i in range(1, len(c) - 1):
            if strada[c[i]]:
                self.assertEqual((c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]),
                                 (c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]))


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

    def test_i_binari_stanno_lontani_da_strade_e_costruzioni(self):
        p = self.piano()
        meta = 350
        self.assertTrue(p.ferrovia.celle)
        for (z, x) in list(p.ferrovia.celle)[:400]:
            tipo = int(p.tipo_strada[z, x])
            # solo i passaggi a livello sono sulla strada; mai su ponti, parapetti, viali
            self.assertIn(tipo, (0, ST.STRADA))
            self.assertNotIn(int(p.cls[z, x]), M.ACQUA)
        for (cx, cz), celle in p.ferrovia.rotaie.items():
            for lx, lz, y, forma, alim, suolo in celle[:30]:
                self.assertLessEqual(abs(y - suolo), FE.SCAVO_MAX + 1)

    def test_il_corridoio_e_riservato_a_chi_viene_dopo(self):
        p = self.piano()
        zz, xx = np.array(sorted(p.ferrovia.celle)).T
        self.assertTrue(p.protetto[zz, xx].all())


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
            zz, xx = np.mgrid[0:lato, 0:lato]
            # colline dolci fra le citta': il binario deve salire e scendere
            e = (90.0 + 160.0 * np.exp(-((xx - 320) ** 2 + (zz - 200) ** 2) / (2 * 60.0 ** 2))
                 + 100.0 * np.exp(-((xx - 200) ** 2 + (zz - 260) ** 2) / (2 * 45.0 ** 2))).astype(np.float32)
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
                rotaie = {}
                for cx, cz in lv.all_chunk_coords("minecraft:overworld"):
                    ch = lv.get_chunk(cx, cz, "minecraft:overworld")
                    ids = [i for i, b in enumerate(ch.block_palette) if b.base_name in ("rail", "powered_rail")]
                    if not ids:
                        continue
                    pal = {i: str(ch.block_palette[i].properties["shape"].py_str) for i in ids}
                    for sy in ch.blocks.sub_chunks:
                        sub = np.asarray(ch.blocks.get_sub_chunk(sy))
                        for i in ids:
                            for lx, ly, lz in zip(*np.nonzero(sub == i)):
                                rotaie[(cx * 16 + int(lx), sy * 16 + int(ly), cz * 16 + int(lz))] = pal[i]
                self.assertGreater(len(rotaie), 100)
                self.assertTrue(any(f.startswith('ascending') for f in rotaie.values()), 'nessuna salita')
                DIR = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
                spezzati = 0
                for (x, y, z), forma in rotaie.items():
                    if forma.startswith("ascending"):
                        fini = [forma.split("_")[1], {"north": "south", "south": "north", "east": "west",
                                                      "west": "east"}[forma.split("_")[1]]]
                    else:
                        fini = forma.split("_")
                    for f in fini:
                        dx, dz = DIR[f]
                        alto = forma.startswith("ascending") and f == forma.split("_")[1]
                        candidati = [(x + dx, y, z + dz), (x + dx, y - 1, z + dz)]
                        if alto:
                            candidati.append((x + dx, y + 1, z + dz))
                        if not any(c in rotaie for c in candidati):
                            spezzati += 1
                # solo i capi delle linee e del binario morto delle stazioni
                self.assertLessEqual(spezzati, 12, spezzati)
            finally:
                lv.close()


if __name__ == "__main__":
    unittest.main()
