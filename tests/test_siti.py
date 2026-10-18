"""Luoghi da scoprire (`siti.py`)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import bauli as BA
from genworld import mappa as M
from genworld import siti as SI
from genworld.citta import Citta

MARE = 62
LATO = 1000


class Finto:
    def __init__(self):
        self.ids = {("air", ()): 0}
        self.id_aria = 0

    def blocco(self, nome, **pr):
        return self.ids.setdefault((nome, tuple(sorted(pr.items()))), len(self.ids))

    def nome(self, i):
        return next((n for (n, _), v in self.ids.items() if v == i), "?")


def mondo():
    """Un quarto di mare profondo, una foresta con un lago, pianure e una collina."""
    cls = np.full((LATO, LATO), M.PIANURA, np.uint8)
    h = np.full((LATO, LATO), 70, np.int32)
    cls[:, :300] = M.OCEANO
    h[:, :300] = 40
    cls[:, 300:310] = M.SPIAGGIA
    h[:, 300:310] = 64
    cls[600:1000, 310:700] = M.FORESTA
    cls[700:740, 400:440] = M.MARE
    h[700:740, 400:440] = 58
    cls[:300, 600:] = M.MONTAGNA
    h[:300, 600:] = 110
    cls[300:600, 400:900] = M.PRATERIA
    return cls, h


def siti(seed=1):
    cls, h = mondo()
    citta = [Citta(z=450, x=500, raggio=40, piazza=(450, 500), murata=True)]
    return SI.pianifica(cls, h, MARE, None, citta, seed=seed), cls, h


class TestPianifica(unittest.TestCase):
    def test_tutti_i_tipi_con_una_mappa_adatta(self):
        s, _, _ = siti()
        tipi = {x.tipo for x in s}
        for t in ("sommersa", "labirinto", "arena", "santuario", "segnale", "pesca", "museo"):
            self.assertIn(t, tipi, t)

    def test_deterministico(self):
        a, _, _ = siti(3)
        b, _, _ = siti(3)
        self.assertEqual([(x.tipo, x.x, x.z, x.nome) for x in a], [(x.tipo, x.x, x.z, x.nome) for x in b])

    def test_distanti_dentro_la_mappa_e_con_nome(self):
        s, _, _ = siti()
        for i, a in enumerate(s):
            self.assertTrue(a.nome)
            x0, z0, x1, z1 = a.bbox()
            self.assertTrue(0 <= x0 and x1 < LATO and 0 <= z0 and z1 < LATO, a)
            for b in s[i + 1:]:
                if "museo" in (a.tipo, b.tipo):
                    continue
                self.assertGreater((a.x - b.x) ** 2 + (a.z - b.z) ** 2, (a.mezzo + b.mezzo) ** 2, (a, b))

    def test_le_maschere_rispettano_il_terreno(self):
        s, cls, h = siti()
        for x in s:
            if x.tipo == "sommersa":
                self.assertTrue(np.isin(cls[x.z, x.x], M.MARINO))
                self.assertLessEqual(x.y, MARE - 10)
            elif x.tipo == "strega":
                self.assertEqual(cls[x.z, x.x], M.FORESTA)
            elif x.tipo == "museo":
                self.assertFalse(np.isin(cls[x.z, x.x], M.ACQUA))
            else:
                self.assertFalse(np.isin(cls[x.z, x.x], M.ACQUA), x.tipo)

    def test_museo_vicino_alla_citta(self):
        s, _, _ = siti()
        m = next(x for x in s if x.tipo == "museo")
        d = ((m.x - 500) ** 2 + (m.z - 450) ** 2) ** 0.5
        self.assertTrue(50 <= d <= 85, d)

    def test_evita_le_zone_protette(self):
        cls, h = mondo()
        evita = np.zeros(cls.shape, bool)
        evita[:, 300:] = True
        s = SI.pianifica(cls, h, MARE, evita, [], seed=1)
        for x in s:
            self.assertLess(x.x + x.mezzo, 300 + 1, x)

    def test_densita_zero_niente(self):
        cls, h = mondo()
        self.assertEqual(SI.pianifica(cls, h, MARE, None, [], densita=0), [])


class TestDisegni(unittest.TestCase):
    def setUp(self):
        self.s, self.cls, self.h = siti()

    def test_ogni_tipo_ha_un_disegno_nei_limiti_con_bottino_noto(self):
        visti = set()
        for x in self.s:
            visti.add(x.tipo)
            d = SI.disegna(x)
            bx0, bz0, bx1, bz1 = x.bbox()
            self.assertTrue(d.prim, x.tipo)
            for p in d.prim:
                if p[0] == "B":
                    _, a0, y0, c0, a1, y1, c1, k, _v = p
                    self.assertIn(k, SI.BLOCCHI)
                    self.assertTrue(bx0 - 3 <= a0 and a1 <= bx1 + 3 and bz0 - 3 <= c0 and c1 <= bz1 + 3,
                                    (x.tipo, p))
                    self.assertTrue(-64 < y0 and y1 < 300, (x.tipo, p))
            for (fx, fy, fz, tipo) in d.forzieri:
                self.assertTrue(tipo in BA.BOTTINO_DIMENSIONI or tipo.startswith("mago"), tipo)
                self.assertTrue(bx0 <= fx <= bx1 and bz0 <= fz <= bz1)
            for c in d.creature:
                self.assertTrue(bx0 <= c.x <= bx1 and bz0 <= c.z <= bz1, (x.tipo, c))
        self.assertGreaterEqual(len(visti), 7)

    def test_i_blocchi_si_traducono(self):
        import PyMCTranslate
        from amulet.api.block import Block
        from amulet_nbt import StringTag
        tr = PyMCTranslate.new_translation_manager()
        v = tr.get_version("java", (26, 2, 0))
        for k, (n, pr) in SI.BLOCCHI.items():
            b = Block("universal_minecraft", n, {a: StringTag(c) for a, c in pr.items()})
            r = v.block.from_universal(b)
            self.assertTrue(r[0].namespaced_name.startswith("minecraft:"), (k, r))
            self.assertNotEqual(r[0].base_name, "air" if k != "aria" else "x", k)

    def test_labirinto_connesso_dall_ingresso_al_centro(self):
        for seme in range(5):
            g = SI._labirinto(SI.N_LABIRINTO, np.random.default_rng(seme))
            lato = g.shape[0]
            libere = {(i, j) for i in range(lato) for j in range(lato) if not g[i, j]}
            self.assertIn((1, 1), libere)
            visti, pila = {(1, 1)}, [(1, 1)]
            while pila:
                i, j = pila.pop()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (i + di, j + dj)
                    if q in libere and q not in visti:
                        visti.add(q)
                        pila.append(q)
            celle = {(1 + 2 * a, 1 + 2 * b) for a in range(SI.N_LABIRINTO) for b in range(SI.N_LABIRINTO)}
            self.assertTrue(celle <= visti)

    def test_ci_sono_creature_boss_con_nome(self):
        nomi = {}
        for x in self.s:
            for c in SI.disegna(x).creature:
                if c.capo:
                    nomi[x.tipo] = (c.specie, c.nome)
        self.assertEqual(nomi["sommersa"][0], "guardiano_anziano")
        self.assertEqual(nomi["arena"][0], "devastatore")
        for t, (sp, n) in nomi.items():
            self.assertTrue(n, t)
        from genworld.torri import SPECIE
        for x in self.s:
            for c in SI.disegna(x).creature:
                self.assertIn(c.specie, SPECIE)
                self.assertTrue(str(c.nbt_tag()["id"]).startswith("minecraft:"))

    def test_luoghi_in_coordinate_di_gioco(self):
        lu = SI.luoghi(self.s, LATO // 2)
        self.assertEqual(len(lu), len(self.s))
        self.assertEqual(lu[0].x, self.s[0].x - LATO // 2)


class TestAltro(unittest.TestCase):
    def test_pellegrini_catena_di_libri_fra_i_santuari(self):
        s, _, _ = siti()
        sant = [x for x in s if x.tipo == "santuario"]
        self.assertGreaterEqual(len(sant), 2)
        libri = SI.libri_pellegrini(s, LATO // 2)
        self.assertEqual(len(libri), len(sant))
        for pos, l in libri.items():
            self.assertLessEqual(len(l.titolo), 32)
            self.assertTrue(all(len(p) < 250 for p in l.pagine))
        ultimi = [l for l in libri.values() if "ultima" in l.pagine[0]]
        self.assertEqual(len(ultimi), 1)
        # le posizioni sono quelle del forziere del santuario
        for x in sant:
            self.assertIn((x.x - LATO // 2, x.y + 1, x.z - 1 - LATO // 2), libri)

    def test_nessun_libro_con_un_solo_santuario(self):
        cls, h = mondo()
        s = [SI.Sito("santuario", 500, 500, 70, 1, "S")]
        self.assertEqual(SI.libri_pellegrini(s, 500), {})

    def test_il_curatore_compra_reperti(self):
        from genworld import commissioni as CM
        x = SI.Sito("museo", 500, 500, 70, 1, "Museo")
        c = CM.curatore(x, seed=3)
        self.assertEqual(c.nome, "Curatore del museo")
        self.assertEqual(c.x, 500.5)
        self.assertGreaterEqual(len(c.offerte), 6)
        self.assertLessEqual(len(c.offerte), 12)
        for o in c.offerte:
            self.assertEqual(o["vende"][0], "emerald")
        from genworld import entita as EN
        EN._scambio(c.offerte[0])

    def test_mulino_ha_le_pale_a_nord_e_il_forziere(self):
        x = SI.Sito("mulino", 100, 100, 70, 1, "M")
        d = SI.disegna(x)
        self.assertEqual([t for *_, t in d.forzieri], ["mulino"])
        self.assertTrue(any(p[0] == "B" and p[7] == "lana" and p[3] == 96 for p in d.prim))


class TestPosa(unittest.TestCase):
    def posa(self, x, quota=60):
        s = Finto()
        tav = SI.Tavolozza(s)
        x0, z0, x1, z1 = x.bbox()
        res = {}
        for cz in range(z0 // 16, z1 // 16 + 1):
            for cx in range(x0 // 16, x1 // 16 + 1):
                out = np.zeros((16, 384, 16), np.uint32)
                ys = np.arange(-64, 320)[None, :, None]
                out[np.broadcast_to(ys < quota, out.shape)] = s.blocco("stone")
                SI.posa(out, tav, cx * 16, cz * 16, -64, [x], [0], meta=0)
                res[(cx, cz)] = out

        def blocco(px, py, pz):
            o = res.get((px // 16, pz // 16))
            return None if o is None else s.nome(int(o[px % 16, py + 64, pz % 16]))
        return s, tav, blocco

    def test_segnale_torre_con_scala_e_forziere(self):
        x = SI.Sito("segnale", 100, 100, 70, seme=1, nome="T")
        s, tav, b = self.posa(x, 69)
        self.assertEqual(b(100, 71, 98), "air")                   # la porta
        self.assertEqual(b(99, 80, 101), "ladder")
        self.assertEqual(b(100, 87, 100), "lantern")
        self.assertEqual(len(tav.bauli), 2)

    def test_sommersa_riempie_di_acqua_dove_c_e_aria(self):
        x = SI.Sito("sommersa", 200, 200, 40, seme=1, nome="S")
        s = Finto()
        tav = SI.Tavolozza(s)
        out = np.full((16, 384, 16), tav.acqua, np.uint32)
        out[:, :(40 + 64), :] = s.blocco("stone")
        for cz in range(11, 15):
            for cx in range(11, 15):
                SI.posa(out, tav, cx * 16, cz * 16, -64, [x], [0], meta=0)
        interno = out[202 - 192, 40 + 64 + 3, 202 - 192]
        self.assertEqual(interno, tav.acqua)
        self.assertTrue(((out == tav.id["prism_mattoni"]).sum()) > 100)

    def test_labirinto_ha_siepe_e_forziere_al_centro(self):
        x = SI.Sito("labirinto", 300, 300, 70, seme=2, nome="L")
        s, tav, b = self.posa(x, 69)
        self.assertEqual(b(300, 70, 300), "chest")
        self.assertEqual(len(tav.bauli), 1)
        siepi = sum(1 for dx in range(-8, 9) for dz in range(-8, 9) if b(300 + dx, 70, 300 + dz) == "leaves")
        self.assertGreater(siepi, 60)

    def test_grotta_scavata_con_acqua_e_pozzo(self):
        x = SI.Sito("grotta", 400, 400, 90, seme=3, nome="G", extra={"fondo": 44})
        s, tav, b = self.posa(x, 89)
        self.assertEqual(b(400, 52, 400), "air")
        self.assertEqual(b(400, 44, 400), "water")
        self.assertEqual(b(412, 60, 400), "ladder")
        self.assertEqual(b(412, 89, 400), "ladder")

    def test_arena_chiusa_con_un_solo_ingresso(self):
        x = SI.Sito("arena", 500, 500, 70, seme=4, nome="A")
        s, tav, b = self.posa(x, 69)
        self.assertEqual(b(500, 71, 512), "air")
        self.assertNotEqual(b(505, 71, 512), "air")
        self.assertNotEqual(b(500, 71, 488), "air")

    def test_strega_su_palafitte_con_scalini(self):
        x = SI.Sito("strega", 600, 600, 70, seme=5, nome="Capanna di Nonna")
        s, tav, b = self.posa(x, 69)
        self.assertEqual(b(600, 74, 600), "carpet")
        self.assertEqual(b(604, 73, 600), "stairs")
        self.assertEqual(b(605, 72, 600), "stairs")
        self.assertEqual(b(605, 71, 600), "planks")
        self.assertEqual(b(600, 72, 600), "air")                  # il pavimento sta a 73


if __name__ == "__main__":
    unittest.main()
