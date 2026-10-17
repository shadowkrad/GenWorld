"""Vita lungo le strade: pietre miliari, santelle, carovane."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import arredi as AR  # noqa: E402
from genworld import bordostrada as BS  # noqa: E402
from genworld import mappa as M  # noqa: E402
from genworld import template as TM  # noqa: E402
from genworld.citta import Citta  # noqa: E402

VER = TM.traduttore((26, 2, 0))


def nome_a(m, x, y, z):
    i = int(m.celle[x, y, z])
    return None if i < 0 else m.tavolozza[i][0]


class TestDisegni(unittest.TestCase):
    def test_il_miliare(self):
        m = BS.miliare().modello(VER)
        self.assertEqual(m.celle.shape, (1, 3, 1))
        self.assertTrue((m.celle >= 0).all())

    def test_la_santella_e_aperta_verso_la_strada(self):
        for verso, (fx, fz) in BS.DIREZIONI.items():
            m = BS.santella(verso).modello(VER)
            self.assertEqual(m.celle.shape, (3, 4, 3))
            cx, cz = 1 + fx, 1 + fz                       # la cella di facciata
            self.assertEqual(int(m.celle[cx, 1, cz]), -1, verso)          # aperta
            self.assertEqual(nome_a(m, cx, 2, cz), "lantern", verso)       # la lanterna appesa
            bx, bz = 1 - fx, 1 - fz                       # il fondo e' chiuso
            self.assertEqual(nome_a(m, bx, 1, bz), "stone_bricks")
            self.assertEqual(nome_a(m, 1, 2, 1), "candle")                 # la candela sull'altare
            self.assertEqual(nome_a(m, 1, 1, 1), "stone_bricks")          # l'altare scolpito
            # il tetto copre tutto
            self.assertTrue((m.celle[:, 3, :] >= 0).all())

    def test_la_carovana_ha_fuoco_carro_e_dimensioni_girate(self):
        for verso in BS.DIREZIONI:
            m = BS.carovana(verso).modello(VER)
            dx, dz = BS.dimensioni_carovana(verso)
            self.assertEqual(m.celle.shape, (dx, 3, dz))
            nomi = {n for n, _ in m.tavolozza}
            self.assertTrue({"campfire", "planks", "hay_block", "barrel"} <= nomi, verso)
            # il fuoco sta dalla parte della strada
            fx, fz = BS.DIREZIONI[verso]
            f = [(x, z) for x in range(dx) for z in range(dz)
                 if nome_a(m, x, 0, z) == "campfire"][0]
            c = [(x, z) for x in range(dx) for z in range(dz) if nome_a(m, x, 0, z) == "planks"]
            cx, cz = np.mean([p[0] for p in c]), np.mean([p[1] for p in c])
            self.assertGreater((f[0] - cx) * fx + (f[1] - cz) * fz, 0, verso)

    def test_i_posti_degli_animali_sono_dentro_o_accanto(self):
        for verso in BS.DIREZIONI:
            dx, dz = BS.dimensioni_carovana(verso)
            for x, z in BS.posti_carovana(verso, 100, 200):
                self.assertTrue(98 <= x <= 100 + dx + 2 and 198 <= z <= 200 + dz + 2, (verso, x, z))


def mondo(lato=600):
    cls = np.full((lato, lato), M.PRATERIA, np.uint8)
    h = np.full((lato, lato), 70, np.int32)
    strade = np.zeros((lato, lato), np.uint8)
    strade[300, 20:580] = 1                          # una strada dritta est-ovest
    strade[20:580, 450] = 1                          # e una nord-sud
    citta = [Citta(z=300, x=60, raggio=20, piazza=(300, 60), murata=True, lato=24)]
    return cls, h, strade, citta


class TestPianifica(unittest.TestCase):
    def esegui(self, **kw):
        cls, h, strade, citta = mondo()
        ris = AR.Risultato()
        occ = np.zeros(cls.shape, bool)
        c = BS.pianifica(ris, strade, citta, h, cls, occ, VER, seed=kw.get("seed", 1))
        return ris, c, strade, citta

    def test_mette_miliari_santelle_e_carovane(self):
        ris, c, *_ = self.esegui()
        self.assertGreater(c["miliare"], 5)
        self.assertGreaterEqual(c["santella"], 2)
        self.assertGreaterEqual(c["carovana"], 1)
        tipi = [a.tipo for a in ris.arredi]
        self.assertEqual(tipi.count("miliare"), c["miliare"])

    def test_determinismo(self):
        a = self.esegui(seed=4)[0]
        b = self.esegui(seed=4)[0]
        self.assertEqual([(x.tipo, x.x, x.z) for x in a.arredi], [(x.tipo, x.x, x.z) for x in b.arredi])

    def test_mai_sulla_strada_ne_vicino_al_paese(self):
        ris, c, strade, citta = self.esegui()
        for a in ris.arredi:
            self.assertFalse(strade[a.z:a.z + a.profondita, a.x:a.x + a.larghezza].any(), a)
            # fuori dal quadrato della citta' (centro x=60, z=300, mezza estensione 24 + margine)
            r = 24 + BS.MARGINE_PAESE
            self.assertFalse(abs(a.x - 60) <= r and abs(a.z - 300) <= r, a)

    def test_distanze_minime(self):
        ris, *_ = self.esegui()
        for tipo, dist in (("miliare", BS.DISTANZA_MILIARI - 6), ("santella", BS.DISTANZA_SANTELLE - 6)):
            pos = [(a.x, a.z) for a in ris.arredi if a.tipo == tipo]
            for i, p in enumerate(pos):
                for q in pos[i + 1:]:
                    self.assertGreaterEqual(np.hypot(p[0] - q[0], p[1] - q[1]), dist, tipo)

    def test_i_mercanti_vanno_con_le_carovane(self):
        ris, c, *_ = self.esegui()
        specie = [a.specie for a in ris.animali]
        self.assertEqual(specie.count("mercante"), c["carovana"])
        self.assertEqual(specie.count("lama_mercante"), 2 * c["carovana"])

    def test_i_modelli_si_riusano(self):
        ris, c, *_ = self.esegui()
        self.assertLessEqual(len(ris.modelli), 1 + 4 + 4)          # miliare + santelle + carovane
        for a in ris.arredi:
            self.assertTrue(0 <= a.modello < len(ris.modelli))

    def test_senza_strade_o_a_densita_zero_niente(self):
        cls, h, strade, citta = mondo()
        ris = AR.Risultato()
        self.assertEqual(sum(BS.pianifica(ris, None, citta, h, cls, None, VER).values()), 0)
        self.assertEqual(sum(BS.pianifica(ris, strade, citta, h, cls, None, VER, densita=0).values()), 0)
        self.assertEqual(ris.arredi, [])

    def test_non_sull_acqua(self):
        cls, h, strade, citta = mondo()
        cls[:, 150:260] = M.FIUME
        ris = AR.Risultato()
        BS.pianifica(ris, strade, citta, h, cls, np.zeros(cls.shape, bool), VER, seed=2)
        for a in ris.arredi:
            self.assertFalse(np.isin(cls[a.z:a.z + a.profondita, a.x:a.x + a.larghezza], M.ACQUA).any())


class TestMercante(unittest.TestCase):
    def test_nbt_del_mercante_e_dei_lama(self):
        from genworld.fauna import Animale
        m = Animale(x=1.5, y=70.0, z=1.5, specie="mercante", seme=1).nbt_tag()
        self.assertEqual(str(m["id"]), "minecraft:wandering_trader")
        self.assertGreater(int(m["DespawnDelay"]), 10 ** 8)
        l = Animale(x=1.5, y=70.0, z=1.5, specie="lama_mercante", seme=2).nbt_tag()
        self.assertEqual(str(l["id"]), "minecraft:trader_llama")
        self.assertNotIn("DespawnDelay", l)


if __name__ == "__main__":
    unittest.main()
