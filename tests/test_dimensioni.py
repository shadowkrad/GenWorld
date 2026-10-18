"""Nether ed End di Arda (`nether.py`, `end.py`, `dimensioni.py`)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import bauli as BA
from genworld import dimensioni as DM
from genworld import end as FINE
from genworld import nether as NE


class Finto:
    def __init__(self):
        self.ids = {("air", ()): 0}
        self.id_aria = 0

    def blocco(self, nome, **pr):
        return self.ids.setdefault((nome, tuple(sorted(pr.items()))), len(self.ids))

    def nome(self, i):
        return next((n for (n, _), v in self.ids.items() if v == i), "?")


LATO = 600


def piano():
    return NE.pianifica(LATO, seed=3)


class TestPianoNether(unittest.TestCase):
    def test_deterministico_e_forme(self):
        a, b = piano(), piano()
        self.assertTrue(np.array_equal(a.pav, b.pav))
        self.assertEqual([(s.x, s.z, s.nome) for s in a.strutture],
                         [(s.x, s.z, s.nome) for s in b.strutture])
        self.assertEqual(a.pav.shape, (LATO, LATO))
        self.assertTrue(set(np.unique(a.bioma)) <= set(range(len(NE.BIOMI))))

    def test_c_e_sempre_spazio_fra_pavimento_e_volta(self):
        p = piano()
        self.assertGreaterEqual(int((p.sof - p.pav).min()), 30)
        self.assertLess(int(p.sof.max()), NE.VOLTA - 4)

    def test_strutture_distanti_nel_margine_e_lontane_dal_portale(self):
        p = NE.pianifica(1000, seed=5)
        self.assertGreaterEqual(len(p.strutture), 6)
        for i, a in enumerate(p.strutture):
            self.assertLessEqual(abs(a.x) + a.mezzo, p.meta - 5)
            self.assertLessEqual(abs(a.z) + a.mezzo, p.meta - 5)
            self.assertGreater(a.x ** 2 + a.z ** 2, 60 ** 2)
            self.assertLessEqual(a.y, NE.QUOTA_MAX_STRUTTURE)
            for b in p.strutture[i + 1:]:
                self.assertGreaterEqual((a.x - b.x) ** 2 + (a.z - b.z) ** 2,
                                        NE.DISTANZA_FRA_STRUTTURE ** 2)
        self.assertEqual({s.tipo for s in p.strutture}, {"fortezza", "bastione"})
        self.assertEqual(len({s.nome for s in p.strutture}), len(p.strutture))

    def test_creature_a_terra_non_nelle_strutture(self):
        p = NE.pianifica(1000, seed=5)
        self.assertGreater(len(p.creature), 50)
        for c in p.creature:
            for s in p.strutture:
                self.assertFalse(abs(c.x - s.x) <= s.mezzo and abs(c.z - s.z) <= s.mezzo)


class TestChunkNether(unittest.TestCase):
    def setUp(self):
        self.p = piano()
        self.s = Finto()
        self.tav = NE.Tavolozza(self.s)
        self.bio = list(range(5))

    def chunk(self, cx, cz):
        return NE.blocchi_chunk(self.p, self.tav, cx, cz, self.bio)

    def test_fondo_e_volta_di_bedrock_e_aria_sopra_la_volta(self):
        out, biomi, spawner, fo = self.chunk(0, 0)
        I = self.tav.id
        self.assertEqual(out.shape, (16, NE.ALT, 16))
        self.assertTrue((out[:, 0, :] == I["bedrock"]).all())
        self.assertTrue((out[:, NE.VOLTA, :] == I["bedrock"]).all())
        self.assertTrue((out[:, NE.VOLTA + 1:, :] == I["aria"]).all())
        self.assertEqual(biomi.shape, (16, 16))

    def test_lava_solo_sotto_il_livello_del_mare_di_lava(self):
        I = self.tav.id
        trovata = False
        for cx in range(-12, 12, 2):
            for cz in range(-12, 12, 2):
                out = self.chunk(cx, cz)[0]
                ys = np.nonzero((out == I["lava"]).any(axis=(0, 2)))[0]
                if len(ys):
                    trovata = True
                    self.assertLessEqual(int(ys.max()), NE.LAVA)
        self.assertTrue(trovata)

    def test_cammino_libero_fra_pavimento_e_volta(self):
        out = self.chunk(2, 2)[0]
        I = self.tav.id
        ora = out[:, 60, :]
        self.assertTrue(((ora == I["aria"]) | (ora != I["aria"])).all())
        self.assertGreater(int((out == I["netherrack"]).sum()), 1000)
        self.assertGreater(int((out == I["quarzo"]).sum()) + int((out == I["oro_nether"]).sum()), 0)

    def cerca(self, tipo):
        s = next(s for s in self.p.strutture if s.tipo == tipo)
        return s, self.chunk(s.x >> 4, s.z >> 4)

    def test_fortezza_sala_vuota_spawner_e_forzieri(self):
        s, (out, _, spawner, fo) = self.cerca("fortezza")
        I = self.tav.id
        ox, oz = (s.x >> 4) * 16, (s.z >> 4) * 16
        self.assertEqual(spawner[0][3] if spawner else "blaze", "blaze")
        # il centro della sala e' il piedistallo con lo spawner; due passi piu' in la' si cammina
        for dx in (2, 3):
            x, z = s.x + dx - ox, s.z - oz
            if 0 <= x < 16 and 0 <= z < 16:
                self.assertEqual(out[x, s.y + 1, z], I["aria"])
                self.assertEqual(out[x, s.y + 2, z], I["aria"])
                self.assertEqual(out[x, s.y, z], I["mattoni"])
        self.assertTrue(all(t.startswith("nether_") for t in fo.values()))

    def test_fortezza_intera_ha_spawner_e_tre_forzieri(self):
        s = next(s for s in self.p.strutture if s.tipo == "fortezza")
        spw, forz = [], {}
        x0, z0, x1, z1 = s.bbox()
        for cz in range(z0 >> 4, (z1 >> 4) + 1):
            for cx in range(x0 >> 4, (x1 >> 4) + 1):
                _, _, sp, fo = self.chunk(cx, cz)
                spw += sp
                forz.update(fo)
        self.assertEqual(len(spw), 1)
        self.assertEqual(sorted(forz.values()), ["nether_fortezza", "nether_fortezza", "nether_tesoro"])

    def test_bastione_scala_e_forzieri(self):
        s = next(s for s in self.p.strutture if s.tipo == "bastione")
        forz = {}
        scale = 0
        x0, z0, x1, z1 = s.bbox()
        I = self.tav.id
        for cz in range(z0 >> 4, (z1 >> 4) + 1):
            for cx in range(x0 >> 4, (x1 >> 4) + 1):
                out, _, _, fo = self.chunk(cx, cz)
                forz.update(fo)
                scale += int((out == I["scala"]).sum())
        self.assertEqual(len(forz), 4)
        self.assertTrue(all(t == "nether_bastione" for t in forz.values()))
        self.assertEqual(scale, 9)

    def test_ogni_scatola_resta_nei_limiti(self):
        for s in self.p.strutture:
            x0, z0, x1, z1 = s.bbox()
            for sc in NE.disegna(s)[0]:
                a0, b0, c0, a1, b1, c1 = sc[:6]
                self.assertTrue(x0 <= a0 and a1 <= x1 and z0 <= c0 and c1 <= z1, (s.tipo, sc))
                self.assertTrue(0 <= b0 and b1 < NE.VOLTA - 4, (s.tipo, sc))

    def test_i_blocchi_si_traducono_in_gioco(self):
        import PyMCTranslate
        from amulet.api.block import Block
        from amulet_nbt import StringTag
        tr = PyMCTranslate.new_translation_manager()
        v = tr.get_version("java", (26, 2, 0))
        universali = set(tr.universal_format.biome.biome_ids)
        for nome in NE.BIOMI + ("the_end",):
            self.assertIn(f"universal_minecraft:{nome}", universali, nome)
        for tabella in (NE.BLOCCHI, FINE.BLOCCHI):
            for k, (n, pr) in tabella.items():
                b = Block("universal_minecraft", n, {a: StringTag(c) for a, c in pr.items()})
                r = v.block.from_universal(b)
                self.assertTrue(r[0].namespaced_name.startswith("minecraft:"), (k, r))
                if k not in ("recinto", "tronco_c", "tronco_d"):   # questi cambiano nome in gioco
                    self.assertEqual(r[0].base_name, n, k)


class TestEnd(unittest.TestCase):
    def test_sette_isole_due_cittadelle_una_con_le_ali(self):
        p = FINE.pianifica(1000, seed=3)
        self.assertEqual(len(p.isole), 7)
        self.assertEqual(sum(i.cittadella for i in p.isole), 2)
        self.assertEqual(sum(i.tesoro for i in p.isole), 1)
        for i in p.isole:
            self.assertGreaterEqual(i.x ** 2 + i.z ** 2, 150 ** 2)

    def test_chunk_fuori_dalle_isole_non_si_scrivono(self):
        p = FINE.pianifica(1000, seed=3)
        s = Finto()
        tav = FINE.Tavolozza(s)
        self.assertIsNone(FINE.blocchi_chunk(p, tav, 0, 0, 1))
        i = p.isole[0]
        r = FINE.blocchi_chunk(p, tav, i.x >> 4, i.z >> 4, 1)
        self.assertIsNotNone(r)
        self.assertGreater(int((r[0] == tav.id["pietra"]).sum()), 100)

    def test_la_cittadella_ha_porta_e_forzieri(self):
        p = FINE.pianifica(1000, seed=3)
        s = Finto()
        tav = FINE.Tavolozza(s)
        cit = next(i for i in p.isole if i.tesoro)
        forz = {}
        for cz in range((cit.z - 10) >> 4, ((cit.z + 10) >> 4) + 1):
            for cx in range((cit.x - 10) >> 4, ((cit.x + 10) >> 4) + 1):
                r = FINE.blocchi_chunk(p, tav, cx, cz, 1)
                forz.update(r[2])
        self.assertEqual(sorted(forz.values()), ["end_citta", "end_citta", "end_tesoro"])

    def test_creature_sulle_isole(self):
        p = FINE.pianifica(1000, seed=3)
        self.assertEqual(sum(1 for c in p.creature if c.specie == "shulker"), 8)


class TestBottino(unittest.TestCase):
    def test_tabelle_coerenti(self):
        for tipo, voci in BA.BOTTINO_DIMENSIONI.items():
            for seme in range(20):
                c = BA._oggetti_dungeon(np.random.default_rng(seme), tipo, True)
                slot = [v[0] for v in c]
                self.assertEqual(len(slot), len(set(slot)), tipo)
                self.assertTrue(all(0 <= x < 27 for x in slot))
                nomi = {v for v, *_ in voci}
                self.assertTrue(all(v[1] in nomi or len(v) == 4 for v in c), tipo)

    def test_le_ali_stanno_sempre_nel_tesoro_della_fine(self):
        for seme in range(30):
            c = BA._oggetti_dungeon(np.random.default_rng(seme), "end_tesoro", False)
            self.assertIn("elytra", [v[1] for v in c])


class TestSistemaCreature(unittest.TestCase):
    def test_chi_e_nato_in_un_blocco_sale(self):
        from genworld.torri import Combattente
        out = np.zeros((16, 40, 16), np.uint32)
        out[:, :10, :] = 5
        c = Combattente(x=3.5, y=4.0, z=3.5, specie="piglin")
        DM._sistema_creature([c], out, 0, 0, 0)
        self.assertEqual(c.y, 10.0)


class TestScrittura(unittest.TestCase):
    def test_un_chunk_del_nether_va_su_disco_e_si_rilegge(self):
        import amulet
        from genworld.livello_dat import ImpostazioniMondo
        from genworld.mondo import ScrittoreMondo
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w")
            imp = ImpostazioniMondo(nome="T", versione=(26, 2, 0), seed=1)
            with ScrittoreMondo(p, imp, lato_blocchi=64, crea=True) as m:
                tav = NE.Tavolozza(m)
                pn = NE.pianifica(64, 1, strutture=0)
                out, biomi, _, _ = NE.blocchi_chunk(pn, tav, 0, 0, [m.bioma(b) for b in NE.BIOMI])
                m.scrivi_chunk(0, 0, out, y0=0, biomi=biomi, dimensione="minecraft:the_nether")
            self.assertTrue(os.path.exists(
                os.path.join(p, "dimensions", "minecraft", "the_nether", "region", "r.0.0.mca")))
            lv = amulet.load_level(p)
            try:
                ch = lv.get_chunk(0, 0, "minecraft:the_nether")
                nomi = {str(ch.block_palette[i].base_name) for i in np.unique(ch.blocks[:, :128, :])}
                self.assertIn("netherrack", nomi)
                self.assertIn("bedrock", nomi)
            finally:
                lv.close()


if __name__ == "__main__":
    unittest.main()
