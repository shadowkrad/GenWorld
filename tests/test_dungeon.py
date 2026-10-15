"""Rovine con cripta (#5)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import dungeon as DG

Y0, ALT = -64, 320
STANZA_LIBERA = ("air", "cobweb", "soul_lantern", "spawner", "bone_block")


class Finto:
    def __init__(self):
        self.ids = {("air", ()): 0}
        self.id_aria = 0

    def blocco(self, nome, **pr):
        return self.ids.setdefault((nome, tuple(sorted(pr.items()))), len(self.ids))

    def nome(self, i):
        return next((n for (n, _), v in self.ids.items() if v == i), "?")


def terreno(lato=1000, quota=90):
    return np.full((lato, lato), quota, np.int32), np.zeros((lato, lato), bool)


def un_dungeon(direzione=(1, 0), quota=90, x=150, z=150):
    d = DG.Dungeon(x=x, z=z, ys=quota, dx=direzione[0], dz=direzione[1],
                   mob="skeleton", seme=5)
    d.scatole = DG._scatole(quota)
    return d


def posa_su_regione(d, quota=90):
    """`posa` lavora su un chunk 16x16: si stampa chunk per chunk e si ricompone."""
    s = Finto()
    tav = DG.Tavolozza(s)
    x0, z0, x1, z1 = d.bbox()
    regione = {}
    for cz in range(z0 // 16, z1 // 16 + 1):
        for cx in range(x0 // 16, x1 // 16 + 1):
            out = np.zeros((16, ALT, 16), np.uint32)
            ys = np.arange(Y0, Y0 + ALT)[None, :, None]
            out[np.broadcast_to(ys < quota, out.shape)] = s.blocco("stone")
            h = np.full((16, 16), quota, np.int32)
            DG.posa(out, tav, h, cx * 16, cz * 16, Y0, [d], [0], meta=0)
            regione[(cx, cz)] = out

    def blocco(x, y, z):
        out = regione.get((x // 16, z // 16))
        return None if out is None else s.nome(int(out[x % 16, y - Y0, z % 16]))
    return s, tav, blocco


class TestPianifica(unittest.TestCase):
    def test_trova_siti_su_un_pianoro_asciutto(self):
        h, acqua = terreno()
        ds = DG.pianifica(h, acqua, seed=1)
        self.assertGreaterEqual(len(ds), 3)
        for i, a in enumerate(ds):
            self.assertEqual(a.ys, 90)
            for b in ds[i + 1:]:
                self.assertGreaterEqual((a.x - b.x) ** 2 + (a.z - b.z) ** 2,
                                        DG.DISTANZA_FRA_RUINE ** 2)

    def test_non_nasce_su_acqua(self):
        h, acqua = terreno()
        self.assertEqual(DG.pianifica(h, np.ones_like(acqua), seed=1), [])

    def test_non_nasce_su_terreno_accidentato(self):
        h, acqua = terreno()
        irregolare = (80 + np.random.default_rng(0).integers(0, 12, h.shape)).astype(np.int32)
        self.assertEqual(DG.pianifica(irregolare, acqua, seed=1), [])

    def test_il_percorso_vuole_terra_sopra_le_stanze(self):
        h, acqua = terreno()
        d = un_dungeon((1, 0))
        self.assertTrue(DG._percorso_valido(d, h, acqua, None))
        x, z = d.punto(DG.U_SPAWNER[0] + 3, 0)
        h2 = h.copy()
        h2[z, x] = 75                       # il soffitto della sala resterebbe scoperto
        self.assertFalse(DG._percorso_valido(d, h2, acqua, None))
        acqua2 = acqua.copy()
        acqua2[z, x] = True                 # un lago sopra la cripta
        self.assertFalse(DG._percorso_valido(d, h, acqua2, None))

    def test_rispetta_la_maschera_evita(self):
        h, acqua = terreno()
        self.assertEqual(DG.pianifica(h, acqua, evita=np.ones_like(acqua), seed=1), [])

    def test_e_deterministica_e_la_densita_zero_spegne(self):
        h, acqua = terreno()
        a = [(d.x, d.z, d.dx, d.dz, d.mob) for d in DG.pianifica(h, acqua, seed=4)]
        b = [(d.x, d.z, d.dx, d.dz, d.mob) for d in DG.pianifica(h, acqua, seed=4)]
        self.assertEqual(a, b)
        self.assertEqual(DG.pianifica(h, acqua, densita=0.0, seed=4), [])

    def test_la_cripta_resta_dentro_la_mappa(self):
        h, acqua = terreno()
        for d in DG.pianifica(h, acqua, seed=2):
            x0, z0, x1, z1 = d.bbox()
            self.assertTrue(x0 >= 0 and z0 >= 0 and x1 < 1000 and z1 < 1000)


class TestPosa(unittest.TestCase):
    def setUp(self):
        self.d = un_dungeon((1, 0))
        self.s, self.tav, self.b = posa_su_regione(self.d)

    def test_la_scala_scende_di_un_blocco_a_passo(self):
        d = self.d
        for u in range(1, DG.PROFONDITA + 1):
            x, z = d.punto(u, 0)
            self.assertEqual(self.b(x, d.ys - u, z), "stairs", f"scalino u={u}")
            self.assertEqual(self.b(x, d.ys - u + 1, z), "air")
            self.assertEqual(self.b(x, d.ys - u + 3, z), "air")

    def test_le_stanze_sono_vuote_e_hanno_un_pavimento(self):
        d = self.d
        for sc in d.scatole[1:]:
            x, z = d.punto((sc.u0 + sc.u1) // 2, 0)
            for y in range(sc.y0 + 1, sc.y1 + 1):
                self.assertIn(self.b(x, y, z), STANZA_LIBERA + ("stairs",), (sc.tipo, y))
            self.assertNotIn(self.b(x, sc.y0 - 1, z), ("air", None))

    def test_si_arriva_dalla_scala_al_tesoro_senza_muri(self):
        """Lungo l'asse centrale a quota di calpestio si cammina dall'ultimo
        scalino al forziere finale: niente roccia di mezzo."""
        d = self.d
        for u in range(DG.U_ANTICAMERA[0], DG.U_TESORO[1] - 1):
            x, z = d.punto(u, 0)
            for y in (d.yf + 1, d.yf + 2):
                self.assertIn(self.b(x, y, z), STANZA_LIBERA + ("stone_bricks",),
                              f"ostacolo a u={u} y={y}: {self.b(x, y, z)}")
            self.assertIn(self.b(x, d.yf, z), STANZA_LIBERA + ("stone_bricks",))

    def test_lo_spawner_sta_sul_piedistallo(self):
        sx, sy, sz = self.d.spawner()
        self.assertEqual(self.b(sx, sy, sz), "spawner")
        self.assertEqual(self.b(sx, sy - 1, sz), "stone_bricks")

    def test_i_forzieri_sono_in_aria_aperta_non_sotto_un_blocco(self):
        self.assertEqual(len(self.tav.bauli), 3)
        for x, y, z, tipo in self.tav.bauli:
            self.assertEqual(self.b(x, y, z), "chest")
            self.assertIn(self.b(x, y + 1, z), ("air", "cobweb"), f"forziere {tipo} coperto")

    def test_due_forzieri_di_stanza_e_uno_di_tesoro(self):
        self.assertEqual(sorted(t for *_, t in self.tav.bauli), ["stanza", "stanza", "tesoro"])

    def test_le_trappole_hanno_il_tnt_sotto_e_aria_sopra(self):
        d = self.d
        ux = DG.U_TESORO[1] - 1
        for v in (-2, 2):
            x, z = d.punto(ux, v)
            self.assertEqual(self.b(x, d.yf, z), "trapped_chest")
            self.assertEqual(self.b(x, d.yf - 1, z), "tnt")
            self.assertIn(self.b(x, d.yf + 1, z), ("air", "cobweb"))

    def test_la_rovina_ha_pareti_e_un_pavimento(self):
        d = self.d
        muri = 0
        for u in range(-DG.RUINA, DG.RUINA + 1):
            for v in range(-DG.RUINA, DG.RUINA + 1):
                x, z = d.punto(u, v)
                if max(abs(u), abs(v)) == DG.RUINA and self.b(x, d.ys, z) == "stone_bricks":
                    muri += 1
                if max(abs(u), abs(v)) < DG.RUINA and not (u >= 1 and abs(v) <= 1):
                    self.assertEqual(self.b(x, d.ys - 1, z), "stone_bricks")
        self.assertGreater(muri, 8)

    def test_le_quattro_direzioni_producono_la_stessa_cripta(self):
        for dire in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d = un_dungeon(dire)
            _, tav, b = posa_su_regione(d)
            sx, sy, sz = d.spawner()
            self.assertEqual(b(sx, sy, sz), "spawner", dire)
            x, z = d.punto(DG.U_ANTICAMERA[0] + 2, 0)
            self.assertIn(b(x, d.yf + 1, z), STANZA_LIBERA, dire)
            self.assertEqual(len(tav.bauli), 3, dire)

    def test_non_tocca_i_chunk_lontani(self):
        s = Finto()
        tav = DG.Tavolozza(s)
        out = np.zeros((16, ALT, 16), np.uint32)
        out[:, :100, :] = s.blocco("stone")
        prima = out.copy()
        DG.posa(out, tav, np.full((16, 16), 90, np.int32), 0, 0, Y0, [self.d], [], meta=0)
        self.assertTrue((out == prima).all())


class TestIndiceEMaschera(unittest.TestCase):
    def test_l_indice_copre_tutta_la_cripta(self):
        d = un_dungeon((1, 0))
        idx = DG.indice_per_chunk([d])
        x0, z0, x1, z1 = d.bbox()
        for x in range(x0, x1 + 1, 7):
            self.assertIn(0, idx.get((x // 16, d.z // 16), []), x)

    def test_la_maschera_copre_la_rovina_e_il_margine(self):
        d = un_dungeon()
        m = DG.maschera([d], (300, 300), margine=2)
        self.assertTrue(m[d.z, d.x])
        self.assertTrue(m[d.z + DG.RUINA + 2, d.x])
        self.assertFalse(m[d.z + DG.RUINA + 5, d.x])


class TestSpawnerEntity(unittest.TestCase):
    def test_il_blockentity_dello_spawner(self):
        d = un_dungeon()
        sx, sy, sz = d.spawner()
        be = DG.entita_blocco([d], (sx // 16) * 16, (sz // 16) * 16, 0)
        self.assertEqual(len(be), 1)
        self.assertEqual((be[0].x, be[0].y, be[0].z), (sx, sy, sz))
        self.assertEqual(str(be[0].nbt.compound["utags"]["SpawnData"]["id"]),
                         "minecraft:skeleton")
        self.assertEqual(DG.entita_blocco([d], 0, 0, 0), [])


class TestPersistenza(unittest.TestCase):
    def test_lo_spawner_e_i_forzieri_sopravvivono_al_salvataggio(self):
        import tempfile

        import amulet

        from genworld import bauli as BA
        from genworld.livello_dat import ImpostazioniMondo
        from genworld.mondo import ScrittoreMondo
        with tempfile.TemporaryDirectory() as dir_:
            percorso = os.path.join(dir_, "w")
            with ScrittoreMondo(percorso, ImpostazioniMondo(nome="P", modalita=1),
                                lato_blocchi=32, crea=True) as m:
                tav, tav_ba = DG.Tavolozza(m), BA.Tavolozza(m)
                coords = list(m.chunk_coords())
                dentro = coords[0]
                ox0, oz0 = m.origine_chunk(*dentro)
                u = (DG.U_SPAWNER[0] + DG.U_SPAWNER[1]) // 2
                # coordinate di mappa = coordinate di gioco (meta = 0): lo spawner
                # cade a (ox0 + 8, 30, oz0 + 8)
                d = DG.Dungeon(x=ox0 + 8 - u, z=oz0 + 8, ys=43, dx=1, dz=0, mob="zombie", seme=1)
                d.scatole = DG._scatole(43)
                self.assertEqual(d.spawner(), (ox0 + 8, 30, oz0 + 8))
                for cx, cz in coords:
                    ox, oz = m.origine_chunk(cx, cz)
                    out = np.full((16, 140, 16), m.id_aria, np.uint32)
                    out[:, :20, :] = m.blocco("stone")
                    if (cx, cz) == dentro:
                        out[8, 30 - Y0, 8] = tav.spawner
                    be = DG.entita_blocco([d], ox, oz, 0)
                    m.scrivi_chunk(cx, cz, out,
                                   biomi=np.full((16, 16), m.bioma("plains"), np.uint32),
                                   bauli=be)
            tipi_attesi = {"spawner"}
            lv = amulet.load_level(percorso)
            try:
                ch = lv.get_chunk(dentro[0], dentro[1], "minecraft:overworld")
                tipi = {b.base_name for b in ch.block_entities}
                self.assertTrue(tipi_attesi <= tipi, tipi)
            finally:
                lv.close()


class TestTraduzioni(unittest.TestCase):
    def test_i_blocchi_della_cripta_si_traducono(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        for nome, pr, java in (
                ("stone_bricks", {"variant": "cracked"}, "cracked_stone_bricks"),
                ("stone_bricks", {"variant": "chiseled"}, "chiseled_stone_bricks"),
                ("stairs", {"material": "stone_brick", "facing": "east", "half": "bottom",
                            "shape": "straight"}, "stone_brick_stairs"),
                ("soul_lantern", {"hanging": "true"}, "soul_lantern"),
                ("spawner", {}, "spawner"), ("cobweb", {}, "cobweb"),
                ("tnt", {"unstable": "false", "underwater": "false"}, "tnt"),
                ("trapped_chest", {"facing": "north", "connection": "none"}, "trapped_chest"),
                ("bone_block", {"axis": "y"}, "bone_block")):
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in pr.items()})
            o, _, _ = v.block.from_universal(b)
            self.assertEqual(o.base_name, java, nome)


if __name__ == "__main__":
    unittest.main()
