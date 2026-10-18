"""Rovine con cripta (#5)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import dungeon as DG

Y0, ALT = -64, 320
# i blocchi che una creatura attraversa; con le trappole dei corridoi (filo e piastre)
STANZA_LIBERA = ("air", "cobweb", "soul_lantern", "spawner", "bone_block",
                 "tripwire", "pressure_plate")


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

    def proprieta(x, y, z):
        out = regione.get((x // 16, z // 16))
        if out is None:
            return {}
        i = int(out[x % 16, y - Y0, z % 16])
        return next((dict(p) for (n, p), v in s.ids.items() if v == i), {})
    blocco.proprieta = proprieta
    return s, tav, blocco


class TestLayoutVariato(unittest.TestCase):
    """Snodi, sale fuori asse, cunicoli e stanze laterali: il percorso resta percorribile."""

    @classmethod
    def setUpClass(cls):
        h, acqua = terreno(900)
        cls.ds = DG.pianifica(h, acqua, densita=40.0, seed=3)

    def test_ci_sono_snodi_e_stanze_laterali(self):
        self.assertGreaterEqual(len(self.ds), 8)
        fuori_asse = [d for d in self.ds if d.vs != 0 or d.vt != 0]
        self.assertGreaterEqual(len(fuori_asse), len(self.ds) // 2)
        self.assertTrue(any(d.laterali for d in self.ds))
        self.assertTrue(any(s.tipo == "cunicolo" for d in self.ds for s in d.scatole))
        self.assertTrue(any(sum(1 for s in d.scatole if s.tipo == "corridoio") > 2 for d in self.ds))

    def test_i_primi_tre_forzieri_sono_sempre_stanza_stanza_tesoro(self):
        for d in self.ds:
            tipi = [t for *_, t in d.forzieri()]
            self.assertEqual(tipi[:3], ["stanza", "stanza", "tesoro"])
            self.assertEqual(len(tipi), 3 + len(d.laterali))

    def test_le_scatole_non_si_sovrappongono_se_non_dove_si_connettono(self):
        for d in self.ds:
            stanze = [s for s in d.scatole if s.tipo in ("anticamera", "spawner", "tesoro", "stanza_laterale")]
            for i, a in enumerate(stanze):
                for b in stanze[i + 1:]:
                    self.assertTrue(a.u1 < b.u0 or b.u1 < a.u0 or a.v1 < b.v0 or b.v1 < a.v0, (a, b))

    def test_dallo_scalino_si_raggiunge_tutto(self):
        """Si stampa la cripta e si fa una visita: dalla fine della scala si arriva allo
        spawner, al forziere del tesoro e a ogni forziere delle stanze laterali, passando
        per celle libere con due blocchi d'aria sopra."""
        for d in self.ds[:8]:
            s, tav, b = posa_su_regione(d, quota=d.ys)
            yf = d.yf
            libere = ("air", "cobweb", "soul_lantern", "bone_block", "spawner", "chest", "trapped_chest",
                      "tripwire", "pressure_plate")

            def libera(x, z):
                return all(b(x, y, z) in libere for y in (yf, yf + 1))
            ux, uz = d.punto(DG.PROFONDITA + 1, 0)
            visti, coda = {(ux, uz)}, [(ux, uz)]
            while coda:
                x, z = coda.pop()
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (x + dx, z + dz)
                    if n not in visti and libera(*n):
                        visti.add(n)
                        coda.append(n)
            sx, sy, sz = d.spawner()
            self.assertTrue(any(abs(sx - x) + abs(sz - z) <= 3 for x, z in visti), "spawner")   # c'e' il piedistallo
            for fx, fy, fz, tipo in d.forzieri():
                vicino = any((fx + dx, fz + dz) in visti for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                self.assertTrue(vicino, f"forziere {tipo} irraggiungibile in {d.x},{d.z}")

    def test_le_trappole_sono_deterministiche_e_varie(self):
        viste = set()
        for d in self.ds:
            assert d.trappole() == d.trappole()
            tipi = [t for t, *_ in d.trappole()]
            self.assertGreaterEqual(len(tipi), 1, "almeno una trappola per cripta")
            self.assertLessEqual(tipi.count("tnt"), 1)
            viste |= set(tipi)
        self.assertGreaterEqual(len(viste), 5, viste)
        self.assertTrue(viste <= set(DG.TIPI_TRAPPOLA))

    def test_ogni_trappola_si_posa_come_dovrebbe(self):
        for d in self.ds[:10]:
            s, tav, b = posa_su_regione(d, quota=d.ys)
            yf = d.yf
            for tipo, u, vc in d.trappole():
                x, z = d.punto(u, vc)
                if tipo == "dardi":
                    self.assertEqual(b(x, yf, z), "tripwire")
                    for lato in (1, -1):
                        xd, zd = d.punto(u, vc + 2 * lato)
                        self.assertEqual(b(xd, yf + 1, zd), "dispenser")
                        xh, zh = d.punto(u, vc + lato)
                        self.assertEqual(b(xh, yf, zh), "tripwire_hook")
                elif tipo == "fossa":
                    xs, zs = d.punto(u, vc)
                    self.assertEqual(b(xs, yf - 4, zs), "pointed_dripstone")
                    self.assertEqual(b(xs, yf, zs), "air")
                    xl, zl = d.punto(u, vc - 1)
                    self.assertEqual(b(xl, yf - 1, zl), "stone_bricks")          # la sponda tiene
                elif tipo == "ragnatele":
                    self.assertEqual(b(x, yf, z), "cobweb")
                elif tipo == "magma":
                    self.assertEqual(b(x, yf - 1, z), "magma_block")
                elif tipo == "infestati":
                    xl, zl = d.punto(u, vc + 2)
                    self.assertEqual(b(xl, yf, zl), "infested_block")
                elif tipo == "tnt":
                    self.assertEqual(b(x, yf - 1, z), "tnt")
                    self.assertEqual(b(x, yf, z), "pressure_plate")

    def test_i_distributori_hanno_il_blockentity(self):
        for d in self.ds[:10]:
            attesi = d.distributori()
            trovati = []
            for cz in range(d.bbox()[1] // 16, d.bbox()[3] // 16 + 1):
                for cx in range(d.bbox()[0] // 16, d.bbox()[2] // 16 + 1):
                    trovati += [(be.x, be.y, be.z) for be in DG.entita_blocco([d], cx * 16, cz * 16, 0)
                                if be.base_name == "dispenser"]
            self.assertEqual(sorted(trovati), sorted((x, y, z) for x, y, z, _ in attesi))

    def test_gli_spawner_sono_meno_asfissianti_del_gioco(self):
        be = DG.be_spawner(0, 0, 0, "zombie", misto=True).nbt.compound["utags"]
        self.assertLessEqual(int(be["SpawnCount"]), 2)
        self.assertLessEqual(int(be["MaxNearbyEntities"]), 3)
        self.assertLessEqual(int(be["RequiredPlayerRange"]), 10)
        self.assertGreaterEqual(int(be["MinSpawnDelay"]), 600)
        self.assertGreaterEqual(int(be["MaxSpawnDelay"]), 1500)
        self.assertLessEqual(int(be["SpawnRange"]), 3)

    def test_il_dritto_resta_valido_per_chi_costruisce_a_mano(self):
        d = un_dungeon()
        self.assertEqual((d.vs, d.vt, d.laterali), (0, 0, []))
        self.assertEqual([t for *_, t in d.forzieri()], ["stanza", "stanza", "tesoro"])


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

    def test_le_scale_salgono_verso_l_ingresso(self):
        """`facing` di una scala e' il verso in cui si sale: scendendo verso +d
        le scale guardano -d. Con il verso sbagliato il dorso alto sta davanti."""
        d = self.d
        atteso = DG.FACCIA[(-d.dx, -d.dz)]
        for u in range(1, DG.PROFONDITA + 1):
            x, z = d.punto(u, 0)
            self.assertEqual(self.b.proprieta(x, d.ys - u, z)["facing"], atteso, u)
        for dire in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dd = un_dungeon(dire)
            _, _, b = posa_su_regione(dd)
            x, z = dd.punto(5, 0)
            self.assertEqual(b.proprieta(x, dd.ys - 5, z)["facing"],
                             DG.FACCIA[(-dire[0], -dire[1])], dire)

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
        be = [b for b in DG.entita_blocco([d], (sx // 16) * 16, (sz // 16) * 16, 0)
              if b.base_name == "spawner"]
        self.assertEqual(len(be), 1)
        self.assertEqual((be[0].x, be[0].y, be[0].z), (sx, sy, sz))
        ut = be[0].nbt.compound["utags"]
        self.assertEqual(str(ut["SpawnData"]["entity"]["id"]), "minecraft:skeleton")
        self.assertEqual(str(ut["SpawnPotentials"][0]["data"]["entity"]["id"]),
                         "minecraft:skeleton")
        self.assertEqual(DG.entita_blocco([d], 0, 0, 0), [])

    def test_lo_spawner_ha_dei_compagni(self):
        be = DG.be_spawner(0, 0, 0, "skeleton", misto=True)
        pot = be.nbt.compound["utags"]["SpawnPotentials"]
        ids = [str(p["data"]["entity"]["id"]) for p in pot]
        self.assertEqual(ids[0], "minecraft:skeleton")
        self.assertEqual(int(pot[0]["weight"]), 4)
        self.assertEqual(ids[1:], [f"minecraft:{m}" for m in DG.COMPAGNI["skeleton"]])
        semplice = DG.be_spawner(0, 0, 0, "skeleton").nbt.compound["utags"]["SpawnPotentials"]
        self.assertEqual(len(semplice), 1)


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
            # il file region, letto a mano: e' quello che legge il gioco
            import glob
            import struct
            import zlib

            import amulet_nbt as nbt
            trovati = []
            for f in glob.glob(os.path.join(percorso, "**", "*.mca"), recursive=True):
                dati = open(f, "rb").read()
                for i in range(1024):
                    off = struct.unpack(">I", b"\x00" + dati[i * 4:i * 4 + 3])[0]
                    if not off:
                        continue
                    pos = off * 4096
                    ln = struct.unpack(">I", dati[pos:pos + 4])[0]
                    tag = nbt.load(zlib.decompress(dati[pos + 5:pos + 4 + ln]),
                                   compressed=False, little_endian=False)
                    for be in tag.compound["block_entities"]:
                        if "spawner" in str(be["id"]):
                            trovati.append(be)
            self.assertEqual(len(trovati), 1)
            be = trovati[0]
            self.assertEqual(str(be["SpawnData"]["entity"]["id"]), "minecraft:zombie")
            self.assertEqual(len(be["SpawnPotentials"]), 1 + len(DG.COMPAGNI["zombie"]))
            self.assertEqual(str(be["SpawnPotentials"][0]["data"]["entity"]["id"]),
                             "minecraft:zombie")
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
