"""Torri isolate con un capo (#6)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import torri as TR

Y0, ALT = -64, 320


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


def una_torre(direzione=(1, 0), ys=90, x=150, z=150):
    return TR.Torre(x=x, z=z, ys=ys, dx=direzione[0], dz=direzione[1],
                    specie="vindicatore", nome="Aldric il Cupo", seme=3)


def posa_su_regione(t, quota=90):
    s = Finto()
    tav = TR.Tavolozza(s)
    x0, z0, x1, z1 = t.bbox()
    regione = {}
    for cz in range(z0 // 16, z1 // 16 + 1):
        for cx in range(x0 // 16, x1 // 16 + 1):
            out = np.zeros((16, ALT, 16), np.uint32)
            ys = np.arange(Y0, Y0 + ALT)[None, :, None]
            out[np.broadcast_to(ys < quota, out.shape)] = s.blocco("stone")
            h = np.full((16, 16), quota, np.int32)
            TR.posa(out, tav, h, cx * 16, cz * 16, Y0, [t], [0], meta=0)
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


class TestPianifica(unittest.TestCase):
    def test_trova_siti_su_un_pianoro_asciutto_e_distanziati(self):
        h, acqua = terreno()
        ts = TR.pianifica(h, acqua, seed=1)
        self.assertGreaterEqual(len(ts), 2)
        for i, a in enumerate(ts):
            for b in ts[i + 1:]:
                self.assertGreaterEqual((a.x - b.x) ** 2 + (a.z - b.z) ** 2,
                                        TR.DISTANZA_FRA_TORRI ** 2)

    def test_non_nasce_su_acqua_su_terreno_accidentato_o_dove_si_evita(self):
        h, acqua = terreno()
        self.assertEqual(TR.pianifica(h, np.ones_like(acqua), seed=1), [])
        irregolare = (80 + np.random.default_rng(0).integers(0, 12, h.shape)).astype(np.int32)
        self.assertEqual(TR.pianifica(irregolare, acqua, seed=1), [])
        self.assertEqual(TR.pianifica(h, acqua, evita=np.ones_like(acqua), seed=1), [])

    def test_e_deterministica_e_la_densita_zero_spegne(self):
        h, acqua = terreno()
        a = [(t.x, t.z, t.specie, t.nome) for t in TR.pianifica(h, acqua, seed=4)]
        b = [(t.x, t.z, t.specie, t.nome) for t in TR.pianifica(h, acqua, seed=4)]
        self.assertEqual(a, b)
        self.assertEqual(TR.pianifica(h, acqua, densita=0.0, seed=4), [])


class TestPosa(unittest.TestCase):
    def setUp(self):
        self.t = una_torre()
        self.s, self.tav, self.b = posa_su_regione(self.t)

    def p(self, u, v):
        return self.t.punto(u, v)

    def blocco(self, u, v, y):
        x, z = self.p(u, v)
        return self.b(x, y, z)

    def proprieta(self, u, v, y):
        x, z = self.p(u, v)
        return self.b.proprieta(x, y, z)

    def test_i_quattro_piani_hanno_aria_in_una_cella_libera_e_un_solaio_sopra(self):
        t = self.t
        for k in range(TR.PIANI):
            base = t.piano(k)
            for y in range(base, base + TR.ALTEZZA_PIANO - 1):
                self.assertEqual(self.blocco(1, 2, y), "air", (k, y))
            self.assertIn(self.blocco(1, 2, base + TR.ALTEZZA_PIANO - 1),
                          ("planks", "stone_bricks"), k)

    def test_le_pareti_chiudono_il_perimetro_tranne_la_porta(self):
        t = self.t
        for k in range(TR.PIANI):
            base = t.piano(k)
            for u in range(-TR.R, TR.R + 1):
                for v in (-TR.R, TR.R):
                    for w in (u, v), (v, u):
                        nome = self.blocco(*w, base + 2)
                        porta = (k == 0 and w == (TR.R, 0))
                        self.assertTrue(nome in ("stone_bricks", "bars", "door") or porta,
                                        f"buco in parete {w} piano {k}: {nome}")

    def test_la_porta_ha_gli_stipiti_l_architrave_e_un_gradino(self):
        t, R = self.t, TR.R
        self.assertEqual(self.blocco(R, 0, t.ys), "door")
        self.assertEqual(self.blocco(R, 0, t.ys + 1), "door")
        self.assertEqual(self.blocco(R, 0, t.ys + 2), "stone_bricks")      # architrave
        for v in (-1, 1):
            for y in range(t.ys, t.ys + 3):
                self.assertEqual(self.blocco(R, v, y), "stone_bricks")
            self.assertEqual(self.blocco(R + 1, v, t.ys + 2), "torch")
        # davanti alla porta niente gradino: la soglia e' a livello, con due blocchi d'aria
        for v in (-1, 0, 1):
            self.assertEqual(self.blocco(R + 1, v, t.ys), "air")
            self.assertEqual(self.blocco(R + 1, v, t.ys + 1), "air")
        for v in (-3, 3):
            self.assertEqual(self.blocco(R + 1, v, t.ys + 3), "wall_banner")

    def test_lo_zoccolo_gira_attorno_alla_torre_tranne_l_ingresso(self):
        t, R = self.t, TR.R
        for u in range(-R, R + 1):
            self.assertEqual(self.blocco(u, R + 1, t.ys), "stairs", f"zoccolo u={u}")
            self.assertEqual(self.blocco(u, -R - 1, t.ys), "stairs")
        for v in range(-R, R + 1):
            self.assertEqual(self.blocco(-R - 1, v, t.ys), "stairs")
        for v in (-4, -3, -2, 2, 3, 4):
            self.assertEqual(self.blocco(R + 1, v, t.ys), "stairs")

    def test_il_tetto_ha_il_parafulmine_di_rame(self):
        t = self.t
        self.assertEqual(self.blocco(0, 0, t.piano(TR.PIANI)), "lightning_rod")
        self.assertEqual(self.proprieta(0, 0, t.piano(TR.PIANI))["material"], "copper")

    def test_la_scala_interna_sale_di_un_blocco_a_cella_con_posto_per_la_testa(self):
        t = self.t
        for k, ((u0, v0), (du, dv)) in TR.RAMPE.items():
            for j in range(5):
                u, v = u0 + du * j, v0 + dv * j
                y = t.piano(k) + j
                self.assertEqual(self.blocco(u, v, y), "stairs", (k, j))
                self.assertEqual(self.blocco(u, v, y + 2), "air", (k, j))

    def test_il_solaio_e_bucato_dove_la_scala_sale_e_intero_altrove(self):
        t = self.t
        for k, ((u0, v0), (du, dv)) in TR.RAMPE.items():
            y = t.piano(k) + 4
            for j in (1, 2, 3):
                self.assertEqual(self.blocco(u0 + du * j, v0 + dv * j, y), "air", (k, j))
            for j in (0,):
                self.assertNotEqual(self.blocco(u0 + du * j, v0 + dv * j, y), "air", (k, j))
            self.assertIn(self.blocco(1, 2, y), ("planks", "stone_bricks"))

    def test_le_scale_guardano_il_verso_della_salita(self):
        """`facing` di una scala e' il verso in cui si sale: il dorso alto sta di la'."""
        t = self.t
        for k, ((u0, v0), (du, dv)) in TR.RAMPE.items():
            atteso = TR.FACCIA[(t.dx * du - t.dz * dv, t.dz * du + t.dx * dv)]
            self.assertEqual(self.proprieta(u0, v0, t.piano(k))["facing"], atteso, k)

    def test_i_due_forzieri_sono_agli_angoli_lontani_dalle_rampe_e_aperti(self):
        t = self.t
        self.assertEqual(sorted(tp for *_, tp in self.tav.bauli), ["stanza", "tesoro"])
        quote = {tp: y for _, y, _, tp in self.tav.bauli}
        self.assertEqual(quote["stanza"], t.piano(2))
        self.assertEqual(quote["tesoro"], t.piano(TR.PIANI - 1))
        for x, y, z, _ in self.tav.bauli:
            self.assertEqual(self.b(x, y, z), "chest")
            self.assertEqual(self.b(x, y + 1, z), "air")
        x, _, z, _ = self.tav.bauli[0]
        celle_rampe = {self.p(u0 + du * j, v0 + dv * j)
                       for _, ((u0, v0), (du, dv)) in TR.RAMPE.items() for j in range(6)}
        self.assertNotIn((x, z), celle_rampe)

    def test_ogni_piano_e_allestito(self):
        t = self.t
        attesi = {0: ("barrel", "crafting_table", "furnace", "anvil", "carpet", "slab"),
                  1: ("bed", "barrel", "candle"),
                  2: ("fletching_table", "smithing_table", "grindstone", "anvil",
                      "cartography_table"),
                  3: ("bookshelf", "lectern", "carpet", "stairs")}
        for k, nomi in attesi.items():
            visti = set()
            for u in range(-3, 4):
                for v in range(-3, 4):
                    for y in range(t.piano(k), t.piano(k) + 3):
                        visti.add(self.blocco(u, v, y))
            for n in nomi:
                self.assertIn(n, visti, f"piano {k}: manca {n}")

    def test_i_letti_hanno_testa_e_piedi_e_la_testata_contro_il_muro(self):
        t = self.t
        y = t.piano(1)
        verso_muro = TR.FACCIA[(-t.dz * -1 + 0, t.dx * -1)]       # direzione di -v
        for u in (-2, 0, 2):
            self.assertEqual(self.blocco(u, -3, y), "bed")
            self.assertEqual(self.blocco(u, -2, y), "bed")
            testa, piedi = self.proprieta(u, -3, y), self.proprieta(u, -2, y)
            self.assertEqual(testa["part"], "head")
            self.assertEqual(piedi["part"], "foot")
            self.assertEqual(testa["facing"], verso_muro)

    def test_i_mobili_non_chiudono_le_celle_delle_guardie_ne_il_centro(self):
        t = self.t
        for k, (u, v) in TR.CELLE_GUARDIE.items():
            self.assertEqual(self.blocco(u, v, t.piano(k)), "air", k)
            self.assertEqual(self.blocco(u, v, t.piano(k) + 1), "air", k)
        self.assertEqual(self.blocco(0, 0, t.piano(TR.PIANI - 1) + 1), "air")

    def test_gli_stendardi_e_le_torce_stanno_attaccati_a_un_muro_pieno(self):
        t = self.t
        trovati = 0
        for y in range(t.ys, t.piano(TR.PIANI)):
            for u in range(-TR.R - 1, TR.R + 2):
                for v in range(-TR.R - 1, TR.R + 2):
                    n = self.blocco(u, v, y)
                    if n in ("torch", "wall_banner"):
                        trovati += 1
                        vicini = [self.blocco(u + a, v + b, y) for a, b in
                                  ((1, 0), (-1, 0), (0, 1), (0, -1))]
                        self.assertTrue(any(x in ("stone_bricks", "bookshelf") for x in vicini),
                                        f"{n} senza muro a ({u},{v},{y})")
        self.assertGreater(trovati, 15)

    def test_il_tetto_ha_pali_con_lanterne_agli_angoli_e_merli(self):
        t = self.t
        cima = t.piano(TR.PIANI)
        for u, v in ((TR.R, TR.R), (TR.R, -TR.R), (-TR.R, TR.R), (-TR.R, -TR.R)):
            self.assertEqual(self.blocco(u, v, cima + 1), "wall")
            self.assertEqual(self.blocco(u, v, cima + 3), "lantern")
        alti = sum(1 for u in range(-TR.R, TR.R + 1) for v in (-TR.R, TR.R)
                   if self.blocco(u, v, cima + 1) == "stone_bricks")
        self.assertGreater(alti, 2)

    def test_le_fasce_scolpite_marcano_i_piani_sul_filo(self):
        t = self.t
        for k in range(1, TR.PIANI):
            y = t.ys - 1 + TR.ALTEZZA_PIANO * k
            self.assertEqual(self.blocco(1, TR.R, y), "stone_bricks")
            self.assertEqual(self.proprieta(1, TR.R, y).get("variant"), "chiseled", k)

    def test_non_ci_sono_piu_scale_a_pioli(self):
        for u in range(-5, 6):
            for v in range(-5, 6):
                for y in range(self.t.ys, self.t.piano(TR.PIANI) + 4):
                    self.assertNotEqual(self.blocco(u, v, y), "ladder")

    def test_le_quattro_direzioni_producono_la_stessa_torre(self):
        for dire in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t = una_torre(dire)
            _, tav, b = posa_su_regione(t)
            x, z = t.punto(TR.R, 0)
            self.assertEqual(b(x, t.ys, z), "door", dire)
            self.assertEqual(len(tav.bauli), 2, dire)
            x, z = t.punto(-3, -3)
            self.assertEqual(b(x, t.piano(0), z), "stairs", dire)

    def test_non_tocca_i_chunk_lontani(self):
        s = Finto()
        tav = TR.Tavolozza(s)
        out = np.zeros((16, ALT, 16), np.uint32)
        out[:, :100, :] = s.blocco("stone")
        prima = out.copy()
        TR.posa(out, tav, np.full((16, 16), 90, np.int32), 0, 0, Y0, [self.t], [], meta=0)
        self.assertTrue((out == prima).all())


class TestVialetto(unittest.TestCase):
    """L'ingresso non deve mai essere tappato: la soglia e' a livello, e se il
    terreno e' piu' basso si scende a gradini con due blocchi d'aria sopra."""

    def prova(self, quota, direzione=(1, 0)):
        t = una_torre(direzione=direzione, ys=90)
        s, tav, b = posa_su_regione(t, quota=quota)
        p = lambda u, v: t.punto(u, v)
        return t, b, p

    def test_terreno_piano_si_entra_dritti(self):
        t, b, p = self.prova(90)
        for u in range(TR.R + 1, TR.R + 5):
            for v in (-1, 0, 1):
                x, z = p(u, v)
                self.assertEqual(b(x, t.ys, z), "air", (u, v))
                self.assertEqual(b(x, t.ys + 1, z), "air", (u, v))
                self.assertNotEqual(b(x, t.ys - 1, z), "air", (u, v))

    def test_terreno_piu_basso_scende_a_gradini_con_la_testa_libera(self):
        for direzione in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t, b, p = self.prova(87, direzione)
            R = TR.R
            # la soglia e' piena fino a ys-1 e a livello della porta
            for v in (-1, 0, 1):
                x, z = p(R + 1, v)
                self.assertNotEqual(b(x, t.ys - 1, z), "air")
                self.assertEqual(b(x, t.ys, z), "air")
            # i gradini: uno per blocco di dislivello, e sopra l'aria
            for i in (1, 2, 3):
                x, z = p(R + 1 + i, 0)
                self.assertEqual(b(x, t.ys - i, z), "stairs", (direzione, i))
                self.assertEqual(b(x, t.ys - i + 1, z), "air", (direzione, i))
                self.assertEqual(b(x, t.ys + 1, z), "air", (direzione, i))
                self.assertNotEqual(b(x, t.ys - i - 1, z), "air", (direzione, i))

    def test_i_gradini_salgono_verso_la_porta(self):
        t, b, p = self.prova(87)
        x, z = p(TR.R + 2, 0)
        self.assertEqual(b.proprieta(x, t.ys - 1, z)["facing"], TR.FACCIA[(-t.dx, -t.dz)])


class TestNomi(unittest.TestCase):
    def test_i_capi_hanno_nomi_diversi_finche_ce_ne_sono(self):
        h, acqua = terreno()
        ts = TR.pianifica(h, acqua, seed=2, densita=2.0)
        self.assertGreater(len(ts), 3)
        nomi = [t.nome for t in ts]
        self.assertEqual(len(nomi), len(set(nomi)))

    def test_esauriti_i_nomi_ne_compone_di_nuovi_senza_ripeterli(self):
        rng = np.random.default_rng(1)
        liberi = list(TR.NOMI_CAPO)
        usati = [TR._nome_unico(rng, liberi) for _ in range(len(TR.NOMI_CAPO) + 3)]
        self.assertEqual(len(usati[:len(TR.NOMI_CAPO)]), len(set(usati[:len(TR.NOMI_CAPO)])))
        self.assertEqual(liberi, [])
        self.assertTrue(all(isinstance(n, str) and n for n in usati))


class TestCombattenti(unittest.TestCase):
    def test_un_capo_in_cima_e_due_guardie(self):
        t = una_torre()
        c = TR.combattenti([t], [(10, 10)])
        self.assertEqual(len(c), 3)
        capo = next(x for x in c if x.capo)
        self.assertEqual(capo.y, float(t.piano(TR.PIANI - 1)))
        self.assertEqual(capo.nome, "Aldric il Cupo")
        guardie = [x for x in c if not x.capo]
        self.assertEqual(sorted(g.y for g in guardie), [float(t.piano(1)), float(t.piano(2))])

    def test_il_capo_vicino_e_di_ferro_e_lontano_di_diamante_con_incantesimi(self):
        t = una_torre()
        vicino = next(x for x in TR.combattenti([t], [(t.x, t.z)]) if x.capo)
        lontano = next(x for x in TR.combattenti([t], [(9999, 9999)]) if x.capo)
        self.assertEqual((vicino.livello, lontano.livello), (0, 1))
        nv, nl = vicino.nbt_tag(), lontano.nbt_tag()
        self.assertEqual(str(nv["equipment"]["head"]["id"]), "minecraft:iron_helmet")
        self.assertEqual(str(nl["equipment"]["head"]["id"]), "minecraft:diamond_helmet")
        self.assertNotIn("components", nv["equipment"]["head"])
        self.assertIn("components", nl["equipment"]["head"])
        self.assertGreater(float(nl["Health"]), float(nv["Health"]))

    def test_il_capo_ha_nome_visibile_vita_attributo_e_arma(self):
        t = una_torre()
        capo = next(x for x in TR.combattenti([t]) if x.capo)
        n = capo.nbt_tag()
        self.assertEqual(str(n["id"]), "minecraft:vindicator")
        self.assertEqual(str(n["CustomName"]), "Aldric il Cupo")
        self.assertEqual(int(n["CustomNameVisible"]), 1)
        self.assertEqual(str(n["equipment"]["mainhand"]["id"]), "minecraft:iron_axe")
        self.assertEqual(str(n["attributes"][0]["id"]), "minecraft:max_health")
        self.assertEqual(float(n["attributes"][0]["base"]), float(n["Health"]))
        for s in ("head", "chest", "legs", "feet"):
            self.assertIn(s, n["equipment"])

    def test_gli_id_di_gioco_e_le_guardie_senza_nome(self):
        for nostro, gioco in TR.SPECIE.items():
            n = TR.Combattente(x=1.0, y=2.0, z=3.0, specie=nostro, seme=1).nbt_tag()
            self.assertEqual(str(n["id"]), f"minecraft:{gioco}")
            self.assertNotIn("CustomName", n)

    def test_ogni_specie_che_puo_essere_capo_ha_un_arma(self):
        for sp in TR.CAPO_PER_ZONA:
            if sp not in TR.MAGHI:
                self.assertIn(sp, TR.ARMA)


class TestMago(unittest.TestCase):
    def mago(self):
        return TR.Torre(x=150, z=150, ys=90, dx=1, dz=0, specie="evocatore",
                        nome="Teofrasto", seme=3)

    def test_capo_evocatore_con_guardie_streghe_e_senza_armatura(self):
        t = self.mago()
        c = TR.combattenti([t], [(t.x, t.z)])
        capo = next(x for x in c if x.capo)
        n = capo.nbt_tag()
        self.assertEqual(str(n["id"]), "minecraft:evoker")
        self.assertEqual(str(n["CustomName"]), "Teofrasto")
        self.assertNotIn("head", n.get("equipment", {}))
        for g in (x for x in c if not x.capo):
            self.assertEqual(g.specie, "strega")
            self.assertEqual(str(g.nbt_tag()["id"]), "minecraft:witch")

    def test_il_nome_del_mago_viene_dalla_sua_lista(self):
        for s in range(40):
            torri = TR.pianifica(*terreno(), densita=12.0, seed=s)
            maghi = [t for t in torri if t.specie == "evocatore"]
            for t in maghi[:len(TR.NOMI_MAGO)]:
                self.assertIn(t.nome, TR.NOMI_MAGO)

    def test_laboratorio_e_biblioteca(self):
        t = self.mago()
        s, tav, blocco = posa_su_regione(t)
        nomi = {s.nome(i) for i in range(len(s.ids))}
        self.assertTrue({"brewing_stand", "cauldron", "enchanting_table", "bookshelf"} <= nomi, nomi)

    def test_i_forzieri_sono_del_mago(self):
        t = self.mago()
        s, tav, blocco = posa_su_regione(t)
        self.assertTrue(tav.bauli)
        self.assertTrue(all(tipo.startswith("mago_") for *_, tipo in tav.bauli))


class TestIndiceEMaschera(unittest.TestCase):
    def test_indice_e_maschera(self):
        t = una_torre()
        idx = TR.indice_per_chunk([t])
        self.assertIn(0, idx.get((t.x // 16, t.z // 16), []))
        m = TR.maschera([t], (300, 300), margine=2)
        self.assertTrue(m[t.z, t.x])
        self.assertTrue(m[t.z + TR.R + 3, t.x])
        self.assertFalse(m[t.z + TR.R + 6, t.x])


class TestTutteLeTraduzioni(unittest.TestCase):
    def test_ogni_blocco_della_tavolozza_si_traduce_in_un_blocco_di_gioco(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        s = Finto()
        TR.Tavolozza(s)
        cattivi = []
        for (nome, prop), _ in s.ids.items():
            if nome == "air":
                continue
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in prop})
            o, _, _ = v.block.from_universal(b)
            if o.namespace != "minecraft":
                cattivi.append((nome, dict(prop)))
        self.assertEqual(cattivi, [])


class TestTraduzioni(unittest.TestCase):
    def test_i_blocchi_si_traducono(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        for nome, pr, java in (("ladder", {"facing": "east"}, "ladder"),
                               ("planks", {"material": "oak"}, "oak_planks"),
                               ("bars", {"material": "iron", "north": "false", "south": "false",
                                         "east": "false", "west": "false"}, "iron_bars"),
                               ("lantern", {"material": "iron", "hanging": "true"}, "lantern")):
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in pr.items()})
            o, _, _ = v.block.from_universal(b)
            self.assertEqual(o.base_name, java, nome)


if __name__ == "__main__":
    unittest.main()
