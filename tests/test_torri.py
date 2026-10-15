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

    def test_i_quattro_piani_hanno_aria_e_un_solaio_sopra(self):
        t = self.t
        x, z = t.punto(1, 1)
        for k in range(TR.PIANI):
            base = t.piano(k)
            for y in range(base, base + TR.ALTEZZA_PIANO - 1):
                self.assertIn(self.b(x, y, z), ("air", "lantern"), (k, y))
            self.assertIn(self.b(x, base + TR.ALTEZZA_PIANO - 1, z),
                          ("planks", "stone_bricks"), k)

    def test_le_pareti_chiudono_il_perimetro_tranne_porta_e_feritoie(self):
        t = self.t
        for k in range(TR.PIANI):
            base = t.piano(k)
            for v in range(-TR.R, TR.R + 1):
                for u in (-TR.R, TR.R):
                    for w in (u, v), (v, u):
                        x, z = t.punto(*w)
                        nome = self.b(x, base + 2, z)
                        porta = (k == 0 and w == (TR.R, 0))
                        feritoia = False
                        self.assertTrue(nome in ("stone_bricks", "bars") or porta or feritoia,
                                        f"buco in parete {w} piano {k}: {nome}")

    def test_la_porta_e_un_varco_alto_due(self):
        t = self.t
        x, z = t.punto(TR.R, 0)
        self.assertEqual(self.b(x, t.ys, z), "air")
        self.assertEqual(self.b(x, t.ys + 1, z), "air")
        self.assertEqual(self.b(x, t.ys + 2, z), "stone_bricks")

    def test_la_scala_a_pioli_sale_dal_piano_terra_al_tetto_attraverso_i_solai(self):
        t = self.t
        x, z = t.punto(-TR.R + 1, 0)
        for y in range(t.ys, t.piano(TR.PIANI)):
            self.assertEqual(self.b(x, y, z), "ladder", f"manca un piolo a y={y}")

    def test_la_scala_e_appoggiata_al_muro_e_guarda_verso_l_interno(self):
        t = self.t
        x, z = t.punto(-TR.R, 0)
        self.assertEqual(self.b(x, t.ys + 2, z), "stone_bricks")

    def test_due_forzieri_uno_al_secondo_piano_uno_in_cima_aperti(self):
        self.assertEqual(sorted(tp for *_, tp in self.tav.bauli), ["stanza", "tesoro"])
        quote = {tp: y for _, y, _, tp in self.tav.bauli}
        self.assertEqual(quote["stanza"], self.t.piano(2))
        self.assertEqual(quote["tesoro"], self.t.piano(TR.PIANI - 1))
        for x, y, z, _ in self.tav.bauli:
            self.assertEqual(self.b(x, y, z), "chest")
            self.assertIn(self.b(x, y + 1, z), ("air", "lantern"))

    def test_merli_sul_tetto(self):
        t = self.t
        cima = t.piano(TR.PIANI)
        alti = sum(1 for u in range(-TR.R, TR.R + 1) for v in (-TR.R, TR.R)
                   if self.b(*(lambda p: (p[0], cima + 1, p[1]))(t.punto(u, v))) == "stone_bricks")
        self.assertGreater(alti, 2)

    def test_le_quattro_direzioni_producono_la_stessa_torre(self):
        for dire in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t = una_torre(dire)
            _, tav, b = posa_su_regione(t)
            x, z = t.punto(TR.R, 0)
            self.assertEqual(b(x, t.ys, z), "air", dire)
            self.assertEqual(len(tav.bauli), 2, dire)

    def test_non_tocca_i_chunk_lontani(self):
        s = Finto()
        tav = TR.Tavolozza(s)
        out = np.zeros((16, ALT, 16), np.uint32)
        out[:, :100, :] = s.blocco("stone")
        prima = out.copy()
        TR.posa(out, tav, np.full((16, 16), 90, np.int32), 0, 0, Y0, [self.t], [], meta=0)
        self.assertTrue((out == prima).all())


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

    def test_ogni_specie_ha_un_arma(self):
        for sp in TR.SPECIE:
            self.assertIn(sp, TR.ARMA)


class TestIndiceEMaschera(unittest.TestCase):
    def test_indice_e_maschera(self):
        t = una_torre()
        idx = TR.indice_per_chunk([t])
        self.assertIn(0, idx.get((t.x // 16, t.z // 16), []))
        m = TR.maschera([t], (300, 300), margine=2)
        self.assertTrue(m[t.z, t.x])
        self.assertTrue(m[t.z + TR.R + 3, t.x])
        self.assertFalse(m[t.z + TR.R + 6, t.x])


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
