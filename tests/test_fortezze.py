"""Fortezze isolate (#6)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import fortezze as FZ
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


def terreno(lato=1200, quota=90):
    return np.full((lato, lato), quota, np.int32), np.zeros((lato, lato), bool)


def una_fortezza(direzione=(1, 0), ys=90, x=150, z=150):
    return FZ.Fortezza(x=x, z=z, ys=ys, dx=direzione[0], dz=direzione[1],
                       specie="vindicatore", nome="Barone Rotger", seme=3)


def posa_su_regione(f, quota=90):
    s = Finto()
    tav = FZ.Tavolozza(s)
    x0, z0, x1, z1 = f.bbox()
    regione = {}
    for cz in range(z0 // 16, z1 // 16 + 1):
        for cx in range(x0 // 16, x1 // 16 + 1):
            out = np.zeros((16, ALT, 16), np.uint32)
            ys = np.arange(Y0, Y0 + ALT)[None, :, None]
            out[np.broadcast_to(ys < quota, out.shape)] = s.blocco("stone")
            h = np.full((16, 16), quota, np.int32)
            FZ.posa(out, tav, h, cx * 16, cz * 16, Y0, [f], [0], meta=0)
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
    def test_trova_siti_su_un_pianoro_asciutto_lontani_fra_loro(self):
        h, acqua = terreno()
        fs = FZ.pianifica(h, acqua, seed=1)
        self.assertGreaterEqual(len(fs), 2)
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                self.assertGreaterEqual((a.x - b.x) ** 2 + (a.z - b.z) ** 2,
                                        FZ.DISTANZA_FRA_FORTEZZE ** 2)

    def test_non_nasce_su_acqua_su_terreno_accidentato_o_dove_si_evita(self):
        h, acqua = terreno()
        self.assertEqual(FZ.pianifica(h, np.ones_like(acqua), seed=1), [])
        irregolare = (80 + np.random.default_rng(0).integers(0, 12, h.shape)).astype(np.int32)
        self.assertEqual(FZ.pianifica(irregolare, acqua, seed=1), [])
        self.assertEqual(FZ.pianifica(h, acqua, evita=np.ones_like(acqua), seed=1), [])

    def test_e_deterministica_i_nomi_sono_diversi_e_la_densita_zero_spegne(self):
        h, acqua = terreno()
        a = [(f.x, f.z, f.nome, f.specie) for f in FZ.pianifica(h, acqua, seed=4)]
        b = [(f.x, f.z, f.nome, f.specie) for f in FZ.pianifica(h, acqua, seed=4)]
        self.assertEqual(a, b)
        nomi = [x[2] for x in a]
        self.assertEqual(len(nomi), len(set(nomi)))
        self.assertEqual(FZ.pianifica(h, acqua, densita=0.0, seed=4), [])


class TestPosa(unittest.TestCase):
    def setUp(self):
        self.f = una_fortezza()
        self.s, self.tav, self.b = posa_su_regione(self.f)

    def blocco(self, u, v, y):
        x, z = self.f.punto(u, v)
        return self.b(x, y, z)

    def test_la_cinta_chiude_il_perimetro_tranne_il_portale(self):
        f, F = self.f, FZ.F
        for y in (f.ys, f.ys + 3, f.ys + 6):
            for t in range(-F, F + 1):
                for (u, v) in ((t, F), (t, -F), (-F, t), (F, t)):
                    nome = self.blocco(u, v, y)
                    portale = (u == F and abs(v) <= 1 and y < f.ys + 4)
                    self.assertTrue(nome in ("stone_bricks", "bars") or portale,
                                    f"buco nella cinta ({u},{v}) y={y}: {nome}")

    def test_il_portale_e_alto_quattro_con_la_saracinesca_alzata_e_gli_stipiti(self):
        f, F = self.f, FZ.F
        for v in (-1, 0, 1):
            for y in range(f.ys, f.ys + 3):
                self.assertEqual(self.blocco(F, v, y), "air")
            self.assertEqual(self.blocco(F, v, f.ys + 3), "bars")
            self.assertEqual(self.blocco(F, v, f.ys + 5), "stone_bricks")      # l'architrave
        for v in (-2, 2):
            self.assertEqual(self.blocco(F, v, f.ys + 2), "stone_bricks")
            self.assertEqual(self.blocco(F + 1, v, f.ys + 3), "torch")
        for v in (-4, 4):
            self.assertEqual(self.blocco(F + 1, v, f.ys + 4), "wall_banner")

    def test_il_camminamento_e_libero_in_cima_alla_cinta_con_i_merli_fuori(self):
        f, F = self.f, FZ.F
        y = f.ys + FZ.ALTEZZA_MURA
        for u in range(-6, 7):
            self.assertEqual(self.blocco(u, F - 1, y), "air", u)
            self.assertEqual(self.blocco(u, F - 1, y - 1), "stone_bricks")
        merli = sum(1 for u in range(-F, F + 1) if self.blocco(u, F, y + 1) == "stone_bricks")
        self.assertGreater(merli, 8)

    def test_le_quattro_torri_d_angolo_hanno_due_piani_il_tetto_e_la_scala_a_pioli(self):
        f, F = self.f, FZ.F
        for su in (1, -1):
            for sv in (1, -1):
                # pareti piene
                self.assertEqual(self.blocco(su * (F - 4), sv * (F - 3), f.ys + 3), "stone_bricks")
                # il piano terra e' vuoto, il solaio e' di assi, il secondo piano e' vuoto
                self.assertEqual(self.blocco(su * (F - 2), sv * (F - 2), f.ys + 1), "air")
                self.assertEqual(self.blocco(su * (F - 2), sv * (F - 2), f.ys + FZ.ALTEZZA_MURA - 1),
                                 "planks")
                self.assertEqual(self.blocco(su * (F - 2), sv * (F - 2), f.ys + FZ.ALTEZZA_MURA + 1),
                                 "air")
                # la scala a pioli dal secondo piano al tetto
                for y in range(f.ys + FZ.ALTEZZA_MURA, f.ys + FZ.ALTEZZA_TORRE_ANGOLO):
                    self.assertEqual(self.blocco(su * (F - 1), sv * (F - 1), y), "ladder", y)
                # il palo con la lanterna in cima
                self.assertEqual(self.blocco(su * F, sv * F, f.ys + FZ.ALTEZZA_TORRE_ANGOLO + 3),
                                 "lantern")

    def test_le_torri_d_angolo_si_raggiungono_dal_cortile_e_dal_camminamento(self):
        f, F = self.f, FZ.F
        for su in (1, -1):
            for sv in (1, -1):
                self.assertEqual(self.blocco(su * (F - 4), sv * (F - 2), f.ys), "air")      # dal cortile
                self.assertEqual(self.blocco(su * (F - 4), sv * (F - 1), f.ys + FZ.ALTEZZA_MURA), "air")
                self.assertEqual(self.blocco(su * (F - 1), sv * (F - 4), f.ys + FZ.ALTEZZA_MURA), "air")

    def test_le_scale_di_accesso_al_camminamento_salgono_di_un_blocco_a_cella(self):
        f, F = self.f, FZ.F
        for sv in (1, -1):
            for j in range(7):
                v = sv * (3 + j)
                self.assertEqual(self.blocco(-F + 2, v, f.ys + j), "stairs", (sv, j))
                x, z = f.punto(-F + 2, v)
                self.assertEqual(self.b.proprieta(x, f.ys + j, z)["facing"],
                                 FZ.FACCIA[(f.dx * 0 - f.dz * sv, f.dz * 0 + f.dx * sv)], (sv, j))
                if j:
                    self.assertEqual(self.blocco(-F + 2, v, f.ys + j - 1), "stone_bricks")
            # in cima si scende sul camminamento
            self.assertEqual(self.blocco(-F + 1, sv * 9, f.ys + FZ.ALTEZZA_MURA - 1), "stone_bricks")
            self.assertEqual(self.blocco(-F + 1, sv * 9, f.ys + FZ.ALTEZZA_MURA), "air")

    def test_il_maschio_e_una_torre_curata_al_centro_col_suo_portone_verso_il_cortile(self):
        f = self.f
        m = f.maschio()
        self.assertEqual(m.dx, f.dx)
        x, z = m.punto(TR.R, 0)
        self.assertEqual(self.b(x, f.ys, z), "door")
        # la scala interna e gli arredi del maschio ci sono
        sx, sz = m.punto(-3, -3)
        self.assertEqual(self.b(sx, m.piano(0), sz), "stairs")
        visti = set()
        for u in range(-3, 4):
            for v in range(-3, 4):
                for y in range(m.piano(1), m.piano(1) + 3):
                    xx, zz = m.punto(u, v)
                    visti.add(self.b(xx, y, zz))
        self.assertIn("bed", visti)

    def test_il_cortile_e_allestito(self):
        f = self.f
        visti = set()
        for u in range(-12, 13):
            for v in range(-12, 13):
                for y in range(f.ys, f.ys + 4):
                    visti.add(self.blocco(u, v, y))
        for atteso in ("smithing_table", "anvil", "furnace", "grindstone", "barrel", "chest",
                       "hay_block", "fence", "campfire", "lantern", "wall_banner", "wall"):
            self.assertIn(atteso, visti, atteso)
        # il viale dal portale al maschio e' di ghiaia
        self.assertEqual(self.blocco(6, 0, f.ys - 1), "gravel")

    def test_il_pozzo_ha_acqua_chiusa_da_un_muretto(self):
        f = self.f
        self.assertEqual(self.blocco(6, 6, f.ys - 1), "water")
        for u, v in ((5, 6), (7, 6), (6, 5), (6, 7)):
            self.assertEqual(self.blocco(u, v, f.ys), "wall")
        self.assertEqual(self.blocco(6, 6, f.ys), "air")

    def test_i_forzieri_sono_tre_e_aperti(self):
        self.assertEqual(sorted(t for *_, t in self.tav.bauli + self.tav.torre.bauli),
                         ["stanza", "stanza", "tesoro"])
        for x, y, z, _ in self.tav.bauli + self.tav.torre.bauli:
            self.assertEqual(self.b(x, y, z), "chest")
            self.assertIn(self.b(x, y + 1, z), ("air", "slab"))      # la fucina ha il tetto a 3
        self.assertEqual(sorted(t for *_, t in self.f.forzieri()), ["stanza", "stanza", "tesoro"])

    def test_le_quattro_direzioni_producono_la_stessa_fortezza(self):
        for dire in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            f = una_fortezza(dire)
            _, tav, b = posa_su_regione(f)
            x, z = f.punto(FZ.F, 0)
            self.assertEqual(b(x, f.ys, z), "air", dire)
            self.assertEqual(b(x, f.ys + 3, z), "bars", dire)
            m = f.maschio()
            mx, mz = m.punto(TR.R, 0)
            self.assertEqual(b(mx, f.ys, mz), "door", dire)
            self.assertEqual(len(tav.bauli + tav.torre.bauli), 3, dire)

    def test_non_tocca_i_chunk_lontani(self):
        s = Finto()
        tav = FZ.Tavolozza(s)
        out = np.zeros((16, ALT, 16), np.uint32)
        out[:, :100, :] = s.blocco("stone")
        prima = out.copy()
        FZ.posa(out, tav, np.full((16, 16), 90, np.int32), 0, 0, Y0, [self.f], [], meta=0)
        self.assertTrue((out == prima).all())


class TestCombattenti(unittest.TestCase):
    def test_capo_guardie_soldati_e_arcieri(self):
        f = una_fortezza()
        c = FZ.combattenti([f], [(10, 10)])
        self.assertEqual(sum(1 for x in c if x.capo), 1)
        capo = next(x for x in c if x.capo)
        self.assertEqual(capo.nome, "Barone Rotger")
        self.assertEqual(capo.y, float(f.maschio().piano(TR.PIANI - 1)))
        arcieri = [x for x in c if x.y == float(f.ys + FZ.ALTEZZA_TORRE_ANGOLO)]
        self.assertEqual(len(arcieri), 4)
        self.assertTrue(all(a.specie == "scheletro" for a in arcieri))
        cortile = [x for x in c if x.y == float(f.ys)]
        self.assertEqual(len(cortile), 4)
        self.assertEqual(len(c), 3 + 4 + 4)

    def test_i_soldati_non_nascono_dentro_i_mobili_ne_nei_muri(self):
        f = una_fortezza()
        _, _, b = posa_su_regione(f)
        for x in FZ.combattenti([f]):
            if x.y == float(f.ys):
                self.assertEqual(b(int(x.x), int(x.y), int(x.z)), "air")
                self.assertEqual(b(int(x.x), int(x.y) + 1, int(x.z)), "air")
            if x.y == float(f.ys + FZ.ALTEZZA_TORRE_ANGOLO):
                self.assertEqual(b(int(x.x), int(x.y), int(x.z)), "air")


class TestIndiceEMaschera(unittest.TestCase):
    def test_indice_e_maschera(self):
        f = una_fortezza()
        idx = FZ.indice_per_chunk([f])
        self.assertIn(0, idx.get((f.x // 16, f.z // 16), []))
        for dx in (-FZ.F - 2, FZ.F + 2):
            self.assertIn(0, idx.get(((f.x + dx) // 16, f.z // 16), []))
        m = FZ.maschera([f], (400, 400), margine=2)
        self.assertTrue(m[f.z, f.x])
        self.assertTrue(m[f.z + FZ.F + 3, f.x])
        self.assertFalse(m[f.z + FZ.F + 6, f.x])


class TestTraduzioni(unittest.TestCase):
    def test_ogni_blocco_della_tavolozza_si_traduce_in_un_blocco_di_gioco(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        s = Finto()
        FZ.Tavolozza(s)
        cattivi = []
        for (nome, prop), _ in s.ids.items():
            if nome == "air":
                continue
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in prop})
            o, _, _ = v.block.from_universal(b)
            if o.namespace != "minecraft":
                cattivi.append((nome, dict(prop)))
        self.assertEqual(cattivi, [])


if __name__ == "__main__":
    unittest.main()
