"""Alberi per bioma (#27): acacia, rovere scuro, giungla, abeti alti, ciliegi."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import biomi as B
from genworld import mappa as M
from genworld import vegetazione as V


class Finta:
    def __init__(self):
        self.tronco = {sp: 100 + sp for sp in V.ALBERI}
        self.foglie = {sp: 200 + sp for sp in V.ALBERI}
        self.cactus, self.arbusto = 30, 31
        self.erba_alta, self.felce = 32, 33
        self.torchflower = 34
        self.azalea, self.azalea_fiorita, self.radici = 35, 36, 37
        self.liana = {f: 700 + i for i, f in enumerate(('north', 'south', 'east', 'west'))}
        self.bambu = {'none': 800, 'small': 801, 'large': 802}
        self.sasso = (900, 901, 902, 903)
        self.tronco_a_terra = {(m, a): 950 + i for i, (m, a) in enumerate(
            (m, a) for m in ('oak', 'birch', 'spruce') for a in ('x', 'z'))}
        self.tappeto_muschio = 960
        self.secca, self.secca_alta = 970, 971
        self.brocca = (975, 976)
        self.lucciole = 977
        self.selvatici = {(f, n): 1100 + 10 * i + n for i, f in enumerate(('north', 'south', 'east', 'west')) for n in (1, 2, 3, 4)}
        self.lettiera = {(f, n): 980 + 10 * i + n for i, f in enumerate(('north', 'south', 'east', 'west')) for n in (1, 2, 3, 4)}
        self.fungo = {V.FUNGO_MARRONE: 961, V.FUNGO_ROSSO: 962}
        self.doppio = {sp: (300 + sp, 400 + sp) for sp in V.DOPPI}
        self.gambo = 500
        self.cappello = {sp: 600 + sp for sp in V.FUNGHI_GIGANTI}
        self.aria = 0


def albero(sp, alt, x=8, z=8):
    return {"x": np.array([x]), "z": np.array([z]), "y": np.array([70]),
            "specie": np.array([sp]), "altezza": np.array([alt]),
            "seme": np.array([7])}


def disegnato(sp, alt):
    out = np.zeros((16, 140, 16), np.uint32)
    V.disegna(out, 0, 0, 0, albero(sp, alt), [0], Finta())
    return out


class TestSemina(unittest.TestCase):
    def test_ogni_bioma_semina_le_sue_specie(self):
        lato = 160
        cls = np.full((lato, lato), M.FORESTA, np.uint8)
        h = np.full((lato, lato), 80, np.int32)
        for nome, attese in (("savanna", V.ACACIA), ("dark_forest", V.ROVERE_SCURO),
                             ("jungle", V.GIUNGLA), ("meadow", V.CILIEGIO),
                             ("giant_spruce_taiga", V.ABETE_ALTO)):
            biomi = np.full((lato, lato), B.INDICE[nome], np.uint8)
            a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi)
            self.assertIn(attese, set(a["specie"].tolist()), nome)

    def test_senza_biomi_nessuna_specie_nuova(self):
        cls = np.full((120, 120), M.FORESTA, np.uint8)
        h = np.full((120, 120), 80, np.int32)
        a = V.semina(cls, h, livello_mare=62, seed=3)
        nuove = {V.ACACIA, V.ROVERE_SCURO, V.GIUNGLA, V.ABETE_ALTO, V.CILIEGIO}
        self.assertFalse(nuove & set(a["specie"].tolist()))

    def test_altezze_nei_limiti(self):
        cls = np.full((160, 160), M.FORESTA, np.uint8)
        h = np.full((160, 160), 80, np.int32)
        biomi = np.full((160, 160), B.INDICE["jungle"], np.uint8)
        a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi)
        g = a["specie"] == V.GIUNGLA
        lo, hi = V.ALTEZZA[V.GIUNGLA]
        self.assertTrue(((a["altezza"][g] >= lo) & (a["altezza"][g] <= hi)).all())


class TestDisegno(unittest.TestCase):
    def test_acacia_ha_chioma_piatta_e_larga(self):
        out = disegnato(V.ACACIA, 7)
        f = 200 + V.ACACIA
        strati = [(out[:, y, :] == f).sum() for y in range(70, 90)]
        self.assertGreaterEqual(max(strati), 15)
        self.assertLessEqual(sum(1 for s in strati if s), 4)

    def test_rovere_scuro_ha_tronco_2x2(self):
        out = disegnato(V.ROVERE_SCURO, 7)
        self.assertEqual(int((out[:, 71, :] == 100 + V.ROVERE_SCURO).sum()), 4)

    def test_giungla_e_alta_e_a_piu_livelli(self):
        out = disegnato(V.GIUNGLA, 14)
        f = 200 + V.GIUNGLA
        con_foglie = [y for y in range(70, 100) if (out[:, y, :] == f).any()]
        self.assertGreater(max(con_foglie) - min(con_foglie), 6)
        self.assertGreater(len(con_foglie), 5)
        self.assertTrue((out[8, 70:84, 8] == 100 + V.GIUNGLA).all())

    def test_abete_alto_supera_l_abete(self):
        a = disegnato(V.ABETE_ALTO, 20)
        self.assertTrue((a[8, 70:89, 8] == 100 + V.ABETE_ALTO).all())

    def test_nessun_disegno_esce_dal_raggio(self):
        for sp, alt in ((V.ACACIA, 7), (V.ROVERE_SCURO, 8), (V.GIUNGLA, 14)):
            out = disegnato(sp, alt)
            xs, _, zs = np.nonzero(out)
            self.assertLessEqual(max(abs(xs - 8).max(), abs(zs - 8).max()), V.RAGGIO_MAX, sp)


class TestPianteNuove(unittest.TestCase):
    def test_le_piante_alte_hanno_due_meta(self):
        for sp in V.DOPPI:
            out = disegnato(sp, 2)
            self.assertEqual(int(out[8, 70, 8]), 300 + sp)
            self.assertEqual(int(out[8, 71, 8]), 400 + sp)

    def test_il_fungo_gigante_ha_gambo_e_cappello_largo(self):
        for sp in V.FUNGHI_GIGANTI:
            out = disegnato(sp, 6)
            self.assertTrue((out[8, 70:75, 8] == 500).all())
            cap = (out == 600 + sp)
            self.assertGreaterEqual(int(cap.sum()), 14)
            xs, _, zs = np.nonzero(cap)
            self.assertGreaterEqual(xs.max() - xs.min(), 3)

    def test_ogni_bioma_nuovo_semina_le_sue_piante(self):
        cls = np.full((200, 200), M.PRATERIA, np.uint8)
        h = np.full((200, 200), 80, np.int32)
        for nome, attese in (("sunflower_plains", V.GIRASOLE), ("taiga", V.FELCE_GRANDE),
                             ("meadow", V.LILLA), ("dark_forest", V.FUNGO_GIGANTE_M),
                             ("savanna", V.TORCHFLOWER)):
            biomi = np.full((200, 200), B.INDICE[nome], np.uint8)
            a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi, scala_densita=3.0)
            self.assertIn(attese, set(a["specie"].tolist()), nome)

    def test_ogni_specie_ha_altezza_e_nome(self):
        for sp in range(41):
            self.assertIn(sp, V.ALTEZZA)
            self.assertIn(sp, V.NOMI_SPECIE)


class TestSeccoELettiera(unittest.TestCase):
    def test_erba_secca_nel_deserto_e_nella_savana(self):
        cls = np.full((200, 200), M.DESERTO, np.uint8)
        h = np.full((200, 200), 80, np.int32)
        a = V.semina(cls, h, livello_mare=62, seed=3, scala_densita=4.0)
        spec = set(a["specie"].tolist())
        self.assertIn(V.SECCA, spec)
        self.assertIn(V.SECCA_ALTA, spec)
        biomi = np.full((200, 200), B.INDICE["savanna"], np.uint8)
        cls2 = np.full((200, 200), M.PRATERIA, np.uint8)
        a = V.semina(cls2, h, livello_mare=62, seed=3, biomi=biomi, scala_densita=3.0)
        self.assertIn(V.SECCA, set(a["specie"].tolist()))

    def test_lettiera_nei_boschi(self):
        cls = np.full((200, 200), M.FORESTA, np.uint8)
        h = np.full((200, 200), 80, np.int32)
        self.assertIn(V.LETTIERA, set(V.semina(cls, h, livello_mare=62, seed=3)["specie"].tolist()))
        biomi = np.full((200, 200), B.INDICE["dark_forest"], np.uint8)
        a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi, scala_densita=3.0)
        self.assertIn(V.LETTIERA, set(a["specie"].tolist()))

    def test_la_pianta_brocca_nella_giungla_e_alta_due(self):
        biomi = np.full((200, 200), B.INDICE["jungle"], np.uint8)
        cls = np.full((200, 200), M.FORESTA, np.uint8)
        h = np.full((200, 200), 80, np.int32)
        a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi, scala_densita=3.0)
        self.assertIn(V.BROCCA, set(a["specie"].tolist()))
        out = disegnato(V.BROCCA, 2)
        self.assertEqual(int((out == 975).sum()), 1)
        self.assertEqual(int((out == 976).sum()), 1)

    def test_lucciole_e_fiori_selvatici_per_bioma(self):
        h = np.full((200, 200), 80, np.int32)
        cls = np.full((200, 200), M.FORESTA, np.uint8)
        for nome, attesa in (("dark_forest", V.LUCCIOLE), ("mangrove_swamp", V.LUCCIOLE),
                             ("meadow", V.FIORI_SELVATICI)):
            biomi = np.full((200, 200), B.INDICE[nome], np.uint8)
            a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi, scala_densita=3.0)
            self.assertIn(attesa, set(a["specie"].tolist()), nome)
        prato = np.full((200, 200), M.PRATERIA, np.uint8)
        a = V.semina(prato, h, livello_mare=62, seed=3, scala_densita=4.0)
        self.assertIn(V.FIORI_SELVATICI, set(a["specie"].tolist()))

    def test_si_disegnano_lucciole_e_fiori_selvatici(self):
        self.assertEqual(int((disegnato(V.LUCCIOLE, 1) == 977).sum()), 1)
        out = disegnato(V.FIORI_SELVATICI, 1)
        n = sum(int((out == v).sum()) for v in set(Finta().selvatici.values()))
        self.assertTrue(1 <= n <= 6)

    def test_si_disegnano(self):
        self.assertEqual(int((disegnato(V.SECCA, 1) == 970).sum()), 1)
        self.assertEqual(int((disegnato(V.SECCA_ALTA, 1) == 971).sum()), 1)
        lett = disegnato(V.LETTIERA, 1)
        nomi = {v for v in np.unique(lett) if 980 <= v <= 1030}
        self.assertGreaterEqual(int(sum((lett == v).sum() for v in nomi)), 1)
        self.assertLessEqual(int(sum((lett == v).sum() for v in nomi)), 5)

    def test_la_traduzione_esiste(self):
        from genworld.vegetazione import Tavolozza
        class S:
            id_aria = 0
            def __init__(self): self.n = {}
            def blocco(self, nome, **p): return self.n.setdefault((nome, tuple(sorted(p.items()))), len(self.n) + 1)
        s = S(); Tavolozza(s)
        self.assertIn(("leaf_litter", (("facing", "north"), ("segment_amount", "1"))), s.n)
        self.assertIn(("plant", (("plant_type", "short_dry_grass"),)), s.n)


class TestPalude(unittest.TestCase):
    def test_la_mangrovia_ha_radici_al_piede_e_chioma_larga(self):
        out = disegnato(V.MANGROVIA, 6)
        self.assertTrue((out[8, 70:76, 8] == 100 + V.MANGROVIA).all())
        self.assertGreaterEqual(int((out[:, 70, :] == 37).sum()), 1)
        self.assertGreaterEqual(int((out == 200 + V.MANGROVIA).sum()), 25)

    def test_la_palude_semina_mangrovie_e_azalee(self):
        cls = np.full((200, 200), M.PIANURA, np.uint8)
        h = np.full((200, 200), 64, np.int32)
        biomi = np.full((200, 200), B.INDICE["mangrove_swamp"], np.uint8)
        a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi, scala_densita=3.0)
        sp = set(a["specie"].tolist())
        self.assertIn(V.MANGROVIA, sp)
        self.assertIn(V.AZALEA, sp)

    def test_il_bioma_palude_nasce_nelle_terre_basse_e_non_in_alto(self):
        cls = np.full((200, 200), M.PIANURA, np.uint8)
        basso = np.full((200, 200), 64, np.int32)
        alto = np.full((200, 200), 90, np.int32)
        idx = B.INDICE["mangrove_swamp"]
        self.assertGreater(int((B.assegna(cls, basso) == idx).sum()), 1000)
        self.assertEqual(int((B.assegna(cls, alto) == idx).sum()), 0)

    def test_il_bioma_palude_traduce_e_ha_il_legno(self):
        self.assertNotIn("mangrove_swamp", B.verifica_traduzioni((26, 2, 0)))
        self.assertEqual(B.LEGNO_BIOMA["mangrove_swamp"], "mangrove")


class TestBambuELiane(unittest.TestCase):
    def test_il_bambu_e_un_gruppo_di_canne_con_foglie_in_cima(self):
        out = disegnato(V.BAMBU, 9)
        canne = (out >= 800) & (out <= 802)
        self.assertGreaterEqual(int(canne.any(axis=1).sum()), 1)
        self.assertTrue((out[8, 70, 8] == 800))
        self.assertIn(802, set(np.unique(out).tolist()))
        self.assertGreaterEqual(int(canne.sum()), 7)

    def test_la_giungla_ha_liane_attaccate_al_tronco(self):
        visti = 0
        for seed in range(30):
            out = np.zeros((16, 140, 16), np.uint32)
            alb = albero(V.GIUNGLA, 14)
            alb["seme"] = np.array([seed])
            V.disegna(out, 0, 0, 0, alb, [0], Finta())
            for f, idv in Finta().liana.items():
                for lx, ly, lz in zip(*np.nonzero(out == idv)):
                    visti += 1
                    dx, dz = {"west": (-1, 0), "east": (1, 0), "north": (0, -1),
                              "south": (0, 1)}[f]
                    self.assertEqual(int(out[lx + dx, ly, lz + dz]), 100 + V.GIUNGLA,
                                     "liana non attaccata al tronco")
        self.assertGreater(visti, 10)

    def test_la_giungla_semina_bambu(self):
        cls = np.full((200, 200), M.FORESTA, np.uint8)
        h = np.full((200, 200), 80, np.int32)
        biomi = np.full((200, 200), B.INDICE["jungle"], np.uint8)
        a = V.semina(cls, h, livello_mare=62, seed=3, biomi=biomi, scala_densita=3.0)
        self.assertIn(V.BAMBU, set(a["specie"].tolist()))


class TestSassiETronchi(unittest.TestCase):
    def test_il_sasso_e_un_grumo_appoggiato_con_i_buchi_sotto_riempiti(self):
        out = disegnato(V.SASSO, 4)
        sassi = (out >= 900) & (out <= 903)
        self.assertGreaterEqual(int(sassi.sum()), 6)
        self.assertTrue((out[8, 70:72, 8] >= 900).any())
        # nessun blocco di sasso e' piu' alto di due sopra la base
        ys = np.nonzero(sassi.any(axis=(0, 2)))[0]
        self.assertLessEqual(int(ys.max() - ys.min()), 2)

    def test_il_tronco_caduto_e_orizzontale_lungo_alt_con_muschio_o_funghi(self):
        out = np.zeros((16, 140, 16), np.uint32)
        V.disegna(out, 0, 0, 0, albero(V.TRONCO_CADUTO, 5), [0], Finta())
        tronchi = (out >= 950) & (out <= 955)
        xs, ys, zs = np.nonzero(tronchi)
        self.assertEqual(len(xs), 5)
        self.assertEqual(len(set(ys.tolist())), 1)                 # tutto alla stessa quota
        self.assertTrue(len(set(xs.tolist())) == 5 or len(set(zs.tolist())) == 5)

    def test_nei_boschi_si_seminano_sassi_e_tronchi_ma_non_in_pianura(self):
        cls = np.full((250, 250), M.FORESTA, np.uint8)
        h = np.full((250, 250), 80, np.int32)
        a = V.semina(cls, h, livello_mare=62, seed=3, scala_densita=2.0)
        sp = a["specie"].tolist()
        self.assertGreater(sp.count(V.SASSO), 5)
        self.assertGreater(sp.count(V.TRONCO_CADUTO), 3)
        cls[:] = M.PIANURA
        b = V.semina(cls, h, livello_mare=62, seed=3, scala_densita=2.0)
        self.assertNotIn(V.SASSO, b["specie"].tolist())
        self.assertNotIn(V.TRONCO_CADUTO, b["specie"].tolist())

    def test_i_sassi_stanno_anche_in_montagna_i_tronchi_solo_nel_bosco(self):
        cls = np.full((250, 250), M.MONTAGNA, np.uint8)
        h = np.full((250, 250), 100, np.int32)
        a = V.semina(cls, h, livello_mare=62, seed=3, scala_densita=3.0)
        self.assertIn(V.SASSO, a["specie"].tolist())
        self.assertNotIn(V.TRONCO_CADUTO, a["specie"].tolist())


class TestTraduzioni(unittest.TestCase):
    def test_i_blocchi_nuovi_si_traducono(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        lati = {"north": "true", "south": "true", "east": "true", "west": "true"}
        for nome, pr, java in (
                ("double_plant", {"plant_type": "sunflower", "half": "upper"}, "sunflower"),
                ("double_plant", {"plant_type": "large_fern", "half": "lower"}, "large_fern"),
                ("double_plant", {"plant_type": "lilac", "half": "lower"}, "lilac"),
                ("double_plant", {"plant_type": "rose_bush", "half": "lower"}, "rose_bush"),
                ("double_plant", {"plant_type": "peony", "half": "lower"}, "peony"),
                ("torchflower", {}, "torchflower"),
                ("moss_carpet", {}, "moss_carpet"),
                ("log", {"axis": "x", "material": "birch", "stripped": "false"}, "birch_log"),
                ("andesite", {"polished": "false"}, "andesite"),
                ("bamboo", {"stage": "0", "leaves": "large", "thickness": "thick"}, "bamboo"),
                ("vine", {"north": "true", "south": "false", "east": "false",
                          "west": "false", "up": "false"}, "vine"),
                ("azalea", {}, "azalea"), ("flowering_azalea", {}, "flowering_azalea"),
                ("mangrove_roots", {}, "mangrove_roots"), ("mud", {}, "mud"),
                ("log", {"material": "mangrove", "axis": "y", "stripped": "false"}, "mangrove_log"),
                ("brown_mushroom_block", dict(lati, up="true", down="false"),
                 "brown_mushroom_block"),
                ("red_mushroom_block", dict(lati, up="true", down="false"),
                 "red_mushroom_block"),
                ("mushroom_stem", dict(lati, up="false", down="false", material="red"),
                 "mushroom_stem")):
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in pr.items()})
            o, _, _ = v.block.from_universal(b)
            self.assertEqual(o.base_name, java, nome)

    def test_il_bioma_giungla_traduce(self):
        self.assertNotIn("jungle", B.verifica_traduzioni((26, 2, 0)))


if __name__ == "__main__":
    unittest.main()
