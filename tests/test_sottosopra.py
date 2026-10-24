"""Il Sottosopra: il Nether come specchio dell'overworld (`sottosopra.py`)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import biomi as B
from genworld import sottosopra as SP


class Finto:
    """Un ScrittoreMondo finto: palette e biomi."""

    def __init__(self):
        self._cache_blocchi = {}
        self.id_aria = self.blocco("air")
        self._bio = {}

    def blocco(self, nome, **pr):
        k = (nome, tuple(sorted(pr.items())))
        if k not in self._cache_blocchi:
            self._cache_blocchi[k] = len(self._cache_blocchi)
        return self._cache_blocchi[k]

    def bioma(self, nome):
        return self._bio.setdefault(nome, len(self._bio) + 1)

    def nome(self, i):
        return next(k for k, v in self._cache_blocchi.items() if v == i)


class TestTraduci(unittest.TestCase):
    def test_terreno(self):
        self.assertEqual(SP.traduci("grass_block", {"snowy": "false"}, 0), ("crimson_nylium", {}))
        self.assertEqual(SP.traduci("grass_block", {}, 1), ("warped_nylium", {}))
        self.assertEqual(SP.traduci("stone", {})[0], "netherrack")
        self.assertEqual(SP.traduci("dirt", {})[0], "netherrack")
        self.assertEqual(SP.traduci("sand", {})[0], "soul_sand")
        self.assertEqual(SP.traduci("deepslate", {})[0], "blackstone")

    def test_acqua_diventa_lava(self):
        self.assertEqual(SP.traduci("water", {"level": "0"}), ("lava", {"level": "0"}))
        self.assertEqual(SP.traduci("water", {})[0], "lava")

    def test_legno_e_pietra_cambiano_materiale_non_forma(self):
        n, p = SP.traduci("stairs", {"material": "oak", "facing": "east", "half": "bottom",
                                     "shape": "straight", "waterlogged": "true"}, 0)
        self.assertEqual(n, "stairs")
        self.assertEqual(p["material"], "crimson")
        self.assertEqual(p["facing"], "east")
        self.assertNotIn("waterlogged", p)
        self.assertEqual(SP.traduci("planks", {"material": "birch"}, 1)[1]["material"], "warped")
        self.assertEqual(SP.traduci("slab", {"material": "stone_brick", "type": "top"})[1]["material"],
                         "polished_blackstone_brick")
        self.assertEqual(SP.traduci("wall", {"material": "brick"})[1]["material"], "nether_brick")
        self.assertEqual(SP.traduci("log", {"axis": "y", "material": "oak", "stripped": "false"}, 0)[1]["material"],
                         "crimson")

    def test_foglie_funghi_e_piante(self):
        self.assertEqual(SP.traduci("leaves", {"material": "oak"}, 0)[0], "nether_wart_block")
        self.assertEqual(SP.traduci("leaves", {"material": "oak"}, 1)[0], "warped_wart_block")
        self.assertEqual(SP.traduci("double_plant", {"plant_type": "tall_grass"})[0], "air")
        self.assertEqual(SP.traduci("kelp", {})[0], "air")
        self.assertEqual(SP.traduci("wheat", {"age": "7"}), ("nether_wart", {"age": "3"}))
        self.assertEqual(SP.traduci("some_tulip", {})[0], "air")

    def test_luci_e_minerali(self):
        self.assertEqual(SP.traduci("torch", {"facing": "up"})[0], "soul_torch")
        self.assertEqual(SP.traduci("lantern", {"material": "iron", "hanging": "true"}),
                         ("soul_lantern", {"hanging": "true"}))
        self.assertEqual(SP.traduci("diamond_ore", {})[0], "ancient_debris")
        self.assertEqual(SP.traduci("coal_ore", {})[0], "nether_quartz_ore")

    def test_il_resto_passa_com_e(self):
        self.assertEqual(SP.traduci("chest", {"facing": "north", "material": "wood"}),
                         ("chest", {"facing": "north", "material": "wood"}))
        self.assertEqual(SP.traduci("obsidian", {}), ("obsidian", {}))
        self.assertEqual(SP.traduci("air", {}), ("air", {}))
        self.assertEqual(SP.traduci("blocco_sconosciuto", {"a": "b"}), ("blocco_sconosciuto", {"a": "b"}))

    def test_ogni_destinazione_si_traduce_in_gioco(self):
        import PyMCTranslate
        from amulet.api.block import Block
        from amulet_nbt import StringTag
        tr = PyMCTranslate.new_translation_manager()
        v = tr.get_version("java", (26, 2, 0))
        sorgenti = [(n, {}) for n in SP.SEMPLICI] + [
            ("stairs", {"material": m, "facing": "north", "half": "bottom", "shape": "straight"})
            for m in list(SP.LEGNI) + list(SP.MATERIALI_PIETRA)] + [
            ("planks", {"material": m}) for m in SP.LEGNI] + [
            ("fence", {"material": m}) for m in SP.LEGNI] + [
            ("leaves", {"material": "oak"}), ("grass_block", {}), ("plant", {"plant_type": "poppy"}),
            ("sapling", {"material": "oak"}), ("lantern", {"material": "iron", "hanging": "false"})]
        visti = set()
        for var in (0, 1):
            for nome, pr in sorgenti:
                n2, p2 = SP.traduci(nome, pr, var)
                if n2 == "air":
                    continue
                valide = SP._proprieta_valide(n2, p2)
                self.assertIsNotNone(valide, (nome, n2))
                if (n2, tuple(sorted(valide.items()))) in visti:
                    continue
                visti.add((n2, tuple(sorted(valide.items()))))
                r = v.block.from_universal(Block("universal_minecraft", n2,
                                                 {k: StringTag(x) for k, x in valide.items()}))
                self.assertTrue(r[0].namespaced_name.startswith("minecraft:"), (nome, n2, r))


class TestSpecchio(unittest.TestCase):
    def test_gli_stessi_indici_diventano_blocchi_del_nether(self):
        s = Finto()
        erba = s.blocco("grass_block", snowy="false")
        acqua = s.blocco("water", level="0")
        pietra = s.blocco("stone")
        out = np.array([[[erba, acqua], [pietra, s.id_aria]]], np.uint32)
        sp = SP.Specchio(s)
        r = sp.specchia(out, 0)
        self.assertEqual(r.shape, out.shape)
        self.assertEqual(s.nome(int(r[0, 0, 0]))[0], "crimson_nylium")
        self.assertEqual(s.nome(int(r[0, 0, 1]))[0], "lava")
        self.assertEqual(s.nome(int(r[0, 1, 0]))[0], "netherrack")
        self.assertEqual(int(r[0, 1, 1]), s.id_aria)
        self.assertEqual(s.nome(int(sp.specchia(out, 1)[0, 0, 0]))[0], "warped_nylium")

    def test_i_blocchi_comparsi_dopo_si_traducono(self):
        s = Finto()
        sp = SP.Specchio(s)
        sp.specchia(np.array([[[s.blocco("stone")]]], np.uint32))
        nuovo = s.blocco("sand")
        r = sp.specchia(np.array([[[nuovo]]], np.uint32))
        self.assertEqual(s.nome(int(r[0, 0, 0]))[0], "soul_sand")

    def test_non_tocca_l_originale(self):
        s = Finto()
        out = np.array([[[s.blocco("stone")]]], np.uint32)
        copia = out.copy()
        SP.Specchio(s).specchia(out)
        self.assertTrue((out == copia).all())

    def test_biomi_per_ogni_bioma_dell_overworld(self):
        s = Finto()
        t = SP.Specchio(s).biomi(None, B.BIOMI)
        self.assertEqual(len(t), len(B.BIOMI))
        nomi = {n for n in s._bio}
        self.assertTrue(nomi <= {"nether", "crimson_forest", "warped_forest", "soul_sand_valley",
                                 "basalt_deltas"})

    def test_variante_e_a_macchie(self):
        v = {SP.variante_chunk(cx, cz) for cx in range(-40, 40) for cz in range(-40, 40)}
        self.assertEqual(v, {0, 1})
        self.assertEqual(SP.variante_chunk(3, 3), SP.variante_chunk(4, 5))


class TestDatapack(unittest.TestCase):
    def test_nether_a_scala_uno_alto_come_l_overworld(self):
        with tempfile.TemporaryDirectory() as d:
            base = SP.scrivi_datapack(d)
            self.assertEqual(SP.scrivi_datapack(d), base)           # idempotente
            mc = json.load(open(os.path.join(base, "pack.mcmeta"), encoding="utf-8"))
            self.assertIn("min_format", mc["pack"])
            dim = json.load(open(os.path.join(base, "data", "minecraft", "dimension_type", "the_nether.json"),
                                 encoding="utf-8"))
            self.assertEqual(dim["coordinate_scale"], 1.0)
            self.assertEqual((dim["min_y"], dim["height"], dim["logical_height"]), (-64, 384, 384))
            self.assertFalse(dim["has_ceiling"])
            self.assertEqual(dim["min_y"] % 16, 0)

    def test_amulet_legge_l_altezza_del_nether_dal_datapack(self):
        import amulet
        from genworld.livello_dat import ImpostazioniMondo
        from genworld.mondo import ScrittoreMondo
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w")
            imp = ImpostazioniMondo(nome="T", versione=(26, 2, 0), sottosopra=True)
            with ScrittoreMondo(p, imp, lato_blocchi=64, crea=True) as m:
                b = m._livello.bounds("minecraft:the_nether")
                self.assertEqual(b.min[1], -64)
                self.assertEqual(b.max[1], 320)
                from amulet.api.chunk import Chunk
                c = Chunk(0, 0)
                c.blocks[:, -60:-50, :] = m.blocco("netherrack")
                c.changed = True
                m._livello.put_chunk(c, "minecraft:the_nether")
            lv = amulet.load_level(p)
            try:
                ch = lv.get_chunk(0, 0, "minecraft:the_nether")
                self.assertEqual(str(ch.block_palette[int(ch.blocks[3, -55, 3])].base_name), "netherrack")
            finally:
                lv.close()


class TestCreature(unittest.TestCase):
    def test_abitanti_e_greggi_diventano_piglin_e_hoglin(self):
        from genworld.entita import Abitante
        from genworld.fauna import Animale
        from genworld.torri import Combattente
        lista = [Abitante(x=1.0, y=70.0, z=2.0, mestiere="fabbro", seme=3),
                 Animale(x=2.0, y=70.0, z=3.0, specie="mucca", seme=4),
                 Animale(x=3.0, y=62.0, z=4.0, specie="merluzzo", seme=5),
                 Animale(x=4.0, y=70.0, z=5.0, specie="gallina", seme=6),
                 Combattente(x=5.0, y=70.0, z=6.0, specie="zombie", seme=7)]
        r = SP.specchia_creature(lista)
        self.assertEqual([c.specie for c in r], ["piglin", "hoglin", "piglin", "zombie"])
        for c in r:
            self.assertTrue(str(c.nbt_tag()["id"]).startswith("minecraft:"))
        self.assertEqual(len({(c.x, c.z) for c in r}), 4)


if __name__ == "__main__":
    unittest.main()
