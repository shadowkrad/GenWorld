"""Verifica del level.dat. Richiede amulet-nbt (vedi README).

Eseguire con l'interprete del venv:
    .venv/bin/python -m unittest discover -s tests -q
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from genworld.livello_dat import (ImpostazioniMondo, costruisci,
                                      data_version_di, scrivi, verifica)
    AMULET = True
except ImportError:
    AMULET = False


@unittest.skipUnless(AMULET, "amulet-nbt non installato in questo interprete")
class TestLevelDat(unittest.TestCase):

    def test_struttura_completa(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "level.dat")
            scrivi(p, ImpostazioniMondo())
            self.assertEqual(verifica(p), [])

    def test_data_version_da_pymctranslate(self):
        self.assertEqual(data_version_di((1, 21, 4)), 4189)
        self.assertEqual(data_version_di((1, 21, 0)), 3953)
        self.assertEqual(ImpostazioniMondo().data_version, 4189)

    def test_data_version_fallback(self):
        """Una versione inesistente ricade sulla piu' alta non superiore."""
        self.assertEqual(data_version_di((1, 21, 3)), data_version_di((1, 21, 2)))

    def test_overworld_e_superflat_vuoto(self):
        """I chunk non scritti devono restare vuoti, non riempirsi di vanilla."""
        d = costruisci(ImpostazioniMondo()).compound["Data"]
        gen = d["WorldGenSettings"]["dimensions"]["minecraft:overworld"]["generator"]
        self.assertEqual(str(gen["type"]), "minecraft:flat")
        self.assertEqual(len(gen["settings"]["layers"]), 1)
        self.assertEqual(str(gen["settings"]["layers"][0]["block"]), "minecraft:air")

    def test_spawn_e_modalita(self):
        d = costruisci(ImpostazioniMondo(spawn=(10, 90, -20), modalita=1)).compound["Data"]
        self.assertEqual(int(d["SpawnX"]), 10)
        self.assertEqual(int(d["SpawnY"]), 90)
        self.assertEqual(int(d["SpawnZ"]), -20)
        self.assertEqual(int(d["GameType"]), 1)

    def test_rileva_level_dat_monco(self):
        """Il verificatore deve bocciare un level.dat come quello di amulet."""
        from amulet_nbt import CompoundTag, IntTag, NamedTag, StringTag
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "level.dat")
            NamedTag(CompoundTag({"Data": CompoundTag({
                "DataVersion": IntTag(4553),
                "LevelName": StringTag("World Created By Amulet"),
            })}), "").save_to(p, compressed=True)
            problemi = verifica(p)
            self.assertTrue(any("WorldGenSettings" in x for x in problemi))
            self.assertTrue(any("GameType" in x for x in problemi))


if __name__ == "__main__":
    unittest.main()
