"""Test degli abitanti e del writer dei file `entities/*.mca`.

Qui non c'e' una libreria da controllare: il file lo scriviamo noi, byte per
byte, perche' amulet non serializza le entita'. Quindi i test sono l'unica
cosa che sta fra questo modulo e un mondo che non si apre.

Si rilegge sempre con un parser indipendente e si pretende esattamente quello
che pretende Minecraft - a partire da `Position`, il campo la cui assenza
faceva scorrere all'infinito "Caricamento del chunk non riuscito".
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from amulet_nbt import load as nbt_load  # noqa: F401
    from genworld.entita import (MESTIERI, Abitante, leggi_chunk,
                                 nbt_abitante, scrivi_regioni)
    NBT = True
except ImportError:
    NBT = False

# le professioni che Minecraft conosce davvero
PROFESSIONI = {
    "armorer", "butcher", "cartographer", "cleric", "farmer", "fisherman",
    "fletcher", "leatherworker", "librarian", "mason", "nitwit", "none",
    "shepherd", "toolsmith", "weaponsmith",
}


@unittest.skipUnless(NBT, "amulet-nbt non installato in questo interprete")
class TestMestieri(unittest.TestCase):

    def test_tutte_le_professioni_esistono(self):
        """Un mestiere inventato non da' errore: il villager nasce
        disoccupato e il banco resta li' inutile."""
        for nostro, (loro, _) in MESTIERI.items():
            self.assertIn(loro, PROFESSIONI, f"{nostro} -> {loro}")

    def test_ogni_mestiere_ha_un_banco(self):
        for nostro, (_, banco) in MESTIERI.items():
            self.assertTrue(banco, f"{nostro} senza banco di lavoro")


@unittest.skipUnless(NBT, "amulet-nbt non installato in questo interprete")
class TestNbt(unittest.TestCase):

    def test_campi_obbligatori(self):
        c = nbt_abitante(Abitante(x=1.5, y=70.0, z=2.5, mestiere="fabbro"))
        for chiave in ("id", "Pos", "Motion", "Rotation", "UUID", "Health",
                       "PersistenceRequired", "VillagerData"):
            self.assertIn(chiave, c)
        self.assertEqual(str(c["id"]), "minecraft:villager")
        self.assertEqual(len(c["UUID"]), 4)

    def test_non_sparisce_quando_il_giocatore_se_ne_va(self):
        c = nbt_abitante(Abitante(x=0.0, y=70.0, z=0.0))
        self.assertEqual(int(c["PersistenceRequired"]), 1)

    def test_la_professione_finisce_nel_nbt(self):
        c = nbt_abitante(Abitante(x=0.0, y=70.0, z=0.0, mestiere="macellaio"))
        self.assertEqual(str(c["VillagerData"]["profession"]),
                         "minecraft:butcher")

    def test_due_abitanti_hanno_uuid_diversi(self):
        a = nbt_abitante(Abitante(x=0.0, y=70.0, z=0.0, seme=1))
        b = nbt_abitante(Abitante(x=0.0, y=70.0, z=0.0, seme=2))
        self.assertNotEqual(list(a["UUID"]), list(b["UUID"]))


@unittest.skipUnless(NBT, "amulet-nbt non installato in questo interprete")
class TestScrittura(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.mondo = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def test_giro_completo(self):
        ab = [Abitante(x=5.5, y=71.0, z=6.5, mestiere="fabbro", seme=1),
              Abitante(x=7.5, y=71.0, z=8.5, mestiere="macellaio", seme=2)]
        st = scrivi_regioni(self.mondo, ab, 4189)
        self.assertEqual(st["abitanti"], 2)
        self.assertEqual(st["chunk"], 1)
        c = leggi_chunk(os.path.join(self.mondo, "entities", "r.0.0.mca"), 0, 0)
        self.assertIsNotNone(c)
        self.assertIn("Position", c)
        self.assertEqual(list(c["Position"]), [0, 0])
        self.assertEqual(len(c["Entities"]), 2)
        self.assertEqual(int(c["DataVersion"]), 4189)

    def test_coordinate_negative(self):
        """Il mondo e' centrato sull'origine: meta' degli abitanti ha
        coordinate negative, e li' sbagliare lo spostamento di bit e' facile."""
        ab = [Abitante(x=-3.5, y=70.0, z=-40.5, mestiere="pastore", seme=1)]
        scrivi_regioni(self.mondo, ab, 4189)
        self.assertTrue(os.path.exists(
            os.path.join(self.mondo, "entities", "r.-1.-1.mca")))
        c = leggi_chunk(os.path.join(self.mondo, "entities", "r.-1.-1.mca"),
                        -1, -3)
        self.assertIsNotNone(c, "chunk non trovato dove dovrebbe stare")
        self.assertEqual(list(c["Position"]), [-1, -3])

    def test_ogni_abitante_sta_nel_suo_chunk(self):
        """Se un abitante finisce nel chunk sbagliato, Minecraft lo sposta o
        lo perde: la posizione nel file deve essere coerente con la Pos."""
        ab = [Abitante(x=float(x) + 0.5, y=70.0, z=float(z) + 0.5, seme=x * 31 + z)
              for x in range(-40, 41, 7) for z in range(-40, 41, 7)]
        scrivi_regioni(self.mondo, ab, 4189)
        import glob
        trovati = 0
        for f in glob.glob(os.path.join(self.mondo, "entities", "*.mca")):
            rx, rz = (int(v) for v in os.path.basename(f).split(".")[1:3])
            for cz in range(rz * 32, rz * 32 + 32):
                for cx in range(rx * 32, rx * 32 + 32):
                    c = leggi_chunk(f, cx, cz)
                    if c is None:
                        continue
                    self.assertEqual(list(c["Position"]), [cx, cz])
                    for e in c["Entities"]:
                        x, _, z = (float(v) for v in e["Pos"])
                        self.assertEqual(int(x) >> 4, cx, f"x {x} nel chunk {cx}")
                        self.assertEqual(int(z) >> 4, cz, f"z {z} nel chunk {cz}")
                        trovati += 1
        self.assertEqual(trovati, len(ab))

    def test_intestazione_e_settori(self):
        """Il formato region non perdona: l'intestazione e' 8 KiB e ogni
        chunk comincia a un multiplo di 4096."""
        scrivi_regioni(self.mondo, [Abitante(x=1.5, y=70.0, z=1.5)], 4189)
        percorso = os.path.join(self.mondo, "entities", "r.0.0.mca")
        dim = os.path.getsize(percorso)
        self.assertEqual(dim % 4096, 0, "file non allineato ai settori")
        self.assertGreaterEqual(dim, 3 * 4096)
        with open(percorso, "rb") as f:
            testa = f.read(4096)
            off = int.from_bytes(testa[0:3], "big")
            quanti = testa[3]
            self.assertGreaterEqual(off, 2, "chunk dentro l'intestazione")
            self.assertGreaterEqual(quanti, 1)
            f.seek(off * 4096)
            lung = int.from_bytes(f.read(4), "big")
            self.assertEqual(f.read(1)[0], 2, "compressione non zlib")
            zlib.decompress(f.read(lung - 1))      # esplode se e' malformato

    def test_senza_abitanti_niente_cartella(self):
        st = scrivi_regioni(self.mondo, [], 4189)
        self.assertEqual(st["abitanti"], 0)
        self.assertFalse(os.path.isdir(os.path.join(self.mondo, "entities")))


if __name__ == "__main__":
    unittest.main()
