"""La riserva per rocksdb: quel poco che amulet gli chiede, con sqlite."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import _compat as C  # noqa: E402


class TestRiservaRocksDB(unittest.TestCase):

    def test_put_e_get(self):
        with tempfile.TemporaryDirectory() as d:
            db = C.RocksDB(os.path.join(d, "db"), options=C._Opzioni(), write_options=C._Opzioni())
            db.put(b"chunk/1", b"\x00\x01\x02")
            self.assertEqual(db.get(b"chunk/1"), b"\x00\x01\x02")
            db.close()

    def test_un_valore_si_sovrascrive(self):
        with tempfile.TemporaryDirectory() as d:
            db = C.RocksDB(os.path.join(d, "db"))
            db.put(b"k", b"uno")
            db.put(b"k", b"due")
            self.assertEqual(db.get(b"k"), b"due")
            db.close()

    def test_una_chiave_assente_da_keyerror_come_il_vero_rocksdb(self):
        with tempfile.TemporaryDirectory() as d:
            db = C.RocksDB(os.path.join(d, "db"))
            with self.assertRaises(KeyError):
                db.get(b"non-c-e")
            db.close()

    def test_delete(self):
        with tempfile.TemporaryDirectory() as d:
            db = C.RocksDB(os.path.join(d, "db"))
            db.put(b"k", b"v")
            db.delete(b"k")
            with self.assertRaises(KeyError):
                db.get(b"k")
            db.close()

    def test_close_si_puo_chiamare_due_volte(self):
        """amulet lo chiama sia esplicitamente sia da `__del__`."""
        with tempfile.TemporaryDirectory() as d:
            db = C.RocksDB(os.path.join(d, "db"))
            db.close()
            db.close()

    def test_i_dati_sopravvivono_a_una_riapertura(self):
        with tempfile.TemporaryDirectory() as d:
            percorso = os.path.join(d, "db")
            db = C.RocksDB(percorso)
            db.put(b"k", b"v" * 100_000)
            db.close()
            db2 = C.RocksDB(percorso)
            self.assertEqual(db2.get(b"k"), b"v" * 100_000)
            db2.close()

    def test_le_opzioni_si_assegnano_liberamente(self):
        o = C._Opzioni()
        o.create_if_missing = True
        o.compression_type = C._TipiDiCompressione.ZstdCompression
        o.sync = False
        o.disable_wal = True

    def test_il_modulo_di_riserva_ha_i_nomi_che_amulet_importa(self):
        m = C._modulo_di_riserva()
        for nome in ("RocksDB", "Options", "WriteOptions", "CompressionType"):
            self.assertTrue(hasattr(m, nome), nome)
        self.assertTrue(m.__genworld_riserva__)
        self.assertEqual(m.CompressionType.ZstdCompression, "zstd")

    def test_dove_rocksdb_funziona_non_si_installa_niente(self):
        """Se `rocksdb` e' gia' in sys.modules ed e' quello vero, la riserva
        non lo sostituisce."""
        import types
        vero = types.ModuleType("rocksdb")
        prima = sys.modules.get("rocksdb")
        sys.modules["rocksdb"] = vero
        try:
            self.assertFalse(C.installa_riserva_rocksdb())
            self.assertIs(sys.modules["rocksdb"], vero)
        finally:
            if prima is None:
                del sys.modules["rocksdb"]
            else:
                sys.modules["rocksdb"] = prima


if __name__ == "__main__":
    unittest.main()
