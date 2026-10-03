"""Riserva per `rocksdb`, quando il suo binario non si carica.

amulet-core tiene la cronologia dei chunk in un database RocksDB temporaneo
(`BaseLevel._history_db`) e lo importa in cima, incondizionatamente: se
`rocksdb` non si carica, `import amulet` fallisce e con lui tutto quello che
scrive o legge un mondo.

Su Windows 11 con **Smart App Control** attivo succede esattamente questo: il
file `_rocksdb.cp312-win_amd64.pyd` non e' firmato e il sistema ne blocca il
caricamento ("Un criterio di controllo dell'applicazione ha bloccato il
file"). Non e' un difetto del progetto e non si aggira: non carichiamo il
binario bloccato, ne' tocchiamo le impostazioni di sicurezza.

Quello che amulet chiede al database pero' e' pochissimo - `put` e `get` di
byte, `close` - e si puo' dare con la libreria standard. Questa riserva
(sqlite3, su disco, nella stessa cartella temporanea) entra in funzione SOLO se
`import rocksdb` non riesce; dove il binario funziona non cambia niente.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import types


class _TipiDiCompressione:
    """Gli unici nomi che amulet usa; la compressione qui non c'e'."""
    NoCompression = "none"
    SnappyCompression = "snappy"
    ZlibCompression = "zlib"
    Bz2Compression = "bz2"
    LZ4Compression = "lz4"
    LZ4HCCompression = "lz4hc"
    ZstdCompression = "zstd"


class _Opzioni:
    """Opzioni di RocksDB: si possono assegnare liberamente e non servono a
    niente (sqlite non le conosce)."""


class RocksDB:
    """Quel poco di `rocksdb.RocksDB` che amulet usa: put, get, delete, close,
    su un file sqlite dentro la cartella data."""

    def __init__(self, path, options=None, write_options=None, *args, **kwargs):
        os.makedirs(path, exist_ok=True)
        self._db = sqlite3.connect(os.path.join(path, "riserva.sqlite"),
                                   isolation_level=None, check_same_thread=False)
        # e' un deposito temporaneo che sparisce con la sessione: niente
        # giornale, niente sincronizzazione (come `disable_wal` e `sync=False`)
        self._db.execute("PRAGMA journal_mode=OFF")
        self._db.execute("PRAGMA synchronous=OFF")
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k BLOB PRIMARY KEY, v BLOB)")

    def put(self, key: bytes, value: bytes, *args, **kwargs) -> None:
        self._db.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                         (bytes(key), bytes(value)))

    def get(self, key: bytes, *args, **kwargs) -> bytes:
        riga = self._db.execute("SELECT v FROM kv WHERE k = ?", (bytes(key),)).fetchone()
        if riga is None:
            raise KeyError(key)                 # come il vero RocksDB
        return bytes(riga[0])

    def delete(self, key: bytes, *args, **kwargs) -> None:
        self._db.execute("DELETE FROM kv WHERE k = ?", (bytes(key),))

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None


def _modulo_di_riserva() -> types.ModuleType:
    m = types.ModuleType("rocksdb")
    m.RocksDB = RocksDB
    m.Options = _Opzioni
    m.WriteOptions = _Opzioni
    m.CompressionType = _TipiDiCompressione
    m.__genworld_riserva__ = True
    return m


def installa_riserva_rocksdb() -> bool:
    """Se `rocksdb` non si carica, mette al suo posto la riserva. Ritorna True
    se l'ha messa. Va chiamata PRIMA di importare amulet."""
    if "rocksdb" in sys.modules:
        return bool(getattr(sys.modules["rocksdb"], "__genworld_riserva__", False))
    try:
        import rocksdb  # noqa: F401
        return False
    except Exception:                           # noqa: BLE001 (DLL bloccata: ImportError, OSError...)
        sys.modules.pop("rocksdb", None)
        sys.modules["rocksdb"] = _modulo_di_riserva()
        return True
