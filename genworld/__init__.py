"""GenWorld - generatore di mondi Minecraft da piante e mappe."""

__version__ = "0.1.0"

# amulet importa `rocksdb` in cima: se il suo binario e' bloccato (Smart App
# Control di Windows 11) `import amulet` fallirebbe. Vedi `_compat.py`.
from ._compat import installa_riserva_rocksdb as _installa_riserva_rocksdb

RISERVA_ROCKSDB = _installa_riserva_rocksdb()
