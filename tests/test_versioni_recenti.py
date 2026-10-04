"""Le versioni di gioco recenti (DataVersion 4786 in su, da 26.2) tengono le
dimensioni in `dimensions/<namespace>/<nome>/`: un mondo appena creato non ha
ancora la cartella `region` e amulet non registrava la dimensione, quindi
la generazione si fermava con DimensionDoesNotExist."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genworld  # noqa: E402,F401
from genworld.livello_dat import ImpostazioniMondo, versioni_supportate  # noqa: E402

try:
    from genworld.mondo import ScrittoreMondo
    AMULET = True
except ImportError:
    AMULET = False


@unittest.skipUnless(AMULET, "amulet non disponibile")
class TestVersioniRecenti(unittest.TestCase):

    def apri(self, versione):
        with tempfile.TemporaryDirectory() as d:
            imp = ImpostazioniMondo(nome="t", modalita=1, spawn=(0, 150, 0),
                                    versione=versione)
            percorso = os.path.join(d, "w")
            with ScrittoreMondo(percorso, imp, lato_blocchi=64, crea=True):
                pass
            return os.path.isdir(os.path.join(percorso, "dimensions", "minecraft",
                                              "overworld", "region"))

    def test_ogni_versione_offerta_si_puo_creare(self):
        for v in versioni_supportate():
            if v >= (1, 21, 4):
                with self.subTest(versione=v):
                    self.apri(v)

    def test_le_versioni_recenti_usano_la_cartella_della_dimensione(self):
        self.assertTrue(self.apri((26, 2, 0)))
        self.assertFalse(self.apri((1, 21, 4)))


if __name__ == "__main__":
    unittest.main()
