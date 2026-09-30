"""Test dell'indice per-chunk degli edifici.

Un edificio va registrato in TUTTI i chunk che il suo ingombro tocca: un chunk
che non sa di doverlo disegnare lo lascia tagliato di netto sul confine.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld.insediamenti import indice_per_chunk  # noqa: E402
from genworld import edifici as E  # noqa: E402


class TestIndicePerChunk(unittest.TestCase):

    def edificio(self, x, z, larghezza, profondita, gronda=1) -> E.Edificio:
        return E.Edificio(x=x, z=z, larghezza=larghezza, profondita=profondita,
                          base=70, piani=1, altezza_piano=3, gronda=gronda)

    def test_un_edificio_ben_dentro_un_chunk_sta_in_uno_solo(self):
        indice = indice_per_chunk([self.edificio(x=5, z=5, larghezza=6, profondita=6)])
        self.assertEqual(set(indice), {(0, 0)})
        self.assertEqual(indice[(0, 0)], [0])

    def test_un_edificio_sul_confine_sta_in_tutti_i_chunk_toccati(self):
        indice = indice_per_chunk([self.edificio(x=12, z=12, larghezza=8, profondita=8)])
        self.assertEqual(set(indice), {(0, 0), (1, 0), (0, 1), (1, 1)})

    def test_piu_edifici_nello_stesso_chunk_non_si_pestano(self):
        indice = indice_per_chunk([self.edificio(2, 2, 5, 5), self.edificio(9, 9, 5, 5)])
        self.assertEqual(indice[(0, 0)], [0, 1])

    def test_nessun_edificio_indice_vuoto(self):
        self.assertEqual(indice_per_chunk([]), {})


if __name__ == "__main__":
    unittest.main()
