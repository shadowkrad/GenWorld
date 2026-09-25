"""Le mappe e i chunk usano due ordini di assi diversi. Non confonderli.

Le mappe sono array di immagine: [riga, colonna] = [z, x].
I chunk sono array di blocchi: (16, altezza, 16) = [x, y, z].

Sbagliare la conversione genera il mondo trasposto rispetto alla mappa, e il
difetto e' quasi invisibile: un continente ribaltato lungo la diagonale sembra
comunque un continente.
"""

import os
import sys
import unittest

import numpy as np

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RADICE)

from genworld import mappa as M
from genworld import vegetazione as V
from genworld.motore import fetta


class TestOrientamento(unittest.TestCase):

    def mappa_asimmetrica(self, n=64):
        """Quote che crescono con x e restano costanti con z: asimmetrica."""
        z, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        return (60 + x).astype(np.int32)      # h[z, x] = 60 + x

    def test_fetta_rispetta_x_e_z(self):
        h = self.mappa_asimmetrica()
        f = fetta(h, sx=16, sz=32)            # chunk che parte da x=16, z=32
        self.assertEqual(f.shape, (16, 16))
        # f[lx, lz] deve valere h[sz+lz, sx+lx] = 60 + (16+lx)
        for lx in range(16):
            for lz in range(0, 16, 5):
                self.assertEqual(int(f[lx, lz]), 60 + 16 + lx,
                                 f"assi scambiati in (lx={lx}, lz={lz})")

    def test_fetta_non_e_il_taglio_ingenuo(self):
        """Il taglio che sembra giusto e' proprio quello sbagliato."""
        h = self.mappa_asimmetrica()
        ingenuo = h[16:32, 32:48]
        self.assertFalse(np.array_equal(fetta(h, 16, 32), ingenuo))

    def test_semina_e_fetta_concordano(self):
        """Un albero seminato su terra deve finire su terra anche nel chunk.

        E' l'invariante che il bug violava: la vegetazione leggeva [z, x] e il
        terreno [x, z], quindi gli alberi cadevano in mare.
        """
        n = 64
        cls = np.full((n, n), M.OCEANO, np.uint8)
        h = np.full((n, n), 50, np.int32)
        cls[:, 40:] = M.PRATERIA          # terra solo per x >= 40
        h[:, 40:] = 80
        alberi = V.semina(cls, h, livello_mare=62, scala_densita=4.0, seed=3)
        self.assertGreater(alberi["x"].size, 0)
        for i in range(alberi["x"].size):
            x, z = int(alberi["x"][i]), int(alberi["z"][i])
            self.assertGreaterEqual(x, 40, "albero seminato nel mare")
            sx, sz = (x // 16) * 16, (z // 16) * 16
            f = fetta(h, sx, sz)
            self.assertGreater(int(f[x - sx, z - sz]), 62,
                               "la fetta del chunk non vede terra dove c'e' l'albero")


if __name__ == "__main__":
    unittest.main()
