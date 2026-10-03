"""I banchi del mercato non devono stare in aria.

Difetto visto in gioco: la base del banco era la mediana del terreno + 1, e
citta', campi e strade muovono le quote dopo che il mercato e' stato disegnato:
i banchi restavano sospesi. Ora la base e' il punto piu' basso a terreno
definitivo, e i pali scendono fino a toccare terra.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import edifici as E  # noqa: E402
from genworld import motore as MO  # noqa: E402


class FintaTavolozza:
    aria = 0

    def staccionata(self, legno):
        return 10

    def b(self, nome, **prop):
        return {"wool": 11}.get(nome, 12)


def terreno(altezze: np.ndarray, H: int = 120) -> np.ndarray:
    """Un chunk (16, H, 16): pietra (id 1) fino all'altezza di ogni colonna
    esclusa, aria (id 0) sopra. `altezze` e' (16, 16)."""
    out = np.zeros((16, H, 16), np.uint32)
    for x in range(16):
        for z in range(16):
            out[x, :int(altezze[x, z]), z] = 1
    return out


class TestPaliDelBanco(unittest.TestCase):

    def disegna(self, quote, base):
        out = terreno(quote)
        b = E.Banco(x=4, z=4, base=base, verso=E.NORD, merce="frutta")
        E.costruisci_banco(out, 0, 0, 0, b, FintaTavolozza())
        return out

    def test_su_terreno_piano_il_banco_appoggia(self):
        out = self.disegna(np.full((16, 16), 70), base=70)
        for gx, gz in ((4, 4), (6, 4), (4, 6), (6, 6)):
            self.assertNotEqual(int(out[gx, 70, gz]), 0)         # palo o bancone
            self.assertEqual(int(out[gx, 71, gz]), 10)           # il palo sale
            self.assertEqual(int(out[gx, 69, gz]), 1)            # terra, mai toccata

    def test_un_palo_in_aria_scende_fino_a_terra(self):
        """La base e' piu' alta del terreno di tre blocchi: i pali riempiono il
        vuoto, non restano sospesi."""
        out = self.disegna(np.full((16, 16), 70), base=73)
        for gx, gz in ((4, 4), (6, 4), (4, 6), (6, 6)):
            for y in list(range(70, 73)) + [74, 75]:             # a y=73 c'e' il bancone
                self.assertEqual(int(out[gx, y, gz]), 10, f"buco nel palo a y={y}")
            self.assertNotEqual(int(out[gx, 73, gz]), 0)

    def test_su_un_pendio_ogni_palo_tocca_terra(self):
        quote = np.full((16, 16), 70)
        quote[:, 6:] = 72                                        # un gradino a meta' banco
        out = self.disegna(quote, base=70)
        for gx, gz in ((4, 4), (6, 4), (4, 6), (6, 6)):
            h = int(quote[gx, gz])
            sotto = [int(out[gx, y, gz]) for y in range(h, 74)]
            self.assertNotIn(0, sotto[:3], f"palo ({gx},{gz}) sospeso sopra y={h}")

    def test_la_discesa_ha_un_limite(self):
        out = self.disegna(np.full((16, 16), 40), base=100)
        self.assertEqual(int(out[4, 100 - 9, 4]), 0, "il palo e' sceso troppo")

    def test_il_tendone_resta_sopra_i_pali(self):
        out = self.disegna(np.full((16, 16), 70), base=70)
        self.assertEqual(int(out[5, 73, 5]), 11)


class TestBaseDefinitiva(unittest.TestCase):

    def test_la_tabella_dei_mestieri_e_a_livello_di_modulo(self):
        self.assertEqual(MO.MERCE_MESTIERE["carne"], "macellaio")
        self.assertEqual(set(MO.MERCE_MESTIERE), set(E.MERCE))


if __name__ == "__main__":
    unittest.main()
