"""Zone del bottino: castello, nave e faro da template."""
import os
import sys
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld.motore import zone_bottino, zone_mestieri  # noqa: E402


class TestZone(unittest.TestCase):
    def test_classifica_per_indice_di_modello(self):
        confini = {"case": (0, 10), "castelli": (10, 11), "navi": (11, 15),
                   "moli": (15, 16), "fari": (16, 17)}
        isolate = [NS(modello=3, x=100, z=100, larghezza=5, profondita=5),     # una casa: niente
                   NS(modello=10, x=200, z=300, larghezza=40, profondita=30),  # castello
                   NS(modello=12, x=50, z=60, larghezza=20, profondita=8),     # nave
                   NS(modello=15, x=70, z=70, larghezza=3, profondita=20),     # molo: niente
                   NS(modello=16, x=400, z=410, larghezza=9, profondita=9)]    # faro
        zone = zone_bottino(NS(isolate=isolate), confini, 500)
        self.assertEqual(zone, [(-300, -200, -260, -170, "castello"),
                                (-450, -440, -430, -432, "nave"),
                                (-100, -90, -91, -81, "faro")])

    def test_le_botteghe_hanno_il_mestiere(self):
        case = [NS(x=10, z=20, x1=18, z1=29, mestiere="fabbro"),
                NS(x=30, z=40, x1=36, z1=46, mestiere="")]
        self.assertEqual(zone_mestieri(case, 5), [(5, 15, 13, 24, "fabbro")])

    def test_senza_confini_nessuna_zona(self):
        self.assertEqual(zone_bottino(NS(isolate=[NS(modello=0, x=1, z=1, larghezza=1, profondita=1)]),
                                      {"case": (0, 5)}, 0), [])


if __name__ == "__main__":
    unittest.main()
