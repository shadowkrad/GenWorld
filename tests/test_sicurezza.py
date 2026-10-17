"""Rete di sicurezza: creature mai nei muri, mai sotto terra."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import sicurezza as SI  # noqa: E402
from genworld.entita import Abitante  # noqa: E402
from genworld.fauna import Animale  # noqa: E402
from genworld.torri import Combattente  # noqa: E402

try:
    from genworld.livello_dat import ImpostazioniMondo
    from genworld.mondo import ScrittoreMondo
    AMULET = True
except ImportError:
    AMULET = False

SUOLO = 60          # il primo blocco libero sta a y=60 (pietra fino a 59)


def mondo(d):
    """Un mondo 32 x 32: pietra fino a 59, aria sopra; un muro di mattoni 3 x 4 x 1
    in (x=4..6, z=4, y=60..63); un blocco pieno 5 x 5 x 3 di pietra in (10..14, 60..62, 10..14)
    (una stanza murata); un tetto di pietra a y=64 sopra la zona 20..23 (un corridoio)."""
    w = os.path.join(d, "w")
    with ScrittoreMondo(w, ImpostazioniMondo(versione=(26, 2, 0)), lato_blocchi=32, crea=True) as m:
        for cx in (-1, 0):
            for cz in (-1, 0):
                out = np.full((16, 384, 16), m.id_aria, np.uint32)
                out[:, 0:SUOLO + 64, :] = m.blocco("stone")           # y indice = y + 64
                out[:, SUOLO + 64:, :] = m.id_aria
                if (cx, cz) == (0, 0):
                    for x in range(4, 7):
                        for y in range(SUOLO, SUOLO + 4):
                            out[x, y + 64, 4] = m.blocco("stone_bricks")
                    out[10:15, SUOLO + 64:SUOLO + 67, 10:15] = m.blocco("stone_bricks")
                    out[8, SUOLO + 64, 8] = m.blocco("water")
                m.scrivi_chunk(cx, cz, out)
    return w


@unittest.skipUnless(AMULET, "amulet non disponibile")
class TestSicurezza(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        cls.w = mondo(cls.d.name)

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()

    def un(self, e):
        fuori, st = SI.rendi_sicure(self.w, [e])
        return fuori[0], st

    def test_chi_sta_bene_resta_dov_e(self):
        e = Abitante(x=2.5, y=float(SUOLO), z=2.5, mestiere="fabbro")
        n, st = self.un(e)
        self.assertEqual((n.x, n.y, n.z), (2.5, float(SUOLO), 2.5))
        self.assertEqual(st["spostate"], 0)
        self.assertEqual(st["controllate"], 1)

    def test_dentro_un_muro_va_nella_cella_libera_piu_vicina(self):
        e = Abitante(x=5.5, y=float(SUOLO), z=4.5, mestiere="fabbro")          # nel muro
        n, st = self.un(e)
        self.assertEqual(st["spostate"], 1)
        self.assertNotEqual((int(n.x), int(n.z)), (5, 4))
        self.assertLessEqual(abs(n.x - 5.5) + abs(n.z - 4.5), 3)
        self.assertEqual(n.y, float(SUOLO))                                     # sul pavimento, non piu' su

    def test_sotto_terra_risale_in_superficie(self):
        e = Abitante(x=2.5, y=float(SUOLO - 3), z=2.5, mestiere="fabbro")
        n, st = self.un(e)
        self.assertEqual(st["spostate"], 1)
        self.assertEqual(n.y, float(SUOLO))

    def test_murato_in_un_blocco_pieno_esce_in_una_cella_valida(self):
        e = Combattente(x=12.5, y=float(SUOLO), z=12.5, specie="zombie")        # al centro del blocco pieno
        n, st = self.un(e)
        self.assertEqual(st["spostate"], 1)
        self.assertEqual(type(n), Combattente)
        s = SI.Sicurezza(self.w)
        try:
            self.assertTrue(s.valida(int(n.x), int(n.y), int(n.z)), (n.x, n.y, n.z))
        finally:
            s.chiudi()

    def test_non_in_acqua(self):
        e = Animale(x=8.5, y=float(SUOLO), z=8.5, specie="mucca")               # nell'acqua
        n, st = self.un(e)
        self.assertEqual(st["spostate"], 1)

    def test_chi_vola_o_nuota_non_si_tocca(self):
        for specie in ("pipistrello", "merluzzo", "delfino"):
            e = Animale(x=5.5, y=float(SUOLO - 3), z=5.5, specie=specie)
            n, st = self.un(e)
            self.assertEqual((n.x, n.y, n.z), (5.5, float(SUOLO - 3), 5.5), specie)
            self.assertEqual(st["controllate"], 0)

    def test_fuori_dai_chunk_scritti_non_si_tocca(self):
        e = Abitante(x=300.5, y=float(SUOLO), z=300.5, mestiere="fabbro")
        n, st = self.un(e)
        self.assertEqual((n.x, n.z), (300.5, 300.5))
        self.assertEqual(st["controllate"], 0)

    def test_il_resto_della_lista_e_l_ordine_restano(self):
        a = Abitante(x=2.5, y=float(SUOLO), z=2.5, mestiere="fabbro")
        b = Abitante(x=5.5, y=float(SUOLO), z=4.5, mestiere="pastore")        # nel muro
        c = Animale(x=1.5, y=float(SUOLO), z=1.5, specie="pecora")
        fuori, st = SI.rendi_sicure(self.w, [a, b, c])
        self.assertEqual(len(fuori), 3)
        self.assertEqual(fuori[0], a)
        self.assertEqual(fuori[2], c)
        self.assertEqual(fuori[1].mestiere, "pastore")
        self.assertNotEqual((fuori[1].x, fuori[1].z), (5.5, 4.5))

    def test_oggetti_non_dataclass_o_carrelli_si_ignorano(self):
        class Strano:
            x, y, z = 5.5, 60.0, 4.5

            def nbt_tag(self):
                return None
        fuori, st = SI.rendi_sicure(self.w, [Strano()])
        self.assertEqual(st["controllate"], 0)
        self.assertEqual(fuori[0].x, 5.5)

    def test_lista_vuota(self):
        self.assertEqual(SI.rendi_sicure(self.w, []), ([], {"controllate": 0, "spostate": 0,
                                                              "irrecuperabili": 0}))


if __name__ == "__main__":
    unittest.main()
