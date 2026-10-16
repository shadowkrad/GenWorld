"""Cache della pianificazione (#17)."""
import dataclasses
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genworld.motore import Opzioni, analizza, pianifica_con_cache  # noqa: E402
from test_motore import mappa_finta  # noqa: E402


def opzioni(d, **kw):
    img = mappa_finta(os.path.join(d, "isola.png"), 256)
    base = dict(immagine=img, uscita=os.path.join(d, "w"), lato=256, vulcani=0,
                alberi=0.3, villaggi=1.0, strade=True, miniere=1.0)
    base.update(kw)
    return Opzioni(**base)


class TestFirma(unittest.TestCase):
    def test_stabile_e_sensibile(self):
        with tempfile.TemporaryDirectory() as d:
            op = opzioni(d)
            f = op.firma_piano()
            self.assertEqual(f, op.firma_piano())
            # dove si scrive e come si chiama non contano
            self.assertEqual(f, dataclasses.replace(op, uscita=op.uscita + "2", nome="x").firma_piano())
            # un'opzione di pianificazione si'
            self.assertNotEqual(f, dataclasses.replace(op, alberi=0.9).firma_piano())
            self.assertNotEqual(f, dataclasses.replace(op, seed=99).firma_piano())

    def test_un_template_cambiato_invalida(self):
        with tempfile.TemporaryDirectory() as d:
            t = os.path.join(d, "tpl")
            os.makedirs(t)
            op = opzioni(d, templates=t)
            f = op.firma_piano()
            with open(os.path.join(t, "casa.nbt"), "wb") as fh:
                fh.write(b"x")
            self.assertNotEqual(f, op.firma_piano())


class TestCachePiano(unittest.TestCase):
    def test_secondo_giro_dalla_cache_e_uguale(self):
        with tempfile.TemporaryDirectory() as d:
            op = opzioni(d)
            a = analizza(op)
            p1, c1 = pianifica_con_cache(op, a)
            self.assertFalse(c1)
            self.assertTrue(os.path.exists(op.cache_piano))
            p2, c2 = pianifica_con_cache(op, a)
            self.assertTrue(c2)
            self.assertTrue(np.array_equal(p1.h, p2.h))
            self.assertEqual(len(p1.edifici), len(p2.edifici))
            self.assertEqual(p1.note, p2.note)
            self.assertEqual(len(p1.abitanti), len(p2.abitanti))

    def test_opzioni_diverse_rifanno_il_piano(self):
        with tempfile.TemporaryDirectory() as d:
            op = opzioni(d)
            a = analizza(op)
            pianifica_con_cache(op, a)
            _, c = pianifica_con_cache(dataclasses.replace(op, alberi=0.8), a)
            self.assertFalse(c)

    def test_cache_illeggibile_si_ignora(self):
        with tempfile.TemporaryDirectory() as d:
            op = opzioni(d)
            a = analizza(op)
            with open(op.cache_piano, "wb") as f:
                f.write(b"non e' un pickle")
            p, c = pianifica_con_cache(op, a)
            self.assertFalse(c)
            self.assertIsNotNone(p.h)
            _, c = pianifica_con_cache(op, a)
            self.assertTrue(c)

    def test_senza_cache_non_scrive_niente(self):
        with tempfile.TemporaryDirectory() as d:
            op = opzioni(d)
            a = analizza(op)
            pianifica_con_cache(op, a, usa=False)
            self.assertFalse(os.path.exists(op.cache_piano))


if __name__ == "__main__":
    unittest.main()
