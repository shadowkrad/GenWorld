"""Test del motore: firma di cache, cache, pianificazione, scrittura.

La scrittura dei chunk richiede amulet e viene saltata dove non c'e'.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld.motore import (Analisi, Opzioni, analizza, fetta,  # noqa: E402
                             pianifica, _leggi_cache, _scrivi_cache)

from genworld.motore import genera  # noqa: E402

try:
    from genworld.mondo import ScrittoreMondo  # noqa: F401
    AMULET = True
except ImportError:
    AMULET = False


def mappa_finta(percorso: str, lato: int = 256) -> str:
    """Un'isola verde su mare blu, con una cima chiara: basta a far girare
    tutta la pipeline senza dipendere da un'immagine reale."""
    zz, xx = np.mgrid[:lato, :lato]
    d = np.hypot(zz - lato / 2, xx - lato / 2)
    rgb = np.zeros((lato, lato, 3), np.uint8)
    rgb[...] = (40, 80, 160)                       # mare
    rgb[d < lato * 0.40] = (90, 150, 80)           # terra
    rgb[d < lato * 0.15] = (170, 170, 165)         # montagna
    Image.fromarray(rgb).save(percorso)
    return percorso


class TestFirma(unittest.TestCase):

    def base(self, **kw) -> Opzioni:
        return Opzioni(immagine="m.png", uscita="/tmp/mondo", **kw)

    def test_alberi_e_villaggi_non_invalidano_l_analisi(self):
        """Sono fasi successive: cambiarli non deve costare una rianalisi,
        che e' la parte lenta."""
        a = self.base(alberi=1.0, villaggi=1.0, strade=True)
        b = self.base(alberi=0.0, villaggi=3.0, strade=False)
        self.assertEqual(a.firma_analisi(), b.firma_analisi())

    def test_i_parametri_del_terreno_la_invalidano(self):
        a = self.base()
        for campo, valore in (("lato", 512), ("fiumi", 40.0), ("vulcani", 0),
                              ("erosione", 0.3), ("adattivo", True),
                              ("ritaglio", 0.1)):
            b = self.base(**{campo: valore})
            self.assertNotEqual(a.firma_analisi(), b.firma_analisi(), campo)

    def test_la_cache_sta_accanto_al_mondo(self):
        self.assertEqual(self.base().cache, "/tmp/mondo_cache.npz")


class TestCache(unittest.TestCase):

    def test_giro_completo(self):
        with tempfile.TemporaryDirectory() as d:
            op = Opzioni(immagine="m.png", uscita=os.path.join(d, "w"), lato=32)
            a = Analisi(cls=np.zeros((32, 32), np.uint8),
                        h=np.full((32, 32), 70, np.int32),
                        livello=np.full((32, 32), 62.0, np.float32),
                        lava=np.zeros((32, 32), bool),
                        colata=np.zeros((32, 32), bool),
                        note=["una nota"])
            _scrivi_cache(op, a)
            b = _leggi_cache(op)
            self.assertIsNotNone(b)
            self.assertTrue((b.h == a.h).all())
            self.assertEqual(b.note, ["una nota"])

    def test_firma_diversa_ignora_la_cache(self):
        with tempfile.TemporaryDirectory() as d:
            op = Opzioni(immagine="m.png", uscita=os.path.join(d, "w"), lato=32)
            _scrivi_cache(op, Analisi(
                cls=np.zeros((32, 32), np.uint8), h=np.zeros((32, 32), np.int32),
                livello=np.zeros((32, 32), np.float32),
                lava=np.zeros((32, 32), bool), colata=np.zeros((32, 32), bool)))
            from dataclasses import replace
            self.assertIsNone(_leggi_cache(replace(op, lato=64)))

    def test_cache_illeggibile_non_esplode(self):
        """Una cache di una versione precedente va ignorata, non fatta
        esplodere: costa qualche secondo, un KeyError costa la generazione."""
        with tempfile.TemporaryDirectory() as d:
            op = Opzioni(immagine="m.png", uscita=os.path.join(d, "w"))
            with open(op.cache, "wb") as fp:
                fp.write(b"non e' un npz")
            self.assertIsNone(_leggi_cache(op))


class TestPipeline(unittest.TestCase):

    def test_analisi_e_pianificazione(self):
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"))
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=192,
                         vulcani=1, fiumi=80.0)
            visti = []
            a = analizza(op, lambda f, t: visti.append(f))
            self.assertEqual(a.cls.shape, (192, 192))
            self.assertTrue((a.h > 62).any(), "nessuna terra emersa")
            self.assertTrue(visti and visti[-1] == 1.0)
            self.assertEqual(sorted(visti), visti, "avanzamento non monotono")

            # seconda chiamata: deve venire dalla cache, e essere identica
            b = analizza(op)
            self.assertTrue((b.h == a.h).all())

            piano = pianifica(op, a)
            self.assertEqual(piano.h.shape, a.h.shape)
            self.assertIsInstance(piano.indice_alberi, dict)

    def test_fetta_traspone(self):
        """La regola che era sbagliata all'inizio: le mappe sono [z, x], i
        chunk sono [x, y, z]."""
        a = np.arange(32 * 32).reshape(32, 32)
        f = fetta(a, 0, 16)
        self.assertEqual(f.shape, (16, 16))
        self.assertEqual(f[3, 5], a[16 + 5, 3])


@unittest.skipUnless(AMULET, "amulet non installato in questo interprete")
class TestScrittura(unittest.TestCase):

    def test_mondo_piccolo_completo(self):
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 128)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=64,
                         vulcani=0, alberi=0.0, villaggi=0.0, strade=False)
            st = genera(op)
            self.assertEqual(st["chunk"], 16)
            self.assertEqual(st["restano"], 0)
            self.assertEqual(st.get("level_dat"), "valido")
            self.assertTrue(os.path.exists(os.path.join(op.uscita, "level.dat")))

    def test_annullamento(self):
        """La finestra deve poter fermare una generazione lunga."""
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 128)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=128,
                         vulcani=0, alberi=0.0, villaggi=0.0, strade=False)
            visti = []

            def avanza(f, t):
                visti.append(f)

            st = genera(op, avanza, ferma=lambda: len(visti) > 4)
            self.assertTrue(st["interrotto"])
            self.assertLess(st["chunk"], st["totale"])


if __name__ == "__main__":
    unittest.main()
