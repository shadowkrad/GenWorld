"""Citta' vere: dal file dei luoghi fino alla pianificazione completa (`motore.pianifica`)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import mappa as M
from genworld.motore import Analisi, Opzioni, pianifica


def analisi(lato=600):
    cls = np.full((lato, lato), M.PIANURA, np.uint8)
    cls[:, :120] = M.OCEANO
    cls[:, 120:135] = M.SPIAGGIA
    h = np.full((lato, lato), 70, np.int32)
    h[:, :120] = 40
    return Analisi(cls=cls, h=h, livello=np.full((lato, lato), 62.0, np.float32),
                   lava=np.zeros((lato, lato), bool), colata=np.zeros((lato, lato), np.float32))


class TestCittaReali(unittest.TestCase):
    def piano(self, **kw):
        lato = 600
        reali = (("Alfa", 0.45, 0.30, True), ("Beta", 0.75, 0.70, False))
        op = Opzioni(immagine="", uscita="mondi/_prova_citta_reali", lato=lato, citta_reali=reali,
                     vulcani=0, miniere=0, dungeon=0, torri=0, fortezze=0, castelli_rovina=0, isole=0,
                     siti=0, accampamenti=0, cimiteri=0, portali=0, incarichi=False, fauna=0,
                     alberi=0, **kw)
        return pianifica(op, analisi(lato))

    def test_le_citta_vere_si_piazzano_con_il_loro_nome(self):
        p = self.piano(solo_citta_reali=True)
        nomi = {l.nome for l in p.luoghi if l.tipo == "citta"}
        self.assertEqual(nomi, {"Alfa", "Beta"})
        a = next(l for l in p.luoghi if l.nome == "Alfa")
        # la citta' sta vicino al punto indicato (in coordinate di gioco: meta' mondo di scarto)
        self.assertLess(abs(a.x - (int(0.45 * 600) - 300)), 70)
        self.assertLess(abs(a.z - (int(0.30 * 600) - 300)), 70)

    def test_senza_solo_citta_reali_ci_sono_anche_i_villaggi_automatici(self):
        solo = self.piano(solo_citta_reali=True)
        con = self.piano(solo_citta_reali=False)
        self.assertGreaterEqual(len([l for l in con.luoghi if l.tipo == "citta"]),
                                len([l for l in solo.luoghi if l.tipo == "citta"]))


if __name__ == "__main__":
    unittest.main()
