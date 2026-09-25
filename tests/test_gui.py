"""Test della finestra.

Qt sa disegnare senza schermo (`QT_QPA_PLATFORM=offscreen`), quindi la
finestra si puo' costruire e interrogare anche qui. Non si prova l'aspetto -
quello si guarda a occhio su `mondi/gui.png`, prodotto da
`esempi/scatto_gui.py` - ma il comportamento: i cursori finiscono nelle
opzioni giuste e il preventivo cambia quando si muovono.
"""

from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PySide6.QtWidgets import QApplication
    from genworld.gui import Finestra
    QT = True
except ImportError:
    QT = False

APP = None


def setUpModule() -> None:  # noqa: N802 (API unittest)
    global APP
    if QT:
        APP = QApplication.instance() or QApplication([])


@unittest.skipUnless(QT, "PySide6 non installato in questo interprete")
class TestFinestra(unittest.TestCase):

    def setUp(self):
        self.f = Finestra()

    def tearDown(self):
        self.f.close()
        self.f.deleteLater()

    def test_i_cursori_finiscono_nelle_opzioni(self):
        self.f.lato.setValue(512)
        self.f.vulcani.setValue(5)
        self.f.fiumi.setValue(90)
        self.f.alberi.setValue(250)
        self.f.villaggi.setValue(0)
        self.f.erosione.setValue(30)
        self.f.strade.setChecked(False)
        op = self.f.opzioni()
        self.assertEqual(op.lato, 512)
        self.assertEqual(op.vulcani, 5)
        self.assertEqual(op.fiumi, 90.0)
        self.assertAlmostEqual(op.alberi, 2.5)
        self.assertAlmostEqual(op.villaggi, 0.0)
        self.assertAlmostEqual(op.erosione, 0.30)
        self.assertFalse(op.strade)

    def test_il_profilo_vuoto_e_None_non_stringa_vuota(self):
        """`Profilo.carica("")` esploderebbe: il motore si aspetta None."""
        self.f.percorso_profilo.setText("   ")
        self.assertIsNone(self.f.opzioni().profilo)

    def test_il_preventivo_segue_il_lato(self):
        self.f.lato.setValue(256)
        piccolo = self.f.pannello._righe[0][2].text()
        self.f.lato.setValue(2048)
        grande = self.f.pannello._righe[0][2].text()
        self.assertNotEqual(piccolo, grande,
                            "la tolleranza sulla forma non e' cambiata col lato")
        self.assertIn("chunk", self.f.nota_lato.text())

    def test_il_preset_riempie_il_territorio(self):
        i = self.f.preset.findData("italia")
        self.f.preset.setCurrentIndex(i)
        self.assertEqual(self.f.larghezza_km.value(), 1290)
        self.assertEqual(self.f.quota_max.value(), 4808)
        terr, insediamenti = self.f.territorio()
        self.assertEqual(terr.nome, "Italia")
        self.assertTrue(insediamenti)

    def test_il_monumento_aggiunge_e_toglie_una_riga(self):
        """Il pannello riusa le righe: quelle di troppo vanno nascoste, non
        lasciate a mostrare il valore di prima."""
        self.f.monumento.setCurrentIndex(0)
        visibili = lambda: sum(1 for r in self.f.pannello._righe  # noqa: E731
                              if r[0].isVisibleTo(self.f.pannello))
        senza = visibili()
        self.f.monumento.setCurrentIndex(self.f.monumento.findData("torre_eiffel"))
        self.assertEqual(visibili(), senza + 1)
        self.f.monumento.setCurrentIndex(0)
        self.assertEqual(visibili(), senza)

    def test_la_scala_degli_oggetti_non_e_toccabile(self):
        """Il punto fermo del progetto: il cursore muove la scala del
        TERRENO, gli oggetti restano 1 m per blocco. Non deve esistere un
        controllo che la cambi."""
        nomi = [n for n in dir(self.f) if "oggetti" in n.lower()]
        self.assertEqual(nomi, [])
        for lato in (256, 1024, 4096):
            self.f.lato.setValue(lato)
            terr, _ = self.f.territorio()
            from genworld.scale import Scala
            self.assertEqual(Scala(lato, terr).scala_oggetti_m_per_blocco, 1.0)


if __name__ == "__main__":
    unittest.main()
