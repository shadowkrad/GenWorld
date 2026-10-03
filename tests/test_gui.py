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
    from genworld.gui import Finestra, VisualizzatoreTemplate
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

    def test_la_versione_di_default_e_1_21_4_ed_e_nella_tenda(self):
        self.assertEqual(self.f.opzioni().versione, (1, 21, 4))
        self.assertGreaterEqual(self.f.versione.count(), 1)
        # ogni voce e' >= 1.21.4: "dalla 1.21.4 in poi", non un elenco che
        # comincia da versioni piu' vecchie mai provate su questo progetto
        for i in range(self.f.versione.count()):
            self.assertGreaterEqual(self.f.versione.itemData(i), (1, 21, 4))

    def test_cambiare_la_tenda_cambia_la_versione_scelta(self):
        # NON `self.f.versione.findData(...)`: vedi il commento nella
        # costruzione della tenda in `gui.py` - PySide6 lo confronta per
        # identita' dell'oggetto Python, quindi una tupla scritta qui
        # troverebbe sempre -1 anche a valori uguali.
        i = next((i for i in range(self.f.versione.count())
                 if self.f.versione.itemData(i) == (1, 21, 9)), -1)
        if i < 0:
            self.skipTest("1.21.9 non nell'elenco di PyMCTranslate installato")
        self.f.versione.setCurrentIndex(i)
        self.assertEqual(self.f.opzioni().versione, (1, 21, 9))

    def test_accampamenti_e_cimiteri_finiscono_nelle_opzioni(self):
        self.f.accampamenti.setValue(50)
        self.f.cimiteri.setValue(0)
        self.f.portali.setValue(0)
        op = self.f.opzioni()
        self.assertAlmostEqual(op.accampamenti, 0.50)
        self.assertAlmostEqual(op.cimiteri, 0.0)
        self.assertAlmostEqual(op.portali, 0.0)

    def test_il_cursore_dei_portali_finisce_nelle_opzioni(self):
        self.f.portali.setValue(150)
        self.assertAlmostEqual(self.f.opzioni().portali, 1.50)

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

    def test_senza_preset_il_preventivo_non_mostra_gli_insediamenti(self):
        """Senza un territorio reale non c'e' niente da confrontare: la riga
        direbbe "ottima" su niente, quindi non si mostra. Restano le altre."""
        visibili = [r[0].text() for r in self.f.pannello._righe
                    if r[0].isVisibleTo(self.f.pannello)]
        self.assertNotIn("Insediamenti", visibili)
        self.assertIn("Scala degli oggetti", visibili)
        self.assertFalse(hasattr(self.f, "preset"))
        self.assertFalse(hasattr(self.f, "monumento"))

    def test_il_territorio_viene_da_lato_e_quota(self):
        self.f.larghezza_km.setValue(50)
        self.f.quota_max.setValue(1000)
        terr, insediamenti = self.f.territorio()
        self.assertEqual(terr.larghezza_m, 50000.0)
        self.assertEqual(terr.quota_max_m, 1000.0)
        self.assertEqual(insediamenti, [])

    def test_senza_case_la_finestra_avvisa_prima_di_generare(self):
        """Un clone pulito non ha template: il programma non deve produrre
        villaggi vuoti senza dirlo."""
        from unittest import mock
        from PySide6.QtWidgets import QMessageBox
        chiese = []
        self.f.percorso.setText(os.path.join(RADICE, "input", "README.md"))
        with mock.patch("genworld.gui.QMessageBox.question",
                        side_effect=lambda *a, **k: chiese.append(a[2]) or QMessageBox.No), \
                mock.patch("genworld.gui.TM.conta_file", return_value=0), \
                mock.patch("os.path.exists", return_value=True):
            self.f._genera()
        self.assertTrue(chiese and "Nessun template" in chiese[0])
        self.assertIsNone(self.f.lavoro, "non doveva partire dopo il no")

    def test_il_menu_template_apre_il_visualizzatore(self):
        """"un menu' che apre la vista per i template da visualizzare" -
        la richiesta dell'utente, parola per parola."""
        self.assertIsNone(self.f._visualizzatore)
        self.f._apri_visualizzatore_template()
        self.assertIsInstance(self.f._visualizzatore, VisualizzatoreTemplate)
        # l'elenco si popola da solo alla costruzione: almeno un gruppo
        # (templates/strutture ha sempre delle case nel repository)
        self.assertGreater(self.f._visualizzatore.albero.topLevelItemCount(), 0)
        self.f._visualizzatore.close()

    def test_riaprire_il_menu_non_crea_una_seconda_finestra(self):
        self.f._apri_visualizzatore_template()
        primo = self.f._visualizzatore
        self.f._apri_visualizzatore_template()
        self.assertIs(self.f._visualizzatore, primo)
        primo.close()

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


RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARTELLA_TEMPLATE = os.path.join(RADICE, "templates")

try:
    import amulet  # noqa: F401
    AMULET = True
except ImportError:
    AMULET = False


@unittest.skipUnless(QT, "PySide6 non installato in questo interprete")
class TestVisualizzatoreTemplate(unittest.TestCase):
    """La finestra aperta dal menu' "Template" - vedi vista_template.py per
    il rendering vero, qui solo il comportamento della finestra."""

    def setUp(self):
        self.v = VisualizzatoreTemplate(CARTELLA_TEMPLATE, (1, 21, 4))

    def tearDown(self):
        self.v.close()
        self.v.deleteLater()

    def test_selezionare_un_intestazione_di_gruppo_non_fa_niente(self):
        gruppo = self.v.albero.topLevelItem(0)
        self.v.albero.setCurrentItem(gruppo)
        self.assertEqual(self.v.info.text(), "Scegli un modello dall'elenco a sinistra.")

    @unittest.skipUnless(AMULET, "amulet non disponibile")
    def test_selezionare_un_file_mostra_dimensioni_e_le_tre_viste(self):
        gruppo = next(self.v.albero.topLevelItem(i)
                     for i in range(self.v.albero.topLevelItemCount())
                     if self.v.albero.topLevelItem(i).childCount() > 0)
        figlio = gruppo.child(0)
        self.v.albero.setCurrentItem(figlio)
        self.assertIn("blocchi", self.v.info.text())
        self.assertIsNotNone(self.v.vista_sopra._pix)
        self.assertIsNotNone(self.v.vista_fronte._pix)
        self.assertIsNotNone(self.v.vista_fianco._pix)

    @unittest.skipUnless(AMULET, "amulet non disponibile")
    def test_la_scheda_3d_si_riempie_e_la_sezione_segue_l_altezza(self):
        gruppo = next(self.v.albero.topLevelItem(i)
                     for i in range(self.v.albero.topLevelItemCount())
                     if self.v.albero.topLevelItem(i).childCount() > 0)
        self.v.albero.setCurrentItem(gruppo.child(0))
        self.assertIsNotNone(self.v.vista_3d._pix)
        self.assertEqual(self.v.sezione.maximum(), self.v._modello.dy)
        prima = self.v.vista_3d._pix.toImage()
        self.v.rotazione.setValue(200)              # girare ridisegna
        self.assertNotEqual(prima, self.v.vista_3d._pix.toImage())

    def test_il_trascinamento_gira_e_tiene_l_inclinazione_nei_limiti(self):
        self.v._trascinato(10, 1000)
        self.assertLessEqual(self.v.inclinazione.value(), 89)
        self.v._trascinato(0, -5000)
        self.assertGreaterEqual(self.v.inclinazione.value(), 5)

    def test_cartella_senza_modelli_non_esplode(self):
        v = VisualizzatoreTemplate("/percorso/che/non/esiste", (1, 21, 4))
        self.assertEqual(v.albero.topLevelItemCount(), 1)   # il messaggio "nessun modello"
        v.close()
        v.deleteLater()


if __name__ == "__main__":
    unittest.main()
