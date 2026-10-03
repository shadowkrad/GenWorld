"""Test della riga di comando dello stimatore e del suo rapporto a testo."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import cli  # noqa: E402
from genworld import report  # noqa: E402
from genworld.fidelity import stima  # noqa: E402
from genworld.presets import TERRITORI  # noqa: E402
from genworld.scale import Scala, Territorio  # noqa: E402


def esegui(*argv) -> str:
    fuori = io.StringIO()
    with contextlib.redirect_stdout(fuori):
        cli.main(list(argv))
    return fuori.getvalue()


class TestCli(unittest.TestCase):

    def test_stima_di_un_preset(self):
        testo = esegui("stima", "--preset", "garda", "--lato", "2048", "--auto-verticale")
        self.assertIn("Forma", testo)
        self.assertIn("Quote", testo)

    def test_stima_di_un_territorio_a_mano(self):
        testo = esegui("stima", "--territorio", "20000x20000", "--quota-max", "1500",
                       "--lato", "1024")
        self.assertIn("Forma", testo)

    def test_confronta_mostra_piu_dimensioni_di_mondo(self):
        testo = esegui("confronta", "--preset", "garda")
        for lato in ("256", "1024", "16384"):
            self.assertIn(lato, testo)

    def test_risolvi_con_errore_fissato(self):
        testo = esegui("risolvi", "--preset", "italia", "--errore", "500")
        self.assertIn("servono almeno", testo)

    def test_risolvi_con_scala_fissata(self):
        testo = esegui("risolvi", "--preset", "garda", "--metri-per-blocco", "1")
        self.assertIn("il mondo viene", testo)

    def test_risolvi_con_mondo_fissato(self):
        testo = esegui("risolvi", "--preset", "garda", "--lato", "2048")
        self.assertIn("Mondo fissato a 2048", testo)

    def test_con_un_monumento(self):
        senza = esegui("stima", "--preset", "garda", "--lato", "1024")
        con = esegui("stima", "--preset", "garda", "--lato", "1024",
                     "--monumenti", "torre_eiffel")
        self.assertNotEqual(senza, con)

    def test_un_preset_sconosciuto_esce_con_un_messaggio(self):
        with self.assertRaises(SystemExit) as e:
            esegui("stima", "--preset", "atlantide")
        self.assertIn("atlantide", str(e.exception))

    def test_un_monumento_sconosciuto_esce_con_un_messaggio(self):
        with self.assertRaises(SystemExit) as e:
            esegui("stima", "--preset", "garda", "--monumenti", "torre_di_babele")
        self.assertIn("torre_di_babele", str(e.exception))

    def test_senza_territorio_ne_preset_esce(self):
        with self.assertRaises(SystemExit):
            esegui("stima")

    def test_un_territorio_malformato_esce(self):
        with self.assertRaises(SystemExit) as e:
            esegui("stima", "--territorio", "grande")
        self.assertIn("formato", str(e.exception))

    def test_un_comando_sconosciuto_esce(self):
        with self.assertRaises(SystemExit):
            esegui("inventa")


def una_stima():
    terr = Territorio("Prova", 20000.0, 20000.0, 1500.0, 0.0)
    return stima(Scala(lato_mondo_blocchi=1024, territorio=terr, livello_mare_y=62), [], [])


class TestReport(unittest.TestCase):

    def test_la_barra_ha_sempre_la_stessa_larghezza(self):
        for p in (0.0, 0.25, 0.5, 1.0):
            self.assertEqual(len(report._barra(p, celle=10, unicode_ok=False)), 10)
            self.assertEqual(len(report._barra(p, celle=10, unicode_ok=True)), 10)

    def test_la_barra_piena_ha_piu_riempimento_di_quella_vuota(self):
        vuota = report._barra(0.0, unicode_ok=False)
        piena = report._barra(1.0, unicode_ok=False)
        self.assertGreater(piena.count("#") + piena.count("="), vuota.count("#") + vuota.count("="))

    def test_il_pannello_ha_una_riga_per_dimensione(self):
        st = una_stima()
        testo = report.pannello(st)
        for d in st.dimensioni:
            self.assertIn(d.nome, testo)

    def test_il_pannello_senza_avvisi_e_piu_corto(self):
        st = una_stima()
        st.avvisi = ["una cosa da sapere " * 6]
        con = report.pannello(st, mostra_avvisi=True)
        senza = report.pannello(st, mostra_avvisi=False)
        self.assertGreater(len(con), len(senza))

    def test_la_tabella_di_confronto_ha_una_riga_per_stima(self):
        terr, _ = TERRITORI["garda"]
        stime = [stima(Scala(lato_mondo_blocchi=l, territorio=terr, livello_mare_y=62), [], [])
                 for l in (512, 1024, 2048)]
        testo = report.tabella_confronto(stime)
        for l in ("512", "1024", "2048"):
            self.assertIn(l, testo)

    def test_avvolgi_rispetta_la_larghezza(self):
        for riga in report._avvolgi("parola " * 60, indent="  ", larghezza=40):
            self.assertLessEqual(len(riga), 40)


if __name__ == "__main__":
    unittest.main()
