"""Test del visualizzatore dei template: proiezioni ortogonali, elenco per
cartella/scopo, sostituzione dei blocchi non tradotti nella preview.

Le proiezioni (`_proietta`/`_colora`) sono funzioni pure - niente amulet,
niente PyMCTranslate - quindi si provano con un modello finto invece che con
un `.nbt` vero, cosi' i test di base girano ovunque. Solo `carica_info` sui
file veri (che chiama `template.carica`, quindi il traduttore) va dietro lo
stesso `skipUnless` degli altri test di questo modulo.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import template as TM  # noqa: E402
from genworld import vista_template as VT  # noqa: E402

try:
    import amulet  # noqa: F401
    AMULET = True
except ImportError:
    AMULET = False

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(RADICE, "templates")


class TestProiezione(unittest.TestCase):
    """Un cubo 2x2x2 con un solo blocco pieno in un angolo: la proiezione
    da ciascun lato deve vedere quel blocco solo dal lato giusto."""

    def celle(self):
        c = np.full((2, 2, 2), -1, np.int32)
        c[0, 0, 0] = 0     # angolo (x=0, y=0, z=0)
        return c

    def test_dall_alto_vede_il_blocco_in_basso(self):
        # asse=1 (y), dal_basso=False -> parte da y=1 e scende: trova il
        # blocco a y=0 solo nella colonna (x=0, z=0)
        mappa = VT._proietta(self.celle(), asse=1, dal_basso=False)
        self.assertEqual(mappa.shape, (2, 2))
        self.assertEqual(int(mappa[0, 0]), 0)
        self.assertEqual(int(mappa[1, 0]), -1)
        self.assertEqual(int(mappa[0, 1]), -1)

    def test_scandire_dal_basso_trova_lo_stesso_blocco_solo_se_e_il_primo(self):
        mappa = VT._proietta(self.celle(), asse=1, dal_basso=True)
        self.assertEqual(int(mappa[0, 0]), 0)

    def test_un_blocco_dietro_un_altro_nasconde_il_secondo(self):
        c = self.celle()
        c[0, 1, 0] = 1     # un secondo blocco piu' in alto, stessa colonna
        # scandendo dal basso si vede prima il blocco 0
        mappa = VT._proietta(c, asse=1, dal_basso=True)
        self.assertEqual(int(mappa[0, 0]), 0)
        # scandendo dall'alto (si parte dalla cima e si scende) si vede
        # prima il blocco 1
        mappa2 = VT._proietta(c, asse=1, dal_basso=False)
        self.assertEqual(int(mappa2[0, 0]), 1)

    def test_colonna_vuota_resta_meno_uno(self):
        mappa = VT._proietta(self.celle(), asse=0, dal_basso=True)
        # (x=0,z=0) e' vuoto lungo l'asse x per qualunque y tranne y=0,z=0
        self.assertEqual(int(mappa[1, 1]), -1)


class TestColorazione(unittest.TestCase):

    def test_colore_noto_e_colore_ignoto(self):
        mappa = np.array([[0, 1], [-1, 0]], np.int32)
        tavolozza = [("stone", {}), ("un_blocco_mai_visto", {})]
        rgb, ignoti = VT._colora(mappa, tavolozza)
        self.assertEqual(ignoti, ["un_blocco_mai_visto"])
        from genworld.anteprima import COLORE_IGNOTO, COLORI
        self.assertEqual(tuple(rgb[0, 0]), COLORI["stone"])
        self.assertEqual(tuple(rgb[0, 1]), COLORE_IGNOTO)
        self.assertEqual(tuple(rgb[1, 0]), tuple(VT.COLORE_VUOTO))

    def test_immagine_si_ingrandisce_con_la_scala(self):
        rgb = np.zeros((3, 4, 3), np.uint8)
        im = VT._immagine(rgb, scala=5)
        self.assertEqual(im.size, (4 * 5, 3 * 5))


class TestColoreDiRiserva(unittest.TestCase):
    """I blocchi che `anteprima.COLORI` non elenca non devono piu' finire
    magenta: rendevano illeggibile la casa che si voleva controllare."""

    def test_legno_prende_il_colore_dal_materiale(self):
        self.assertEqual(VT.colore_blocco("trapdoor", {"material": "spruce"}),
                         VT._LEGNO["spruce"])
        self.assertEqual(VT.colore_blocco("spruce_shelf", {}), VT._LEGNO["spruce"])

    def test_la_tintura_viene_dalla_proprieta_color(self):
        self.assertEqual(VT.colore_blocco("stained_terracotta", {"color": "red"}),
                         VT._TINTE["red"])

    def test_i_blocchi_di_prima_ora_hanno_un_colore(self):
        for n in ("fence_gate", "button", "brick_block", "flower_pot",
                  "cobbled_deepslate", "moss_carpet", "iron_chain", "bush"):
            self.assertIsNotNone(VT.colore_blocco(n, {}), n)

    def test_un_nome_sconosciuto_resta_sconosciuto(self):
        self.assertIsNone(VT.colore_blocco("un_blocco_mai_visto", {}))


class TestRender3D(unittest.TestCase):

    def _modello(self):
        celle = np.full((4, 3, 4), -1, np.int32)
        celle[:, 0, :] = 0                       # pavimento
        celle[1:3, 1:3, 1:3] = 1                 # un blocco al centro
        return TM.Modello(nome="prova", celle=celle,
                          tavolozza=[("stone", {}), ("planks", {"material": "oak"})])

    def test_produce_un_immagine_non_vuota(self):
        im = VT.render_3d(self._modello(), yaw=35, pitch=30, lato=200)
        self.assertGreater(im.width, 0)
        colori = {c for _, c in im.getcolors(maxcolors=1 << 20)}
        self.assertGreater(len(colori), 2, "solo lo sfondo: non ha disegnato niente")

    def test_girando_il_modello_l_immagine_cambia(self):
        m = self._modello()
        a = VT.render_3d(m, yaw=0, lato=160).tobytes()
        b = VT.render_3d(m, yaw=90, lato=160).tobytes()
        self.assertNotEqual(a, b)

    def test_la_sezione_toglie_gli_strati_alti(self):
        m = self._modello()
        piena = VT.render_3d(m, taglio=None, lato=160)
        bassa = VT.render_3d(m, taglio=1, lato=160)      # solo il pavimento
        self.assertNotEqual(piena.tobytes(), bassa.tobytes())
        # il modello originale non viene toccato dal taglio
        self.assertEqual(int((m.celle[:, 1:, :] >= 0).sum()), 8)

    def test_un_modello_di_sola_aria_non_esplode(self):
        celle = np.full((2, 2, 2), -1, np.int32)
        m = TM.Modello(nome="vuoto", celle=celle, tavolozza=[("stone", {})])
        self.assertGreater(VT.render_3d(m, lato=100).width, 0)


class TestElenco(unittest.TestCase):
    """`elenca()` legge solo i nomi/percorsi - non serve amulet."""

    def test_raggruppa_per_cartella_di_scopo(self):
        voci = VT.elenca(TEMPLATES)
        cartelle = {v.cartella_scopo for v in voci}
        # almeno le cartelle che questa sessione ha popolato devono esserci
        self.assertIn("Cimiteri (templates/cimiteri)", cartelle)
        self.assertIn("Portali (templates/portali)", cartelle)

    def test_solo_nbt_vengono_elencati(self):
        voci = VT.elenca(TEMPLATES)
        # templates/altro/lantern-post-....mcstructure non e' un .nbt: il
        # loader delle case lo ignora, e cosi' deve fare l'elenco
        nomi = {v.nome for v in voci}
        self.assertNotIn("lantern-post-xb3x5zmc", nomi)

    def test_cartella_assente_non_esplode(self):
        self.assertEqual(VT.elenca("/percorso/che/non/esiste/di/sicuro"), [])


@unittest.skipUnless(AMULET and os.path.isdir(os.path.join(TEMPLATES, "portali")),
                     "amulet o templates/portali non disponibili")
class TestCaricaInfoSuFileVeri(unittest.TestCase):

    def test_il_portale_carica_senza_blocchi_ignoti(self):
        info = VT.carica_info(os.path.join(TEMPLATES, "portali",
                                           "portal-del-neth-54ojw1ng.nbt"))
        self.assertEqual(info.ignoti, [])
        self.assertGreater(info.dx, 0)
        self.assertGreater(len(info.blocchi), 0)
        # le tre proiezioni sono immagini PIL vere, non None
        for im in (info.sopra, info.fronte, info.fianco):
            self.assertGreater(im.width, 0)
            self.assertGreater(im.height, 0)

    def test_il_cimitero_non_ha_piu_i_blocchi_non_tradotti_nella_preview(self):
        """`carica_info` applica la stessa sostituzione di `avamposti.py`
        per cimitero/portale (vedi `_CARTELLE_CON_SOSTITUZIONE`): la preview
        deve mostrare quello che finisce davvero nel mondo, non i nomi
        grezzi del file - altrimenti sembrerebbe rotta anche se la
        generazione vera la aggiusta."""
        percorso = os.path.join(TEMPLATES, "cimiteri", "cementerio-grav-9ggj8p63.nbt")
        m = TM.carica(percorso)
        nomi_grezzi = {n for n, _ in m.tavolozza}
        from genworld.avamposti import _BLOCCHI_RECENTI
        self.assertTrue(nomi_grezzi & set(_BLOCCHI_RECENTI),
                        "il test presuppone un cimitero coi blocchi non tradotti")

        info = VT.carica_info(percorso)
        self.assertEqual(set(info.ignoti) & set(_BLOCCHI_RECENTI), set())

    def test_una_casa_non_passa_per_la_sostituzione_del_cimitero(self):
        """Le case (templates/strutture) NON passano per
        `sostituisci_blocchi_recenti`: e' una scelta di `motore.py` per
        cimitero/portale, non generale - vedi il commento in cima al
        modulo. Qui si verifica solo che `carica_info` non esploda su una
        cartella diversa, non che ogni casa sia perfettamente tradotta
        (molte non lo sono alla versione di default - vedi il doc di
        progetto)."""
        strutture = os.path.join(TEMPLATES, "strutture")
        primo = next((f for f in sorted(os.listdir(strutture))
                     if f.lower().endswith(".nbt")), None)
        if primo is None:
            self.skipTest("nessuna casa in templates/strutture")
        info = VT.carica_info(os.path.join(strutture, primo))
        self.assertGreater(info.dx, 0)


if __name__ == "__main__":
    unittest.main()
