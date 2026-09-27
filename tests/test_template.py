"""Test delle case da template `.nbt`.

Il pezzo delicato non e' leggere il file - quello o funziona o esplode - ma la
ROTAZIONE. Una casa va girata verso la strada, e girare una struttura non e'
girare un array: un tronco con `axis=x` diventa `axis=z`, una scala che guarda
a nord guarda a est, uno steccato collegato a ovest si collega a nord. Se si
gira l'array e si lasciano stare le proprieta' si ottiene una casa dalla
pianta giusta fatta tutta di pezzi storti, ed e' un difetto che dall'alto non
si vede.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from genworld import template as T  # noqa: E402

try:
    import PyMCTranslate  # noqa: F401
    TRADUTTORE = True
except ImportError:
    TRADUTTORE = False


class TestRotazioneProprieta(unittest.TestCase):

    def test_la_direzione_gira_in_orario(self):
        p = {"facing": "north"}
        self.assertEqual(T.ruota_proprieta(p, 1)["facing"], "east")
        self.assertEqual(T.ruota_proprieta(p, 2)["facing"], "south")
        self.assertEqual(T.ruota_proprieta(p, 3)["facing"], "west")
        self.assertEqual(T.ruota_proprieta(p, 4)["facing"], "north")

    def test_l_asse_di_un_tronco_si_scambia(self):
        self.assertEqual(T.ruota_proprieta({"axis": "x"}, 1)["axis"], "z")
        self.assertEqual(T.ruota_proprieta({"axis": "z"}, 1)["axis"], "x")
        self.assertEqual(T.ruota_proprieta({"axis": "y"}, 1)["axis"], "y")
        self.assertEqual(T.ruota_proprieta({"axis": "x"}, 2)["axis"], "x")

    def test_su_e_giu_non_girano(self):
        self.assertEqual(T.ruota_proprieta({"facing": "up"}, 1)["facing"], "up")

    def test_i_collegamenti_di_uno_steccato_seguono(self):
        p = {"north": "true", "east": "false", "south": "false", "west": "false"}
        g = T.ruota_proprieta(p, 1)
        self.assertEqual(g["east"], "true")
        self.assertEqual(g["north"], "false")

    def test_la_rotazione_di_un_cartello_e_di_sedicesimi(self):
        self.assertEqual(T.ruota_proprieta({"rotation": "0"}, 1)["rotation"], "4")
        self.assertEqual(T.ruota_proprieta({"rotation": "14"}, 1)["rotation"], "2")

    def test_quattro_quarti_tornano_al_punto_di_partenza(self):
        p = {"facing": "west", "axis": "x", "north": "true", "rotation": "3"}
        self.assertEqual(T.ruota_proprieta(p, 4), p)

    def test_le_proprieta_senza_direzione_non_si_toccano(self):
        p = {"half": "upper", "material": "oak", "shape": "straight"}
        self.assertEqual(T.ruota_proprieta(p, 1), p)


class TestScelta(unittest.TestCase):

    def modello(self, dx, dz, porta=0, dy=10):
        return T.Modello(nome=f"{dx}x{dz}",
                         celle=np.zeros((dx, dy, dz), np.int32),
                         tavolozza=[("air", {})], porta=porta)

    def test_entra_solo_quello_che_ci_sta(self):
        ms = [self.modello(7, 7), self.modello(13, 9)]
        rng = np.random.default_rng(0)
        s = T.scegli(ms, 8, 8, 0, rng)
        self.assertIsNotNone(s)
        self.assertEqual(s[0], 0)
        self.assertIsNone(T.scegli([ms[1]], 8, 8, 0, rng))

    def test_la_porta_guarda_la_strada(self):
        ms = [self.modello(7, 9, porta=T.NORD)]
        rng = np.random.default_rng(0)
        for verso in range(4):
            s = T.scegli(ms, 12, 12, verso, rng)
            self.assertIsNotNone(s)
            k, quarti = s
            self.assertEqual((ms[k].porta + quarti) % 4, verso)

    def test_meglio_girata_male_che_un_buco_nella_fila(self):
        """Se nessuna rotazione fa guardare la porta dalla parte giusta, si
        costruisce lo stesso: un vuoto in una schiera si vede di piu'."""
        ms = [self.modello(12, 7, porta=T.NORD)]
        rng = np.random.default_rng(0)
        # 12x7 entra solo a 0 e 2 quarti; la porta chiede EST
        s = T.scegli(ms, 12, 8, T.EST, rng)
        self.assertIsNotNone(s)

    def test_una_casa_troppo_alta_si_scarta(self):
        ms = [self.modello(7, 7, dy=40)]
        self.assertIsNone(T.scegli(ms, 20, 20, 0, np.random.default_rng(0),
                                   altezza_massima=24))


@unittest.skipUnless(TRADUTTORE, "PyMCTranslate non installato")
class TestAndataERitorno(unittest.TestCase):
    """Si esportano le case parametriche in `.nbt` e si rileggono: e' l'unico
    controllo che tocca davvero il formato, la traduzione e la rotazione
    insieme."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        subprocess.run(
            [sys.executable, os.path.join(RADICE, "esempi", "esporta_template.py"),
             "--cartella", cls.dir.name, "--quante", "3"],
            check=True, capture_output=True)
        cls.modelli = T.carica_cartella(cls.dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_si_rileggono_tutti(self):
        self.assertEqual(len(self.modelli), 3)

    def test_hanno_una_porta_e_guarda_a_nord(self):
        for m in self.modelli:
            self.assertEqual(m.porta, T.NORD, m.nome)

    def test_i_nomi_sono_universali_non_di_gioco(self):
        """Se qui comparisse `oak_planks` invece di `planks` vorrebbe dire
        che la traduzione non e' avvenuta, e il mondo si scriverebbe pieno di
        blocchi che Minecraft non conosce - senza dare errore."""
        nomi = {n for m in self.modelli for n, _ in m.tavolozza}
        self.assertIn("planks", nomi)
        self.assertNotIn("oak_planks", nomi)

    def test_la_casa_non_e_vuota(self):
        for m in self.modelli:
            pieni = int((m.celle >= 0).sum())
            self.assertGreater(pieni, m.dx * m.dz, m.nome)

    def test_ruotando_l_ingombro_si_scambia(self):
        m = self.modelli[0]
        self.assertEqual(m.ingombro(0), (m.dx, m.dz))
        self.assertEqual(m.ingombro(1), (m.dz, m.dx))


class FintoScrittore:
    """Assegna un id progressivo a ogni (nome, proprieta')."""

    def __init__(self):
        self.voci = {}

    def blocco(self, nome, **prop):
        chiave = (nome, tuple(sorted(prop.items())))
        return self.voci.setdefault(chiave, len(self.voci) + 1)


class TestPosa(unittest.TestCase):

    def catalogo(self):
        # un modello asimmetrico: 3 lungo x, 2 lungo z, cosi' la rotazione
        # si vede
        celle = np.full((3, 2, 2), -1, np.int32)
        celle[0, 0, 0] = 1
        celle[2, 0, 0] = 2
        m = T.Modello(nome="prova", celle=celle,
                      tavolozza=[("air", {}), ("stone", {}),
                                 ("log", {"axis": "x"})])
        return T.Catalogo([m], FintoScrittore()), m

    def test_il_meno_uno_non_si_tocca(self):
        cat, m = self.catalogo()
        out = np.full((16, 20, 16), 99, np.uint32)
        T.costruisci(out, 0, 0, 0, cat, 0, 0, 4, 4, 5)
        # solo due celle scritte
        self.assertEqual(int((out != 99).sum()), 2)

    def test_va_dove_gli_si_dice(self):
        cat, m = self.catalogo()
        out = np.full((16, 20, 16), 99, np.uint32)
        T.costruisci(out, 0, 0, 0, cat, 0, 0, 4, 6, 5)
        ids = cat.id_palette(0, 0)
        self.assertEqual(int(out[4, 5, 6]), int(ids[1]))
        self.assertEqual(int(out[6, 5, 6]), int(ids[2]))

    def test_girando_di_un_quarto_il_tronco_cambia_asse(self):
        cat, m = self.catalogo()
        dritto = cat.id_palette(0, 0)[2]
        girato = cat.id_palette(0, 1)[2]
        self.assertNotEqual(int(dritto), int(girato))
        chiavi = {v: k for k, v in cat._scrittore.voci.items()}
        self.assertEqual(dict(chiavi[int(dritto)][1])["axis"], "x")
        self.assertEqual(dict(chiavi[int(girato)][1])["axis"], "z")

    def test_non_sfora_il_chunk(self):
        """Una casa sta a cavallo di piu' chunk: quella che cade fuori dal
        chunk corrente va ignorata, non riportata dentro."""
        cat, m = self.catalogo()
        out = np.full((16, 20, 16), 99, np.uint32)
        T.costruisci(out, 0, 0, 0, cat, 0, 0, 15, 15, 5)
        self.assertEqual(int((out != 99).sum()), 1)   # solo l'angolo

    def test_la_fondazione_riempie_il_vuoto_e_si_ferma_sul_pieno(self):
        cat, m = self.catalogo()
        out = np.full((16, 20, 16), 0, np.uint32)     # 0 = aria
        out[:, :3, :] = 7                             # terreno fino a y=2
        T.fondazione(out, 0, 0, 0, cat, 0, 0, 4, 4, 6, blocco=5, aria=0)
        self.assertEqual(int(out[4, 5, 4]), 5)
        self.assertEqual(int(out[4, 3, 4]), 5)
        self.assertEqual(int(out[4, 2, 4]), 7)        # il terreno non si tocca


if __name__ == "__main__":
    unittest.main()
