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

    def modello(self, dx, dz, porta=0, dy=10, stili=frozenset()):
        return T.Modello(nome=f"{dx}x{dz}",
                         celle=np.zeros((dx, dy, dz), np.int32),
                         tavolozza=[("air", {})], porta=porta, stili=stili)

    def test_entra_solo_quello_che_ci_sta(self):
        ms = [self.modello(7, 7), self.modello(13, 9)]
        rng = np.random.default_rng(0)
        s = T.scegli(ms, 8, 8, 0, rng)
        self.assertIsNotNone(s)
        self.assertEqual(s[0], 0)

    def test_nessuno_entra_si_prende_il_meno_peggio(self):
        """Non c'e' piu' un generatore parametrico a cui tornare: se niente
        entra nel lotto si prende comunque un modello vero, quello con la
        minor eccedenza, invece di lasciare il lotto vuoto."""
        rng = np.random.default_rng(0)
        s = T.scegli([self.modello(13, 9)], 8, 8, 0, rng)
        self.assertIsNotNone(s)
        self.assertEqual(s[0], 0)

    def test_fra_due_che_non_entrano_vince_la_minor_eccedenza(self):
        ms = [self.modello(20, 20), self.modello(10, 10)]
        rng = np.random.default_rng(0)
        s = T.scegli(ms, 8, 8, 0, rng)
        self.assertIsNotNone(s)
        self.assertEqual(s[0], 1)          # 10x10 sfora di meno di 20x20

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

    def test_una_casa_fuori_stile_non_compare(self):
        """A differenza dell'orientamento, lo stile e' un vincolo rigido: una
        casa deserto non deve mai spuntare in un lotto bosco, nemmeno se e'
        l'unica che entrerebbe nel lotto."""
        ms = [self.modello(7, 7, stili=frozenset({"deserto"}))]
        rng = np.random.default_rng(0)
        self.assertIsNone(T.scegli(ms, 8, 8, 0, rng, stile="bosco"))
        self.assertIsNotNone(T.scegli(ms, 8, 8, 0, rng, stile="deserto"))

    def test_una_casa_senza_stili_compare_ovunque(self):
        """Il caso comune: un modello senza `stili` non ha vincoli, cosi' un
        file buttato nella cartella senza toccare altro funziona subito."""
        ms = [self.modello(7, 7)]
        rng = np.random.default_rng(0)
        for stile in ("bosco", "montagna", "deserto", "prato", None):
            self.assertIsNotNone(T.scegli(ms, 8, 8, 0, rng, stile=stile))

    def test_lo_stile_resta_rigido_anche_nel_fallback_di_orientamento(self):
        """Il fallback 'meglio girata male' vale per l'orientamento, non deve
        far passare uno stile sbagliato."""
        ms = [self.modello(12, 7, porta=T.NORD, stili=frozenset({"deserto"}))]
        rng = np.random.default_rng(0)
        self.assertIsNone(T.scegli(ms, 12, 8, T.EST, rng, stile="bosco"))


@unittest.skipUnless(TRADUTTORE, "PyMCTranslate non installato")
class TestAssegnazionePerVillaggio(unittest.TestCase):
    """Un modello non si ripete dentro lo stesso villaggio, e una casa non
    sporge mai dal lotto: se non entra niente, il lotto resta senza casa."""

    def modello(self, dx, dz, porta=0, dy=10):
        return T.Modello(nome=f"{dx}x{dz}",
                         celle=np.zeros((dx, dy, dz), np.int32),
                         tavolozza=[("air", {})], porta=porta)

    def lotto(self, larghezza, profondita, villaggio, gronda=0, palafitta=False):
        from genworld.edifici import Edificio
        return Edificio(x=0, z=0, larghezza=larghezza, profondita=profondita,
                        base=64, gronda=gronda, palafitta=palafitta,
                        villaggio=villaggio, porta=0)

    def test_senza_sporgere_un_lotto_troppo_piccolo_non_ha_casa(self):
        rng = np.random.default_rng(0)
        self.assertIsNone(T.scegli([self.modello(13, 9)], 8, 8, 0, rng,
                                   sporgere=False))
        self.assertEqual(T.candidati_lotto([self.modello(13, 9)], 8, 8, 0), [])

    def test_i_candidati_entrano_tutti_per_intero(self):
        ms = [self.modello(7, 7), self.modello(9, 12), self.modello(20, 20)]
        for k, quarti in T.candidati_lotto(ms, 10, 10, 0):
            ix, iz = ms[k].ingombro(quarti)
            self.assertLessEqual(ix, 10)
            self.assertLessEqual(iz, 10)

    def test_nessun_modello_si_ripete_nello_stesso_villaggio(self):
        ms = [self.modello(7, 7), self.modello(8, 8), self.modello(9, 9),
              self.modello(6, 6)]
        lotti = [self.lotto(12, 12, villaggio=0) for _ in range(4)]
        scelte = T.assegna(ms, lotti, np.random.default_rng(1))
        self.assertEqual(len(scelte), 4)
        self.assertEqual(len({k for k, _ in scelte.values()}), 4)

    def test_finiti_i_modelli_i_lotti_restano_senza_casa(self):
        ms = [self.modello(7, 7), self.modello(8, 8)]
        lotti = [self.lotto(12, 12, villaggio=0) for _ in range(5)]
        scelte = T.assegna(ms, lotti, np.random.default_rng(1))
        self.assertEqual(len(scelte), 2)

    def test_due_villaggi_possono_avere_la_stessa_casa(self):
        ms = [self.modello(7, 7)]
        lotti = [self.lotto(12, 12, villaggio=0), self.lotto(12, 12, villaggio=1)]
        scelte = T.assegna(ms, lotti, np.random.default_rng(1))
        self.assertEqual(len(scelte), 2)

    def test_il_lotto_con_meno_scelta_non_resta_a_bocca_asciutta(self):
        """Il lotto piccolo ha un solo modello possibile, quello grande ne ha
        due: se il grande scegliesse per primo potrebbe prendere l'unico del
        piccolo."""
        ms = [self.modello(6, 6), self.modello(10, 10)]
        for seme in range(20):
            lotti = [self.lotto(12, 12, villaggio=0), self.lotto(7, 7, villaggio=0)]
            scelte = T.assegna(ms, lotti, np.random.default_rng(seme))
            self.assertEqual(len(scelte), 2, f"seme {seme}")
            self.assertEqual(scelte[1][0], 0)

    def test_la_palafitta_non_prende_una_casa(self):
        ms = [self.modello(7, 7)]
        lotti = [self.lotto(12, 12, villaggio=0, palafitta=True)]
        self.assertEqual(T.assegna(ms, lotti, np.random.default_rng(0)), {})

    def test_e_deterministica(self):
        ms = [self.modello(7, 7), self.modello(8, 8), self.modello(9, 9)]
        lotti = [self.lotto(12, 12, villaggio=n % 2) for n in range(6)]
        a = T.assegna(ms, lotti, np.random.default_rng(5))
        b = T.assegna(ms, lotti, np.random.default_rng(5))
        self.assertEqual(a, b)


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


class TestManifestoStili(unittest.TestCase):
    """`stili.json` e' facoltativo: una cartella che non ce l'ha si comporta
    come prima (nessun vincolo)."""

    def test_cartella_senza_manifesto(self):
        self.assertEqual(T.carica_stili("/percorso/che/non/esiste"), {})

    def test_manifesto_letto_e_applicato(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "stili.json"), "w") as f:
                json.dump({"casa": ["deserto", "prato"]}, f)
            self.assertEqual(T.carica_stili(d), {"casa": frozenset({"deserto", "prato"})})

    @unittest.skipUnless(TRADUTTORE, "PyMCTranslate non installato")
    def test_carica_cartella_applica_gli_stili_del_manifesto(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(
                [sys.executable, os.path.join(RADICE, "esempi", "esporta_template.py"),
                 "--cartella", d, "--quante", "1"],
                check=True, capture_output=True)
            nome = os.listdir(d)[0][:-4]
            with open(os.path.join(d, "stili.json"), "w") as f:
                json.dump({nome: ["deserto"]}, f)
            modelli = T.carica_cartella(d)
            self.assertEqual(modelli[0].stili, frozenset({"deserto"}))

    @unittest.skipUnless(TRADUTTORE, "PyMCTranslate non installato")
    def test_carica_cartelle_unisce_piu_cartelle(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            for d, n in ((d1, 2), (d2, 3)):
                subprocess.run(
                    [sys.executable, os.path.join(RADICE, "esempi", "esporta_template.py"),
                     "--cartella", d, "--quante", str(n)],
                    check=True, capture_output=True)
            modelli = T.carica_cartelle([d1, d2])
            self.assertEqual(len(modelli), 5)


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

    def test_vietato_impedisce_di_disegnare_sopra_il_lotto_vicino(self):
        """Il difetto segnalato dall'utente: un modello che sporge oltre il
        proprio lotto non deve mai scrivere sopra quello di un ALTRO
        edificio, o la casa disegnata per seconda "mangia" un pezzo di
        quella disegnata per prima."""
        cat, m = self.catalogo()
        out = np.full((16, 20, 16), 99, np.uint32)
        vietato = np.zeros((16, 16), bool)
        vietato[6, 6] = True     # proprio dove cadrebbe il secondo tronco
        T.costruisci(out, 0, 0, 0, cat, 0, 0, 4, 4, 5, vietato=vietato)
        ids = cat.id_palette(0, 0)
        self.assertEqual(int(out[4, 5, 4]), int(ids[1]))   # la prima cella, permessa
        self.assertEqual(int(out[6, 5, 6]), 99)            # la seconda, bloccata

    def test_vietato_non_tocca_le_altre_colonne(self):
        cat, m = self.catalogo()
        out = np.full((16, 20, 16), 99, np.uint32)
        vietato = np.zeros((16, 16), bool)   # nessuna colonna vietata
        T.costruisci(out, 0, 0, 0, cat, 0, 0, 4, 6, 5, vietato=vietato)
        ids = cat.id_palette(0, 0)
        self.assertEqual(int(out[4, 5, 6]), int(ids[1]))
        self.assertEqual(int(out[6, 5, 6]), int(ids[2]))

    def test_vietato_ferma_anche_la_fondazione(self):
        cat, m = self.catalogo()
        out = np.full((16, 20, 16), 0, np.uint32)
        out[:, :3, :] = 7
        vietato = np.zeros((16, 16), bool)
        vietato[4, 4] = True
        T.fondazione(out, 0, 0, 0, cat, 0, 0, 4, 4, 6, blocco=5, aria=0,
                     vietato=vietato)
        self.assertEqual(int(out[4, 5, 4]), 0)   # non riempita: la colonna e' vietata
        self.assertEqual(int(out[4, 3, 4]), 0)


if __name__ == "__main__":
    unittest.main()
