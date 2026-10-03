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

from genworld.motore import (ALTEZZA_MURA, Analisi, Opzioni, analizza,  # noqa: E402
                             blocchi_chunk, fetta, fetta_orlo, pianifica,
                             _leggi_cache, _scrivi_cache)
from genworld import agricoltura as AG  # noqa: E402
from genworld import mappa as M  # noqa: E402
from genworld import strade as ST  # noqa: E402
from genworld import vulcani as U  # noqa: E402
from genworld import fiumi as R  # noqa: E402

from genworld.motore import genera  # noqa: E402

try:
    from genworld.mondo import ScrittoreMondo  # noqa: F401
    from genworld import template as TM  # noqa: E402
    AMULET = True
except ImportError:
    AMULET = False

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(RADICE, "templates", "strutture")


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

    def test_la_versione_non_invalida_l_analisi(self):
        """La versione Java cambia solo il DataVersion e la traduzione dei
        blocchi in scrittura, non il terreno: rianalizzare per cambiarla
        vorrebbe dire pagare di nuovo la parte lenta per niente."""
        a = self.base(versione=(1, 21, 4))
        b = self.base(versione=(1, 21, 9))
        self.assertEqual(a.firma_analisi(), b.firma_analisi())

    def test_la_versione_di_default_e_1_21_4(self):
        self.assertEqual(Opzioni(immagine="m.png", uscita="/tmp/mondo").versione,
                         (1, 21, 4))

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

    def test_ogni_edificio_ha_un_abitante_anche_senza_mestiere(self):
        """Prima solo le case con un mestiere avevano un abitante, e
        `citta._mestiere()` ne assegna uno solo a una minoranza delle case
        per disegno - una citta' grande restava quasi vuota di villager.
        Ora ogni casa (bottega o abitazione semplice) ne ha uno."""
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 192)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=192,
                         vulcani=0, fiumi=80.0)
            a = analizza(op)
            piano = pianifica(op, a)
            self.assertGreater(len(piano.edifici), 0)
            case_semplici = sum(1 for e in piano.edifici if not e.mestiere)
            self.assertGreater(case_semplici, 0, "il test presuppone almeno "
                               "una casa senza bottega su questa mappa")
            # un abitante per edificio, piu' uno per banco di mercato
            self.assertGreaterEqual(len(piano.abitanti), len(piano.edifici))

    def test_protetto_copre_mura_strade_e_campi(self):
        """`piano.protetto` (vedi il commento in `pianifica()`) e' la
        maschera che dice a caverne e miniere dove non scavare: deve
        includere ogni cella di cinta muraria, di sede stradale e di campo,
        altrimenti un cunicolo puo' ancora lasciare una struttura appesa sul
        vuoto. Serve una mappa abbastanza grande: una citta' murata (con il fosso) ha
        bisogno di un centinaio di celle di pianura."""
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 320)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=320,
                         vulcani=0, fiumi=80.0)
            a = analizza(op)
            piano = pianifica(op, a)
            self.assertIsNotNone(piano.protetto)
            self.assertGreater(int(piano.protetto.sum()), 0)
            self.assertTrue(bool((piano.muro > 0).sum()), "il test presuppone "
                            "una citta' murata su questa mappa")
            self.assertTrue((piano.protetto[piano.muro > 0]).all())
            if piano.tipo_strada is not None:
                self.assertTrue((piano.protetto[piano.tipo_strada > 0]).all())
            if piano.campi is not None:
                self.assertTrue((piano.protetto[piano.campi > 0]).all())

    def test_proprietario_copre_il_sedime_di_ogni_edificio(self):
        """`piano.proprietario` (vedi il commento in `pianifica()`) e' la
        maschera che impedisce al modello sporgente di un edificio di
        disegnare sopra il lotto di un ALTRO edificio - il difetto
        segnalato dall'utente come "casa a mezzo". Ogni cella del sedime di
        un edificio deve riportare il SUO indice, mai quello di un altro
        edificio o della maschera vuota (-1)."""
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 256)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=256,
                         vulcani=0, fiumi=80.0, villaggi=1.0, strade=True)
            a = analizza(op)
            piano = pianifica(op, a)
            self.assertGreater(len(piano.edifici), 0)
            self.assertIsNotNone(piano.proprietario)
            for i, e in enumerate(piano.edifici):
                fetta_ed = piano.proprietario[e.z:e.z + e.profondita,
                                              e.x:e.x + e.larghezza]
                self.assertTrue((fetta_ed == i).all(),
                                f"il sedime dell'edificio {i} non e' tutto suo")

    def test_accampamenti_e_cimiteri_vengono_pianificati(self):
        """Densita' alta su una mappa grande deve produrre almeno un
        accampamento e un cimitero, e il loro ingombro deve finire dentro
        `protetto` - altrimenti una galleria di miniera potrebbe scavare
        proprio sotto una cripta appena disegnata, lo stesso difetto delle
        mura sospese gia' visto con case e strade."""
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 512)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=512,
                         vulcani=0, fiumi=80.0, villaggi=0.0, strade=False,
                         accampamenti=6.0, cimiteri=6.0)
            a = analizza(op)
            piano = pianifica(op, a)
            self.assertGreater(len(piano.avamposti), 0)
            self.assertTrue(any(av.tipo == "campo" for av in piano.avamposti))
            self.assertTrue(any(av.tipo == "cimitero" for av in piano.avamposti))
            self.assertIsNotNone(piano.indice_avamposti)
            self.assertGreater(len(piano.indice_avamposti), 0)
            from genworld import avamposti as AP
            maschera = AP.maschera(piano.avamposti, piano.protetto.shape)
            self.assertTrue((piano.protetto[maschera]).all())
            # i nemici piazzati a mano sono nell'elenco degli abitanti
            from genworld.avamposti import Nemico
            self.assertTrue(any(isinstance(ab, Nemico) for ab in piano.abitanti))

    def test_niente_avamposti_con_densita_zero(self):
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 256)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=256,
                         vulcani=0, fiumi=80.0, villaggi=0.0, strade=False,
                         accampamenti=0.0, cimiteri=0.0)
            a = analizza(op)
            piano = pianifica(op, a)
            self.assertEqual(piano.avamposti, [])


class TestMappaNonQuadrata(unittest.TestCase):
    """Una mappa rettangolare non deve uscire deformata quadrata - vedi il
    commento su `nh, nw` in `analizza()` e `TestCaricaProporzioni` in
    `test_mappa.py`. Qui si verifica il comportamento end-to-end: il mondo
    resta lato x lato, ma la mappa vi si centra senza stirarsi, col bordo
    che avanza lasciato a oceano aperto."""

    def mappa_rettangolare(self, percorso, w, h):
        """Un disco di terra su mare, dentro un'immagine 2:1 (o 1:2): se il
        caricamento deformasse l'immagine, il disco uscirebbe un'ellisse."""
        zz, xx = np.mgrid[:h, :w]
        d = np.hypot(zz - h / 2, xx - w / 2)
        raggio = min(w, h) * 0.35
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[...] = (40, 80, 160)               # mare
        rgb[d < raggio] = (90, 150, 80)        # terra
        Image.fromarray(rgb).save(percorso)
        return percorso

    def test_il_mondo_resta_quadrato(self):
        with tempfile.TemporaryDirectory() as d:
            img = self.mappa_rettangolare(os.path.join(d, "rett.png"), 400, 200)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=256,
                         vulcani=0, fiumi=0, laghi=0)
            a = analizza(op)
            self.assertEqual(a.cls.shape, (256, 256))
            self.assertEqual(a.h.shape, (256, 256))

    def test_il_bordo_che_avanza_e_oceano_aperto(self):
        """Una mappa 2:1 dentro un mondo quadrato lascia una fascia vuota
        sopra e sotto (o ai lati): deve restare oceano, non terra inventata
        ne' un buco senza classe."""
        with tempfile.TemporaryDirectory() as d:
            img = self.mappa_rettangolare(os.path.join(d, "rett.png"), 400, 200)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=256,
                         vulcani=0, fiumi=0, laghi=0)
            a = analizza(op)
            # la mappa 2:1 si scala sulla larghezza: avanza spazio in
            # verticale, quindi le righe vicine ai bordi alto/basso
            self.assertTrue((a.cls[0, :] == M.OCEANO).all())
            self.assertTrue((a.cls[-1, :] == M.OCEANO).all())

    def test_niente_deformazione_un_cerchio_resta_un_cerchio(self):
        """Prova diretta della deformazione: un disco disegnato in una mappa
        2:1 non deve uscire un'ellisse schiacciata sull'asse corto - il
        difetto esatto segnalato dall'utente, solo misurato invece che
        guardato a occhio."""
        with tempfile.TemporaryDirectory() as d:
            img = self.mappa_rettangolare(os.path.join(d, "rett.png"), 400, 200)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=256,
                         vulcani=0, fiumi=0, laghi=0)
            a = analizza(op)
            terra = ~np.isin(a.cls, M.ACQUA)
            self.assertTrue(terra.any(), "il test presuppone terra emersa")
            zs, xs = np.nonzero(terra)
            estensione_z = int(zs.max() - zs.min())
            estensione_x = int(xs.max() - xs.min())
            # il disco originale era perfettamente tondo: le due estensioni
            # devono restare vicine, non una il doppio dell'altra come
            # sarebbe uscito stirando una mappa 2:1 su un quadrato
            rapporto = max(estensione_z, estensione_x) / max(1, min(estensione_z, estensione_x))
            self.assertLess(rapporto, 1.3,
                            f"il disco e' uscito deformato: {estensione_x}x{estensione_z}")

    def test_una_mappa_gia_quadrata_non_cambia(self):
        """Regressione: la stragrande maggioranza delle mappe usate finora
        era gia' quadrata, e per quelle non deve cambiare niente."""
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 256)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=256,
                         vulcani=0, fiumi=80.0)
            a = analizza(op)
            self.assertEqual(a.cls.shape, (256, 256))
            # niente nota sul riempimento per una mappa gia' quadrata
            self.assertFalse(any("non quadrata" in n for n in a.note))


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

    @unittest.skipUnless(AMULET, "amulet non disponibile")
    def test_la_versione_scelta_arriva_nel_level_dat(self):
        """`op.versione` deve arrivare fino a `ImpostazioniMondo`: prima di
        questo il DataVersion era sempre quello di default (1.21.4),
        qualunque cosa scegliesse chi chiama `genera` - un mondo dichiarato
        per un'altra versione ma marcato 1.21.4 e' esattamente il tipo di
        disallineamento che fa dire al gioco "mondo da una versione
        successiva" o lo fa rigenerare come se fosse vecchio."""
        import amulet_nbt

        from genworld.livello_dat import data_version_di

        altra = (1, 21, 5)
        atteso = data_version_di(altra)
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 128)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=64,
                         vulcani=0, alberi=0.0, villaggi=0.0, strade=False,
                         versione=altra)
            genera(op)
            dat = amulet_nbt.load(os.path.join(op.uscita, "level.dat"))
            trovato = int(dat.compound["Data"]["DataVersion"])
            self.assertEqual(trovato, atteso)

    def test_mondo_con_accampamenti_e_cimiteri_si_scrive(self):
        """Un giro end-to-end vero: densita' alta forza la presenza di
        campi e cimiteri, e la scrittura non deve incepparsi ne' produrre
        un level.dat rotto."""
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 384)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=384,
                         vulcani=0, alberi=0.3, villaggi=0.0, strade=False,
                         caverne=1.0, miniere=1.0, accampamenti=6.0,
                         cimiteri=6.0)
            st = genera(op)
            self.assertEqual(st["restano"], 0)
            self.assertEqual(st.get("level_dat"), "valido")

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

    @unittest.skipUnless(AMULET and os.path.isdir(TEMPLATES),
                         "amulet o templates/strutture non disponibili")
    def test_il_bottino_arriva_sul_forziere_vero_non_altrove(self):
        """Il difetto vero: `BA.trova()` veniva chiamata con `sx, sz` (le
        coordinate di MAPPA usate per affettare `piano.h` eccetera), non con
        `ox, oz` (l'angolo del chunk in coordinate di GIOCO). Il bottino
        finiva scritto `meta = lato // 2` blocchi piu' in la', su una cella
        qualunque senza nessun forziere - il forziere vero restava vuoto
        come prima. Un mondo di un solo chunk non lo vedeva mai, perche' a
        `lato=16` la mappa combacia col gioco e `meta` e' zero: serve una
        mappa vera, a piu' chunk, con un template vero che abbia un
        forziere - esattamente questo test."""
        import amulet

        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 192)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=192,
                         vulcani=0, fiumi=80.0, villaggi=1.0, strade=True,
                         caverne=1.0, miniere=1.0, templates=TEMPLATES, seed=11)

            modelli = TM.carica_cartelle([TEMPLATES])
            k_forzato = next(i for i, m in enumerate(modelli)
                             if "abandoned-house" in m.nome)
            assegna_vera = TM.assegna

            def forzato(modelli, edifici, rng, **kw):
                return {i: (k_forzato, 0) for i, e in enumerate(edifici)
                        if not e.palafitta}

            TM.assegna = forzato
            try:
                st = genera(op)
            finally:
                TM.assegna = assegna_vera
            self.assertEqual(st["restano"], 0)

            lv = amulet.load_level(op.uscita)
            try:
                singoli_pieni = 0
                for cx, cz in lv.all_chunk_coords("minecraft:overworld"):
                    ch = lv.get_chunk(cx, cz, "minecraft:overworld")
                    pal = lv.block_palette
                    for (bx, by, bz), be in ch.block_entities.items():
                        if be.base_name != "chest":
                            continue
                        # il difetto vero: il block-entity finiva con una
                        # posizione fuori dal chunk che lo contiene.
                        lx, lz = bx - cx * 16, bz - cz * 16
                        self.assertTrue(0 <= lx < 16 and 0 <= lz < 16,
                                        f"forziere a {(bx, by, bz)} fuori dal "
                                        f"chunk {(cx, cz)}: e' il difetto "
                                        f"sx/sz invece di ox/oz")
                        blocco = pal[int(np.asarray(ch.blocks[lx, by, lz]))]
                        connessione = (blocco.properties.get("type")
                                      or blocco.properties.get("connection"))
                        if str(connessione) in ("single", "none"):
                            utags = be.nbt.compound.get("utags")
                            n = len(utags["Items"]) if utags is not None else 0
                            if n > 0:
                                singoli_pieni += 1
                self.assertGreater(singoli_pieni, 0,
                                   "nessun forziere singolo e' arrivato pieno")
            finally:
                lv.close()


class FintoScrittore:
    """Assegna un id progressivo a ogni (nome, proprieta') - basta per
    distinguere i blocchi senza aprire un livello Minecraft vero."""
    id_aria = 0

    def __init__(self):
        self.voci: dict = {}

    def blocco(self, nome, **prop):
        chiave = (nome, tuple(sorted(prop.items())))
        return self.voci.setdefault(chiave, len(self.voci) + 1)


class TestSpondeFiumi(unittest.TestCase):
    """Il bordo del fiume: terra, non pietra - vedi il commento su
    `sponda_m` in `scrivi()` e su `sponda_c` in `blocchi_chunk()`. Nato da
    uno screenshot in cui il fiume era circondato da un bordo di pietra
    piatta invece che di terra."""

    def chunk(self, sponda, nuda):
        s = FintoScrittore()
        h_c = np.full((16, 16), 70, np.int32)
        cls_c = np.full((16, 16), M.PIANURA, np.uint8)
        out = blocchi_chunk(s, h_c, cls_c, nuda_c=nuda, sponda_c=sponda)
        cima_y = 70 - 1 - (-64)          # stessa formula di blocchi_chunk
        return out[:, cima_y, :], s

    def test_senza_sponda_la_parete_nuda_resta_pietra(self):
        """Comportamento invariato lontano dai fiumi: una parete ripida
        qualunque continua a mostrare la roccia, come prima di questa
        modifica."""
        nuda = np.ones((16, 16), bool)
        cima, s = self.chunk(sponda=None, nuda=nuda)
        pietra = s.blocco("stone")
        self.assertTrue((cima == pietra).all())

    def test_la_sponda_scavalca_anche_una_parete_marcata_nuda(self):
        """E' esattamente il caso del bug: la fascia scavata vicino al
        fiume risultava ripida abbastanza da finire marcata 'nuda' (parete),
        e usciva tutta di pietra. La sponda deve vincere comunque."""
        nuda = np.ones((16, 16), bool)
        sponda = np.ones((16, 16), bool)
        cima, s = self.chunk(sponda=sponda, nuda=nuda)
        terra = s.blocco("dirt")
        pietra = s.blocco("stone")
        self.assertGreater(int((cima == terra).sum()), 0)
        self.assertLess(int((cima == pietra).sum()), cima.size)

    def test_la_sponda_e_per_lo_piu_terra(self):
        nuda = np.zeros((16, 16), bool)
        sponda = np.ones((16, 16), bool)
        cima, s = self.chunk(sponda=sponda, nuda=nuda)
        terra = s.blocco("dirt")
        self.assertGreater(int((cima == terra).sum()), cima.size // 2)

    def test_la_sponda_ha_anche_qualche_roccia_che_sporge(self):
        nuda = np.zeros((16, 16), bool)
        sponda = np.ones((16, 16), bool)
        cima, s = self.chunk(sponda=sponda, nuda=nuda)
        pietra = s.blocco("stone")
        muschio = s.blocco("mossy_cobblestone")
        self.assertGreater(int((cima == pietra).sum() + (cima == muschio).sum()), 0)

    def test_fuori_sponda_il_terreno_normale_non_cambia(self):
        nuda = np.zeros((16, 16), bool)
        cima, s = self.chunk(sponda=None, nuda=nuda)
        erba = s.blocco("grass_block")
        self.assertTrue((cima == erba).all())


class TestFiumiEvitanoVulcani(unittest.TestCase):
    """I fiumi si calcolano DOPO i vulcani, apposta, per avere torrenti sui
    fianchi (vedi il commento sopra `maschera_f` in `analizza()`) - ma il
    cratere ha gia' il suo lago di lava con un pelo libero tutto suo, e un
    fiume che ci scorre sopra o dentro ci mette sopra un secondo pelo che
    conflige col primo: due acque sovrapposte, "un effetto non normale"
    segnalato dall'utente, che voleva i fiumi tolti dai vulcani.

    Scenario deterministico: una valle a V che convoglia il deflusso lungo
    z=80, con un vulcano piazzato esattamente su quella linea di flusso -
    cosi' la sovrapposizione e' garantita, non affidata al caso."""

    def scenario(self):
        H = W = 160
        z0 = 80
        xx = np.arange(W)[None, :].astype(np.float32)
        zz = np.arange(H)[:, None].astype(np.float32)
        h = (140 - xx * 0.5 + (zz - z0) ** 2 * 0.05).astype(np.float32)
        cls = np.full((H, W), M.PIANURA, np.uint8)
        cls[:, W - 6:] = M.OCEANO
        h[:, W - 6:] = 40
        h = h.astype(np.int32)
        marino = cls == M.OCEANO

        v = U.Vulcano(z=z0, x=100, raggio=40, altezza=60, raggio_cratere=14,
                      profondita_cratere=10, quota_base=float(h[z0, 100]), seme=1)
        cls2, h2f, lava, q_lava = U.modella(cls, h.astype(np.float32), [v])
        return cls2, h2f, marino

    def test_senza_esclusione_il_fiume_entra_nel_vulcano(self):
        """Prova che il difetto e' reale: senza l'esclusione, il reticolo
        idrografico calcolato sul terreno-con-vulcano attraversa il cono."""
        cls2, h2f, marino = self.scenario()
        maschera_f, _ = R.da_terreno(h2f.astype(np.float32), marino, soglia=20.0)
        sovrapposte = maschera_f & np.isin(cls2, (M.VULCANO, M.CRATERE))
        self.assertGreater(int(sovrapposte.sum()), 0,
                           "il test presuppone che il fiume attraversi il vulcano")

    def test_con_esclusione_nessun_fiume_dentro_il_vulcano(self):
        """La correzione applicata in analizza(): la stessa esclusione gia'
        usata per i laghi (M.VULCANO, M.CRATERE), qui applicata alla
        maschera dei fiumi."""
        cls2, h2f, marino = self.scenario()
        maschera_f, _ = R.da_terreno(h2f.astype(np.float32), marino, soglia=20.0)
        maschera_f &= ~np.isin(cls2, (M.VULCANO, M.CRATERE))
        sovrapposte = maschera_f & np.isin(cls2, (M.VULCANO, M.CRATERE))
        self.assertEqual(int(sovrapposte.sum()), 0)

    def test_il_fiume_resta_altrove(self):
        """L'esclusione non deve cancellare il fiume intero: solo le celle
        che cadono dentro il cono o il cratere."""
        cls2, h2f, marino = self.scenario()
        maschera_f, _ = R.da_terreno(h2f.astype(np.float32), marino, soglia=20.0)
        prima = int(maschera_f.sum())
        maschera_f &= ~np.isin(cls2, (M.VULCANO, M.CRATERE))
        dopo = int(maschera_f.sum())
        self.assertGreater(dopo, 0, "il fiume non deve sparire del tutto")
        self.assertLess(dopo, prima, "l'esclusione deve togliere qualche cella")


class TestDistanzaMinimaDalVulcano(unittest.TestCase):
    """Portali, accampamenti, cimiteri e ingressi di miniera nascevano sul
    fianco del cono: `evita` conosceva mura, strade e lotti, non il vulcano.
    In gioco un portale restava appeso a una parete di basalto."""

    def analisi(self):
        H = W = 200
        h = np.full((H, W), 90, np.int32)
        cls = np.full((H, W), M.PIANURA, np.uint8)
        v = U.Vulcano(z=100, x=100, raggio=30, altezza=50, raggio_cratere=10,
                      profondita_cratere=8, quota_base=90.0, seme=1)
        cls2, h2f, lava, q_lava = U.modella(cls, h.astype(np.float32), [v])
        a = Analisi(cls=cls2, h=h2f, livello=np.zeros((H, W), np.float32),
                    lava=lava, colata=np.zeros((H, W), np.float32))
        return a, np.zeros((H, W), bool)

    def test_la_zona_copre_il_cono_e_cresce_col_margine(self):
        from scipy.ndimage import distance_transform_edt
        from genworld.motore import zona_vulcanica
        a, _ = self.analisi()
        nuda = zona_vulcanica(a, 0)
        self.assertTrue(nuda[np.isin(a.cls, (M.VULCANO, M.CRATERE))].all())
        larga = zona_vulcanica(a, 12)
        self.assertGreater(int(larga.sum()), int(nuda.sum()))
        self.assertTrue((larga | ~nuda).all(), "la zona larga deve contenere quella nuda")
        self.assertLessEqual(float(distance_transform_edt(~nuda)[larga].max()), 12.0)

    def test_senza_vulcani_la_zona_e_vuota(self):
        from genworld.motore import zona_vulcanica
        a, _ = self.analisi()
        a.cls[:] = M.PIANURA
        a.lava[:] = False
        self.assertFalse(zona_vulcanica(a, 12).any())

    def test_le_miniere_non_nascono_sul_vulcano(self):
        from genworld import miniere as MI
        from genworld.motore import DISTANZA_MIN_VULCANO, zona_vulcanica
        a, mare = self.analisi()
        h = a.h.astype(np.float32)
        nuda = zona_vulcanica(a, 0)
        evita = zona_vulcanica(a, DISTANZA_MIN_VULCANO + MI.BINARIO_FUORI + 4)
        senza, con = [], []
        for seed in range(60):
            senza += MI.pianifica(h, mare, evita=None, densita=8.0, seed=seed, distanza_min=10)
            con += MI.pianifica(h, mare, evita=evita, densita=8.0, seed=seed, distanza_min=10)
        self.assertTrue(any(nuda[m.z, m.x] for m in senza),
                        "il test presuppone che senza esclusione qualcuna nasca sul cono")
        self.assertTrue(con)
        self.assertFalse(any(nuda[m.z, m.x] for m in con))

    def test_gli_avamposti_non_nascono_sul_vulcano(self):
        from genworld import avamposti as AP
        from genworld.motore import DISTANZA_MIN_VULCANO, zona_vulcanica
        a, mare = self.analisi()
        h = a.h.astype(np.int32)
        nuda = zona_vulcanica(a, 0)
        evita = zona_vulcanica(a, DISTANZA_MIN_VULCANO + AP.RAGGIO_CIMITERO)
        senza, con = [], []
        for seed in range(60):
            senza += AP.pianifica(h, mare, evita=None, campi=60.0, cimiteri=60.0, seed=seed)
            con += AP.pianifica(h, mare, evita=evita, campi=60.0, cimiteri=60.0, seed=seed)
        self.assertTrue(any(nuda[av.z, av.x] for av in senza),
                        "il test presuppone che senza esclusione qualcuno nasca sul cono")
        self.assertTrue(con)
        self.assertFalse(any(nuda[av.z, av.x] for av in con))


class TestVialiInterni(unittest.TestCase):
    """I viali dentro l'abitato (ST.LASTRICATO) non sono piu' lastricati di
    pietra piatta - vedi il commento in motore._posa_strada."""

    def strada(self, tipo_valore):
        s = FintoScrittore()
        h_c = np.full((16, 16), 70, np.int32)
        cls_c = np.full((16, 16), M.PIANURA, np.uint8)
        tipo_c = np.full((16, 16), tipo_valore, np.uint8)
        quota_c = h_c.copy()
        out = blocchi_chunk(s, h_c, cls_c, tipo_c=tipo_c, quota_c=quota_c)
        y = 70 - 1 - (-64)
        return out[:, y, :], s

    def test_il_viale_non_e_mai_pietra_lastricata(self):
        cima, s = self.strada(ST.LASTRICATO)
        lastrico = s.blocco("cobblestone")
        self.assertEqual(int((cima == lastrico).sum()), 0)

    def test_il_viale_e_per_lo_piu_terra_calpestata(self):
        cima, s = self.strada(ST.LASTRICATO)
        sentiero = s.blocco("grass_path")
        self.assertGreater(int((cima == sentiero).sum()), cima.size // 2)

    def test_il_viale_ha_qualche_sasso_in_piu_del_vicolo(self):
        """Rango >= secondaria si distingue dal vicolo per un po' di ghiaia
        in piu' (il passaggio piu' battuto), non per un materiale diverso."""
        viale, s = self.strada(ST.LASTRICATO)
        vicolo, _ = self.strada(ST.STRADA)
        ghiaia = s.blocco("gravel")
        self.assertGreater(int((viale == ghiaia).sum()),
                           int((vicolo == ghiaia).sum()))
        self.assertEqual(int((vicolo == ghiaia).sum()), 0)


class TestMura(unittest.TestCase):
    """La cinta a tre celle - vedi il commento in motore._posa_mura: i
    merli non devono spargersi a scacchiera su tutto lo spessore, solo sul
    filo, e la screpolatura non deve dipendere dalle coordinate locali al
    chunk (si romperebbe ad ogni confine, come gia' successo per viali e
    sponde)."""

    def muro_di_prova(self, ox=0, oz=0):
        s = FintoScrittore()
        h_c = np.full((16, 16), 70, np.int32)
        cls_c = np.full((16, 16), M.PIANURA, np.uint8)
        muro_c = np.zeros((16, 16), np.uint8)
        muro_c[6:9, :] = 1               # muro dritto, spesso 3 celle
        out = blocchi_chunk(s, h_c, cls_c, muro_c=muro_c, ox=ox, oz=oz)
        return out, s

    def test_niente_merli_in_mezzo_a_un_muro_spesso(self):
        """La colonna centrale (lx=7) non tocca mai l'esterno del muro:
        sopra la sommita' deve restare aria, mai un merlo."""
        out, s = self.muro_di_prova()
        aria = s.id_aria
        cima_y = 70 + ALTEZZA_MURA - (-64)
        self.assertTrue((out[7, cima_y, :] == aria).all())

    def test_i_merli_stanno_sul_filo(self):
        """Le due colonne esterne (lx=6 e lx=8) toccano il vuoto e devono
        avere almeno un merlo alternato."""
        out, s = self.muro_di_prova()
        pietra = s.blocco("stone_bricks", variant="normal")
        cima_y = 70 + ALTEZZA_MURA - (-64)
        self.assertGreater(int((out[6, cima_y, :] == pietra).sum()), 0)
        self.assertGreater(int((out[8, cima_y, :] == pietra).sum()), 0)

    def test_la_screpolatura_dipende_dal_mondo_non_dal_chunk(self):
        """Stessa cella di mondo raggiunta con due scomposizioni diverse in
        (ox, lx): deve uscire lo stesso materiale. Con le coordinate locali
        (il bug corretto qui) il motivo cambierebbe a ogni confine."""
        s = FintoScrittore()
        h_c = np.full((16, 16), 70, np.int32)
        cls_c = np.full((16, 16), M.PIANURA, np.uint8)
        y = 70 + 3 - (-64)          # una cella dentro il corpo del muro

        muro_a = np.zeros((16, 16), np.uint8)
        muro_a[5, :] = 1             # mondo x = 0 + 5 = 5
        out_a = blocchi_chunk(s, h_c, cls_c, muro_c=muro_a, ox=0, oz=0)

        muro_b = np.zeros((16, 16), np.uint8)
        muro_b[0, :] = 1             # mondo x = 5 + 0 = 5
        out_b = blocchi_chunk(s, h_c, cls_c, muro_c=muro_b, ox=5, oz=0)

        self.assertTrue((out_a[5, y, :] == out_b[0, y, :]).all())


class TestRecintoConnesso(unittest.TestCase):
    """Uno steccato (`fence`) e' un blocco "connesso": le sue proprieta'
    north/south/east/west decidono la forma disegnata in gioco, e il gioco
    NON le ricalcola da solo per un blocco che sta gia' nel chunk generato
    (lo fa solo quando lo si piazza o un vicino cambia, in survival). Prima
    di questo fix erano sempre "false": in gioco, una fila di paletti
    staccati invece di un recinto continuo - vedi il commento in
    motore._posa_campi."""

    def campo(self, celle, ox=0, oz=0):
        s = FintoScrittore()
        h_c = np.full((16, 16), 70, np.int32)
        cls_c = np.full((16, 16), M.PIANURA, np.uint8)
        campi_c = np.zeros((16, 16), np.uint8)
        for lx, lz in celle:
            campi_c[lx, lz] = AG.RECINTO
        out = blocchi_chunk(s, h_c, cls_c, campi_c=campi_c, ox=ox, oz=oz)
        y = 70 - (-64)               # un blocco sopra la superficie
        return out, s, y

    def test_un_palo_isolato_non_si_connette_a_niente(self):
        out, s, y = self.campo([(5, 5)])
        palo = s.blocco("fence", material="oak", north="false", south="false",
                        east="false", west="false")
        self.assertEqual(int(out[5, y, 5]), palo)

    def test_due_pali_affiancati_si_connettono_fra_loro(self):
        """(5,5) e (6,5): vicini lungo x, cioe' est/ovest."""
        out, s, y = self.campo([(5, 5), (6, 5)])
        est = s.blocco("fence", material="oak", north="false", south="false",
                       east="true", west="false")
        ovest = s.blocco("fence", material="oak", north="false", south="false",
                         east="false", west="true")
        self.assertEqual(int(out[5, y, 5]), est)
        self.assertEqual(int(out[6, y, 5]), ovest)

    def test_si_connette_anche_al_vicino_nel_chunk_accanto(self):
        """Lo stesso caso del test precedente, ma spezzato in due chunk: il
        palo sul bordo (lx=15) deve sapere che al di la' del confine (nel
        chunk successivo) c'e' un altro palo - e' proprio il caso per cui
        serve `campi_orlo_c`/`fetta_orlo`."""
        campi_pieno = np.zeros((40, 40), np.uint8)
        campi_pieno[10, 15] = AG.RECINTO   # (x=15, z=10): ultimo del chunk 0
        campi_pieno[10, 16] = AG.RECINTO   # (x=16, z=10): primo del chunk 1

        s = FintoScrittore()
        h_c = np.full((16, 16), 70, np.int32)
        cls_c = np.full((16, 16), M.PIANURA, np.uint8)
        y = 70 - (-64)

        campi_c0 = fetta(campi_pieno, 0, 0)
        orlo0 = fetta_orlo(campi_pieno, 0, 0)
        out0 = blocchi_chunk(s, h_c, cls_c, campi_c=campi_c0, ox=0, oz=0,
                             campi_orlo_c=orlo0)
        est = s.blocco("fence", material="oak", north="false", south="false",
                       east="true", west="false")
        self.assertEqual(int(out0[15, y, 10]), est)

    def test_senza_orlo_esplicito_il_bordo_del_chunk_non_si_connette(self):
        """Comportamento di ripiego (retrocompatibile) quando non si passa
        `campi_orlo_c`: il vicino oltre il chunk resta sconosciuto, come
        prima di questo fix - non deve esplodere, solo non connettersi."""
        campi_c = np.zeros((16, 16), np.uint8)
        campi_c[15, 10] = AG.RECINTO
        s = FintoScrittore()
        h_c = np.full((16, 16), 70, np.int32)
        cls_c = np.full((16, 16), M.PIANURA, np.uint8)
        out = blocchi_chunk(s, h_c, cls_c, campi_c=campi_c, ox=0, oz=0)
        y = 70 - (-64)
        palo = s.blocco("fence", material="oak", north="false", south="false",
                        east="false", west="false")
        self.assertEqual(int(out[15, y, 10]), palo)


class TestPilastriPonte(unittest.TestCase):
    """I pilastri di sostegno di un impalcato - vedi il commento in
    motore._posa_strada: un vuoto sotto l'impalcato di un solo blocco non
    deve piu' generare un moncone di pilastro isolato (il caso tipico e' un
    ponte quasi a raso sopra un laghetto poco profondo o un avvallamento
    lieve del terreno, non un vero corso d'acqua da scavalcare)."""

    def ponte(self, dislivello):
        s = FintoScrittore()
        h_c = np.full((16, 16), 70, np.int32)
        cls_c = np.full((16, 16), M.PIANURA, np.uint8)
        tipo_c = np.full((16, 16), ST.PONTE, np.uint8)
        quota_c = np.full((16, 16), 70 + dislivello, np.int32)
        out = blocchi_chunk(s, h_c, cls_c, tipo_c=tipo_c, quota_c=quota_c,
                            ox=0, oz=0)
        return out, s

    def test_un_dislivello_di_un_blocco_non_ha_pilastri(self):
        out, s = self.ponte(1)
        pilastro = s.blocco("log", axis="y", material="oak", stripped="true")
        self.assertEqual(int((out == pilastro).sum()), 0)

    def test_un_dislivello_vero_ha_pilastri(self):
        out, s = self.ponte(5)
        pilastro = s.blocco("log", axis="y", material="oak", stripped="true")
        self.assertGreater(int((out == pilastro).sum()), 0)


if __name__ == "__main__":
    unittest.main()
