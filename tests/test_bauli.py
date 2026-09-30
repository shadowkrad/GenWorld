"""Test del bottino dei forzieri.

`template.carica()` non porta con se' il tag NBT di un blocco (vedi il
docstring di `bauli.py`): un forziere che arriva da un file `.nbt` di casa
e' sempre vuoto quando si stampa il template. Qui si controlla che
`bauli.trova()` lo trovi DOPO, dal blocco gia' scritto, e gli dia un
contenuto - senza toccare i forzieri doppi, che restano vuoti apposta.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from amulet_nbt import load as nbt_load  # noqa: F401
    from genworld import bauli as BA
    from genworld.livello_dat import ImpostazioniMondo
    from genworld.mondo import ScrittoreMondo
    AMULET = True
except ImportError:
    AMULET = False


class FintoScrittore:
    """Stessa idea di `test_motore.FintoScrittore`: un id progressivo per
    ogni (nome, proprieta'), niente livello Minecraft vero da aprire."""
    id_aria = 0

    def __init__(self):
        self.voci: dict = {}

    def blocco(self, nome, **prop):
        chiave = (nome, tuple(sorted(prop.items())))
        return self.voci.setdefault(chiave, len(self.voci) + 1)


@unittest.skipUnless(AMULET, "amulet non installato in questo interprete")
class TestTrova(unittest.TestCase):

    ALTEZZA = 40
    Y0 = 0

    def chunk_con_forziere(self, facing="north", connection="none"):
        s = FintoScrittore()
        tav = BA.Tavolozza(s)
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        idf = s.blocco("chest", facing=facing, connection=connection,
                       material="wood")
        out[5, 10, 6] = idf
        return out, s, tav, idf

    def test_un_forziere_singolo_prende_un_contenuto(self):
        out, s, tav, idf = self.chunk_con_forziere()
        bauli = BA.trova(out, tav, 0, 0, self.Y0, seed=1)
        self.assertEqual(len(bauli), 1)
        be = bauli[0]
        self.assertEqual((be.x, be.y, be.z), (5, 10, 6))
        items = be.nbt.compound["utags"]["Items"]
        self.assertGreaterEqual(len(items), 2)
        self.assertLessEqual(len(items), 5)
        for voce in items:
            self.assertTrue(str(voce["id"]).startswith("minecraft:"))
            self.assertGreater(int(voce["count"]), 0)

    def test_le_quattro_orientazioni_si_trovano_tutte(self):
        s = FintoScrittore()
        tav = BA.Tavolozza(s)
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        posizioni = {"north": (1, 1), "south": (2, 2), "east": (3, 3),
                    "west": (4, 4)}
        for facing, (x, z) in posizioni.items():
            out[x, 10, z] = s.blocco("chest", facing=facing, connection="none",
                                     material="wood")
        bauli = BA.trova(out, tav, 0, 0, self.Y0, seed=1)
        self.assertEqual(len(bauli), 4)

    def test_un_forziere_doppio_resta_vuoto(self):
        """`connection="left"`/`"right"` e' un baule a due battenti con
        l'inventario condiviso - riempirne solo una meta' si nota, quindi
        si lascia stare."""
        for connection in ("left", "right"):
            out, s, tav, idf = self.chunk_con_forziere(connection=connection)
            bauli = BA.trova(out, tav, 0, 0, self.Y0, seed=1)
            self.assertEqual(bauli, [])

    def test_niente_forzieri_niente_bottino(self):
        s = FintoScrittore()
        tav = BA.Tavolozza(s)
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        self.assertEqual(BA.trova(out, tav, 0, 0, self.Y0, seed=1), [])

    def test_il_contenuto_e_deterministico_sulla_posizione(self):
        """Lo stesso forziere, rigenerato con lo stesso seme, ha sempre lo
        stesso bottino - altrimenti due lotti generati due volte
        divergerebbero senza che nessuno abbia cambiato niente."""
        out, s, tav, idf = self.chunk_con_forziere()
        a = BA.trova(out, tav, 0, 0, self.Y0, seed=1)
        b = BA.trova(out, tav, 0, 0, self.Y0, seed=1)
        self.assertEqual(a[0].nbt.compound["utags"]["Items"],
                         b[0].nbt.compound["utags"]["Items"])

    def test_coordinate_assolute_con_offset_di_chunk(self):
        out, s, tav, idf = self.chunk_con_forziere()
        bauli = BA.trova(out, tav, 32, -16, -64, seed=1)
        be = bauli[0]
        self.assertEqual((be.x, be.y, be.z), (32 + 5, -64 + 10, -16 + 6))

    def test_ender_chest_e_trapped_chest_non_si_riempiono(self):
        """Solo il `chest` di legno e' la dispensa di una casa: un ender
        chest e' lo storage personale del giocatore, un trapped chest
        aziona un meccanismo - riempirli non avrebbe senso."""
        s = FintoScrittore()
        tav = BA.Tavolozza(s)
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        out[1, 10, 1] = s.blocco("ender_chest", facing="north")
        out[2, 10, 2] = s.blocco("trapped_chest", facing="north",
                                 connection="none")
        self.assertEqual(BA.trova(out, tav, 0, 0, self.Y0, seed=1), [])


@unittest.skipUnless(AMULET, "amulet non installato in questo interprete")
class TestPersistenzaSuMondoVero(unittest.TestCase):
    """Il difetto vero non si vedeva su un chunk solo.

    `BlockEntity("minecraft", "chest", ...)` con un NBT piatto sopravviveva
    al salvataggio/rilettura quando il mondo aveva UN chunk soltanto, e
    tornava vuoto ("Items" azzerato) appena la sessione di scrittura ne
    toccava piu' di uno - anche un chunk qualunque degli altri tre, senza
    forzieri. Un test su un solo chunk (come quelli sopra) non l'avrebbe mai
    visto: qui si scrive un mondo vero di piu' chunk con `ScrittoreMondo` e
    si rilegge con `amulet.load_level`, la stessa strada di
    `test_mondo_valido.py`.
    """

    def test_il_bottino_sopravvive_su_piu_chunk(self):
        import amulet

        with tempfile.TemporaryDirectory() as d:
            percorso = os.path.join(d, "w")
            imp = ImpostazioniMondo(nome="Prova", modalita=1)
            posizioni = {}
            with ScrittoreMondo(percorso, imp, lato_blocchi=32, crea=True) as m:
                pietra = m.blocco("stone")
                tav = BA.Tavolozza(m)
                prato = m.bioma("plains")
                for cx, cz in m.chunk_coords():
                    ox, oz = m.origine_chunk(cx, cz)
                    b = np.full((16, 40, 16), m.id_aria, np.uint32)
                    b[:, :10, :] = pietra
                    # ogni chunk ha il suo forziere, cosi' un difetto che
                    # colpisse solo alcuni chunk non passerebbe inosservato
                    idf = m.blocco("chest", facing="north", connection="none",
                                   material="wood")
                    b[8, 10, 8] = idf
                    bauli = BA.trova(b, tav, ox, oz, -64, seed=1)
                    self.assertEqual(len(bauli), 1)
                    posizioni[(cx, cz)] = (bauli[0].x, bauli[0].y, bauli[0].z)
                    m.scrivi_chunk(cx, cz, b,
                                   biomi=np.full((16, 16), prato, np.uint32),
                                   bauli=bauli)

            self.assertGreaterEqual(len(posizioni), 4,
                                    "il test serve un mondo di piu' chunk")

            lv = amulet.load_level(percorso)
            try:
                for (cx, cz), (gx, gy, gz) in posizioni.items():
                    ch = lv.get_chunk(cx, cz, "minecraft:overworld")
                    be = ch.block_entities.get((gx, gy, gz))
                    self.assertIsNotNone(
                        be, f"nessun block-entity a {(gx, gy, gz)} nel chunk "
                            f"{(cx, cz)}: il forziere e' tornato vuoto")
                    utags = be.nbt.compound["utags"]
                    items = utags["Items"]
                    self.assertGreaterEqual(
                        len(items), 2,
                        f"forziere del chunk {(cx, cz)} tornato senza "
                        f"contenuto dopo il salvataggio/rilettura")
            finally:
                lv.close()


if __name__ == "__main__":
    unittest.main()
