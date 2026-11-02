"""Scrittura in parallelo (#16): stesso mondo di quella seriale."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genworld import parallelo as PA  # noqa: E402
from genworld.motore import Opzioni, analizza, pianifica, scrivi  # noqa: E402
from test_motore import mappa_finta  # noqa: E402


class TestRipartizione(unittest.TestCase):
    def test_coordinate_come_lo_scrittore(self):
        c = PA.coordinate_chunk(64)
        self.assertEqual(len(c), 16)
        self.assertEqual(c[0], (-2, -2))
        self.assertEqual(c[-1], (1, 1))

    def test_ogni_chunk_in_un_solo_gruppo(self):
        c = PA.coordinate_chunk(1000)
        gruppi = PA.ripartisci(c, 8)
        tutti = sorted(k for g in gruppi for k in g)
        self.assertEqual(tutti, list(range(len(c))))
        self.assertLessEqual(len(gruppi), 8)

    def test_una_regione_non_e_divisa_fra_gruppi(self):
        c = PA.coordinate_chunk(1000)
        gruppi = PA.ripartisci(c, 8)
        dove = {}
        for n, g in enumerate(gruppi):
            for k in g:
                cx, cz = c[k]
                dove.setdefault((cx // 32, cz // 32), set()).add(n)
        self.assertTrue(all(len(v) == 1 for v in dove.values()))

    def test_carico_bilanciato(self):
        c = PA.coordinate_chunk(1000)
        gruppi = PA.ripartisci(c, 4)
        pesi = sorted(len(g) for g in gruppi)
        self.assertLessEqual(pesi[-1], 2 * pesi[0] + 1)


def _chunk_come_testo(percorso):
    """{(cx, cz): array di stringhe dei blocchi, n. di block entity}"""
    import amulet
    lv = amulet.load_level(percorso)
    fuori = {}
    try:
        dim = "minecraft:overworld"
        for cx, cz in sorted(lv.all_chunk_coords(dim)):
            ch = lv.get_chunk(cx, cz, dim)
            righe = []
            for cy in sorted(ch.blocks.sub_chunks):
                sub = ch.blocks.get_sub_chunk(cy)
                nomi = np.array([str(ch.block_palette[i]) for i in range(len(ch.block_palette))],
                                dtype=object)
                righe.append((cy, nomi[sub]))
            fuori[(cx, cz)] = (righe, len(ch.block_entities))
    finally:
        lv.close()
    return fuori


class TestEquivalenza(unittest.TestCase):
    def test_il_mondo_parallelo_e_uguale_al_seriale(self):
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 256)
            base = dict(immagine=img, lato=256, vulcani=0, alberi=0.3, villaggi=1.0,
                        strade=True, caverne=1.0, miniere=1.0, accampamenti=4.0,
                        cimiteri=4.0)
            op_s = Opzioni(uscita=os.path.join(d, "seriale"), **base)
            op_p = Opzioni(uscita=os.path.join(d, "parallelo"), **base)
            a = analizza(op_s)
            piano = pianifica(op_s, a)
            st_s = scrivi(op_s, a, piano)
            st_p = PA.scrivi_parallelo(op_p, a, piano, processi=2)
            self.assertEqual(st_s["chunk"], st_p["chunk"])
            self.assertEqual(st_p["restano"], 0)
            self.assertEqual(st_p.get("level_dat"), "valido")
            self.assertEqual(st_s.get("abitanti"), st_p.get("abitanti"))
            # nessun avanzo dei mondi di lavoro accanto al mondo
            self.assertEqual([n for n in os.listdir(d) if n.startswith("genworld_")], [])
            s = _chunk_come_testo(op_s.uscita)
            p = _chunk_come_testo(op_p.uscita)
            self.assertEqual(sorted(s), sorted(p))
            for k in s:
                self.assertEqual(s[k][1], p[k][1], f"block entity del chunk {k}")
                self.assertEqual(len(s[k][0]), len(p[k][0]))
                for (cy1, r1), (cy2, r2) in zip(s[k][0], p[k][0]):
                    self.assertEqual(cy1, cy2)
                    self.assertTrue((r1 == r2).all(), f"blocchi diversi nel chunk {k}, y {cy1}")


if __name__ == "__main__":
    unittest.main()


class TestArrayCondivisi(unittest.TestCase):
    def test_gli_array_grandi_vanno_su_disco_e_tornano_mappati(self):
        import dataclasses
        import tempfile
        import numpy as np
        from genworld import parallelo as PA

        @dataclasses.dataclass
        class Piano:
            grande: np.ndarray
            piccolo: np.ndarray
            dizionario: dict
            elenco: list
            numero: int = 3

        grande = np.arange(PA.SOGLIA_ARRAY // 4 + 10, dtype=np.int32)
        p = Piano(grande=grande, piccolo=np.arange(5), dizionario={"a": grande, "b": np.arange(3)},
                  elenco=[grande, 7])
        with tempfile.TemporaryDirectory() as d:
            q = PA.su_disco((p, "x"), d)
            self.assertIsInstance(q[0].grande, PA.RifArray)
            self.assertIsInstance(q[0].dizionario["a"], PA.RifArray)
            self.assertEqual(q[0].grande.percorso, q[0].dizionario["a"].percorso)      # una copia sola
            self.assertIs(q[0].piccolo, p.piccolo)                                       # i piccoli restano
            self.assertEqual(q[0].numero, 3)
            self.assertEqual(len(os.listdir(d)), 1)
            r = PA.da_disco(q)
            self.assertTrue(np.array_equal(r[0].grande, grande))
            self.assertTrue(np.array_equal(r[0].dizionario["a"], grande))
            self.assertEqual(r[0].elenco[1], 7)
            self.assertEqual(r[1], "x")
            self.assertFalse(r[0].grande.flags.writeable)          # sola lettura: niente memoria impegnata in piu'
            del r


class TestRiparazione(unittest.TestCase):
    def test_le_coordinate_dei_chunk_non_salvati_si_leggono_dal_log(self):
        testo = ("INFO - boh\nERROR - Error saving chunk Chunk(-405, -512, UnboundedPartial3DArray(dtype=x), "
                 "EntityList([]))\nERROR - Error saving chunk Chunk(7, 3, Unb)\n"
                 "ERROR - Error saving chunk Chunk(7, 3, Unb)\nERROR - altro Chunk(1, 2, x)")
        self.assertEqual(PA.chiavi_da_log(testo), [(-405, -512), (7, 3)])


class TestRiparazioneMondo(unittest.TestCase):
    def test_i_chunk_mancanti_si_rifanno_uguali_a_quelli_di_un_mondo_intero(self):
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 256)
            base = dict(immagine=img, lato=256, vulcani=0, alberi=0.3, villaggi=1.0, strade=True,
                        caverne=1.0, miniere=1.0, accampamenti=4.0, cimiteri=4.0)
            op_r = Opzioni(uscita=os.path.join(d, "intero"), **base)
            op_p = Opzioni(uscita=os.path.join(d, "riparato"), **base)
            a = analizza(op_r)
            piano = pianifica(op_r, a)
            scrivi(op_r, a, piano)
            tutti = PA.coordinate_chunk(256)
            mancanti = [3, 17, 200]
            resto = [k for k in range(len(tutti)) if k not in mancanti]
            scrivi(op_p, a, piano, da=0, indici=resto, lavoratore=True)
            self.assertEqual(len(_chunk_come_testo(op_p.uscita)), len(tutti) - 3)
            from genworld.motore import mappe_globali
            st = scrivi(op_p, a, piano, da=1, indici=mancanti, lavoratore=True,
                        precaricati={**__import__("genworld.motore", fromlist=["carica_modelli"])
                                     .carica_modelli(op_p, piano), "mappe": mappe_globali(op_p, a, piano)})
            self.assertEqual(st["chunk"], 3)
            r = _chunk_come_testo(op_r.uscita)
            p = _chunk_come_testo(op_p.uscita)
            self.assertEqual(sorted(r), sorted(p))
            for k in mancanti:
                coord = tutti[k]
                for (c1, b1), (c2, b2) in zip(r[coord][0], p[coord][0]):
                    self.assertTrue((b1 == b2).all(), f"blocchi diversi nel chunk riparato {coord}")


class TestRipresa(unittest.TestCase):
    def test_una_scrittura_interrotta_si_riprende_e_il_mondo_e_uguale_all_intero(self):
        import pickle
        from dataclasses import replace
        from genworld.motore import carica_modelli, con_uscita, mappe_globali
        with tempfile.TemporaryDirectory() as d:
            img = mappa_finta(os.path.join(d, "isola.png"), 256)
            base_opz = dict(immagine=img, lato=256, vulcani=0, alberi=0.3, villaggi=1.0, strade=True,
                            caverne=1.0, miniere=1.0, accampamenti=4.0, cimiteri=4.0)
            op_s = Opzioni(uscita=os.path.join(d, "seriale"), **base_opz)
            op_r = Opzioni(uscita=os.path.join(d, "ripreso"), **base_opz)
            a = analizza(op_s)
            piano = pianifica(op_s, a)
            scrivi(op_s, a, piano)
            # la situazione dopo un'interruzione: cartella di lavoro con i dati e due mondi a meta'
            coords = PA.coordinate_chunk(256)
            gruppi = PA.ripartisci(coords, 2)
            base = os.path.join(d, "genworld_prova")
            os.makedirs(base)
            if piano.cls is not None:
                a = replace(a, cls=piano.cls, livello=piano.livello)
            pre = carica_modelli(op_r, piano)
            pre["mappe"] = mappe_globali(op_r, a, piano)
            with open(os.path.join(base, "dati.pkl"), "wb") as f:
                pickle.dump(PA.su_disco((op_r, a, piano, pre), base), f)
            for k, g in enumerate(gruppi):
                meta = g[:len(g) // 2]                      # solo la prima meta' dei chunk del gruppo
                scrivi(con_uscita(op_r, os.path.join(base, f"w{k}"), op_r.nome), a, piano,
                       indici=meta, lavoratore=True, precaricati=pre)
            presenti = PA.chunk_presenti(os.path.join(base, "w0"))
            self.assertTrue(0 < len(presenti) < len(gruppi[0]))
            st = PA.riprendi_parallelo(base, processi=2)
            self.assertEqual(st["chunk"], len(coords))
            self.assertEqual(st.get("level_dat"), "valido")
            self.assertFalse(os.path.exists(base))                    # a fine lavoro la cartella sparisce
            s = _chunk_come_testo(op_s.uscita)
            r = _chunk_come_testo(op_r.uscita)
            self.assertEqual(sorted(s), sorted(r))
            for k in s:
                self.assertEqual(s[k][1], r[k][1], f"block entity del chunk {k}")
                for (c1, b1), (c2, b2) in zip(s[k][0], r[k][0]):
                    self.assertTrue((b1 == b2).all(), f"blocchi diversi nel chunk {k}")
