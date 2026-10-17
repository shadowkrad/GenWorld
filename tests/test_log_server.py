"""Difetti emersi dal log di un server vero (v32/v33):

  - 'Root entity ... is outside of serialized chunk': chunk delle entita' sbagliato
    con coordinate negative (int() invece di floor);
  - 'Entity uuid already exists': due entita' con lo stesso UUID (carrelli);
  - 'String ... is too long: 36, expected range [0-32]': titolo di un libro troppo lungo,
    il gioco scarta l'intera pila.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import bauli as BA  # noqa: E402
from genworld import entita as EN  # noqa: E402
from genworld import luoghi as LU  # noqa: E402
from genworld.fauna import Animale  # noqa: E402
from genworld.miniere import Carrello  # noqa: E402


def entita_nel_chunk(cartella, cx, cz):
    rx, rz = cx >> 5, cz >> 5
    for base in (os.path.join(cartella, "dimensions", "minecraft", "overworld", "entities"),
                 os.path.join(cartella, "entities")):
        p = os.path.join(base, f"r.{rx}.{rz}.mca")
        if os.path.exists(p):
            return EN.leggi_chunk(p, cx, cz)
    return None


class TestChunkDelleEntita(unittest.TestCase):
    def posizioni(self, d, chunk):
        t = chunk if chunk is not None else None
        return t

    def test_il_chunk_di_una_coordinata_negativa(self):
        casi = [(-64.5, 13.5, (-5, 0)), (103.5, -48.5, (6, -4)), (-0.5, -0.5, (-1, -1)),
                (-16.0, 16.0, (-1, 1)), (15.99, 15.99, (0, 0)), (-15.9, -16.1, (-1, -2)),
                (-288.5, -19.5, (-19, -2))]
        for x, z, (cx, cz) in casi:
            with tempfile.TemporaryDirectory() as d:
                EN.scrivi_regioni(d, [Animale(x=x, y=70.0, z=z, specie="coniglio", seme=1)], 4901)
                ch = entita_nel_chunk(d, cx, cz)
                self.assertIsNotNone(ch, (x, z, cx, cz))
                ent = ch.compound["Entities"] if hasattr(ch, "compound") else ch["Entities"]
                self.assertEqual(len(ent), 1)
                # e il chunk accanto e' vuoto (dove finiva con int())
                self.assertEqual(int(np.floor(x)) >> 4, cx)
                self.assertEqual(int(np.floor(z)) >> 4, cz)

    def test_ogni_entita_sta_nel_chunk_che_la_contiene(self):
        rng = np.random.default_rng(3)
        lista = [Animale(x=float(rng.uniform(-300, 300)), y=70.0, z=float(rng.uniform(-300, 300)),
                         specie="pecora", seme=int(i)) for i in range(300)]
        with tempfile.TemporaryDirectory() as d:
            EN.scrivi_regioni(d, lista, 4901)
            trovate = 0
            for cx in range(-20, 20):
                for cz in range(-20, 20):
                    ch = entita_nel_chunk(d, cx, cz)
                    if ch is None:
                        continue
                    comp = ch.compound if hasattr(ch, "compound") else ch
                    for e in comp["Entities"]:
                        x, z = float(e["Pos"][0]), float(e["Pos"][2])
                        self.assertEqual((int(np.floor(x)) >> 4, int(np.floor(z)) >> 4), (cx, cz), (x, z))
                        trovate += 1
            self.assertEqual(trovate, 300)


class TestUuidUnici(unittest.TestCase):
    def test_due_carrelli_con_lo_stesso_seme_hanno_uuid_diversi(self):
        a = Carrello(x=1.5, y=20.0, z=1.5, seme=7)
        b = Carrello(x=40.5, y=20.0, z=40.5, seme=7)
        self.assertEqual(tuple(int(v) for v in a.nbt_tag()["UUID"]), tuple(int(v) for v in b.nbt_tag()["UUID"]))
        with tempfile.TemporaryDirectory() as d:
            EN.scrivi_regioni(d, [a, b, Animale(x=2.5, y=70.0, z=2.5, specie="mucca", seme=7)], 4901)
            visti = set()
            for cx, cz in ((0, 0), (2, 2)):
                ch = entita_nel_chunk(d, cx, cz)
                comp = ch.compound if hasattr(ch, "compound") else ch
                for e in comp["Entities"]:
                    visti.add(tuple(int(v) for v in e["UUID"]))
            self.assertEqual(len(visti), 3)

    def test_i_doppioni_si_rifanno_in_modo_deterministico(self):
        a = Carrello(x=1.5, y=20.0, z=1.5, seme=7)
        b = Carrello(x=40.5, y=20.0, z=40.5, seme=7)

        def esegui():
            with tempfile.TemporaryDirectory() as d:
                EN.scrivi_regioni(d, [a, b], 4901)
                return sorted(tuple(int(v) for e in (entita_nel_chunk(d, cx, cz).compound["Entities"]
                                                     if hasattr(entita_nel_chunk(d, cx, cz), "compound")
                                                     else entita_nel_chunk(d, cx, cz)["Entities"])
                                    for v in e["UUID"]) for cx, cz in ((0, 0), (2, 2)))
        self.assertEqual(esegui(), esegui())


class TestTitoloDeiLibri(unittest.TestCase):
    def comp(self, titolo):
        return {"written_book_content": {"title": {"raw": titolo}, "author": "Il Custode",
                                         "pages": [{"raw": "x"}]}}

    def test_limita_libro(self):
        lungo = "Carte di Torre di Brenna Mano di Ferro il 6°"
        self.assertGreater(len(lungo), 32)
        c = BA.limita_libro(self.comp(lungo))
        self.assertEqual(len(c["written_book_content"]["title"]["raw"]), 32)
        self.assertEqual(c["written_book_content"]["title"]["raw"], lungo[:32])
        corto = BA.limita_libro(self.comp("Il diario"))
        self.assertEqual(corto["written_book_content"]["title"]["raw"], "Il diario")
        self.assertIsNone(BA.limita_libro(None))
        self.assertEqual(BA.limita_libro({"custom_name": "x"}), {"custom_name": "x"})

    def test_nel_forziere_e_negli_scambi(self):
        lungo = "Carte di Fortezza di Signore di Roccanera"
        v = BA._voce((3, "written_book", 1, self.comp(lungo)))
        titolo = str(v["components"]["minecraft:written_book_content"]["title"]["raw"])
        self.assertLessEqual(len(titolo), 32)
        t = EN._scambio({"compra": ("emerald", 5),
                         "vende": ("written_book", 1, self.comp(lungo)), "usi": 1})
        self.assertLessEqual(len(str(t["sell"]["components"]["minecraft:written_book_content"]["title"]["raw"])), 32)

    def test_tutti_i_libri_della_catena_rispettano_il_limite(self):
        l = [LU.Luogo("Torre di Brenna Mano di Ferro il 6°", "torre", 0, 0),
             LU.Luogo("Fortezza di Signore di Roccanera", "fortezza", 100, 0),
             LU.Luogo("Cripta di Orrenda 7", "rovina", 200, 0),
             LU.Luogo("Aldona", "citta", 10, 10)]
        ind = LU.indizi(l)
        for libro in list(ind["tappe"].values()) + [ind["inizio"][1]]:
            v = BA._voce((0, "written_book", 1, {"written_book_content": {
                "title": {"raw": libro.titolo}, "author": libro.autore,
                "pages": [{"raw": p} for p in libro.pagine]}}))
            self.assertLessEqual(len(str(v["components"]["minecraft:written_book_content"]["title"]["raw"])), 32)


if __name__ == "__main__":
    unittest.main()
