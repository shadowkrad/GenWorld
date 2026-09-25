"""Test dei biomi.

La prova che conta e' l'ultima: si genera un mondo, si riaprono i file region
e si confrontano i biomi letti con la mappa che li ha prodotti. E' lo stesso
metodo dello spike sulle quote, e serve per lo stesso motivo: un bioma
sbagliato non si vede da nessuna parte finche' non apri il gioco.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import biomi as B  # noqa: E402
from genworld.mappa import (CRATERE, DESERTO, FIUME, FORESTA, MARE, MONTAGNA,  # noqa: E402
                            NEVE, OCEANO, PIANURA, VULCANO)

try:
    import amulet  # noqa: F401
    from genworld.motore import Opzioni, analizza, genera
    AMULET = True
except ImportError:
    AMULET = False


def scenario(lato=120, quota=70):
    cls = np.full((lato, lato), PIANURA, np.uint8)
    h = np.full((lato, lato), quota, np.int32)
    return cls, h


class TestTavolozza(unittest.TestCase):

    def test_nomi_unici_e_indicizzati(self):
        self.assertEqual(len(set(B.BIOMI)), len(B.BIOMI))
        self.assertEqual(len(B.INDICE), len(B.BIOMI))

    def test_ogni_bioma_ha_un_colore_di_anteprima(self):
        self.assertEqual([n for n in B.BIOMI if n not in B.COLORI], [])

    @unittest.skipUnless(AMULET, "PyMCTranslate non installato")
    def test_tutti_i_nomi_traducono(self):
        """La trappola di `oak_log`, seconda edizione: un nome MODERNO non e'
        un nome universale. `universal_minecraft:snowy_plains` non esiste -
        l'universale e' `snowy_tundra` - e from_universal su un nome
        sconosciuto lo restituisce tale e quale, senza errore."""
        self.assertEqual(B.verifica_traduzioni(), [])

    @unittest.skipUnless(AMULET, "PyMCTranslate non installato")
    def test_un_nome_moderno_non_traduce(self):
        """Il controllo funziona: se non fosse cosi' sarebbe inutile."""
        import PyMCTranslate
        tr = PyMCTranslate.new_translation_manager()
        v = tr.get_version("java", (1, 21, 4))
        r = v.biome.from_universal("universal_minecraft:snowy_plains")
        self.assertFalse(r.startswith("minecraft:"),
                         "se questo passa, la trappola non esiste piu'")


class TestLatitudine(unittest.TestCase):

    def test_neve_in_alto_mette_il_freddo_in_alto(self):
        cls, _ = scenario()
        cls[:12, :] = NEVE
        lat = B.latitudine(cls)
        self.assertIsNotNone(lat)
        self.assertGreater(lat[0, 0], lat[-1, 0])

    def test_neve_in_basso_ribalta_l_asse(self):
        cls, _ = scenario()
        cls[-12:, :] = NEVE
        lat = B.latitudine(cls)
        self.assertLess(lat[0, 0], lat[-1, 0])

    def test_niente_neve_niente_gradiente(self):
        """Senza indizi non si inventa un nord."""
        cls, _ = scenario()
        self.assertIsNone(B.latitudine(cls))

    def test_neve_al_centro_e_quota_non_latitudine(self):
        cls, _ = scenario()
        cls[54:66, :] = NEVE
        self.assertIsNone(B.latitudine(cls))


class TestFreddo(unittest.TestCase):

    def test_sale_con_la_quota(self):
        cls, h = scenario()
        h[:, :60] = 70
        h[:, 60:] = 170
        f = B.freddo(cls, h)
        self.assertGreater(f[:, 60:].mean(), f[:, :60].mean() + 0.3)

    def test_il_deserto_resta_caldo_anche_in_quota(self):
        cls, h = scenario()
        cls[:, 60:] = DESERTO
        h[:, :] = 160
        f = B.freddo(cls, h)
        self.assertLess(f[:, 60:].max(), 0.2)

    def test_resta_fra_zero_e_uno(self):
        cls, h = scenario()
        h[:] = np.random.default_rng(0).integers(-40, 280, h.shape)
        f = B.freddo(cls, h)
        self.assertGreaterEqual(f.min(), 0.0)
        self.assertLessEqual(f.max(), 1.0)


class TestAssegnazione(unittest.TestCase):

    def nomi(self, cls, h):
        b = B.assegna(cls, h)
        return {B.BIOMI[i] for i in np.unique(b)}, b

    def test_l_acqua_prende_solo_biomi_d_acqua(self):
        """Un bioma di terra sull'oceano si vede subito: cambia il colore
        dell'acqua e ci nascono i mob sbagliati."""
        cls, h = scenario()
        cls[:] = OCEANO
        cls[:, :30] = MARE
        cls[:, 30:40] = FIUME
        h[:] = 50
        _, b = self.nomi(cls, h)
        for i in np.unique(b):
            n = B.BIOMI[int(i)]
            self.assertTrue("ocean" in n or "river" in n, f"{n} non e' acqua")

    def test_il_vulcano_e_badlands_non_un_bioma_del_nether(self):
        """`basalt_deltas` sarebbe stato perfetto a vedersi e sbagliato a
        giocarsi: nell'overworld porta la nebbia del Nether e i suoi mob."""
        cls, h = scenario()
        cls[40:60, 40:60] = VULCANO
        cls[48:52, 48:52] = CRATERE
        b = B.assegna(cls, h)
        self.assertEqual(B.BIOMI[int(b[50, 50])], "badlands")
        self.assertEqual(B.BIOMI[int(b[42, 42])], "badlands")

    def test_la_quota_cambia_il_bioma_a_parita_di_classe(self):
        cls, h = scenario()
        cls[:] = MONTAGNA
        h[:, :60] = 70
        h[:, 60:] = 185
        b = B.assegna(cls, h)
        basso = {B.BIOMI[int(i)] for i in np.unique(b[:, :55])}
        alto = {B.BIOMI[int(i)] for i in np.unique(b[:, 65:])}
        self.assertTrue(basso.isdisjoint(alto) or basso != alto,
                        f"stesso bioma in pianura e in vetta: {basso}")

    def test_dentro_una_classe_c_e_varieta(self):
        """Un continente coperto da un unico `forest` si legge come finto."""
        cls, h = scenario(lato=400)
        cls[:] = FORESTA
        b = B.assegna(cls, h)
        self.assertGreaterEqual(len(np.unique(b)), 2)

    def test_indici_validi(self):
        cls, h = scenario()
        rng = np.random.default_rng(1)
        cls[:] = rng.integers(0, 12, cls.shape)
        b = B.assegna(cls, h)
        self.assertLess(int(b.max()), len(B.BIOMI))

    def test_statistiche_ordinate(self):
        cls, h = scenario()
        cls[:80, :] = OCEANO
        h[:80, :] = 50
        s = B.statistiche(B.assegna(cls, h))
        valori = list(s.values())
        self.assertEqual(valori, sorted(valori, reverse=True))


@unittest.skipUnless(AMULET, "amulet non installato in questo interprete")
class TestScrittiSuDisco(unittest.TestCase):

    def test_i_biomi_riletti_corrispondono_alla_mappa(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            # isola verde con una cima chiara, come in test_motore
            lato = 192
            zz, xx = np.mgrid[:lato, :lato]
            dd = np.hypot(zz - lato / 2, xx - lato / 2)
            rgb = np.zeros((lato, lato, 3), np.uint8)
            rgb[...] = (40, 80, 160)
            rgb[dd < lato * 0.40] = (90, 150, 80)
            rgb[dd < lato * 0.15] = (240, 240, 245)
            img = os.path.join(d, "isola.png")
            Image.fromarray(rgb).save(img)

            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=64,
                         vulcani=0, alberi=0.0, villaggi=0.0, strade=False)
            genera(op)
            a = analizza(op)

            import amulet
            from genworld.motore import fetta
            lv = amulet.load_level(op.uscita)
            try:
                nomi = {}
                confrontate = diverse = 0
                meta = op.lato // 2
                for cx in range(-2, 2):
                    for cz in range(-2, 2):
                        ch = lv.get_chunk(cx, cz, "minecraft:overworld")
                        letti = np.asarray(ch.biomes[:, 4:5, :])[:, 0, :]
                        sx, sz = cx * 16 + meta, cz * 16 + meta
                        atteso = fetta(a.biomi, sx, sz)[::4, ::4]
                        for i in np.unique(letti):
                            if i not in nomi:
                                nomi[i] = str(lv.biome_palette[int(i)]).split(":")[-1]
                        ottenuto = np.vectorize(lambda i: B.INDICE[nomi[i]])(letti)
                        confrontate += ottenuto.size
                        diverse += int((ottenuto != atteso).sum())
            finally:
                lv.close()

            self.assertEqual(confrontate, 16 * 16)   # 16 chunk x 4x4 biomi
            self.assertEqual(diverse, 0,
                             f"{diverse} biomi diversi su {confrontate}")

    def test_il_mondo_non_e_tutto_plains(self):
        """Il difetto che ha fatto nascere questo modulo: ogni chunk usciva
        `plains`, cioe' lo stesso clima dal deserto alla tundra."""
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            lato = 192
            zz, xx = np.mgrid[:lato, :lato]
            dd = np.hypot(zz - lato / 2, xx - lato / 2)
            rgb = np.full((lato, lato, 3), (40, 80, 160), np.uint8)
            rgb[dd < lato * 0.40] = (90, 150, 80)
            rgb[dd < lato * 0.15] = (240, 240, 245)
            img = os.path.join(d, "isola.png")
            Image.fromarray(rgb).save(img)
            op = Opzioni(immagine=img, uscita=os.path.join(d, "w"), lato=64,
                         vulcani=0, alberi=0.0, villaggi=0.0, strade=False)
            genera(op)
            import amulet
            lv = amulet.load_level(op.uscita)
            try:
                visti = set()
                for cx in range(-2, 2):
                    for cz in range(-2, 2):
                        ch = lv.get_chunk(cx, cz, "minecraft:overworld")
                        for i in np.unique(np.asarray(ch.biomes[:, 4:5, :])):
                            visti.add(str(lv.biome_palette[int(i)]))
            finally:
                lv.close()
            self.assertGreater(len(visti), 2, f"solo {visti}")


if __name__ == "__main__":
    unittest.main()
