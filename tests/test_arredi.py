"""Test degli arredi urbani.

Il pezzo delicato non e' disegnarli ma che siano VERI in gioco: ogni blocco
deve essere traducibile (un nome che PyMCTranslate non conosce si scrive
benissimo e non compare), le staccionate devono essere collegate (un mondo
scritto a mano non ricalcola le forme), e un arredo non deve mai finire sopra
una casa, una strada o un'altra cosa gia' posata.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from genworld import arredi as A
    from genworld import mappa as M
    from genworld import template as TM
    from genworld.citta import Citta
    from genworld.edifici import Edificio, NORD, SUD, EST, OVEST
    import PyMCTranslate  # noqa: F401
    DISPONIBILE = True
except ImportError:
    DISPONIBILE = False


@unittest.skipUnless(DISPONIBILE, "amulet/PyMCTranslate non disponibili")
class TestDisegni(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ver = TM.traduttore()

    def modello(self, d):
        return d.modello(self.ver)

    def nomi(self, m):
        return {n for n, _ in m.tavolozza}

    def test_gli_arredi_fissi_si_costruiscono_e_si_traducono(self):
        for fabbrica, dim in ((A.lampione, (1, 4, 1)), (A.campana, (3, 3, 1)),
                              (A.fontana, (9, 11, 9)), (A.fontana_piccola, (5, 3, 5)), (A.pozzo, (3, 4, 3)),
                              (A.panchina, (3, 1, 1))):
            m = self.modello(fabbrica())
            self.assertEqual(m.celle.shape, dim)
            self.assertGreater(int((m.celle >= 0).sum()), 0)

    def test_il_lampione_ha_la_lanterna_in_cima(self):
        m = self.modello(A.lampione())
        self.assertIn("lantern", self.nomi(m))
        n, _ = m.tavolozza[int(m.celle[0, 3, 0])]
        self.assertEqual(n, "lantern")

    def test_la_campana_c_e(self):
        self.assertIn("bell", self.nomi(self.modello(A.campana())))

    def test_la_fontana_ha_acqua_chiusa_da_un_anello(self):
        d = A.fontana_piccola()
        m = self.modello(d)
        self.assertIn("water", self.nomi(m))
        acqua = [i for i, (n, _) in enumerate(m.tavolozza) if n == "water"][0]
        for x, _, z in zip(*np.nonzero(m.celle == acqua)):
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                vicino = m.celle[x + dx, 0, z + dz]
                self.assertNotEqual(vicino, -1, "l'acqua ha un lato scoperto e scorrerebbe via")

    def test_la_fontana_grande_ha_l_acqua_sigillata_in_ogni_piatto(self):
        """Bacino e piatti: ogni cella d'acqua ha un solido sotto e sui
        quattro lati. Le tende sono vetro, non acqua che scorre."""
        m = self.modello(A.fontana())
        acqua = [i for i, (n, _) in enumerate(m.tavolozza) if n == "water"][0]
        livelli = set()
        for x, y, z in zip(*np.nonzero(m.celle == acqua)):
            livelli.add(int(y))
            if y > 0:                        # a y=0 sotto c'e' il terreno
                self.assertNotEqual(m.celle[x, y - 1, z], -1)
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                self.assertNotEqual(m.celle[x + dx, y, z + dz], -1)
        self.assertEqual(livelli, {0, 4, 8})
        self.assertIn("stained_glass", " ".join(n for n, _ in m.tavolozza))

    def test_un_blocco_non_traducibile_solleva_invece_di_sparire(self):
        d = A.Disegno("prova", 1, 1, 1)
        d.metti(0, 0, 0, "un_blocco_mai_visto")
        with self.assertRaises(ValueError):
            d.modello(self.ver)

    def test_la_staccionata_e_collegata_ai_vicini(self):
        rng = np.random.default_rng(0)
        g, _ = A.recinto(8, 6, SUD, rng)
        m = self.modello(g)
        for x in range(8):
            for z in range(6):
                i = int(m.celle[x, 0, z])
                if i < 0 or m.tavolozza[i][0] != "fence":
                    continue
                p = m.tavolozza[i][1]
                collegata = [p[k] for k in ("north", "south", "east", "west")]
                self.assertIn("true", collegata, f"palo isolato in ({x},{z})")

    def test_il_cancello_sta_sul_lato_della_strada(self):
        rng = np.random.default_rng(0)
        for verso, cella in ((NORD, lambda x, z: z == 0), (SUD, lambda x, z: z == 5),
                             (OVEST, lambda x, z: x == 0), (EST, lambda x, z: x == 7)):
            g, _ = A.recinto(8, 6, verso, rng)
            m = self.modello(g)
            gate = [i for i, (n, _) in enumerate(m.tavolozza) if n == "fence_gate"]
            self.assertEqual(len(gate), 1)
            for x, _, z in zip(*np.nonzero(m.celle == gate[0])):
                self.assertTrue(cella(int(x), int(z)), f"cancello sul lato sbagliato per {verso}")

    def test_le_bestie_stanno_dentro_e_non_sul_fieno(self):
        rng = np.random.default_rng(3)
        g, bestie = A.recinto(9, 7, NORD, rng)
        m = self.modello(g)
        self.assertGreater(len(bestie), 0)
        for lx, lz in bestie:
            self.assertTrue(1 <= lx < 8 and 1 <= lz < 6, "fuori dallo steccato")
            self.assertEqual(int(m.celle[int(lx), 0, int(lz)]), -1,
                             "un animale dentro un blocco")

    def test_il_giardino_ha_un_varco_sul_lato_della_strada(self):
        rng = np.random.default_rng(0)
        g = A.giardino(8, 6, NORD, rng)
        m = self.modello(g)
        riga_strada = [int(m.celle[x, 0, 0]) for x in range(8)]
        self.assertGreaterEqual(riga_strada.count(-1), 2, "niente varco d'ingresso")
        riga_opposta = [int(m.celle[x, 0, 5]) for x in range(8)]
        self.assertEqual(riga_opposta.count(-1), 0, "la siepe dovrebbe chiudere dietro")

    def test_i_riempitivi_prendono_la_misura_del_lotto(self):
        rng = np.random.default_rng(0)
        for w, d in ((6, 6), (7, 11), (13, 8)):
            self.assertEqual(self.modello(A.giardino(w, d, NORD, rng)).celle.shape[::2], (w, d))
            g, _ = A.recinto(w, d, EST, rng)
            self.assertEqual(self.modello(g).celle.shape[::2], (w, d))
        self.assertEqual(self.modello(A.piazzetta(8, 9, NORD, rng)).celle.shape[::2], (8, 9))


def _edificio(x, z, w, d, villaggio=0, porta=NORD):
    return Edificio(x=x, z=z, larghezza=w, profondita=d, base=70, gronda=0,
                    porta=porta, villaggio=villaggio)


@unittest.skipUnless(DISPONIBILE, "amulet/PyMCTranslate non disponibili")
class TestPianificazione(unittest.TestCase):
    """Un paese finto su un prato piatto: quattro lotti, uno con casa."""

    def scenario(self, con_strada=False):
        H = W = 120
        h = np.full((H, W), 70, np.int32)
        cls = np.full((H, W), M.PIANURA, np.uint8)
        edifici = [_edificio(20, 20, 8, 8), _edificio(34, 20, 8, 8),
                   _edificio(20, 34, 9, 7), _edificio(34, 34, 12, 10)]
        citta = [Citta(z=40, x=40, raggio=35, piazza=(60, 60))]
        strada = np.zeros((H, W), np.uint8)
        vie = np.zeros((H, W), np.uint8)
        if con_strada:
            strada[48:51, 10:110] = 1                  # una via che taglia il paese
            vie[49, 10:110] = 3
        return dict(edifici=edifici, scelte={0: (0, 0)}, citta=citta, h=h, cls=cls,
                    vie=vie, tipo_strada=strada, muro=np.zeros((H, W), np.uint8),
                    campi=None, banchi=[])

    def piano(self, **kw):
        s = self.scenario(kw.pop("con_strada", False))
        s.update(kw)
        return A.pianifica(**s, seed=3), s

    def test_ogni_lotto_senza_casa_riceve_un_arredo(self):
        ris, s = self.piano()
        for i, e in enumerate(s["edifici"]):
            if i == 0:
                continue
            coperto = any(a.x == e.x and a.z == e.z for a in ris.arredi) or any(
                b.x >= e.x and b.x1 <= e.x + e.larghezza and b.z >= e.z
                and b.z1 <= e.z + e.profondita for b in ris.banchi)
            self.assertTrue(coperto, f"lotto {i} lasciato vuoto")

    def test_il_lotto_con_casa_non_riceve_niente(self):
        ris, s = self.piano()
        e = s["edifici"][0]
        for a in ris.arredi:
            self.assertFalse(a.x < e.x + e.larghezza and a.x + a.larghezza > e.x
                             and a.z < e.z + e.profondita and a.z + a.profondita > e.z,
                             f"{a.tipo} sopra una casa")

    def test_ogni_paese_ha_una_campana_e_una_fontana_o_un_pozzo(self):
        ris, _ = self.piano()
        tipi = [a.tipo for a in ris.arredi]
        self.assertEqual(tipi.count("campana"), 1)
        self.assertEqual(sum(t in ("fontana", "pozzo") for t in tipi), 1)
        self.assertIn("fontana", tipi, "una citta' (raggio 35) vuole la fontana")

    def test_un_borgo_ha_il_pozzo(self):
        s = self.scenario()
        s["citta"] = [Citta(z=40, x=40, raggio=18, piazza=(60, 60))]
        ris = A.pianifica(**s, seed=3)
        tipi = [a.tipo for a in ris.arredi]
        self.assertIn("pozzo", tipi)
        self.assertNotIn("fontana", tipi)

    def test_nessun_arredo_ne_si_sovrappone_a_un_altro(self):
        ris, s = self.piano(con_strada=True)
        occ = np.zeros(s["h"].shape, int)
        for a in ris.arredi:
            occ[a.z:a.z + a.profondita, a.x:a.x + a.larghezza] += 1
        self.assertEqual(int(occ.max()), 1, "due arredi sulla stessa cella")

    def test_niente_sulla_carreggiata(self):
        ris, s = self.piano(con_strada=True)
        carreggiata = s["tipo_strada"] > 0
        for a in ris.arredi:
            if a.tipo in ("giardino", "recinto", "piazzetta", "lampione"):
                m = ris.modelli[a.modello]
                # per un lampione conta solo il palo (strato 0): il braccio e la
                # lanterna stanno in alto, SOPRA la strada
                celle = m.celle[:, :1, :] if a.tipo == "lampione" else m.celle
                pieno = (celle >= 0).any(axis=1)                   # (x, z)
                fp = carreggiata[a.z:a.z + a.profondita, a.x:a.x + a.larghezza].T
                self.assertFalse((pieno & fp).any(), f"{a.tipo} sopra la strada")
            else:
                self.assertFalse(carreggiata[a.z:a.z + a.profondita,
                                             a.x:a.x + a.larghezza].any(), a.tipo)

    def test_il_lampione_ha_il_braccio_con_la_lanterna_appesa(self):
        for verso in ("+x", "-x", "+z", "-z"):
            d = A.lampione_a_braccio(verso)
            nomi = {n for (n, _) in d.blocchi.values()}
            self.assertIn("lantern", nomi)
            self.assertIn("dark_oak_fence", nomi)
            self.assertNotIn("dark_oak_log", nomi, "il lampione e' di staccionata, non di tronco")
            lanterna = [k for k, (n, p) in d.blocchi.items() if n == "lantern"][0]
            braccio = [k for k, (n, p) in d.blocchi.items()
                       if n == "dark_oak_fence" and k[1] == 4 and k != (lanterna[0], 4, lanterna[2])]
            self.assertEqual(lanterna[1], 3, "la lanterna pende sotto il braccio")
            self.assertTrue(any(k[1] == 4 for k in d.blocchi if k[0] == lanterna[0]
                                and k[2] == lanterna[2]), "lanterna senza braccio sopra")
            self.assertEqual(sum(1 for k in d.blocchi if k[1] == 0), 1, "un solo palo a terra")
        self.assertEqual(A.lampione_a_braccio("+x").dim, (2, 5, 1))
        self.assertEqual(A.lampione_a_braccio("-z").dim, (1, 5, 2))

    def test_i_lampioni_stanno_lungo_la_via_e_distanziati(self):
        ris, _ = self.piano(con_strada=True)
        pali = [(a.x, a.z) for a in ris.arredi if a.tipo == "lampione"]
        self.assertGreater(len(pali), 0)
        for i, (x, z) in enumerate(pali):
            self.assertLessEqual(abs(z - 49), 5, "lontano dalla via")
            for (x2, z2) in pali[i + 1:]:
                self.assertGreaterEqual((x - x2) ** 2 + (z - z2) ** 2, 81)

    def test_la_strada_e_illuminata_per_tutta_la_lunghezza(self):
        """Anche fuori dal paese: da un capo all'altro non ci sono tratti
        scuri piu' lunghi del passo di campagna."""
        ris, _ = self.piano(con_strada=True)
        xs = sorted(a.x for a in ris.arredi if a.tipo == "lampione" and 46 <= a.z <= 52)
        self.assertGreaterEqual(len(xs), 6)
        self.assertLessEqual(xs[0], 10 + A.PASSO_LAMPIONI_CAMPAGNA + 2)
        self.assertGreaterEqual(xs[-1], 109 - A.PASSO_LAMPIONI_CAMPAGNA - 2)
        for a, b in zip(xs, xs[1:]):
            self.assertLessEqual(b - a, 2 * A.PASSO_LAMPIONI_CAMPAGNA, "tratto al buio")

    def test_densita_zero_non_arreda_niente(self):
        ris, _ = self.piano(densita=0.0)
        self.assertEqual(ris.arredi, [])
        self.assertEqual(ris.banchi, [])

    def test_e_deterministica(self):
        a, _ = self.piano()
        b, _ = self.piano()
        self.assertEqual([(x.tipo, x.x, x.z) for x in a.arredi],
                         [(x.tipo, x.x, x.z) for x in b.arredi])

    def test_la_zona_da_evitare_e_rispettata(self):
        s = self.scenario()
        evita = np.zeros(s["h"].shape, bool)
        evita[15:60, 15:60] = True                      # tutto il paese
        ris = A.pianifica(**s, seed=3, evita=evita)
        for a in ris.arredi:
            self.assertFalse(evita[a.z:a.z + a.profondita, a.x:a.x + a.larghezza].any(),
                             f"{a.tipo} nella zona vietata")

    def test_le_bestie_dei_recinti_hanno_un_animale_vero(self):
        for seme in range(12):
            s = self.scenario()
            ris = A.pianifica(**s, seed=seme)
            for an in ris.animali:
                self.assertIn(an.specie, ("mucca", "pecora", "gallina", "maiale"))
                self.assertEqual(an.y, 70.0)


@unittest.skipUnless(DISPONIBILE, "amulet/PyMCTranslate non disponibili")
class TestIndiceEMaschera(unittest.TestCase):

    def test_un_arredo_a_cavallo_di_due_chunk_e_registrato_in_tutti_e_due(self):
        a = A.Arredo("giardino", 12, 3, 8, 6, 70, 0)
        idx = A.indice_per_chunk([a])
        self.assertEqual(set(idx), {(0, 0), (1, 0)})

    def test_la_maschera_copre_l_ingombro(self):
        a = A.Arredo("pozzo", 5, 7, 3, 3, 70, 0)
        m = A.maschera([a], (30, 30))
        self.assertEqual(int(m.sum()), 9)
        self.assertTrue(m[7:10, 5:8].all())
        self.assertEqual(int(A.maschera([a], (30, 30), margine=1).sum()), 25)


if __name__ == "__main__":
    unittest.main()
