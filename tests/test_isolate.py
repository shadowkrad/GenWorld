"""Test delle case isolate: i template troppo alti per un villaggio, sparsi
fuori dai paesi, ognuno al massimo una volta per mappa."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import isolate as IS  # noqa: E402
from genworld import mappa as M  # noqa: E402
from genworld import template as TM  # noqa: E402


def modello(dx, dy, dz, stili=frozenset()):
    return TM.Modello(nome=f"{dx}x{dy}x{dz}", celle=np.zeros((dx, dy, dz), np.int32),
                      tavolozza=[("stone", {})], stili=stili)


def mondo(lato=400):
    h = np.full((lato, lato), 70, np.int32)
    cls = np.full((lato, lato), M.PIANURA, np.uint8)
    evita = np.zeros((lato, lato), bool)
    return h, cls, evita


class TestCandidati(unittest.TestCase):

    def test_solo_i_modelli_troppo_alti_per_un_lotto(self):
        ms = [modello(9, 10, 9), modello(20, 30, 20), modello(15, 25, 15)]
        self.assertEqual(IS.candidati(ms, 400), [1, 2])

    def test_gli_esclusi_a_mano_non_sono_case(self):
        ms = [modello(20, 30, 20, frozenset({"_escluso"})), modello(20, 30, 20)]
        self.assertEqual(IS.candidati(ms, 400), [1])

    def test_un_modello_enorme_su_una_mappa_piccola_non_ci_sta(self):
        ms = [modello(109, 30, 65), modello(20, 30, 20)]
        self.assertEqual(IS.candidati(ms, 256), [1])          # 109 > 256 // 5


class TestPianificazione(unittest.TestCase):

    def piano(self, ms, seed=1, **kw):
        h, cls, evita = mondo(kw.pop("lato", 400))
        h = kw.pop("h", h)
        cls = kw.pop("cls", cls)
        evita = kw.pop("evita", evita)
        return IS.pianifica(ms, h, cls, evita, 62, seed=seed, **kw), (h, cls, evita)

    def test_ogni_modello_al_massimo_una_volta(self):
        ms = [modello(20, 30, 20), modello(24, 28, 18), modello(18, 26, 26)]
        case, _ = self.piano(ms, densita=10.0)
        usati = [c.modello for c in case]
        self.assertEqual(len(usati), len(set(usati)))
        self.assertEqual(len(case), 3)

    def test_i_modelli_da_villaggio_non_diventano_case_isolate(self):
        ms = [modello(9, 10, 9), modello(20, 30, 20)]
        case, _ = self.piano(ms, densita=10.0)
        self.assertEqual({c.modello for c in case}, {1})

    def test_l_ingombro_e_quello_girato(self):
        ms = [modello(12, 30, 20)]
        case, _ = self.piano(ms, densita=10.0)
        for c in case:
            self.assertEqual((c.larghezza, c.profondita), ms[0].ingombro(c.quarti))

    def test_non_stanno_nella_zona_da_evitare_ne_a_meno_di_un_margine(self):
        ms = [modello(20, 30, 20), modello(22, 30, 22)]
        evita = np.zeros((400, 400), bool)
        evita[100:300, 100:300] = True
        case, _ = self.piano(ms, evita=evita, densita=10.0, margine=8)
        self.assertGreater(len(case), 0)
        for c in case:
            fp = evita[max(0, c.z - 8):c.z + c.profondita + 8, max(0, c.x - 8):c.x + c.larghezza + 8]
            self.assertFalse(fp.any(), "troppo vicina alla zona vietata")

    def test_non_stanno_nell_acqua(self):
        ms = [modello(20, 30, 20)]
        cls = np.full((400, 400), M.OCEANO, np.uint8)
        cls[150:250, 150:250] = M.PIANURA                    # un'isola
        case, _ = self.piano(ms, cls=cls, densita=10.0)
        for c in case:
            fp = cls[c.z:c.z + c.profondita, c.x:c.x + c.larghezza]
            self.assertTrue((fp == M.PIANURA).all())

    def test_non_stanno_sul_dirupo(self):
        ms = [modello(20, 30, 20)]
        h = (70 + (np.arange(400)[None, :] * 3)).repeat(400, axis=0).astype(np.int32)   # 3 blocchi per cella
        case, _ = self.piano(ms, h=h, densita=10.0)
        self.assertEqual(case, [])

    def test_non_stanno_sotto_il_livello_del_mare(self):
        ms = [modello(20, 30, 20)]
        h = np.full((400, 400), 60, np.int32)
        case, _ = self.piano(ms, h=h, densita=10.0)
        self.assertEqual(case, [])

    def test_lo_stile_del_modello_e_rispettato(self):
        ms = [modello(20, 30, 20, frozenset({"deserto"}))]
        case, _ = self.piano(ms, densita=10.0)         # tutto pianura = stile prato
        self.assertEqual(case, [])

    def test_stanno_lontane_fra_loro(self):
        ms = [modello(20, 30, 20), modello(22, 30, 22), modello(18, 30, 18), modello(24, 30, 24)]
        case, _ = self.piano(ms, densita=10.0, distanza_min=80)
        for i, a in enumerate(case):
            for b in case[i + 1:]:
                ca = (a.x + a.larghezza / 2, a.z + a.profondita / 2)
                cb = (b.x + b.larghezza / 2, b.z + b.profondita / 2)
                self.assertGreaterEqual(np.hypot(ca[0] - cb[0], ca[1] - cb[1]), 79)

    def test_densita_zero_non_ne_mette(self):
        case, _ = self.piano([modello(20, 30, 20)], densita=0.0)
        self.assertEqual(case, [])

    def test_e_deterministica(self):
        ms = [modello(20, 30, 20), modello(22, 30, 22)]
        a, _ = self.piano(ms, seed=5, densita=10.0)
        b, _ = self.piano(ms, seed=5, densita=10.0)
        self.assertEqual(a, b)

    def test_un_seme_diverso_da_un_paese_diverso(self):
        ms = [modello(20, 30, 20)]
        a, _ = self.piano(ms, seed=1, densita=10.0)
        b, _ = self.piano(ms, seed=2, densita=10.0)
        self.assertNotEqual([(c.x, c.z) for c in a], [(c.x, c.z) for c in b])

    def test_il_numero_segue_la_dimensione_della_mappa(self):
        ms = [modello(20 + i, 30, 20) for i in range(30)]
        piccola, _ = self.piano(ms, lato=400)
        grande, _ = self.piano(ms, lato=1000)
        self.assertGreater(len(grande), len(piccola))


class TestIndiceEMaschera(unittest.TestCase):

    def test_un_maniero_a_cavallo_di_piu_chunk_e_registrato_in_tutti(self):
        c = IS.CasaIsolata(x=10, z=10, larghezza=40, profondita=20, base=70, modello=0, quarti=0)
        idx = IS.indice_per_chunk([c])
        self.assertEqual(set(idx), {(cx, cz) for cx in range(0, 4) for cz in range(0, 2)})

    def test_la_maschera_ha_l_ingombro_e_il_margine(self):
        c = IS.CasaIsolata(x=10, z=10, larghezza=20, profondita=10, base=70, modello=0, quarti=0)
        self.assertEqual(int(IS.maschera([c], (100, 100)).sum()), 200)
        self.assertEqual(int(IS.maschera([c], (100, 100), margine=2).sum()), 24 * 14)


if __name__ == "__main__":
    unittest.main()
