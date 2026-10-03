"""Test del caricamento immagine: una mappa non quadrata non va deformata.

Nato da una segnalazione dell'utente: una mappa rettangolare, caricata,
usciva "modificata e deformata quadrata". Causa: `carica()` ridimensionava
sempre a `(lato, lato)`, stirando a incastro qualunque proporzione avesse
l'originale. Vedi anche `TestMappaNonQuadrata` in `test_motore.py` per la
verifica end-to-end (centratura nel mondo quadrato, bordo a oceano).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import mappa as M  # noqa: E402


def salva_rgb(rgb: np.ndarray, percorso: str) -> str:
    Image.fromarray((rgb * 255).astype(np.uint8)).save(percorso)
    return percorso


class TestCaricaProporzioni(unittest.TestCase):

    def immagine(self, w, h):
        im = np.random.default_rng(0).integers(0, 255, (h, w, 3), np.uint8)
        d = tempfile.mkdtemp()
        return salva_rgb(im.astype(np.float32) / 255.0, os.path.join(d, "m.png"))

    def test_una_mappa_quadrata_resta_lato_x_lato_come_prima(self):
        """Comportamento invariato per il caso comune: nessuna regressione."""
        p = self.immagine(300, 300)
        rgb = M.carica(p, lato=256)
        self.assertEqual(rgb.shape[:2], (256, 256))

    def test_una_mappa_rettangolare_non_diventa_quadrata(self):
        """Il caso segnalato: 400x200 (2:1) non deve uscire 256x256."""
        p = self.immagine(400, 200)
        rgb = M.carica(p, lato=256)
        self.assertNotEqual(rgb.shape[:2], (256, 256))

    def test_le_proporzioni_originali_sono_mantenute(self):
        """Il lato piu' lungo diventa `lato`, l'altro si accorcia in
        proporzione - non un ridimensionamento indipendente per asse, che e'
        esattamente cio' che deforma l'immagine."""
        p = self.immagine(400, 200)          # rapporto 2:1
        rgb = M.carica(p, lato=256)
        h, w = rgb.shape[:2]
        self.assertEqual(w, 256, "il lato piu' lungo (larghezza) deve diventare lato")
        self.assertAlmostEqual(w / h, 2.0, delta=0.02)

    def test_stessa_prova_con_laltro_orientamento(self):
        p = self.immagine(200, 400)          # rapporto 1:2, verticale stavolta
        rgb = M.carica(p, lato=256)
        h, w = rgb.shape[:2]
        self.assertEqual(h, 256)
        self.assertAlmostEqual(h / w, 2.0, delta=0.02)

    def test_senza_lato_nessun_ridimensionamento(self):
        p = self.immagine(400, 200)
        rgb = M.carica(p)
        self.assertEqual(rgb.shape[:2], (200, 400))
