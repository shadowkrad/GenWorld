"""Test dell'altimetria: quanto e' alto un rilievo, e quanto e' ripida la costa.

Nasce da uno screenshot di un deserto pieno di guglie: cime alte come tre
montagne e larghe come una casa. La causa era che la quota dipendeva solo dal
colore - una macchia di cinque pixel di "montagna" diventava un picco di
centoventi blocchi su quindici celle - mentre in natura la quota di un
massiccio dipende dalla sua LARGHEZZA.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import mappa as M  # noqa: E402


def scacchiera(lato, raggio, sfondo=M.DESERTO, dentro=M.MONTAGNA):
    """Una macchia tonda di `dentro` in mezzo a `sfondo`."""
    zz, xx = np.ogrid[:lato, :lato]
    d = np.hypot(zz - lato / 2, xx - lato / 2)
    return np.where(d <= raggio, dentro, sfondo).astype(np.uint8)


class TestLarghezzaEQuota(unittest.TestCase):

    def quota_massima(self, raggio, lato=200):
        cls = scacchiera(lato, raggio)
        rug = np.zeros((lato, lato), np.float32)
        return float(M.altimetria(cls, rug).max())

    def test_un_rilievo_stretto_non_regge_la_sua_quota(self):
        stretto = self.quota_massima(3)
        largo = self.quota_massima(40)
        self.assertLess(stretto, largo - 15,
                        f"largo {largo:.0f}, stretto {stretto:.0f}: "
                        f"la quota non dipende dalla larghezza")

    def test_un_massiccio_largo_arriva_alla_quota_piena(self):
        q = Q = M.Quote()
        self.assertGreater(self.quota_massima(40), Q.base[M.MONTAGNA] - 6)

    def test_un_puntino_resta_una_collina(self):
        Q = M.Quote()
        self.assertLess(self.quota_massima(2), Q.quota_collina + 8)

    def test_la_guglia_non_esiste_piu(self):
        """La misura del difetto: quanto sporge una cima rispetto al terreno
        nel raggio di dodici celle."""
        from scipy.ndimage import uniform_filter
        lato = 200
        cls = scacchiera(lato, 4)
        h = M.altimetria(cls, np.zeros((lato, lato), np.float32))
        sporgenza = float((h - uniform_filter(h, 25)).max())
        self.assertLess(sporgenza, 20.0, f"sporge di {sporgenza:.0f} blocchi")


class TestCosta(unittest.TestCase):

    def scenario(self):
        lato = 120
        cls = np.full((lato, lato), M.MONTAGNA, np.uint8)
        cls[:, :30] = M.MARE
        return cls, np.zeros((lato, lato), np.float32)

    def test_la_costa_non_e_un_muro(self):
        cls, rug = self.scenario()
        h = M.altimetria(cls, rug)
        marino = np.isin(cls, M.MARINO)
        gz, gx = np.gradient(h)
        pend = np.hypot(gz, gx)
        # sulla battigia: due celle di terra a ridosso dell'acqua
        from scipy.ndimage import binary_dilation
        battigia = binary_dilation(marino, iterations=2) & ~marino
        # 6,5 misurato: il salto vero e' fra terra e fondale, e quello
        # resta - una falesia esiste. Senza il tetto si arrivava a 24.
        self.assertLess(float(pend[battigia].max()), 8.0)

    def test_senza_il_tetto_sarebbe_un_muro(self):
        cls, rug = self.scenario()
        q = M.Quote(); q.pendenza_costa = 0.0
        h = M.altimetria(cls, rug, q)
        from scipy.ndimage import binary_dilation
        marino = np.isin(cls, M.MARINO)
        battigia = binary_dilation(marino, iterations=2) & ~marino
        gz, gx = np.gradient(h)
        senza = float(np.hypot(gz, gx)[battigia].max())
        q2 = M.Quote()
        h2 = M.altimetria(cls, rug, q2)
        gz, gx = np.gradient(h2)
        con = float(np.hypot(gz, gx)[battigia].max())
        self.assertGreater(senza, con, "il tetto di costa non fa niente")

    def test_il_mare_resta_sotto_il_pelo(self):
        cls, rug = self.scenario()
        h = M.altimetria(cls, rug)
        marino = np.isin(cls, M.MARINO)
        self.assertLess(float(h[marino].max()), 62)


if __name__ == "__main__":
    unittest.main()
