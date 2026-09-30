"""Test dell'indice per-chunk degli edifici.

Nato dal bug delle "case smezzate": un modello da template piu' grande del
lotto (l'ultimo ripiego di `template.scegli()`) puo' sporgere oltre i chunk
gia' registrati da `indice_per_chunk`, che al momento della pianificazione
non sa ancora quale modello verra' scelto. Vedi il commento su
`estendi_indice_per_modello`.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld.insediamenti import estendi_indice_per_modello, indice_per_chunk  # noqa: E402
from genworld import edifici as E  # noqa: E402


class TestEstendiIndice(unittest.TestCase):

    def test_un_modello_dentro_al_lotto_non_serve_estensione(self):
        """Se il modello sta nei chunk gia' registrati, l'indice non cambia:
        aggiungere solo dove serve evita richiamate doppie a costruisci()."""
        indice = {(0, 0): [5]}
        estendi_indice_per_modello(indice, 5, px=2, pz=2, ix=6, iz=6)
        self.assertEqual(indice, {(0, 0): [5]})

    def test_un_modello_che_sporge_in_un_altro_chunk_viene_aggiunto_li(self):
        """Il caso del bug: il lotto e' vicino al bordo del chunk (0,0) e il
        modello scelto sporge dentro il chunk (1,0). Senza l'estensione
        quella meta' della casa non verrebbe mai disegnata."""
        indice = {(0, 0): [3]}
        # angolo a x=12, largo 10: arriva a x=22, oltre il chunk 0 (0..15)
        estendi_indice_per_modello(indice, 3, px=12, pz=4, ix=10, iz=6)
        self.assertIn((1, 0), indice)
        self.assertIn(3, indice[(1, 0)])
        self.assertIn(3, indice[(0, 0)])

    def test_non_duplica_lo_stesso_edificio_nello_stesso_chunk(self):
        indice: dict = {}
        estendi_indice_per_modello(indice, 7, px=0, pz=0, ix=20, iz=20)
        estendi_indice_per_modello(indice, 7, px=0, pz=0, ix=20, iz=20)
        for chunk_ids in indice.values():
            self.assertEqual(chunk_ids.count(7), 1)

    def test_copre_tutti_i_chunk_toccati_da_un_modello_molto_grande(self):
        """Un modello che spazia su 3x2 chunk deve comparire in tutti e sei,
        non solo al primo e all'ultimo angolo."""
        indice: dict = {}
        estendi_indice_per_modello(indice, 0, px=0, pz=0, ix=40, iz=20)
        attesi = {(cx, cz) for cx in range(3) for cz in range(2)}
        self.assertEqual(set(indice.keys()), attesi)


class TestIndicePerChunkVsEstensione(unittest.TestCase):
    """L'indice di partenza (lotto) e quello esteso (modello reale) devono
    combinarsi senza buchi anche nel caso reale end-to-end."""

    def edificio(self, x, z, larghezza, profondita, gronda=1) -> E.Edificio:
        return E.Edificio(x=x, z=z, larghezza=larghezza, profondita=profondita,
                          base=70, piani=1, altezza_piano=3, gronda=gronda)

    def test_modello_oversize_richiede_l_estensione(self):
        """Riproduce la geometria del bug: un lotto piccolo (6x6), ben dentro
        al chunk (0, 0) anche col margine di `ingombro`, e un modello scelto
        per ripiego molto piu' grande (40x6) - il modello centrato sul lotto
        sporge sia nel chunk (-1, 0) sia nel chunk (1, 0), che il solo lotto
        non registra mai."""
        ed = self.edificio(x=3, z=3, larghezza=6, profondita=6)
        edifici = [ed]
        indice = indice_per_chunk(edifici)
        self.assertEqual(set(indice.keys()), {(0, 0)})

        ix, iz = 40, 6
        px = ed.x + (ed.larghezza - ix) // 2
        pz = ed.z + (ed.profondita - iz) // 2
        estendi_indice_per_modello(indice, 0, px, pz, ix, iz)
        for chunk in ((-1, 0), (0, 0), (1, 0)):
            self.assertIn(chunk, indice, f"manca {chunk}")
            self.assertIn(0, indice[chunk])


if __name__ == "__main__":
    unittest.main()
