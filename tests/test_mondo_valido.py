"""Controlli su un mondo scritto, dal punto di vista di Minecraft.

Questi test nascono da un difetto vero, visto solo aprendo il gioco: il
terreno era perfetto e in sovrimpressione scorreva all'infinito

    Caricamento del chunk in [-11, -6] non riuscito

Nel registro: `Failed to parse chunk [x, z] position info`,
`ArrayIndexOutOfBoundsException: Index 0 out of bounds for length 0`, 1193
volte in una partita. La causa non era nei file region - quelli erano stati
riletti e confrontati piu' volte - ma nella cartella `entities/`, che amulet
scrive con un chunk-entita' vuoto e senza `Position` per ogni chunk.

Morale, e motivo di questo file: rileggere quello che si e' scritto non
basta, se si rilegge solo la parte a cui si sta pensando.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from amulet_nbt import load as nbt_load
    from genworld.livello_dat import ImpostazioniMondo, verifica
    from genworld.mondo import ScrittoreMondo
    AMULET = True
except ImportError:
    AMULET = False


def chunk_grezzo(percorso_mca: str, cx: int, cz: int) -> bytes | None:
    """Legge un chunk da un file region senza passare da amulet.

    Di proposito: amulet e' anche quello che li ha scritti, e un controllo
    fatto con lo stesso strumento che ha prodotto il file non e' un
    controllo.
    """
    with open(percorso_mca, "rb") as f:
        testa = f.read(4096)
        i = ((cx & 31) + (cz & 31) * 32) * 4
        off = int.from_bytes(testa[i:i + 3], "big")
        if off == 0:
            return None
        f.seek(off * 4096)
        lung = int.from_bytes(f.read(4), "big")
        f.read(1)                       # tipo di compressione
        return zlib.decompress(f.read(lung - 1))


@unittest.skipUnless(AMULET, "amulet non installato in questo interprete")
class TestMondoValido(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.percorso = os.path.join(self.dir.name, "w")
        imp = ImpostazioniMondo(nome="Prova", modalita=1)
        with ScrittoreMondo(self.percorso, imp, lato_blocchi=64, crea=True) as m:
            pietra = m.blocco("stone")
            prato = m.bioma("plains")
            for cx, cz in m.chunk_coords():
                b = np.full((16, 100, 16), m.id_aria, np.uint32)
                b[:, :70, :] = pietra
                m.scrivi_chunk(cx, cz, b, biomi=np.full((16, 16), prato, np.uint32))

    def tearDown(self):
        self.dir.cleanup()

    def test_nessuna_cartella_entities_rotta(self):
        """Il difetto di partenza. Se un giorno la cartella tornera', dovra'
        contenere chunk validi - e il test sotto lo pretende."""
        cartella = os.path.join(self.percorso, "entities")
        if not os.path.isdir(cartella):
            return
        for nome in os.listdir(cartella):
            if not nome.endswith(".mca"):
                continue
            for cz in range(-2, 2):
                for cx in range(-2, 2):
                    grezzo = chunk_grezzo(os.path.join(cartella, nome), cx, cz)
                    if grezzo is None:
                        continue
                    c = nbt_load(grezzo).compound
                    self.assertIn("Position", c,
                                  "chunk-entita' senza Position: e' esattamente "
                                  "l'ArrayIndexOutOfBounds che vede Minecraft")
                    self.assertEqual(len(c["Position"]), 2)

    def test_ogni_chunk_ha_quello_che_minecraft_legge(self):
        """Minecraft legge xPos/zPos/yPos, Status e sections. Un chunk senza
        Status non viene caricato, uno senza yPos nemmeno."""
        mca = os.path.join(self.percorso, "region", "r.-1.-1.mca")
        trovati = 0
        for cz in range(-2, 0):
            for cx in range(-2, 0):
                grezzo = chunk_grezzo(mca, cx, cz)
                self.assertIsNotNone(grezzo, f"chunk {cx},{cz} assente")
                c = nbt_load(grezzo).compound
                for chiave in ("DataVersion", "xPos", "zPos", "yPos",
                               "Status", "sections", "Heightmaps"):
                    self.assertIn(chiave, c, f"chunk {cx},{cz} senza {chiave}")
                self.assertEqual(int(c["xPos"]), cx)
                self.assertEqual(int(c["zPos"]), cz)
                self.assertEqual(str(c["Status"]), "minecraft:full")
                self.assertGreater(int(c["DataVersion"]), 0)
                trovati += 1
        self.assertEqual(trovati, 4)

    def test_il_level_dat_non_fa_lamentare_il_gioco(self):
        """`key missing: DragonFight` compariva nel registro di ogni
        apertura: Minecraft lo cerca anche se l'End non l'hai mai visto."""
        problemi = verifica(os.path.join(self.percorso, "level.dat"))
        self.assertEqual(problemi, [])
        d = nbt_load(os.path.join(self.percorso, "level.dat")).compound["Data"]
        self.assertIn("DragonFight", d)
        self.assertIn("Gateways", d["DragonFight"])

    def test_niente_session_lock_appeso(self):
        self.assertFalse(os.path.exists(os.path.join(self.percorso, "session.lock")))


if __name__ == "__main__":
    unittest.main()
