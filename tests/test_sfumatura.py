"""Sfumatura del bordo: blending_data sui chunk piu' esterni."""
import glob
import os
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import sfumatura as SF
from genworld.livello_dat import ImpostazioniMondo
from genworld.mondo import ScrittoreMondo


def mondo(percorso, lato=96):
    with ScrittoreMondo(percorso, ImpostazioniMondo(nome="P", modalita=1), lato_blocchi=lato,
                        crea=True) as m:
        pietra = m.blocco("stone")
        for cx, cz in m.chunk_coords():
            out = np.full((16, 140, 16), m.id_aria, np.uint32)
            out[:, :40, :] = pietra
            m.scrivi_chunk(cx, cz, out, biomi=np.full((16, 16), m.bioma("plains"), np.uint32))


def chunk_grezzi(percorso):
    import amulet_nbt as nbt
    fuori = {}
    for f in (glob.glob(os.path.join(percorso, "dimensions", "minecraft", "overworld",
                                     "region", "*.mca"))
              + glob.glob(os.path.join(percorso, "region", "*.mca"))):
        dati = open(f, "rb").read()
        for i in range(1024):
            off = int.from_bytes(dati[i * 4:i * 4 + 3], "big")
            if not off:
                continue
            pos = off * 4096
            ln = struct.unpack(">I", dati[pos:pos + 4])[0]
            tag = nbt.load(zlib.decompress(dati[pos + 5:pos + 4 + ln]),
                           compressed=False, little_endian=False)
            c = tag.compound
            fuori[(int(c["xPos"]), int(c["zPos"]))] = c
    return fuori


class TestSfumatura(unittest.TestCase):
    def test_i_chunk_di_bordo_prendono_il_tag_gli_interni_no(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w")
            mondo(p, lato=96)                       # 6 x 6 chunk
            st = SF.segna_bordo(p, anelli=1)
            chunk = chunk_grezzi(p)
            self.assertEqual(st["chunk"], len(chunk))
            xs = sorted({c[0] for c in chunk})
            zs = sorted({c[1] for c in chunk})
            esterni = {(x, z) for (x, z) in chunk
                       if x in (xs[0], xs[-1]) or z in (zs[0], zs[-1])}
            self.assertEqual(st["segnati"], len(esterni))
            for pos, c in chunk.items():
                self.assertEqual("blending_data" in c, pos in esterni, pos)
            bd = chunk[sorted(esterni)[0]]["blending_data"]
            self.assertEqual((int(bd["min_section"]), int(bd["max_section"])),
                             (SF.SEZIONE_MIN, SF.SEZIONE_MAX))

    def test_piu_anelli_segnano_piu_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w")
            mondo(p, lato=96)
            st = SF.segna_bordo(p, anelli=2)
            self.assertEqual(st["segnati"], 32)     # 6 x 6 con i 4 chunk centrali interni

    def test_il_resto_del_chunk_resta_intatto_e_amulet_lo_rilegge(self):
        import amulet
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w")
            mondo(p, lato=64)
            prima = chunk_grezzi(p)
            SF.segna_bordo(p, anelli=1)
            dopo = chunk_grezzi(p)
            self.assertEqual(set(prima), set(dopo))
            for pos in prima:
                self.assertEqual(int(prima[pos]["DataVersion"]), int(dopo[pos]["DataVersion"]))
                self.assertEqual(len(prima[pos]["sections"]), len(dopo[pos]["sections"]))
            lv = amulet.load_level(p)
            try:
                cx, cz = sorted(dopo)[0]
                ch = lv.get_chunk(cx, cz, "minecraft:overworld")
                self.assertEqual(ch.block_palette[ch.blocks.get_sub_chunk(-4)[0, 0, 0]].base_name,
                                 "stone")
            finally:
                lv.close()

    def test_una_seconda_passata_non_duplica_ne_rompe(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w")
            mondo(p, lato=64)
            SF.segna_bordo(p, anelli=1)
            SF.segna_bordo(p, anelli=1)
            chunk = chunk_grezzi(p)
            self.assertEqual(sum("blending_data" in c for c in chunk.values()), 12)
            for c in chunk.values():
                if "blending_data" in c:
                    self.assertEqual(len([k for k in c.keys() if k == "blending_data"]), 1)

    def test_senza_region_non_fa_nulla(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(SF.segna_bordo(d), {"chunk": 0, "segnati": 0, "file": 0})


if __name__ == "__main__":
    unittest.main()
