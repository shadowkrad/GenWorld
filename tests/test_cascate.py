"""Cascate (#55): salti del fiume, colonna in caduta, pareti ai lati."""
import numpy as np

from genworld import cascate as CAS
from genworld import mappa as M

MARE = 62
Y0, ALT = 0, 120


class Finto:
    def __init__(self):
        self.ids = {("air", ()): 0}
        self.id_aria = 0

    def blocco(self, nome, **pr):
        k = (nome, tuple(sorted(pr.items())))
        if k not in self.ids:
            self.ids[k] = len(self.ids)
        return self.ids[k]

    def nome(self, i):
        return next(n for (n, _), v in self.ids.items() if v == i)


def fiume(h_monte=72, h_valle=64, lato=16):
    """Un fiume lungo x: a monte (x < 8) pelo h_monte, a valle pelo h_valle,
    larghezza 3 celle in z (6..8), terra attorno."""
    cls = np.full((lato, lato), M.PIANURA, np.uint8)
    liv = np.full((lato, lato), float(MARE), np.float32)
    cls[6:9, :] = M.FIUME
    liv[6:9, :8] = h_monte
    liv[6:9, 8:] = h_valle
    return cls, liv


def test_trova_il_salto():
    cls, liv = fiume()
    caduta, parete = CAS.trova(liv, cls, MARE)
    # la cella di valle accanto al salto
    assert (caduta[6:9, 8] == 72).all()
    assert caduta.sum() == 3 * 72
    # le pareti: le celle di terra accanto alla colonna (z=5 e z=9)
    assert (parete[5, 8] == 72) and (parete[9, 8] == 72)
    assert parete[6:9, :].max() == 0          # sull'acqua non c'e' parete


def test_nessun_salto_piccolo_o_in_salita():
    cls, liv = fiume(h_monte=65, h_valle=64)
    assert not CAS.trova(liv, cls, MARE)[0].any()
    cls, liv = fiume(h_monte=64, h_valle=72)         # il "salto" ha il verso opposto
    caduta, _ = CAS.trova(liv, cls, MARE)
    assert (caduta[6:9, 7] == 72).all() and not caduta[6:9, 8:].any()


def test_niente_cascata_nel_mare_o_lava():
    cls, liv = fiume()
    cls[:, :] = M.OCEANO
    assert not CAS.trova(liv, cls, MARE)[0].any()
    cls, liv = fiume()
    lava = np.ones(cls.shape, bool)
    assert not CAS.trova(liv, cls, MARE, lava)[0].any()


def _chunk(s):
    cls, liv = fiume()
    h = np.full((16, 16), 60, np.int32)
    h[cls == M.PIANURA] = 66                     # la riva poco sopra il pelo di valle
    out = np.zeros((16, ALT, 16), np.uint32)
    out[np.arange(ALT)[None, :, None] < h.T[:, None, :]] = s.blocco("stone")
    # acqua come la scrive blocchi_chunk: fino al pelo di ogni cella
    liv_i = np.maximum(liv.astype(np.int32), MARE).T
    ys = np.arange(ALT)[None, :, None]
    out[(out == 0) & (ys <= liv_i[:, None, :]) & (cls.T == M.FIUME)[:, None, :]] = s.blocco("water")
    return out, h, liv, cls


def test_posa_colonna_e_pareti():
    s = Finto()
    out, h, liv, cls = _chunk(s)
    caduta, parete = CAS.trova(liv, cls, MARE)
    # ATTENZIONE: fetta() gira gli assi in [x, z]; qui il test lavora in [x, z]
    # direttamente: le mappe si trasponi per comodita'.
    n = CAS.posa(out, s, h.T, liv.T, caduta.T, parete.T, 0, 0, Y0, MARE)
    assert n == 3
    cade = s.blocco("water", falling="true", flowing="true", level="0")
    # la colonna: x=8, z in 6..8, y da 65 a 72 e' acqua che cade
    for z in (6, 7, 8):
        assert (out[8, 65:73, z] == cade).all()
        assert out[8, 73, z] == 0                              # sopra il pelo di monte, aria
        assert out[8, 64, z] != cade                           # sotto, il fiume di valle
    # le pareti ai lati: pietra da sopra la riva (66) fino a 72, mai acqua
    for z in (5, 9):
        col = out[8, 66:73, z]
        assert (col != 0).all() and (col != cade).all()
    # nessuna acqua in caduta con aria sotto: ogni blocco che cade poggia su altra acqua
    for z in (6, 7, 8):
        for y in range(65, 73):
            assert out[8, y - 1, z] != 0


def test_idempotente_su_chunk_senza_cascate():
    s = Finto()
    out = np.zeros((16, ALT, 16), np.uint32)
    z = np.zeros((16, 16), np.int16)
    assert CAS.posa(out, s, np.full((16, 16), 60), np.full((16, 16), MARE), z, z, 0, 0, Y0) == 0
    assert not out.any()
