"""Test dei vulcani.

Le prove riguardano cose misurabili - il cono e' piu' alto del terreno, il
cratere e' un invaso, le colate scendono - non l'aspetto, che si giudica a
occhio sul confronto in mondi/vulcani.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genworld.mappa import CRATERE, MARE, PIANURA, VULCANO  # noqa: E402
from genworld import vulcani as V  # noqa: E402


def terreno(lato: int = 240, quota: int = 80):
    cls = np.full((lato, lato), PIANURA, np.uint8)
    cls[:, :30] = MARE
    h = np.full((lato, lato), quota, np.int32)
    return cls, h


def un_vulcano(z=120, x=140, raggio=40, altezza=60, quota_base=80):
    return V.Vulcano(z=z, x=x, raggio=raggio, altezza=altezza,
                     raggio_cratere=8, profondita_cratere=12,
                     quota_base=quota_base, seme=1234)


def test_il_cono_si_somma_al_terreno():
    """Un vulcano su una collina eredita la collina, non la spiana."""
    cls, h = terreno()
    h[100:150, 120:170] = 110  # collina piu' alta del cono in quel punto
    _, h2, _, _ = V.modella(cls, h, [un_vulcano()])
    assert h2.max() >= h.max()
    assert (h2 >= np.minimum(h, 110)).mean() > 0.99


def test_il_cratere_e_un_invaso():
    """Il fondo del cratere sta sotto l'orlo, e c'e' lava dentro."""
    v = un_vulcano()
    cls, h = terreno()
    c, h2, lava, q = V.modella(cls, h, [v])
    fondo = h2[v.z, v.x]
    orlo = h2[v.z, v.x + v.raggio_cratere + 1]
    assert fondo < orlo, f"fondo {fondo} non sotto l'orlo {orlo}"
    assert lava[v.z, v.x]
    assert q[v.z, v.x] > fondo


def test_le_classi_sono_assegnate():
    v = un_vulcano()
    cls, h = terreno()
    c, _, _, _ = V.modella(cls, h, [v])
    assert (c == VULCANO).sum() > 0
    assert (c == CRATERE).sum() > 0
    assert c[v.z, v.x] == CRATERE
    # il mare resta mare anche se un cono lo sfiorasse
    assert (c[:, :30] == MARE).all()


def test_la_rugosita_rompe_la_simmetria_radiale():
    """Il motivo della patch: un cono liscio ha il gradiente esattamente
    radiale e le colate scendono in linea retta. Misuriamo quanto la quota
    varia lungo un anello a raggio costante."""
    v = un_vulcano()
    cls, h = terreno()
    zz, xx = np.ogrid[:cls.shape[0], :cls.shape[1]]
    d = np.sqrt((zz - v.z) ** 2 + (xx - v.x) ** 2)
    # anello a raggio COSTANTE: una fascia larga misurerebbe la pendenza
    # radiale del cono invece della variazione lungo la circonferenza
    anello = np.abs(d - v.raggio * 0.5) < 0.5

    _, liscio, _, _ = V.modella(cls, h, [v], rugosita=0.0)
    _, solcato, _, _ = V.modella(cls, h, [v], rugosita=0.18)
    assert liscio[anello].std() < 1.0
    assert solcato[anello].std() > 3.0


def test_le_colate_scendono():
    v = un_vulcano()
    cls, h = terreno()
    c, h2, _, _ = V.modella(cls, h, [v])
    col = V.colate(h2, c, [v], per_vulcano=6)
    assert col.sum() > 20
    zz, xx = np.nonzero(col)
    d = np.hypot(zz - v.z, xx - v.x)
    # nessuna colata dentro il cratere, e nessuna piu' in alto del punto di
    # partenza sull'orlo
    assert d.min() >= v.raggio_cratere - 1
    assert h2[col].max() <= v.quota_orlo + 2


def test_le_colate_si_fermano_in_pianura():
    """Senza la soglia di pendenza l'inerzia tirava righe rette per mezza
    mappa: una colata non deve uscire di molto dal cono."""
    v = un_vulcano()
    cls, h = terreno(lato=400)
    c, h2, _, _ = V.modella(cls, h, [v])
    col = V.colate(h2, c, [v], per_vulcano=8)
    zz, xx = np.nonzero(col)
    d = np.hypot(zz - v.z, xx - v.x)
    assert d.max() < v.raggio * 1.6, f"colata arrivata a {d.max():.0f}"


def test_i_siti_stanno_su_terra_alta_e_separati():
    cls, h = terreno(lato=400, quota=40)
    h[:, 200:] = 120  # altopiano a est
    cls[:, :60] = MARE
    vs = V.scegli_siti(cls, h, quanti=3, separazione=90, seed=3)
    assert len(vs) == 3
    for v in vs:
        assert cls[v.z, v.x] != MARE
        assert v.x > 60
    for i, a in enumerate(vs):
        for b in vs[i + 1:]:
            assert np.hypot(a.z - b.z, a.x - b.x) >= 90


def test_senza_vulcani_non_cambia_nulla():
    cls, h = terreno()
    c, h2, lava, q = V.modella(cls, h, [])
    assert (c == cls).all()
    assert (h2 == h).all()
    assert not lava.any()
    assert not V.colate(h, cls, []).any()


def test_statistiche():
    v = un_vulcano()
    cls, h = terreno()
    c, h2, lava, _ = V.modella(cls, h, [v])
    col = V.colate(h2, c, [v])
    s = V.statistiche([v], c, lava, col)
    assert s["vulcani"] == 1
    assert s["cono"] > 0 and s["cratere"] > 0
    assert s["quota_massima"] == int(v.quota_base + v.altezza)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
