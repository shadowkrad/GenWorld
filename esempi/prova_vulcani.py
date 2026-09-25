"""Confronto visivo delle rugosita' del cono vulcanico.

Genera un pannello per ogni valore di rugosita' con ombreggiatura e colate in
arancione, ritagliato su un solo vulcano perche' il difetto (le colate a
raggiera) si vede solo da vicino.

    python esempi/prova_vulcani.py 0 0.10 0.16 0.24
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genworld import vulcani as V  # noqa: E402

RADICE = Path(__file__).resolve().parents[1]


def ombreggia(h: np.ndarray) -> np.ndarray:
    """Hillshade classico: sole a 315 gradi, 45 di altezza."""
    gz, gx = np.gradient(h.astype(np.float32))
    az, alt = np.radians(315.0), np.radians(45.0)
    pendenza = np.arctan(np.hypot(gz, gx) * 2.0)
    aspetto = np.arctan2(-gz, gx)
    v = (np.sin(alt) * np.cos(pendenza) +
         np.cos(alt) * np.sin(pendenza) * np.cos(az - aspetto))
    return np.clip(v, 0, 1)


def pannello(h: np.ndarray, colata: np.ndarray, lava: np.ndarray,
             fetta: tuple[slice, slice]) -> Image.Image:
    om = ombreggia(h)[fetta]
    hh = h[fetta]
    base = 0.25 + 0.75 * (hh - hh.min()) / max(float(np.ptp(hh)), 1.0)
    g = np.clip(om * 0.75 + base * 0.45, 0, 1)
    rgb = np.dstack([g, g, g])
    rgb[colata[fetta]] = (1.0, 0.45, 0.05)
    rgb[lava[fetta]] = (1.0, 0.72, 0.15)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def main() -> None:
    valori = [float(a) for a in sys.argv[1:]] or [0.0, 0.16, 0.28]
    h0 = np.load(RADICE / "mondi" / "arda_h.npy")
    cls0 = np.load(RADICE / "mondi" / "arda_cls.npy")

    vs = V.scegli_siti(cls0, np.round(h0).astype(np.int32), quanti=3, seed=7)
    print(f"{len(vs)} vulcani: " +
          ", ".join(f"({v.z},{v.x}) r={v.raggio} h={v.altezza:.0f}" for v in vs))
    v0 = vs[0]
    m = int(v0.raggio * 2.6)
    fetta = (slice(max(0, v0.z - m), v0.z + m), slice(max(0, v0.x - m), v0.x + m))

    immagini = []
    for r in valori:
        c, h, lava, _ = V.modella(cls0, np.round(h0).astype(np.int32), vs, rugosita=r)
        col = V.colate(h, c, vs)
        print(f"rugosita {r:.2f}: colate={int(col.sum())} "
              f"quota_max={int(h.max())}")
        immagini.append(pannello(h, col, lava, fetta))

    larghezza = sum(i.width for i in immagini) + 4 * (len(immagini) - 1)
    fuori = Image.new("RGB", (larghezza, immagini[0].height), (30, 30, 34))
    x = 0
    for im in immagini:
        fuori.paste(im, (x, 0))
        x += im.width + 4
    fuori = fuori.resize((fuori.width * 2, fuori.height * 2), Image.NEAREST)
    dest = RADICE / "mondi" / "vulcani.png"
    fuori.save(dest)
    print("scritto", dest, "|", " | ".join(f"rugosita {v:.2f}" for v in valori))


if __name__ == "__main__":
    main()
