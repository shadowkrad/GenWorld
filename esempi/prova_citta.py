"""Disegna la pianta di una citta' per guardarla.

Una pianta urbana si giudica a occhio: si vede subito se gli isolati sono
troppo piccoli, se le case guardano la strada, se la cinta gira dove deve.

    python esempi/prova_citta.py [indice_del_sito]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genworld import citta as C  # noqa: E402
from genworld import edifici as E  # noqa: E402
from genworld import insediamenti as I  # noqa: E402
from genworld.motore import Opzioni, analizza  # noqa: E402

RADICE = Path(__file__).resolve().parents[1]

COLORI = {
    "fondo": (34, 40, 34), "area": (86, 104, 74), "acqua": (44, 78, 130),
    C.VICOLO: (150, 138, 116), C.SECONDARIA: (186, 172, 146),
    C.ASSE: (222, 210, 182),
    "muro": (120, 120, 126), "porta": (210, 120, 60),
    "casa": (150, 86, 60), "facciata": (240, 200, 120),
    "piazza": (240, 240, 210),
}


def disegna(cls, h, area, rango, edifici, muro, piazza, fetta) -> Image.Image:
    sz, sx = fetta
    forma = (sz.stop - sz.start, sx.stop - sx.start)
    img = np.zeros((*forma, 3), np.uint8)
    img[:] = COLORI["fondo"]
    from genworld.mappa import ACQUA
    img[np.isin(cls[sz, sx], ACQUA)] = COLORI["acqua"]
    img[area[sz, sx]] = COLORI["area"]
    for r in (C.VICOLO, C.SECONDARIA, C.ASSE):
        img[rango[sz, sx] == r] = COLORI[r]
    m = muro[sz, sx]
    img[m == 1] = COLORI["muro"]
    img[m == 2] = COLORI["porta"]

    for ed in edifici:
        z0, z1 = ed.z - sz.start, ed.z1 - sz.start
        x0, x1 = ed.x - sx.start, ed.x1 - sx.start
        if z1 <= 0 or x1 <= 0 or z0 >= forma[0] or x0 >= forma[1]:
            continue
        img[max(0, z0):max(0, z1), max(0, x0):max(0, x1)] = COLORI["casa"]
        # la facciata: il lato dov'e' la porta, per vedere se guarda la strada
        if ed.porta == E.NORD and 0 <= z0 < forma[0]:
            img[z0, max(0, x0):max(0, x1)] = COLORI["facciata"]
        elif ed.porta == E.SUD and 0 <= z1 - 1 < forma[0]:
            img[z1 - 1, max(0, x0):max(0, x1)] = COLORI["facciata"]
        elif ed.porta == E.OVEST and 0 <= x0 < forma[1]:
            img[max(0, z0):max(0, z1), x0] = COLORI["facciata"]
        elif ed.porta == E.EST and 0 <= x1 - 1 < forma[1]:
            img[max(0, z0):max(0, z1), x1 - 1] = COLORI["facciata"]

    pz, px = piazza[0] - sz.start, piazza[1] - sx.start
    if 0 <= pz < forma[0] and 0 <= px < forma[1]:
        img[max(0, pz - 1):pz + 2, max(0, px - 1):px + 2] = COLORI["piazza"]
    return Image.fromarray(img, "RGB")


def main() -> None:
    op = Opzioni(immagine=str(RADICE / "input" / "mappa_arda.png"),
                 uscita=str(RADICE / "mondi" / "arda"), nome="Arda")
    a = analizza(op)
    siti = I.scegli_siti(a.cls, a.h, livello_mare=62, seed=31)
    ed, h, vie, muro, citta, banchi = C.pianifica(a.cls, a.h, a.livello, siti, 62, seed=31)
    print(C.statistiche(ed, vie, muro, citta, banchi))

    immagini = []
    for k in range(min(3, len(citta))):
        c = citta[k]
        area = C.contorno(a.cls, a.h, c.z, c.x, c.raggio, 62, seed=31 * 31 + k)
        m = int(c.raggio * 1.25)
        fetta = (slice(max(0, c.z - m), c.z + m), slice(max(0, c.x - m), c.x + m))
        dentro = [e for e in ed
                  if fetta[0].start - 12 < e.z < fetta[0].stop + 12
                  and fetta[1].start - 12 < e.x < fetta[1].stop + 12]
        print(f"  sito {k}: r={c.raggio}, {len(dentro)} edifici, murata={c.murata}")
        immagini.append(disegna(a.cls, a.h, area, vie, dentro, muro, c.piazza, fetta))

    alta = max(i.height for i in immagini)
    larga = sum(i.width for i in immagini) + 6 * (len(immagini) - 1)
    fuori = Image.new("RGB", (larga, alta), (20, 20, 24))
    x = 0
    for im in immagini:
        fuori.paste(im, (x, 0))
        x += im.width + 6
    k = max(1, 900 // max(fuori.width, 1))
    fuori = fuori.resize((fuori.width * k, fuori.height * k), Image.NEAREST)
    dest = RADICE / "mondi" / "citta.png"
    fuori.save(dest)
    print("scritto", dest)


if __name__ == "__main__":
    main()
