"""Spike di fase 0: scrivere un mondo Minecraft vero e verificarlo.

Genera un mondo 256x256 con una heightmap deliberatamente riconoscibile
(una collina centrale, un'ondulazione regolare, una valle fluviale diagonale
che si riempie d'acqua), lo salva in formato Anvil e ne produce l'anteprima
PNG rileggendo i file region.

    python esempi/spike_piatto.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld.anteprima import rendi
from genworld.livello_dat import ImpostazioniMondo, verifica
from genworld.mondo import ScrittoreMondo, colonne_da_altezze
from genworld.rumore import dettaglio, quantizza

LATO = 256
LIVELLO_MARE = 62
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# il percorso finale dipende da --liscio, vedi main()
MONDI = os.path.join(RADICE, "mondi")


def heightmap(lato: int, rumore: bool = True, seed: int = 7) -> np.ndarray:
    """Rilievo di prova: collina + ondulazione + valle fluviale diagonale.

    Con `rumore=False` si ottiene la forma liscia pura, che una volta
    quantizzata a blocchi mostra le curve di livello concentriche: e' il
    difetto che si vede subito aprendo il mondo in gioco.
    """
    c = lato / 2
    x = np.arange(lato) - c
    z = np.arange(lato) - c
    X, Z = np.meshgrid(x, z, indexing="ij")

    r = np.sqrt(X ** 2 + Z ** 2)
    h = np.full((lato, lato), 66.0)
    h += 58.0 * np.exp(-((r / 70.0) ** 2))                  # collina centrale
    h += 6.0 * np.sin(X / 18.0) * np.cos(Z / 22.0)          # ondulazione
    h -= 18.0 * np.exp(-(((X + Z) / 26.0) ** 2))            # valle diagonale

    if not rumore:
        return np.floor(h + 0.5).astype(np.int32)

    # dettaglio frattale per il carattere del versante, poi quantizzazione
    # ditherata per non far comparire le curve di livello a gradini
    h = h + dettaglio(h, ampiezza_base=0.8, ampiezza_pendenza=4.0, seed=seed)
    return quantizza(h, forza=1.0, seed=seed)


def main() -> None:
    t0 = time.time()
    rumore = "--liscio" not in sys.argv
    nome_mondo = "spike" if rumore else "spike_liscio"
    imp = ImpostazioniMondo(
        nome="GenWorld - Spike" + ("" if rumore else " (liscio)"),
        modalita=1,
        spawn=(0, 130, 0),
    )
    alt = heightmap(LATO, rumore=rumore)
    print(f"heightmap {LATO}x{LATO} ({'con rumore' if rumore else 'liscia'}): quote da {alt.min()} a {alt.max()}")

    percorso = os.path.join(MONDI, nome_mondo)
    os.makedirs(MONDI, exist_ok=True)
    with ScrittoreMondo(percorso, imp, lato_blocchi=LATO) as m:
        meta = LATO // 2
        for cx, cz in m.chunk_coords():
            ox, oz = m.origine_chunk(cx, cz)
            fetta = alt[ox + meta: ox + meta + 16, oz + meta: oz + meta + 16]
            blocchi = colonne_da_altezze(fetta, m, livello_mare=LIVELLO_MARE)
            m.scrivi_chunk(cx, cz, blocchi, bioma="plains")
        scritti = m.chunk_scritti

    t1 = time.time()
    print(f"scritti {scritti} chunk in {t1 - t0:.1f}s -> {percorso}")

    problemi = verifica(os.path.join(percorso, "level.dat"))
    print("level.dat:", "valido" if not problemi else f"PROBLEMI {problemi}")

    png = os.path.join(MONDI, f"{nome_mondo}_anteprima.png")
    st = rendi(percorso, png, ombreggiatura=1.0)
    print(f"anteprima {st['larghezza']}x{st['altezza']} da {st['chunk']} chunk "
          f"(quote {st['quota_min']:.0f}-{st['quota_max']:.0f}) -> {png}")
    if st["blocchi_ignoti"]:
        print("  blocchi senza colore:", st["blocchi_ignoti"])
    print(f"totale {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
