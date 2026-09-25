"""Genera un mondo Minecraft da una mappa disegnata (riga di comando).

La pipeline vera sta in `genworld/motore.py`: qui c'e' solo l'interfaccia a
riga di comando. La finestra (`genworld/gui.py`) chiama le stesse funzioni,
cosi' le due strade non possono divergere.

Una mappa grande non entra sempre in una sola esecuzione, quindi si puo'
lavorare a lotti:

    python esempi/genera_mappa.py --lato 832 --lotto 0 --chunk-per-lotto 700
    python esempi/genera_mappa.py --lato 832 --lotto 1 --chunk-per-lotto 700
    ...
    python esempi/genera_mappa.py --anteprima

Senza `--chunk-per-lotto` il mondo si genera tutto in una volta.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld.motore import (LIVELLO_MARE, SUPERFICIE, Opzioni, analizza,  # noqa: F401,E402
                             blocchi_chunk, fetta, genera, pianifica, scrivi)

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONDI = os.path.join(RADICE, "mondi")
PERCORSO = os.path.join(MONDI, "arda")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--immagine", default=os.path.join(RADICE, "input", "mappa_arda.png"))
    ap.add_argument("--uscita", default=PERCORSO, help="cartella del mondo")
    ap.add_argument("--nome", default="Arda")
    ap.add_argument("--lato", type=int, default=832)
    ap.add_argument("--ritaglio", type=float, default=0.025,
                    help="frazione di cornice da togliere per lato")
    ap.add_argument("--lotto", type=int, default=0)
    ap.add_argument("--chunk-per-lotto", type=int, default=0,
                    help="0 = tutto in una volta")
    ap.add_argument("--anteprima", action="store_true")
    ap.add_argument("--profilo", help="profilo JSON prodotto dal Calibratore Mappe: "
                    "rettangolo di interesse + campioni col contagocce")
    ap.add_argument("--vulcani", type=int, default=3, metavar="N",
                    help="quanti vulcani piazzare (0 = nessuno)")
    ap.add_argument("--strade", action="store_true", default=True,
                    help="traccia la rete stradale e i ponti (attivo)")
    ap.add_argument("--senza-strade", dest="strade", action="store_false")
    ap.add_argument("--villaggi", type=float, default=1.0, metavar="SCALA",
                    help="densita' degli insediamenti (0 = nessuno)")
    ap.add_argument("--fiumi", type=float, default=150.0, metavar="SOGLIA",
                    help="area drenata minima perche' una cella sia fiume "
                         "(0 = nessun fiume). Piu' bassa = reticolo piu' fitto")
    ap.add_argument("--alberi", type=float, default=1.0, metavar="SCALA",
                    help="densita' della vegetazione (0 = nessun albero)")
    ap.add_argument("--erosione", type=float, default=0.0, metavar="GOCCE",
                    help="gocce per cella di erosione idraulica (0 = spenta). "
                         "Utile sui terreni lisci; su mappe con rilievo dipinto "
                         "forte il rilievo c'e' gia' e l'erosione aggiunge poco")
    ap.add_argument("--adattivo", action="store_true",
                    help="classificazione relativa (Lab + raggruppamento) invece "
                         "delle soglie HSV tarate a mano")
    a = ap.parse_args()

    if a.anteprima:
        from genworld.anteprima import rendi
        png = a.uscita.rstrip("/\\") + "_anteprima.png"
        st = rendi(a.uscita, png, y0=-64, y1=220, ombreggiatura=1.0)
        print(f"anteprima {st['larghezza']}x{st['altezza']} da {st['chunk']} chunk "
              f"(quote {st['quota_min']:.0f}-{st['quota_max']:.0f}) -> {png}")
        if st["blocchi_ignoti"]:
            print("  blocchi senza colore:", st["blocchi_ignoti"])
        return

    op = Opzioni(immagine=a.immagine, uscita=a.uscita, nome=a.nome, lato=a.lato,
                 ritaglio=a.ritaglio, profilo=a.profilo, adattivo=a.adattivo,
                 erosione=a.erosione, fiumi=a.fiumi, vulcani=a.vulcani,
                 alberi=a.alberi, villaggi=a.villaggi, strade=a.strade)

    t0 = time.time()
    primo = a.lotto == 0
    analisi = analizza(op)
    if primo:
        for n in analisi.note:
            print(n)
        print(f"mappa {op.lato}x{op.lato}: quote {analisi.h.min()}..{analisi.h.max()}")

    piano = pianifica(op, analisi)
    if primo:
        for n in piano.note:
            print(n)

    quanti = a.chunk_per_lotto
    st = scrivi(op, analisi, piano, da=a.lotto * quanti if quanti else 0,
                quanti=quanti)
    da = a.lotto * quanti if quanti else 0
    print(f"lotto {a.lotto}: {st['chunk']} chunk da {da} di {st['totale']} "
          f"in {time.time() - t0:.0f}s ({st['restano']} rimanenti)")
    if "level_dat" in st:
        print("level.dat:", st["level_dat"])


if __name__ == "__main__":
    main()
