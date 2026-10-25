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
    ap.add_argument("--altezza", type=int, default=0,
                    help="mondo rettangolare: --lato blocchi in x per --altezza in z (0 = quadrato). "
                         "Solo con --dem; l'origine resta a meta' del lato")
    ap.add_argument("--ritaglio", type=float, default=0.025,
                    help="frazione di cornice da togliere per lato")
    ap.add_argument("--lotto", type=int, default=0)
    ap.add_argument("--chunk-per-lotto", type=int, default=0,
                    help="0 = tutto in una volta")
    ap.add_argument("--templates", default=";".join(
                        c for c in (os.path.join(RADICE, "templates", "strutture"),)
                        if os.path.isdir(c)),
                    help="cartelle dei .nbt da usare come case, separate da ';' "
                         "('' per i lotti senza casa: non c'e' piu' un "
                         "generatore parametrico di riserva). Di default solo "
                         "'templates/strutture': 'templates/case' contiene le "
                         "case esportate dal vecchio generatore parametrico e "
                         "va aggiunta a mano solo se la si vuole di nuovo")
    ap.add_argument("--templates-cimitero", default=";".join(
                        c for c in (os.path.join(RADICE, "templates", "cimiteri"),)
                        if os.path.isdir(c)),
                    help="cartelle dei .nbt da usare come cimitero, separate da "
                         "';' ('' per il ripiego disegnato da codice). Di "
                         "default 'templates/cimiteri' se esiste")
    ap.add_argument("--templates-portale", default=";".join(
                        c for c in (os.path.join(RADICE, "templates", "portali"),)
                        if os.path.isdir(c)),
                    help="cartelle dei .nbt da usare come portale, separate da "
                         "';' ('' per nessun portale: non c'e' un ripiego "
                         "disegnato da codice per questa struttura). Di "
                         "default 'templates/portali' se esiste")
    ap.add_argument("--templates-castello", default=";".join(
                        c for c in (os.path.join(RADICE, "templates", "castelli"),)
                        if os.path.isdir(c)),
                    help="cartelle di schemi (.schem, .litematic, .nbt) per il "
                         "castello, uno per mappa. Di default 'templates/castelli'")
    ap.add_argument("--templates-nave", default=";".join(
                        c for c in (os.path.join(RADICE, "templates", "navi"),)
                        if os.path.isdir(c)),
                    help="cartelle di schemi per le navi in mare. Di default "
                         "'templates/navi'")
    ap.add_argument("--templates-molo", default=";".join(
                        c for c in (os.path.join(RADICE, "templates", "moli"),)
                        if os.path.isdir(c)),
                    help="cartelle con lo schema del molo (banchina con casetta) e "
                         "`meta.json`. Di default 'templates/moli'; senza, il molo e' "
                         "disegnato da codice")
    ap.add_argument("--templates-faro", default=";".join(
                        c for c in (os.path.join(RADICE, "templates", "fari"),)
                        if os.path.isdir(c)),
                    help="cartelle con il faro: schemi o mondi (cartelle con level.dat) "
                         "e `meta.json` con la regione. Di default 'templates/fari'")
    ap.add_argument("--castelli", type=float, default=1.0, metavar="SCALA",
                    help="0 = nessun castello")
    ap.add_argument("--dimensioni", action=argparse.BooleanOptionalAction, default=True,
                    help="scrive anche il Nether e l'End di Arda (default: si')")
    ap.add_argument("--isole", type=float, default=1.0, metavar="SCALA",
                    help="isole del tesoro in mare aperto (0 = nessuna)")
    ap.add_argument("--navi", type=float, default=1.0, metavar="SCALA",
                    help="densita' delle navi in mare (0 = nessuna)")
    ap.add_argument("--porti", type=float, default=1.0, metavar="SCALA",
                    help="molo, barche ormeggiate e faro per gli abitati sulla costa "
                         "(0 = nessuno). Le barche vengono dal pacchetto delle navi")
    ap.add_argument("--anteprima", action="store_true")
    ap.add_argument("--profilo", help="profilo JSON prodotto dal Calibratore Mappe: "
                    "rettangolo di interesse + campioni col contagocce")
    ap.add_argument("--vulcani", type=int, default=3, metavar="N",
                    help="quanti vulcani piazzare (0 = nessuno)")
    ap.add_argument("--sottosopra", action=argparse.BooleanOptionalAction, default=False,
                    help="il Nether e' lo specchio dell'overworld: stessa geografia, materiali del "
                         "Nether, lava al posto dell'acqua (scala 1:1, via datapack). "
                         "--no-sottosopra (il predefinito): il Nether disegnato da noi, con fortezze e bastioni")
    ap.add_argument("--decadimento", type=float, default=0.5, metavar="0..1",
                    help="solo con --sottosopra: quanto sono in rovina le costruzioni (0 intatte)")
    ap.add_argument("--dem", default="", metavar="FILE.npz",
                    help="quote reali preparate con esempi/prepara_dem.py, al posto dell'immagine")
    ap.add_argument("--solo-citta-reali", action="store_true",
                    help="niente villaggi automatici: solo le citta' del file --luoghi-reali")
    ap.add_argument("--luoghi-reali", default="", metavar="FILE.json",
                    help="citta' e vulcani veri (lat/lon) da piazzare dove stanno; serve --dem")
    ap.add_argument("--vulcani-px", metavar="X,Y;X,Y", default="",
                    help="posizioni dei vulcani in PIXEL dell'immagine originale (come le legge "
                         "un editor di immagini); se ci sono, sono le uniche e --vulcani non conta")
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
    ap.add_argument("--arredi", type=float, default=1.0, metavar="SCALA",
                    help="densita' degli arredi urbani sui lotti senza casa e "
                         "lungo le vie (0 = lotti vuoti)")
    ap.add_argument("--isolate", type=float, default=1.0, metavar="SCALA",
                    help="densita' delle case isolate fuori dai villaggi (i "
                         "template troppo alti per un lotto; 0 = nessuna)")
    ap.add_argument("--fauna", type=float, default=1.0, metavar="SCALA",
                    help="densita' della fauna, selvatica e da cortile (0 = nessuna)")
    ap.add_argument("--laghi", type=float, default=1.0, metavar="SCALA",
                    help="riempie le conche chiuse trovate nel rilievo "
                         "(0 = terreno asciutto anche nei bacini senza sbocco)")
    ap.add_argument("--miniere", type=float, default=1.0, metavar="SCALA",
                    help="densita' dei condotti artificiali (pozzi, gallerie, "
                         "binari e filoni lungo il percorso; 0 = nessuna)")
    ap.add_argument("--accampamenti", type=float, default=1.0, metavar="SCALA",
                    help="densita' degli accampamenti di nemici (0 = nessuno)")
    ap.add_argument("--cimiteri", type=float, default=1.0, metavar="SCALA",
                    help="densita' dei cimiteri (0 = nessuno)")
    ap.add_argument("--portali", type=float, default=1.0, metavar="SCALA",
                    help="densita' dei portali (0 = nessuno; comunque nessuno "
                         "senza --templates-portale, non c'e' un ripiego)")
    ap.add_argument("--erosione", type=float, default=0.0, metavar="GOCCE",
                    help="gocce per cella di erosione idraulica (0 = spenta). "
                         "Utile sui terreni lisci; su mappe con rilievo dipinto "
                         "forte il rilievo c'e' gia' e l'erosione aggiunge poco")
    ap.add_argument("--adattivo", action="store_true",
                    help="classificazione relativa (Lab + raggruppamento) invece "
                         "delle soglie HSV tarate a mano")
    ap.add_argument("--versione", default="1.21.4", metavar="X.Y.Z",
                    help="versione Java di riferimento (DataVersion di "
                         "level.dat e traduzione dei blocchi dei template), "
                         "es. 1.21.9. Dalla 1.21.4 in poi")
    ap.add_argument("--forme-citta", default="miste",
                    help="forma delle citta' murate: 'miste' (a sorte, fissata per ogni "
                         "citta'), 'quadrato', 'rettangolo', 'ovale', 'tondo', oppure un "
                         "elenco separato da virgole (es. tondo,ovale,rettangolo,quadrato) "
                         "percorso in ordine dalle citta' murate")
    ap.add_argument("--cache-piano", action="store_true",
                    help="salva/rilegge la pianificazione (accanto al mondo, *_piano.pkl): "
                         "vale solo con le stesse opzioni, gli stessi template e lo stesso codice")
    ap.add_argument("--processi", type=int, default=1, metavar="N",
                    help="scrive i chunk con N processi (0 = uno per core, al massimo 8; "
                         "1 = come prima). Non si combina con --chunk-per-lotto")
    ap.add_argument("--fuori-mappa", choices=("naturale", "vuoto"), default="naturale",
                    help="cosa c'e' oltre la mappa: 'naturale' (terreno normale di "
                         "Minecraft generato al volo, nessun bordo) o 'vuoto' (aria)")
    a = ap.parse_args()

    try:
        versione = tuple(int(n) for n in a.versione.split("."))
        if len(versione) != 3:
            raise ValueError
    except ValueError:
        ap.error(f"--versione vuole X.Y.Z, non {a.versione!r}")

    if a.anteprima:
        from genworld.anteprima import rendi
        png = a.uscita.rstrip("/\\") + "_anteprima.png"
        st = rendi(a.uscita, png, y0=-64, y1=220, ombreggiatura=1.0)
        print(f"anteprima {st['larghezza']}x{st['altezza']} da {st['chunk']} chunk "
              f"(quote {st['quota_min']:.0f}-{st['quota_max']:.0f}) -> {png}")
        if st["blocchi_ignoti"]:
            print("  blocchi senza colore:", st["blocchi_ignoti"])
        return

    vulcani_pos = ()
    if a.vulcani_px:
        from genworld.profilo import Profilo
        punti = [tuple(float(v) for v in c.split(",")) for c in a.vulcani_px.split(";") if c.strip()]
        prof_ = Profilo.carica(a.profilo) if a.profilo else Profilo()
        vulcani_pos = tuple(prof_.frazioni_da_pixel(a.immagine, a.lato, punti))
    citta_reali = ()
    if a.dem and a.luoghi_reali:
        import json
        from genworld import dem as DEM
        dati = json.load(open(a.luoghi_reali, encoding="utf-8"))
        d_ = DEM.carica(a.dem)
        passo_cella = d_["passo"] / min(a.lato / d_["elev"].shape[1], (a.altezza or a.lato) / d_["elev"].shape[0])
        if d_.get("crs") == "EPSG:4326":
            passo_cella *= 111_320.0
        vul = dati.get("vulcani", [])
        fr = DEM.frazioni_da_lonlat(d_, a.lato, [(v["lon"], v["lat"]) for v in vul], a.altezza or None)
        vulcani_pos = tuple(
            (fx, fz, max(5, round(v["raggio_km"] * 1000 / passo_cella)), v["altezza"],
             max(2, round(v.get("cratere_km", 1.0) * 1000 / passo_cella)),
             int(bool(v.get("spento", False))), int(bool(v.get("lago", False))))
            for (fx, fz), v in zip(fr, vul))
        cit = dati.get("citta", [])
        frc = DEM.frazioni_da_lonlat(d_, a.lato, [(c["lon"], c["lat"]) for c in cit], a.altezza or None)
        citta_reali = tuple((c["nome"], fx, fz, bool(c.get("grande"))) for (fx, fz), c in zip(frc, cit)
                            if 0 < fx < 1 and 0 < fz < 1)
    op = Opzioni(immagine=a.immagine, vulcani_posizioni=vulcani_pos, dem=a.dem, sottosopra=a.sottosopra, decadimento=a.decadimento, citta_reali=citta_reali, solo_citta_reali=a.solo_citta_reali, uscita=a.uscita, nome=a.nome, lato=a.lato, altezza=a.altezza,
                 ritaglio=a.ritaglio, profilo=a.profilo, adattivo=a.adattivo,
                 erosione=a.erosione, fiumi=a.fiumi, vulcani=a.vulcani,
                 alberi=a.alberi, villaggi=a.villaggi, strade=a.strade,
                 templates=a.templates, templates_cimitero=a.templates_cimitero,
                 templates_portale=a.templates_portale,
                 templates_castello=a.templates_castello,
                 templates_nave=a.templates_nave, templates_molo=a.templates_molo,
                 templates_faro=a.templates_faro, castelli=a.castelli, navi=a.navi, porti=a.porti, isole=a.isole, dimensioni=a.dimensioni,
                 fauna=a.fauna, arredi=a.arredi, isolate=a.isolate, laghi=a.laghi,
                 miniere=a.miniere, accampamenti=a.accampamenti,
                 cimiteri=a.cimiteri, portali=a.portali, versione=versione,
                 fuori_mappa=a.fuori_mappa, forme_citta=a.forme_citta)

    from genworld import template as TM
    if a.villaggi > 0 and TM.conta_file(a.templates) == 0:
        print("ATTENZIONE: " + TM.AVVISO_SENZA_CASE)

    t0 = time.time()
    primo = a.lotto == 0
    analisi = analizza(op)
    if primo:
        for n in analisi.note:
            print(n)
        print(f"mappa {op.lato}x{op.lato}: quote {analisi.h.min()}..{analisi.h.max()}")

    if a.cache_piano:
        from genworld.motore import pianifica_con_cache
        piano, dalla_cache = pianifica_con_cache(op, analisi)
        if dalla_cache:
            print("piano riletto dalla cache")
    else:
        piano = pianifica(op, analisi)
    if primo:
        for n in piano.note:
            print(n)

    quanti = a.chunk_per_lotto
    if a.processi != 1 and not quanti:
        from genworld.parallelo import scrivi_parallelo
        st = scrivi_parallelo(op, analisi, piano, processi=a.processi)
    else:
        st = scrivi(op, analisi, piano, da=a.lotto * quanti if quanti else 0,
                    quanti=quanti)
    da = a.lotto * quanti if quanti else 0
    print(f"lotto {a.lotto}: {st['chunk']} chunk da {da} di {st['totale']} "
          f"in {time.time() - t0:.0f}s ({st['restano']} rimanenti)")
    if "level_dat" in st:
        print("level.dat:", st["level_dat"])


if __name__ == "__main__":
    main()
