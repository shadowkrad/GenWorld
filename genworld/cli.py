"""Riga di comando dello stimatore.

    python -m genworld.cli stima --preset garda --lato 2048 --auto-verticale
    python -m genworld.cli confronta --preset italia
    python -m genworld.cli risolvi --preset italia --errore 500
"""

from __future__ import annotations

import argparse
import sys

from .fidelity import stima as calcola_stima
from .presets import MONUMENTI, TERRITORI
from .report import pannello, tabella_confronto
from .scale import Scala, Territorio, _fmt_m, mondo_per_errore, mondo_per_scala

LATI_CONFRONTO = (256, 512, 1024, 2048, 4096, 8192, 16384)


def _territorio(args) -> tuple[Territorio, list]:
    if args.preset:
        chiave = args.preset.lower()
        if chiave not in TERRITORI:
            sys.exit(f"preset sconosciuto: {args.preset}. Disponibili: {', '.join(TERRITORI)}")
        terr, fabbrica = TERRITORI[chiave]
        return terr, fabbrica()
    if not args.territorio:
        sys.exit("serve --preset oppure --territorio LARGHEZZAxALTEZZA (in metri)")
    try:
        larg, alt = (float(v) for v in args.territorio.lower().split("x"))
    except ValueError:
        sys.exit("formato --territorio non valido, atteso es. 20000x20000")
    terr = Territorio("Area personalizzata", larg, alt, args.quota_max, 0.0)
    return terr, []


def _landmark(args) -> list:
    if not args.monumenti:
        return []
    out = []
    for nome in args.monumenti.split(","):
        chiave = nome.strip().lower()
        if chiave not in MONUMENTI:
            sys.exit(f"monumento sconosciuto: {nome}. Disponibili: {', '.join(MONUMENTI)}")
        out.append(MONUMENTI[chiave])
    return out


def _costruisci_scala(args, terr) -> Scala:
    s = Scala(
        lato_mondo_blocchi=args.lato,
        territorio=terr,
        scala_oggetti_m_per_blocco=args.scala_oggetti,
        esagerazione_verticale=args.esagerazione,
        livello_mare_y=args.livello_mare,
    )
    if args.auto_verticale:
        s = s.con_esagerazione_ottimale()
    return s


def cmd_stima(args) -> None:
    terr, insediamenti = _territorio(args)
    scala = _costruisci_scala(args, terr)
    st = calcola_stima(
        scala,
        insediamenti=insediamenti,
        landmark=_landmark(args),
        fedelta_terreno=args.fedelta_terreno,
    )
    print(pannello(st))


def cmd_confronta(args) -> None:
    terr, insediamenti = _territorio(args)
    stime = []
    for lato in LATI_CONFRONTO:
        args.lato = lato
        scala = _costruisci_scala(args, terr)
        stime.append(
            calcola_stima(
                scala,
                insediamenti=insediamenti,
                landmark=_landmark(args),
                fedelta_terreno=args.fedelta_terreno,
            )
        )
    print(f"\n  {terr.nome} - effetto del budget di blocchi\n")
    print(tabella_confronto(stime))


def cmd_risolvi(args) -> None:
    """I tre modi di lavorare: fisso il mondo, la scala, oppure la fedelta'."""
    terr, _ = _territorio(args)
    print(f"\n  {terr.nome}: {_fmt_m(terr.larghezza_m)} x {_fmt_m(terr.altezza_m)}\n")

    if args.errore:
        lato = mondo_per_errore(terr, args.errore)
        print(f"  Fedelta' fissata a +/- {_fmt_m(args.errore)}")
        print(f"  -> servono almeno {lato} x {lato} blocchi ({lato ** 2 / 1e6:.1f} M colonne)")
    elif args.metri_per_blocco:
        lato = mondo_per_scala(terr, args.metri_per_blocco)
        print(f"  Scala fissata a 1 blocco = {_fmt_m(args.metri_per_blocco)}")
        print(f"  -> il mondo viene {lato} x {lato} blocchi ({lato ** 2 / 1e6:.1f} M colonne)")
    else:
        mpb = terr.lato_max_m / args.lato
        print(f"  Mondo fissato a {args.lato} x {args.lato} blocchi")
        print(f"  -> 1 blocco = {_fmt_m(mpb)}, forma fedele entro +/- {_fmt_m(mpb / 2)}")
    print()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="genworld",
        description="Stimatore di fedelta' per la generazione di mondi Minecraft.",
    )
    sub = p.add_subparsers(dest="comando", required=True)

    def comuni(sp):
        sp.add_argument("--preset", help=f"uno fra: {', '.join(TERRITORI)}")
        sp.add_argument("--territorio", help="estensione reale in metri, es. 20000x20000")
        sp.add_argument("--quota-max", type=float, default=0.0, help="quota massima in metri")
        sp.add_argument("--lato", type=int, default=1024, help="lato del mondo in blocchi")
        sp.add_argument("--scala-oggetti", type=float, default=1.0,
                        help="metri per blocco per le strutture (di norma 1.0)")
        sp.add_argument("--esagerazione", type=float, default=1.0,
                        help="esagerazione verticale")
        sp.add_argument("--auto-verticale", action="store_true",
                        help="calcola l'esagerazione che usa tutto il budget verticale")
        sp.add_argument("--livello-mare", type=int, default=62)
        sp.add_argument("--fedelta-terreno", type=float, default=1.0,
                        help="levetta 0..1 di semplificazione delle geometrie")
        sp.add_argument("--monumenti", help="elenco separato da virgole, es. torre_eiffel,colosseo")

    comuni(sub.add_parser("stima", help="pannello di fedelta' per una configurazione"))
    comuni(sub.add_parser("confronta", help="stessa mappa a diverse dimensioni di mondo"))
    sp_r = sub.add_parser("risolvi", help="fisso mondo, scala oppure fedelta'")
    comuni(sp_r)
    sp_r.add_argument("--errore", type=float, help="errore massimo tollerato in metri")
    sp_r.add_argument("--metri-per-blocco", type=float, help="scala desiderata")

    args = p.parse_args(argv)
    {"stima": cmd_stima, "confronta": cmd_confronta, "risolvi": cmd_risolvi}[args.comando](args)


if __name__ == "__main__":
    main()
