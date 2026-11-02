"""Esporta un mondo Java di GenWorld in Bedrock (.mcworld).

    python esempi/esporta_bedrock.py mondi/italia mondi/italia.mcworld --nome "Italia"

Il file .mcworld si apre con un doppio clic (Minecraft Bedrock lo importa). Limiti nel
docstring di `genworld/bedrock.py`: niente Nether/Sottosopra, datapack e creature.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from genworld import bedrock


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mondo", help="cartella del mondo Java")
    ap.add_argument("uscita", help="file .mcworld (o cartella) da scrivere")
    ap.add_argument("--nome", default=None, help="nome del mondo in Bedrock")
    a = ap.parse_args()

    def avanza(fatti, totale):
        print(f"\r{fatti}/{totale} chunk", end="", flush=True)

    r = bedrock.esporta(a.mondo, a.uscita, nome=a.nome, avanza=avanza)
    print()
    for d, n in r.items():
        print(f"{d}: {n} chunk")
    print("scritto", a.uscita)


if __name__ == "__main__":
    main()
