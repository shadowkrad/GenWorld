"""Verifica un mondo gia' generato: chunk mancanti, pezzi sospesi, acqua che trabocca.

    python esempi/verifica_mondo.py mondi/prova_case --lato 1000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genworld  # noqa: E402,F401  (prima di amulet: ripiego per rocksdb)
from genworld.verifica import verifica_mondo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mondo")
    ap.add_argument("--lato", type=int, required=True, help="lato del mondo in blocchi")
    a = ap.parse_args()
    rap = verifica_mondo(a.mondo, a.lato,
                         lambda f: print(f"\r{f:4.0%}", end="", file=sys.stderr))
    print()
    print("\n".join(rap.righe()))
    return 0 if rap.pulito else 1


if __name__ == "__main__":
    sys.exit(main())
