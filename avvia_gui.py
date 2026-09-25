"""Punto di ingresso per PyInstaller.

`python -m genworld.gui` funziona benissimo, ma PyInstaller vuole un file da
cui partire, non un modulo. Questo e' quel file, e non fa nient'altro.
"""

import sys

from genworld.gui import main

if __name__ == "__main__":
    sys.exit(main())
