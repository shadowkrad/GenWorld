"""Apre la finestra offscreen, la riempie e ne salva uno scatto.

Serve a guardare la GUI senza uno schermo: Qt sa disegnare su un buffer
(`QT_QPA_PLATFORM=offscreen`) e la finestra si fotografa con `grab()`.

    python esempi/scatto_gui.py [--analizza]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from genworld.gui import Finestra  # noqa: E402
from genworld.motore import analizza  # noqa: E402

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _senza_dialoghi() -> None:
    """Le finestrelle modali bloccherebbero uno scatto automatico."""
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)


def main() -> None:
    app = QApplication([])
    f = Finestra()
    f.resize(1240, 800)
    f.show()
    app.processEvents()

    if "--analizza" in sys.argv:
        # si chiama il motore direttamente: il thread non serve a uno scatto
        a = analizza(f.opzioni())
        f._analisi_pronta(a)
        app.processEvents()

    if "--genera" in sys.argv:
        # questa invece e' la prova vera: il bottone, il thread, la barra
        _senza_dialoghi()
        f.lato.setValue(int(sys.argv[sys.argv.index("--genera") + 1]))
        f.bottone_genera.click()
        visti = []
        f.lavoro.avanzato.connect(lambda fr, t: visti.append((fr, t)))
        while f.lavoro is not None:
            app.processEvents()
            QThread.msleep(30)
        print(f"avanzamenti ricevuti: {len(visti)}, ultimo {visti[-1] if visti else '-'}")
        print("stato:", f.stato.text())

    dest = os.path.join(RADICE, "mondi", "gui.png")
    f.grab().save(dest)
    print("scritto", dest)


if __name__ == "__main__":
    main()
