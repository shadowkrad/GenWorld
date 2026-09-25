"""GenWorld - la finestra.

E' il punto da cui tutto era cominciato: "un applicativo per pc". Il motore
esisteva gia' e si guidava da riga di comando; qui si tengono insieme le due
cose che finora stavano separate - i cursori e il **pannello di fedelta' che
si aggiorna mentre li trascini**.

Tre regole di questa finestra:

1. Il pannello di fedelta' e' un preventivo, non un voto. Mostra una riga per
   dimensione, non un numero unico, perche' un numero unico nasconde proprio
   la cosa che interessa: *cosa* si perde.
2. Le **strutture non si rimpiccioliscono** con il terreno. La scala degli
   oggetti resta 1 m per blocco e non c'e' nessun controllo per cambiarla: il
   cursore muove solo la scala del terreno, e il pannello mostra il fattore di
   esagerazione che ne risulta.
3. Il lavoro pesante sta in un thread. Una generazione da 2704 chunk dura
   minuti: la finestra deve restare viva, mostrare a che punto e' e potersi
   annullare.

Avvio:

    python -m genworld.gui
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from dataclasses import replace

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QSpinBox, QTabWidget,
                               QVBoxLayout, QWidget)

from . import mappa as M
from .fidelity import Stima, stima as calcola_stima
from .motore import LIVELLO_MARE, Analisi, Opzioni, analizza, genera
from .presets import MONUMENTI, TERRITORI
from .scale import Scala, Territorio, _fmt_m

def _radice() -> str:
    """La cartella del progetto: input/, mondi/, i profili.

    Impacchettato con PyInstaller, `__file__` finisce dentro il bundle
    temporaneo e `mondi/` verrebbe scritto li', cioe' buttato via alla
    chiusura. Congelato, la radice e' la cartella dell'eseguibile.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


RADICE = _radice()


# --------------------------------------------------------------------------
# Pezzi di interfaccia
# --------------------------------------------------------------------------

class Barra(QWidget):
    """Barra 0..1 disegnata a mano.

    Una QProgressBar farebbe quasi la stessa cosa, ma qui il colore deve
    dipendere dal punteggio: il pannello si legge con la coda dell'occhio
    mentre si trascina un cursore, e il colore e' la lettura piu' veloce.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._v = 0.0
        self.setMinimumSize(120, 14)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def imposta(self, v: float) -> None:
        self._v = max(0.0, min(1.0, float(v)))
        self.update()

    def paintEvent(self, ev) -> None:  # noqa: N802 (API Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 3, -1, -3)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(228, 228, 232))
        p.drawRoundedRect(r, 4, 4)
        if self._v > 0:
            pieno = QColor(*_colore(self._v))
            p.setBrush(pieno)
            rr = r.adjusted(0, 0, -int(r.width() * (1 - self._v)), 0)
            p.drawRoundedRect(rr, 4, 4)
        p.end()


def _colore(v: float) -> tuple[int, int, int]:
    if v >= 0.85:
        return (56, 142, 76)
    if v >= 0.65:
        return (120, 160, 60)
    if v >= 0.45:
        return (206, 166, 42)
    if v >= 0.25:
        return (214, 122, 38)
    return (196, 66, 52)


class PannelloFedelta(QGroupBox):
    """Le righe dello stimatore, ridisegnate a ogni movimento dei cursori."""

    def __init__(self, parent=None):
        super().__init__("Preventivo di fedelta'", parent)
        self._righe: list[tuple[QLabel, Barra, QLabel, QLabel]] = []
        v = QVBoxLayout(self)
        self.intestazione = QLabel("—")
        self.intestazione.setWordWrap(True)
        f = self.intestazione.font()
        f.setBold(True)
        self.intestazione.setFont(f)
        v.addWidget(self.intestazione)

        self.griglia = QGridLayout()
        self.griglia.setColumnStretch(1, 1)
        v.addLayout(self.griglia)

        self.regime = QLabel("—")
        self.regime.setWordWrap(True)
        v.addWidget(self.regime)
        self.avvisi = QLabel("")
        self.avvisi.setWordWrap(True)
        self.avvisi.setStyleSheet("color: #a4501a;")
        v.addWidget(self.avvisi)
        v.addStretch(1)

    def _riga(self, i: int):
        while len(self._righe) <= i:
            nome, barra = QLabel(), Barra()
            valore, giudizio = QLabel(), QLabel()
            giudizio.setStyleSheet("color: #666;")
            r = len(self._righe)
            self.griglia.addWidget(nome, r, 0)
            self.griglia.addWidget(barra, r, 1)
            self.griglia.addWidget(valore, r, 2)
            self.griglia.addWidget(giudizio, r, 3)
            self._righe.append((nome, barra, valore, giudizio))
        return self._righe[i]

    def mostra(self, s: Stima) -> None:
        sc = s.scala
        self.intestazione.setText(
            f"{sc.lato_mondo_blocchi} x {sc.lato_mondo_blocchi} blocchi   ·   "
            f"1 blocco = {_fmt_m(sc.metri_per_blocco)}   ·   "
            f"territorio {_fmt_m(sc.territorio.larghezza_m)} x "
            f"{_fmt_m(sc.territorio.altezza_m)}")
        for i, d in enumerate(s.dimensioni):
            nome, barra, valore, giudizio = self._riga(i)
            nome.setText(d.nome)
            barra.imposta(d.punteggio)
            valore.setText(d.valore)
            giudizio.setText(d.giudizio)
            for w in (nome, barra, valore, giudizio):
                w.setVisible(True)
            nome.setToolTip(d.nota)
            barra.setToolTip(d.nota)
        for i in range(len(s.dimensioni), len(self._righe)):
            for w in self._righe[i]:
                w.setVisible(False)

        self.regime.setText(
            f"Regime: <b>{s.regime}</b> — gli oggetti sono {sc.fattore_esagerazione:.0f}x "
            f"fuori scala rispetto al terreno, e cosi' devono restare: una casa "
            f"resta una casa in cui si entra.")
        self.avvisi.setText("\n".join("• " + a for a in s.avvisi))


class Anteprima(QLabel):
    """Immagine che si adatta al riquadro senza deformarsi."""

    def __init__(self, testo: str, parent=None):
        super().__init__(testo, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setStyleSheet("background: #1e1e22; color: #888; border-radius: 4px;")
        self._pix: QPixmap | None = None

    def imposta_immagine(self, im) -> None:
        """`im` e' una PIL.Image RGB."""
        im = im.convert("RGB")
        dati = im.tobytes("raw", "RGB")
        # i byte devono sopravvivere alla QImage: senza questo riferimento
        # l'immagine mostra spazzatura appena il garbage collector passa
        self._dati = dati
        q = QImage(dati, im.width, im.height, im.width * 3, QImage.Format_RGB888)
        self._pix = QPixmap.fromImage(q)
        self._ridisegna()

    def _ridisegna(self) -> None:
        if self._pix is None:
            return
        self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation))

    def resizeEvent(self, ev):  # noqa: N802 (API Qt)
        super().resizeEvent(ev)
        self._ridisegna()


# --------------------------------------------------------------------------
# Lavoro in secondo piano
# --------------------------------------------------------------------------

class Lavoro(QThread):
    """Esegue una funzione lunga fuori dal thread della finestra.

    La funzione riceve `avanza(frazione, testo)` e non deve MAI toccare un
    widget: l'avanzamento passa da un segnale, che Qt consegna al thread
    giusto.
    """

    avanzato = Signal(float, str)
    finito = Signal(object)
    fallito = Signal(str)

    def __init__(self, funzione, parent=None):
        super().__init__(parent)
        self._f = funzione
        self.stop = threading.Event()

    def run(self) -> None:  # noqa: D102
        try:
            r = self._f(lambda f, t: self.avanzato.emit(float(f), str(t)),
                        self.stop.is_set)
        except Exception:
            self.fallito.emit(traceback.format_exc())
            return
        self.finito.emit(r)


# --------------------------------------------------------------------------
# Finestra
# --------------------------------------------------------------------------

class Finestra(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenWorld — da una mappa a un mondo Minecraft")
        self.resize(1240, 800)
        self.analisi: Analisi | None = None
        self.lavoro: Lavoro | None = None

        radice = QHBoxLayout(self)
        radice.addWidget(self._colonna_sinistra(), 0)
        radice.addWidget(self._colonna_destra(), 1)
        self._aggiorna_stima()

    # -- costruzione ------------------------------------------------------

    def _colonna_sinistra(self) -> QWidget:
        corpo = QWidget()
        v = QVBoxLayout(corpo)
        v.setContentsMargins(0, 0, 0, 0)

        # Mappa ---------------------------------------------------------
        g = QGroupBox("Mappa")
        f = QFormLayout(g)
        self.percorso = QLineEdit(os.path.join(RADICE, "input", "mappa_arda.png"))
        b = QPushButton("Sfoglia…")
        b.setFixedWidth(84)
        b.clicked.connect(self._scegli_immagine)
        r1 = QHBoxLayout()
        r1.addWidget(self.percorso, 1)
        r1.addWidget(b, 0)
        f.addRow("Immagine", r1)
        self.percorso_profilo = QLineEdit()
        self.percorso_profilo.setPlaceholderText("facoltativo")
        bp = QPushButton("Sfoglia…")
        bp.setFixedWidth(84)
        bp.clicked.connect(self._scegli_profilo)
        r2 = QHBoxLayout()
        r2.addWidget(self.percorso_profilo, 1)
        r2.addWidget(bp, 0)
        f.addRow("Profilo", r2)
        self.adattivo = QCheckBox("Classificazione adattiva (mappe con tinte inusuali)")
        f.addRow(self.adattivo)
        self.bottone_analizza = QPushButton("Analizza la mappa")
        self.bottone_analizza.clicked.connect(self._analizza)
        f.addRow(self.bottone_analizza)
        v.addWidget(g)

        # Mondo ---------------------------------------------------------
        g = QGroupBox("Mondo")
        f = QFormLayout(g)
        self.lato = QSpinBox()
        self.lato.setRange(128, 4096)
        self.lato.setSingleStep(64)
        self.lato.setValue(832)
        self.lato.valueChanged.connect(self._aggiorna_stima)
        self.nota_lato = QLabel()
        f.addRow("Lato in blocchi", self.lato)
        f.addRow("", self.nota_lato)
        v.addWidget(g)

        # Elementi ------------------------------------------------------
        g = QGroupBox("Cosa generare")
        f = QFormLayout(g)
        self.vulcani = self._cursore(f, "Vulcani", 0, 6, 3, passo=1)
        self.fiumi = self._cursore(f, "Fiumi (soglia)", 0, 400, 150, passo=10)
        self.alberi = self._cursore(f, "Alberi", 0, 300, 100, suffisso="%")
        self.villaggi = self._cursore(f, "Villaggi", 0, 300, 100, suffisso="%")
        self.erosione = self._cursore(f, "Erosione", 0, 100, 0, suffisso="%")
        self.strade = QCheckBox("Strade e ponti")
        self.strade.setChecked(True)
        f.addRow(self.strade)
        v.addWidget(g)

        # Territorio ----------------------------------------------------
        g = QGroupBox("Territorio reale (solo per il preventivo)")
        f = QFormLayout(g)
        self.preset = QComboBox()
        self.preset.addItem("— personalizzato —", None)
        for k, (t, _) in TERRITORI.items():
            self.preset.addItem(t.nome, k)
        # si parte da un territorio vero: con "personalizzato" la riga degli
        # insediamenti sarebbe verde piena senza che nessuno l'abbia chiesta,
        # e un preventivo che dice "ottima" su niente non aiuta
        self.preset.setCurrentIndex(max(1, self.preset.findData("garda")))
        self.preset.currentIndexChanged.connect(self._preset_cambiato)
        f.addRow("Preset", self.preset)
        self.larghezza_km = QSpinBox()
        self.larghezza_km.setRange(1, 5000)
        self.larghezza_km.setValue(20)
        self.larghezza_km.setSuffix(" km")
        self.larghezza_km.valueChanged.connect(self._aggiorna_stima)
        f.addRow("Lato del territorio", self.larghezza_km)
        self.quota_max = QSpinBox()
        self.quota_max.setRange(0, 9000)
        self.quota_max.setValue(2218)
        self.quota_max.setSuffix(" m")
        self.quota_max.valueChanged.connect(self._aggiorna_stima)
        f.addRow("Quota massima", self.quota_max)
        self.auto_verticale = QCheckBox("Esagerazione verticale automatica")
        self.auto_verticale.setChecked(True)
        self.auto_verticale.stateChanged.connect(self._aggiorna_stima)
        f.addRow(self.auto_verticale)
        self.monumento = QComboBox()
        self.monumento.addItem("— nessuno —", None)
        for k, lm in MONUMENTI.items():
            self.monumento.addItem(lm.nome, k)
        self.monumento.currentIndexChanged.connect(self._aggiorna_stima)
        f.addRow("Monumento di prova", self.monumento)
        v.addWidget(g)
        v.addStretch(1)

        rotolo = QScrollArea()
        rotolo.setWidget(corpo)
        rotolo.setWidgetResizable(True)
        rotolo.setFrameShape(QFrame.NoFrame)
        rotolo.setFixedWidth(430)
        return rotolo

    def _cursore(self, form: QFormLayout, etichetta: str, minimo: int,
                 massimo: int, valore: int, passo: int = 1,
                 suffisso: str = "") -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(minimo, massimo)
        s.setSingleStep(passo)
        s.setValue(valore)
        eco = QLabel(f"{valore}{suffisso}")
        eco.setMinimumWidth(52)
        eco.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        s.valueChanged.connect(lambda v: eco.setText(f"{v}{suffisso}"))
        riga = QHBoxLayout()
        riga.addWidget(s, 1)
        riga.addWidget(eco, 0)
        form.addRow(etichetta, riga)
        return s

    def _colonna_destra(self) -> QWidget:
        corpo = QWidget()
        v = QVBoxLayout(corpo)
        v.setContentsMargins(0, 0, 0, 0)

        self.schede = QTabWidget()
        self.vista_classi = Anteprima("Analizza la mappa per vedere le classi")
        self.vista_quote = Anteprima("…e l'altimetria dedotta")
        self.vista_mondo = Anteprima("Il mondo generato comparira' qui")
        self.schede.addTab(self.vista_classi, "Classi")
        self.schede.addTab(self.vista_quote, "Quote")
        self.schede.addTab(self.vista_mondo, "Mondo")
        v.addWidget(self.schede, 1)

        self.pannello = PannelloFedelta()
        self.pannello.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        v.addWidget(self.pannello, 0)

        # Altezza fissa: il riepilogo e' lungo e variabile, e senza un
        # limite si mangiava la riga dei bottoni quando compariva un avviso.
        self.registro = QLabel("")
        self.registro.setWordWrap(True)
        self.registro.setFixedHeight(44)
        self.registro.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.registro.setStyleSheet("color: #555;")
        v.addWidget(self.registro, 0)

        riga = QHBoxLayout()
        self.avanzamento = QProgressBar()
        self.avanzamento.setRange(0, 1000)
        self.stato = QLabel("pronto")
        self.stato.setMinimumWidth(240)
        self.bottone_genera = QPushButton("Genera il mondo")
        self.bottone_genera.clicked.connect(self._genera)
        f = QFont(self.bottone_genera.font())
        f.setBold(True)
        self.bottone_genera.setFont(f)
        self.bottone_annulla = QPushButton("Annulla")
        self.bottone_annulla.setEnabled(False)
        self.bottone_annulla.clicked.connect(self._annulla)
        riga.addWidget(self.avanzamento, 1)
        riga.addWidget(self.stato, 0)
        riga.addWidget(self.bottone_annulla, 0)
        riga.addWidget(self.bottone_genera, 0)
        v.addLayout(riga)
        return corpo

    # -- opzioni ----------------------------------------------------------

    def opzioni(self) -> Opzioni:
        nome = os.path.splitext(os.path.basename(self.percorso.text()))[0] or "mondo"
        return Opzioni(
            immagine=self.percorso.text(),
            uscita=os.path.join(RADICE, "mondi", nome),
            nome=nome.replace("_", " ").title(),
            lato=int(self.lato.value()),
            profilo=self.percorso_profilo.text().strip() or None,
            adattivo=self.adattivo.isChecked(),
            erosione=self.erosione.value() / 100.0,
            fiumi=float(self.fiumi.value()),
            vulcani=int(self.vulcani.value()),
            alberi=self.alberi.value() / 100.0,
            villaggi=self.villaggi.value() / 100.0,
            strade=self.strade.isChecked(),
        )

    def territorio(self) -> tuple[Territorio, list]:
        chiave = self.preset.currentData()
        if chiave:
            t, fabbrica = TERRITORI[chiave]
            return t, fabbrica()
        lato_m = self.larghezza_km.value() * 1000.0
        return Territorio("Area personalizzata", lato_m, lato_m,
                          float(self.quota_max.value()), 0.0), []

    def _preset_cambiato(self) -> None:
        chiave = self.preset.currentData()
        if chiave:
            t, _ = TERRITORI[chiave]
            for w, val in ((self.larghezza_km, int(t.lato_max_m / 1000)),
                           (self.quota_max, int(t.quota_max_m))):
                w.blockSignals(True)
                w.setValue(min(val, w.maximum()))
                w.blockSignals(False)
        self._aggiorna_stima()

    # -- preventivo -------------------------------------------------------

    def _aggiorna_stima(self) -> None:
        terr, insediamenti = self.territorio()
        sc = Scala(lato_mondo_blocchi=int(self.lato.value()), territorio=terr,
                   livello_mare_y=LIVELLO_MARE)
        if self.auto_verticale.isChecked():
            sc = sc.con_esagerazione_ottimale()
        chiave = self.monumento.currentData()
        landmark = [MONUMENTI[chiave]] if chiave else []
        self.pannello.mostra(calcola_stima(sc, insediamenti, landmark))
        n = self.lato.value() // 16
        self.nota_lato.setText(f"{n} x {n} = {n * n} chunk")

    # -- azioni -----------------------------------------------------------

    def _scegli_immagine(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Scegli la mappa", os.path.join(RADICE, "input"),
            "Immagini (*.png *.jpg *.jpeg *.bmp *.webp)")
        if p:
            self.percorso.setText(p)
            # un profilo appartiene a una mappa: accanto all'immagine c'e'
            # spesso il suo <nome>.profilo.json prodotto dal Calibratore
            candidato = os.path.splitext(p)[0] + ".profilo.json"
            if os.path.exists(candidato):
                self.percorso_profilo.setText(candidato)

    def _scegli_profilo(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Scegli il profilo", os.path.join(RADICE, "input"),
            "Profili (*.json)")
        if p:
            self.percorso_profilo.setText(p)

    def _occupata(self, si: bool) -> None:
        self.bottone_genera.setEnabled(not si)
        self.bottone_analizza.setEnabled(not si)
        self.bottone_annulla.setEnabled(si)

    def _analizza(self) -> None:
        if self.lavoro is not None:
            return
        op = self.opzioni()
        if not os.path.exists(op.immagine):
            QMessageBox.warning(self, "GenWorld", "Immagine non trovata.")
            return
        self._occupata(True)
        self.lavoro = Lavoro(lambda avanza, ferma: analizza(op, avanza), self)
        self.lavoro.avanzato.connect(self._avanzato)
        self.lavoro.finito.connect(self._analisi_pronta)
        self.lavoro.fallito.connect(self._errore)
        self.lavoro.start()

    def _genera(self) -> None:
        if self.lavoro is not None:
            return
        op = self.opzioni()
        if not os.path.exists(op.immagine):
            QMessageBox.warning(self, "GenWorld", "Immagine non trovata.")
            return
        if os.path.exists(op.uscita):
            r = QMessageBox.question(
                self, "GenWorld",
                f"La cartella\n{op.uscita}\nesiste gia' e verra' riscritta. Procedo?")
            if r != QMessageBox.Yes:
                return
        self._occupata(True)
        self.lavoro = Lavoro(lambda avanza, ferma: genera(op, avanza, ferma), self)
        self.lavoro.avanzato.connect(self._avanzato)
        self.lavoro.finito.connect(lambda st: self._fatto(op, st))
        self.lavoro.fallito.connect(self._errore)
        self.lavoro.start()

    def _annulla(self) -> None:
        if self.lavoro is not None:
            self.lavoro.stop.set()
            self.stato.setText("annullamento…")

    def _avanzato(self, f: float, testo: str) -> None:
        self.avanzamento.setValue(int(f * 1000))
        self.stato.setText(testo)

    def _analisi_pronta(self, a: Analisi) -> None:
        self.analisi = a
        self.lavoro = None
        self._occupata(False)
        self.registro.setText("  ·  ".join(a.note))
        self.vista_classi.imposta_immagine(M.anteprima_classi(a.cls))
        self.vista_quote.imposta_immagine(M.anteprima_quote(a.h, LIVELLO_MARE))
        self.schede.setCurrentIndex(0)
        self.stato.setText("analisi pronta")

    def _fatto(self, op: Opzioni, st: dict) -> None:
        self.lavoro = None
        self._occupata(False)
        if st.get("interrotto"):
            self.stato.setText("annullato")
            return
        self.registro.setText("  ·  ".join(st.get("note", [])))
        self.stato.setText(f"{st['chunk']} chunk in {st['secondi']:.0f}s")
        self.avanzamento.setValue(1000)
        try:
            from .anteprima import rendi
            png = op.uscita + "_anteprima.png"
            rendi(op.uscita, png, y0=-64, y1=220, ombreggiatura=1.0)
            from PIL import Image
            self.vista_mondo.imposta_immagine(Image.open(png))
            self.schede.setCurrentIndex(2)
        except Exception:
            # l'anteprima e' un di piu': se non riesce, il mondo c'e' lo stesso
            self.vista_mondo.setText("mondo generato (anteprima non riuscita)")
        QMessageBox.information(
            self, "GenWorld",
            f"Mondo scritto in\n{op.uscita}\n\nPer giocarlo, copia la cartella "
            r"in %APPDATA%\.minecraft\saves\.")

    def closeEvent(self, ev):  # noqa: N802 (API Qt)
        # chiudere la finestra mentre un thread scrive chunk lascerebbe il
        # mondo a meta' e il processo appeso: si chiede lo stop e si aspetta
        if self.lavoro is not None:
            self.lavoro.stop.set()
            self.lavoro.wait(5000)
        super().closeEvent(ev)

    def _errore(self, testo: str) -> None:
        self.lavoro = None
        self._occupata(False)
        self.stato.setText("errore")
        QMessageBox.critical(self, "GenWorld", testo[-2000:])


def main() -> int:
    app = QApplication(sys.argv)
    f = Finestra()
    f.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
