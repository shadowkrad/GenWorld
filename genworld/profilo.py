"""Profilo di lettura di una mappa: rettangolo di interesse e campioni.

Nasce da una misura: su cinque mappe reali la classificazione automatica
funziona su due e fallisce su due. I modi di fallire non sono sottigliezze di
taratura, sono cose che una persona vede in un secondo e un algoritmo no - una
cornice illustrata, un riquadro di legenda, un mare dipinto dello stesso colore
della terra.

Quindi si smette di indovinare e si chiedono due gesti:

  1. un RETTANGOLO attorno alla mappa vera, che elimina in un colpo cornici,
     legende, cartigli e rose dei venti;
  2. qualche CLIC col contagocce che dice "questo e' mare, questo e' montagna,
     questo e' decorazione da ignorare".

Il profilo si salva e si riusa: vale per tutte le mappe dello stesso stile.

Le coordinate sono normalizzate 0..1 sull'immagine ORIGINALE, così il profilo
resta valido a qualunque risoluzione la si rilegga.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import mappa as M
from .classi_guidate import DECORO, classifica_guidata
from .normalizza import a_lab

NOME_CLASSE = {
    "oceano": M.OCEANO, "mare": M.MARE, "spiaggia": M.SPIAGGIA,
    "deserto": M.DESERTO, "pianura": M.PIANURA, "prateria": M.PRATERIA,
    "foresta": M.FORESTA, "montagna": M.MONTAGNA, "neve": M.NEVE,
    "decoro": DECORO,
}
CLASSE_NOME = {v: k for k, v in NOME_CLASSE.items()}


@dataclass
class Profilo:
    """Come leggere una mappa. Serializzabile in JSON."""

    nome: str = "senza nome"
    # rettangolo di interesse in coordinate normalizzate: x0, y0, x1, y1
    roi: tuple[float, float, float, float] | None = None
    # classe -> elenco di punti (x, y) normalizzati sull'immagine originale
    campioni: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    # regolarizzazione spaziale: finestra del filtro di mediana, in pixel
    pulizia: int = 11
    soglia_montagna: float = 0.30
    # quota per classe, in blocchi; None usa i valori predefiniti
    quote: dict[int, float] | None = None
    # maschera dipinta a mano: {"w":.., "h":.., "rle": base64}. Ha l'ultima
    # parola su terra e acqua, in coordinate dell'immagine ORIGINALE.
    maschera: dict | None = None

    # -- serializzazione ----------------------------------------------------

    def a_dizionario(self) -> dict:
        return {
            "nome": self.nome,
            "roi": list(self.roi) if self.roi else None,
            "campioni": {CLASSE_NOME[k]: [list(p) for p in v]
                         for k, v in self.campioni.items() if v},
            "pulizia": self.pulizia,
            "soglia_montagna": self.soglia_montagna,
            "quote": ({CLASSE_NOME[k]: v for k, v in self.quote.items()}
                      if self.quote else None),
            "maschera": self.maschera,
        }

    @classmethod
    def da_dizionario(cls, d: dict) -> "Profilo":
        campioni = {}
        for nome, punti in (d.get("campioni") or {}).items():
            chiave = NOME_CLASSE.get(nome.strip().lower())
            if chiave is None:
                raise ValueError(f"classe sconosciuta nel profilo: {nome!r}")
            campioni[chiave] = [tuple(p) for p in punti]
        roi = d.get("roi")
        quote = d.get("quote")
        return cls(
            nome=d.get("nome", "senza nome"),
            roi=tuple(roi) if roi else None,
            campioni=campioni,
            pulizia=int(d.get("pulizia", 11)),
            soglia_montagna=float(d.get("soglia_montagna", 0.30)),
            quote=({NOME_CLASSE[k]: float(v) for k, v in quote.items()}
                   if quote else None),
            maschera=d.get("maschera") or None,
        )

    def salva(self, percorso: str) -> None:
        with open(percorso, "w", encoding="utf-8") as f:
            json.dump(self.a_dizionario(), f, indent=2, ensure_ascii=False)

    @classmethod
    def carica(cls, percorso: str) -> "Profilo":
        with open(percorso, encoding="utf-8") as f:
            return cls.da_dizionario(json.load(f))

    # -- applicazione -------------------------------------------------------

    def n_campioni(self) -> int:
        return sum(len(v) for v in self.campioni.values())

    def leggi_immagine(self, percorso: str, lato: int) -> np.ndarray:
        """Carica l'immagine applicando il rettangolo di interesse.

        Come `mappa.carica()`: il ridimensionamento mantiene le proporzioni
        (il ROI puo' non essere quadrato quanto il mondo) invece di stirare
        l'immagine a incastro - l'array tornato puo' quindi non essere
        quadrato."""
        im = Image.open(percorso).convert("RGB")
        if self.roi:
            w, h = im.size
            x0, y0, x1, y1 = self.roi
            im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        w, h = im.size
        scala = lato / max(w, h)
        nw, nh = max(1, round(w * scala)), max(1, round(h * scala))
        im = im.resize((nw, nh), Image.LANCZOS)
        return np.asarray(im).astype(np.float32) / 255.0

    def _campioni_nel_ritaglio(self) -> dict[int, list[tuple[float, float]]]:
        """Riporta i punti dalle coordinate dell'originale a quelle del ritaglio."""
        if not self.roi:
            return self.campioni
        x0, y0, x1, y1 = self.roi
        lx, ly = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        fuori = {}
        for k, punti in self.campioni.items():
            dentro = [((x - x0) / lx, (y - y0) / ly) for x, y in punti
                      if x0 <= x <= x1 and y0 <= y <= y1]
            if dentro:
                fuori[k] = dentro
        return fuori

    def classifica(self, rgb: np.ndarray, rug: np.ndarray | None = None) -> np.ndarray:
        """Classifica un'immagine GIA' ritagliata secondo questo profilo."""
        from .classi_guidate import campiona
        punti = self._campioni_nel_ritaglio()
        if not punti:
            raise ValueError("il profilo non ha campioni dentro il rettangolo")
        if rug is None:
            rug = M.rugosita(rgb)
        camp = campiona(rgb, punti)
        cls = classifica_guidata(rgb, camp, rug,
                                 soglia_montagna=self.soglia_montagna,
                                 pulisci=self.pulizia)
        return self.applica_maschera(cls)

    # -- maschera dipinta ---------------------------------------------------

    def maschera_dipinta(self, forma: tuple[int, int]) -> np.ndarray | None:
        """La maschera dell'utente, ritagliata sul rettangolo e riscalata.

        Valori: 0 nessuna indicazione, 1 acqua, 2 terra.
        """
        if not self.maschera:
            return None
        m = _decodifica_rle(self.maschera)
        if self.roi:
            h, w = m.shape
            x0, y0, x1, y1 = self.roi
            m = m[int(y0 * h): max(int(y1 * h), int(y0 * h) + 1),
                  int(x0 * w): max(int(x1 * w), int(x0 * w) + 1)]
        if m.size == 0:
            return None
        return np.asarray(
            Image.fromarray(m, "L").resize((forma[1], forma[0]), Image.NEAREST)
        )

    def applica_maschera(self, cls: np.ndarray) -> np.ndarray:
        """Il pennello ha l'ultima parola su terra e acqua.

        Non sovrascrive la CLASSE, solo l'appartenenza: dove l'utente ha detto
        "acqua" e il colore diceva terra si mette mare, e viceversa pianura.
        Dove i due sono d'accordo resta il dettaglio della classificazione.
        """
        m = self.maschera_dipinta(cls.shape)
        if m is None:
            return cls
        acqua = np.isin(cls, M.ACQUA)
        cls = cls.copy()
        cls[(m == 1) & ~acqua] = M.MARE
        cls[(m == 2) & acqua] = M.PIANURA
        return cls

    def quote_effettive(self) -> M.Quote:
        q = M.Quote()
        if self.quote:
            q.base.update(self.quote)
        return q


def _decodifica_rle(d: dict) -> np.ndarray:
    """Maschera compressa a lunghezze di corsa: terzetti (valore, conta lo, hi).

    Formato scelto perche' e' banale da produrre in JavaScript e da leggere
    qui, e una maschera 192x192 a tre valori sta in pochi kilobyte di base64
    dentro il JSON del profilo, senza file a parte da tenere allineati.
    """
    w, h = int(d["w"]), int(d["h"])
    grezzo = base64.b64decode(d["rle"])
    out = np.zeros(w * h, dtype=np.uint8)
    pos = 0
    for i in range(0, len(grezzo) - 2, 3):
        n = grezzo[i + 1] | (grezzo[i + 2] << 8)
        if n == 0:
            continue
        fine = min(pos + n, out.size)
        out[pos:fine] = grezzo[i]
        pos = fine
        if pos >= out.size:
            break
    return out.reshape(h, w)


def codifica_rle(m: np.ndarray) -> dict:
    """Inverso di _decodifica_rle, per i test e per gli strumenti da riga."""
    h, w = m.shape
    piatto = m.ravel().astype(np.uint8)
    fuori = bytearray()
    i = 0
    while i < piatto.size:
        v = piatto[i]
        j = i
        while j < piatto.size and piatto[j] == v and j - i < 65535:
            j += 1
        n = j - i
        fuori += bytes((int(v), n & 0xFF, (n >> 8) & 0xFF))
        i = j
    return {"w": w, "h": h, "rle": base64.b64encode(bytes(fuori)).decode()}


def maschera_decoro(cls: np.ndarray) -> np.ndarray:
    """Dove non c'e' mappa. Va trattata come acqua profonda o lasciata vuota."""
    return cls == DECORO
