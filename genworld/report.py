"""Rendering testuale del pannello di fedelta'.

E' l'anteprima che in futuro stara' sotto i cursori della GUI, aggiornata in
tempo reale mentre si trascina. Qui vive come testo perche' i conti sono gia'
tutti veri e si possono guardare senza aspettare una finestra Qt.
"""

from __future__ import annotations

import sys

from .fidelity import Stima
from .scale import _fmt_m, _num

LARGHEZZA = 80


def _supporta_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


def _barra(punteggio: float, celle: int = 10, unicode_ok: bool | None = None) -> str:
    if unicode_ok is None:
        unicode_ok = _supporta_unicode()
    pieno, vuoto = ("#", ".") if not unicode_ok else ("█", "░")
    n = int(round(max(0.0, min(1.0, punteggio)) * celle))
    return pieno * n + vuoto * (celle - n)


def pannello(stima: Stima, mostra_avvisi: bool = True) -> str:
    s = stima.scala
    t = s.territorio
    u = _supporta_unicode()
    riga = ("─" if u else "-") * LARGHEZZA

    out: list[str] = []
    out.append(riga)
    out.append(f"  {t.nome.upper()}")
    out.append(
        f"  Mondo {s.lato_mondo_blocchi} x {s.lato_mondo_blocchi} blocchi"
        f"   |   1 blocco = {_fmt_m(s.metri_per_blocco)}"
    )
    out.append(
        f"  Territorio {_fmt_m(t.larghezza_m)} x {_fmt_m(t.altezza_m)}"
        f"   |   {_num(t.area_km2)} km2"
    )
    out.append(
        f"  Verticale: 1 blocco = {_fmt_m(s.metri_per_blocco_verticale)}"
        f"   |   esagerazione {s.esagerazione_verticale:.1f}x"
        f"   |   tetto {_fmt_m(s.quota_max_rappresentabile_m)}"
    )
    out.append(riga)

    for d in stima.dimensioni:
        nome = d.nome[:24].ljust(24)
        valore = d.valore[:26].ljust(26)
        out.append(f"  {nome}{valore}{_barra(d.punteggio, unicode_ok=u)}  {d.giudizio}")
        if d.nota:
            out.append(f"  {'':24}{d.nota}")

    out.append(riga)
    out.append(f"  REGIME: {stima.regime}")
    out.append(f"  Indice sintetico {stima.indice * 100:.0f}/100  (solo per confrontare alternative)")

    if mostra_avvisi and stima.avvisi:
        out.append(riga)
        out.append("  AVVISI")
        for a in stima.avvisi:
            out.extend(_avvolgi(a, indent="   - "))

    out.append(riga)
    return "\n".join(out)


def _avvolgi(testo: str, indent: str = "  ", larghezza: int = LARGHEZZA) -> list[str]:
    parole = testo.split()
    righe: list[str] = []
    corrente = indent
    for p in parole:
        if len(corrente) + len(p) + 1 > larghezza:
            righe.append(corrente.rstrip())
            corrente = " " * len(indent) + p + " "
        else:
            corrente += p + " "
    if corrente.strip():
        righe.append(corrente.rstrip())
    return righe


def tabella_confronto(stime: list[Stima]) -> str:
    """Confronto fra piu' dimensioni di mondo per lo stesso territorio."""
    u = _supporta_unicode()
    riga = ("─" if u else "-") * LARGHEZZA
    out = [riga]
    out.append(
        "  " + "lato".rjust(6) + "m/blocco".rjust(12) + "esag.".rjust(9)
        + "forma".rjust(11) + "insed.".rjust(11) + "  regime"
    )
    out.append(riga)
    for st in stime:
        s = st.scala
        d = {x.nome: x for x in st.dimensioni}
        forma = d["Forma (coste, confini)"].valore.replace("+/- ", "")
        insed = d["Insediamenti"].valore.split(" (")[0]
        out.append(
            "  "
            + str(s.lato_mondo_blocchi).rjust(6)
            + _fmt_m(s.metri_per_blocco).rjust(12)
            + f"{s.fattore_esagerazione:.0f}x".rjust(9)
            + forma.rjust(11)
            + insed.rjust(11)
            + "  " + st.regime
        )
    out.append(riga)
    return "\n".join(out)
