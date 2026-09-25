"""Stimatore di fedelta'.

Prende una configurazione di scala e dichiara IN ANTICIPO quanto il mondo
generato potra' assomigliare alla mappa di partenza.

Scelta di progetto: niente voto unico. Un "affidabilita' 34%" non dice
all'utente cosa sta perdendo. Si riporta una riga per dimensione, in unita'
che una persona capisce ("la costa entro +/- 350 m", "9 villaggi su 143").
L'indice sintetico esiste solo per ordinare le alternative, ed e' dichiarato
come tale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .scale import Insediamento, Landmark, Scala, _fmt_m


# --------------------------------------------------------------------------
# Quanta parte del mondo puo' essere coperta da insediamenti prima che la
# mappa smetta di sembrare un territorio e diventi un agglomerato unico.
# --------------------------------------------------------------------------
COPERTURA_MAX_INSEDIAMENTI = 0.25


@dataclass
class Dimensione:
    """Una riga del pannello di fedelta'."""

    nome: str
    valore: str          # leggibile da un umano
    punteggio: float     # 0..1, solo per la barra
    nota: str = ""

    @property
    def giudizio(self) -> str:
        p = self.punteggio
        if p >= 0.85:
            return "ottima"
        if p >= 0.65:
            return "buona"
        if p >= 0.45:
            return "media"
        if p >= 0.25:
            return "ridotta"
        return "molto ridotta"


@dataclass
class Stima:
    scala: Scala
    dimensioni: list[Dimensione]
    regime: str
    indice: float
    avvisi: list[str] = field(default_factory=list)
    insediamenti_tenuti: list[Insediamento] = field(default_factory=list)
    landmark: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# Punteggi
# --------------------------------------------------------------------------

def _punteggio_log(valore: float, ottimo: float, pessimo: float) -> float:
    """Mappa un valore su 0..1 con andamento logaritmico.

    `ottimo` -> 1.0, `pessimo` -> 0.0. Funziona sia crescente che decrescente.
    """
    if valore <= min(ottimo, pessimo):
        return 1.0 if ottimo < pessimo else 0.0
    if valore >= max(ottimo, pessimo):
        return 0.0 if ottimo < pessimo else 1.0
    return 1.0 - math.log10(valore / ottimo) / math.log10(pessimo / ottimo)


def _fedelta_geometrica(scala: Scala, tolleranza_blocchi: float) -> Dimensione:
    """Quanto resta fedele la FORMA (coste, confini, tracciati).

    Nota non ovvia: l'errore relativo all'estensione vale sempre
    (0.5 + tolleranza) / lato_mondo_blocchi, quindi la fedelta' della forma
    dipende SOLO dalla risoluzione in blocchi, non da quanto e' grande il
    territorio. Riprodurre l'Italia o un lago a 1000 blocchi da' la stessa
    fedelta' di forma; cambia l'errore in metri, non la riconoscibilita'.
    """
    errore_m = (0.5 + tolleranza_blocchi) * scala.metri_per_blocco
    relativo = (0.5 + tolleranza_blocchi) / scala.lato_mondo_blocchi
    punteggio = _punteggio_log(relativo, ottimo=0.0005, pessimo=0.02)
    return Dimensione(
        nome="Forma (coste, confini)",
        valore=f"+/- {_fmt_m(errore_m)}",
        punteggio=punteggio,
        nota=f"{relativo * 100:.2f}% dell'estensione",
    )


def _fedelta_altimetrica(scala: Scala) -> Dimensione:
    errore_m = scala.metri_per_blocco_verticale / 2
    relativo = errore_m / scala.territorio.dislivello_m
    punteggio = _punteggio_log(relativo, ottimo=0.001, pessimo=0.05)
    esag = scala.esagerazione_verticale
    nota = f"esagerazione verticale {esag:.1f}x" if esag != 1 else "nessuna esagerazione"
    return Dimensione(
        nome="Quote (rilievo)",
        valore=f"+/- {_fmt_m(errore_m)}",
        punteggio=punteggio,
        nota=nota,
    )


def _seleziona_insediamenti(
    scala: Scala, insediamenti: Sequence[Insediamento], copertura_max: float
) -> tuple[list[Insediamento], Dimensione]:
    """Operatore di SELEZIONE: quali insediamenti stanno nel budget di area.

    Gli insediamenti sono costruiti a scala oggetti, quindi occupano l'area
    giocabile a prescindere da quanto e' compresso il terreno. E' qui che il
    fattore di esagerazione si fa sentire davvero.
    """
    if not insediamenti:
        return [], Dimensione("Insediamenti", "nessuno indicato", 1.0)

    area_mondo = scala.lato_mondo_blocchi ** 2
    budget = area_mondo * copertura_max

    ordinati = sorted(insediamenti, key=lambda i: (-i.importanza, i.lato_blocchi))
    tenuti: list[Insediamento] = []
    usata = 0.0
    for ins in ordinati:
        if usata + ins.area_blocchi <= budget:
            tenuti.append(ins)
            usata += ins.area_blocchi

    quota = len(tenuti) / len(insediamenti)
    punteggio = _punteggio_log(max(quota, 1e-4), ottimo=1.0, pessimo=0.01)
    dim = Dimensione(
        nome="Insediamenti",
        valore=f"{len(tenuti)} su {len(insediamenti)} ({quota * 100:.0f}%)",
        punteggio=punteggio,
        nota=f"occupano il {usata / area_mondo * 100:.0f}% del mondo",
    )
    return tenuti, dim


def _coerenza_scala(scala: Scala) -> Dimensione:
    f = scala.fattore_esagerazione
    punteggio = _punteggio_log(max(f, 1.0), ottimo=2.0, pessimo=200.0)
    return Dimensione(
        nome="Scala degli oggetti",
        valore=f"{f:.0f}x fuori scala" if f >= 2 else "in scala",
        punteggio=punteggio,
        nota=scala.regime,
    )


def _fedelta_landmark(scala: Scala, landmark: Sequence[Landmark]) -> tuple[list[dict], Dimensione | None]:
    if not landmark:
        return [], None
    rese = [scala.resa_landmark(lm) for lm in landmark]
    media = sum(r["fedelta"] for r in rese) / len(rese)
    compressi = [r for r in rese if r["compresso"]]
    valore = f"{media * 100:.0f}% dell'altezza reale"
    nota = (
        f"{len(compressi)} su {len(rese)} compressi dal tetto verticale"
        if compressi
        else "tutti a grandezza naturale"
    )
    return rese, Dimensione("Monumenti", valore, media, nota)


# --------------------------------------------------------------------------
# Stima completa
# --------------------------------------------------------------------------

PESI = {
    "Forma (coste, confini)": 0.25,
    "Quote (rilievo)": 0.20,
    "Insediamenti": 0.25,
    "Scala degli oggetti": 0.20,
    "Monumenti": 0.10,
}


def stima(
    scala: Scala,
    insediamenti: Sequence[Insediamento] = (),
    landmark: Sequence[Landmark] = (),
    fedelta_terreno: float = 1.0,
    copertura_max: float = COPERTURA_MAX_INSEDIAMENTI,
) -> Stima:
    """Calcola il preventivo di fedelta'.

    `fedelta_terreno` e' la levetta 0..1: 1 = nessuna semplificazione oltre la
    quantizzazione, 0 = semplificazione aggressiva (fino a 10 blocchi di
    tolleranza sulle geometrie).
    """
    fedelta_terreno = min(1.0, max(0.0, fedelta_terreno))
    tolleranza_blocchi = (1.0 - fedelta_terreno) * 10.0

    dimensioni = [
        _fedelta_geometrica(scala, tolleranza_blocchi),
        _fedelta_altimetrica(scala),
    ]
    tenuti, dim_ins = _seleziona_insediamenti(scala, insediamenti, copertura_max)
    dimensioni.append(dim_ins)
    dimensioni.append(_coerenza_scala(scala))

    rese, dim_lm = _fedelta_landmark(scala, landmark)
    if dim_lm is not None:
        dimensioni.append(dim_lm)

    peso_tot = sum(PESI.get(d.nome, 0.1) for d in dimensioni)
    indice = sum(d.punteggio * PESI.get(d.nome, 0.1) for d in dimensioni) / peso_tot

    return Stima(
        scala=scala,
        dimensioni=dimensioni,
        regime=scala.regime,
        indice=indice,
        avvisi=_avvisi(scala, insediamenti, tenuti, rese),
        insediamenti_tenuti=tenuti,
        landmark=rese,
    )


def _avvisi(
    scala: Scala,
    insediamenti: Sequence[Insediamento],
    tenuti: Sequence[Insediamento],
    rese: Sequence[dict],
) -> list[str]:
    out: list[str] = []
    t = scala.territorio

    if t.quota_max_m > scala.quota_max_rappresentabile_m:
        out.append(
            f"Il rilievo verra' tagliato sopra {_fmt_m(scala.quota_max_rappresentabile_m)}: "
            f"la quota massima del territorio e' {_fmt_m(t.quota_max_m)}. "
            f"Alza l'esagerazione verticale a {scala.esagerazione_verticale_ottimale():.1f}x "
            f"oppure abbassa il livello del mare."
        )

    if scala.fattore_esagerazione > 100:
        out.append(
            "Oltre 100x di esagerazione la mappa non e' piu' usabile per orientarsi: "
            "gli insediamenti vanno spostati di chilometri per fare spazio agli edifici. "
            "Valuta una modalita' a soli landmark, senza pretesa di mappa continua."
        )

    if insediamenti and len(tenuti) / len(insediamenti) < 0.1:
        out.append(
            f"Sopravvive meno del 10% degli insediamenti ({len(tenuti)}/{len(insediamenti)}): "
            "servira' tipificazione aggressiva, non semplice selezione."
        )

    compressi = [r for r in rese if r["compresso"]]
    for r in compressi:
        out.append(
            f"{r['nome']}: {r['blocchi']} blocchi invece di {r['blocchi_ideali']} "
            f"({r['fedelta'] * 100:.0f}%), limite verticale di Minecraft."
        )

    ottimale = scala.esagerazione_verticale_ottimale()
    if t.quota_max_m > 0 and scala.esagerazione_verticale < ottimale * 0.5:
        out.append(
            f"Il budget verticale e' sottoutilizzato: con esagerazione {ottimale:.1f}x "
            "il rilievo sarebbe molto piu' leggibile."
        )

    return out
