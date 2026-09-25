"""Modello delle scale.

Principio fondante di GenWorld: la scala del TERRENO e la scala degli OGGETTI
sono indipendenti.

Il terreno viene compresso per far stare un territorio grande in pochi blocchi.
Gli oggetti (case, ponti, palafitte, monumenti) restano a dimensione giocabile:
se comprimo il terreno 100 volte, la Torre Eiffel non diventa alta 3 blocchi.

Il rapporto fra le due scale e' il "fattore di esagerazione", ed e' il numero da
cui discende tutta la generalizzazione cartografica a valle.

Questo modulo non genera niente: fa solo conti. E' volutamente senza dipendenze.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Limiti del formato Minecraft Java Edition (dalla 1.18 in poi)
# --------------------------------------------------------------------------
Y_MIN = -64
Y_MAX = 320
ALTEZZA_TOTALE = Y_MAX - Y_MIN          # 384 blocchi
LIVELLO_MARE_DEFAULT = 62


# --------------------------------------------------------------------------
# Ingombro a scala oggetti (blocchi di lato) per tipo di insediamento.
# Sono dimensioni GIOCABILI: un villaggio in cui cammini e' ~96 blocchi.
# --------------------------------------------------------------------------
LATO_BLOCCHI_PER_TIPO = {
    "villaggio": 96,
    "paese": 192,
    "citta": 384,
    "metropoli": 640,
}


@dataclass(frozen=True)
class Territorio:
    """L'area reale da riprodurre, in metri."""

    nome: str
    larghezza_m: float
    altezza_m: float
    quota_max_m: float = 0.0
    quota_min_m: float = 0.0

    @property
    def lato_max_m(self) -> float:
        return max(self.larghezza_m, self.altezza_m)

    @property
    def area_km2(self) -> float:
        return self.larghezza_m * self.altezza_m / 1e6

    @property
    def dislivello_m(self) -> float:
        return max(1.0, self.quota_max_m - self.quota_min_m)


@dataclass(frozen=True)
class Landmark:
    """Un monumento o edificio notevole che deve restare riconoscibile."""

    nome: str
    altezza_m: float
    larghezza_m: float = 0.0


@dataclass(frozen=True)
class Insediamento:
    """Una citta' o un villaggio candidato a comparire nel mondo."""

    nome: str
    tipo: str = "villaggio"     # villaggio | paese | citta | metropoli
    importanza: float = 0.5     # 0..1, guida la selezione quando lo spazio manca

    @property
    def lato_blocchi(self) -> int:
        return LATO_BLOCCHI_PER_TIPO.get(self.tipo, 96)

    @property
    def area_blocchi(self) -> int:
        return self.lato_blocchi ** 2


@dataclass
class Scala:
    """Configurazione di scala di un mondo.

    L'utente fissa il budget (`lato_mondo_blocchi`) e il territorio; tutto il
    resto e' derivato. Gli oggetti restano a `scala_oggetti_m_per_blocco`,
    che di norma vale 1.0 e non va toccata.
    """

    lato_mondo_blocchi: int
    territorio: Territorio
    scala_oggetti_m_per_blocco: float = 1.0
    esagerazione_verticale: float = 1.0
    livello_mare_y: int = LIVELLO_MARE_DEFAULT

    # -- scala orizzontale --------------------------------------------------

    @property
    def metri_per_blocco(self) -> float:
        """Metri reali rappresentati da un blocco in orizzontale."""
        return self.territorio.lato_max_m / self.lato_mondo_blocchi

    @property
    def fattore_esagerazione(self) -> float:
        """Di quanto gli oggetti sono fuori scala rispetto al terreno."""
        return self.metri_per_blocco / self.scala_oggetti_m_per_blocco

    # -- scala verticale ----------------------------------------------------

    @property
    def metri_per_blocco_verticale(self) -> float:
        return self.metri_per_blocco / self.esagerazione_verticale

    @property
    def blocchi_sopra_mare(self) -> int:
        return Y_MAX - self.livello_mare_y

    @property
    def quota_max_rappresentabile_m(self) -> float:
        """Oltre questa quota il rilievo viene tagliato."""
        return self.blocchi_sopra_mare * self.metri_per_blocco_verticale

    def esagerazione_verticale_ottimale(self) -> float:
        """Esagerazione che usa tutto il budget verticale senza tagliare nulla.

        Ricavata da:  quota_max / (mpb_orizzontale / esag) = blocchi_sopra_mare
        """
        if self.territorio.quota_max_m <= 0:
            return 1.0
        return (
            self.blocchi_sopra_mare
            * self.metri_per_blocco
            / self.territorio.quota_max_m
        )

    def con_esagerazione_ottimale(self) -> "Scala":
        return Scala(
            lato_mondo_blocchi=self.lato_mondo_blocchi,
            territorio=self.territorio,
            scala_oggetti_m_per_blocco=self.scala_oggetti_m_per_blocco,
            esagerazione_verticale=self.esagerazione_verticale_ottimale(),
            livello_mare_y=self.livello_mare_y,
        )

    # -- regime di rappresentazione ----------------------------------------

    @property
    def regime(self) -> str:
        f = self.fattore_esagerazione
        if f < 5:
            return "riproduzione realistica"
        if f < 20:
            return "mappa illustrata"
        if f < 100:
            return "plastico con monumenti"
        return "caricatura"

    # -- landmark -----------------------------------------------------------

    def resa_landmark(self, lm: Landmark) -> dict:
        """Quanto di un monumento riusciamo a costruire nel budget verticale."""
        ideali = lm.altezza_m / self.scala_oggetti_m_per_blocco
        disponibili = float(self.blocchi_sopra_mare)
        if ideali <= disponibili:
            return {
                "nome": lm.nome,
                "blocchi": int(round(ideali)),
                "blocchi_ideali": int(round(ideali)),
                "fedelta": 1.0,
                "compresso": False,
            }
        return {
            "nome": lm.nome,
            "blocchi": int(disponibili),
            "blocchi_ideali": int(round(ideali)),
            "fedelta": disponibili / ideali,
            "compresso": True,
        }

    # -- riepilogo ----------------------------------------------------------

    def riepilogo(self) -> str:
        t = self.territorio
        return (
            f"Mondo: {self.lato_mondo_blocchi} x {self.lato_mondo_blocchi} blocchi"
            f"  |  1 blocco = {_fmt_m(self.metri_per_blocco)}"
            f"  |  territorio: {_fmt_m(t.larghezza_m)} x {_fmt_m(t.altezza_m)}"
        )


# --------------------------------------------------------------------------
# I tre modi di lavorare: fissi il mondo, fissi la scala, oppure fissi la
# fedelta' e ti fai dire quanto mondo serve.
# --------------------------------------------------------------------------

def mondo_per_scala(territorio: Territorio, metri_per_blocco: float) -> int:
    """Fisso la scala -> quanto viene grande il mondo."""
    return int(math.ceil(territorio.lato_max_m / metri_per_blocco))


def mondo_per_errore(
    territorio: Territorio, errore_max_m: float, tolleranza_blocchi: float = 0.0
) -> int:
    """Fisso la fedelta' geometrica -> lato minimo del mondo.

    L'errore di quantizzazione e' mezzo blocco, piu' l'eventuale tolleranza di
    semplificazione espressa in blocchi.
    """
    if errore_max_m <= 0:
        raise ValueError("errore_max_m deve essere positivo")
    mpb = errore_max_m / (0.5 + tolleranza_blocchi)
    return mondo_per_scala(territorio, mpb)


def _num(x: float, decimali: int = 0) -> str:
    """Numero con convenzione italiana: punto per le migliaia, virgola per i decimali."""
    s = f"{x:,.{decimali}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _fmt_m(m: float) -> str:
    """Formatta una distanza in metri scegliendo l'unita' leggibile."""
    if m >= 100_000:
        return f"{_num(m / 1000)} km"
    if m >= 1000:
        return f"{_num(m / 1000, 1)} km"
    if m >= 1:
        return f"{_num(m)} m"
    return f"{_num(m * 100)} cm"
