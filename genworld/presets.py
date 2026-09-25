"""Territori, insediamenti e monumenti di esempio.

Servono a provare lo stimatore senza avere ancora un importer di dati reali.
Le quote sono in metri sul livello del mare, le estensioni sono bounding box.
"""

from __future__ import annotations

from .scale import Insediamento, Landmark, Territorio

# --------------------------------------------------------------------------
# Territori
# --------------------------------------------------------------------------

ITALIA = Territorio(
    nome="Italia",
    larghezza_m=1_000_000,     # ~1000 km E-O
    altezza_m=1_290_000,       # ~1290 km N-S
    quota_max_m=4808,          # Monte Bianco
    quota_min_m=0,
)

GARDA = Territorio(
    nome="Lago di Garda",
    larghezza_m=20_000,
    altezza_m=20_000,
    quota_max_m=2218,          # Monte Baldo
    quota_min_m=65,            # livello del lago
)

VAL_ORCIA = Territorio(
    nome="Val d'Orcia",
    larghezza_m=15_000,
    altezza_m=15_000,
    quota_max_m=821,
    quota_min_m=200,
)

ETNA = Territorio(
    nome="Etna",
    larghezza_m=45_000,
    altezza_m=45_000,
    quota_max_m=3357,
    quota_min_m=0,
)


# --------------------------------------------------------------------------
# Monumenti
# --------------------------------------------------------------------------

MONUMENTI = {
    "torre_eiffel": Landmark("Torre Eiffel", 330, 125),
    "colosseo": Landmark("Colosseo", 48, 189),
    "duomo_milano": Landmark("Duomo di Milano", 108, 158),
    "torre_pisa": Landmark("Torre di Pisa", 57, 15),
    "mole": Landmark("Mole Antonelliana", 167, 50),
    "burj_khalifa": Landmark("Burj Khalifa", 828, 80),
}


# --------------------------------------------------------------------------
# Insediamenti
# --------------------------------------------------------------------------

def insediamenti_italia() -> list[Insediamento]:
    """Approssimazione della gerarchia urbana italiana: 143 elementi."""
    out: list[Insediamento] = []

    for nome in ("Roma", "Milano", "Napoli"):
        out.append(Insediamento(nome, "metropoli", importanza=1.0))

    citta = (
        "Torino", "Palermo", "Genova", "Bologna", "Firenze", "Bari",
        "Catania", "Venezia", "Verona", "Messina", "Padova", "Trieste",
    )
    for nome in citta:
        out.append(Insediamento(nome, "citta", importanza=0.8))

    for i in range(40):
        out.append(Insediamento(f"Paese {i + 1}", "paese", importanza=0.5))

    for i in range(88):
        out.append(Insediamento(f"Villaggio {i + 1}", "villaggio", importanza=0.2))

    return out


def insediamenti_garda() -> list[Insediamento]:
    """I centri rivieraschi del basso e medio Garda."""
    paesi = (
        "Sirmione", "Desenzano", "Peschiera", "Bardolino", "Garda",
        "Lazise", "Salo", "Gargnano", "Malcesine", "Limone", "Riva", "Torri",
    )
    out = [Insediamento(n, "paese", importanza=0.7) for n in paesi]
    out += [Insediamento(f"Borgo {i + 1}", "villaggio", importanza=0.3) for i in range(18)]
    return out


TERRITORI = {
    "italia": (ITALIA, insediamenti_italia),
    "garda": (GARDA, insediamenti_garda),
    "valorcia": (VAL_ORCIA, lambda: [
        Insediamento(n, "villaggio", importanza=0.6)
        for n in ("Pienza", "Montalcino", "San Quirico", "Castiglione d'Orcia",
                  "Radicofani", "Bagno Vignoni")
    ]),
    "etna": (ETNA, lambda: [
        Insediamento("Catania", "citta", importanza=1.0),
        Insediamento("Acireale", "paese", importanza=0.7),
        Insediamento("Zafferana", "paese", importanza=0.6),
        Insediamento("Nicolosi", "villaggio", importanza=0.6),
        Insediamento("Linguaglossa", "villaggio", importanza=0.5),
        Insediamento("Randazzo", "villaggio", importanza=0.5),
    ]),
}
