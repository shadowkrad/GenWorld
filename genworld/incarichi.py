"""Incarichi: un committente per citta', con una missione verso un luogo d'avventura.

In ogni citta' vive un abitante con un nome (il committente). Per un smeraldo
vende un libro che racconta l'incarico; chi gli porta l'oggetto della missione
riceve la ricompensa. Tre tipi, scelti dal luogo:

  uccidi   - il capo di una torre, una fortezza o un castello lascia cadere un
             sigillo (nella mano libera, probabilita' 1);
  recupera - una reliquia sta nel forziere del tesoro di una cripta;
  salva    - la lettera di un prigioniero sta nel forziere di una cella del castello.

L'oggetto ha un nome, una descrizione e un'etichetta `custom_data`: e' questa che
il villager riconosce nello scambio, quindi un lingotto d'oro qualunque non vale.

Tutto e' deterministico sul seme: stessa mappa, stessi incarichi. Le coordinate
dei luoghi sono quelle di GIOCO (mappa - meta), come in `luoghi.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import luoghi as LU

NOMI = ("Mastro Ottone", "Dama Berenice", "Frate Anselmo", "Notaio Corvino", "Ser Lamberto",
        "Madonna Isotta", "Mastro Gervasio", "Cancelliere Orsino", "Dama Fiammetta",
        "Podesta' Tebaldo", "Scriba Almerico", "Madonna Clotilde")

# id dell'oggetto, titolo del libro, come si chiama l'oggetto
OGGETTO = {
    "uccidi": ("gold_ingot", "Sigillo di {meta}"),
    "recupera": ("amethyst_shard", "Reliquia di {meta}"),
    "salva": ("paper", "Lettera del prigioniero di {meta}"),
}


@dataclass
class Incarico:
    uid: str
    tipo: str                       # uccidi | recupera | salva
    citta: str
    committente: str
    x: int                          # dove sta il committente (gioco)
    z: int
    meta: str                       # nome del luogo
    meta_tipo: str
    mx: int                         # dove sta il luogo (gioco)
    mz: int
    oggetto: dict                   # {"id", "nome", "lore", "uid"}
    ricompensa: tuple               # (id, quanti)
    chiave: tuple = ()              # uccidi: (x, z) di mappa del capo; altri: (x, y, z) di gioco
    pagine: tuple = ()


def _spec(tipo: str, uid: str, meta: str, citta: str) -> dict:
    id_, titolo = OGGETTO[tipo]
    lore = {"uccidi": [f"Strappato al signore di {meta}.", f"Portalo a chi lo attende a {citta}."],
            "recupera": [f"Custodita nelle profondita' di {meta}.", f"La aspettano a {citta}."],
            "salva": [f"Scritta in fretta, nelle celle di {meta}.", f"Va consegnata a {citta}."]}[tipo]
    return {"id": id_, "nome": titolo.format(meta=meta)[:48], "lore": lore, "uid": uid}


def _ricompensa(dist: int, lontano: bool) -> tuple:
    if lontano or dist > 300:
        return ("diamond", 3 + min(5, dist // 150))
    return ("emerald", 12 + min(28, dist // 20))


def _pagine(tipo: str, nome: str, citta: str, l: LU.Luogo, qui: LU.Luogo, ogg: str) -> tuple:
    dove = f"{l.nome}, {LU._passi(qui, l)} da {citta}."
    corpo = {
        "uccidi": (f"Il signore di {l.nome} tiene in ostaggio la nostra pace.",
                   f"Portami '{ogg}': lo porta addosso. Ti paghero' bene."),
        "recupera": (f"In fondo alle cripte di {l.nome} giace qualcosa che ci appartiene.",
                     f"Riportami '{ogg}' dal forziere piu' profondo."),
        "salva": (f"Un nostro uomo marcisce in una cella di {l.nome}.",
                  f"Cercalo e portami '{ogg}': la sua lettera dira' dove e' stato venduto."),
    }[tipo]
    return (f"{nome}, {citta}.", f"Il luogo e' {dove}", *corpo, "La ricompensa e' tua alla consegna.")


def pianifica(citta: list, luoghi: list, strutture: dict, meta: int,
              seed: int = 0) -> list[Incarico]:
    """`citta`: oggetti Citta (x, z di mappa); `luoghi`: la lista di `LU.raccogli`;
    `strutture`: {(tipo, x, z): (chiave_capo | None, forziere | None)} dove tipo/x/z sono
    quelli del Luogo (coordinate di gioco). Dà a ogni citta' il luogo d'avventura
    libero piu' vicino; se i luoghi finiscono, ne riusa il piu' vicino senza trofeo."""
    rng = np.random.default_rng(seed + 5454)
    mete = [l for l in luoghi if l.tipo in LU.TIPI_AVVENTURA and (l.tipo, l.x, l.z) in strutture]
    cl = {(l.x, l.z): l for l in luoghi if l.tipo == "citta"}
    usate: set = set()
    nomi_usati: set = set()
    fuori: list[Incarico] = []
    for c in citta:
        qui = cl.get((c.x - meta, c.z - meta))
        if qui is None or not mete:
            continue
        liberi = [l for l in mete if (l.tipo, l.x, l.z) not in usate]
        if not liberi:
            break
        l = min(liberi, key=lambda m: LU.distanza(qui, m))
        usate.add((l.tipo, l.x, l.z))
        capo, forziere = strutture[(l.tipo, l.x, l.z)]
        if l.tipo == "rovina":
            tipo = "recupera"
        elif l.tipo == "castello" and forziere is not None and rng.random() < 0.5:
            tipo = "salva"
        else:
            tipo = "uccidi"
        dist = LU.distanza(qui, l)
        uid = f"inc{len(fuori) + 1}_{abs(c.x * 31 + c.z) % 9973}"
        spec = _spec(tipo, uid, l.nome, qui.nome)
        scelte = [n for n in NOMI if n not in nomi_usati] or list(NOMI)
        nome = scelte[int(rng.integers(len(scelte)))]
        nomi_usati.add(nome)
        if tipo == "uccidi":
            chiave = capo
        else:
            chiave = forziere
        fuori.append(Incarico(
            uid=uid, tipo=tipo, citta=qui.nome, committente=nome,
            x=0, z=0, meta=l.nome, meta_tipo=l.tipo, mx=l.x, mz=l.z, oggetto=spec,
            ricompensa=_ricompensa(dist, l.tipo in ("castello", "fortezza")),
            chiave=tuple(chiave) if chiave else (),
            pagine=_pagine(tipo, nome, qui.nome, l, qui, spec["nome"])))
        fuori[-1].x, fuori[-1].z = qui.x, qui.z
    return fuori


def trofei(incarichi: list[Incarico]) -> dict:
    """{(x, z) di MAPPA del capo: oggetto}: per `combattenti(..., trofei=)`."""
    return {i.chiave: i.oggetto for i in incarichi if i.tipo == "uccidi" and len(i.chiave) == 2}


def oggetti_forziere(incarichi: list[Incarico]) -> dict:
    """{(x, y, z) di GIOCO del forziere: [oggetto]}: per `bauli.trova(..., oggetti=)`."""
    fuori: dict = {}
    for i in incarichi:
        if i.tipo != "uccidi" and len(i.chiave) == 3:
            fuori.setdefault(i.chiave, []).append(i.oggetto)
    return fuori


def offerte(i: Incarico) -> tuple:
    """I due scambi del committente: il libro e la consegna."""
    libro = {"compra": ("emerald", 1),
             "vende": ("written_book", 1, {"written_book_content": {
                 "title": {"raw": f"Incarico: {i.meta}"[:32]}, "author": i.committente,
                 "pages": [{"raw": p} for p in i.pagine]}}),
             "usi": 1}
    consegna = {"compra": (i.oggetto["id"], 1, {"custom_data": {"incarico": i.uid}}),
                "vende": (i.ricompensa[0], i.ricompensa[1], None),
                "usi": 1}
    return (libro, consegna)


def abitanti(incarichi: list[Incarico], strade, altezze, citta: list, meta: int, seed: int = 0) -> list:
    """Un committente per incarico, su una strada fra 4 e 9 blocchi dalla piazza.
    `strade`: array (tipo_strada) o None; `altezze`: h. Coordinate di mappa."""
    from .entita import Abitante
    fuori = []
    if strade is None:
        return fuori
    H, W = strade.shape
    for n, i in enumerate(incarichi):
        c = next((c for c in citta if c.x - meta == i.x and c.z - meta == i.z), None)
        if c is None:
            continue
        pz, px = c.piazza
        z0, z1, x0, x1 = max(0, pz - 9), min(H, pz + 10), max(0, px - 9), min(W, px + 10)
        zz, xx = np.nonzero(strade[z0:z1, x0:x1] > 0)
        if zz.size == 0:
            continue
        d = (zz + z0 - pz) ** 2 + (xx + x0 - px) ** 2
        ok = np.nonzero((d >= 16) & (d <= 81))[0]
        k = int(ok[len(ok) // 2]) if ok.size else int(np.argmin(d))
        z, x = int(zz[k]) + z0, int(xx[k]) + x0
        fuori.append(Abitante(x=x + 0.5, y=float(altezze[z, x]), z=z + 0.5, mestiere="cartografo",
                              livello=5, seme=(seed * 7919 + n * 104729 + 17) & 0x7FFFFFFF, nome=i.committente,
                              offerte=offerte(i)))
    return fuori


def sezione_elenco(incarichi: list[Incarico]) -> str:
    """Il testo per `luoghi.md`: per chi vuole sapere cosa c'e' da fare."""
    if not incarichi:
        return ""
    righe = ["", "## Incarichi", ""]
    for i in incarichi:
        righe.append(f"- **{i.committente}** ({i.citta}, {i.x}, {i.z}): {i.tipo} a {i.meta} "
                     f"({i.mx}, {i.mz}) - oggetto «{i.oggetto['nome']}», "
                     f"ricompensa {i.ricompensa[1]} {i.ricompensa[0]}")
    return "\n".join(righe) + "\n"
