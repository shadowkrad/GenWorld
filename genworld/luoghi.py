"""Luoghi con nome, cartelli, elenco e catena di indizi (#7, #8).

Ogni luogo generato (citta', miniera, rovina, torre, accampamento, cimitero,
portale) ha un nome. I nomi finiscono in tre posti:

* **cartelli** davanti alle rovine, alle torri e alle miniere;
* un **elenco** (`luoghi.md` e `luoghi.json`) accanto al mondo, con le coordinate:
  e' la mappa dell'avventura per chi gioca (e per chi prova il mondo);
* una **catena di indizi**: un libro in una casa della citta' di partenza indica
  la prima rovina o torre, e il tesoro di ognuna contiene un libro che indica la
  successiva. L'ultima chiude la pista.

Tutto in coordinate di GIOCO (la mappa meno mezzo lato).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

CITTA = ("Monteverde", "Valdargento", "Pontealto", "Portobruno", "Roccaserena",
         "Colledoro", "Fontenera", "Borgochiaro", "Villabianca", "Casalpino",
         "Rivabella", "Selvascura", "Campolupo", "Altamarea", "Pietrafonda",
         "Granvalle", "Sorgenera", "Mirabruna", "Cerchiodoro", "Altopasso")
MINIERE = ("Miniera del Corvo", "Miniera dell'Eco", "Miniera della Lanterna",
           "Miniera di Ferrovecchio", "Miniera del Nano Muto", "Miniera Profonda",
           "Miniera dei Tre Pozzi", "Miniera di Pietrabuia", "Miniera del Rame Verde",
           "Miniera dell'Ultima Vena")
ROVINE = ("Cripta di Mortelia", "Cripta dei Sussurri", "Cripta di Orrenda",
          "Tomba del Re Smarrito", "Cripta della Veglia", "Rovine di Altomuro",
          "Cripta dell'Osso", "Sepolcro dei Cento", "Cripta di Valdoscura")
ACCAMPAMENTI = ("Campo dei Predoni", "Accampamento del Lupo", "Campo dei Senzanome",
                "Bivacco dell'Esule", "Campo del Corvo", "Accampamento delle Ceneri")
CIMITERI = ("Cimitero degli Impiccati", "Camposanto dei Dimenticati", "Cimitero di Valdoro",
            "Giardino dei Pianti", "Cimitero del Silenzio")
PORTALI = ("Portale Antico", "Il Varco Spezzato", "Portale delle Ceneri", "Arco del Confine")

TIPI_AVVENTURA = ("rovina", "torre")

# otto direzioni, a partire da est, in senso orario guardando la mappa dall'alto
# (+x est, +z sud)
DIREZIONI = ("est", "sud-est", "sud", "sud-ovest", "ovest", "nord-ovest", "nord", "nord-est")


@dataclass
class Luogo:
    nome: str
    tipo: str          # citta, miniera, rovina, torre, accampamento, cimitero, portale
    x: int             # coordinate di gioco
    z: int
    y: int = 0
    nota: str = ""


@dataclass
class Cartello:
    x: int             # coordinate di MAPPA
    z: int
    rotazione: int     # 0..15 come nel gioco
    righe: tuple


def direzione(dx: float, dz: float) -> str:
    """Il punto cardinale (italiano) di uno spostamento (dx, dz)."""
    ang = math.degrees(math.atan2(dz, dx)) % 360       # 0 = est, 90 = sud
    return DIREZIONI[int(((ang + 22.5) % 360) // 45)]


def distanza(a: Luogo, b: Luogo) -> int:
    return int(round(math.hypot(a.x - b.x, a.z - b.z)))


def _scegli(rng: np.random.Generator, elenco: tuple, usati: set) -> str:
    liberi = [n for n in elenco if n not in usati]
    if liberi:
        nome = liberi[int(rng.integers(0, len(liberi)))]
    else:
        nome = f"{elenco[int(rng.integers(0, len(elenco)))]} {int(rng.integers(2, 9))}"
        while nome in usati:
            nome += "'"
    usati.add(nome)
    return nome


def raccogli(citta: list, miniere: list, dungeon: list, torri: list, avamposti: list,
             meta: int, seed: int = 0) -> list[Luogo]:
    """Dà un nome a ogni luogo e lo mette in coordinate di gioco."""
    rng = np.random.default_rng(seed + 8888)
    usati: set = set()
    fuori: list[Luogo] = []
    for c in citta:
        fuori.append(Luogo(_scegli(rng, CITTA, usati), "citta", c.x - meta, c.z - meta,
                           nota=f"raggio {c.raggio}" + (", murata" if c.murata else "")))
    for mn in miniere:
        fuori.append(Luogo(_scegli(rng, MINIERE, usati), "miniera", mn.x - meta, mn.z - meta,
                           mn.y_superficie))
    for d in dungeon:
        fuori.append(Luogo(_scegli(rng, ROVINE, usati), "rovina", d.x - meta, d.z - meta, d.ys,
                           nota=f"cripta con {d.mob}"))
    for t in torri:
        fuori.append(Luogo(f"Torre di {t.nome}", "torre", t.x - meta, t.z - meta, t.ys,
                           nota=f"capo: {t.nome}"))
    for av in avamposti:
        tab = {"campo": ("accampamento", ACCAMPAMENTI), "cimitero": ("cimitero", CIMITERI),
               "portale": ("portale", PORTALI)}[av.tipo]
        fuori.append(Luogo(_scegli(rng, tab[1], usati), tab[0], av.x - meta, av.z - meta, av.y))
    return fuori


# --------------------------------------------------------------------------
# Elenco
# --------------------------------------------------------------------------

def scrivi_elenco(luoghi: list[Luogo], cartella: str, nome_mondo: str = "") -> tuple[str, str]:
    """Scrive `luoghi.md` e `luoghi.json` in `cartella`. Ritorna i due percorsi."""
    os.makedirs(cartella, exist_ok=True)
    ordine = ("citta", "torre", "rovina", "miniera", "accampamento", "cimitero", "portale")
    titoli = {"citta": "Citta'", "torre": "Torri con un capo", "rovina": "Rovine con cripta",
              "miniera": "Miniere", "accampamento": "Accampamenti", "cimitero": "Cimiteri",
              "portale": "Portali"}
    righe = [f"# Luoghi{(' di ' + nome_mondo) if nome_mondo else ''}", "",
             "Coordinate di gioco (x, z). Per andarci: `/tp @s x ~ z` in creativa.", ""]
    for tipo in ordine:
        del_tipo = sorted((l for l in luoghi if l.tipo == tipo), key=lambda l: l.nome)
        if not del_tipo:
            continue
        righe += [f"## {titoli[tipo]}", ""]
        for l in del_tipo:
            extra = f" - {l.nota}" if l.nota else ""
            y = f", y={l.y}" if l.y else ""
            righe.append(f"- **{l.nome}**: x={l.x}, z={l.z}{y}{extra}")
        righe.append("")
    md = os.path.join(cartella, "luoghi.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(righe))
    js = os.path.join(cartella, "luoghi.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump([l.__dict__ for l in luoghi], f, ensure_ascii=False, indent=1)
    return md, js


# --------------------------------------------------------------------------
# Catena di indizi
# --------------------------------------------------------------------------

@dataclass
class Libro:
    titolo: str
    autore: str
    pagine: tuple


def catena(luoghi: list[Luogo]) -> tuple[Luogo | None, list[Luogo]]:
    """(citta' di partenza, tappe in ordine). La partenza e' la citta' piu' vicina
    al centro del mondo; le tappe sono rovine e torri, sempre la piu' vicina a
    quella appena lasciata."""
    citta = [l for l in luoghi if l.tipo == "citta"]
    tappe = [l for l in luoghi if l.tipo in TIPI_AVVENTURA]
    if not citta or not tappe:
        return None, []
    partenza = min(citta, key=lambda c: c.x ** 2 + c.z ** 2)
    ordine: list[Luogo] = []
    qui = partenza
    resto = list(tappe)
    while resto:
        prossima = min(resto, key=lambda l: distanza(qui, l))
        ordine.append(prossima)
        resto.remove(prossima)
        qui = prossima
    return partenza, ordine


def _passi(a: Luogo, b: Luogo) -> str:
    d = distanza(a, b)
    approx = int(round(d / 10.0)) * 10
    return f"a circa {approx} passi a {direzione(b.x - a.x, b.z - a.z)}"


def libro_iniziale(partenza: Luogo, prima: Luogo) -> Libro:
    return Libro(
        "Il diario del viandante", "Un viandante",
        (f"Ho girato a lungo le strade di {partenza.nome}. Una voce corre fra i vecchi: "
         f"{_passi(partenza, prima)} da qui sorge {prima.nome}.",
         "Chi ci e' andato non e' sempre tornato, ma chi e' tornato aveva le tasche piene. "
         "Porta una buona lama e qualcosa da mangiare."))


def libro_tappa(da: Luogo, a: Luogo | None) -> Libro:
    if a is None:
        return Libro("L'ultima pagina", "Il Custode",
                     (f"Chi legge ha seguito la pista fino in fondo, a {da.nome}.",
                      "Il tesoro e' tuo. Non lasciare che altri sappiano da dove viene."))
    return Libro(
        f"Carte di {da.nome}", "Il Custode",
        (f"Quello che ho nascosto qui non e' tutto. Il resto sta in un altro luogo: "
         f"{a.nome}, {_passi(da, a)} da questo posto.",
         "Segui la strada con prudenza: il capo di quel posto non e' meno testardo del mio."))


def indizi(luoghi: list[Luogo]) -> dict:
    """{'inizio': (citta, Libro) | None, 'tappe': {(tipo, x, z): Libro}} - le
    chiavi delle tappe sono tipo e coordinate di GIOCO del luogo."""
    partenza, tappe = catena(luoghi)
    if partenza is None:
        return {"inizio": None, "tappe": {}}
    fuori = {"inizio": (partenza, libro_iniziale(partenza, tappe[0])), "tappe": {}}
    for i, l in enumerate(tappe):
        successiva = tappe[i + 1] if i + 1 < len(tappe) else None
        fuori["tappe"][(l.tipo, l.x, l.z)] = libro_tappa(l, successiva)
    return fuori


# --------------------------------------------------------------------------
# Cartelli
# --------------------------------------------------------------------------

ROTAZIONE = {(0, 1): 0, (-1, 0): 4, (0, -1): 8, (1, 0): 12}     # verso dove guarda


def _righe(*testi: str) -> tuple:
    r = list(testi)[:4]
    return tuple(r + [""] * (4 - len(r)))


def cartelli(dungeon: list, torri: list, miniere: list, luoghi: list[Luogo],
             meta: int) -> list[Cartello]:
    """Un cartello davanti a ogni rovina, torre e miniera, col nome del luogo."""
    per_posto = {(l.tipo, l.x, l.z): l for l in luoghi}
    fuori: list[Cartello] = []
    for d in dungeon:
        l = per_posto.get(("rovina", d.x - meta, d.z - meta))
        x, z = d.punto(-4 - 2, 2)            # fuori dalla rovina, sul lato dell'ingresso
        fuori.append(Cartello(x, z, ROTAZIONE[(-d.dx, -d.dz)],
                              _righe(l.nome if l else "Rovine", "", "Non scendere", "da solo")))
    for t in torri:
        l = per_posto.get(("torre", t.x - meta, t.z - meta))
        x, z = t.punto(4 + 2, 2)
        fuori.append(Cartello(x, z, ROTAZIONE[(t.dx, t.dz)],
                              _righe(l.nome if l else "Torre", "", "Qui comanda", t.nome)))
    for mn in miniere:
        r = mn.rampa
        if r is None:
            continue
        l = per_posto.get(("miniera", mn.x - meta, mn.z - meta))
        x = mn.x + r.dx * -5 - r.dz * 3
        z = mn.z + r.dz * -5 + r.dx * 3
        fuori.append(Cartello(x, z, ROTAZIONE[(-r.dx, -r.dz)],
                              _righe(l.nome if l else "Miniera", "", "Pericolo di crollo")))
    return fuori


def entita_cartello(c: Cartello, y: int, meta: int):
    """Il `BlockEntity` (formato universale di amulet) di un cartello."""
    from amulet.api.block_entity import BlockEntity
    from amulet_nbt import ByteTag, CompoundTag, ListTag, NamedTag, StringTag
    testo = lambda r: ListTag([StringTag(t) for t in r])
    utags = CompoundTag({
        "front_text": CompoundTag({"color": StringTag("black"), "has_glowing_text": ByteTag(0),
                                   "java_nbt": testo(c.righe)}),
        "back_text": CompoundTag({"java_nbt": testo(("", "", "", ""))}),
        "is_waxed": ByteTag(1),
    })
    return BlockEntity("universal_minecraft", "sign", c.x - meta, y, c.z - meta,
                       NamedTag(CompoundTag({"utags": utags})))
