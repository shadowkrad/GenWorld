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
           "Miniera dell'Ultima Vena", "Miniera del Lupo", "Miniera del Re Cieco",
           "Miniera del Fumo", "Miniera dell'Argento Nero", "Miniera del Carbone",
           "Miniera del Cristallo", "Miniera del Vento", "Miniera del Gufo",
           "Miniera della Pioggia", "Miniera del Sale", "Miniera del Sole Spento",
           "Miniera di Roccaforte", "Miniera dei Ciottoli", "Miniera dell'Orso",
           "Miniera della Volpe Rossa", "Miniera dei Sospiri", "Miniera del Falco",
           "Miniera della Luna Cava", "Miniera del Ghiaccio", "Miniera di Valpiombo",
           "Miniera del Cinghiale", "Miniera dello Stagno", "Miniera della Brace",
           "Miniera dell'Eremita", "Miniera dei Due Fratelli", "Miniera del Salice")
ROVINE = ("Cripta di Mortelia", "Cripta dei Sussurri", "Cripta di Orrenda",
          "Tomba del Re Smarrito", "Cripta della Veglia", "Rovine di Altomuro",
          "Cripta dell'Osso", "Sepolcro dei Cento", "Cripta di Valdoscura")
ACCAMPAMENTI = ("Campo dei Predoni", "Accampamento del Lupo", "Campo dei Senzanome",
                "Bivacco dell'Esule", "Campo del Corvo", "Accampamento delle Ceneri")
CIMITERI = ("Cimitero degli Impiccati", "Camposanto dei Dimenticati", "Cimitero di Valdoro",
            "Giardino dei Pianti", "Cimitero del Silenzio")
ISOLE = ("Isola del Corvo Bianco", "Isola dei Naufraghi", "Scoglio del Capitano",
         "Isola della Croce Rossa", "Isola di Sabbia Nera", "Isolotto del Pirata Muto",
         "Isola delle Due Palme", "Secca del Tesoro")
PORTALI = ("Portale Antico", "Il Varco Spezzato", "Portale delle Ceneri", "Arco del Confine")

TIPI_AVVENTURA = ("rovina", "torre", "fortezza", "castello", "isola")

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
             meta: int, seed: int = 0, fortezze: list | None = None,
             castelli: list | None = None, isole: list | None = None) -> list[Luogo]:
    """Dà un nome a ogni luogo e lo mette in coordinate di gioco."""
    rng = np.random.default_rng(seed + 8888)
    usati: set = set()
    fuori: list[Luogo] = []
    for c in citta:
        fuori.append(Luogo(_scegli(rng, CITTA, usati), "citta", c.x - meta, c.z - meta,
                           nota=f"raggio {c.raggio}" + (", murata" if c.murata else "")
                           + (f", {c.forma}" if getattr(c, "forma", "quadrato") != "quadrato" else "")))
    for mn in miniere:
        fuori.append(Luogo(_scegli(rng, MINIERE, usati), "miniera", mn.x - meta, mn.z - meta,
                           mn.y_superficie))
    for d in dungeon:
        fuori.append(Luogo(_scegli(rng, ROVINE, usati), "rovina", d.x - meta, d.z - meta, d.ys,
                           nota=f"cripta con {d.mob}"))
    for t in torri:
        fuori.append(Luogo(f"Torre di {t.nome}", "torre", t.x - meta, t.z - meta, t.ys,
                           nota=f"capo: {t.nome}"))
    for fz in fortezze or []:
        fuori.append(Luogo(f"Fortezza di {fz.nome}", "fortezza", fz.x - meta, fz.z - meta, fz.ys,
                           nota=f"capo: {fz.nome}"))
    for cs in castelli or []:
        fuori.append(Luogo(cs.nome, "castello", cs.x - meta, cs.z - meta, cs.ys,
                           nota=f"in rovina, signore: {cs.capo}"))
    for isl in isole or []:
        fuori.append(Luogo(_scegli(rng, ISOLE, usati), "isola", isl.x - meta, isl.z - meta, isl.cima,
                           nota="una croce rossa sulla sabbia: il forziere e' sotto"))
    for av in avamposti:
        tab = {"campo": ("accampamento", ACCAMPAMENTI), "cimitero": ("cimitero", CIMITERI),
               "portale": ("portale", PORTALI)}[av.tipo]
        fuori.append(Luogo(_scegli(rng, tab[1], usati), tab[0], av.x - meta, av.z - meta, av.y))
    return fuori


# --------------------------------------------------------------------------
# Elenco
# --------------------------------------------------------------------------

def scrivi_elenco(luoghi: list[Luogo], cartella: str, nome_mondo: str = "",
                  extra_md: str = "") -> tuple[str, str]:
    """Scrive `luoghi.md` e `luoghi.json` in `cartella`. Ritorna i due percorsi."""
    os.makedirs(cartella, exist_ok=True)
    ordine = ("citta", "torre", "fortezza", "castello", "isola", "rovina", "miniera", "accampamento", "cimitero", "portale")
    titoli = {"citta": "Citta'", "torre": "Torri con un capo", "fortezza": "Fortezze",
              "castello": "Castelli in rovina", "isola": "Isole del tesoro", "rovina": "Rovine con cripta",
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
        f.write("\n".join(righe) + extra_md)
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


def libro_di_bottega(qui: Luogo, a: Luogo, mestiere: str = "mercante") -> Libro:
    """Cio' che un bottegaio racconta a chi paga: dove sta un luogo d'avventura."""
    return Libro(
        f"Voci di {qui.nome}"[:32] if qui.nome else "Voci di bottega", f"Un {mestiere}",
        (f"Chi passa di qui racconta di un posto chiamato {a.nome}, {_passi(qui, a)} da questa bottega.",
         "Non e' un luogo per chi viaggia leggero: porta cibo, una buona lama e delle torce."))


def offerte_bottega(abitanti: list, luoghi: list[Luogo], meta: int) -> list:
    """Ai cartografi e ai librai che stanno in citta' si aggiunge uno scambio:
    5 smeraldi per un libro che indica la rovina o la torre piu' vicina.
    Ritorna gli abitanti, con le offerte dove servono."""
    import dataclasses
    mete = [l for l in luoghi if l.tipo in TIPI_AVVENTURA]
    citta = [l for l in luoghi if l.tipo == "citta"]
    if not mete:
        return abitanti
    fuori = []
    for a in abitanti:
        if getattr(a, "mestiere", None) in ("cartografo", "libraio"):
            qui = Luogo("", "bottega", int(a.x - meta), int(a.z - meta))
            vicina = min(citta, key=lambda c: distanza(qui, c)) if citta else None
            if vicina is not None:
                qui.nome = vicina.nome
            target = min(mete, key=lambda l: distanza(qui, l))
            libro = libro_di_bottega(qui, target, "cartografo" if a.mestiere == "cartografo" else "libraio")
            offerta = {"compra": ("emerald", 5),
                       "vende": ("written_book", 1, {"written_book_content": {
                           "title": {"raw": libro.titolo}, "author": libro.autore,
                           "pages": [{"raw": p} for p in libro.pagine]}}),
                       "usi": 3}
            a = dataclasses.replace(a, offerte=tuple(a.offerte) + (offerta,))
        fuori.append(a)
    return fuori


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
             meta: int, fortezze: list | None = None,
             castelli: list | None = None) -> list[Cartello]:
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
    for fz in fortezze or []:
        l = per_posto.get(("fortezza", fz.x - meta, fz.z - meta))
        x, z = fz.punto(14 + 3, 4)            # davanti al portale, a lato del viale
        fuori.append(Cartello(x, z, ROTAZIONE[(fz.dx, fz.dz)],
                              _righe(l.nome[:15] if l else "Fortezza", l.nome[15:30] if l else "",
                                     "Fermati", "e annunciati")))
    for cs in castelli or []:
        l = per_posto.get(("castello", cs.x - meta, cs.z - meta))
        x, z = cs.punto(19 + 3, 6)             # fuori dal portale sfondato
        fuori.append(Cartello(x, z, ROTAZIONE[(cs.dx, cs.dz)],
                              _righe(l.nome[:15] if l else "Castello", "", "Chi entra non", "torna uguale")))
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


# --------------------------------------------------------------------------
# Cartelli ai bivi (#10)
# --------------------------------------------------------------------------

RAGGIO_BIVIO = 5
DISTANZA_FRA_BIVI = 70


def _rami(strada: np.ndarray, x: int, z: int, r: int = RAGGIO_BIVIO) -> int:
    """Quante strade partono da (x, z): gli archi di strada su un anello di
    raggio r. Una strada dritta ne ha 2, un bivio 3, un incrocio 4."""
    H, W = strada.shape
    n = 32
    valori = []
    for k in range(n):
        a = 2 * math.pi * k / n
        px, pz = int(round(x + r * math.cos(a))), int(round(z + r * math.sin(a)))
        valori.append(bool(0 <= px < W and 0 <= pz < H and strada[pz, px]))
    return sum(1 for k in range(n) if valori[k] and not valori[k - 1])


def trova_bivi(strada: np.ndarray, passo: int = 4) -> list[tuple[int, int]]:
    """Centri dei bivi (3 o piu' rami), distanti fra loro almeno
    `DISTANZA_FRA_BIVI` celle. `strada` e' una maschera booleana (z, x)."""
    H, W = strada.shape
    candidati = []
    zz, xx = np.nonzero(strada[::passo, ::passo])
    for z, x in zip((zz * passo).tolist(), (xx * passo).tolist()):
        if RAGGIO_BIVIO < x < W - RAGGIO_BIVIO and RAGGIO_BIVIO < z < H - RAGGIO_BIVIO                 and _rami(strada, x, z) >= 3:
            candidati.append((x, z))
    scelti: list[tuple[int, int]] = []
    for x, z in candidati:
        if all((x - a) ** 2 + (z - b) ** 2 >= DISTANZA_FRA_BIVI ** 2 for a, b in scelti):
            scelti.append((x, z))
    return scelti


def cartelli_bivi(strada: np.ndarray, luoghi: list[Luogo], meta: int,
                  massimo: int = 40) -> list[Cartello]:
    """Un cartello accanto a ogni bivio con i due luoghi piu' vicini (che non
    siano a portata di mano): nome e direzione."""
    if not luoghi:
        return []
    H, W = strada.shape
    fuori: list[Cartello] = []
    for x, z in trova_bivi(strada)[:massimo]:
        gx, gz = x - meta, z - meta
        qui = Luogo("", "", gx, gz)
        vicini = sorted((l for l in luoghi if 25 < distanza(qui, l) < 600 and l.tipo != "isola"),
                        key=lambda l: distanza(qui, l))[:2]
        if not vicini:
            continue
        # la cella libera piu' vicina accanto alla strada
        posto = None
        for r in range(2, 7):
            for dx, dz in ((r, 0), (-r, 0), (0, r), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)):
                px, pz = x + dx, z + dz
                if 0 <= px < W and 0 <= pz < H and not strada[pz, px]:
                    posto = (px, pz, dx, dz)
                    break
            if posto:
                break
        if posto is None:
            continue
        px, pz, dx, dz = posto
        # guarda verso il bivio
        vx, vz = (-dx, -dz)
        ver = (int(np.sign(vx)), 0) if abs(vx) >= abs(vz) else (0, int(np.sign(vz)))
        righe = []
        for l in vicini:
            righe.append(l.nome[:15])
            righe.append(f"{direzione(l.x - gx, l.z - gz)} {distanza(qui, l)}")
        fuori.append(Cartello(px, pz, ROTAZIONE[ver], _righe(*righe)))
    return fuori
