"""Edifici parametrici.

Un edificio e' una funzione di pochi parametri - larghezza, profondita', piani,
palette, orientamento - non un modello copiato. Cosi' la varieta' e' infinita e
la coerenza col bioma viene gratis: capanna di tronchi in foresta, casa di
arenaria nel deserto, abete e pietra in montagna.

La PALAFITTA non e' una tipologia a parte ma un modificatore: si attiva quando
il sedime cade sull'acqua. Si piantano i pali dal fondale, si stende la
piattaforma, e da li' in su costruisce lo stesso identico generatore di casa,
che non sa nemmeno di stare sull'acqua. E' il pezzo di progetto piu' elegante
di tutto il programma: un villaggio lacustre viene fuori senza scrivere codice
per i villaggi lacustri.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mappa import (DESERTO, FORESTA, MONTAGNA, NEVE, PIANURA, PRATERIA,
                    SPIAGGIA)

NORD, EST, SUD, OVEST = range(4)
DIREZIONE = {NORD: "north", EST: "east", SUD: "south", OVEST: "west"}
OPPOSTO = {NORD: SUD, SUD: NORD, EST: OVEST, OVEST: EST}


@dataclass(frozen=True)
class Palette:
    """I materiali di uno stile costruttivo."""
    muro: tuple           # (nome, proprieta')
    telaio: tuple
    pavimento: tuple
    tetto: str            # materiale delle scale del tetto
    basamento: tuple
    legno: str            # materiale di porte, staccionate, botole


def _planks(m): return ("planks", {"material": m})
def _log(m, stripped="false"): return ("log", {"axis": "y", "material": m, "stripped": stripped})


PALETTE = {
    "bosco": Palette(_planks("oak"), _log("oak"), _planks("oak"), "oak",
                     ("cobblestone", {}), "oak"),
    "montagna": Palette(("stone_bricks", {"variant": "normal"}), _log("spruce"),
                        _planks("spruce"), "spruce", ("cobblestone", {}), "spruce"),
    "deserto": Palette(("sandstone", {"variant": "normal"}), ("sandstone", {"variant": "normal"}),
                       _planks("birch"), "birch",
                       ("sandstone", {"variant": "normal"}), "birch"),
    "prato": Palette(_planks("oak"), _log("spruce"), _planks("spruce"), "spruce",
                     ("cobblestone", {}), "oak"),
}

PALETTE_PER_CLASSE = {
    FORESTA: "bosco", PRATERIA: "prato", PIANURA: "prato",
    MONTAGNA: "montagna", NEVE: "montagna",
    DESERTO: "deserto", SPIAGGIA: "deserto",
}


@dataclass
class Edificio:
    """Un edificio pianificato, in coordinate della mappa."""
    x: int                 # angolo minimo
    z: int
    larghezza: int         # lungo x
    profondita: int        # lungo z
    base: int              # quota del pavimento
    piani: int = 1
    altezza_piano: int = 4
    porta: int = NORD
    stile: str = "prato"
    palafitta: bool = False
    fondale: int = 0       # quota del fondale, per i pali
    gronda: int = 1        # di quanto il tetto sporge oltre i muri
    mestiere: str = ""     # "" = abitazione, altrimenti bottega
    seme: int = 0

    @property
    def x1(self) -> int: return self.x + self.larghezza
    @property
    def z1(self) -> int: return self.z + self.profondita
    @property
    def falde(self) -> int:
        """Quanti gradini di falda: il tetto e' a due acque, e la pendenza e'
        di un blocco per blocco - la sola che in Minecraft non lasci buchi.

        La campata deve essere DISPARI, altrimenti le due falde si incontrano
        su due file e il colmo non esiste: resta una scanalatura lunga quanto
        la casa. Quando viene pari si accorcia la gronda di un lato, che e'
        uno sfalsamento di un blocco che non si nota, al contrario del
        solco sul colmo, che si nota da lontano.
        """
        celle = min(self.larghezza, self.profondita) + 2 * self.gronda
        if celle % 2 == 0:
            celle -= 1
        return celle // 2 + 1

    @property
    def colmo(self) -> int:
        return self.base + self.piani * self.altezza_piano + self.falde

    def ingombro_tetto(self) -> tuple[int, int, int, int]:
        """Riquadro che il tetto occupa: i muri piu' la gronda.

        E' questo, non il sedime, l'ingombro vero di una casa vista
        dall'alto - ed e' quello che va tenuto libero quando se ne mette
        un'altra accanto.
        """
        return (self.x - self.gronda, self.z - self.gronda,
                self.x1 + self.gronda, self.z1 + self.gronda)


class TavolozzaEdilizia:
    """Id dei blocchi da costruzione, risolti sul livello aperto."""

    def __init__(self, scrittore):
        s = scrittore
        self.aria = s.id_aria
        self._s = s
        self.blocco = {}
        for nome, p in PALETTE.items():
            self.blocco[(nome, "muro")] = s.blocco(p.muro[0], **p.muro[1])
            self.blocco[(nome, "telaio")] = s.blocco(p.telaio[0], **p.telaio[1])
            self.blocco[(nome, "pavimento")] = s.blocco(p.pavimento[0], **p.pavimento[1])
            self.blocco[(nome, "basamento")] = s.blocco(p.basamento[0], **p.basamento[1])
            self.blocco[(nome, "palo")] = s.blocco(
                "log", axis="y", material=p.legno, stripped="true")
        self.vetro = s.blocco("glass_pane", north="false", south="false",
                              east="false", west="false")
        self.vetro_pieno = s.blocco("glass")
        self.torcia = s.blocco("torch", facing="up")

    def scala(self, materiale: str, verso: int, meta: str = "bottom") -> int:
        return self._s.blocco("stairs", facing=DIREZIONE[verso], half=meta,
                              material=materiale, shape="straight")

    def porta(self, materiale: str, verso: int, meta: str) -> int:
        return self._s.blocco("door", facing=DIREZIONE[verso], half=meta,
                              hinge="left", material=materiale,
                              open="false", powered="false")

    def b(self, nome: str, **proprieta: str) -> int:
        """Un blocco qualunque, per l'arredo."""
        return self._s.blocco(nome, **proprieta)

    def staccionata(self, materiale: str) -> int:
        return self._s.blocco("fence", material=materiale, north="false",
                              south="false", east="false", west="false")

    def trave(self, materiale: str, asse: str) -> int:
        """Un tronco COLTO DI TRAVERSO. E' mezzo mestiere del graticcio: il
        montante e' un tronco in piedi, il corrente e' lo stesso tronco
        sdraiato, e finche' erano tutti in piedi le facciate restavano
        quadretti di assi tutti uguali."""
        return self._s.blocco("log", axis=asse, material=materiale,
                              stripped="false")


# --------------------------------------------------------------------------
# Costruzione
# --------------------------------------------------------------------------

def costruisci(out: np.ndarray, y0: int, ox: int, oz: int,
               ed: Edificio, tav: TavolozzaEdilizia) -> None:
    """Stampa un edificio dentro l'array di chunk (16, H, 16)."""
    H = out.shape[1]
    pal = PALETTE[ed.stile]
    rng = np.random.default_rng(ed.seme)

    def posa(gx: int, gy: int, gz: int, blocco: int) -> None:
        lx, lz, ly = gx - ox, gz - oz, gy - y0
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 <= ly < H:
            out[lx, ly, lz] = blocco

    def vuota(gx: int, gy: int, gz: int) -> None:
        posa(gx, gy, gz, tav.aria)

    def leggi(gx: int, gy: int, gz: int):
        lx, lz, ly = gx - ox, gz - oz, gy - y0
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 <= ly < H:
            return out[lx, ly, lz]
        return None

    m = tav.blocco[(ed.stile, "muro")]
    telaio = tav.blocco[(ed.stile, "telaio")]
    pavimento = tav.blocco[(ed.stile, "pavimento")]
    basamento = tav.blocco[(ed.stile, "basamento")]
    palo = tav.blocco[(ed.stile, "palo")]

    x0, z0 = ed.x, ed.z
    x1, z1 = ed.x1 - 1, ed.z1 - 1

    # --- palafitta: pali dal fondale e piattaforma -----------------------
    if ed.palafitta:
        for gx in range(x0 - 1, x1 + 2):
            for gz in range(z0 - 1, z1 + 2):
                bordo = gx in (x0 - 1, x1 + 1) or gz in (z0 - 1, z1 + 1)
                angolo_palo = ((gx - x0) % 3 == 0) and ((gz - z0) % 3 == 0)
                if bordo or angolo_palo:
                    for gy in range(ed.fondale, ed.base):
                        posa(gx, gy, gz, palo)
                posa(gx, ed.base - 1, gz, pavimento)
    else:
        # Fondazione, non pavimento. Il lotto e' spianato, ma il raccordo
        # lascia il bordo un po' piu' basso: con un solo strato a base-1 la
        # casa appoggiava su un'aria di uno o due blocchi lungo un lato. Qui
        # si scende finche' non si trova del pieno, al massimo sei blocchi.
        for gx in range(x0, x1 + 1):
            for gz in range(z0, z1 + 1):
                posa(gx, ed.base - 1, gz, basamento)
                for giu in range(2, 8):
                    sotto_di_qui = leggi(gx, ed.base - giu, gz)
                    if sotto_di_qui is None or sotto_di_qui != tav.aria:
                        break
                    posa(gx, ed.base - giu, gz, basamento)

    altezza = ed.piani * ed.altezza_piano

    # --- muri a graticcio -------------------------------------------------
    # La facciata non e' una parete di assi. E' un telaio di legno riempito di
    # muratura: montanti verticali ogni tre o quattro blocchi, un corrente
    # orizzontale a ogni solaio, e in mezzo il tamponamento. Senza il telaio
    # si ottiene una scatola, ed e' quello che si vedeva dall'alto.
    passo_montante = 3 if min(ed.larghezza, ed.profondita) >= 8 else 4
    corrente_x = tav.trave(pal.legno, "x")
    corrente_z = tav.trave(pal.legno, "z")

    for piano in range(ed.piani):
        y_base = ed.base + piano * ed.altezza_piano
        for k in range(ed.altezza_piano):
            gy = y_base + k
            ultimo = k == ed.altezza_piano - 1
            for gx in range(x0, x1 + 1):
                for gz in range(z0, z1 + 1):
                    sul_bordo = gx in (x0, x1) or gz in (z0, z1)
                    if not sul_bordo:
                        vuota(gx, gy, gz)
                        continue
                    angolo = gx in (x0, x1) and gz in (z0, z1)
                    if angolo:
                        posa(gx, gy, gz, telaio)
                    elif ultimo:
                        # il corrente corre nel verso del muro su cui sta
                        posa(gx, gy, gz,
                             corrente_x if gz in (z0, z1) else corrente_z)
                    elif k == 0 and piano == 0:
                        posa(gx, gy, gz, basamento)   # zoccolo di pietra
                    elif ((gz in (z0, z1) and (gx - x0) % passo_montante == 0)
                          or (gx in (x0, x1) and (gz - z0) % passo_montante == 0)):
                        posa(gx, gy, gz, telaio)      # montante
                    else:
                        posa(gx, gy, gz, m)
        # solaio fra i piani, col vano scala aperto
        if piano < ed.piani - 1:
            for gx in range(x0 + 1, x1):
                for gz in range(z0 + 1, z1):
                    if (gx, gz) == (x0 + 1, z0 + 1):
                        continue          # il buco per la scala
                    posa(gx, y_base + ed.altezza_piano - 1, gz, pavimento)

    # --- porta -----------------------------------------------------------
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    if ed.porta == NORD:   px, pz = cx, z0
    elif ed.porta == SUD:  px, pz = cx, z1
    elif ed.porta == OVEST: px, pz = x0, cz
    else:                  px, pz = x1, cz

    # --- finestre --------------------------------------------------------
    # Due blocchi di altezza e incassate fra i montanti. Quelle di prima erano
    # un vetro solo a mezza altezza, che da fuori si legge come un puntino.
    for piano in range(ed.piani):
        gy = ed.base + piano * ed.altezza_piano + 1
        for gx in range(x0 + 1, x1):
            if (gx - x0) % passo_montante == 0:
                continue
            for gz in (z0, z1):
                if (gx, gz) == (px, pz):
                    continue
                posa(gx, gy, gz, tav.vetro)
                posa(gx, gy + 1, gz, tav.vetro)
        for gz in range(z0 + 1, z1):
            if (gz - z0) % passo_montante == 0:
                continue
            for gx in (x0, x1):
                if (gx, gz) == (px, pz):
                    continue
                posa(gx, gy, gz, tav.vetro)
                posa(gx, gy + 1, gz, tav.vetro)

    posa(px, ed.base, pz, tav.porta(pal.legno, OPPOSTO[ed.porta], "lower"))
    posa(px, ed.base + 1, pz, tav.porta(pal.legno, OPPOSTO[ed.porta], "upper"))
    posa(px, ed.base + 2, pz, telaio)          # architrave

    # --- tetto a due falde ------------------------------------------------
    # Il padiglione di prima era una piramide: quattro falde che convergono in
    # un punto. Sta bene su una villa, non su una casa di paese, e vista
    # dall'alto da' un paese di tegole a rombi tutte uguali. Qui c'e' un colmo
    # vero, due falde, e due timpani murati alle testate.
    cima = ed.base + altezza
    lungo_x = ed.larghezza >= ed.profondita
    g = ed.gronda
    passi = ed.falde

    # estremi della falda, in coordinate di gronda
    if lungo_x:
        a0, a1 = x0 - g, x1 + g          # lungo il colmo
        b0, b1 = z0 - g, z1 + g          # lungo la pendenza
    else:
        a0, a1 = z0 - g, z1 + g
        b0, b1 = x0 - g, x1 + g
    if (b1 - b0) % 2 == 1:
        b1 -= 1                          # campata dispari: vedi `falde`

    for k in range(passi):
        # La falda comincia SOTTO il filo di gronda, non sopra: alzandola di
        # un blocco - com'era - restava una feritoia aperta fra la testa del
        # muro e la falda, lungo tutti e due i lati lunghi. Da dentro si
        # vedeva il cielo, da fuori sembrava una casa non finita.
        gy = cima - g + k
        q0, q1 = b0 + k, b1 - k
        if q0 > q1:
            break
        for a in range(a0, a1 + 1):
            for q, verso in ((q0, NORD if lungo_x else OVEST),
                             (q1, SUD if lungo_x else EST)):
                if q0 == q1:
                    # il colmo: una trave nel verso del colmo, non due scale
                    # che si scontrano
                    gx, gz = (a, q) if lungo_x else (q, a)
                    posa(gx, gy, gz, corrente_x if lungo_x else corrente_z)
                    continue
                gx, gz = (a, q) if lungo_x else (q, a)
                posa(gx, gy, gz, tav.scala(pal.tetto, verso))
        # timpano: il muro triangolare alle due testate, sotto la falda
        if q0 < q1:
            for a in (a0 + g, a1 - g):
                for q in range(q0 + 1, q1):
                    gx, gz = (a, q) if lungo_x else (q, a)
                    posa(gx, gy, gz, m)
        # NIENTE svuotamento del sottotetto. C'era, e alla prima falda -
        # quella che passa esattamente sul filo dei muri - cancellava il
        # corrente di testa lungo tutti e due i lati lunghi: la casa restava
        # senza l'ultimo corso, con il tetto appoggiato sul vuoto. Sotto la
        # falda c'e' gia' aria, perche' l'interno dei muri e' stato svuotato
        # mentre li si alzava.

    # --- comignolo --------------------------------------------------------
    # Un tetto senza fumaiolo e' un tetto da modellino. Sale da dentro casa,
    # buca la falda e sporge di due blocchi.
    if ed.larghezza >= 6 and ed.profondita >= 6:
        fx = x0 + 1 if ed.porta != OVEST else x1 - 1
        fz = z0 + 1 if ed.porta != NORD else z1 - 1
        gola = cima + passi + 1
        for gy in range(ed.base, gola + 1):
            posa(fx, gy, fz, basamento)
        # il fuoco sta DENTRO la canna, due blocchi sotto la bocca: il fumo
        # esce dal comignolo, che e' l'unico motivo per cui un comignolo c'e'
        posa(fx, gola - 1, fz, tav.b("campfire", facing="north", lit="true",
                                     signal_fire="false", waterlogged="false"))
        vuota(fx, gola, fz)

    # --- interni ---------------------------------------------------------
    _arreda(posa, ed, tav, pal, rng)

    # --- dettagli --------------------------------------------------------
    if rng.random() < 0.6:
        posa(px, ed.base + 1, pz + (1 if ed.porta == NORD else -1), tav.aria)
    # torcia accanto alla porta
    dx = 1 if ed.porta in (NORD, SUD) else 0
    dz = 0 if ed.porta in (NORD, SUD) else 1
    posa(px + dx, ed.base + 2, pz + dz, tav.torcia)


COLORI_LETTO = ("red", "white", "blue", "yellow", "green", "brown")

# Il banco di lavoro di ogni mestiere, piu' quello che ci sta attorno.
# In Minecraft il banco non e' decorazione: e' il "posto di lavoro" che un
# abitante rivendica, quindi metterlo giusto non e' un vezzo, e' quello che
# fa funzionare la bottega.
BOTTEGA = {
    "fabbro":        ("smithing_table", ("anvil", "furnace", "cauldron")),
    "armaiolo":      ("grindstone",     ("anvil", "furnace")),
    "corazzaio":     ("blast_furnace",  ("anvil", "cauldron")),
    "falegname":     ("fletching_table", ("crafting_table", "barrel")),
    "scalpellino":   ("stonecutter",    ("crafting_table",)),
    "macellaio":     ("smoker",         ("cauldron", "barrel")),
    "fruttivendolo": ("composter",      ("barrel", "hay_block")),
    "pescatore":     ("barrel",         ("cauldron", "barrel")),
    "pastore":       ("loom",           ("wool", "barrel")),
    "libraio":       ("lectern",        ("bookshelf", "bookshelf")),
    "cartografo":    ("cartography_table", ("bookshelf", "barrel")),
    "conciatore":    ("cauldron",       ("barrel", "cauldron")),
    "speziale":      ("brewing_stand",  ("bookshelf", "cauldron")),
}

# Come si posa ciascun blocco d'arredo: alcuni vogliono proprieta', e senza
# quelle non arrivano in gioco. E' la regola di `oak_log`, applicata
# all'arredamento.
def _arredo(tav: TavolozzaEdilizia, nome: str, verso: int = NORD) -> int:
    d = DIREZIONE[verso]
    if nome in ("smithing_table", "fletching_table", "crafting_table",
                "cartography_table", "bookshelf"):
        return tav.b(nome)
    if nome == "anvil":
        return tav.b("anvil", facing=d, damage="0")
    if nome == "grindstone":
        return tav.b("grindstone", face="floor", facing=d)
    if nome in ("furnace", "blast_furnace", "smoker"):
        return tav.b(nome, facing=d, lit="false")
    if nome == "composter":
        return tav.b("composter", level="0")
    if nome == "barrel":
        return tav.b("barrel", facing="up", open="false")
    if nome in ("loom", "stonecutter", "lectern"):
        if nome == "lectern":
            return tav.b("lectern", facing=d, has_book="false", powered="false")
        return tav.b(nome, facing=d)
    if nome == "cauldron":
        return tav.b("cauldron", level="0")
    if nome == "brewing_stand":
        return tav.b("brewing_stand", has_bottle_0="false",
                     has_bottle_1="false", has_bottle_2="false")
    if nome == "hay_block":
        return tav.b("hay_block", axis="y")
    if nome == "wool":
        return tav.b("wool", color="white")
    return tav.b(nome)


def _arreda(posa, ed: Edificio, tav: TavolozzaEdilizia, pal: Palette,
            rng: np.random.Generator) -> None:
    """Mette dentro quello che rende una casa una casa, o una bottega.

    Finche' gli edifici erano gusci vuoti la differenza non si vedeva
    dall'alto, e infatti nessuno se n'era accorto - ma entrarci dentro era
    entrare in una scatola. Qui non si arreda a caso: il letto sta lontano
    dalla porta, il focolare contro un muro, la scala in un angolo sotto il
    buco nel solaio, e il banco del mestiere dove lo vede chi entra.
    """
    ix0, iz0 = ed.x + 1, ed.z + 1
    ix1, iz1 = ed.x1 - 2, ed.z1 - 2
    if ix1 - ix0 < 2 or iz1 - iz0 < 2:
        return                    # troppo piccola per arredarla

    colore = COLORI_LETTO[rng.integers(0, len(COLORI_LETTO))]
    scala_su = tav.b("ladder", facing="south")

    for piano in range(ed.piani):
        y = ed.base + piano * ed.altezza_piano
        ultimo = piano == ed.piani - 1

        # scala verso il piano sopra, nell'angolo del buco nel solaio
        if not ultimo:
            for k in range(ed.altezza_piano):
                posa(ix0, y + k, iz0, scala_su)

        # letto lontano dalla porta: testa nell'angolo opposto
        if rng.random() < (0.9 if ultimo else 0.4):
            posa(ix1, y, iz1, tav.b("bed", facing="south", occupied="false",
                                    part="head", color=colore))
            posa(ix1, y, iz1 - 1, tav.b("bed", facing="south", occupied="false",
                                        part="foot", color=colore))

        if piano == 0:
            if ed.mestiere in BOTTEGA:
                banco, attorno = BOTTEGA[ed.mestiere]
                # il banco guarda chi entra: sta sul muro opposto alla porta
                posa(ix0 + 1, y, iz1, _arredo(tav, banco, NORD))
                for k, nome in enumerate(attorno):
                    gx = ix0 + 2 + k
                    if gx <= ix1:
                        posa(gx, y, iz1, _arredo(tav, nome, NORD))
                posa(ix1, y, iz0, tav.b("chest", facing="south", type="single"))
            else:
                # abitazione: focolare e banco contro il muro sud
                posa(ix0 + 1, y, iz1, tav.b("furnace", facing="north",
                                            lit="false"))
                posa(ix0 + 2, y, iz1, tav.b("crafting_table"))
                posa(ix1, y, iz0, tav.b("chest", facing="south", type="single"))
        elif rng.random() < 0.5:
            posa(ix0 + 1, y, iz1, tav.b("bookshelf"))

        # luce: una torcia sul pavimento basta, e non serve un muro a reggerla
        posa(ix0 + 1, y, iz0 + 1, tav.torcia)
        if ix1 - ix0 >= 4 and rng.random() < 0.5:
            posa(ix1 - 1, y, iz1 - 1, tav.torcia)


# --------------------------------------------------------------------------
# Banchi del mercato
# --------------------------------------------------------------------------

@dataclass
class Banco:
    """Un banco di mercato: quattro pali, una tenda, un bancone, la merce."""
    x: int
    z: int
    base: int
    verso: int = NORD          # da che parte si serve il cliente
    merce: str = "frutta"
    seme: int = 0

    @property
    def x1(self) -> int: return self.x + 3
    @property
    def z1(self) -> int: return self.z + 3


MERCE = {
    "frutta":  (("melon", {}), ("pumpkin", {}), ("hay_block", {"axis": "y"})),
    "carne":   (("smoker", {"facing": "north", "lit": "false"}),
                ("cauldron", {"level": "0"}), ("campfire", {"facing": "north",
                 "lit": "true", "signal_fire": "false", "waterlogged": "false"})),
    "pesce":   (("barrel", {"facing": "up", "open": "false"}),
                ("cauldron", {"level": "0"}),
                ("barrel", {"facing": "up", "open": "false"})),
    "verdura": (("composter", {"level": "0"}), ("hay_block", {"axis": "y"}),
                ("composter", {"level": "0"})),
}
TENDA = {"frutta": "red", "carne": "brown", "pesce": "light_blue",
         "verdura": "green"}


def costruisci_banco(out: np.ndarray, y0: int, ox: int, oz: int,
                     b: Banco, tav: TavolozzaEdilizia) -> None:
    """Un banco di mercato dentro l'array di chunk.

    Tre blocchi per tre, che e' la misura giusta: un banco piu' grande di
    cosi' non e' un banco, e' un negozio, e i negozi sono le case che ci
    stanno attorno.
    """
    H = out.shape[1]

    def posa(gx, gy, gz, blocco):
        lx, lz, ly = gx - ox, gz - oz, gy - y0
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 <= ly < H:
            out[lx, ly, lz] = blocco

    palo = tav.staccionata("oak")
    tenda = tav.b("wool", color=TENDA.get(b.merce, "white"))
    y = b.base
    x0, z0, x1, z1 = b.x, b.z, b.x1 - 1, b.z1 - 1

    for gx, gz in ((x0, z0), (x1, z0), (x0, z1), (x1, z1)):
        for k in range(3):
            posa(gx, y + k, gz, palo)
    for gx in range(x0, x1 + 1):
        for gz in range(z0, z1 + 1):
            posa(gx, y + 3, gz, tenda)

    # il bancone sta sul lato opposto a chi serve, cosi' il cliente ci arriva
    merce = MERCE.get(b.merce, MERCE["frutta"])
    if b.verso in (NORD, SUD):
        linea = [(gx, z0 if b.verso == SUD else z1) for gx in range(x0, x1 + 1)]
    else:
        linea = [(x0 if b.verso == EST else x1, gz) for gz in range(z0, z1 + 1)]
    for k, (gx, gz) in enumerate(linea):
        nome, prop = merce[k % len(merce)]
        posa(gx, y, gz, tav.b(nome, **prop))


def indice_banchi(banchi: list[Banco], passo: int = 16) -> dict:
    fuori: dict[tuple[int, int], list[int]] = {}
    for i, b in enumerate(banchi):
        for cz in range((b.z - 1) // passo, (b.z1 + 1) // passo + 1):
            for cx in range((b.x - 1) // passo, (b.x1 + 1) // passo + 1):
                fuori.setdefault((cx, cz), []).append(i)
    return fuori


def posto_di_lavoro(out: np.ndarray, y0: int, ox: int, oz: int,
                    ed: "Edificio", tav: TavolozzaEdilizia) -> None:
    """Mette il banco del mestiere dentro una casa di template.

    Una casa disegnata a mano non sa niente dei nostri mestieri, ma il banco
    non e' arredamento: in Minecraft e' il **posto di lavoro** che l'abitante
    rivendica, e senza quello il fabbro che ci abita dentro resta un
    disoccupato con il cappello da fabbro.

    Si cerca una cella d'aria al piano terra con del pieno sotto, partendo dal
    centro e allargandosi: cosi' il banco finisce in mezzo alla stanza se c'e'
    una stanza, e non finisce dentro un muro se non c'e'.
    """
    nome = BOTTEGA.get(ed.mestiere)
    if not nome:
        return
    banco = _arredo(tav, nome[0], NORD)
    H = out.shape[1]
    cx, cz = (ed.x + ed.x1) // 2, (ed.z + ed.z1) // 2
    ly = ed.base - y0
    if not (1 <= ly < H):
        return
    for r in range(0, max(ed.larghezza, ed.profondita) // 2 + 1):
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if max(abs(dx), abs(dz)) != r:
                    continue
                lx, lz = cx + dx - ox, cz + dz - oz
                if not (0 <= lx < 16 and 0 <= lz < 16):
                    continue
                if out[lx, ly, lz] != tav.aria:
                    continue
                if out[lx, ly - 1, lz] == tav.aria:
                    continue
                out[lx, ly, lz] = banco
                return


def punto_lavoro(ed: Edificio) -> tuple[float, float, float]:
    """Dove sta l'abitante, dentro la sua casa: in mezzo alla stanza."""
    return ((ed.x + ed.x1) / 2.0, float(ed.base), (ed.z + ed.z1) / 2.0)


def ingombro(ed: Edificio, margine: int = 2) -> tuple[int, int, int, int]:
    """Riquadro (x0, z0, x1, z1) che l'edificio puo' toccare, gronda inclusa."""
    m = margine + ed.gronda
    return (ed.x - m, ed.z - m, ed.x1 + m, ed.z1 + m)
