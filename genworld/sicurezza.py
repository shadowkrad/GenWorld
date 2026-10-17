"""Rete di sicurezza per le creature: mai nei muri, mai sotto terra.

Villager, animali, mostri e guardiani si piazzano in coordinate decise dalla
pianificazione, PRIMA che i blocchi esistano. Se la costruzione sotto e' diversa
da come la si immaginava (un mobile al centro della stanza, un pavimento a un'altra
quota, un muro dove si pensava ci fosse aria, una roccia sopra un corridoio) la
creatura nasce incastonata o sotto terra, e in gioco soffoca o resta bloccata.

Qui, a mondo scritto, ogni creatura si controlla sui BLOCCHI VERI: la sua cella e
quella sopra devono essere libere e sotto deve esserci qualcosa su cui stare. Se non
e' cosi' la si sposta nella cella valida piu' vicina (prima in orizzontale, poi un po'
in su o in giu'); se non ce n'e' nessuna vicino, si cerca verso l'alto nella stessa
colonna. Chi vola (pipistrelli), nuota (pesci) o va su rotaia (carrelli) non si tocca.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np

DIMENSIONE = "minecraft:overworld"
Y_MIN, Y_MAX = -64, 320

# blocchi in cui una creatura puo' stare (non pieni)
PASSABILI = frozenset({
    "air", "cave_air", "void_air", "carpet", "moss_carpet", "torch", "wall_torch", "soul_torch",
    "redstone_torch", "wall_banner", "banner", "wall_sign", "sign", "hanging_sign", "button",
    "lever", "pressure_plate", "plant", "tall_grass", "double_plant", "snow", "candle", "string",
    "vine", "cobweb", "rail", "powered_rail", "detector_rail", "activator_rail", "ladder",
    "tripwire", "tripwire_hook", "flower_pot", "sweet_berry_bush", "wheat", "carrots", "potatoes",
    "beetroots", "sugar_cane", "leaf_litter", "wildflowers", "pink_petals", "sculk_vein",
    "glow_lichen", "hanging_roots", "cave_vines", "pointed_dripstone", "amethyst_cluster",
    "small_amethyst_bud", "medium_amethyst_bud", "large_amethyst_bud", "fern", "firefly_bush",
    "dead_bush", "short_grass", "seagrass", "kelp", "lily_pad", "end_rod",
})
LIQUIDI = frozenset({"water", "lava", "flowing_water", "flowing_lava", "bubble_column"})

# Chi brucia alla luce del giorno (i non-morti normali) e in cosa si ripiega se non c'e'
# un posto al coperto: lo zombi diventa husk, lo scheletro e lo stray un predone (balestra),
# che non brucia. L'elmo non basta: in pochi secondi di sole si spacca.
BRUCIANO = frozenset({"zombie", "scheletro", "stray"})
CONVERSIONE = {"zombie": "husk", "scheletro": "predone", "stray": "predone"}
RAGGIO_COPERTURA = 14

# chi non si sposta
SPECIE_LIBERE = frozenset({"pipistrello", "merluzzo", "salmone", "pesce_tropicale", "calamaro",
                           "delfino", "pesce_palla", "calamaro_luminoso", "tartaruga", "ape",
                           "pappagallo", "rana"})


def _libera_da_spostare(e) -> bool:
    """Vero se la creatura va controllata (cammina sul terreno)."""
    if not dataclasses.is_dataclass(e):
        return False
    if getattr(e, "specie", None) in SPECIE_LIBERE:
        return False
    if type(e).__name__ in ("Carrello", "Manichino"):
        return False
    return True


class _Chunk:
    """Un chunk letto: gli id dei blocchi e per ognuno se e' passabile / liquido."""

    def __init__(self, livello, cx: int, cz: int):
        ch = livello.get_chunk(cx, cz, DIMENSIONE)
        self.ids = np.asarray(ch.blocks[:, Y_MIN:Y_MAX, :])
        pal = livello.block_palette
        self.passabile: dict[int, bool] = {}
        self.liquido: dict[int, bool] = {}
        for bid in np.unique(self.ids):
            nome = pal[int(bid)].base_name
            self.passabile[int(bid)] = nome in PASSABILI
            self.liquido[int(bid)] = nome in LIQUIDI

    def blocco(self, lx: int, y: int, lz: int):
        i = y - Y_MIN
        if not (0 <= i < self.ids.shape[1]):
            return None
        return int(self.ids[lx, i, lz])

    def coperto(self, lx: int, y: int, lz: int) -> bool:
        """C'e' un blocco pieno (o liquido) sopra la testa, a y + 2 o piu' in alto: il sole
        non arriva. Le foglie contano (fermano il cielo come un tetto)."""
        if not hasattr(self, "_pieni"):
            lut = np.zeros(int(self.ids.max()) + 1, bool)
            for b, libero in self.passabile.items():
                lut[b] = not libero
            pieno = lut[self.ids]
            # quanti blocchi pieni stanno da qui in su
            self._pieni = np.flip(np.cumsum(np.flip(pieno, axis=1), axis=1), axis=1)
        i = y + 2 - Y_MIN
        if i >= self._pieni.shape[1]:
            return False
        return bool(self._pieni[lx, i, lz] > 0)


class Sicurezza:
    def __init__(self, percorso: str):
        import amulet
        self.livello = amulet.load_level(percorso)
        self._chunk: dict[tuple[int, int], _Chunk | None] = {}

    def chiudi(self) -> None:
        self.livello.close()

    def chunk(self, cx: int, cz: int):
        if (cx, cz) not in self._chunk:
            try:
                self._chunk[(cx, cz)] = _Chunk(self.livello, cx, cz)
            except Exception:                                    # noqa: BLE001 - chunk assente
                self._chunk[(cx, cz)] = None
        return self._chunk[(cx, cz)]

    def _cella(self, x: int, y: int, z: int):
        ch = self.chunk(x // 16, z // 16)
        if ch is None:
            return None
        b = ch.blocco(x % 16, y, z % 16)
        return None if b is None else (ch, b)

    def passabile(self, x: int, y: int, z: int):
        c = self._cella(x, y, z)
        return None if c is None else c[0].passabile[c[1]]

    def valida(self, x: int, y: int, z: int) -> bool:
        """Cella libera, cella sopra libera, e un appoggio pieno (non liquido) sotto."""
        a, b = self.passabile(x, y, z), self.passabile(x, y + 1, z)
        sotto = self._cella(x, y - 1, z)
        if not a or not b or sotto is None:
            return False
        ch, bid = sotto
        return (not ch.passabile[bid]) and (not ch.liquido[bid])

    def al_coperto(self, x: int, y: int, z: int) -> bool:
        ch = self.chunk(x // 16, z // 16)
        return ch is not None and ch.coperto(x % 16, y, z % 16)

    def trova(self, x: int, y: int, z: int, raggio: int = 3, su: int = 24, coperto: bool = False):
        """La cella valida piu' vicina a (x, y, z), o None. Con `coperto` anche al riparo
        dal sole."""
        def buona(a, b, c):
            return self.valida(a, b, c) and (not coperto or self.al_coperto(a, b, c))
        if buona(x, y, z):
            return (x, y, z)
        migliore = None
        for dy in range(-6 if not coperto else -8, 7 if not coperto else 9):
            for dx, dz in itertools.product(range(-raggio, raggio + 1), repeat=2):
                costo = abs(dy) * 2 + (dx * dx + dz * dz) ** 0.5
                if migliore is not None and costo >= migliore[0]:
                    continue
                if buona(x + dx, y + dy, z + dz):
                    migliore = (costo, (x + dx, y + dy, z + dz))
        if migliore is not None:
            return migliore[1]
        if coperto:
            return None
        for dy in range(7, su + 1):                              # verso l'alto, nella stessa colonna
            if self.valida(x, y + dy, z):
                return (x, y + dy, z)
        return None


def rendi_sicure(percorso_mondo: str, entita: list) -> tuple[list, dict]:
    """Ritorna (entita' con le posizioni corrette, statistiche). Coordinate di
    GIOCO. Le entita' sono dataclass con x, y, z; le altre non si toccano."""
    st = {"controllate": 0, "spostate": 0, "irrecuperabili": 0, "al_coperto": 0, "convertite": 0}
    controllo = [i for i, e in enumerate(entita) if _libera_da_spostare(e)]
    if not controllo:
        return list(entita), st
    s = Sicurezza(percorso_mondo)
    fuori = list(entita)
    try:
        for i in controllo:
            e = entita[i]
            x, y, z = int(np.floor(e.x)), int(np.floor(e.y + 1e-6)), int(np.floor(e.z))
            if s.chunk(x // 16, z // 16) is None:
                continue
            st["controllate"] += 1
            corrente = e
            if not s.valida(x, y, z):
                nuova = s.trova(x, y, z)
                if nuova is None:
                    st["irrecuperabili"] += 1
                    continue
                x, y, z = nuova
                corrente = dataclasses.replace(e, x=x + 0.5, y=float(y), z=z + 0.5)
                fuori[i] = corrente
                st["spostate"] += 1
            # i non-morti bruciano al sole: devono stare al coperto (torre, sala, cripta)
            if getattr(corrente, "specie", None) in BRUCIANO and not s.al_coperto(x, y, z):
                riparo = s.trova(x, y, z, raggio=RAGGIO_COPERTURA, coperto=True)
                if riparo is not None:
                    fuori[i] = dataclasses.replace(corrente, x=riparo[0] + 0.5, y=float(riparo[1]),
                                                   z=riparo[2] + 0.5)
                    st["al_coperto"] += 1
                elif not getattr(corrente, "capo", False):
                    fuori[i] = dataclasses.replace(corrente, specie=CONVERSIONE[corrente.specie])
                    st["convertite"] += 1
    finally:
        s.chiudi()
    return fuori, st
