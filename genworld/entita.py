"""Abitanti: scrittura a mano dei file `entities/*.mca`.

Perche' a mano. Amulet non serializza le entita': si puo' riempire
`chunk.entities` con oggetti Entity perfettamente formati e quello che finisce
sul disco resta un chunk-entita' di 22 byte con dentro solo `DataVersion: 0`.
E' lo stesso file che faceva scorrere all'infinito

    Caricamento del chunk in [-11, -6] non riuscito

e che abbiamo risolto cancellando la cartella. Ora la cartella serve, quindi
va scritta bene: formato Anvil, intestazione di 8 KiB, settori da 4096 byte,
NBT compresso con zlib, e dentro `Position` ed `Entities` come li vuole
Minecraft.

E' la seconda volta che tocca rifare da zero quello che amulet scrive male -
la prima era il `level.dat`. Il criterio e' sempre lo stesso: se una cosa la
legge Minecraft, la si scrive nel modo in cui Minecraft la legge, e la si
rilegge con un parser proprio per essere sicuri.
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass, field

SETTORE = 4096

# amulet_nbt si importa dentro le funzioni che lo usano: la tabella dei
# mestieri serve anche a `citta.py`, che gira benissimo senza mezza libreria
# di Minecraft installata - e i suoi test pure.

# Mestieri di Minecraft, con il loro banco di lavoro.
# I nomi a sinistra sono i nostri, quelli a destra sono di gioco: cosi' il
# resto del programma parla italiano e la traduzione sta in un posto solo.
MESTIERI = {
    "fabbro":        ("toolsmith",    "smithing_table"),
    "armaiolo":      ("weaponsmith",  "grindstone"),
    "corazzaio":     ("armorer",      "blast_furnace"),
    "falegname":     ("fletcher",     "fletching_table"),
    "scalpellino":   ("mason",        "stonecutter"),
    "macellaio":     ("butcher",      "smoker"),
    "fruttivendolo": ("farmer",       "composter"),
    "pescatore":     ("fisherman",    "barrel"),
    "pastore":       ("shepherd",     "loom"),
    "libraio":       ("librarian",    "lectern"),
    "cartografo":    ("cartographer", "cartography_table"),
    "conciatore":    ("leatherworker", "cauldron"),
    "speziale":      ("cleric",       "brewing_stand"),
}


@dataclass
class Abitante:
    """Un villager da scrivere nei file. Coordinate in blocchi del mondo."""
    x: float
    y: float
    z: float
    mestiere: str = "fruttivendolo"
    livello: int = 2
    seme: int = 0

    @property
    def professione(self) -> str:
        return MESTIERI.get(self.mestiere, ("none", ""))[0]


def _uuid(seme: int):
    """Quattro interi. Minecraft ne assegna uno se manca, ma un mondo con
    cento entita' senza UUID e' un mondo che si aspetta che il gioco faccia
    ordine per noi."""
    import random

    from amulet_nbt import IntArrayTag
    r = random.Random(seme)
    return IntArrayTag([r.randint(-2 ** 31, 2 ** 31 - 1) for _ in range(4)])


def nbt_abitante(a: Abitante, bioma: str = "plains"):
    from amulet_nbt import (ByteTag, CompoundTag, DoubleTag, FloatTag, IntTag,
                            ListTag, StringTag)
    return CompoundTag({
        "id": StringTag("minecraft:villager"),
        "Pos": ListTag([DoubleTag(a.x), DoubleTag(a.y), DoubleTag(a.z)]),
        "Motion": ListTag([DoubleTag(0.0), DoubleTag(0.0), DoubleTag(0.0)]),
        "Rotation": ListTag([FloatTag(0.0), FloatTag(0.0)]),
        "UUID": _uuid(a.seme),
        "Health": FloatTag(20.0),
        "Air": IntTag(300),
        "Fire": IntTag(-1),
        "FallDistance": FloatTag(0.0),
        "Invulnerable": ByteTag(0),
        "OnGround": ByteTag(1),
        # senza questo, un abitante lontano dal giocatore puo' sparire
        "PersistenceRequired": ByteTag(1),
        "VillagerData": CompoundTag({
            "level": IntTag(a.livello),
            "profession": StringTag(f"minecraft:{a.professione}"),
            "type": StringTag(f"minecraft:{bioma}"),
        }),
        "Xp": IntTag(0),
    })


def _chunk_nbt(cx: int, cz: int, abitanti: list[Abitante],
               data_version: int) -> bytes:
    from amulet_nbt import (CompoundTag, IntArrayTag, IntTag, ListTag,
                            NamedTag)
    radice = CompoundTag({
        "DataVersion": IntTag(data_version),
        # QUESTO e' il campo che mancava nei file scritti da amulet, ed e' da
        # qui che nasceva l'ArrayIndexOutOfBoundsException: Minecraft legge
        # pos[0] e trovava un array vuoto.
        "Position": IntArrayTag([cx, cz]),
        "Entities": ListTag([nbt_abitante(a) for a in abitanti]),
    })
    return NamedTag(radice, "").save_to(compressed=False)


def scrivi_regioni(percorso_mondo: str, abitanti: list[Abitante],
                   data_version: int) -> dict:
    """Scrive `entities/r.X.Z.mca` per tutti gli abitanti dati."""
    if not abitanti:
        return {"abitanti": 0, "chunk": 0, "regioni": 0}

    per_chunk: dict[tuple[int, int], list[Abitante]] = {}
    for a in abitanti:
        chiave = (int(a.x) >> 4, int(a.z) >> 4)
        per_chunk.setdefault(chiave, []).append(a)

    per_regione: dict[tuple[int, int], dict] = {}
    for (cx, cz), gruppo in per_chunk.items():
        per_regione.setdefault((cx >> 5, cz >> 5), {})[(cx, cz)] = gruppo

    cartella = os.path.join(percorso_mondo, "entities")
    os.makedirs(cartella, exist_ok=True)
    for (rx, rz), chunk in per_regione.items():
        _scrivi_regione(os.path.join(cartella, f"r.{rx}.{rz}.mca"),
                        chunk, data_version)
    return {"abitanti": len(abitanti), "chunk": len(per_chunk),
            "regioni": len(per_regione)}


def _scrivi_regione(percorso: str, chunk: dict, data_version: int) -> None:
    """Un file region: 4096 byte di posizioni, 4096 di date, poi i settori."""
    posizioni = bytearray(SETTORE)
    date = bytearray(SETTORE)
    corpo = bytearray()
    settore = 2                      # i primi due sono l'intestazione

    for (cx, cz), abitanti in sorted(chunk.items()):
        dati = zlib.compress(_chunk_nbt(cx, cz, abitanti, data_version))
        blocco = len(dati) + 5       # 4 di lunghezza + 1 di compressione
        quanti = (blocco + SETTORE - 1) // SETTORE
        corpo += (len(dati) + 1).to_bytes(4, "big") + b"\x02" + dati
        corpo += b"\x00" * (quanti * SETTORE - blocco)

        i = ((cx & 31) + (cz & 31) * 32) * 4
        posizioni[i:i + 3] = settore.to_bytes(3, "big")
        posizioni[i + 3] = quanti
        date[i:i + 4] = (0).to_bytes(4, "big")
        settore += quanti

    with open(percorso, "wb") as f:
        f.write(posizioni)
        f.write(date)
        f.write(corpo)


def leggi_chunk(percorso_mca: str, cx: int, cz: int):
    """Rilegge un chunk-entita' senza passare da amulet.

    Serve ai test: un controllo fatto con lo stesso strumento che ha scritto
    il file non e' un controllo. Qui non c'e' nemmeno lo strumento - il file
    lo scriviamo noi - quindi il parser indipendente e' obbligatorio.
    """
    with open(percorso_mca, "rb") as f:
        testa = f.read(SETTORE)
        i = ((cx & 31) + (cz & 31) * 32) * 4
        off = int.from_bytes(testa[i:i + 3], "big")
        if off == 0:
            return None
        f.seek(off * SETTORE)
        lung = int.from_bytes(f.read(4), "big")
        f.read(1)
        from amulet_nbt import load as nbt_load
        return nbt_load(zlib.decompress(f.read(lung - 1))).compound
