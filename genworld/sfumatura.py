"""Sfumatura del bordo: il terreno generato dal gioco si raccorda al nostro.

Oltre la mappa il mondo continua da solo (`livello_dat`, `fuori_mappa`). Senza
accorgimenti, il generatore vanilla non sa nulla dei nostri chunk: il primo
chunk nuovo accanto al bordo ha il SUO terreno e fra i due c'e' un taglio
verticale.

Minecraft ha gia' il rimedio per i mondi aggiornati da versioni vecchie: un
chunk con il tag `blending_data` dice al generatore "sono un chunk vecchio,
sfuma il terreno nuovo verso di me". Qui si mette quel tag sui chunk piu'
esterni. Si fa a mondo finito, riscrivendo i file region: amulet non sa
scrivere quel tag.
"""
from __future__ import annotations

import glob
import os
import struct
import zlib

SETTORE = 4096
SEZIONE_MIN = -4         # -64 / 16
SEZIONE_MAX = 20         # 320 / 16, escluso
ANELLI = 3               # file di chunk piu' esterni che portano il tag


def _leggi_regione(percorso: str) -> dict | None:
    """{(cx, cz): (nbt non compresso, marca temporale)} o None se il formato
    non e' quello che sappiamo riscrivere (compressione diversa da zlib)."""
    with open(percorso, "rb") as f:
        dati = f.read()
    nome = os.path.basename(percorso).split(".")
    rx, rz = int(nome[1]), int(nome[2])
    fuori = {}
    for i in range(1024):
        off = int.from_bytes(dati[i * 4:i * 4 + 3], "big")
        if not off:
            continue
        pos = off * SETTORE
        lunghezza = struct.unpack(">I", dati[pos:pos + 4])[0]
        if dati[pos + 4] != 2:
            return None
        grezzo = zlib.decompress(dati[pos + 5:pos + 4 + lunghezza])
        data = int.from_bytes(dati[SETTORE + i * 4:SETTORE + i * 4 + 4], "big")
        fuori[(rx * 32 + i % 32, rz * 32 + i // 32)] = (grezzo, data)
    return fuori


def _scrivi_regione(percorso: str, chunk: dict) -> None:
    posizioni = bytearray(SETTORE)
    date = bytearray(SETTORE)
    corpo = bytearray()
    settore = 2
    for (cx, cz), (grezzo, data) in sorted(chunk.items()):
        compresso = zlib.compress(grezzo)
        blocco = len(compresso) + 5
        quanti = (blocco + SETTORE - 1) // SETTORE
        corpo += (len(compresso) + 1).to_bytes(4, "big") + b"\x02" + compresso
        corpo += b"\x00" * (quanti * SETTORE - blocco)
        i = ((cx & 31) + (cz & 31) * 32) * 4
        posizioni[i:i + 3] = settore.to_bytes(3, "big")
        posizioni[i + 3] = quanti
        date[i:i + 4] = data.to_bytes(4, "big")
        settore += quanti
    with open(percorso, "wb") as f:
        f.write(posizioni)
        f.write(date)
        f.write(corpo)


def _con_tag(grezzo: bytes) -> bytes:
    import amulet_nbt as nbt
    tag = nbt.load(grezzo, compressed=False, little_endian=False)
    tag.compound["blending_data"] = nbt.CompoundTag({
        "min_section": nbt.IntTag(SEZIONE_MIN),
        "max_section": nbt.IntTag(SEZIONE_MAX),
    })
    return tag.save_to(compressed=False, little_endian=False)


def segna_bordo(percorso_mondo: str, anelli: int = ANELLI,
                dimensione: str = "minecraft/overworld") -> dict:
    """Mette `blending_data` sui chunk che hanno, entro `anelli` chunk, almeno
    un vicino mancante. Ritorna {"chunk": quanti, "segnati": quanti, "file": n}."""
    # dalla 26.1 le dimensioni stanno in `dimensions/<namespace>/<nome>/region`;
    # prima l'overworld era `region` nella radice del mondo
    files = sorted(glob.glob(os.path.join(percorso_mondo, "dimensions",
                                          *dimensione.split("/"), "region", "r.*.*.mca")))
    if not files and dimensione == "minecraft/overworld":
        files = sorted(glob.glob(os.path.join(percorso_mondo, "region", "r.*.*.mca")))
    regioni = {}
    for f in files:
        dati = _leggi_regione(f)
        if dati is None:
            continue
        regioni[f] = dati
    presenti = {c for dati in regioni.values() for c in dati}
    segnati = 0
    for f, dati in regioni.items():
        cambiato = False
        for (cx, cz), (grezzo, data) in list(dati.items()):
            bordo = any((cx + dx, cz + dz) not in presenti
                        for dx in range(-anelli, anelli + 1)
                        for dz in range(-anelli, anelli + 1))
            if bordo:
                dati[(cx, cz)] = (_con_tag(grezzo), data)
                segnati += 1
                cambiato = True
        if cambiato:
            _scrivi_regione(f, dati)
    return {"chunk": len(presenti), "segnati": segnati, "file": len(regioni)}
