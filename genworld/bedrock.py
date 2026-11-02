"""Da un mondo Java generato a un mondo Bedrock (`.mcworld`).

Amulet sa scrivere il formato LevelDB di Bedrock e traduce i blocchi dal formato universale:
qui si copiano i chunk dell'overworld e dell'End dal mondo Java al nuovo mondo e lo si
impacchetta in uno zip con estensione `.mcworld`, che Minecraft Bedrock apre con un doppio clic.

Limiti, che dipendono da Bedrock e non da noi:
* il Nether di Bedrock e' alto 128 blocchi: quello di Arda (256) e il Sottosopra (che usa un
  datapack) non si esportano; in Bedrock si va nel Nether vanilla;
* i pacchetti di dati (datapack) non esistono: niente biomi o dimensioni personalizzati;
* le creature (abitanti, mostri, carrelli) stanno nei file `entities/` di Java e non si
  copiano: restano i blocchi e i contenitori (forzieri, spawner) con il loro contenuto.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile

DIMENSIONI = ("minecraft:overworld", "minecraft:the_end")
VERSIONE_BEDROCK = (1, 21, 50)


def esporta(mondo_java: str, uscita: str, nome: str | None = None, ogni: int = 1500,
            avanza=None, versione: tuple = VERSIONE_BEDROCK) -> dict:
    """Converte `mondo_java` in `uscita` (un `.mcworld`, oppure una cartella se non finisce
    cosi'). Ritorna {dimensione: chunk copiati}."""
    import amulet
    from amulet.api.level import World
    from amulet.api.selection import SelectionBox, SelectionGroup
    from amulet.level.formats.leveldb_world import LevelDBFormat

    zip_finale = uscita.lower().endswith(".mcworld")
    cartella = tempfile.mkdtemp(prefix="genworld_bedrock_") if zip_finale else uscita
    if not zip_finale and os.path.exists(cartella):
        shutil.rmtree(cartella)
    lv = amulet.load_level(mondo_java)
    copiati: dict = {}
    try:
        presenti = [d for d in DIMENSIONI if d in lv.dimensions]
        limiti = {}
        for d in presenti:
            b = lv.bounds(d)
            lo, hi = b.min, b.max
            limiti[d] = SelectionGroup([SelectionBox(
                (-30000000, int(lo[1]), -30000000), (30000000, int(hi[1]), 30000000))])
        wr = LevelDBFormat(cartella)
        wr.create_and_open("bedrock", versione, limiti, overwrite=True)
        wr.save()
        wr.close()
        bd = World(cartella, LevelDBFormat(cartella))
        try:
            totale = sum(len(list(lv.all_chunk_coords(d))) for d in presenti)
            fatti = 0
            for d in presenti:
                n = 0
                for cx, cz in lv.all_chunk_coords(d):
                    bd.put_chunk(lv.get_chunk(cx, cz, d), d)
                    n += 1
                    fatti += 1
                    if n % ogni == 0:
                        bd.save()
                        if avanza:
                            avanza(fatti, totale)
                copiati[d] = n
            bd.save()
        finally:
            bd.close()
    finally:
        lv.close()
    if nome:
        with open(os.path.join(cartella, "levelname.txt"), "w", encoding="utf-8") as f:
            f.write(nome)
    if zip_finale:
        if os.path.exists(uscita):
            os.remove(uscita)
        with zipfile.ZipFile(uscita, "w", zipfile.ZIP_DEFLATED) as z:
            for radice, _, files in os.walk(cartella):
                for fn in files:
                    p = os.path.join(radice, fn)
                    z.write(p, os.path.relpath(p, cartella))
        shutil.rmtree(cartella, ignore_errors=True)
    return copiati
