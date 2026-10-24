"""Le dimensioni del portale: il Nether e l'End di Arda scritti accanto all'overworld.

Si scrivono a mondo finito (vedi `motore.passaggi_finali`), PRIMA delle creature
dell'overworld: `ScrittoreMondo` a ogni chiusura butta le cartelle `entities/` perche'
amulet le scrive rotte, e le nostre devono arrivare dopo l'ultima chiusura.
"""
from __future__ import annotations

import math

from . import bauli as BA
from . import end as FINE
from . import entita as EN
from . import nether as NE

NETHER = "minecraft:the_nether"
END = "minecraft:the_end"


def _sistema_creature(creature: list, blocchi, ox: int, oz: int, aria: int) -> None:
    """Porta in alto chi sarebbe nato dentro un blocco (funghi giganti, rilievi)."""
    for c in creature:
        lx, lz = math.floor(c.x) - ox, math.floor(c.z) - oz
        if not (0 <= lx < 16 and 0 <= lz < 16):
            continue
        y = int(c.y)
        for _ in range(12):
            if 1 <= y < blocchi.shape[1] - 2 and blocchi[lx, y, lz] == aria and blocchi[lx, y + 1, lz] == aria:
                break
            y += 1
        c.y = float(y)


def _per_chunk(creature: list) -> dict:
    fuori: dict = {}
    for c in creature:
        fuori.setdefault((math.floor(c.x) >> 4, math.floor(c.z) >> 4), []).append(c)
    return fuori


def _chunk_isole(pe) -> set:
    fuori = set()
    for i in pe.isole:
        x0, z0, x1, z1 = i.bbox()
        for cz in range(z0 >> 4, (z1 >> 4) + 1):
            for cx in range(x0 >> 4, (x1 >> 4) + 1):
                fuori.add((cx, cz))
    return fuori


def scrivi(op) -> dict:
    """Scrive Nether ed End in `op.uscita`. Ritorna le statistiche e il testo per `luoghi.md`."""
    from .dungeon import be_spawner
    from .livello_dat import ImpostazioniMondo
    from .mondo import ScrittoreMondo

    imp = ImpostazioniMondo(nome=op.nome, modalita=1, spawn=(0, 150, 0),
                            versione=op.versione, seed=op.seed, fuori_mappa=op.fuori_mappa,
                            sottosopra=getattr(op, 'sottosopra', False))
    pn = NE.pianifica(op.lato, seed=op.seed)
    pe = FINE.pianifica(op.lato, seed=op.seed)
    guardie_n: list = []
    for s in pn.strutture:
        guardie_n += NE.disegna(s)[3]
    creature_n = pn.creature + guardie_n
    creature_e = pe.creature
    per_n, per_e = _per_chunk(creature_n), _per_chunk(creature_e)
    scritti_n = scritti_e = 0

    with ScrittoreMondo(op.uscita, imp, lato_blocchi=op.lato, crea=False) as m:
        tn = NE.Tavolozza(m)
        te = FINE.Tavolozza(m)
        id_n = [m.bioma(b) for b in NE.BIOMI]
        id_e = m.bioma("the_end")
        for cx, cz in ([] if getattr(op, "sottosopra", False) else m.chunk_coords()):
            blocchi, biomi, spawner, forzieri = NE.blocchi_chunk(pn, tn, cx, cz, id_n)
            ox, oz = cx * 16, cz * 16
            be = BA.trova(blocchi, tn.bauli, ox, oz, 0, seed=op.seed, dungeon=forzieri)
            be += [be_spawner(x, y, z, mob) for x, y, z, mob in spawner]
            _sistema_creature(per_n.get((cx, cz), []), blocchi, ox, oz, tn.id["aria"])
            m.scrivi_chunk(cx, cz, blocchi, y0=0, biomi=biomi, bauli=be, dimensione=NETHER)
            scritti_n += 1
        # l'End non dipende dalla grandezza della mappa: si scrivono i chunk delle isole
        for cx, cz in sorted(_chunk_isole(pe)):
            ox, oz = cx * 16, cz * 16
            r = FINE.blocchi_chunk(pe, te, cx, cz, id_e)
            if r is None:
                continue
            blocchi, biomi, forzieri = r
            be = BA.trova(blocchi, te.bauli, ox, oz, 0, seed=op.seed, dungeon=forzieri)
            _sistema_creature(per_e.get((cx, cz), []), blocchi, ox, oz, te.id["aria"])
            m.scrivi_chunk(cx, cz, blocchi, y0=0, biomi=biomi, bauli=be, dimensione=END)
            scritti_e += 1

    dv = imp.data_version
    if not getattr(op, "sottosopra", False):
        EN.scrivi_regioni(op.uscita, creature_n, dv, dimensione="the_nether")
    EN.scrivi_regioni(op.uscita, creature_e, dv, dimensione="the_end")
    return {"nether_chunk": scritti_n, "end_chunk": scritti_e,
            "nether_strutture": len(pn.strutture), "end_isole": len(pe.isole),
            "creature_dimensioni": len(creature_n) + len(creature_e),
            "md": NE.sezione_elenco(pn) + FINE.sezione_elenco(pe)}
