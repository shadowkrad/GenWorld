"""Mappe del tesoro (#7): una mappa vera, disegnata col nostro terreno.

Nei forzieri del tesoro di una rovina, di una torre, di una fortezza e di un
castello, accanto al libro che indica la tappa successiva (vedi `luoghi.catena`),
c'e' una MAPPA: il paesaggio attorno alla tappa, con una croce rossa sul posto.

Una mappa di Minecraft e' un oggetto `filled_map` con un numero (`map_id`) e un
file nel mondo con i 128 x 128 colori. Qui il file lo scriviamo noi, dal terreno
che abbiamo pianificato: classi, quote, livello dell'acqua, strade e case.

Formato (dalla documentazione del gioco; NON provato in gioco):
  - dal DataVersion 4786 (26.1+): `data/minecraft/maps/<id>.dat`, e il contatore
    `last_id.dat` (scritto sia in `data/minecraft/` sia in `data/`, perche' la
    documentazione non dice dove); prima: `data/map_<id>.dat` e `data/idcounts.dat`;
  - ogni file e' un NBT compresso: {data: {scale, dimension, locked, xCenter,
    zCenter, trackingPosition, unlimitedTracking, banners, frames, colors},
    DataVersion};
  - colore = colore_base * 4 + ombra, con ombra 0 = scuro, 1 = normale,
    2 = chiaro, 3 = scurissimo.
"""
from __future__ import annotations

import os

import numpy as np

from . import mappa as M

SCALA = 2                    # 4 blocchi per pixel: 512 x 512 blocchi di paesaggio
LATO = 128

# colori base (indice -> usato qui)
NESSUNO, ERBA, SABBIA, FUOCO, PIANTA, NEVE, ARGILLA, TERRA, PIETRA, ACQUA, LEGNO = (
    0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13)
GRIGIO, NERO, ROSSO_SCURO = 21, 29, 28

BASE_CLASSE = {
    M.OCEANO: ACQUA, M.MARE: ACQUA, M.FIUME: ACQUA,
    M.SPIAGGIA: SABBIA, M.DESERTO: SABBIA,
    M.PIANURA: ERBA, M.PRATERIA: ERBA, M.FORESTA: PIANTA,
    M.MONTAGNA: PIETRA, M.NEVE: NEVE, M.VULCANO: GRIGIO, M.CRATERE: FUOCO,
}


def colore(base: int, ombra: int) -> int:
    return base * 4 + ombra


def _a_byte(v: np.ndarray) -> np.ndarray:
    """Gli id di colore arrivano a 247: nel file sono byte CON segno."""
    return np.asarray(v, np.uint8).view(np.int8)


def disegna(cls: np.ndarray, h: np.ndarray, livello: np.ndarray | None, meta: int,
            cx: int, cz: int, edifici: list | None = None,
            tipo_strada: np.ndarray | None = None, destino: tuple | None = None,
            qui: tuple | None = None, scala: int = SCALA) -> np.ndarray:
    """I 16384 colori (int8) di una mappa centrata su (cx, cz) in coordinate di
    GIOCO. `destino` e `qui` sono (x, z) di gioco: la croce e il punto."""
    passo = 2 ** scala
    H, W = cls.shape
    j, i = np.mgrid[0:LATO, 0:LATO]
    gx = cx + (i - LATO // 2) * passo + passo // 2       # blocco di gioco al centro del pixel
    gz = cz + (j - LATO // 2) * passo + passo // 2
    mx, mz = gx + meta, gz + meta
    dentro = (mx >= 0) & (mx < W) & (mz >= 0) & (mz < H)
    sx, sz = np.clip(mx, 0, W - 1), np.clip(mz, 0, H - 1)
    classe = cls[sz, sx]
    quota = h[sz, sx].astype(np.float32)
    nord = h[np.clip(sz - passo, 0, H - 1), sx].astype(np.float32)

    base = np.zeros((LATO, LATO), np.int32)
    for k, b in BASE_CLASSE.items():
        base[classe == k] = b
    ombra = np.full((LATO, LATO), 1, np.int32)

    # rilievo: piu' alto del vicino a nord = chiaro, piu' basso = scuro
    d = quota - nord
    ombra = np.where(d >= 3, 2, np.where(d >= -1, 1, np.where(d >= -4, 0, 3)))

    # acqua: piu' profonda = piu' scura
    acqua = base == ACQUA
    if livello is not None:
        pelo = np.maximum(livello[sz, sx], 62)
    else:
        pelo = np.full(quota.shape, 62.0)
    prof = np.clip(pelo - quota, 0, None)
    ombra = np.where(acqua, np.where(prof <= 3, 2, np.where(prof <= 10, 1, 0)), ombra)

    # strade e case, dal piano
    if tipo_strada is not None:
        base = np.where((tipo_strada[sz, sx] > 0) & ~acqua, TERRA, base)
    if edifici:
        e = np.zeros((H, W), bool)
        for ed in edifici:
            e[max(0, ed.z):ed.z + ed.profondita, max(0, ed.x):ed.x + ed.larghezza] = True
        base = np.where(e[sz, sx] & ~acqua, LEGNO, base)
        ombra = np.where(e[sz, sx] & ~acqua, 2, ombra)

    base = np.where(dentro, base, NESSUNO)
    out = np.where(base == NESSUNO, 0, base * 4 + ombra).astype(np.int32)

    def pixel(x, z):
        return (int(round((x - cx - passo // 2) / passo)) + LATO // 2,
                int(round((z - cz - passo // 2) / passo)) + LATO // 2)

    def metti(p, q, c):
        if 0 <= p < LATO and 0 <= q < LATO:
            out[q, p] = c

    if qui is not None:                                  # un punto: dov'e' il forziere
        p, q = pixel(*qui)
        for dp in (-1, 0, 1):
            for dq in (-1, 0, 1):
                metti(p + dp, q + dq, colore(NERO, 1))
        metti(p, q, colore(NEVE, 2))
    if destino is not None:                              # la croce
        p, q = pixel(*destino)
        for r in range(-5, 6):
            for spessore in (-1, 0, 1):                  # contorno scuro
                metti(p + r + spessore, q + r, colore(NERO, 1))
                metti(p + r + spessore, q - r, colore(NERO, 1))
        for r in range(-5, 6):
            metti(p + r, q + r, colore(FUOCO, 2))
            metti(p + r, q - r, colore(FUOCO, 2))
    return _a_byte(out.reshape(-1))


def scrivi_file(cartella_mondo: str, numero: int, colori: np.ndarray, cx: int, cz: int,
                data_version: int, scala: int = SCALA) -> str:
    """Scrive la mappa `numero`. Ritorna il percorso del file."""
    from amulet_nbt import (ByteArrayTag, ByteTag, CompoundTag, IntTag, ListTag,
                            NamedTag, StringTag)
    nuovo = data_version >= 4786
    if nuovo:
        dove = os.path.join(cartella_mondo, "data", "minecraft", "maps")
        percorso = os.path.join(dove, f"{numero}.dat")
    else:
        dove = os.path.join(cartella_mondo, "data")
        percorso = os.path.join(dove, f"map_{numero}.dat")
    os.makedirs(dove, exist_ok=True)
    dati = CompoundTag({
        "scale": ByteTag(scala),
        "dimension": StringTag("minecraft:overworld"),
        "trackingPosition": ByteTag(0),
        "unlimitedTracking": ByteTag(0),
        "locked": ByteTag(1),                    # non si aggiorna e non si ridimensiona
        "xCenter": IntTag(cx),
        "zCenter": IntTag(cz),
        "banners": ListTag([], 10),
        "frames": ListTag([], 10),
        "colors": ByteArrayTag(np.asarray(colori, np.int8)),
    })
    NamedTag(CompoundTag({"data": dati, "DataVersion": IntTag(data_version)}), "").save_to(
        percorso, compressed=True)
    return percorso


def scrivi_ultimo(cartella_mondo: str, ultimo: int, data_version: int) -> str:
    """Il contatore delle mappe: la prossima creata in gioco avra' `ultimo + 1`."""
    from amulet_nbt import CompoundTag, IntTag, NamedTag
    nuovo = data_version >= 4786
    dove = os.path.join(cartella_mondo, "data", "minecraft") if nuovo else os.path.join(cartella_mondo, "data")
    percorso = os.path.join(dove, "last_id.dat" if nuovo else "idcounts.dat")
    os.makedirs(dove, exist_ok=True)
    contenuto = NamedTag(CompoundTag({"data": CompoundTag({"map": IntTag(ultimo)}),
                                      "DataVersion": IntTag(data_version)}), "")
    contenuto.save_to(percorso, compressed=True)
    if nuovo:                                  # per sicurezza anche nella vecchia posizione
        contenuto.save_to(os.path.join(cartella_mondo, "data", "last_id.dat"), compressed=True)
    return percorso


def oggetto_mappa(numero: int, luogo: str, uid: str, verso: str = "") -> dict:
    """La descrizione dell'oggetto per `bauli`: una `filled_map` con il suo numero."""
    return {"id": "filled_map", "nome": f"Mappa: {luogo}"[:48],
            "lore": [f"Una croce rossa segna {luogo}.", verso] if verso else [f"Una croce rossa segna {luogo}."],
            "uid": uid, "map_id": numero}


def pianifica(luoghi: list, tappe_forzieri: dict, meta: int) -> list[dict]:
    """Una mappa per ogni tappa che ha una tappa successiva.

    `luoghi`: dal piano (coordinate di gioco). `tappe_forzieri`: {(tipo, x, z) di
    gioco: (x, y, z) di gioco del forziere con il libro}, cioe' dove cade il libro
    della tappa. Ritorna dizionari {numero, chiave (x,y,z), centro, destino, qui,
    luogo, oggetto}."""
    from . import luoghi as LU
    partenza, tappe = LU.catena(luoghi)
    fuori = []
    for i, l in enumerate(tappe[:-1]):
        sede = tappe_forzieri.get((l.tipo, l.x, l.z))
        if sede is None:
            continue
        prossima = tappe[i + 1]
        numero = len(fuori) + 1
        uid = f"mappa{numero}"
        fuori.append({
            "numero": numero, "chiave": tuple(sede), "centro": (prossima.x, prossima.z),
            "destino": (prossima.x, prossima.z), "qui": (sede[0], sede[2]),
            "luogo": prossima.nome,
            "oggetto": oggetto_mappa(numero, prossima.nome, uid,
                                     LU._passi(l, prossima)),
        })
    return fuori


def scrivi_tutte(cartella_mondo: str, mappe: list[dict], cls, h, livello, meta: int,
                 data_version: int, edifici=None, tipo_strada=None) -> int:
    """Scrive i file delle mappe e il contatore. Ritorna quante."""
    if not mappe:
        return 0
    for m in mappe:
        cx, cz = m["centro"]
        colori = disegna(cls, h, livello, meta, cx, cz, edifici, tipo_strada,
                         destino=m["destino"], qui=m["qui"])
        scrivi_file(cartella_mondo, m["numero"], colori, cx, cz, data_version)
    scrivi_ultimo(cartella_mondo, max(m["numero"] for m in mappe), data_version)
    return len(mappe)
