"""Esporta le case parametriche come file `.nbt` di blocco struttura.

Serve a due cose. La prima e' avere qualcosa in `templates/case/` appena si
scarica il progetto, cosi' la generazione da template funziona subito invece
di funzionare "dopo che ti sei costruito dieci case". La seconda, piu' utile,
e' dare un punto di partenza da modificare: si carica il .nbt in Minecraft con
un blocco struttura, si sistema quello che non piace - ed e' molto piu' facile
sistemare una casa che disegnarne una da zero - e si risalva con lo stesso
nome.

    python esempi/esporta_template.py
    python esempi/esporta_template.py --quante 12 --cartella templates/case
"""

import argparse
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import edifici as E  # noqa: E402

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONE = (1, 21, 4)
DATA_VERSION = 4189


class TavolozzaRaccolta:
    """Finge di essere il livello aperto e si limita a raccogliere i blocchi."""

    def __init__(self):
        self.voci: list[tuple[str, tuple]] = []
        self._indice: dict = {}
        self.aria = self._id("air", {})
        self.blocco = {}
        for nome, p in E.PALETTE.items():
            self.blocco[(nome, "muro")] = self._id(p.muro[0], p.muro[1])
            self.blocco[(nome, "telaio")] = self._id(p.telaio[0], p.telaio[1])
            self.blocco[(nome, "pavimento")] = self._id(p.pavimento[0], p.pavimento[1])
            self.blocco[(nome, "basamento")] = self._id(p.basamento[0], p.basamento[1])
            self.blocco[(nome, "palo")] = self._id(
                "log", {"axis": "y", "material": p.legno, "stripped": "true"})
        self.vetro = self._id("glass_pane", dict(north="false", south="false",
                                                 east="false", west="false"))
        self.vetro_pieno = self._id("glass", {})
        self.torcia = self._id("torch", {"facing": "up"})

    def _id(self, nome: str, prop: dict) -> int:
        chiave = (nome, tuple(sorted(prop.items())))
        if chiave not in self._indice:
            self._indice[chiave] = len(self.voci)
            self.voci.append((nome, chiave[1]))
        return self._indice[chiave]

    # l'interfaccia che `edifici.costruisci` si aspetta
    def b(self, nome, **prop): return self._id(nome, prop)
    def scala(self, materiale, verso, meta="bottom"):
        return self._id("stairs", dict(facing=E.DIREZIONE[verso], half=meta,
                                       material=materiale, shape="straight"))
    def porta(self, materiale, verso, meta):
        return self._id("door", dict(facing=E.DIREZIONE[verso], half=meta,
                                     hinge="left", material=materiale,
                                     open="false", powered="false"))
    def staccionata(self, materiale):
        return self._id("fence", dict(material=materiale, north="false",
                                      south="false", east="false", west="false"))
    def trave(self, materiale, asse):
        return self._id("log", dict(axis=asse, material=materiale,
                                    stripped="false"))


def a_minecraft(voci, versione=VERSIONE):
    """Dal namespace universale ai nomi di gioco, che e' quello che legge il
    blocco struttura."""
    import PyMCTranslate
    from amulet.api.block import Block
    from amulet_nbt import StringTag

    ver = PyMCTranslate.new_translation_manager().get_version("java", versione)
    fuori = []
    for nome, prop in voci:
        b = Block("universal_minecraft", nome,
                  {k: StringTag(v) for k, v in prop})
        gioco = ver.block.from_universal(b)[0]
        fuori.append((gioco.namespaced_name,
                      {k: str(v.py_str if hasattr(v, "py_str") else v)
                       for k, v in gioco.properties.items()}))
    return fuori


def scrivi_nbt(percorso, celle, voci_gioco, aria_id):
    """Il formato del blocco struttura: size, palette, blocks."""
    from amulet_nbt import (CompoundTag, IntTag, ListTag, NamedTag, StringTag)

    dx, dy, dz = celle.shape
    palette = ListTag([
        CompoundTag({"Name": StringTag(nome)} | (
            {"Properties": CompoundTag({k: StringTag(v) for k, v in prop.items()})}
            if prop else {}))
        for nome, prop in voci_gioco])

    blocchi = []
    for px in range(dx):
        for py in range(dy):
            for pz in range(dz):
                stato = int(celle[px, py, pz])
                blocchi.append(CompoundTag({
                    "state": IntTag(stato),
                    "pos": ListTag([IntTag(px), IntTag(py), IntTag(pz)]),
                }))

    radice = CompoundTag({
        "DataVersion": IntTag(DATA_VERSION),
        "size": ListTag([IntTag(dx), IntTag(dy), IntTag(dz)]),
        "palette": palette,
        "blocks": ListTag(blocchi),
        "entities": ListTag([]),
    })
    grezzo = NamedTag(radice, "").save_to(compressed=False)
    with gzip.open(percorso, "wb") as f:
        f.write(grezzo)


def una_casa(larg, prof, piani, stile, mestiere, seme):
    """Costruisce la casa in un array e lo ritaglia sull'ingombro vero."""
    tav = TavolozzaRaccolta()
    margine, alto = 4, 40
    ed = E.Edificio(x=margine, z=margine, larghezza=larg, profondita=prof,
                    base=8, piani=piani, stile=stile, gronda=1,
                    mestiere=mestiere, seme=seme)
    lato_x = larg + 2 * margine
    lato_z = prof + 2 * margine
    out = np.full((lato_x, alto, lato_z), tav.aria, np.int32)
    E.costruisci(out, 0, 0, 0, ed, tav)

    # ritaglio: via l'aria di contorno, ma si tiene il piano di posa a y=base-1
    pieno = out != tav.aria
    if not pieno.any():
        raise RuntimeError("casa vuota")
    xs = np.flatnonzero(pieno.any(axis=(1, 2)))
    zs = np.flatnonzero(pieno.any(axis=(0, 1)))
    ys = np.flatnonzero(pieno.any(axis=(0, 2)))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    z0, z1 = int(zs.min()), int(zs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return tav, out[x0:x1, y0:y1, z0:z1], ed.base - y0


MODELLI = [
    # (larghezza, profondita', piani, stile, mestiere, nome)
    # Le misure piccole ci sono perche' servono: i lotti di una pianta
    # organica sono quasi tutti piccoli, e un catalogo di sole case grandi
    # lascia tre quarti del paese al generatore parametrico.
    (5, 5, 1, "prato", "", "capanna_prato"),
    (6, 5, 1, "bosco", "", "capanna_bosco"),
    (6, 6, 1, "prato", "fruttivendolo", "bottega_frutta"),
    (7, 6, 2, "bosco", "", "casetta_due_piani"),
    (7, 7, 1, "prato", "", "casetta_prato"),
    (9, 7, 1, "bosco", "", "casa_bosco"),
    (9, 8, 2, "prato", "", "casa_due_piani"),
    (11, 8, 2, "bosco", "falegname", "bottega_falegname"),
    (9, 9, 2, "montagna", "scalpellino", "bottega_scalpellino"),
    (11, 9, 3, "prato", "", "casa_alta"),
    (8, 8, 1, "deserto", "", "casa_deserto"),
    (10, 8, 2, "montagna", "", "casa_montagna"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartella", default=os.path.join(RADICE, "templates", "case"))
    ap.add_argument("--quante", type=int, default=len(MODELLI))
    args = ap.parse_args()

    os.makedirs(args.cartella, exist_ok=True)
    for i, (larg, prof, piani, stile, mest, nome) in enumerate(MODELLI[:args.quante]):
        tav, celle, _base = una_casa(larg, prof, piani, stile, mest, 1000 + i)
        voci = a_minecraft(tav.voci)
        percorso = os.path.join(args.cartella, f"{nome}.nbt")
        scrivi_nbt(percorso, celle, voci, tav.aria)
        print(f"{nome}.nbt  {celle.shape[0]}x{celle.shape[1]}x{celle.shape[2]}")
    print(f"\n{args.quante} template in {args.cartella}")


if __name__ == "__main__":
    main()
