"""Scrive la checklist di controllo in gioco di un mondo generato.

Legge `luoghi.json` (scritto dalla generazione accanto al mondo) e la cache
dell'analisi (`<mondo>_cache.npz`, per i biomi) e produce un file markdown con
i comandi `/tp` per ogni cosa da guardare: i luoghi con nome e un punto per
bioma. Uso:

    python esempi/crea_checklist.py mondi/mappa_arda --uscita docs/checklist_in_gioco.md
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import biomi as B  # noqa: E402

BIOMI_DA_VEDERE = (
    ("badlands", "sabbia rossa e fasce di terracotta"),
    ("mangrove_swamp", "mangrovie con radici, fango, rane"),
    ("savanna", "acacie piatte, lama, asini"),
    ("jungle", "alberi altissimi, bambu', liane, panda"),
    ("taiga", "abeti, felci grandi"),
    ("snowy_tundra", "neve sottile sugli alberi, stray"),
    ("dark_forest", "rovere scuro, funghi giganti"),
    ("meadow", "ciliegi, lilla', rose, peonie, api"),
    ("sunflower_plains", "girasoli"),
    ("warm_ocean", "coralli e pesci tropicali"),
    ("desert", "cactus, cammelli, husk negli accampamenti"),
)


def punto_per_bioma(biomi, h, nome, meta):
    m = biomi == B.INDICE[nome]
    z, x = np.nonzero(m)
    if not len(x):
        return None
    i = int(np.argmin((x - x.mean()) ** 2 + (z - z.mean()) ** 2))
    return int(x[i]) - meta, int(h[z[i], x[i]]) + 6, int(z[i]) - meta, int(len(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mondo")
    ap.add_argument("--uscita", default="")
    a = ap.parse_args()
    luoghi = json.load(open(os.path.join(a.mondo, "luoghi.json"), encoding="utf-8"))
    righe = [f"# Checklist di controllo in gioco - {os.path.basename(a.mondo.rstrip('/'))}", "",
             "Per ogni punto: `/tp @s x y z` in creativa o spettatore, guarda, segna OK o difetto.", ""]

    def sezione(titolo, tipo, cosa):
        elenco = [l for l in luoghi if l["tipo"] == tipo]
        if not elenco:
            return
        righe.extend([f"## {titolo}", "", cosa, ""])
        for l in sorted(elenco, key=lambda l: l["nome"]):
            y = (l["y"] or 80) + 6
            righe.append(f"- [ ] **{l['nome']}** - `/tp @s {l['x']} {y} {l['z']}`"
                         + (f" ({l['nota']})" if l.get("nota") else ""))
        righe.append("")

    sezione("Torri con un capo", "torre",
            "Torre di pietra a quattro piani: porta, scala a pioli, capo con nome in cima, due guardie, "
            "due forzieri; cartello davanti.")
    sezione("Incarichi", "citta",
            "In ogni citta' un abitante con un nome sopra la testa, su una strada vicino alla piazza "
            "(l'elenco e' in luoghi.md, sezione Incarichi). Comprare il libro (1 smeraldo) dice dove "
            "andare. Il capo della torre/fortezza/castello lascia cadere un oggetto con nome e "
            "descrizione (lingotto d'oro); la reliquia (scheggia di ametista) sta nel forziere del "
            "tesoro della rovina; la lettera (carta) nella cella del castello. Riportandolo al "
            "committente lo scambio e' attivo solo con QUELL'oggetto (un lingotto qualunque non vale) "
            "e paga smeraldi o diamanti. Controllare che il committente non perda la professione.")
    sezione("Castelli in rovina", "castello",
            "Cinta con brecce e torri spezzate, portale sfondato, cortile invaso da erba, macerie, ossa, "
            "alberi morti e tombe; il maschio sventrato con trono (il capo siede li'), tavola rotta, "
            "scala rovinata; torre dello spawner (SW). Dalla sala, a sinistra, una scala scende alla "
            "cripta: anticamera, DUE fili d'inciampo con distributori di frecce, fossa con le punte "
            "(sponda stretta su un lato), sala dello spawner, celle con prigionieri, caveau con piastre "
            "sul TNT, forzieri-esca e il tesoro. Il cartello e' fuori dal portale.")
    sezione("Fortezze", "fortezza",
            "Cinta di pietra 29x29 con quattro torri d'angolo (scala a pioli, arciere sul tetto), portale "
            "con saracinesca alzata, camminamento con merli (scale d'accesso sul lato di fondo), cortile "
            "con pozzo, fucina con cassa, stalla, bracieri, lampioni e stendardi; il maschio al centro e' "
            "una torre curata con il capo (nome, armatura) e quattro soldati nel cortile.")
    sezione("Rovine con cripta", "rovina",
            "Rovina in superficie, scala di 14 blocchi, anticamera, sala dello spawner, tesoro con due "
            "forzieri-trappola sul TNT. Cartello davanti. Lo spawner sta circa 41 blocchi dalla rovina.")
    sezione("Miniere", "miniera",
            "Imbocco con cartello, collina del posto, binari. Nell'ultimo piano: camera finale con pozza.")
    sezione("Citta'", "citta",
            "Piazza con fontana (parafulmine), campana, 2 stendardi, 2 bracieri, lampioni di rame; "
            "giardini con zucche e alveari; legno degli arredi del bioma; gatti sulle strade; "
            "un libro 'Il diario del viandante' in alcune case della citta' piu' vicina al centro.")
    sezione("Accampamenti", "accampamento", "Cerchio calpestato, falo', predone con balestra, cavalli.")
    sezione("Cimiteri", "cimitero", "Forzieri con ossa e refurtiva; scheletro (stray sulla neve).")

    cache = os.path.splitext(a.mondo.rstrip("/\\"))[0] + "_cache.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        lato = d["h"].shape[0]
        meta = lato // 2
        righe.extend(["## Biomi", ""])
        for nome, cosa in BIOMI_DA_VEDERE:
            p = punto_per_bioma(d["biomi"], d["h"], nome, meta)
            if p is None:
                righe.append(f"- {nome}: assente in questa mappa")
            else:
                righe.append(f"- [ ] **{nome}** ({p[3]} celle) - `/tp @s {p[0]} {p[1]} {p[2]}` - {cosa}")
        righe.append("")

    righe.extend([
        "## Bordo della mappa", "",
        "- [ ] Vola oltre un bordo (es. `/tp @s 480 120 0` e poi verso est): il terreno continua e si "
        "raccorda al nostro senza un taglio netto.", "",
        "## Catena degli indizi", "",
        "- [ ] In una casa della citta' piu' vicina al centro: libro che indica la prima tappa. "
        "Seguila: il tesoro di ogni rovina/torre ha un libro che indica la successiva.",
        "- [ ] Un cartografo o un libraio vende per 5 smeraldi un libro con la rovina o la torre piu' vicina.",
        "- [ ] Cartelli ai bivi con i due luoghi piu' vicini e la direzione.", "",
        "## Sottoterra", "",
        "- [ ] Grotte con dripstone, muschio, pozze, sculk profondo, geodi di ametista (vedi `docs/`).", ""])
    testo = "\n".join(righe)
    if a.uscita:
        with open(a.uscita, "w", encoding="utf-8") as f:
            f.write(testo)
        print("scritta", a.uscita)
    else:
        print(testo)


if __name__ == "__main__":
    main()
