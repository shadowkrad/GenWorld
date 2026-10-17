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
            "due forzieri; cartello davanti. Alcune sono TORRI DEL MAGO: capo evocatore (senza armatura), "
            "streghe di guardia, alambicchi e calderoni al primo piano, librerie e tavoli per gli incantesimi "
            "al secondo, forzieri con pozioni, libri incantati e roba da alchimista.")
    # Incarichi: dal `luoghi.md` scritto dalla generazione
    md = os.path.join(a.mondo, "luoghi.md")
    if os.path.exists(md):
        import re
        testo_md = open(md, encoding="utf-8").read()
        voci = re.findall(r"- \*\*(.+?)\*\* \((.+?), (-?\d+), (-?\d+)\): (\w+) a (.+?) "
                          r"\((-?\d+), (-?\d+)\) - oggetto «(.+?)», ricompensa (.+)", testo_md)
        if voci:
            righe.extend([
                "## Incarichi", "",
                "Un abitante con il nome sopra la testa, su una strada vicino alla piazza. Per 1 smeraldo "
                "vende un libro con l'incarico. Il capo (uccidi) lascia cadere l'oggetto, la reliquia "
                "(recupera) sta nel forziere del tesoro, la lettera (salva) nella cella del castello. "
                "Lo scambio di consegna vale solo con QUELL'oggetto (un lingotto qualunque no) e paga "
                "smeraldi o diamanti. Controlla che il committente non perda la professione.", ""])
            for nome, citta_n, x, z, tipo, meta_n, mx, mz, ogg, ric in voci:
                righe.append(f"- [ ] **{nome}** ({citta_n}) - `/tp @s {x} 100 {z}` - {tipo} a {meta_n} "
                             f"(`/tp @s {mx} 100 {mz}`), oggetto «{ogg}», ricompensa {ric}")
            righe.append("")

    # Cascate: dai salti dell'analisi (approssimati: la citta' puo' ritoccare i fiumi)
    cache_c = os.path.splitext(a.mondo.rstrip("/\\"))[0] + "_cache.npz"
    if os.path.exists(cache_c):
        from genworld import cascate as CAS
        dc = np.load(cache_c, allow_pickle=True)
        caduta, _ = CAS.trova(dc["livello"], dc["cls"], 62, dc["lava"])
        zz, xx = np.nonzero(caduta)
        if len(zz):
            m0 = dc["h"].shape[0] // 2
            salto = caduta[zz, xx] - np.maximum(np.floor(dc["livello"][zz, xx]), 62)
            scelti = []
            for i in np.argsort(-salto):
                if all(abs(int(xx[i]) - sx) + abs(int(zz[i]) - sz) > 60 for sx, sz in scelti):
                    scelti.append((int(xx[i]), int(zz[i])))
                if len(scelti) == 5:
                    break
            righe.extend([
                "## Cascate", "",
                "Dove un fiume scende di 3+ blocchi: colonna d'acqua in caduta con pareti di roccia "
                "ai lati. Controlla che l'acqua cada senza spargersi sul prato, che la pozza non si "
                "svuoti e che a valle il fiume prosegua. Cinque dei salti piu' alti e distanti fra loro.", ""])
            for sx, sz in scelti:
                y = int(caduta[sz, sx]) + 8
                righe.append(f"- [ ] salto di {int(caduta[sz, sx] - max(np.floor(dc['livello'][sz, sx]), 62))}"
                             f" blocchi - `/tp @s {sx - m0} {y} {sz - m0}`")
            righe.append("")
    righe.extend([
        "## Isole del tesoro", "",
        "In mare aperto, lontano dalle coste (vedi la sezione Isole del tesoro di `luoghi.md`): un'isola di "
        "sabbia con il cuore d'erba, due palme di giungla e una croce di sabbia rossa. Sotto il centro della "
        "croce, a due blocchi di profondita', un forziere con oro e gemme (e il libro/la mappa della tappa "
        "successiva). Sono tappe della catena degli indizi: la mappa del forziere precedente mostra la croce.", "",
        "- [ ] l'isola si vede dal mare, con la spiaggia e il fianco dolce, senza muri d'acqua o buchi;",
        "- [ ] la croce e' una X di sabbia rossa; scavando al centro, a due blocchi, c'e' il forziere;",
        "- [ ] le palme hanno la chioma e non sono tagliate.", "",
        "## Megaliti e ruderi", "",
        "Sulle colline e nelle pianure, lontano da paesi e strade (nelle note della generazione: `ruderi antichi`): "
        "cerchi di pietre con un trilite e una candela al centro, templi in rovina con colonne spezzate a quote "
        "diverse e dolmen con la lastra sopra, tutti con muschio.", "",
        "- [ ] si vedono come ruderi antichi, non come blocchi sparsi; le pietre poggiano a terra e non sono sospese;",
        "- [ ] i muschi e le pietre non coprono alberi o altre strutture.", "",
        "## Lungo le strade", "",
        "- [ ] Pietre miliari (un pilastrino di mattoni) ogni cinquanta blocchi circa fuori dai paesi.",
        "- [ ] Santelle: un'edicola di pietra aperta verso la strada, con candela e lanterna.",
        "- [ ] Carovane: un carro con il carico, un fuoco acceso e un mercante ambulante con due lama.",
        "- [ ] Ponti di pietra ad archi (se ci sono ponti alti): piloni ogni sette celle e archi a tutto sesto.", "",
        "## Campagna", "",
        "- [ ] Vigneti: filari di staccionata con azalea fiorita sopra; uliveti con olivi bassi. Sulle colline vicine ai paesi.",
        "- [ ] Lucciole (cespugli) nei boschi scuri e nelle paludi; fiori selvatici nei prati.", "",
        "## Abitanti e forzieri", "",
        "- [ ] I villager stanno dentro le case, in piedi e non nei muri o sotto il pavimento.",
        "- [ ] I forzieri non sono piu' vuoti, nemmeno quelli doppi e le botti; il bottino segue il mestiere "
        "della bottega (il fabbro ha ferro e picconi, il falegname assi e asce...).",
        "- [ ] Forni, fornaci e affumicatoi hanno il carbone.",
        "- [ ] Mini quest: in ogni abitato 2-3 popolani con un nome, due richieste ciascuno (materiali in "
        "cambio di attrezzi con nome e incantesimi). Elenco in `luoghi.md`.", "",
        "## Altro da controllare", "",
        "- [ ] Grotte lussureggianti: oltre alle pozze, ruscelli sinuosi di acqua ferma in un solco di una cella, "
        "chiusi dalla roccia (non scorrono e non si svuotano); si interrompono ai gradini del pavimento.",
        "- [ ] Giungla: ogni tanto una pianta brocca (`pitcher_plant`), alta due blocchi.",
        "- [ ] Erba secca (savana, deserto) e lettiera di foglie sotto gli alberi dei boschi e del bosco scuro.",
        "- [ ] Lampioni di rame veri (non bulbi) nelle citta', di ossidazione diversa da uno all'altro.",
        "- [ ] I moli di legno procedurali (se ci sono) hanno il legno del bioma, non sempre abete.",
        "- [ ] Il tetto delle torri ha un parafulmine di rame al centro.",
        "- [ ] I forzieri di nave, faro e castello da template hanno il loro bottino (oro e provviste di bordo; "
        "carbone, torce, canna da pesca; armi, libri, mele d'oro), non la dispensa di una casa.",
        "- [ ] Le costruzioni di arenaria (se ce n'e') hanno blocchi lisci, tagliati e scolpiti mescolati.",
        "- [ ] Gli spawner delle cripte fanno uscire ogni tanto un mob diverso (zombie con scheletro, ragno...).",
        "- [ ] Nel log all'apertura del mondo non devono comparire 'Failed to load chunk'.", ""])
    righe.extend([
        "## Mappe del tesoro", "",
        "Nel forziere del tesoro di ogni rovina, torre, fortezza e castello, accanto al libro, c'e' una "
        "mappa (`Mappa: <luogo>`) del paesaggio attorno alla tappa successiva, con una croce rossa sul "
        "posto e un puntino bianco dov'e' il forziere. Dopo aver aperto il forziere:", "",
        "- [ ] la mappa si vede (non e' bianca) e mostra terra, mare, montagne e la croce rossa al centro;",
        "- [ ] aprendola nel mondo la croce cade sul luogo giusto (la descrizione dice quale);",
        "- [ ] il nome e la descrizione dell'oggetto si leggono e non si aggiorna camminando (e' bloccata);",
        "- [ ] se e' bianca o dice 'mappa sconosciuta': scrivimelo, il formato del file di mappa in 26.x "
        "non e' stato provato (`data/minecraft/maps/<n>.dat`).", ""])
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
            "Rovina in superficie, scala di 14 blocchi, corridoi con snodi e 0-3 stanze laterali (cunicoli), "
            "sala dello spawner (ora piu' mite: pochi mob, si puo' arrivare al tesoro), trappole varie nei "
            "corridoi (dardi, fossa, ragnatele, magma, muri infestati, piastre sul TNT), tesoro con due "
            "forzieri-trappola. Cartello davanti. Lo spawner sta circa 41 blocchi dalla rovina.")
    sezione("Miniere", "miniera",
            "Imbocco con cartello, collina del posto, binari. Gallerie con piu' svolte e diramazioni, sale di "
            "sosta 5x5 agli incroci (pali, lanterna, arnesi), lanterne alte: NON si deve sbattere la testa "
            "sulle rampe. Nell'ultimo piano: camera finale con pozza.")
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
