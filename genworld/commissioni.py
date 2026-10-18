"""Mini quest: le commissioni dei popolani.

Oltre al committente degli incarichi (vedi `incarichi.py`, uno per citta', verso un
luogo d'avventura), in ogni abitato vivono due o tre popolani con un nome sopra la
testa e una commissione ciascuno: "portami questi materiali e ti do questo
attrezzo". Si fa in un attimo, con quello che si trova in giro, e serve a farsi
un equipaggiamento: un piccone incantato dal fabbro, una balestra dal falegname,
un libro di Mending dal libraio.

Ognuno ha due richieste, una facile e una piu' difficile; alcune mandano a
cercare le cose nel mondo: schegge di ametista nei geodi, bacche luminose nelle
grotte, ossa e occhi di ragno dai mostri, perle dell'Ender. Lo scambio si puo'
fare una volta sola (`usi`: 1).

Come per il committente, un villager con scambi propri non ne ha di normali: sono
popolani della storia, non botteghe. Tutto e' deterministico sul seme.
"""
from __future__ import annotations

import numpy as np

NOMI_PROPRI = ("Ugo", "Berta", "Ottavio", "Livia", "Remigio", "Agnese", "Tullio", "Ginevra",
               "Osvaldo", "Perpetua", "Aldo", "Clelia", "Baldo", "Ildegarda", "Nestore",
               "Fosca", "Gualtiero", "Lucilla", "Amerigo", "Teodora", "Rolando", "Viola",
               "Cunibert", "Dina")

# mestiere -> (titolo, [richiesta facile, richiesta difficile]). Una richiesta:
# compra = una o due pile da portare, vende = (oggetto, quanti, componenti), nome e
# frase di chi la chiede.
Q = {
    "fabbro": ("Fabbro", [
        {"compra": [("coal", 16), ("iron_ingot", 6)],
         "vende": ("iron_pickaxe", 1, {"enchantments": {"minecraft:efficiency": 2, "minecraft:unbreaking": 2}}),
         "nome": "Piccone del fabbro", "frase": "Il forno ha fame: carbone e ferro, e ti batto un piccone."},
        {"compra": [("diamond", 2), ("iron_ingot", 12)],
         "vende": ("diamond_axe", 1, {"enchantments": {"minecraft:efficiency": 3, "minecraft:unbreaking": 3}}),
         "nome": "Ascia temprata", "frase": "Con due diamanti ti faccio un'ascia che taglia le querce come burro."}]),
    "armaiolo": ("Armaiolo", [
        {"compra": [("iron_ingot", 6), ("string", 4)],
         "vende": ("bow", 1, {"enchantments": {"minecraft:power": 2, "minecraft:unbreaking": 1}}),
         "nome": "Arco del cacciatore", "frase": "Ferro e corda per un arco che non tradisce."},
        {"compra": [("iron_ingot", 12), ("leather", 8)],
         "vende": ("iron_sword", 1, {"enchantments": {"minecraft:sharpness": 3, "minecraft:unbreaking": 2}}),
         "nome": "Spada del capitano", "frase": "Una lama da capitano, se mi porti il materiale."}]),
    "corazzaio": ("Corazzaio", [
        {"compra": [("iron_ingot", 14)],
         "vende": ("iron_chestplate", 1, {"enchantments": {"minecraft:protection": 2}}),
         "nome": "Corazza della guardia", "frase": "Quattordici lingotti e ti faccio una corazza da guardia."},
        {"compra": [("diamond", 3), ("leather", 6)],
         "vende": ("diamond_helmet", 1, {"enchantments": {"minecraft:protection": 3, "minecraft:respiration": 2}}),
         "nome": "Elmo del palombaro", "frase": "Un elmo di diamante che fa respirare anche sott'acqua."}]),
    "falegname": ("Falegname", [
        {"compra": [("oak_log", 32)],
         "vende": ("iron_axe", 1, {"enchantments": {"minecraft:efficiency": 2}}),
         "nome": "Ascia del boscaiolo", "frase": "Trentadue tronchi di quercia e l'ascia e' tua."},
        {"compra": [("spruce_log", 24), ("string", 8)],
         "vende": ("crossbow", 1, {"enchantments": {"minecraft:quick_charge": 2, "minecraft:unbreaking": 2}}),
         "nome": "Balestra d'abete", "frase": "Abete e corda: una balestra che carica in un soffio."}]),
    "scalpellino": ("Scalpellino", [
        {"compra": [("cobblestone", 48)],
         "vende": ("iron_pickaxe", 1, {"enchantments": {"minecraft:efficiency": 3}}),
         "nome": "Piccone dello scalpellino", "frase": "Cinquanta pietre da squadrare e ti do il mio piccone buono."},
        {"compra": [("amethyst_shard", 6), ("stone_bricks", 16)],
         "vende": ("enchanted_book", 1, {"stored_enchantments": {"minecraft:fortune": 2}}),
         "nome": "Segreto dello scalpellino", "frase": "L'ametista dei geodi sotto le colline: portamene, e ti insegno a trovare le gemme."}]),
    "macellaio": ("Macellaio", [
        {"compra": [("beef", 8), ("coal", 4)],
         "vende": ("cooked_beef", 16, None),
         "nome": "Scorta del macellaio", "frase": "Carne cruda e carbone: te la restituisco arrosto, in abbondanza."},
        {"compra": [("spider_eye", 6), ("bone", 10)],
         "vende": ("golden_apple", 2, None),
         "nome": "Mele del macellaio", "frase": "Ossa e occhi di ragno: i mostri della notte pagano bene."}]),
    "pescatore": ("Pescatore", [
        {"compra": [("cod", 12), ("string", 4)],
         "vende": ("fishing_rod", 1, {"enchantments": {"minecraft:lure": 2, "minecraft:luck_of_the_sea": 2}}),
         "nome": "Canna del pescatore", "frase": "Merluzzi e filo: la canna piu' fortunata del porto."},
        {"compra": [("salmon", 8), ("ink_sac", 6)],
         "vende": ("enchanted_book", 1, {"stored_enchantments": {"minecraft:luck_of_the_sea": 3}}),
         "nome": "Libro del mare", "frase": "Salmoni e inchiostro di calamaro, per un segreto del mare."}]),
    "fruttivendolo": ("Fruttivendolo", [
        {"compra": [("wheat", 20)],
         "vende": ("golden_carrot", 8, None),
         "nome": "Carote d'oro", "frase": "Un sacco di grano e ti pago in carote d'oro."},
        {"compra": [("glow_berries", 8), ("sweet_berries", 8)],
         "vende": ("diamond_hoe", 1, {"enchantments": {"minecraft:efficiency": 4, "minecraft:unbreaking": 3}}),
         "nome": "Zappa d'oro", "frase": "Le bacche luminose crescono nelle grotte verdi: portamele e avrai la zappa migliore."}]),
    "pastore": ("Pastore", [
        {"compra": [("white_wool", 16)],
         "vende": ("shears", 1, {"enchantments": {"minecraft:efficiency": 4, "minecraft:unbreaking": 2}}),
         "nome": "Forbici del pastore", "frase": "Lana bianca: ne ho bisogno per l'inverno. In cambio, le mie forbici."},
        {"compra": [("leather", 6), ("string", 8)],
         "vende": ("saddle", 1, None),
         "nome": "Sella del pastore", "frase": "Cuoio e corda e ti cucio una sella."}]),
    "libraio": ("Libraio", [
        {"compra": [("paper", 12), ("ink_sac", 3)],
         "vende": ("enchanted_book", 1, {"stored_enchantments": {"minecraft:unbreaking": 3}}),
         "nome": "Libro della durata", "frase": "Carta e inchiostro per copiare un libro raro."},
        {"compra": [("lapis_lazuli", 10), ("book", 3)],
         "vende": ("enchanted_book", 1, {"stored_enchantments": {"minecraft:mending": 1}}),
         "nome": "Libro della Riparazione", "frase": "Il lapislazzuli dalle miniere: ti scrivo il segreto che ripara gli attrezzi."}]),
    "cartografo": ("Cartografo", [
        {"compra": [("paper", 10), ("ink_sac", 4)],
         "vende": ("spyglass", 1, None),
         "nome": "Cannocchiale del cartografo", "frase": "Carta e inchiostro per un cannocchiale che arriva lontano."},
        {"compra": [("emerald", 6), ("compass", 1)],
         "vende": ("ender_eye", 3, None),
         "nome": "Occhi del cartografo", "frase": "Smeraldi e una bussola: te ne do tre per cercare le fortezze."}]),
    "conciatore": ("Conciatore", [
        {"compra": [("leather", 12)],
         "vende": ("leather_boots", 1, {"enchantments": {"minecraft:feather_falling": 3, "minecraft:unbreaking": 2}}),
         "nome": "Stivali del viandante", "frase": "Dodici pelli e ti cucio stivali che ammortizzano le cadute."},
        {"compra": [("rabbit_hide", 8), ("leather", 12)],
         "vende": ("iron_horse_armor", 1, None),
         "nome": "Barda del conciatore", "frase": "Pelli di coniglio e di vacca per una barda da cavallo."}]),
    "speziale": ("Speziale", [
        {"compra": [("spider_eye", 4), ("sugar", 4)],
         "vende": ("potion", 1, {"potion_contents": {"potion": "minecraft:swiftness"}}),
         "nome": "Filtro di rapidita'", "frase": "Occhi di ragno e zucchero per un filtro che ti fa correre."},
        {"compra": [("gunpowder", 6), ("glowstone_dust", 4)],
         "vende": ("potion", 1, {"potion_contents": {"potion": "minecraft:healing"}}),
         "nome": "Elisir di guarigione", "frase": "Polvere da sparo e polvere luminosa: guarisco chi vuole tornare dalle caverne."}]),
}


# Le commissioni SEMPLICI: portare una quantita' di roba comune, in cambio di smeraldi. Una per
# mestiere, sempre la prima dell'elenco di chi le offre: servono a chi comincia e a chi ha
# solo cio' che raccoglie lungo la strada. mestiere -> (cosa, quanti, smeraldi, frase)
SEMPLICI = {
    "fabbro": ("coal", 20, 4, "Il forno non ha carbone: venti pezzi e ti pago in smeraldi."),
    "armaiolo": ("flint", 12, 3, "Selce per le punte delle frecce: dodici pezzi."),
    "corazzaio": ("iron_ingot", 10, 4, "Dieci lingotti di ferro per la fucina."),
    "falegname": ("oak_planks", 32, 3, "Trentadue assi di quercia per una consegna."),
    "scalpellino": ("cobblestone", 64, 4, "Un carico di ciottoli per un muro."),
    "macellaio": ("beef", 12, 4, "Dodici bistecche crude: la bottega e' vuota."),
    "pescatore": ("cod", 16, 4, "Sedici merluzzi per il mercato del venerdi'."),
    "fruttivendolo": ("carrot", 24, 3, "Ventiquattro carote, fresche."),
    "pastore": ("white_wool", 12, 4, "Dodici di lana bianca per le coperte."),
    "libraio": ("paper", 8, 2, "Otto fogli di carta per copiare i registri."),
    "cartografo": ("paper", 10, 3, "Dieci fogli di carta per le mappe."),
    "conciatore": ("leather", 10, 3, "Dieci pelli per la bottega."),
    "speziale": ("sugar", 12, 3, "Dodici di zucchero per gli sciroppi."),
}

# Gli ACQUIRENTI: comprano quello che il giocatore ha in tasca, a prezzo fisso, tutte le volte
# che vuole (16 scambi). tipo -> (mestiere, titolo, [(cosa, quanti, smeraldi)])
ACQUIRENTI = {
    "minerali": ("scalpellino", "Compratore di minerali", [
        ("coal", 16, 1), ("raw_iron", 8, 1), ("iron_ingot", 6, 1), ("raw_copper", 12, 1),
        ("copper_ingot", 9, 1), ("raw_gold", 6, 1), ("gold_ingot", 4, 1), ("redstone", 12, 1),
        ("lapis_lazuli", 10, 1), ("amethyst_shard", 3, 1), ("diamond", 1, 4), ("cobblestone", 48, 1)]),
    "bottino": ("armaiolo", "Compratore di bottino", [
        ("bone", 12, 1), ("rotten_flesh", 10, 1), ("spider_eye", 6, 1), ("gunpowder", 6, 1),
        ("string", 8, 1), ("arrow", 16, 1), ("ender_pearl", 1, 3), ("blaze_rod", 2, 2),
        ("ghast_tear", 1, 4), ("nautilus_shell", 1, 5), ("phantom_membrane", 3, 2)]),
    "pelli": ("conciatore", "Compratore di pelli", [
        ("leather", 8, 1), ("rabbit_hide", 8, 1), ("feather", 12, 1), ("white_wool", 12, 1),
        ("ink_sac", 8, 1), ("glow_ink_sac", 4, 2), ("honeycomb", 6, 1), ("honey_bottle", 3, 1)]),
    "raccolto": ("fruttivendolo", "Compratore di raccolto", [
        ("wheat", 20, 1), ("carrot", 16, 1), ("potato", 16, 1), ("beetroot", 16, 1),
        ("melon_slice", 16, 1), ("pumpkin", 6, 1), ("apple", 10, 1), ("sweet_berries", 14, 1),
        ("glow_berries", 10, 1), ("sugar_cane", 20, 1)]),
    "pesca": ("pescatore", "Compratore di pesce", [
        ("cod", 10, 1), ("salmon", 8, 1), ("tropical_fish", 4, 1), ("pufferfish", 3, 1),
        ("kelp", 24, 1), ("lily_pad", 12, 1), ("prismarine_shard", 6, 1), ("nautilus_shell", 1, 5)]),
    "legname": ("falegname", "Compratore di legname", [
        ("oak_log", 16, 1), ("spruce_log", 16, 1), ("birch_log", 16, 1), ("acacia_log", 16, 1),
        ("dark_oak_log", 16, 1), ("jungle_log", 16, 1), ("stick", 32, 1), ("bamboo", 24, 1)]),
    "curiosita": ("libraio", "Antiquario", [
        ("music_disc_13", 1, 4), ("music_disc_cat", 1, 4), ("name_tag", 1, 3), ("saddle", 1, 3),
        ("book", 6, 1), ("paper", 20, 1), ("compass", 1, 2), ("clock", 1, 2), ("spyglass", 1, 3),
        ("golden_apple", 1, 4), ("experience_bottle", 4, 2)]),
}


def _offerta(tit: str, citta: str, nome_pnj: str, r: dict) -> dict:
    """Uno scambio nel formato di `entita._scambio`."""
    id_, quanti, comp = r["vende"]
    comp = dict(comp or {})
    if id_ != "emerald":                    # gli smeraldi restano smeraldi veri, si impilano
        comp["custom_name"] = r["nome"]
        comp["lore"] = [r["frase"][:60], f"Commissione di {nome_pnj}, {citta}"[:60]]
    offerta = {"compra": r["compra"][0],
               "vende": (id_, quanti, comp or None), "usi": 1}
    if len(r["compra"]) > 1:
        offerta["compra2"] = r["compra"][1]
    return offerta


def _semplice(mestiere: str) -> dict:
    cosa, quanti, smeraldi, frase = SEMPLICI[mestiere]
    return {"compra": [(cosa, quanti)], "vende": ("emerald", smeraldi, None),
            "nome": "Smeraldi", "frase": frase}


def offerte(mestiere: str, citta: str, nome_pnj: str) -> tuple:
    """Tre richieste: la semplice (smeraldi) e le due con attrezzo."""
    tit, righe = Q[mestiere]
    return tuple(_offerta(tit, citta, nome_pnj, r) for r in [_semplice(mestiere), *righe])


def offerte_acquirente(tipo: str) -> tuple:
    """Gli scambi di un acquirente: la merce del giocatore contro smeraldi, ripetibili."""
    _m, _t, voci = ACQUIRENTI[tipo]
    return tuple({"compra": (cosa, quanti), "vende": ("emerald", smeraldi, None), "usi": 16}
                 for cosa, quanti, smeraldi in voci)


CURATORE = [("nautilus_shell", 1, 6), ("heart_of_the_sea", 1, 24), ("amethyst_shard", 8, 4),
            ("prismarine_crystals", 8, 4), ("music_disc_13", 1, 8), ("music_disc_cat", 1, 8),
            ("echo_shard", 1, 10), ("archer_pottery_sherd", 1, 6), ("sponge", 2, 5),
            ("ancient_debris", 1, 12), ("nether_star", 1, 36), ("totem_of_undying", 1, 14)]


def curatore(sito, seed: int = 0):
    """Il curatore di un museo (`siti.py`): compra reperti rari contro smeraldi. In coordinate di mappa."""
    from .entita import Abitante
    return Abitante(
        x=sito.x + 0.5, y=float(sito.y + 1), z=sito.z + 2.5, mestiere="libraio", livello=5,
        seme=(seed * 6271 + sito.x * 7919 + sito.z * 104729 + 31) & 0x7FFFFFFF,
        nome="Curatore del museo",
        offerte=tuple({"compra": (cosa, quanti), "vende": ("emerald", smeraldi, None), "usi": 8}
                      for cosa, quanti, smeraldi in CURATORE))


def pianifica(citta: list, luoghi: list, strade, altezze, meta: int, occupati: set | None = None,
              seed: int = 0, per_citta: tuple = (3, 5), acquirenti: tuple = (2, 3)) -> list:
    """Gli abitanti-commissione e gli acquirenti. `citta`: oggetti Citta (x, z, piazza,
    murata); `luoghi`: per i nomi delle citta' (coordinate di gioco); `strade`:
    tipo_strada; `occupati`: celle (x, z) di mappa gia' prese da altri abitanti.
    Per citta' da 3 a 5 popolani con commissioni e da 2 a 3 acquirenti."""
    from .entita import Abitante
    fuori: list = []
    if strade is None:
        return fuori
    rng = np.random.default_rng(seed + 6161)
    H, W = strade.shape
    occ = set(occupati or ())
    nomi_citta = {(l.x, l.z): l.nome for l in luoghi if l.tipo == "citta"}
    liberi = list(NOMI_PROPRI)
    mestieri = list(Q)
    for n_c, c in enumerate(citta):
        nome_citta = nomi_citta.get((c.x - meta, c.z - meta), "")
        pz, px = c.piazza
        z0, z1, x0, x1 = max(0, pz - 18), min(H, pz + 19), max(0, px - 18), min(W, px + 19)
        zz, xx = np.nonzero(strade[z0:z1, x0:x1] > 0)
        if zz.size == 0:
            continue
        d2 = (zz + z0 - pz) ** 2 + (xx + x0 - px) ** 2
        ok = np.nonzero((d2 >= 25) & (d2 <= 324))[0]
        if ok.size == 0:
            continue
        quanti = per_citta[1] if c.murata else per_citta[0]
        scelti = [("commissione", mestieri[int(i)])
                  for i in rng.choice(len(mestieri), size=quanti, replace=False)]
        n_acq = acquirenti[1] if c.murata else acquirenti[0]
        tipi = list(ACQUIRENTI)
        scelti += [("acquirente", tipi[int(i)])
                   for i in rng.choice(len(tipi), size=min(n_acq, len(tipi)), replace=False)]
        for genere, chiave in scelti:
            mestiere = chiave if genere == "commissione" else ACQUIRENTI[chiave][0]
            for _ in range(30):                          # una cella di strada non gia' presa
                k = int(ok[int(rng.integers(0, ok.size))])
                z, x = int(zz[k]) + z0, int(xx[k]) + x0
                if all((x + dx, z + dz) not in occ for dx in (-1, 0, 1) for dz in (-1, 0, 1)):
                    break
            else:
                continue
            occ.add((x, z))
            proprio = liberi.pop(int(rng.integers(0, len(liberi)))) if liberi else "Anonimo"
            titolo = Q[mestiere][0] if genere == "commissione" else ACQUIRENTI[chiave][1]
            nome = f"{titolo} {proprio}"
            fuori.append(Abitante(
                x=x + 0.5, y=float(altezze[z, x]), z=z + 0.5, mestiere=mestiere, livello=5,
                seme=(seed * 6271 + n_c * 104729 + len(fuori) * 7919 + 29) & 0x7FFFFFFF,
                nome=nome,
                offerte=(offerte(mestiere, nome_citta, nome) if genere == "commissione"
                         else offerte_acquirente(chiave))))
    return fuori


def guardie(citta: list, strade, altezze, seed: int = 0) -> list:
    """Le PATTUGLIE: due golem di ferro sulle strade attorno alla piazza di ogni citta' (quattro
    se e' murata), a distanza diversa dal centro: difendono gli abitanti dai mostri."""
    from .fauna import Animale
    fuori: list = []
    if strade is None:
        return fuori
    rng = np.random.default_rng(seed + 7171)
    H, W = strade.shape
    for n_c, c in enumerate(citta):
        pz, px = c.piazza
        z0, z1, x0, x1 = max(0, pz - 26), min(H, pz + 27), max(0, px - 26), min(W, px + 27)
        zz, xx = np.nonzero(strade[z0:z1, x0:x1] > 0)
        if zz.size == 0:
            continue
        d2 = (zz + z0 - pz) ** 2 + (xx + x0 - px) ** 2
        for k in range(4 if c.murata else 2):
            lo, hi = (30, 100) if k % 2 == 0 else (150, 600)
            ok = np.nonzero((d2 >= lo) & (d2 <= hi))[0]
            if ok.size == 0:
                continue
            i = int(ok[int(rng.integers(0, ok.size))])
            z, x = int(zz[i]) + z0, int(xx[i]) + x0
            fuori.append(Animale(x=x + 0.5, y=float(altezze[z, x]), z=z + 0.5, specie="golem",
                                 seme=(seed * 6271 + n_c * 104729 + k * 7919 + 73) & 0x7FFFFFFF))
    return fuori


def sezione_elenco(abitanti: list) -> str:
    """Per `luoghi.md`: chi chiede cosa."""
    if not abitanti:
        return ""
    righe = ["", "## Commissioni dei popolani", ""]
    for a in abitanti:
        if a.offerte and a.offerte[0].get("usi", 1) > 1:                  # un acquirente
            righe.append(f"- **{a.nome}** compra: " + ", ".join(
                f"{o['compra'][1]} {o['compra'][0]} per {o['vende'][1]} smeraldi" for o in a.offerte))
            continue
        for o in a.offerte:
            comp = o["vende"][2] or {}
            chiede = " + ".join(f"{n} {i}" for i, n in
                                [o["compra"][:2], *([o["compra2"][:2]] if "compra2" in o else [])])
            righe.append(f"- **{a.nome}**: {chiede} -> {o['vende'][1]} {o['vende'][0]} "
                         f"«{comp.get('custom_name', '')}»")
    return "\n".join(righe) + "\n"
