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


def _offerta(tit: str, citta: str, nome_pnj: str, r: dict) -> dict:
    """Uno scambio nel formato di `entita._scambio`."""
    id_, quanti, comp = r["vende"]
    comp = dict(comp or {})
    comp["custom_name"] = r["nome"]
    comp["lore"] = [r["frase"][:60], f"Commissione di {nome_pnj}, {citta}"[:60]]
    offerta = {"compra": r["compra"][0],
               "vende": (id_, quanti, comp), "usi": 1}
    if len(r["compra"]) > 1:
        offerta["compra2"] = r["compra"][1]
    return offerta


def offerte(mestiere: str, citta: str, nome_pnj: str) -> tuple:
    tit, righe = Q[mestiere]
    return tuple(_offerta(tit, citta, nome_pnj, r) for r in righe)


def pianifica(citta: list, luoghi: list, strade, altezze, meta: int, occupati: set | None = None,
              seed: int = 0, per_citta: tuple = (2, 3)) -> list:
    """Gli abitanti-commissione. `citta`: oggetti Citta (x, z, piazza, murata);
    `luoghi`: per i nomi delle citta' (coordinate di gioco); `strade`: tipo_strada;
    `occupati`: celle (x, z) di mappa gia' prese da altri abitanti."""
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
        scelti = [mestieri[int(i)] for i in rng.choice(len(mestieri), size=quanti, replace=False)]
        for mestiere in scelti:
            for _ in range(20):                          # una cella di strada non gia' presa
                k = int(ok[int(rng.integers(0, ok.size))])
                z, x = int(zz[k]) + z0, int(xx[k]) + x0
                if all((x + dx, z + dz) not in occ for dx in (-1, 0, 1) for dz in (-1, 0, 1)):
                    break
            else:
                continue
            occ.add((x, z))
            proprio = liberi.pop(int(rng.integers(0, len(liberi)))) if liberi else "Anonimo"
            titolo = Q[mestiere][0]
            nome = f"{titolo} {proprio}"
            fuori.append(Abitante(
                x=x + 0.5, y=float(altezze[z, x]), z=z + 0.5, mestiere=mestiere, livello=5,
                seme=(seed * 6271 + n_c * 104729 + len(fuori) * 7919 + 29) & 0x7FFFFFFF,
                nome=nome, offerte=offerte(mestiere, nome_citta, nome)))
    return fuori


def sezione_elenco(abitanti: list) -> str:
    """Per `luoghi.md`: chi chiede cosa."""
    if not abitanti:
        return ""
    righe = ["", "## Commissioni dei popolani", ""]
    for a in abitanti:
        for o in a.offerte:
            comp = o["vende"][2] or {}
            chiede = " + ".join(f"{n} {i}" for i, n in
                                [o["compra"][:2], *([o["compra2"][:2]] if "compra2" in o else [])])
            righe.append(f"- **{a.nome}**: {chiede} -> {o['vende'][1]} {o['vende'][0]} "
                         f"«{comp.get('custom_name', '')}»")
    return "\n".join(righe) + "\n"
