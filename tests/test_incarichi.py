"""Incarichi: committente, oggetto etichettato, trofeo del capo, forziere."""
import numpy as np

from genworld import bauli as BA
from genworld import entita as EN
from genworld import incarichi as IC
from genworld import luoghi as LU
from genworld import torri as TR
from genworld.citta import Citta

META = 500


def _mondo():
    citta = [Citta(z=600, x=600, raggio=30, piazza=(600, 600)),
             Citta(z=800, x=300, raggio=30, piazza=(800, 300))]
    luoghi = [LU.Luogo("Aldona", "citta", 100, 100), LU.Luogo("Brevia", "citta", -200, 300),
              LU.Luogo("Torre di Aldric", "torre", 180, 100),
              LU.Luogo("Rocca Nera", "castello", -150, 500),
              LU.Luogo("Cripta Muta", "rovina", 400, 400)]
    strutture = {("torre", 180, 100): ((680, 600), None),
                 ("castello", -150, 500): ((350, 1000), (-140, -20, 505)),
                 ("rovina", 400, 400): (None, (402, -20, 398))}
    return citta, luoghi, strutture


def test_un_incarico_per_citta_e_determinismo():
    citta, luoghi, strutture = _mondo()
    a = IC.pianifica(citta, luoghi, strutture, META, seed=3)
    b = IC.pianifica(citta, luoghi, strutture, META, seed=3)
    assert len(a) == 2 and a == b
    assert len({i.uid for i in a}) == 2
    assert len({i.committente for i in a}) == 2
    # ogni luogo serve una sola citta'
    assert len({(i.meta_tipo, i.mx, i.mz) for i in a}) == 2


def test_tipo_dal_luogo():
    citta, luoghi, strutture = _mondo()
    inc = IC.pianifica(citta[:1], luoghi, strutture, META, seed=1)[0]
    assert inc.tipo == "uccidi" and inc.meta_tipo == "torre"
    assert IC.trofei([inc]) == {(680, 600): inc.oggetto}
    assert IC.oggetti_forziere([inc]) == {}


def test_recupera_e_salva_vanno_nel_forziere():
    citta, luoghi, strutture = _mondo()
    solo_rovina = {k: v for k, v in strutture.items() if k[0] == "rovina"}
    inc = IC.pianifica(citta[:1], luoghi, solo_rovina, META, seed=1)[0]
    assert inc.tipo == "recupera"
    assert IC.oggetti_forziere([inc]) == {(402, -20, 398): [inc.oggetto]}
    assert IC.trofei([inc]) == {}


def test_oggetto_etichettato_e_offerte():
    citta, luoghi, strutture = _mondo()
    inc = IC.pianifica(citta[:1], luoghi, strutture, META, seed=1)[0]
    libro, consegna = IC.offerte(inc)
    assert libro["compra"] == ("emerald", 1) and libro["vende"][0] == "written_book"
    assert consegna["compra"][2] == {"custom_data": {"incarico": inc.uid}}
    assert consegna["vende"][0] in ("emerald", "diamond") and consegna["usi"] == 1
    comp = BA.componenti_oggetto(inc.oggetto)
    assert comp["custom_data"] == {"incarico": inc.uid}


def test_committente_nbt():
    citta, luoghi, strutture = _mondo()
    inc = IC.pianifica(citta[:1], luoghi, strutture, META, seed=1)
    strade = np.zeros((1000, 1000), dtype=np.uint8)
    strade[600, 606] = 1
    h = np.full((1000, 1000), 70, dtype=np.int32)
    ab = IC.abitanti(inc, strade, h, citta, META, seed=1)
    assert len(ab) == 1 and ab[0].nome == inc[0].committente
    tag = ab[0].nbt_tag()
    assert tag["CustomName"].py_str == inc[0].committente
    assert tag["Xp"].py_int == 250
    ricette = tag["Offers"]["Recipes"]
    assert len(ricette) == 2
    comp = ricette[1]["buy"]["components"]
    assert comp["minecraft:custom_data"]["incarico"].py_str == inc[0].uid


def test_capo_porta_il_trofeo():
    spec = {"id": "gold_ingot", "nome": "Sigillo", "lore": ["x"], "uid": "inc1"}
    t = TR.Combattente(x=1.5, y=70.0, z=1.5, specie="vindicatore", capo=True,
                       nome="Aldric", trofeo=spec)
    tag = t.nbt_tag()
    off = tag["equipment"]["offhand"]
    assert off["id"].py_str == "minecraft:gold_ingot"
    assert off["components"]["minecraft:custom_data"]["incarico"].py_str == "inc1"
    assert tag["drop_chances"]["offhand"].py_float == 1.0
    senza = TR.Combattente(x=1.5, y=70.0, z=1.5, specie="vindicatore", capo=True, nome="A")
    assert "offhand" not in senza.nbt_tag()["equipment"]


def test_combattenti_assegna_il_trofeo_al_capo_giusto():
    t = TR.Torre(x=100, z=100, ys=70, dx=1, dz=0, nome="Aldric")
    spec = {"id": "gold_ingot", "nome": "S", "lore": [], "uid": "u"}
    capi = [c for c in TR.combattenti([t], trofei={(100, 100): spec}) if c.capo]
    assert len(capi) == 1 and capi[0].trofeo == spec
    altri = [c for c in TR.combattenti([t]) if c.capo]
    assert altri[0].trofeo is None


def test_sezione_elenco():
    citta, luoghi, strutture = _mondo()
    inc = IC.pianifica(citta, luoghi, strutture, META, seed=1)
    md = IC.sezione_elenco(inc)
    assert "## Incarichi" in md and inc[0].committente in md
    assert IC.sezione_elenco([]) == ""


def test_elenco_dei_luoghi_include_gli_incarichi(tmp_path):
    citta, luoghi, strutture = _mondo()
    inc = IC.pianifica(citta, luoghi, strutture, META, seed=1)
    md, _ = LU.scrivi_elenco(luoghi, str(tmp_path), "Prova", extra_md=IC.sezione_elenco(inc))
    testo = open(md, encoding="utf-8").read()
    assert "## Incarichi" in testo and inc[0].committente in testo
