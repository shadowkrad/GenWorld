"""Castello in rovina (#6)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import castello as CS

Y0, ALT = -64, 320
LIBERO = ("air", "cobweb", "bone_block", "head", "carpet", "plant", "soul_lantern",
          "moss_carpet", "brown_mushroom", "red_mushroom", "candle")


class Finto:
    def __init__(self):
        self.ids = {("air", ()): 0}
        self.id_aria = 0

    def blocco(self, nome, **pr):
        return self.ids.setdefault((nome, tuple(sorted(pr.items()))), len(self.ids))

    def nome(self, i):
        return next((n for (n, _), v in self.ids.items() if v == i), "?")


def terreno(lato=1600, quota=90):
    return np.full((lato, lato), quota, np.int32), np.zeros((lato, lato), bool)


def un_castello(direzione=(1, 0), ys=90, x=250, z=250, seme=4):
    return CS.Castello(x=x, z=z, ys=ys, dx=direzione[0], dz=direzione[1],
                       nome="Castel Malvento", capo="Re Spettro Aldovrando",
                       specie="scheletro_nero", seme=seme)


def posa_su_regione(c, quota=90):
    s = Finto()
    tav = CS.Tavolozza(s)
    x0, z0, x1, z1 = c.bbox()
    regione = {}
    for cz in range(z0 // 16, z1 // 16 + 1):
        for cx in range(x0 // 16, x1 // 16 + 1):
            out = np.zeros((16, ALT, 16), np.uint32)
            ys = np.arange(Y0, Y0 + ALT)[None, :, None]
            out[np.broadcast_to(ys < quota, out.shape)] = s.blocco("stone")
            h = np.full((16, 16), quota, np.int32)
            CS.posa(out, tav, h, cx * 16, cz * 16, Y0, [c], [0], meta=0)
            regione[(cx, cz)] = out

    def blocco(x, y, z):
        out = regione.get((x // 16, z // 16))
        return None if out is None else s.nome(int(out[x % 16, y - Y0, z % 16]))

    def proprieta(x, y, z):
        out = regione.get((x // 16, z // 16))
        if out is None:
            return {}
        i = int(out[x % 16, y - Y0, z % 16])
        return next((dict(p) for (n, p), v in s.ids.items() if v == i), {})
    blocco.proprieta = proprieta
    return s, tav, blocco


class TestPianifica(unittest.TestCase):
    def test_trova_siti_e_li_distanzia(self):
        h, acqua = terreno()
        cs = CS.pianifica(h, acqua, seed=1)
        self.assertGreaterEqual(len(cs), 1)
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                self.assertGreaterEqual((a.x - b.x) ** 2 + (a.z - b.z) ** 2,
                                        CS.DISTANZA_FRA_CASTELLI ** 2)

    def test_non_nasce_su_acqua_su_terreno_accidentato_o_dove_si_evita(self):
        h, acqua = terreno()
        self.assertEqual(CS.pianifica(h, np.ones_like(acqua), seed=1), [])
        irregolare = (80 + np.random.default_rng(0).integers(0, 14, h.shape)).astype(np.int32)
        self.assertEqual(CS.pianifica(irregolare, acqua, seed=1), [])
        self.assertEqual(CS.pianifica(h, acqua, evita=np.ones_like(acqua), seed=1), [])

    def test_vuole_terra_sopra_la_cripta(self):
        h, acqua = terreno()
        c = un_castello()
        self.assertTrue(CS._valido(c, h, acqua, None))
        x, z = c.punto(CS.U_SPAWNER[0] + 25, CS.VC)           # fuori dalla cinta: il soffitto sporgerebbe
        h2 = h.copy()
        h2[z, x] = c.yf + 5
        self.assertFalse(CS._valido(c, h2, acqua, None))

    def test_e_deterministica_i_nomi_sono_diversi_e_la_densita_zero_spegne(self):
        h, acqua = terreno()
        a = [(c.x, c.z, c.nome, c.capo) for c in CS.pianifica(h, acqua, seed=4, densita=2.0)]
        b = [(c.x, c.z, c.nome, c.capo) for c in CS.pianifica(h, acqua, seed=4, densita=2.0)]
        self.assertEqual(a, b)
        self.assertEqual(len({x[2] for x in a}), len(a))
        self.assertEqual(CS.pianifica(h, acqua, densita=0.0, seed=4), [])

    def test_la_cripta_resta_dentro_la_mappa(self):
        h, acqua = terreno()
        for c in CS.pianifica(h, acqua, seed=2, densita=2.0):
            x0, z0, x1, z1 = c.bbox()
            self.assertTrue(x0 >= 0 and z0 >= 0 and x1 < 1600 and z1 < 1600)


class TestSuperficie(unittest.TestCase):
    def setUp(self):
        self.c = un_castello()
        self.s, self.tav, self.b = posa_su_regione(self.c)

    def blocco(self, u, v, y):
        x, z = self.c.punto(u, v)
        return self.b(x, y, z)

    def test_la_cinta_ha_brecce_e_tratti_interi(self):
        c, K = self.c, CS.K
        altezze = []
        for t in range(-K + 6, K - 5):
            for (u, v) in ((t, K), (-K, t), (t, -K)):
                h = 0
                for y in range(c.ys, c.ys + CS.ALTEZZA_MURA + 1):
                    if self.blocco(u, v, y) == "stone_bricks":
                        h += 1
                altezze.append(h)
        self.assertTrue(any(h == 0 for h in altezze), "nessuna breccia")
        self.assertTrue(any(h >= CS.ALTEZZA_MURA for h in altezze), "nessun tratto intero")
        self.assertTrue(any(0 < h < CS.ALTEZZA_MURA - 1 for h in altezze), "nessun tratto spezzato")

    def test_il_portale_e_aperto_largo_cinque_e_ci_sono_macerie_e_sbarre_cadute(self):
        c, K = self.c, CS.K
        for v in range(-2, 3):
            for y in (c.ys + 4, c.ys + 5):
                self.assertNotEqual(self.blocco(K, v, y), "stone_bricks", (v, y))
        visti = {self.blocco(u, v, y) for u in (K - 1, K) for v in range(-2, 3)
                 for y in range(c.ys, c.ys + 6)}
        self.assertTrue({"bars", "mossy_cobblestone"} & visti)

    def test_le_torri_d_angolo_sono_vuote_dentro_e_spezzate_in_cima(self):
        c, K = self.c, CS.K
        for su in (1, -1):
            for sv in (1, -1):
                self.assertIn(self.blocco(su * (K - 3), sv * (K - 3), c.ys + 3),
                              LIBERO + ("spawner", "chest", "planks"), (su, sv))
                alte = [sum(1 for y in range(c.ys, c.ys + 20) if self.blocco(su * K, sv * (K - 6 + k), y)
                            == "stone_bricks") for k in range(7)]
                self.assertGreater(max(alte) - min(alte), 1, "la torre non e' spezzata")

    def test_il_maschio_e_sventrato_sulla_fronte_e_ha_la_sala_del_trono(self):
        c, mc = self.c, CS.CENTRO_MASCHIO
        # lo squarcio sulla fronte: sopra i 4 blocchi non c'e' piu' muro
        for v in range(-3, 4):
            self.assertNotEqual(self.blocco(mc + CS.MEZZO_MASCHIO, v, c.ys + 6), "stone_bricks", v)
        # il rialzo, il trono e il capo che lo presidia
        self.assertEqual(self.blocco(mc - 5, 0, c.ys + 1), "stairs")
        self.assertEqual(self.blocco(mc - 5, 0, c.ys), "stone_bricks")
        u, v, y = c.posto_capo()
        self.assertEqual(self.blocco(u, v, y), "air")
        self.assertEqual(self.blocco(u, v, y - 1), "stone_bricks")

    def test_la_sala_e_arredata_e_in_rovina(self):
        c, mc = self.c, CS.CENTRO_MASCHIO
        visti = set()
        for u in range(mc - 6, mc + 7):
            for v in range(-6, 7):
                for y in range(c.ys, c.ys + 8):
                    visti.add(self.blocco(u, v, y))
        for atteso in ("slab", "stairs", "carpet", "candle", "barrel", "bookshelf", "campfire",
                       "cobweb", "chest"):
            self.assertIn(atteso, visti, atteso)

    def test_il_cortile_e_invaso_da_erba_macerie_ossa_alberi_morti_e_tombe(self):
        c = self.c
        visti = set()
        for u in range(-16, 17):
            for v in range(-16, 17):
                for y in range(c.ys - 1, c.ys + 6):
                    visti.add(self.blocco(u, v, y))
        for atteso in ("grass_block", "mossy_cobblestone", "bone_block", "log", "campfire",
                       "wall", "gravel", "plant"):
            self.assertIn(atteso, visti, atteso)

    def test_la_torre_dello_spawner_ha_lo_spawner_e_un_forziere(self):
        c, K = self.c, CS.K
        x, y, z, mob = c.spawner()[0]
        self.assertEqual(self.b(x, y, z), "spawner")
        self.assertEqual(self.b(x, y - 1, z), "stone_bricks")
        u, v, yy, tipo = c.forzieri_locali()[2]
        self.assertEqual(self.blocco(u, v, yy), "chest")

    def test_i_forzieri_di_superficie_sono_aperti(self):
        murato = self.c.punto(*self.c.forzieri_locali()[7][:2])
        for x, y, z, tipo in self.tav.bauli:
            if y >= self.c.ys:
                self.assertEqual(self.b(x, y, z), "chest")
                if (x, z) == murato:                    # il forziere murato dietro lo scaffale
                    self.assertEqual(self.b(x, y + 1, z), "stone_bricks")
                else:
                    self.assertIn(self.b(x, y + 1, z), LIBERO, tipo)

    def test_i_mobili_della_sala_non_chiudono_le_sentinelle(self):
        c = self.c
        for m in CS.combattenti([c]):
            if not hasattr(m, "specie") or m.y != float(c.ys):
                continue
            x, z = int(m.x), int(m.z)
            nome = self.b(x, c.ys, z)
            self.assertIn(nome, LIBERO + ("plant", "log"), (m.specie, nome, x, z))


class TestCripta(unittest.TestCase):
    def setUp(self):
        self.c = un_castello()
        self.s, self.tav, self.b = posa_su_regione(self.c)

    def blocco(self, u, v, y):
        x, z = self.c.punto(u, v)
        return self.b(x, y, z)

    def prop(self, u, v, y):
        x, z = self.c.punto(u, v)
        return self.b.proprieta(x, y, z)

    def test_la_scala_scende_dalla_sala_con_il_verso_giusto(self):
        c = self.c
        for s_ in range(1, CS.PROFONDITA + 1):
            u = -11 + s_
            self.assertEqual(self.blocco(u, CS.VC, c.ys - s_), "stairs", s_)
            self.assertEqual(self.blocco(u, CS.VC, c.ys - s_ + 1), "air")
            self.assertEqual(self.blocco(u, CS.VC, c.ys - s_ + 2), "air")
            # `facing` e' il verso in cui si sale: verso la sala, cioe' -u
            self.assertEqual(self.prop(u, CS.VC, c.ys - s_)["facing"], c._faccia(-1, 0))
        # in cima la scala e' nella sala, sotto il soffitto della cripta c'e' roccia
        self.assertEqual(self.blocco(-10, CS.VC, c.ys), "air")

    def test_la_scala_non_riempie_la_sala_di_blocchi(self):
        c = self.c
        for u in range(-12, 0):
            for v in range(CS.VC - 1, CS.VC + 2):
                for y in range(c.ys, c.ys + 4):
                    self.assertNotEqual(self.blocco(u, v, y), "stone_bricks", (u, v, y))

    def test_dalla_scala_si_arriva_al_caveau_a_piedi_tranne_la_fossa_e_le_celle(self):
        c = self.c
        for u in list(range(4, 32)) + list(range(37, 49)) + list(range(49, 64)):
            for y in (c.yf, c.yf + 1):
                n = self.blocco(u, CS.VC, y)
                self.assertIn(n, LIBERO + ("tripwire", "tripwire_hook", "stone_bricks", "spawner",
                                           "pressure_plate", "stairs", "moss_block"), (u, y, n))

    def test_i_fili_d_inciampo_hanno_ganci_filo_e_i_distributori(self):
        c = self.c
        for u in CS.STAZIONI_FILO:
            self.assertEqual(self.blocco(u, CS.VC, c.yf), "tripwire")
            self.assertEqual(self.blocco(u, CS.VC + 1, c.yf), "tripwire_hook")
            self.assertEqual(self.blocco(u, CS.VC - 1, c.yf), "tripwire_hook")
            for lato in (1, -1):
                self.assertEqual(self.blocco(u, CS.VC + 2 * lato, c.yf + 1), "dispenser")
                # appoggiato al muro dietro il gancio, e guarda il corridoio
                self.assertEqual(self.blocco(u, CS.VC + 2 * lato, c.yf), "stone_bricks")
                self.assertEqual(self.prop(u, CS.VC + 2 * lato, c.yf + 1)["facing"], c._faccia(0, -lato))
        self.assertEqual(len(c.distributori()), 6)         # quattro nella cripta, due nella sala

    def test_i_ganci_guardano_il_corridoio_e_il_filo_corre_in_traverso(self):
        c = self.c
        u = CS.STAZIONI_FILO[0]
        self.assertEqual(self.prop(u, CS.VC + 1, c.yf)["facing"], c._faccia(0, -1))
        self.assertEqual(self.prop(u, CS.VC - 1, c.yf)["facing"], c._faccia(0, 1))
        f = self.prop(u, CS.VC, c.yf)
        lungo = ("north", "south") if c.asse_v() == "z" else ("east", "west")
        trasverso = ("east", "west") if c.asse_v() == "z" else ("north", "south")
        self.assertTrue(all(f[d] == "true" for d in lungo), f)
        self.assertTrue(all(f[d] == "false" for d in trasverso), f)

    def test_la_fossa_ha_le_punte_e_una_sponda_libera(self):
        c = self.c
        for u in range(CS.FOSSA_U[0], CS.FOSSA_U[1] + 1):
            for v in (CS.VC, CS.VC + 1):
                self.assertEqual(self.blocco(u, v, c.yf - 1), "air", (u, v))
                self.assertEqual(self.blocco(u, v, c.yf - 4), "pointed_dripstone", (u, v))
            self.assertEqual(self.blocco(u, CS.VC - 1, c.yf - 1), "stone_bricks", u)   # la sponda

    def test_il_caveau_ha_piastre_sopra_il_tnt_esche_sul_tnt_e_il_vero_tesoro(self):
        c = self.c
        ui = CS.U_CAVEAU[0]
        for uu in (ui + 1, ui + 2):
            for v in range(CS.VC - 2, CS.VC + 3):
                self.assertEqual(self.blocco(uu, v, c.yf), "pressure_plate")
                self.assertEqual(self.blocco(uu, v, c.yf - 1), "tnt")
        uc = CS.U_CAVEAU[1] - 1
        for v in (CS.VC - 2, CS.VC + 2):
            self.assertEqual(self.blocco(uc, v, c.yf), "trapped_chest")
            self.assertEqual(self.blocco(uc, v, c.yf - 1), "tnt")
        self.assertEqual(self.blocco(uc, CS.VC, c.yf), "chest")
        self.assertIn(self.blocco(uc, CS.VC, c.yf + 1), LIBERO)

    def test_le_celle_hanno_le_sbarre_e_la_cavita_dietro(self):
        c = self.c
        for (u0, u1) in CS.CELLE_U:
            for lato in (1, -1):
                for u in range(u0, u1 + 1):
                    for y in range(c.yf, c.yf + 3):
                        self.assertEqual(self.blocco(u, CS.VC + 2 * lato, y), "bars")
                    self.assertIn(self.blocco(u, CS.VC + 3 * lato, c.yf + 1), LIBERO)

    def test_gli_spawner_sono_tre_e_stanno_su_un_piedistallo(self):
        c = self.c
        sp = c.spawner()
        self.assertEqual(len(sp), 3)
        self.assertEqual([m for *_, m in sp][2], "cave_spider")
        for (x, y, z, mob) in sp:
            self.assertEqual(self.b(x, y, z), "spawner", mob)
            self.assertEqual(self.b(x, y - 1, z), "stone_bricks", mob)

    def test_i_forzieri_veri_sono_nove_e_con_i_tipi_giusti(self):
        c = self.c
        locali = c.forzieri_locali()
        self.assertEqual(len(locali), 9)
        self.assertEqual(sorted(t for *_, t in locali), ["stanza"] * 7 + ["tesoro"] * 2)
        veri = self.tav.bauli
        self.assertEqual(len(veri), 9)
        for x, y, z, tipo in veri:
            self.assertEqual(self.b(x, y, z), "chest", (x, y, z, tipo))
        self.assertEqual(sorted(t for *_, t in c.forzieri()), sorted(t for *_, t in locali))

    def test_le_guardie_e_i_prigionieri_nascono_in_aria_libera(self):
        c = self.c
        n = 0
        for m in CS.combattenti([c]):
            if m.y < c.ys - 3:
                n += 1
                x, z = int(m.x), int(m.z)
                self.assertIn(self.b(x, int(m.y), z), LIBERO + ("hay_block",), (m.specie, x, z))
                if m.specie != "pipistrello":
                    self.assertIn(self.b(x, int(m.y) + 1, z), LIBERO, (m.specie, x, z))
                    self.assertNotIn(self.b(x, int(m.y) - 1, z), ("air", None))
        self.assertGreaterEqual(n, 12)

    def test_le_quattro_direzioni_producono_lo_stesso_castello(self):
        for dire in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = un_castello(dire)
            s, tav, b = posa_su_regione(c)
            x, z = c.punto(CS.U_CAVEAU[1] - 1, CS.VC)
            self.assertEqual(b(x, c.yf, z), "chest", dire)
            sx, sy, sz, _ = c.spawner()[1]
            self.assertEqual(b(sx, sy, sz), "spawner", dire)
            self.assertEqual(len(tav.bauli), 9, dire)


class TestRifiniture(unittest.TestCase):
    def setUp(self):
        self.c = un_castello()
        self.s, self.tav, self.b = posa_su_regione(self.c)

    def blocco(self, u, v, y):
        x, z = self.c.punto(u, v)
        return self.b(x, y, z)

    def prop(self, u, v, y):
        x, z = self.c.punto(u, v)
        return self.b.proprieta(x, y, z)

    def test_la_sala_del_trono_ha_il_suo_filo_d_inciampo_con_due_distributori(self):
        c, M = self.c, CS.MEZZO_MASCHIO
        uf = CS.FILO_SALA_U
        self.assertEqual(self.blocco(uf, M - 1, c.ys + 1), "tripwire_hook")
        self.assertEqual(self.blocco(uf, -M + 1, c.ys + 1), "tripwire_hook")
        for v in range(-M + 2, M - 1):
            self.assertEqual(self.blocco(uf, v, c.ys + 1), "tripwire", v)
        for lato in (1, -1):
            self.assertEqual(self.blocco(uf, M * lato, c.ys + 1), "stone_bricks")
            self.assertEqual(self.blocco(uf, M * lato, c.ys + 2), "dispenser")
            self.assertEqual(self.prop(uf, M * lato, c.ys + 2)["facing"], c._faccia(0, -lato))
        # il filo non e' murato dentro un mobile: sopra la tavola, a un blocco dal pavimento
        self.assertEqual(self.blocco(uf, 0, c.ys + 2), "air")

    def test_il_filo_corre_lungo_l_asse_giusto_in_ogni_orientamento(self):
        for dire in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = un_castello(dire)
            s, tav, b = posa_su_regione(c)
            x, z = c.punto(CS.STAZIONI_FILO[0], CS.VC)
            f = b.proprieta(x, c.yf, z)
            lungo = ("north", "south") if c.asse_v() == "z" else ("east", "west")
            self.assertTrue(all(f[d] == "true" for d in lungo), (dire, f))
            x, z = c.punto(CS.FILO_SALA_U, 0)
            f = b.proprieta(x, c.ys + 1, z)
            self.assertTrue(all(f[d] == "true" for d in lungo), (dire, f))

    def test_i_due_nascondigli(self):
        c = self.c
        (u, v, y, t) = c.forzieri_locali()[7]
        self.assertEqual(self.blocco(u, v, y), "chest")
        # lo scaffale davanti: per arrivarci bisogna romperlo
        self.assertEqual(self.blocco(u - 1, v, y), "bookshelf")
        (u, v, y, t) = c.forzieri_locali()[8]
        self.assertEqual(self.blocco(u, v, y), "chest")
        self.assertEqual(self.blocco(u, v, y - 1), "bone_block")
        for yy in range(y + 1, c.ys):
            self.assertEqual(self.blocco(u, v, yy), "air")

    def test_terra_bruciata_vicino_alle_brecce(self):
        c, K = self.c, CS.K
        visti = set()
        for u in range(-K - 4, K + 5):
            for v in range(-K - 4, K + 5):
                visti.add(self.blocco(u, v, c.ys - 1))
        self.assertTrue({"blackstone", "magma_block", "coal_block"} & visti)

    def test_il_carro_rovesciato_e_i_lumi_dell_anima_nel_cortile(self):
        c = self.c
        self.assertEqual(self.blocco(9, 9, c.ys), "barrel")
        self.assertEqual(self.blocco(10, 9, c.ys), "hay_block")
        self.assertEqual(self.blocco(3, 6, c.ys), "soul_lantern")
        self.assertEqual(self.blocco(-14, 6, c.ys), "candle")

    def test_la_cripta_ha_mattoni_infestati_muschio_e_funghi(self):
        c = self.c
        visti = set()
        for u in range(4, 73):
            for v in range(CS.VC - 5, CS.VC + 6):
                for y in range(c.yf - 1, c.yf + 6):
                    visti.add(self.blocco(u, v, y))
        for atteso in ("infested_block", "moss_block", "moss_carpet", "candle"):
            self.assertIn(atteso, visti, atteso)
        self.assertTrue({"brown_mushroom", "red_mushroom"} & visti)

    def test_l_edera_cresce_dentro_le_torri(self):
        c, K = self.c, CS.K
        trovata = 0
        for su in (1, -1):
            for sv in (1, -1):
                for a in range(K - 5, K):
                    for b in range(K - 5, K):
                        for y in range(c.ys + 2, c.ys + 6):
                            if self.blocco(su * a, sv * b, y) == "vine":
                                trovata += 1
        self.assertGreater(trovata, 4)

    def test_gli_arcieri_stanno_sul_camminamento_intero(self):
        c = self.c
        posti = CS.posti_sulle_mura(c)
        self.assertEqual(len(posti), 4)
        for (u, v) in posti:
            self.assertEqual(self.blocco(u, v, c.ys + CS.ALTEZZA_MURA - 1), "stone_bricks")
            self.assertEqual(self.blocco(u, v, c.ys + CS.ALTEZZA_MURA), "air")
            self.assertEqual(self.blocco(u, v, c.ys + CS.ALTEZZA_MURA + 1), "air")
        arcieri = [m for m in CS.combattenti([c]) if getattr(m, "y", 0) == float(c.ys + CS.ALTEZZA_MURA)
                   and getattr(m, "specie", "") == "predone"]
        self.assertEqual(len(arcieri), 4)

    def test_il_luogotenente_sorveglia_il_caveau_e_i_pipistrelli_volano_nella_cripta(self):
        c = self.c
        m = CS.combattenti([c], [(c.x, c.z)])
        luog = next(x for x in m if getattr(x, "nome", "") == "Il Custode del Caveau")
        self.assertTrue(luog.capo)
        self.assertEqual(luog.y, float(c.yf))
        x, z = int(luog.x), int(luog.z)
        self.assertIn(self.b(x, c.yf, z), LIBERO)
        pip = [x for x in m if getattr(x, "specie", "") == "pipistrello"]
        self.assertEqual(len(pip), 5)
        for p in pip:
            self.assertEqual(self.b(int(p.x), int(p.y), int(p.z)), "air")

    def test_i_manichini_hanno_un_armatura_arrugginita_e_stanno_su_una_cella_libera(self):
        c = self.c
        man = [m for m in CS.combattenti([c]) if m.__class__.__name__ == "Manichino"]
        self.assertEqual(len(man), 4)
        for m in man:
            n = m.nbt_tag()
            self.assertEqual(str(n["id"]), "minecraft:armor_stand")
            self.assertEqual(int(n["ShowArms"]), 1)
            self.assertEqual(str(n["equipment"]["head"]["id"]), "minecraft:chainmail_helmet")
            self.assertEqual(str(n["equipment"]["chest"]["id"]), "minecraft:chainmail_chestplate")
            self.assertIn("mainhand", n["equipment"])
            x, z = int(m.x), int(m.z)
            self.assertIn(self.b(x, c.ys, z), LIBERO, (x, z))
            self.assertIn(self.b(x, c.ys + 1, z), LIBERO, (x, z))


class TestAbitantiEEntita(unittest.TestCase):
    def test_il_capo_e_il_seguito(self):
        c = un_castello()
        m = CS.combattenti([c], [(10, 10)])
        capi = [x for x in m if getattr(x, "capo", False)]
        self.assertEqual(len(capi), 2)                                   # il re e il luogotenente
        re = next(x for x in capi if x.nome == "Re Spettro Aldovrando")
        u, v, y = c.posto_capo()
        x, z = c.punto(u, v)
        self.assertEqual((int(re.x), re.y, int(re.z)), (x, float(y), z))
        specie = {x.specie for x in m if hasattr(x, "specie")}
        for atteso in ("zombie", "scheletro", "ragno", "ragno_grotta", "husk", "pipistrello",
                       "vindicatore"):
            self.assertIn(atteso, specie)
        self.assertGreaterEqual(len(m), 35)
        for x in m:
            n = x.nbt_tag()
            self.assertTrue(str(n["id"]).startswith("minecraft:"))
        ids = {str(x.nbt_tag()["id"]) for x in m}
        self.assertTrue({"minecraft:bat", "minecraft:armor_stand", "minecraft:husk"} <= ids)

    def test_lontano_dalle_citta_il_capo_e_di_diamante(self):
        c = un_castello()
        vicino = next(x for x in CS.combattenti([c], [(c.x, c.z)]) if x.capo)
        lontano = next(x for x in CS.combattenti([c], [(9999, 9999)]) if x.capo)
        self.assertEqual((vicino.livello, lontano.livello), (0, 1))

    def test_gli_spawner_e_i_distributori_diventano_block_entity_nel_chunk_giusto(self):
        c = un_castello()
        tutti = []
        for cz in range(c.bbox()[1] // 16 - 1, c.bbox()[3] // 16 + 2):
            for cx in range(c.bbox()[0] // 16 - 1, c.bbox()[2] // 16 + 2):
                tutti += CS.entita_blocco([c], cx * 16, cz * 16, 0)
        tipi = sorted(b.base_name for b in tutti)
        self.assertEqual(tipi, ["dispenser"] * 6 + ["spawner"] * 3)
        sp = [b for b in tutti if b.base_name == "spawner"]
        mob = sorted(str(b.nbt.compound["utags"]["SpawnData"]["entity"]["id"]) for b in sp)
        self.assertIn("minecraft:cave_spider", mob)
        disp = next(b for b in tutti if b.base_name == "dispenser")
        items = disp.nbt.compound["utags"]["Items"]
        self.assertEqual(str(items[0]["id"]), "minecraft:arrow")
        self.assertEqual(int(items[0]["count"]), 64)


class TestPersistenza(unittest.TestCase):
    def test_distributore_e_spawner_arrivano_nel_file_del_mondo_con_i_loro_oggetti(self):
        import glob
        import struct
        import tempfile
        import zlib

        import amulet_nbt as nbt

        from genworld.livello_dat import ImpostazioniMondo
        from genworld.mondo import ScrittoreMondo
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w")
            with ScrittoreMondo(p, ImpostazioniMondo(nome="P", modalita=1, versione=(26, 2, 0)),
                                lato_blocchi=32, crea=True) as m:
                tav = CS.Tavolozza(m)
                coords = list(m.chunk_coords())
                ox0, oz0 = m.origine_chunk(*coords[0])
                for k, (cx, cz) in enumerate(coords):
                    ox, oz = m.origine_chunk(cx, cz)
                    out = np.full((16, 140, 16), m.id_aria, np.uint32)
                    out[:, :20, :] = m.blocco("stone")
                    be = []
                    if k == 0:
                        out[3, 94, 3] = tav.distributore["north"]
                        out[8, 94, 8] = tav.spawner
                        be = [CS.DG.be_dispenser(ox0 + 3, 30, oz0 + 3),
                              CS.DG.be_spawner(ox0 + 8, 30, oz0 + 8, "cave_spider")]
                    m.scrivi_chunk(cx, cz, out,
                                   biomi=np.full((16, 16), m.bioma("plains"), np.uint32), bauli=be)
            trovati = {}
            for f in glob.glob(os.path.join(p, "**", "*.mca"), recursive=True):
                dati = open(f, "rb").read()
                for i in range(1024):
                    off = struct.unpack(">I", bytes([0]) + dati[i * 4:i * 4 + 3])[0]
                    if not off:
                        continue
                    pos = off * 4096
                    ln = struct.unpack(">I", dati[pos:pos + 4])[0]
                    tag = nbt.load(zlib.decompress(dati[pos + 5:pos + 4 + ln]),
                                   compressed=False, little_endian=False)
                    for be in tag.compound["block_entities"]:
                        trovati[str(be["id"])] = be
            disp = trovati["minecraft:dispenser"]
            self.assertEqual(str(disp["Items"][0]["id"]), "minecraft:arrow")
            self.assertEqual(int(disp["Items"][0]["count"]), 64)
            sp = trovati["minecraft:mob_spawner"]
            self.assertEqual(str(sp["SpawnData"]["entity"]["id"]), "minecraft:cave_spider")


class TestIndiceEMaschera(unittest.TestCase):
    def test_indice_e_maschera(self):
        c = un_castello()
        idx = CS.indice_per_chunk([c])
        x0, z0, x1, z1 = c.bbox()
        for (x, z) in ((x0, z0), (x1, z1), (c.x, c.z)):
            self.assertIn(0, idx.get((x // 16, z // 16), []))
        m = CS.maschera([c], (600, 600), margine=2)
        self.assertTrue(m[c.z, c.x])
        self.assertTrue(m[c.z + CS.K + 4, c.x])
        self.assertFalse(m[c.z + CS.K + 8, c.x])


class TestTraduzioni(unittest.TestCase):
    def test_ogni_blocco_della_tavolozza_si_traduce_in_un_blocco_di_gioco(self):
        import PyMCTranslate
        import amulet_nbt as n
        from amulet.api.block import Block
        v = PyMCTranslate.new_translation_manager().get_version("java", (26, 2, 0))
        s = Finto()
        CS.Tavolozza(s)
        cattivi = []
        for (nome, prop), _ in s.ids.items():
            if nome == "air":
                continue
            b = Block("universal_minecraft", nome, {k: n.StringTag(x) for k, x in prop})
            o, _, _ = v.block.from_universal(b)
            if o.namespace != "minecraft":
                cattivi.append((nome, dict(prop)))
        self.assertEqual(cattivi, [])


if __name__ == "__main__":
    unittest.main()
