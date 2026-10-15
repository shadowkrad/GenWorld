"""Luoghi con nome, cartelli, elenco e catena di indizi (#7, #8)."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genworld import bauli as BA
from genworld import dungeon as DG
from genworld import luoghi as LU
from genworld import torri as TR


def citta(x, z, raggio=30, murata=False):
    return NS(x=x, z=z, raggio=raggio, murata=murata)


def mondo():
    """Una citta' al centro, due rovine, una torre, una miniera, un campo."""
    meta = 500
    cs = [citta(500, 500), citta(800, 300, 20, True)]
    ds = []
    for x, z in ((600, 520), (650, 700)):
        d = DG.Dungeon(x=x, z=z, ys=90, dx=1, dz=0, mob="zombie", seme=1)
        d.scatole = DG._scatole(90)
        ds.append(d)
    ts = [TR.Torre(x=420, z=380, ys=90, dx=1, dz=0, nome="Aldric il Cupo", seme=0)]
    mn = NS(x=520, z=450, y_superficie=80, rampa=NS(dx=0, dz=1))
    av = [NS(x=300, z=600, y=70, tipo="campo"), NS(x=380, z=640, y=70, tipo="cimitero")]
    return meta, cs, [mn], ds, ts, av


class TestNomi(unittest.TestCase):
    def test_ogni_luogo_ha_un_nome_unico_e_coordinate_di_gioco(self):
        meta, cs, ms, ds, ts, av = mondo()
        luoghi = LU.raccogli(cs, ms, ds, ts, av, meta, seed=1)
        self.assertEqual(len(luoghi), 2 + 1 + 2 + 1 + 2)
        nomi = [l.nome for l in luoghi]
        self.assertEqual(len(nomi), len(set(nomi)), "nomi ripetuti")
        centro = next(l for l in luoghi if l.tipo == "citta" and l.nota.startswith("raggio 30"))
        self.assertEqual((centro.x, centro.z), (0, 0))
        torre = next(l for l in luoghi if l.tipo == "torre")
        self.assertEqual(torre.nome, "Torre di Aldric il Cupo")

    def test_e_deterministico_e_ci_sono_abbastanza_nomi_per_molti_luoghi(self):
        meta, cs, ms, ds, ts, av = mondo()
        a = [l.nome for l in LU.raccogli(cs, ms, ds, ts, av, meta, seed=3)]
        b = [l.nome for l in LU.raccogli(cs, ms, ds, ts, av, meta, seed=3)]
        self.assertEqual(a, b)
        molte = [citta(100 + i, 100 + i) for i in range(60)]
        nomi = [l.nome for l in LU.raccogli(molte, [], [], [], [], 0, seed=1)]
        self.assertEqual(len(nomi), len(set(nomi)))


class TestDirezioni(unittest.TestCase):
    def test_punti_cardinali(self):
        self.assertEqual(LU.direzione(10, 0), "est")
        self.assertEqual(LU.direzione(-10, 0), "ovest")
        self.assertEqual(LU.direzione(0, 10), "sud")
        self.assertEqual(LU.direzione(0, -10), "nord")
        self.assertEqual(LU.direzione(10, -10), "nord-est")
        self.assertEqual(LU.direzione(-10, 10), "sud-ovest")


class TestElenco(unittest.TestCase):
    def test_scrive_md_e_json_con_le_coordinate(self):
        meta, cs, ms, ds, ts, av = mondo()
        luoghi = LU.raccogli(cs, ms, ds, ts, av, meta, seed=1)
        with tempfile.TemporaryDirectory() as d:
            md, js = LU.scrivi_elenco(luoghi, d, "Arda")
            testo = open(md, encoding="utf-8").read()
            self.assertIn("# Luoghi di Arda", testo)
            self.assertIn("Torre di Aldric il Cupo", testo)
            self.assertIn("x=-80, z=-120", testo)
            dati = json.load(open(js, encoding="utf-8"))
            self.assertEqual(len(dati), len(luoghi))
            self.assertEqual({d_["tipo"] for d_ in dati},
                             {"citta", "miniera", "rovina", "torre", "accampamento", "cimitero"})


class TestCatena(unittest.TestCase):
    def test_parte_dalla_citta_piu_vicina_al_centro_e_visita_ogni_tappa_una_volta(self):
        meta, cs, ms, ds, ts, av = mondo()
        luoghi = LU.raccogli(cs, ms, ds, ts, av, meta, seed=1)
        partenza, tappe = LU.catena(luoghi)
        self.assertEqual((partenza.x, partenza.z), (0, 0))
        self.assertEqual(len(tappe), 3)
        self.assertEqual(len({(t.x, t.z) for t in tappe}), 3)
        self.assertTrue(all(t.tipo in LU.TIPI_AVVENTURA for t in tappe))
        # la prima tappa e' la piu' vicina alla partenza
        self.assertEqual(LU.distanza(partenza, tappe[0]),
                         min(LU.distanza(partenza, t) for t in tappe))

    def test_senza_citta_o_senza_tappe_nessuna_catena(self):
        self.assertEqual(LU.catena([]), (None, []))
        self.assertEqual(LU.indizi([LU.Luogo("A", "citta", 0, 0)])["inizio"], None)

    def test_gli_indizi_puntano_alla_tappa_successiva_e_l_ultima_chiude(self):
        meta, cs, ms, ds, ts, av = mondo()
        luoghi = LU.raccogli(cs, ms, ds, ts, av, meta, seed=1)
        ind = LU.indizi(luoghi)
        partenza, libro0 = ind["inizio"]
        _, tappe = LU.catena(luoghi)
        self.assertIn(tappe[0].nome, " ".join(libro0.pagine))
        self.assertIn("passi", " ".join(libro0.pagine))
        for i, t in enumerate(tappe):
            libro = ind["tappe"][(t.tipo, t.x, t.z)]
            testo = " ".join(libro.pagine)
            if i + 1 < len(tappe):
                self.assertIn(tappe[i + 1].nome, testo)
            else:
                self.assertIn("tesoro e' tuo", testo)
        for libro in [libro0] + list(ind["tappe"].values()):
            for p in libro.pagine:
                self.assertLessEqual(len(p), 255)
            self.assertLessEqual(len(libro.titolo), 32)


class TestCartelli(unittest.TestCase):
    def test_un_cartello_davanti_a_ogni_rovina_torre_e_miniera(self):
        meta, cs, ms, ds, ts, av = mondo()
        luoghi = LU.raccogli(cs, ms, ds, ts, av, meta, seed=1)
        cart = LU.cartelli(ds, ts, ms, luoghi, meta)
        self.assertEqual(len(cart), 2 + 1 + 1)
        torre = next(c for c in cart if "Torre di" in c.righe[0])
        self.assertEqual(torre.righe[3], "Aldric il Cupo")
        for c in cart:
            self.assertEqual(len(c.righe), 4)
            self.assertTrue(all(len(r) <= 15 or len(r) <= 25 for r in c.righe))
            self.assertIn(c.rotazione, (0, 4, 8, 12))

    def test_il_cartello_guarda_fuori_dalla_struttura(self):
        d = DG.Dungeon(x=100, z=100, ys=90, dx=1, dz=0, mob="zombie", seme=1)
        d.scatole = DG._scatole(90)
        c = LU.cartelli([d], [], [], [LU.Luogo("Cripta", "rovina", 0, 0)], 0)[0]
        self.assertEqual(c.rotazione, LU.ROTAZIONE[(-1, 0)])       # l'ingresso e' sul lato -u
        self.assertLess(c.x, 100 - DG.RUINA)

    def test_il_blockentity_del_cartello(self):
        c = LU.Cartello(10, 20, 4, ("Riga uno", "", "", ""))
        be = LU.entita_cartello(c, 70, 5)
        self.assertEqual((be.x, be.y, be.z), (5, 70, 15))
        ut = be.nbt.compound["utags"]
        self.assertEqual([str(x) for x in ut["front_text"]["java_nbt"]][0], "Riga uno")


class TestLibriNeiForzieri(unittest.TestCase):
    def chunk(self):
        class F:
            id_aria = 0
            def __init__(self):
                self.v = {}
            def blocco(self, n, **p):
                return self.v.setdefault((n, tuple(sorted(p.items()))), len(self.v) + 1)
        s = F()
        tav = BA.Tavolozza(s)
        out = np.zeros((16, 40, 16), np.uint32)
        out[5, 10, 6] = s.blocco("chest", facing="north", connection="none", material="wood")
        return out, tav

    def test_il_libro_di_tappa_finisce_nel_forziere_giusto(self):
        out, tav = self.chunk()
        libro = LU.Libro("Carte di X", "Il Custode", ("Pagina uno", "Pagina due"))
        b = BA.trova(out, tav, 0, 0, 0, seed=1, dungeon={(5, 10, 6): "tesoro"},
                     libri={(5, 10, 6): libro})
        libri = [v for v in b[0].nbt.compound["utags"]["Items"] if "written_book" in str(v["id"])]
        self.assertEqual(len(libri), 1)
        c = libri[0]["components"]["minecraft:written_book_content"]
        self.assertEqual(str(c["title"]["raw"]), "Carte di X")
        self.assertEqual([str(p["raw"]) for p in c["pages"]], ["Pagina uno", "Pagina due"])
        slots = [int(v["Slot"]) for v in b[0].nbt.compound["utags"]["Items"]]
        self.assertEqual(len(slots), len(set(slots)), "due pile nello stesso slot")

    def test_il_libro_iniziale_sta_in_alcune_case_della_citta_di_partenza(self):
        libro = LU.Libro("Il diario", "Un viandante", ("Ciao",))
        con, senza = 0, 0
        for ox in range(0, 800, 16):
            out, tav = self.chunk()
            b = BA.trova(out, tav, ox, 0, 0, seed=1, libro_citta=(ox + 6, 6, 400, libro))
            tiene = any("written_book" in str(v["id"])
                        for v in b[0].nbt.compound["utags"]["Items"])
            con += tiene
            senza += not tiene
        self.assertGreater(con, 3)
        self.assertGreater(senza, 3)

    def test_nessun_libro_lontano_dalla_citta_o_in_un_forziere_di_dungeon_o_cripta(self):
        libro = LU.Libro("Il diario", "Un viandante", ("Ciao",))
        for ox in range(0, 400, 16):
            out, tav = self.chunk()
            lontano = BA.trova(out, tav, ox, 0, 0, seed=1, libro_citta=(9000, 9000, 50, libro))
            cripta = BA.trova(out, tav, ox, 0, 0, seed=1, cripte=[(ox + 5, 6, 3)],
                              libro_citta=(ox + 6, 6, 400, libro))
            for b in lontano + cripta:
                self.assertFalse(any("written_book" in str(v["id"])
                                     for v in b.nbt.compound["utags"]["Items"]))


class TestPersistenza(unittest.TestCase):
    def test_cartello_e_libro_arrivano_nel_file_del_mondo(self):
        import glob
        import struct
        import zlib

        import amulet_nbt as nbt

        from genworld.livello_dat import ImpostazioniMondo
        from genworld.mondo import ScrittoreMondo
        with tempfile.TemporaryDirectory() as dir_:
            percorso = os.path.join(dir_, "w")
            libro = LU.Libro("Carte di X", "Il Custode", ("Pagina uno",))
            with ScrittoreMondo(percorso, ImpostazioniMondo(nome="P", modalita=1,
                                                            versione=(26, 2, 0)),
                                lato_blocchi=32, crea=True) as m:
                tav_ba = BA.Tavolozza(m)
                coords = list(m.chunk_coords())
                ox0, oz0 = m.origine_chunk(*coords[0])
                for k, (cx, cz) in enumerate(coords):
                    ox, oz = m.origine_chunk(cx, cz)
                    out = np.full((16, 140, 16), m.id_aria, np.uint32)
                    out[:, :20, :] = m.blocco("stone")
                    be = []
                    if k == 0:
                        out[3, 84, 3] = m.blocco("sign", material="oak", rotation="4")
                        be.append(LU.entita_cartello(LU.Cartello(ox0 + 3, oz0 + 3, 4,
                                                                  ("Torre di Prova", "", "Qui comanda", "Tizio")),
                                                     20, 0))
                        out[8, 84, 8] = m.blocco("chest", facing="north", connection="none",
                                                 material="wood")
                        be += BA.trova(out, tav_ba, ox, oz, -64, seed=1,
                                       dungeon={(ox0 + 8, 20, oz0 + 8): "tesoro"},
                                       libri={(ox0 + 8, 20, oz0 + 8): libro})
                    m.scrivi_chunk(cx, cz, out,
                                   biomi=np.full((16, 16), m.bioma("plains"), np.uint32),
                                   bauli=be)
            cartelli, forzieri = [], []
            for f in glob.glob(os.path.join(percorso, "**", "*.mca"), recursive=True):
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
                        if "sign" in str(be["id"]):
                            cartelli.append(be)
                        if "chest" in str(be["id"]):
                            forzieri.append(be)
            self.assertEqual(len(cartelli), 1)
            msg = [str(x) for x in cartelli[0]["front_text"]["messages"]]
            self.assertEqual(msg, ["Torre di Prova", "", "Qui comanda", "Tizio"])
            self.assertEqual(len(forzieri), 1)
            libri = [v for v in forzieri[0]["Items"] if "written_book" in str(v["id"])]
            self.assertEqual(len(libri), 1)
            c = libri[0]["components"]["minecraft:written_book_content"]
            self.assertEqual(str(c["title"]["raw"]), "Carte di X")
            self.assertEqual(str(c["pages"][0]["raw"]), "Pagina uno")


if __name__ == "__main__":
    unittest.main()
