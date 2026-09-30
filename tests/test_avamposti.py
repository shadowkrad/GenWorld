"""Test di campi, cimiteri e portali: pianificazione, indice per chunk, posa."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genworld import avamposti as AV  # noqa: E402
from genworld import template as TM  # noqa: E402

try:
    import amulet  # noqa: F401
    AMULET = True
except ImportError:
    AMULET = False

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_CIMITERO = os.path.join(RADICE, "templates", "cimiteri")
TEMPLATES_PORTALE = os.path.join(RADICE, "templates", "portali")
STRUTTURE = os.path.join(RADICE, "templates", "strutture")


class FintoScrittore:
    """Stesso aiuto di test_miniere.py: un id progressivo per ogni
    (nome, proprieta')."""
    id_aria = 0

    def __init__(self):
        self.voci: dict = {}

    def blocco(self, nome, **prop):
        chiave = (nome, tuple(sorted(prop.items())))
        return self.voci.setdefault(chiave, len(self.voci) + 1)


class TestAvamposto(unittest.TestCase):

    def test_il_raggio_dipende_dal_tipo(self):
        campo = AV.Avamposto(x=0, z=0, y=70, tipo="campo")
        cimitero = AV.Avamposto(x=0, z=0, y=70, tipo="cimitero")
        self.assertEqual(campo.raggio, AV.RAGGIO_CAMPO)
        self.assertEqual(cimitero.raggio, AV.RAGGIO_CIMITERO)


class TestPianifica(unittest.TestCase):

    def scena(self, n=300):
        h = np.full((n, n), 90.0, np.float32)
        mare = np.zeros((n, n), bool)
        mare[:6, :] = True
        return h, mare

    def test_ne_pianifica_almeno_uno_di_ciascun_tipo(self):
        h, mare = self.scena()
        av = AV.pianifica(h, mare, campi=3.0, cimiteri=3.0, seed=3)
        tipi = {a.tipo for a in av}
        self.assertEqual(tipi, {"campo", "cimitero"})

    def test_niente_campi_con_densita_zero(self):
        h, mare = self.scena()
        av = AV.pianifica(h, mare, campi=0.0, cimiteri=3.0, seed=3)
        self.assertFalse(any(a.tipo == "campo" for a in av))
        self.assertTrue(any(a.tipo == "cimitero" for a in av))

    def test_niente_cimiteri_con_densita_zero(self):
        h, mare = self.scena()
        av = AV.pianifica(h, mare, campi=3.0, cimiteri=0.0, seed=3)
        self.assertFalse(any(a.tipo == "cimitero" for a in av))
        self.assertTrue(any(a.tipo == "campo" for a in av))

    def test_niente_del_tutto_con_densita_zero(self):
        h, mare = self.scena()
        av = AV.pianifica(h, mare, campi=0.0, cimiteri=0.0, seed=3)
        self.assertEqual(av, [])

    def test_niente_ingressi_nell_acqua(self):
        h, mare = self.scena()
        for a in AV.pianifica(h, mare, campi=3.0, cimiteri=3.0, seed=17):
            self.assertFalse(mare[a.z, a.x])

    def test_evita_tiene_fuori_gli_avamposti(self):
        h, mare = self.scena()
        evita = np.ones(h.shape, bool)
        av = AV.pianifica(h, mare, evita=evita, campi=3.0, cimiteri=3.0, seed=19)
        self.assertEqual(av, [])

    def test_due_avamposti_non_si_toccano(self):
        h, mare = self.scena()
        av = AV.pianifica(h, mare, campi=5.0, cimiteri=5.0, seed=23)
        for i, a in enumerate(av):
            for b in av[i + 1:]:
                distanza = max(abs(a.x - b.x), abs(a.z - b.z))
                self.assertGreaterEqual(distanza, a.raggio + b.raggio,
                                        "due avamposti si sovrappongono")


class TestIndicePerChunk(unittest.TestCase):

    def test_il_campo_e_nel_suo_chunk(self):
        av = AV.Avamposto(x=20, z=20, y=90, tipo="campo")
        idx = AV.indice_per_chunk([av])
        self.assertIn(0, idx.get((1, 1), []))

    def test_un_cimitero_grande_tocca_piu_chunk(self):
        """Il cimitero e' largo 11 blocchi: vicino a un bordo di chunk deve
        registrarsi anche in quello accanto, in OGNI chunk che il suo
        ingombro tocca, non solo ai due angoli - a differenza del piccolo
        castelletto delle miniere, qui il raggio e' abbastanza grande da
        poter attraversare un chunk intero di mezzo."""
        av = AV.Avamposto(x=15, z=40, y=90, tipo="cimitero")
        idx = AV.indice_per_chunk([av])
        # da x=15-5=10 a x=15+5=20 attraversa i chunk 0 e 1
        for cx in (0, 1):
            self.assertIn(0, idx.get((cx, 2), []), f"manca il chunk ({cx}, 2)")


class TestMaschera(unittest.TestCase):

    def test_la_maschera_copre_l_ingombro(self):
        av = AV.Avamposto(x=20, z=20, y=90, tipo="campo")
        m = AV.maschera([av], (40, 40))
        self.assertTrue(m[20, 20])
        self.assertTrue(m[20 - AV.RAGGIO_CAMPO, 20])
        self.assertFalse(m[20 - AV.RAGGIO_CAMPO - 2, 20])


class TestNemici(unittest.TestCase):

    def test_ogni_campo_ha_dei_nemici(self):
        av = AV.Avamposto(x=20, z=20, y=90, tipo="campo")
        nem = AV.nemici([av], seed=1)
        self.assertGreater(len(nem), 0)
        self.assertTrue(all(n.specie in ("zombie", "scheletro") for n in nem))

    def test_il_cimitero_ha_almeno_uno_scheletro(self):
        av = AV.Avamposto(x=20, z=20, y=90, tipo="cimitero")
        nem = AV.nemici([av], seed=1)
        self.assertTrue(any(n.specie == "scheletro" for n in nem))

    def test_niente_nemici_senza_avamposti(self):
        self.assertEqual(AV.nemici([], seed=1), [])


class TestPosaCampo(unittest.TestCase):

    Y0 = -64
    ALTEZZA = 284
    SOLIDO = 9999

    def colonna_piena(self, quota_terreno=100):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        out[:, :quota_terreno - self.Y0, :] = self.SOLIDO
        return out

    def test_il_fuoco_e_al_centro(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="campo")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        self.assertEqual(int(out[8, base, 8]), tav.falo)

    def test_le_panche_sono_ai_quattro_lati(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="campo")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        self.assertEqual(int(out[8, base, 7]), tav.panca_x)
        self.assertEqual(int(out[8, base, 9]), tav.panca_x)
        self.assertEqual(int(out[7, base, 8]), tav.panca_z)
        self.assertEqual(int(out[9, base, 8]), tav.panca_z)

    def test_i_barili_sono_sugli_angoli_opposti(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="campo")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        r = AV.RAGGIO_CAMPO
        self.assertEqual(int(out[8 + r - 1, base, 8 + r - 1]), tav.barile)
        self.assertEqual(int(out[8 - r + 1, base, 8 - r + 1]), tav.barile)

    def test_la_staccionata_ha_un_varco_a_sud(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="campo")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        r = AV.RAGGIO_CAMPO
        self.assertNotEqual(int(out[8, base, 8 + r]), tav.staccionata)

    def test_protetto_c_blocca_il_campo(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="campo")
        protetto_c = np.zeros((16, 16), bool)
        protetto_c[8, 8] = True
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0], protetto_c=protetto_c)
        base = 90 - self.Y0
        self.assertNotEqual(int(out[8, base, 8]), tav.falo)


class TestPosaCimitero(unittest.TestCase):

    Y0 = -64
    ALTEZZA = 284
    SOLIDO = 9999

    def colonna_piena(self, quota_terreno=100):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        out[:, :quota_terreno - self.Y0, :] = self.SOLIDO
        return out

    def test_la_cripta_ha_un_ragnatela_al_centro(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="cimitero")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        self.assertEqual(int(out[8, base, 8]), tav.ragnatela)

    def test_la_cripta_ha_un_tetto(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="cimitero")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        self.assertEqual(int(out[8, base + 3, 8]), tav.cripta_a)

    def test_il_varco_della_cripta_resta_libero(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="cimitero")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        colonna = out[8, base:base + 3, 9]
        self.assertTrue((colonna != tav.cripta_a).all())
        self.assertTrue((colonna != tav.cripta_b).all())

    def test_il_muro_di_cinta_ha_un_cancello_a_sud(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="cimitero")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        r = AV.RAGGIO_CIMITERO
        self.assertEqual(int(out[8, base, 8 + r]), tav.cancello)
        self.assertEqual(int(out[8, base, 8 - r]), tav.muro)

    def test_nessuna_lapide_sul_vialetto_centrale(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="cimitero")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        colonna = out[8, base, 6:11]
        self.assertTrue((colonna != tav.lapide).all())

    def test_ci_sono_lapidi_nella_griglia(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="cimitero")
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0])
        self.assertTrue((out == tav.lapide).any())

    def test_protetto_c_blocca_il_cimitero(self):
        out = self.colonna_piena(90)
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=8, z=8, y=90, tipo="cimitero")
        protetto_c = np.zeros((16, 16), bool)
        protetto_c[8, 8] = True
        AV.posa(out, tav, 0, 0, self.Y0, [av], [0], protetto_c=protetto_c)
        base = 90 - self.Y0
        self.assertNotEqual(int(out[8, base, 8]), tav.ragnatela)


class TestConfineChunk(unittest.TestCase):
    """Stessa idea di TestIngresso in test_miniere.py: una struttura vicina
    al bordo del chunk deve comparire per intero, non "smezzata"."""

    Y0 = -64
    ALTEZZA = 284
    SOLIDO = 9999

    def colonna_piena(self, quota_terreno=100):
        out = np.zeros((16, self.ALTEZZA, 16), np.uint32)
        out[:, :quota_terreno - self.Y0, :] = self.SOLIDO
        return out

    def test_il_cimitero_attraversa_davvero_il_confine_del_chunk(self):
        s = FintoScrittore()
        tav = AV.Tavolozza(s)
        av = AV.Avamposto(x=15, z=8, y=90, tipo="cimitero")
        out1 = self.colonna_piena(90)
        out2 = self.colonna_piena(90)
        AV.posa(out1, tav, 0, 0, self.Y0, [av], [0])
        AV.posa(out2, tav, 16, 0, self.Y0, [av], [0])
        base = 90 - self.Y0
        # il lato vicino del muro (x=15-5=10) sta nel PRIMO chunk
        self.assertEqual(int(out1[10, base, 12]), tav.muro)
        # il lato lontano (x=15+5=20) sta nel SECONDO chunk (locale x=4) -
        # senza il margine giusto in `indice_per_chunk` non verrebbe mai
        # disegnato, esattamente come il castelletto "smezzato" delle
        # miniere prima della correzione
        self.assertEqual(int(out2[4, base, 12]), tav.muro)


class TestAvampostoNonQuadrato(unittest.TestCase):
    """Un cimitero da template puo' non essere quadrato (mezzo_x != mezzo_z)
    - a differenza del ripiego procedurale, sempre 11x11. `raggio` (usato
    solo per la spaziatura) e i vecchi test sopra non devono cambiare
    comportamento; `indice_per_chunk`/`maschera` invece devono rispettare i
    due assi separatamente."""

    def test_di_default_resta_quadrato_come_prima(self):
        av = AV.Avamposto(x=0, z=0, y=70, tipo="cimitero")
        self.assertEqual(av.mezzo_x, AV.RAGGIO_CIMITERO)
        self.assertEqual(av.mezzo_z, AV.RAGGIO_CIMITERO)
        self.assertEqual(av.raggio, AV.RAGGIO_CIMITERO)
        self.assertEqual(av.modello, -1)

    def test_raggio_e_il_massimo_dei_due_assi(self):
        av = AV.Avamposto(x=0, z=0, y=70, tipo="cimitero", mezzo_x=9, mezzo_z=20)
        self.assertEqual(av.raggio, 20)

    def test_indice_per_chunk_rispetta_gli_assi_separati(self):
        # mezzo_x piccolo, mezzo_z grande: attraversa piu' chunk in z, non in x
        av = AV.Avamposto(x=8, z=8, y=90, tipo="cimitero", mezzo_x=3, mezzo_z=30)
        idx = AV.indice_per_chunk([av])
        chunk_x = {cx for cx, cz in idx}
        chunk_z = {cz for cx, cz in idx}
        self.assertEqual(chunk_x, {0}, "mezzo_x=3 non deve uscire dal chunk 0 in x")
        self.assertGreater(len(chunk_z), 1, "mezzo_z=30 deve attraversare piu' chunk in z")

    def test_maschera_rispetta_gli_assi_separati(self):
        av = AV.Avamposto(x=20, z=20, y=90, tipo="cimitero", mezzo_x=3, mezzo_z=10)
        m = AV.maschera([av], (60, 60))
        self.assertTrue(m[20, 20 + 3])
        self.assertFalse(m[20, 20 + 5])
        self.assertTrue(m[20 + 10, 20])
        self.assertFalse(m[20 + 12, 20])


@unittest.skipUnless(AMULET and os.path.isdir(TEMPLATES_CIMITERO),
                     "amulet o templates/cimiteri non disponibili")
class TestCimiteroDaTemplate(unittest.TestCase):
    """Il cimitero dell'utente (vedi `carica_cimiteri`), non piu' quello
    disegnato da codice - "sostituisci quello che hai creato tu"."""

    def modelli(self):
        return AV.carica_cimiteri([TEMPLATES_CIMITERO])

    def test_carica_almeno_un_modello(self):
        modelli = self.modelli()
        self.assertGreater(len(modelli), 0)

    def test_nessun_blocco_non_tradotto_resta_nella_tavolozza(self):
        """La trappola documentata in `template.py`: un nome non tradotto si
        scrive benissimo e in gioco non c'e'. `_BLOCCHI_RECENTI` copre i
        blocchi noti non tradotti sotto la 1.21.9 - verificato scrivendo e
        rileggendo un mondo vero (vedi il doc di progetto), qui si verifica
        solo che la sostituzione sia stata applicata."""
        modelli = self.modelli()
        for m in modelli:
            nomi = {n for n, _ in m.tavolozza}
            comuni = nomi & set(AV._BLOCCHI_RECENTI)
            self.assertEqual(comuni, set(),
                             f"{m.nome}: blocchi non sostituiti: {comuni}")

    def test_pianifica_sceglie_un_modello_e_le_dimensioni_vere(self):
        h = np.full((300, 300), 90.0, np.float32)
        mare = np.zeros((300, 300), bool)
        modelli = self.modelli()
        av = AV.pianifica(h, mare, campi=0.0, cimiteri=5.0, seed=7,
                          modelli_cimitero=modelli)
        self.assertTrue(av, "il test presuppone almeno un cimitero piazzato")
        m = modelli[0]
        for a in av:
            self.assertGreaterEqual(a.modello, 0)
            self.assertEqual(a.mezzo_x, (m.dx + 1) // 2)
            self.assertEqual(a.mezzo_z, (m.dz + 1) // 2)

    def test_posa_disegna_il_modello_non_il_ripiego(self):
        """Un blocco che esiste solo nel template (la porta) deve comparire
        nel punto giusto; il ripiego procedurale non ha porte.

        `costruisci()`/`posa()` lavorano su UN chunk (16, H, 16) alla volta,
        con `ox`/`oz` l'angolo del chunk in coordinate di mappa - esattamente
        come le chiama `motore.scrivi()` (vedi `TestPosa` in
        test_template.py, stessa convenzione). Il cimitero e' piu' grande di
        un chunk, quindi il test sceglie il chunk che contiene la porta
        invece di inventarsi un array grande quanto tutto il modello, cosa
        che `costruisci()` non supporta (il ritaglio e' sempre largo 16)."""
        modelli = self.modelli()
        m = modelli[0]
        s = FintoScrittore()
        cat = TM.Catalogo(modelli, s)
        aria = s.id_aria
        basamento = s.blocco("cobblestone")

        id_porta = {i for i, (n, _) in enumerate(m.tavolozza) if n == "door"}
        self.assertTrue(id_porta, "il test presuppone un template con una porta")
        dove = np.argwhere(np.isin(m.celle, list(id_porta)))
        self.assertTrue(len(dove), "la porta non compare nelle celle del modello")
        dlx, dly, dlz = (int(v) for v in dove[0])
        # `out` contiene gli ID RISOLTI da `cat.id_palette` (vedi
        # `template.costruisci`: `ids[pezzo]`), non gli indici grezzi della
        # tavolozza - il confronto va fatto sugli ID risolti, non sugli
        # indici, altrimenti il test puo' risultare vero per coincidenza
        # (un altro blocco con lo stesso ID di un indice di porta).
        ids_risolti = cat.id_palette(0, 0)
        id_porta_risolti = {int(ids_risolti[i]) for i in id_porta}

        base = 0
        x0, z0 = 1000, 2000          # angolo del modello, lontano da 0
        wx, wz = x0 + dlx, z0 + dlz  # dove finisce la porta in coordinate di mappa
        ox, oz = (wx // 16) * 16, (wz // 16) * 16   # il chunk che la contiene

        y0 = base - 5
        H = base - y0 + dly + 5
        out = np.full((16, H, 16), aria, np.uint32)
        av = AV.Avamposto(x=x0 + m.dx // 2, z=z0 + m.dz // 2, y=base,
                          tipo="cimitero", modello=0,
                          mezzo_x=(m.dx + 1) // 2, mezzo_z=(m.dz + 1) // 2)
        AV.posa(out, AV.Tavolozza(s), ox, oz, y0, [av], [0], cat_cimitero=cat,
               aria_cimitero=aria, basamento_cimitero=basamento)
        self.assertTrue(np.isin(out, list(id_porta_risolti)).any(),
                        "la porta del template non e' stata disegnata")

    def test_nemici_trova_un_punto_vicino_alla_ragnatela(self):
        modelli = self.modelli()
        m = modelli[0]
        av = AV.Avamposto(x=100, z=100, y=90, tipo="cimitero", modello=0,
                          mezzo_x=(m.dx + 1) // 2, mezzo_z=(m.dz + 1) // 2)
        nem = AV.nemici([av], seed=1, modelli_cimitero=modelli)
        self.assertTrue(any(n.specie == "scheletro" for n in nem),
                        "il template ha una ragnatela: ci si aspetta uno scheletro")


class TestPortaleSenzaModelli(unittest.TestCase):
    """Niente ripiego per il portale, a differenza del cimitero: senza un
    template non ne compare nessuno, qualunque sia `portali` - vedi il
    commento in cima al modulo. Non serve amulet: non si carica nessun
    file."""

    def test_senza_modelli_non_si_piazza_nessun_portale(self):
        h = np.full((300, 300), 90.0, np.float32)
        mare = np.zeros((300, 300), bool)
        av = AV.pianifica(h, mare, campi=0.0, cimiteri=0.0, portali=1000.0,
                          seed=7, modelli_portale=None)
        self.assertFalse(any(a.tipo == "portale" for a in av))


@unittest.skipUnless(AMULET and os.path.isdir(TEMPLATES_PORTALE),
                     "amulet o templates/portali non disponibili")
class TestPortaleDaTemplate(unittest.TestCase):
    """Il portale ("dislocare in maniera casuale sulla mappa") - stesso
    meccanismo del cimitero da template, ma senza ripiego procedurale."""

    def modelli(self):
        return AV.carica_portali([TEMPLATES_PORTALE])

    def test_carica_almeno_un_modello(self):
        modelli = self.modelli()
        self.assertGreater(len(modelli), 0)

    def test_pianifica_sceglie_un_modello_e_le_dimensioni_vere(self):
        h = np.full((400, 400), 90.0, np.float32)
        mare = np.zeros((400, 400), bool)
        modelli = self.modelli()
        av = AV.pianifica(h, mare, campi=0.0, cimiteri=0.0, portali=1000.0,
                          seed=7, modelli_portale=modelli)
        self.assertTrue(av, "il test presuppone almeno un portale piazzato")
        m = modelli[0]
        for a in av:
            self.assertEqual(a.tipo, "portale")
            self.assertGreaterEqual(a.modello, 0)
            self.assertEqual(a.mezzo_x, (m.dx + 1) // 2)
            self.assertEqual(a.mezzo_z, (m.dz + 1) // 2)

    def test_posa_disegna_il_modello(self):
        """Stesso schema di `TestCimiteroDaTemplate.test_posa_disegna_il_
        modello_non_il_ripiego`: un chunk (16, H, 16), l'angolo scelto in
        modo che il blocco cercato ci cada dentro."""
        modelli = self.modelli()
        m = modelli[0]
        s = FintoScrittore()
        cat = TM.Catalogo(modelli, s)
        aria = s.id_aria
        basamento = s.blocco("cobblestone")

        id_frame = {i for i, (n, _) in enumerate(m.tavolozza)
                   if n in ("obsidian", "nether_portal", "nether_bricks")}
        self.assertTrue(id_frame, "il test presuppone un portale con una cornice")
        dove = np.argwhere(np.isin(m.celle, list(id_frame)))
        self.assertTrue(len(dove), "la cornice non compare nelle celle del modello")
        dlx, dly, dlz = (int(v) for v in dove[0])
        # confronto sugli ID RISOLTI, non sugli indici grezzi di tavolozza -
        # vedi il commento gemello in test_posa_disegna_il_modello_non_il_
        # ripiego (TestCimiteroDaTemplate)
        ids_risolti = cat.id_palette(0, 0)
        id_frame_risolti = {int(ids_risolti[i]) for i in id_frame}

        base = 0
        x0, z0 = 3000, 4000
        wx, wz = x0 + dlx, z0 + dlz
        ox, oz = (wx // 16) * 16, (wz // 16) * 16

        y0 = base - 5
        H = base - y0 + dly + 5
        out = np.full((16, H, 16), aria, np.uint32)
        av = AV.Avamposto(x=x0 + m.dx // 2, z=z0 + m.dz // 2, y=base,
                          tipo="portale", modello=0,
                          mezzo_x=(m.dx + 1) // 2, mezzo_z=(m.dz + 1) // 2)
        AV.posa(out, AV.Tavolozza(s), ox, oz, y0, [av], [0], cat_portale=cat,
               aria_portale=aria, basamento_portale=basamento)
        self.assertTrue(np.isin(out, list(id_frame_risolti)).any(),
                        "la cornice del portale non e' stata disegnata")

    def test_nessun_nemico_piazzato_a_mano_per_un_portale(self):
        modelli = self.modelli()
        m = modelli[0]
        av = AV.Avamposto(x=100, z=100, y=90, tipo="portale", modello=0,
                          mezzo_x=(m.dx + 1) // 2, mezzo_z=(m.dz + 1) // 2)
        nem = AV.nemici([av], seed=1, modelli_cimitero=None)
        self.assertEqual(nem, [])


class TestStatistichePerTipo(unittest.TestCase):
    """Regressione: `statistiche()` contava 'cimitero' tutto cio' che non
    era 'campo', il che avrebbe contato un portale come cimitero appena
    aggiunto un terzo tipo."""

    def test_i_tre_tipi_si_contano_separatamente(self):
        av = [AV.Avamposto(x=0, z=0, y=0, tipo="campo"),
             AV.Avamposto(x=50, z=50, y=0, tipo="cimitero"),
             AV.Avamposto(x=100, z=100, y=0, tipo="portale"),
             AV.Avamposto(x=150, z=150, y=0, tipo="portale")]
        st = AV.statistiche(av)
        self.assertEqual(st, {"avamposti": 4, "campi": 1, "cimiteri": 1, "portali": 2})


class TestSicurezzaFileEstranei(unittest.TestCase):
    """I file che l'utente ha buttato in `templates/strutture/` insieme alle
    case vere (il cimitero, il portale, e un cactus/pozzo decorativi non
    ancora usati da nessuna parte) non devono MAI essere scelti come casa -
    sarebbe un lotto con un cimitero al posto dell'abitazione.
    `templates/cimiteri`, `templates/portali` e `templates/altro` sono la
    sistemazione vera; questo test protegge la rete di sicurezza per le
    copie che restano in `templates/strutture` finche' non vengono tolte a
    mano (questa sessione non ha potuto cancellarle sul PC dell'utente -
    vedi il doc di progetto)."""

    ESTRANEI = ("cementerio-grav-9ggj8p63", "decorated-cactu-b079aqr8",
               "pozo-rstico-hydk8c1z", "portal-del-neth-54ojw1ng")

    @unittest.skipUnless(os.path.isfile(os.path.join(STRUTTURE, "stili.json")),
                         "templates/strutture/stili.json non disponibile")
    def test_i_file_estranei_hanno_uno_stile_escluso(self):
        stili = TM.carica_stili(STRUTTURE)
        for nome in self.ESTRANEI:
            self.assertEqual(stili.get(nome), frozenset({"_escluso"}),
                             f"{nome} non e' marcato escluso in stili.json")

    def test_uno_stile_escluso_non_viene_mai_scelto(self):
        """Prova diretta su `template.scegli()`: un modello con
        `stili={'_escluso'}` non deve mai comparire fra i candidati per
        nessuno dei quattro stili veri."""
        finto = TM.Modello(nome="finto-escluso",
                           celle=np.zeros((3, 3, 3), np.int32),
                           tavolozza=[("air", {})], stili=frozenset({"_escluso"}))
        rng = np.random.default_rng(0)
        for stile in ("bosco", "montagna", "deserto", "prato"):
            s = TM.scegli([finto], 10, 10, 0, rng, stile=stile)
            self.assertIsNone(s, f"il modello escluso e' stato scelto per {stile}")


if __name__ == "__main__":
    unittest.main()
