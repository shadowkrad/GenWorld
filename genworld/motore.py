"""Il motore: da un'immagine a un mondo Minecraft, in una funzione.

Questo modulo era il corpo di `esempi/genera_mappa.py`. E' stato estratto
perche' adesso ci sono due modi di guidarlo - la riga di comando e la finestra
- e una pipeline duplicata sarebbe una pipeline che diverge. La CLI e la GUI
chiamano le stesse due funzioni:

    analizza(op)             immagine -> classi, quote, acque (con cache)
    genera(op, avanza=...)   analisi -> file region, con avanzamento

L'ordine delle fasi non e' negoziabile, e il perche' e' scritto nei commenti
lungo il codice: vulcani prima dei fiumi, insediamenti e strade prima della
vegetazione, e tutto prima di scrivere un solo chunk.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from . import agricoltura as AG
from . import batimetria as BAT
from . import biomi as B
from . import citta as CT
from . import edifici as E
from . import entita as EN
from . import fiumi as R
from . import insediamenti as I
from . import mappa as M
from . import normalizza as N
from . import sottosuolo as SS
from . import strade as ST
from . import stratigrafia as SG
from . import template as TM
from . import vegetazione as V
from . import vulcani as U
from .classi_auto import classifica_adattiva
from .erosione import erodi
from .profilo import Profilo, maschera_decoro
from .rumore import dettaglio, quantizza

LIVELLO_MARE = 62

# Blocchi di superficie per classe di terreno.
SUPERFICIE = {
    M.OCEANO:   ("gravel", "stone"),
    M.MARE:     ("sand", "sandstone"),
    M.FIUME:    ("gravel", "stone"),
    M.VULCANO:  ("basalt", "blackstone"),
    M.CRATERE:  ("magma_block", "blackstone"),
    M.SPIAGGIA: ("sand", "sandstone"),
    M.DESERTO:  ("sand", "sandstone"),
    M.PIANURA:  ("grass_block", "dirt"),
    M.PRATERIA: ("grass_block", "dirt"),
    M.FORESTA:  ("grass_block", "dirt"),
    M.MONTAGNA: ("stone", "stone"),
    M.NEVE:     ("snow_block", "stone"),
}


# --------------------------------------------------------------------------
# Opzioni
# --------------------------------------------------------------------------

@dataclass
class Opzioni:
    """Tutto quello che decide come viene il mondo.

    Un solo oggetto, cosi' la chiave di cache e' la firma dell'analisi e non
    una lista di parametri da tenere allineata a mano in due punti.
    """

    immagine: str
    uscita: str                      # cartella del mondo da creare
    nome: str = "Mondo"
    lato: int = 832
    ritaglio: float = 0.025
    profilo: str | None = None
    adattivo: bool = False
    erosione: float = 0.0
    fiumi: float = 150.0
    vulcani: int = 3
    alberi: float = 1.0
    villaggi: float = 1.0
    strade: bool = True
    caverne: float = 1.0
    # Cartella dei `.nbt` di blocco struttura da usare come case. Vuota o
    # inesistente: si usa il generatore parametrico, come prima.
    templates: str = ""
    seed: int = 11

    @property
    def cache(self) -> str:
        return self.uscita.rstrip("/\\") + "_cache.npz"

    def modo(self) -> str:
        """Chiave di cache: cambiando modo di classificare va rifatta l'analisi."""
        if self.profilo:
            return "profilo:" + os.path.basename(self.profilo)
        return "adattivo" if self.adattivo else "soglie"

    def firma_analisi(self) -> tuple:
        """Solo i parametri che cambiano il risultato di `analizza`.

        Alberi, villaggi e strade NON ci sono: si calcolano dopo, e cambiarli
        non deve invalidare l'analisi, che e' la parte lenta.
        """
        return (int(self.lato), self.modo(), float(self.erosione),
                float(self.fiumi), int(self.vulcani), float(self.ritaglio),
                os.path.abspath(self.immagine))


@dataclass
class Analisi:
    cls: np.ndarray
    h: np.ndarray
    livello: np.ndarray
    lava: np.ndarray
    colata: np.ndarray
    biomi: np.ndarray | None = None
    note: list[str] = field(default_factory=list)


Avanzamento = Callable[[float, str], None]


def _nulla(frazione: float, testo: str) -> None:
    pass


# --------------------------------------------------------------------------
# Analisi
# --------------------------------------------------------------------------

def analizza(op: Opzioni, avanza: Avanzamento = _nulla,
             usa_cache: bool = True) -> Analisi:
    """Immagine -> (classi, quote, acque). Risultato messo in cache."""
    if usa_cache:
        cached = _leggi_cache(op)
        if cached is not None:
            avanza(1.0, "analisi ripresa dalla cache")
            return cached

    note: list[str] = []
    avanza(0.02, "lettura dell'immagine")

    prof = Profilo.carica(op.profilo) if op.profilo else None
    if prof is not None:
        # Il rettangolo di interesse sostituisce il ritaglio automatico: e'
        # la persona a dire dov'e' la mappa, e coglie cornici illustrate e
        # riquadri di legenda che nessuna euristica trova.
        note.append(f"profilo: {prof.nome} - {prof.n_campioni()} campioni"
                    + (f", rettangolo {prof.roi}" if prof.roi else ", nessun rettangolo"))
        rgb = prof.leggi_immagine(op.immagine, op.lato)
    else:
        rgb = M.carica(op.immagine, lato=op.lato, ritaglio=op.ritaglio)

    avanza(0.12, "maschera delle scritte")
    testo = M.maschera_testo(rgb)
    rgb = M.ripara(rgb, testo)
    fam, info = N.famiglia(rgb)
    note.append(f"famiglia riconosciuta: {fam}")
    if fam == "grigi":
        note.append("ATTENZIONE: mappa senza colore. La classificazione per "
                    "colore non puo' funzionare, serve una legenda dichiarata.")

    avanza(0.22, "classificazione")
    rug = M.rugosita(rgb, maschera=testo)
    if prof is not None:
        cls = prof.classifica(rgb, rug)
    elif op.adattivo:
        cls = classifica_adattiva(rgb, rug)[0]
    else:
        cls = M.classifica(rgb, rug)

    # Il decoro non e' terreno: dove c'e', il mondo resta acqua profonda
    # invece di inventare un continente dentro una cornice.
    deco = maschera_decoro(cls)
    if deco.any():
        note.append(f"decoro escluso: {deco.mean() * 100:.1f}% dell'area")
        cls[deco] = M.OCEANO

    avanza(0.34, "altimetria dedotta")
    h = M.altimetria(cls, rug, prof.quote_effettive() if prof else None)

    # L'erosione va PRIMA del dettaglio frattale: applicata dopo, lavora su un
    # campo gia' dominato dal rumore e non si vede. Misurato su Arda: applicata
    # in coda cambia la pendenza media da 1,51 a 1,52, cioe' niente.
    if op.erosione > 0:
        avanza(0.40, "erosione idraulica")
        terra = ~np.isin(cls, M.ACQUA)
        h = erodi(h, gocce_per_cella=op.erosione, passi=44, maschera=terra,
                  quota_minima=float(LIVELLO_MARE + 1), seed=op.seed)
    h = h + dettaglio(h, ampiezza_base=0.7, ampiezza_pendenza=3.5, seed=op.seed)
    h = quantizza(h, forza=1.0, seed=op.seed)
    # IL FONDALE si calcola, non si taglia. Il vecchio codice imponeva un
    # tetto (-3 al mare, -12 all'oceano) e prendeva il minimo: siccome sotto
    # non c'era niente che generasse rilievo, tutto si appiattiva contro quel
    # tetto. Risultato misurato su Arda: il 53% dell'acqua a esattamente y=54,
    # un piano unico, e un muro di 67 blocchi al confine fra le due classi.
    # Sostituire invece di tagliare risolve anche il vecchio problema degli
    # isolotti da un blocco, perche' la quota marina non viene piu' dal
    # rumore.
    avanza(0.44, "fondale")
    h = BAT.applica(h, cls, livello_mare=LIVELLO_MARE, seed=op.seed)
    # i fiumi restano dove sono: scorrono in quota, non sono mare
    fiume = cls == M.FIUME
    if fiume.any():
        h = np.where(fiume, np.minimum(h, LIVELLO_MARE - 1), h)
    h = np.clip(h, -60, 300).astype(np.int32)
    st = BAT.statistiche(h, cls, LIVELLO_MARE)
    if st["acqua"]:
        note.append(
            f"fondale: profondita' media {st['prof_media']:.0f} blocchi, "
            f"massima {st['prof_max']:.0f}, {st['quote_distinte']} quote "
            f"distinte, la piu' diffusa copre il "
            f"{st['quota_piu_diffusa'] * 100:.0f}% dell'acqua, "
            f"{st['scalini']} scalini oltre 8 blocchi")

    # VULCANI. Vanno prima dei fiumi: sono l'unico elemento che si SOVRAPPONE
    # al terreno invece di dedurlo, e il deflusso deve poter scendere dai loro
    # fianchi. Calcolarli dopo darebbe coni senza torrenti.
    lava = np.zeros(h.shape, bool)
    colata = np.zeros(h.shape, np.float32)
    q_lava = np.zeros(h.shape, np.float32)
    if op.vulcani > 0:
        avanza(0.52, "vulcani")
        coni = U.scegli_siti(cls, h, quanti=op.vulcani,
                             livello_mare=LIVELLO_MARE, seed=9)
        if coni:
            cls, h, lava, q_lava = U.modella(cls, h, coni)
            colata = U.colate(h, cls, coni)
            st = U.statistiche(coni, cls, lava, colata)
            note.append(f"vulcani: {st['vulcani']}, cono {st['cono']} celle, "
                        f"cratere {st['cratere']}, lago di lava "
                        f"{st['lago_di_lava']}, colate {st['colate']} "
                        f"(di cui {st['colate_incandescenti']} incandescenti), "
                        f"cima a quota {st['quota_massima']}")

    # Fiumi. NON estratti dal disegno: calcolati dal terreno. Tre tentativi di
    # estrazione sono falliti perche' a questa risoluzione un fiume disegnato
    # e' largo un pixel e il suo colore si confonde con le ombre del rilievo.
    livello = np.full(h.shape, float(LIVELLO_MARE), np.float32)
    if lava.any():
        livello = np.where(lava, q_lava, livello)
    if op.fiumi > 0:
        avanza(0.68, "reticolo idrografico")
        marino = np.isin(cls, M.MARINO)
        maschera_f, acc = R.da_terreno(h, marino, soglia=op.fiumi)
        if maschera_f.any():
            livello, hf = R.livella(maschera_f, h, marino,
                                    livello_mare=LIVELLO_MARE)
            # la lava sta nel cratere: il suo pelo e' il suo, non quello del mare
            livello = np.where(lava, q_lava, livello)
            h = np.round(hf).astype(np.int32)
            cls = cls.copy()
            cls[maschera_f] = M.FIUME
            st = R.statistiche(maschera_f, livello, LIVELLO_MARE)
            note.append(f"fiumi: {st['celle']} celle in {st['corsi']} corsi, "
                        f"quota massima {st['quota_max']:.0f} "
                        f"(bacino maggiore {int(acc[~marino].max())} celle)")

    # I biomi per ultimi: dipendono dalle classi, e i fiumi le cambiano.
    # Calcolati prima, un corso d'acqua sarebbe rimasto pianura.
    avanza(0.88, "biomi")
    bio = B.assegna(cls, h, livello_mare=LIVELLO_MARE, seed=op.seed)
    conteggio = B.statistiche(bio)
    principali = list(conteggio.items())[:6]
    note.append("biomi: " + ", ".join(f"{k} {v}" for k, v in principali)
                + (f" (+{len(conteggio) - 6} altri)" if len(conteggio) > 6 else ""))

    a = Analisi(cls=cls, h=h, livello=livello, lava=lava, colata=colata,
                biomi=bio, note=note)
    _scrivi_cache(op, a)
    avanza(1.0, "analisi completata")
    return a


def _leggi_cache(op: Opzioni) -> Analisi | None:
    if not os.path.exists(op.cache):
        return None
    # Una cache scritta da una versione precedente non ha le chiavi nuove.
    # Va ignorata, non fatta esplodere: il costo di ricalcolare l'analisi
    # e' qualche secondo, quello di un KeyError e' una generazione persa.
    try:
        d = np.load(op.cache, allow_pickle=False)
        firma = tuple(str(x) for x in d["firma"])
        if firma != tuple(str(x) for x in op.firma_analisi()):
            return None
        # `biomi` puo' mancare: un array vuoto e' il modo di scrivere None
        # in un npz senza pickle, che qui e' disattivato di proposito.
        bio = d["biomi"]
        return Analisi(cls=d["cls"], h=d["h"], livello=d["livello"],
                       lava=d["lava"], colata=d["colata"],
                       biomi=bio if bio.size else None,
                       note=[str(x) for x in d["note"]])
    except (KeyError, ValueError, OSError):
        return None


def _scrivi_cache(op: Opzioni, a: Analisi) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(op.cache)) or ".", exist_ok=True)
    np.savez_compressed(
        op.cache, cls=a.cls, h=a.h, livello=a.livello, lava=a.lava,
        colata=a.colata,
        biomi=a.biomi if a.biomi is not None else np.zeros(0, np.uint8),
        note=np.array(a.note, dtype=object).astype("U400"),
        firma=np.array([str(x) for x in op.firma_analisi()], dtype="U400"))


# --------------------------------------------------------------------------
# Pianificazione (tutto quello che modifica il terreno)
# --------------------------------------------------------------------------

@dataclass
class Piano:
    h: np.ndarray                    # terreno DOPO spianamenti e sedi stradali
    edifici: list
    indice_edifici: dict
    tipo_strada: np.ndarray | None
    quota_strada: np.ndarray | None
    alberi: dict | None
    indice_alberi: dict
    muro: np.ndarray | None = None   # 1 = cinta, 2 = porta, 3 = torre
    scarpata: np.ndarray | None = None   # fronte dei gradoni urbani
    citta: list = field(default_factory=list)
    abitanti: list = field(default_factory=list)
    banchi: list = field(default_factory=list)
    indice_banchi: dict = field(default_factory=dict)
    campi: np.ndarray | None = None
    poderi: list = field(default_factory=list)
    caverne: dict = field(default_factory=dict)
    indice_caverne: dict = field(default_factory=dict)
    note: list[str] = field(default_factory=list)


def pianifica(op: Opzioni, a: Analisi, avanza: Avanzamento = _nulla) -> Piano:
    """Insediamenti, strade e vegetazione, in quest'ordine obbligato.

    Tre fasi modificano il terreno - gli insediamenti spianano i lotti, le
    strade spianano la sede, la vegetazione si semina su quel che resta - e
    spianare mentre si scrivono i chunk e' impossibile: il chunk accanto e'
    gia' chiuso. Quindi tutto qui, prima di scrivere una sola colonna.
    """
    note: list[str] = []
    h = a.h
    avanza(0.05, "pianta urbana")
    siti = (I.scegli_siti(a.cls, h, livello_mare=LIVELLO_MARE,
                          celle_per_villaggio=int(45_000 / max(op.villaggi, 1e-3)),
                          seed=31) if op.villaggi > 0 else [])
    edifici, h, vie, muro, citta, banchi, urbano = CT.pianifica(
        a.cls, h, a.livello, siti, livello_mare=LIVELLO_MARE, seed=31)
    indice_ed = I.indice_per_chunk(edifici) if edifici else {}
    # Un abitante per bottega: il villager sta nella sua stanza, accanto al
    # banco che rivendichera' come posto di lavoro.
    abitanti = [EN.Abitante(x=px, y=py + 0.0, z=pz, mestiere=e.mestiere,
                            seme=e.seme)
                for e in edifici if e.mestiere
                for px, py, pz in (E.punto_lavoro(e),)]
    # e uno per banco, davanti al suo bancone
    MERCE_MESTIERE = {"frutta": "fruttivendolo", "carne": "macellaio",
                      "pesce": "pescatore", "verdura": "fruttivendolo"}
    abitanti += [EN.Abitante(x=b.x + 1.5, y=float(b.base), z=b.z + 1.5,
                             mestiere=MERCE_MESTIERE.get(b.merce, "fruttivendolo"),
                             seme=b.seme)
                 for b in banchi]
    if edifici:
        st = CT.statistiche(edifici, vie, muro, citta, banchi)
        note.append(
            f"insediamenti: {st['edifici']} edifici in {st['insediamenti']} "
            f"centri ({st['citta']} con le mura); vie {st['assi']} di asse, "
            f"{st['secondarie']} secondarie, {st['vicoli']} di vicolo; "
            f"{st['mura']} celle di cinta, {st['porte']} di porta, "
            f"{st['banchi']} banchi di mercato")
        mest = CT.conteggio_mestieri(edifici)
        if mest:
            note.append("botteghe: " + ", ".join(f"{k} {v}" for k, v in mest.items()))

    # I CAMPI vanno qui: dopo le case e le mura (devono sapere cosa evitare)
    # e prima delle strade, che cosi' possono costeggiarli invece di
    # attraversarli, e prima della semina, che non deve piantare un bosco in
    # mezzo al grano.
    campi = None
    poderi: list = []
    if op.villaggi > 0 and citta:
        avanza(0.25, "campi e frutteti")
        preso = np.zeros(a.cls.shape, bool)
        preso |= vie > 0
        preso |= muro > 0
        for e in edifici:
            x0, z0, x1, z1 = e.ingombro_tetto()
            preso[max(0, z0 - 1):z1 + 1, max(0, x0 - 1):x1 + 1] = True
        for b in banchi:
            preso[max(0, b.z - 2):b.z1 + 2, max(0, b.x - 2):b.x1 + 2] = True
        poderi, h, campi, frutta = AG.pianifica(
            a.cls, h, preso, citta, livello_mare=LIVELLO_MARE,
            scala=op.villaggi, seed=31)
        if poderi:
            st = AG.statistiche(poderi, campi, frutta)
            note.append(
                f"campagna: {st['poderi']} poderi e {st['frutteti']} frutteti, "
                f"{st['arato']} celle arate, {st['canali']} di canale, "
                f"{st['alberi_da_frutto']} alberi da frutto, "
                f"{st['bacche']} cespugli di bacche")

    tipo_strada = quota_strada = None
    if op.strade and edifici:
        avanza(0.35, "rete stradale e ponti")
        ancore = [c.porte for c in citta]
        tipo_strada, quota_strada, h = ST.pianifica(
            a.cls, h, a.livello, siti, edifici, livello_mare=LIVELLO_MARE,
            vie=vie, ancore=ancore, campi=campi, seed=31)
        st = ST.statistiche(tipo_strada)
        note.append(f"strade: {st['strada']} celle di sede, "
                    f"{st['lastricato']} lastricate, {st['ponte']} di "
                    f"ponte ({st['parapetto']} di parapetto)")

    # ULTIMA REGOLA SUL TERRENO. Citta', campi e strade hanno tutti il
    # permesso di muovere le quote; nessuno di loro sa dove passa l'acqua. Le
    # sponde si ricontrollano qui, quando nessuno le tocchera' piu'.
    fiume = a.cls == M.FIUME
    if fiume.any():
        prima = h.copy()
        h = R.puntella(h, fiume, a.livello, livello_mare=LIVELLO_MARE)
        quante = int((h != prima).sum())
        if quante:
            note.append(f"sponde ripuntellate: {quante} celle "
                        f"(fiumi rimasti sospesi dopo citta' e strade)")

    # I fronti dei gradoni si leggono ADESSO, quando nessuno muovera' piu' il
    # terreno: sono una proprieta' delle quote finali, non una decisione.
    scarpata = None
    if urbano.any():
        scarpata = CT.scarpate(h, urbano)
        if scarpata.any():
            note.append(f"terrazze: {int(scarpata.sum())} celle di fronte "
                        f"(muri di sostegno)")

    avanza(0.62, "caverne")
    caverne = (SS.scava_caverne(h, livello_mare=LIVELLO_MARE, seed=19)
               if op.caverne > 0 else {"x": np.zeros(0, np.int32)})
    if caverne["x"].size:
        st = SS.statistiche(caverne)
        note.append(f"sottosuolo: {st['sfere']} celle di cunicolo fra "
                    f"y={st['quota_min']} e y={st['quota_max']}, "
                    f"raggio medio {st['raggio_medio']:.1f}")

    avanza(0.75, "vegetazione")
    alberi = (V.semina(a.cls, h, livello_mare=LIVELLO_MARE,
                       scala_densita=op.alberi, seed=21)
              if op.alberi > 0 else None)
    if alberi is not None and campi is not None and campi.any():
        # niente bosco spontaneo dentro un podere
        dentro = binary_dilation_campi(campi)
        tieni = ~dentro[alberi["z"], alberi["x"]]
        alberi = {k: v[tieni] for k, v in alberi.items()}
    if alberi is not None and muro is not None and muro.any():
        # niente bosco dentro le mura
        from scipy.ndimage import binary_fill_holes as _riempi
        urbano = _riempi(muro > 0)
        tieni = ~urbano[alberi["z"], alberi["x"]]
        alberi = {k: v[tieni] for k, v in alberi.items()}
    if alberi is not None and tipo_strada is not None:
        # niente alberi in mezzo alla carreggiata
        from scipy.ndimage import binary_dilation as _dil
        occupato = _dil(tipo_strada > 0, iterations=1)
        tieni = ~occupato[alberi["z"], alberi["x"]]
        alberi = {k: v[tieni] for k, v in alberi.items()}
    if alberi is not None and edifici:
        # via gli alberi dentro le case: un abete in salotto non ci va
        tieni = np.ones(alberi["x"].size, bool)
        for ed in edifici:
            x0, z0, x1, z1 = E.ingombro(ed, margine=1)
            tieni &= ~((alberi["x"] >= x0) & (alberi["x"] < x1)
                       & (alberi["z"] >= z0) & (alberi["z"] < z1))
        if not tieni.all():
            alberi = {k: v[tieni] for k, v in alberi.items()}
    if alberi is not None and poderi:
        # gli alberi da frutto entrano nello stesso elenco: li disegna il
        # generatore di alberi, che non sa e non deve sapere che sono un
        # frutteto
        alberi = {k: np.concatenate([alberi[k], frutta[k]]) for k in alberi}
    elif poderi and frutta["x"].size:
        alberi = frutta
    if alberi is not None:
        note.append("vegetazione: " + (", ".join(f"{k} {v}" for k, v in
                                                 V.conteggio(alberi).items())
                                       or "nessuna"))
    indice = V.indice_per_chunk(alberi, op.lato) if alberi else {}
    avanza(1.0, "pianificazione completata")
    return Piano(h=h, edifici=edifici, indice_edifici=indice_ed,
                 tipo_strada=tipo_strada, quota_strada=quota_strada,
                 alberi=alberi, indice_alberi=indice, muro=muro,
                 scarpata=scarpata,
                 citta=citta, abitanti=abitanti, banchi=banchi,
                 campi=campi, poderi=poderi, caverne=caverne,
                 indice_caverne=SS.indice_per_chunk(caverne) if caverne["x"].size else {},
                 indice_banchi=E.indice_banchi(banchi) if banchi else {},
                 note=note)


# --------------------------------------------------------------------------
# Scrittura
# --------------------------------------------------------------------------

def fetta(a: np.ndarray, sx: int, sz: int) -> np.ndarray:
    """Ritaglia un chunk da una mappa e la gira nell'ordine dei blocchi.

    QUI STAVA UN BUG, ed e' sopravvissuto a lungo. Le mappe - classi, quote,
    rugosita' - sono array di immagine, indicizzati [riga, colonna] cioe'
    [z, x]. Gli array di blocchi di un chunk sono (16, altezza, 16) con il
    primo asse su x, cioe' [x, y, z]. Ritagliare con `a[sx:sx+16, sz:sz+16]`
    sembra giusto e invece scambia i due assi: il mondo viene generato
    TRASPOSTO rispetto alla mappa.

    Non si vedeva. Un continente ribaltato lungo la diagonale sembra comunque
    un continente, e il confronto a occhio con la mappa non lo tradisce. Il
    test dello spike che confrontava 65.536 colonne una per una lo mancava
    perche' usava la stessa convenzione sbagliata su entrambi i lati.
    E' saltato fuori solo quando la vegetazione, che legge correttamente
    [z, x], ha piantato alberi in mezzo al mare: due sistemi di coordinate in
    disaccordo si vedono, uno sbagliato da solo no.
    """
    return a[sz:sz + 16, sx:sx + 16].T


def blocchi_chunk(scrittore, h_c, cls_c, liv_c=None, tipo_c=None,
                  quota_c=None, lava_c=None, colata_c=None, muro_c=None,
                  campi_c=None, ox=0, oz=0, y0=-64, y1=220,
                  roccia=None, scarto_c=None, nuda_c=None, neve_c=None,
                  manto_c=None, scarpata_c=None):
    """Colonne di un chunk: (16, H, 16) di id di palette, vettoriale."""
    hh = h_c.astype(np.int32)[:, None, :]
    ys = np.arange(y0, y1, dtype=np.int32)[None, :, None]

    out = np.full((16, y1 - y0, 16), scrittore.id_aria, dtype=np.uint32)
    solido = ys < hh
    out[solido] = scrittore.blocco("stone")
    out[:, 0, :] = scrittore.blocco("bedrock")

    # --- strati di roccia -------------------------------------------------
    # PRIMA della superficie: gli strati stanno dentro la montagna, l'erba
    # sopra. L'ordine inverso darebbe banchi di tufo in mezzo a un prato.
    if roccia is not None:
        scarto = (np.zeros((16, 16), np.int32) if scarto_c is None
                  else np.asarray(scarto_c).astype(np.int32))[:, None, :]
        SG.applica(out, roccia, ys, solido, scarto)

    cima = ys == hh - 1
    sotto = solido & (ys >= hh - 4) & ~cima
    # Dove il pendio e' una parete non cresce niente: la superficie resta lo
    # strato di roccia che c'e' sotto. E' cosi' che si ottengono le pareti,
    # senza generarle: basta non coprirle di erba.
    nuda = (np.zeros((16, 16), bool) if nuda_c is None
            else np.asarray(nuda_c).astype(bool))[:, None, :]
    for k, (sup, sub) in SUPERFICIE.items():
        m = (cls_c == k)[:, None, :] & ~nuda
        out[cima & m] = scrittore.blocco(sup)
        out[sotto & m] = scrittore.blocco(sub)

    # --- fronti dei gradoni -----------------------------------------------
    # Dove la citta' e' terrazzata, il salto fra un ripiano e l'altro si veste
    # di pietra: un muro di sostegno si legge come una cosa costruita, un
    # taglio di terra nuda sembra un difetto del terreno.
    if scarpata_c is not None and np.any(scarpata_c):
        sc = np.asarray(scarpata_c).astype(bool)
        sasso = scrittore.blocco("cobblestone")
        muschio = scrittore.blocco("mossy_cobblestone")
        for lx, lz in zip(*np.nonzero(sc)):
            cima_y = int(h_c[lx, lz]) - 1
            for k in range(4):
                ly = cima_y - k - y0
                if 0 <= ly < out.shape[1]:
                    out[lx, ly, lz] = (muschio if (lx + lz + k) % 5 == 0
                                       else sasso)

    # --- neve -------------------------------------------------------------
    # Non una quota netta ma una fascia: prima il manto sottile a chiazze,
    # poi il blocco pieno. Una linea delle nevi disegnata col righello si
    # riconosce da chilometri.
    if roccia is not None and neve_c is not None:
        nv = np.asarray(neve_c).astype(np.uint8)
        piena = (nv == 2)[:, None, :]
        out[cima & piena] = roccia.neve
        manto = np.nonzero(nv == 1)
        for lx, lz in zip(*manto):
            ly = int(h_c[lx, lz]) - y0
            if 0 <= ly < out.shape[1] and out[lx, ly, lz] == scrittore.id_aria:
                out[lx, ly, lz] = roccia.manto

    # Il livello dell'acqua NON e' globale: un fiume scorre in quota, e
    # riempirlo fino al livello del mare lo trasformerebbe in un canyon
    # asciutto o in un braccio di mare.
    liv = (np.full((16, 16), LIVELLO_MARE, np.int32) if liv_c is None
           else np.asarray(liv_c).astype(np.int32))[:, None, :]
    liquido = (~solido) & (ys <= liv)
    out[liquido] = scrittore.blocco("water")
    if lava_c is not None and np.any(lava_c):
        # la lava riempie il cratere fino al SUO pelo, non a quello del mare
        m = np.asarray(lava_c)[:, None, :]
        out[liquido & m] = scrittore.blocco("lava", falling="false",
                                            flowing="false", level="0")
    # --- manto del cono ---------------------------------------------------
    # Prima il cono era basalto e basta, dalla base all'orlo. Un vulcano vero
    # e' fatto di colate sovrapposte di eta' diverse, e si vede.
    if manto_c is not None:
        mc = np.asarray(manto_c)
        vulcanico = np.isin(cls_c, (M.VULCANO, M.CRATERE))
        for k, (sup, sub) in enumerate((("basalt", "blackstone"),
                                        ("smooth_basalt", "blackstone"),
                                        ("blackstone", "blackstone"),
                                        ("tuff", "blackstone"))):
            mm = (vulcanico & (mc == k))[:, None, :]
            out[cima & mm] = scrittore.blocco(sup, axis="y") if "basalt" in sup \
                else scrittore.blocco(sup)
            out[sotto & mm] = scrittore.blocco(sub)

    if colata_c is not None and np.any(colata_c):
        cima_i = (hh - 1 - y0)
        magma = scrittore.blocco("magma_block")
        basalto = scrittore.blocco("basalt", axis="y")
        lava_viva = scrittore.blocco("lava", falling="false", flowing="false",
                                     level="0")
        cc = np.asarray(colata_c, dtype=np.float32)
        for lx in range(16):
            for lz in range(16):
                t = float(cc[lx, lz])
                if t <= 0:
                    continue
                yy = int(cima_i[lx, 0, lz])
                if not (0 <= yy < out.shape[1]):
                    continue
                if t <= U.SOGLIA_LIQUIDA:
                    # lava viva: sta in un canale, non spalmata sul pendio.
                    # Il blocco sotto resta magma cosi' la colata non cola in
                    # una caverna e non sparisce dentro la montagna.
                    out[lx, yy, lz] = lava_viva
                    if yy > 0:
                        out[lx, yy - 1, lz] = magma
                elif t <= U.SOGLIA_CROSTA:
                    out[lx, yy, lz] = magma
                else:
                    out[lx, yy, lz] = basalto

    if tipo_c is not None:
        _posa_strada(out, scrittore, np.asarray(tipo_c),
                     np.asarray(quota_c).astype(np.int32),
                     np.asarray(h_c).astype(np.int32), ox, oz, y0)
    if campi_c is not None and np.any(campi_c):
        _posa_campi(out, scrittore, np.asarray(campi_c),
                    np.asarray(h_c).astype(np.int32), y0)
    if muro_c is not None and np.any(muro_c):
        _posa_mura(out, scrittore, np.asarray(muro_c),
                   np.asarray(h_c).astype(np.int32), y0)
    return out


def _pendenza(h: np.ndarray) -> np.ndarray:
    """Blocchi di quota per cella di pianta. E' la misura che distingue un
    prato da una parete, e serve a tre cose diverse: dove non ditherare, dove
    non far crescere l'erba, dove non far attaccare la neve."""
    gz, gx = np.gradient(h.astype(np.float32))
    return np.hypot(gz, gx)


def binary_dilation_campi(campi):
    from scipy.ndimage import binary_dilation as _d
    return _d(campi > 0, iterations=1)


def _posa_campi(out, s, campi, h_c, y0):
    """Terra arata, colture, canali, recinti.

    La superficie del terreno sta a h-1, quindi l'arato sostituisce QUELLO e
    la coltura cresce a h. Sbagliare di uno qui significa grano che spunta
    dentro la terra o terra arata sospesa.
    """
    H = out.shape[1]
    aria = s.id_aria
    arato = s.blocco("farmland", moisture="7")
    acqua = s.blocco("water", level="0")
    sentiero = s.blocco("grass_path")
    palo = s.blocco("fence", material="oak", north="false", south="false",
                    east="false", west="false")
    semi = [s.blocco(n, age="7") for n in AG.COLTURE]

    for lx in range(16):
        for lz in range(16):
            c = int(campi[lx, lz])
            if c == AG.NIENTE:
                continue
            y = int(h_c[lx, lz]) - 1 - y0
            if not (0 <= y < H - 2):
                continue
            if c >= AG.ARATO:
                out[lx, y, lz] = arato
                out[lx, y + 1, lz] = semi[(c - AG.ARATO) % len(semi)]
            elif c == AG.CANALE:
                out[lx, y, lz] = acqua
                out[lx, y + 1, lz] = aria
            elif c == AG.SENTIERO:
                out[lx, y, lz] = sentiero
                out[lx, y + 1, lz] = aria
            elif c == AG.RECINTO:
                out[lx, y + 1, lz] = palo
            elif c == AG.PRATO:
                out[lx, y + 1, lz] = aria


ALTEZZA_MURA = 7
ALTEZZA_TORRE = 11


def _posa_mura(out, s, muro, h_c, y0):
    """Cinta muraria, torri e varchi.

    Il muro non ha una quota propria: segue il terreno, come un muro vero.
    Merlatura a denti alterni, e sopra la porta un architrave, cosi' il varco
    si legge come una porta e non come un pezzo di muro che manca.

    Due correzioni nate da uno screenshot in cui le mura sembravano macerie
    sparse: e' cresciuta da cinque a sette blocchi - sotto i cinque, con le
    case a due piani accanto, una cinta non si distingue da un muretto - e
    soprattutto adesso ha le FONDAMENTA. Prima partiva dalla quota del
    terreno e saliva; dove il terreno accanto scendeva, sotto il muro
    restava il vuoto e si vedevano blocchi appesi.
    """
    H = out.shape[1]
    pietra = s.blocco("stone_bricks", variant="normal")
    mattone = s.blocco("stone_bricks", variant="mossy")
    incrinata = s.blocco("stone_bricks", variant="cracked")
    aria = s.id_aria
    for lx in range(16):
        for lz in range(16):
            m = int(muro[lx, lz])
            if m == 0:
                continue
            base = int(h_c[lx, lz]) - y0
            if m in (1, 3):
                # fondamenta: si scende finche' non si trova del pieno
                for giu in range(1, 9):
                    k = base - giu
                    if k < 0 or out[lx, k, lz] != aria:
                        break
                    out[lx, k, lz] = pietra
                alta = ALTEZZA_TORRE if m == 3 else ALTEZZA_MURA
                cima = base + alta
                for k in range(base, min(cima, H)):
                    if (lx + lz) % 7 == 0:
                        out[lx, k, lz] = mattone
                    elif (lx * 3 + lz) % 11 == 0:
                        out[lx, k, lz] = incrinata
                    else:
                        out[lx, k, lz] = pietra
                # merli: un dente sì e uno no
                if (lx + lz) % 2 == 0 and 0 <= cima < H:
                    out[lx, cima, lz] = pietra
            else:
                # varco: si sgombera il passaggio e si mette l'architrave
                for k in range(base, min(base + 5, H)):
                    out[lx, k, lz] = aria
                for k in range(base + 5, min(base + ALTEZZA_MURA + 1, H)):
                    if 0 <= k < H:
                        out[lx, k, lz] = pietra


def _posa_strada(out, s, tipo, quota, h_c, ox, oz, y0):
    """Sede stradale, impalcati, pilastri e parapetti.

    Il ponte non e' un oggetto piazzato: e' il tratto di strada che si trova
    sull'acqua o in aria e deve comunque stare alla quota della sede. Da qui
    discende tutto - l'impalcato e' la sede, i pilastri scendono fino a dove
    c'e' terreno, il parapetto sta sui bordi.
    """
    sentiero = s.blocco("grass_path")
    lastrico = s.blocco("cobblestone")
    assi = s.blocco("planks", material="oak")
    pilastro = s.blocco("log", axis="y", material="oak", stripped="true")
    staccionata = s.blocco("fence", material="oak", north="false", south="false",
                           east="false", west="false")
    aria = s.id_aria
    H = out.shape[1]

    for lx in range(16):
        for lz in range(16):
            t = int(tipo[lx, lz])
            if t == 0:
                continue
            y = int(quota[lx, lz]) - 1 - y0
            if not (0 <= y < H):
                continue
            if t in (ST.STRADA, ST.LASTRICATO):
                out[lx, y, lz] = sentiero if t == ST.STRADA else lastrico
                out[lx, y + 1:min(y + 4, H), lz] = aria
                continue

            out[lx, y, lz] = assi                    # impalcato
            out[lx, y + 1:min(y + 5, H), lz] = aria  # luce sopra
            if t == ST.PARAPETTO and y + 1 < H:
                out[lx, y + 1, lz] = staccionata

            # pilastri a intervalli regolari, fino al terreno o al fondale
            gx, gz = ox + lx, oz + lz
            if gx % 5 == 0 and gz % 5 == 0:
                fondo = max(0, int(h_c[lx, lz]) - 1 - y0)
                if fondo < y:
                    out[lx, fondo:y, lz] = pilastro


def scrivi(op: Opzioni, a: Analisi, piano: Piano,
           avanza: Avanzamento = _nulla,
           da: int = 0, quanti: int = 0,
           ferma: Callable[[], bool] | None = None) -> dict:
    """Scrive i chunk. `da`/`quanti` servono alla CLI a lavorare a lotti.

    `ferma` e' interrogato a ogni chunk: la finestra deve poter annullare una
    generazione lunga senza chiudere il programma.
    """
    # amulet si importa QUI e non in cima: l'analisi, la pianificazione e i
    # loro test non ne hanno bisogno, e chi vuole solo guardare una mappa non
    # deve installare mezza libreria di Minecraft.
    from .livello_dat import ImpostazioniMondo, verifica
    from .mondo import ScrittoreMondo

    imp = ImpostazioniMondo(nome=op.nome, modalita=1, spawn=(0, 150, 0))
    os.makedirs(os.path.dirname(os.path.abspath(op.uscita)) or ".", exist_ok=True)
    primo = da == 0
    t0 = time.time()
    scritti = 0
    interrotto = False

    with ScrittoreMondo(op.uscita, imp, lato_blocchi=op.lato, crea=primo) as m:
        tav_ss = SS.Tavolozza(m)
        roccia = SG.TavolozzaRoccia(m, seed=op.seed)
        # Gli strati fanno parte della roccia, quindi il sottosuolo puo'
        # scavarli e sostituirli come farebbe con la pietra liscia.
        tav_ss.aggiungi_rocce(roccia.ids)
        pend = _pendenza(piano.h)
        clima = B.freddo(a.cls, piano.h, livello_mare=LIVELLO_MARE, seed=op.seed)
        nuda_m, neve_m = SG.superficie_montana(piano.h, pend, seed=op.seed,
                                               freddo=clima)
        # La parete nuda vale sulla terra ferma: sott'acqua non si vede, sul
        # basalto di un vulcano sarebbe un banco di granito in mezzo a una
        # colata, e sulla sabbia una scogliera di pietra in mezzo alle dune.
        nuda_m &= ~np.isin(a.cls, M.MARINO + (M.FIUME, M.VULCANO, M.CRATERE,
                                              M.DESERTO, M.SPIAGGIA))
        neve_m = np.where(np.isin(a.cls, M.MARINO + (M.FIUME, M.CRATERE)),
                          0, neve_m).astype(np.uint8)
        scarto_m = SG.piega(max(piano.h.shape),
                            seed=op.seed)[:piano.h.shape[0], :piano.h.shape[1]]
        manto_m = U.manto(a.cls, seed=op.seed)
        tav = V.Tavolozza(m) if piano.alberi is not None else None
        tav_ed = E.TavolozzaEdilizia(m) if piano.edifici else None

        # --- case da template ------------------------------------------
        # La scelta si fa QUI, una volta per edificio, e non dentro il ciclo
        # dei chunk: una casa sta a cavallo di quattro chunk, e sceglierne il
        # modello quattro volte con quattro tirate di dado significa
        # costruirne quattro diverse una dentro l'altra.
        modelli = TM.carica_cartella(op.templates) if op.templates else []
        cat = TM.Catalogo(modelli, m) if modelli else None
        scelte: dict[int, tuple[int, int]] = {}
        if cat is not None and piano.edifici:
            rng_t = np.random.default_rng(op.seed * 7717 + 3)
            for i, e in enumerate(piano.edifici):
                if e.palafitta:
                    continue          # la palafitta e' un modificatore, non una casa
                s = TM.scegli(modelli, e.larghezza + 2 * e.gronda,
                              e.profondita + 2 * e.gronda, e.porta, rng_t)
                if s is not None:
                    scelte[i] = s
            st = TM.statistiche(modelli)
            avanza(0.0, f"{st['modelli']} template, {len(scelte)} case su "
                        f"{len(piano.edifici)}")

        # gli id del palette dei biomi si risolvono una volta sola, ma DOPO
        # l'apertura del livello: il palette appartiene al mondo, non a noi
        id_bioma = (np.array([m.bioma(n) for n in B.BIOMI], np.uint32)
                    if a.biomi is not None else None)
        tutti = list(m.chunk_coords())
        fino = len(tutti) if quanti <= 0 else min(da + quanti, len(tutti))
        if da >= len(tutti):
            return {"chunk": 0, "totale": len(tutti), "restano": 0,
                    "secondi": 0.0, "interrotto": False}
        meta = op.lato // 2
        n = max(1, fino - da)
        for i, (cx, cz) in enumerate(tutti[da:fino]):
            if ferma is not None and ferma():
                interrotto = True
                break
            ox, oz = m.origine_chunk(cx, cz)
            sx, sz = ox + meta, oz + meta
            blocchi = blocchi_chunk(
                m, fetta(piano.h, sx, sz), fetta(a.cls, sx, sz),
                fetta(a.livello, sx, sz),
                fetta(piano.tipo_strada, sx, sz) if piano.tipo_strada is not None else None,
                fetta(piano.quota_strada, sx, sz) if piano.quota_strada is not None else None,
                fetta(a.lava, sx, sz), fetta(a.colata, sx, sz),
                muro_c=fetta(piano.muro, sx, sz) if piano.muro is not None else None,
                campi_c=fetta(piano.campi, sx, sz) if piano.campi is not None else None,
                ox=sx, oz=sz,
                roccia=roccia, scarto_c=fetta(scarto_m, sx, sz),
                nuda_c=fetta(nuda_m, sx, sz), neve_c=fetta(neve_m, sx, sz),
                manto_c=fetta(manto_m, sx, sz),
                scarpata_c=(fetta(piano.scarpata, sx, sz)
                            if piano.scarpata is not None else None))
            # Il sottosuolo PRIMA di tutto il resto: le caverne scavano, e
            # devono scavare nella roccia, non dentro una casa o sotto una
            # strada gia' posata.
            SS.posa(blocchi, tav_ss, fetta(piano.h, sx, sz), sx, sz, -64,
                    piano.caverne,
                    piano.indice_caverne.get((sx // 16, sz // 16), ()),
                    seed=19)
            if tav is not None:
                # Si disegnano anche gli alberi dei chunk vicini: uno piantato
                # a un blocco dal bordo sporge qui, e filtrando per centro
                # invece che per ingombro si otterrebbero alberi tagliati a
                # meta' lungo ogni linea di chunk.
                quali = V.alberi_vicini(piano.indice_alberi, sx // 16, sz // 16)
                if quali:
                    V.disegna(blocchi, -64, sx, sz, piano.alberi, quali, tav)
            if tav_ed is not None:
                # gli edifici dopo gli alberi: una casa vince su un ramo
                for k in piano.indice_edifici.get((sx // 16, sz // 16), ()):
                    ed = piano.edifici[k]
                    if k in scelte:
                        mi, quarti = scelte[k]
                        ix, iz = modelli[mi].ingombro(quarti)
                        px = ed.x + (ed.larghezza - ix) // 2
                        pz = ed.z + (ed.profondita - iz) // 2
                        TM.fondazione(blocchi, -64, sx, sz, cat, mi, quarti,
                                      px, pz, ed.base,
                                      tav_ed.blocco[(ed.stile, "basamento")],
                                      tav_ed.aria)
                        TM.costruisci(blocchi, -64, sx, sz, cat, mi, quarti,
                                      px, pz, ed.base)
                        if ed.mestiere:
                            E.posto_di_lavoro(blocchi, -64, sx, sz, ed, tav_ed)
                    else:
                        E.costruisci(blocchi, -64, sx, sz, ed, tav_ed)
                for k in piano.indice_banchi.get((sx // 16, sz // 16), ()):
                    E.costruisci_banco(blocchi, -64, sx, sz, piano.banchi[k],
                                       tav_ed)
            bio_c = (id_bioma[fetta(a.biomi, sx, sz)]
                     if id_bioma is not None else None)
            m.scrivi_chunk(cx, cz, blocchi, biomi=bio_c)
            scritti += 1
            if i % 25 == 0 or i == n - 1:
                avanza((i + 1) / n, f"chunk {da + i + 1} di {len(tutti)}")

    restano = len(tutti) - (da + scritti)
    fuori = {"chunk": scritti, "totale": len(tutti), "restano": restano,
             "secondi": time.time() - t0, "interrotto": interrotto}

    # Gli abitanti si scrivono ALLA FINE e da soli: `ScrittoreMondo` cancella
    # la cartella `entities/` a ogni chiusura, perche' amulet la scrive rotta.
    # Quella che mettiamo noi deve arrivare dopo l'ultima chiusura.
    if restano == 0 and not interrotto and piano.abitanti:
        from dataclasses import replace as _replace

        from .livello_dat import ImpostazioniMondo as _Imp
        # ATTENZIONE alle coordinate. Tutta la pianificazione lavora in
        # coordinate di MAPPA (0..lato); il mondo e' centrato sull'origine,
        # quindi la coordinata di gioco e' la mappa meno mezzo lato. I blocchi
        # lo sanno perche' `scrivi` passa `ox=sx`; le entita' no, e la prima
        # versione le ha messe in regioni che nel mondo non esistono -
        # r.0.0/r.1.1 invece di r.-1.-1/r.0.0. Nessun errore, nessun avviso:
        # abitanti scritti in un angolo di mondo vuoto.
        meta = op.lato // 2
        in_gioco = [_replace(a, x=a.x - meta, z=a.z - meta)
                    for a in piano.abitanti]
        st = EN.scrivi_regioni(op.uscita, in_gioco, _Imp().data_version)
        fuori["abitanti"] = st["abitanti"]

    if restano == 0 and not interrotto:
        fuori["level_dat"] = "valido" if not verifica(
            os.path.join(op.uscita, "level.dat")) else "PROBLEMI"
    return fuori


# --------------------------------------------------------------------------
# Tutto insieme
# --------------------------------------------------------------------------

def genera(op: Opzioni, avanza: Avanzamento = _nulla,
           ferma: Callable[[], bool] | None = None) -> dict:
    """Analisi + pianificazione + scrittura, con un solo avanzamento 0..1.

    I pesi delle tre fasi sono misurati su Arda 832x832: l'analisi e la
    pianificazione insieme valgono circa un terzo del tempo, la scrittura il
    resto. Meglio una barra che mente poco che tre barre in fila.
    """
    def parziale(da: float, a_: float):
        return lambda f, t: avanza(da + (a_ - da) * max(0.0, min(1.0, f)), t)

    a = analizza(op, parziale(0.0, 0.20))
    piano = pianifica(op, a, parziale(0.20, 0.33))
    st = scrivi(op, a, piano, parziale(0.33, 1.0), ferma=ferma)
    st["note"] = a.note + piano.note
    st["quote"] = (int(piano.h.min()), int(piano.h.max()))
    st["terra_emersa"] = float((piano.h > LIVELLO_MARE).mean())
    return st


def con_uscita(op: Opzioni, cartella: str, nome: str) -> Opzioni:
    """Copia delle opzioni che punta a un altro mondo."""
    return replace(op, uscita=cartella, nome=nome)
