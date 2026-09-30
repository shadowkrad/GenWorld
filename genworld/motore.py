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
from . import arredi as AR
from . import avamposti as AP
from . import batimetria as BAT
from . import biomi as B
from . import bauli as BA
from . import citta as CT
from . import edifici as E
from . import entita as EN
from . import fauna as FA
from . import fiumi as R
from . import insediamenti as I
from . import isolate as IS
from . import laghi as LG
from . import mappa as M
from . import miniere as MI
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

# Distanza minima, in celle, fra il vulcano (cono, cratere, lago di lava e
# colate) e le costruzioni sparse a caso sulla mappa: ingressi di miniera,
# accampamenti, cimiteri, portali. Senza, niente li tratteneva dal nascere
# sul fianco del cono, e in gioco un portale o un pozzo restavano appesi a
# una parete di basalto. Le case non serve: stanno gia' solo su terreno
# abitabile e poco ripido (`insediamenti.scegli_siti`).
DISTANZA_MIN_VULCANO = 12

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
    # Bacini chiusi senza sbocco: 0 lascia quelle conche terreno asciutto
    # invece di riempirle. Cambia il terreno (vedi `firma_analisi`), quindi
    # sta con fiumi e vulcani, non con caverne/miniere che non lo toccano.
    laghi: float = 1.0
    # Moltiplicatore di densita' dei condotti artificiali (pozzi, gallerie,
    # binari, filoni lungo il percorso). 0 li spegne del tutto. Non cambia
    # il terreno, quindi non entra in `firma_analisi` - stessa idea di
    # `caverne`.
    miniere: float = 1.0
    # Moltiplicatore di densita' di accampamenti di nemici e cimiteri (vedi
    # `avamposti.py`). 0 spegne l'uno o l'altro. Non toccano il terreno
    # (solo cio' che ci sta sopra), quindi fuori da `firma_analisi` come
    # `caverne` e `miniere`.
    accampamenti: float = 1.0
    cimiteri: float = 1.0
    # Stessa idea per il portale (vedi `avamposti.py`): a differenza di
    # accampamenti/cimiteri pero' non ha un ripiego disegnato da codice, e
    # senza un template configurato (`templates_portale`) non ne compare
    # nessuno qualunque sia questo valore - vedi `avamposti.pianifica`.
    portali: float = 1.0
    # Moltiplicatore di densita' degli arredi urbani (lampioni, giardini,
    # recinti, bazar sui lotti senza casa: vedi `arredi.py`). 0 li spegne e
    # lascia i lotti vuoti. Non tocca il terreno, quindi fuori da
    # `firma_analisi`.
    arredi: float = 1.0
    # Moltiplicatore di densita' delle case isolate (i template troppo alti per
    # un villaggio, sparsi fuori dai paesi: vedi `isolate.py`). 0 le spegne.
    isolate: float = 1.0
    # Moltiplicatore di densita' della fauna, selvatica e da cortile. 0 la
    # spegne del tutto (restano solo gli abitanti dei villaggi).
    fauna: float = 1.0
    # Cartelle dei `.nbt` di blocco struttura da usare come case, separate da
    # ";" (una sola va benissimo). Non c'e' piu' un generatore parametrico di
    # riserva: vuota o senza cartelle valide vuol dire lotti senza casa, non
    # case disegnate dal codice. GUI e CLI la impostano sempre a una cartella
    # esistente, quindi in pratica non capita.
    templates: str = ""
    # Stessa idea, per il cimitero (vedi `avamposti.carica_cimiteri`). Vuota
    # o senza cartelle valide: il cimitero torna al ripiego disegnato da
    # codice (vedi `avamposti.py`), non un errore.
    templates_cimitero: str = ""
    # Stessa idea, per il portale (vedi `avamposti.carica_portali`). Vuota o
    # senza cartelle valide: nessun portale, non un errore - non c'e' un
    # ripiego disegnato da codice per questa struttura (vedi `avamposti.py`).
    templates_portale: str = ""
    seed: int = 11
    # Versione Java di riferimento per il mondo scritto: decide sia il
    # DataVersion in level.dat sia la traduzione dei blocchi dei template (li
    # traduce `PyMCTranslate`, vedi `template.carica`). Non tocca il terreno
    # ne' la pianificazione - stessa idea di `templates`, quindi fuori da
    # `firma_analisi`.
    versione: tuple[int, int, int] = (1, 21, 4)

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
                float(self.fiumi), int(self.vulcani), float(self.laghi),
                float(self.ritaglio), os.path.abspath(self.immagine))


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

    # La mappa puo' non essere quadrata: M.carica()/Profilo.leggi_immagine()
    # la ridimensionano mantenendo le proporzioni (il lato piu' lungo diventa
    # `op.lato`), MAI stirandola - una mappa rettangolare stirata a incastro
    # usciva visibilmente deformata, segnalato dall'utente. Il mondo resta
    # comunque quadrato (lato x lato): qui la mappa si centra dentro il
    # quadrato, e lo spazio che avanza sul lato piu' corto resta oceano
    # aperto. Fatto ORA, prima di `altimetria()`: la sua sfocatura (vedi
    # `morbidezza`) trasforma da sola il bordo in un pendio, non un dirupo -
    # rifarlo dopo (su `h` gia' calcolato) avrebbe lasciato uno scalino
    # netto proprio al bordo della mappa.
    nh, nw = cls.shape
    if (nh, nw) != (op.lato, op.lato):
        oy, ox = (op.lato - nh) // 2, (op.lato - nw) // 2
        cls_mappa, rug_mappa = cls, rug
        cls = np.full((op.lato, op.lato), M.OCEANO, np.uint8)
        cls[oy:oy + nh, ox:ox + nw] = cls_mappa
        rug = np.zeros((op.lato, op.lato), np.float32)
        rug[oy:oy + nh, ox:ox + nw] = rug_mappa
        riempimento = 100.0 - (nh * nw) / (op.lato ** 2) * 100.0
        note.append(f"mappa non quadrata ({nw}x{nh} su {op.lato}x{op.lato}): "
                    f"proporzioni mantenute, {riempimento:.1f}% del mondo e' "
                    "oceano aperto attorno alla mappa")

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
    # `marino` serve sia ai fiumi (soglia del deflusso verso il mare) sia ai
    # laghi (un bacino non e' un lago se sfocia gia' nel mare): calcolato
    # una volta sola, prima di entrambi i blocchi.
    marino = np.isin(cls, M.MARINO)
    if op.fiumi > 0:
        avanza(0.68, "reticolo idrografico")
        maschera_f, acc = R.da_terreno(h, marino, soglia=op.fiumi)
        # I vulcani si SOMMANO al terreno (vedi sopra) e i fiumi si calcolano
        # DOPO apposta, per avere torrenti sui fianchi - ma il cratere ha
        # gia' il suo lago di lava con un pelo libero tutto suo (vedi
        # `livello` sopra), e un fiume che ci scorre sopra o dentro ci mette
        # sopra un secondo pelo che confligge col primo: due acque
        # sovrapposte, un effetto innaturale segnalato dall'utente ("nei
        # vulcani li toglierei"). Stessa esclusione gia' usata per i laghi
        # qualche riga sotto (`esclusi`), qui applicata anche al fiume
        # perche' si calcola prima che quella maschera esista.
        if op.vulcani > 0 and maschera_f.any():
            maschera_f &= ~np.isin(cls, (M.VULCANO, M.CRATERE))
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

    # LAGHI. Dopo i fiumi apposta: un bacino che i fiumi hanno gia' riempito
    # (`cls == FIUME`) va escluso, altrimenti una conca attraversata da un
    # corso d'acqua verrebbe ritrovata come lago e il suo pelo, calcolato con
    # una logica diversa, entrerebbe in conflitto con quello del fiume.
    if op.laghi > 0:
        avanza(0.78, "laghi")
        esclusi = np.isin(cls, (M.FIUME, M.VULCANO, M.CRATERE))
        maschera_l, livello_l = LG.trova(h, marino, esclusi)
        if maschera_l.any():
            h = np.round(LG.scava(h.astype(np.float32), maschera_l,
                                  livello_l)).astype(np.int32)
            livello = np.where(maschera_l, livello_l, livello).astype(np.float32)
            cls = cls.copy()
            cls[maschera_l] = M.FIUME
            st = LG.statistiche(maschera_l, livello_l)
            note.append(f"laghi: {st['celle']} celle in {st['bacini']} bacini, "
                        f"quota media {st['quota_media']:.0f}")

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
    protetto: np.ndarray | None = None   # superficie costruita: qui caverne
                                          # e miniere non scavano
    caverne: dict = field(default_factory=dict)
    indice_caverne: dict = field(default_factory=dict)
    miniere: list = field(default_factory=list)
    indice_miniere: dict = field(default_factory=dict)
    avamposti: list = field(default_factory=list)
    indice_avamposti: dict = field(default_factory=dict)
    # {indice edificio: (indice modello, rotazione)}: quale casa da template va
    # su quale lotto. Si decide qui e non in `scrivi()`, perche' chi abita una
    # casa (e chi arreda un lotto rimasto vuoto) deve saperlo in pianificazione.
    scelte: dict = field(default_factory=dict)
    isolate: list = field(default_factory=list)
    indice_isolate: dict = field(default_factory=dict)
    arredi: list = field(default_factory=list)
    arredi_modelli: list = field(default_factory=list)
    indice_arredi: dict = field(default_factory=dict)
    proprietario: np.ndarray | None = None   # indice edificio per cella (-1
                                              # libera, -2 banco); vedi il
                                              # commento in `pianifica()`
    note: list[str] = field(default_factory=list)


def zona_vulcanica(a: Analisi, margine: int) -> np.ndarray:
    """Le celle del vulcano (cono, cratere, lago di lava, colate) allargate di
    `margine` celle: dove una costruzione sparsa non deve nascere. Il margine
    conta dal CENTRO della costruzione, quindi chi ha un ingombro grande deve
    sommarci il proprio raggio."""
    v = np.isin(a.cls, (M.VULCANO, M.CRATERE)) | a.lava.astype(bool) | (a.colata > 0)
    if margine <= 0 or not v.any():
        return v
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~v) <= margine


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
    # Proprietario: quale edificio (indice in `edifici`) possiede ciascuna
    # cella del suo SEDIME - il lotto vero, mai sovrapposto a quello di un
    # altro edificio (`citta.lotti()` lo garantisce, con un test dedicato).
    # NON l'ingombro del modello scelto poi: `template.assegna` sceglie solo
    # modelli che ci stanno per intero, ma il tetto sporge di una gronda oltre
    # il sedime. Serve da rete di sicurezza: se due ingombri si toccano, la
    # casa disegnata per seconda "mangerebbe" un pezzo di quella disegnata per
    # prima (il difetto segnalato come "casa a mezzo"), e `vietato_c` lo evita.
    proprietario = np.full(a.cls.shape, -1, np.int32)
    for i, e in enumerate(edifici):
        proprietario[e.z:e.z + e.profondita, e.x:e.x + e.larghezza] = i
    for b in banchi:
        proprietario[b.z:b.z1, b.x:b.x1] = -2    # -2: banco, mai di un edificio
    # Un abitante per casa, bottega o no. Prima solo le case con un mestiere
    # avevano un abitante (`if e.mestiere`), e `_mestiere()` in citta.py ne
    # assegna uno solo al 20-55% delle case per disegno ("una citta' fatta
    # di sole botteghe e' un centro commerciale, non una citta'") - quindi
    # una citta' grande, fatta per lo piu' di semplici abitazioni, restava
    # con pochissimi abitanti in giro: il difetto segnalato come "pochi
    # villager nelle citta' murate". Un villaggio piccolo ha comunque poche
    # case, quindi pochi abitanti anche cosi' - e' la citta' grande, con
    # tante case, che ne beneficia. `mestiere=""` non e' un errore: risolve
    # a professione "none" (villager senza mestiere) in `Abitante.professione`,
    # che e' esattamente cosa serve per chi vive in una casa qualunque.
    #
    # Le case da template si assegnano QUI, una volta per lotto e senza
    # ripetere lo stesso modello dentro un villaggio (`TM.assegna`). Un lotto
    # in cui non entra nessun modello libero resta senza casa, e senza
    # abitante: un villager in un lotto vuoto e' un villager in mezzo al prato.
    cartelle_t = [c.strip() for c in op.templates.split(";") if c.strip()]
    modelli_case = (TM.carica_cartelle(cartelle_t, versione=op.versione)
                    if cartelle_t and edifici else [])
    scelte = (TM.assegna(modelli_case, edifici,
                         np.random.default_rng(op.seed * 7717 + 3))
              if modelli_case else {})
    if modelli_case:
        note.append(f"case: {len(scelte)} su {len(edifici)} lotti hanno una "
                    f"casa ({len({k for k, _ in scelte.values()})} modelli "
                    f"diversi su {len(modelli_case)})")
    abitanti = [EN.Abitante(x=px, y=py + 0.0, z=pz, mestiere=e.mestiere,
                            seme=e.seme)
                for i, e in enumerate(edifici)
                if not modelli_case or i in scelte
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

    # PROTEZIONE DEL SOTTOSUOLO. Caverne e miniere si scavano dentro
    # `blocchi` (vedi `scrivi()`) DOPO che mura, strade e campi sono gia'
    # stati disegnati - il commento "il sottosuolo prima di tutto il resto"
    # descrive l'ordine voluto (le caverne devono scavare nella roccia, non
    # dentro una casa gia' posata), ma nel codice l'ordine e' l'opposto:
    # `blocchi_chunk()` disegna le strutture per prima, `SS.posa()` e
    # `MI.posa()` scavano dopo. Un cunicolo o una galleria che passa proprio
    # sotto una cinta muraria toglie la roccia su cui il muro e' stato
    # disegnato: il muro sopra resta l'immagine gia' scritta, sotto resta il
    # vuoto - "la murata che si interrompe e rimane sospesa" segnalato in
    # uno screenshot. Le caverne (vermi a passeggio su tutta la mappa, senza
    # nessuna nozione di dove sta un insediamento) sono il caso piu' facile
    # da colpire; le gallerie delle miniere evitano gia' di NASCERE dentro
    # un insediamento (`evita`, sotto) ma possono comunque attraversarne uno
    # durante il percorso. Questa maschera dice a entrambe "qui non si
    # scava", a prescindere da quota o raggio.
    protetto = np.zeros(a.cls.shape, bool)
    if muro is not None:
        protetto |= muro > 0
    if tipo_strada is not None:
        protetto |= tipo_strada > 0
    if campi is not None:
        protetto |= campi > 0
    for e in edifici:
        x0, z0, x1, z1 = e.ingombro_tetto()
        protetto[max(0, z0 - 1):z1 + 1, max(0, x0 - 1):x1 + 1] = True

    avanza(0.62, "caverne")
    caverne = (SS.scava_caverne(h, livello_mare=LIVELLO_MARE, seed=19)
               if op.caverne > 0 else {"x": np.zeros(0, np.int32)})
    if caverne["x"].size:
        st = SS.statistiche(caverne)
        note.append(f"sottosuolo: {st['sfere']} celle di cunicolo fra "
                    f"y={st['quota_min']} e y={st['quota_max']}, "
                    f"raggio medio {st['raggio_medio']:.1f}")

    # MINIERE. Sparse a caso su tutta la mappa, non vicino ai paesi (vedi
    # `miniere.pianifica`): l'unico legame con gli insediamenti e' negativo,
    # un ingresso non deve capitare dentro un lotto, sotto una strada o
    # dentro le mura.
    avanza(0.68, "miniere")
    miniere: list = []
    if op.miniere > 0:
        evita = np.zeros(a.cls.shape, bool)
        if muro is not None and muro.any():
            evita |= muro > 0
        if tipo_strada is not None:
            evita |= tipo_strada > 0
        for e in edifici:
            x0, z0, x1, z1 = e.ingombro_tetto()
            evita[max(0, z0 - 1):z1 + 1, max(0, x0 - 1):x1 + 1] = True
        # l'ingresso e' una capanna di RAGGIO_INGRESSO celle attorno al pozzo
        evita |= zona_vulcanica(a, DISTANZA_MIN_VULCANO + MI.RAGGIO_INGRESSO)
        mare_m = np.isin(a.cls, M.MARINO)
        acqua_m = mare_m | (a.cls == M.FIUME)
        miniere = MI.pianifica(h, mare_m, acqua=acqua_m, evita=evita,
                               densita=op.miniere, seed=51)
        if miniere:
            st = MI.statistiche(miniere)
            note.append(f"miniere: {st['miniere']} pozzi, {st['gallerie']} "
                        f"gallerie, {st['lunghezza']} blocchi di percorso, "
                        f"fondo medio y={st['quota_media_fondo']:.0f}")

    # ACCAMPAMENTI E CIMITERI. Stessa idea delle miniere: sparsi a caso,
    # lontani dai paesi (`evita`), mai in acqua. A differenza delle miniere
    # pero' hanno un ingombro vero in superficie (un fuoco, un muro di
    # cinta...), quindi entrano anche in `protetto` - altrimenti una
    # galleria di miniera o un cunicolo potrebbero scavare proprio sotto un
    # cimitero appena disegnato, lasciandolo sospeso sul vuoto, lo stesso
    # difetto gia' visto con le mura delle citta'.
    avanza(0.71, "accampamenti, cimiteri e portali")
    avamposti: list = []
    if op.accampamenti > 0 or op.cimiteri > 0 or op.portali > 0:
        evita_av = np.zeros(a.cls.shape, bool)
        if muro is not None and muro.any():
            evita_av |= muro > 0
        if tipo_strada is not None:
            evita_av |= tipo_strada > 0
        for e in edifici:
            x0, z0, x1, z1 = e.ingombro_tetto()
            evita_av[max(0, z0 - 1):z1 + 1, max(0, x0 - 1):x1 + 1] = True
        mare_av = np.isin(a.cls, M.MARINO) | (a.cls == M.FIUME)
        # Il cimitero e il portale da template (vedi avamposti.py): caricati
        # qui solo per le loro dimensioni, cosi' la spaziatura fra avamposti
        # e la maschera di protezione (`AP.maschera` sotto) usano l'ingombro
        # VERO invece del ripiego 11x11 - la stessa scelta viene rifatta in
        # `scrivi()` per posare i blocchi, come gia' succede per le case
        # (`TM.carica_cartelle` chiamata di nuovo li', non tenuta in cache
        # fra le due funzioni).
        cartelle_cim = [c.strip() for c in op.templates_cimitero.split(";") if c.strip()]
        modelli_cim = AP.carica_cimiteri(cartelle_cim, versione=op.versione) if cartelle_cim else []
        cartelle_portale = [c.strip() for c in op.templates_portale.split(";") if c.strip()]
        modelli_portale = (AP.carica_portali(cartelle_portale, versione=op.versione)
                           if cartelle_portale else [])
        # `AP.pianifica` guarda `evita` solo nel centro dell'avamposto: il
        # raggio del piu' grande fra i modelli va sommato alla distanza
        raggio_av = max(AP.RAGGIO_CAMPO,
                        *AP._mezza_estensione(modelli_cim or None, AP.RAGGIO_CIMITERO),
                        *AP._mezza_estensione(modelli_portale or None, AP.RAGGIO_PORTALE))
        evita_av |= zona_vulcanica(a, DISTANZA_MIN_VULCANO + raggio_av)
        avamposti = AP.pianifica(h, mare_av, evita=evita_av,
                                 campi=op.accampamenti, cimiteri=op.cimiteri,
                                 portali=op.portali, seed=61,
                                 modelli_cimitero=modelli_cim or None,
                                 modelli_portale=modelli_portale or None)
        if avamposti:
            protetto |= AP.maschera(avamposti, a.cls.shape)
            st = AP.statistiche(avamposti)
            note.append(f"avamposti: {st['campi']} accampamenti, "
                        f"{st['cimiteri']} cimiteri, {st['portali']} portali")

    # ARREDI. Dopo strade, campi e avamposti: servono le quote finali e
    # sapere cosa e' gia' occupato. Riempiono i lotti rimasti senza casa
    # (giardini, recinti con le bestie, bazar, piazzette) e mettono in ogni
    # insediamento una campana, una fontana o un pozzo, dei lampioni.
    avanza(0.73, "arredi")
    arredi_r = AR.Risultato()
    if op.arredi > 0 and citta and edifici:
        arredi_r = AR.pianifica(
            edifici, scelte, citta, h, a.cls, vie, tipo_strada, muro, campi,
            banchi, seed=71, densita=op.arredi, versione=op.versione,
            evita=zona_vulcanica(a, DISTANZA_MIN_VULCANO))
        if arredi_r.arredi:
            protetto |= AR.maschera(arredi_r.arredi, a.cls.shape)
        for b in arredi_r.banchi:
            banchi.append(b)
            # il banco sta dentro un lotto senza casa: il lotto resta di quel
            # edificio (nessuna casa da tenere lontana dal vicino)
            protetto[b.z:b.z1, b.x:b.x1] = True
            abitanti.append(EN.Abitante(
                x=b.x + 1.5, y=float(b.base), z=b.z + 1.5,
                mestiere=MERCE_MESTIERE.get(b.merce, "fruttivendolo"),
                seme=b.seme))
        abitanti = abitanti + arredi_r.animali
        if arredi_r.arredi or arredi_r.banchi:
            note.append("arredi: " + ", ".join(
                f"{k} {v}" for k, v in AR.statistiche(arredi_r).items() if v))

    # CASE ISOLATE. I modelli troppo alti per un lotto, sparsi fuori dai
    # paesi: manieri, case sull'albero, fattorie. Vanno dopo tutto il resto
    # perche' il loro criterio e' negativo: dove NON c'e' niente.
    isolate: list = []
    if op.isolate > 0 and modelli_case:
        from scipy.ndimage import binary_dilation as _dil_is
        evita_is = np.zeros(a.cls.shape, bool)
        for e in edifici:
            x0, z0, x1, z1 = e.ingombro_tetto()
            evita_is[max(0, z0 - 1):z1 + 1, max(0, x0 - 1):x1 + 1] = True
        for b in banchi:
            evita_is[max(0, b.z - 1):b.z1 + 1, max(0, b.x - 1):b.x1 + 1] = True
        if tipo_strada is not None:
            evita_is |= tipo_strada > 0
        if muro is not None:
            evita_is |= muro > 0
        if campi is not None:
            evita_is |= campi > 0
        if urbano.any():
            evita_is |= _dil_is(urbano, iterations=8)
        evita_is |= _dil_is(np.isin(a.cls, M.MARINO) | (a.cls == M.FIUME),
                            iterations=4)
        if avamposti:
            evita_is |= AP.maschera(avamposti, a.cls.shape, margine=3)
        for mn in miniere:
            evita_is[max(0, mn.z - 6):mn.z + 7, max(0, mn.x - 6):mn.x + 7] = True
        if arredi_r.arredi:
            evita_is |= AR.maschera(arredi_r.arredi, a.cls.shape, margine=2)
        evita_is |= zona_vulcanica(a, DISTANZA_MIN_VULCANO + 10)
        isolate = IS.pianifica(modelli_case, h, a.cls, evita_is, LIVELLO_MARE,
                               seed=83, densita=op.isolate)
        if isolate:
            protetto |= IS.maschera(isolate, a.cls.shape)
            note.append(f"case isolate: {len(isolate)} "
                        f"({', '.join(modelli_case[c.modello].nome for c in isolate)})")

    avanza(0.75, "vegetazione")
    alberi = (V.semina(a.cls, h, livello_mare=LIVELLO_MARE,
                       scala_densita=op.alberi, seed=21)
              if op.alberi > 0 else None)
    if alberi is not None and op.alberi > 0:
        # la canna da zucchero e' nello stesso elenco: la pianta chi semina
        # gli alberi, la disegna lo stesso `disegna()` - non serve sapere qui
        # che e' un caso a parte, la sua regola (bordo dell'acqua) e' gia'
        # stata applicata in semina_canna()
        canna = V.semina_canna(a.cls, h, livello_mare=LIVELLO_MARE, seed=21)
        if canna["x"].size:
            alberi = V.unisci(alberi, canna)
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
    if alberi is not None and scarpata is not None and scarpata.any():
        # niente chioma a ridosso di uno scalino urbano. Il fronte del
        # gradone e' roccia FORZATA (vedi `scarpata_c` in `blocchi_chunk`,
        # disegnato prima degli alberi): dove la chioma la sfiora, quelle
        # celle non sono aria e `V.disegna()` (che pianta le foglie solo
        # dove trova aria) le salta - il tronco pero' si disegna sempre e
        # vince comunque, quindi sporge da una chioma bucata su un lato.
        # Il raggio della chioma e' 2, quindi 2 celle di distanza dal
        # gradone bastano perche' nessuna foglia ci sbatta contro.
        from scipy.ndimage import binary_dilation as _dil_sc
        vicino_scarpata = _dil_sc(scarpata, iterations=2)
        tieni = ~vicino_scarpata[alberi["z"], alberi["x"]]
        alberi = {k: v[tieni] for k, v in alberi.items()}
    if alberi is not None:
        # Stesso difetto sul bordo di un fiume o di un lago: la sponda
        # (vedi `sponda_m` in `scrivi()`, larga 3 celle dall'acqua) e'
        # anche lei terra/roccia forzata. Un albero seminato appena fuori
        # da quella fascia ha comunque la chioma abbastanza vicina da
        # perdere dei blocchi sul lato dell'acqua - 3 di sponda + 2 di
        # raggio di chioma, 5 celle dal filo dell'acqua.
        fiume_veg = a.cls == M.FIUME
        if fiume_veg.any():
            from scipy.ndimage import binary_dilation as _dil_fv
            vicino_sponda = _dil_fv(fiume_veg, iterations=5) & ~fiume_veg
            tieni = ~vicino_sponda[alberi["z"], alberi["x"]]
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
    if alberi is not None and avamposti:
        # via gli alberi dentro un accampamento o un cimitero: un albero
        # cresciuto in mezzo alla cripta e' lo stesso difetto di un abete in
        # salotto, poche righe sopra.
        dentro_av = AP.maschera(avamposti, a.cls.shape)
        tieni = ~dentro_av[alberi["z"], alberi["x"]]
        if not tieni.all():
            alberi = {k: v[tieni] for k, v in alberi.items()}
    if alberi is not None and isolate:
        # niente albero dentro una casa isolata (ne' addosso alle sue pareti)
        dentro_is = IS.maschera(isolate, a.cls.shape, margine=2)
        tieni = ~dentro_is[alberi["z"], alberi["x"]]
        if not tieni.all():
            alberi = {k: v[tieni] for k, v in alberi.items()}
    if alberi is not None and arredi_r.arredi:
        # niente albero dentro un giardino, un pozzo, sotto un lampione
        dentro_ar = AR.maschera(arredi_r.arredi, a.cls.shape, margine=1)
        tieni = ~dentro_ar[alberi["z"], alberi["x"]]
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

    avanza(0.9, "fauna")
    if op.fauna > 0:
        selvatici = FA.semina_selvatica(a.cls, h, livello_mare=LIVELLO_MARE,
                                        scala_densita=op.fauna, seed=41)
        if selvatici:
            # stessa idea del "preso" dei campi: un'unica maschera di
            # esclusione, non un animale-per-un-animale contro ogni muro,
            # ogni strada e ogni edificio.
            esclusa = np.zeros(a.cls.shape, bool)
            if muro is not None and muro.any():
                from scipy.ndimage import binary_fill_holes as _riempi_f
                esclusa |= _riempi_f(muro > 0)
            if tipo_strada is not None:
                from scipy.ndimage import binary_dilation as _dil_f
                esclusa |= _dil_f(tipo_strada > 0, iterations=1)
            for ed in edifici:
                x0, z0, x1, z1 = E.ingombro(ed, margine=1)
                esclusa[max(0, z0):z1, max(0, x0):x1] = True
            if arredi_r.arredi:
                esclusa |= AR.maschera(arredi_r.arredi, a.cls.shape, margine=1)
            if isolate:
                esclusa |= IS.maschera(isolate, a.cls.shape, margine=1)
            if avamposti:
                # niente cervo o coniglio in mezzo a un accampamento di
                # nemici: sopravviverebbe poco, ed e' comunque un posto
                # dove un animale mite non capiterebbe da solo
                esclusa |= AP.maschera(avamposti, a.cls.shape)
            selvatici = [an for an in selvatici
                        if not esclusa[int(an.z), int(an.x)]]
        animali = selvatici + (FA.per_poderi(poderi, seed=41) if poderi else [])
    else:
        animali = []
    if animali:
        note.append("fauna: " + ", ".join(f"{k} {v}" for k, v in
                                          FA.conteggio(animali).items()))
        abitanti = abitanti + animali

    carrelli = MI.carrelli(miniere, seed=53) if miniere else []
    if carrelli:
        note.append(f"carrelli: {len(carrelli)}")
        abitanti = abitanti + carrelli

    nemici = AP.nemici(avamposti, seed=63, modelli_cimitero=modelli_cim) if avamposti else []
    if nemici:
        note.append(f"nemici: {len(nemici)} fra accampamenti e cimiteri")
        abitanti = abitanti + nemici

    avanza(1.0, "pianificazione completata")
    return Piano(h=h, edifici=edifici, indice_edifici=indice_ed,
                 tipo_strada=tipo_strada, quota_strada=quota_strada,
                 alberi=alberi, indice_alberi=indice, muro=muro,
                 scarpata=scarpata,
                 citta=citta, abitanti=abitanti, banchi=banchi,
                 campi=campi, poderi=poderi, protetto=protetto, caverne=caverne,
                 indice_caverne=SS.indice_per_chunk(caverne) if caverne["x"].size else {},
                 miniere=miniere,
                 indice_miniere=MI.indice_per_chunk(miniere) if miniere else {},
                 avamposti=avamposti,
                 indice_avamposti=AP.indice_per_chunk(avamposti) if avamposti else {},
                 scelte=scelte,
                 isolate=isolate,
                 indice_isolate=IS.indice_per_chunk(isolate) if isolate else {},
                 arredi=arredi_r.arredi, arredi_modelli=arredi_r.modelli,
                 indice_arredi=(AR.indice_per_chunk(arredi_r.arredi)
                                if arredi_r.arredi else {}),
                 indice_banchi=E.indice_banchi(banchi) if banchi else {},
                 proprietario=proprietario,
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


def fetta_orlo(a: np.ndarray, sx: int, sz: int, margine: int = 1) -> np.ndarray:
    """Come `fetta()`, ma con un orlo di `margine` celle intorno al chunk.

    Serve a chi deve sapere com'e' fatto anche il vicino oltre il bordo del
    chunk - non per disegnarlo, solo per guardarlo. Il caso che l'ha reso
    necessario e' la connessione degli steccati: uno steccato scritto sempre
    con i quattro lati "false" si vede in gioco come una fila di paletti
    staccati, perche' il gioco non ricalcola da solo la connessione di un
    blocco gia' generato (lo fa solo per un blocco appena piazzato o quando
    un vicino cambia). Va calcolata qui, e per farlo serve il vicino - che
    puo' stare nel chunk accanto. Fuori mappa si considera "niente", non un
    errore: un orlo che sporge dal bordo del mondo e' normale.
    """
    H, W = a.shape
    lato = 16 + 2 * margine
    fuori = np.zeros((lato, lato), a.dtype)
    z0, z1 = sz - margine, sz + 16 + margine
    x0, x1 = sx - margine, sx + 16 + margine
    zz0, zz1 = max(0, z0), min(H, z1)
    xx0, xx1 = max(0, x0), min(W, x1)
    if zz0 < zz1 and xx0 < xx1:
        fuori[zz0 - z0:zz1 - z0, xx0 - x0:xx1 - x0] = a[zz0:zz1, xx0:xx1]
    return fuori.T


def blocchi_chunk(scrittore, h_c, cls_c, liv_c=None, tipo_c=None,
                  quota_c=None, lava_c=None, colata_c=None, muro_c=None,
                  campi_c=None, ox=0, oz=0, y0=-64, y1=220,
                  roccia=None, scarto_c=None, nuda_c=None, neve_c=None,
                  manto_c=None, scarpata_c=None, sponda_c=None,
                  campi_orlo_c=None, tipo_orlo_c=None):
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

    # --- sponde dei fiumi ---------------------------------------------------
    # La fascia che il fiume scava per restare in un solco (vedi
    # `fiumi.livella`, `scavo_rive`) e' esclusa a monte dalla parete nuda
    # apposta - vedi il commento su `sponda_m` in `scrivi()`. Un fiume e'
    # terra scavata, non una cava: qui la superficie e' terra, per lo piu'
    # battuta (`dirt`), ogni tanto piu' brulla (`coarse_dirt`), e ogni tanto
    # una roccia che sporge - un bordo tutto uguale si legge finto quanto un
    # bordo tutto in pietra. Solo lo strato esposto: sotto resta quello che
    # la classe del terreno gia' prevede.
    if sponda_c is not None and np.any(sponda_c):
        sp = np.asarray(sponda_c).astype(bool)
        terra = scrittore.blocco("dirt")
        brulla = scrittore.blocco("coarse_dirt")
        roccia_sp = scrittore.blocco("stone")
        muschio_sp = scrittore.blocco("mossy_cobblestone")
        for lx, lz in zip(*np.nonzero(sp)):
            cima_y = int(h_c[lx, lz]) - 1 - y0
            if not (0 <= cima_y < out.shape[1]):
                continue
            # hash sulle coordinate DI MONDO, non locali al chunk: con quelle
            # locali il motivo si ripete identico a ogni confine di chunk.
            hv = ((ox + lx) * 73856093) ^ ((oz + lz) * 19349663)
            r = hv % 100
            if r < 8:
                out[lx, cima_y, lz] = muschio_sp if r < 3 else roccia_sp
            elif r < 20:
                out[lx, cima_y, lz] = brulla
            else:
                out[lx, cima_y, lz] = terra

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
        tipo_orlo = (np.asarray(tipo_orlo_c) if tipo_orlo_c is not None
                    else np.pad(np.asarray(tipo_c), 1))
        _posa_strada(out, scrittore, np.asarray(tipo_c), tipo_orlo,
                     np.asarray(quota_c).astype(np.int32),
                     np.asarray(h_c).astype(np.int32), ox, oz, y0)
    if campi_c is not None and np.any(campi_c):
        campi_orlo = (np.asarray(campi_orlo_c) if campi_orlo_c is not None
                     else np.pad(np.asarray(campi_c), 1))
        _posa_campi(out, scrittore, np.asarray(campi_c), campi_orlo,
                    np.asarray(h_c).astype(np.int32), y0)
    if muro_c is not None and np.any(muro_c):
        _posa_mura(out, scrittore, np.asarray(muro_c),
                   np.asarray(h_c).astype(np.int32), ox, oz, y0)
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


def _posa_campi(out, s, campi, campi_orlo, h_c, y0):
    """Terra arata, colture, canali, recinti.

    La superficie del terreno sta a h-1, quindi l'arato sostituisce QUELLO e
    la coltura cresce a h. Sbagliare di uno qui significa grano che spunta
    dentro la terra o terra arata sospesa.

    Il recinto (`fence`) e' un blocco "connesso": la sua forma in gioco
    dipende dalle quattro proprieta' north/south/east/west, che dicono se si
    collega al vicino su quel lato. Scriverle sempre "false" (come faceva
    questa funzione prima) e' un errore dello stesso tipo di quello descritto
    in `mondo.ScrittoreMondo.blocco` per i blocchi senza proprieta': si salva
    senza errori, ma in gioco il gioco NON ricalcola da solo la connessione
    di un blocco che sta gia' nel chunk generato (lo fa solo quando quel
    blocco viene piazzato o un vicino cambia in survival) - il risultato e'
    una fila di paletti isolati invece di uno steccato. Qui la connessione si
    calcola guardando `campi_orlo`, che ha un cella di margine oltre il
    chunk apposta per sapere se il vicino di un palo sul bordo e' un altro
    palo di recinto (o un vuoto) anche quando sta nel chunk accanto.
    """
    H = out.shape[1]
    aria = s.id_aria
    arato = s.blocco("farmland", moisture="7")
    acqua = s.blocco("water", level="0")
    sentiero = s.blocco("grass_path")
    semi = [s.blocco(n, age="7") for n in AG.COLTURE]

    def connesso(lx: int, lz: int) -> bool:
        return int(campi_orlo[lx, lz]) == AG.RECINTO

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
                # +1 di margine: campi_orlo[lx+1, lz+1] e' la cella (lx, lz).
                palo = s.blocco(
                    "fence", material="oak",
                    north="true" if connesso(lx + 1, lz) else "false",
                    south="true" if connesso(lx + 1, lz + 2) else "false",
                    east="true" if connesso(lx + 2, lz + 1) else "false",
                    west="true" if connesso(lx, lz + 1) else "false")
                out[lx, y + 1, lz] = palo
            elif c == AG.PRATO:
                out[lx, y + 1, lz] = aria


ALTEZZA_MURA = 7
ALTEZZA_TORRE = 11
# Quanto in profondita' possono scendere le fondamenta prima di arrendersi.
# In pratica il ciclo sotto non trova mai aria all'interno di `blocchi_chunk`:
# `out[lx, base-1, lz]` e' gia' pietra piena per costruzione (e' lo stesso
# `h_c` che riempie `solido` all'inizio della funzione), quindi la vera causa
# di un muro "appeso" non era la profondita' delle fondamenta ma un cunicolo
# o una galleria scavati DOPO, nello stesso punto (vedi `protetto` in
# `pianifica()` e `protetto_c` in `sottosuolo.posa()`/`miniere.posa()`, che
# e' il fix vero). Il ciclo resta com'e', a costo zero, come margine di
# sicurezza per il caso in cui in futuro qualcos'altro scriva aria sotto una
# cinta prima che le fondamenta vengano posate.
PROFONDITA_FONDAMENTA = 24


def _posa_mura(out, s, muro, h_c, ox, oz, y0):
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

    Altre due correzioni, dallo stesso giro di screenshot dopo che la cinta
    e' diventata spessa tre celle: la screpolatura usava le coordinate
    LOCALI al chunk (`lx + lz`), quindi il motivo ricomincia da capo a ogni
    confine - la stessa trappola gia' vista per viali e sponde, qui rimasta.
    E la merlatura, con lo stesso difetto, marcava un dente si' e uno no su
    OGNI cella piena, spessore compreso: su un muro spesso tre celle viene
    fuori una scacchiera di denti sparsi su tutta la sommita', non una fila
    di merli lungo il bordo. Qui i merli restano solo sul FILO - le celle
    piene con almeno un vicino non pieno - e la screpolatura si legge sulle
    coordinate di mondo.
    """
    H = out.shape[1]
    pietra = s.blocco("stone_bricks", variant="normal")
    mattone = s.blocco("stone_bricks", variant="mossy")
    incrinata = s.blocco("stone_bricks", variant="cracked")
    aria = s.id_aria

    # Il filo esterno/interno del muro: qualunque cella piena con almeno un
    # vicino non pieno nello stesso chunk. Fuori dal ritaglio locale si
    # considera pieno (il muro prosegue nel chunk accanto), altrimenti si
    # inventerebbe un bordo - e quindi merli - proprio sul confine del chunk.
    pieno = np.isin(muro, (1, 3))
    bordo = np.zeros_like(pieno)
    bordo[1:, :] |= pieno[1:, :] & ~pieno[:-1, :]
    bordo[:-1, :] |= pieno[:-1, :] & ~pieno[1:, :]
    bordo[:, 1:] |= pieno[:, 1:] & ~pieno[:, :-1]
    bordo[:, :-1] |= pieno[:, :-1] & ~pieno[:, 1:]
    bordo &= pieno

    for lx in range(16):
        for lz in range(16):
            m = int(muro[lx, lz])
            if m == 0:
                continue
            base = int(h_c[lx, lz]) - y0
            if m in (1, 3):
                # fondamenta: si scende finche' non si trova del pieno
                for giu in range(1, PROFONDITA_FONDAMENTA + 1):
                    k = base - giu
                    if k < 0 or out[lx, k, lz] != aria:
                        break
                    out[lx, k, lz] = pietra
                alta = ALTEZZA_TORRE if m == 3 else ALTEZZA_MURA
                cima = base + alta
                hv = ((ox + lx) * 73856093) ^ ((oz + lz) * 19349663)
                r = hv % 100
                blocco = mattone if r < 14 else (incrinata if r < 23 else pietra)
                for k in range(base, min(cima, H)):
                    out[lx, k, lz] = blocco
                # merli: un dente si' e uno no, solo sul filo del muro
                if bordo[lx, lz] and (ox + lx + oz + lz) % 2 == 0 and 0 <= cima < H:
                    out[lx, cima, lz] = pietra
            else:
                # varco: si sgombera il passaggio e si mette l'architrave
                for k in range(base, min(base + 5, H)):
                    out[lx, k, lz] = aria
                for k in range(base + 5, min(base + ALTEZZA_MURA + 1, H)):
                    if 0 <= k < H:
                        out[lx, k, lz] = pietra


def _posa_strada(out, s, tipo, tipo_orlo, quota, h_c, ox, oz, y0):
    """Sede stradale, impalcati, pilastri e parapetti.

    Il ponte non e' un oggetto piazzato: e' il tratto di strada che si trova
    sull'acqua o in aria e deve comunque stare alla quota della sede. Da qui
    discende tutto - l'impalcato e' la sede, i pilastri scendono fino a dove
    c'e' terreno, il parapetto sta sui bordi.
    """
    # I viali dentro l'abitato (ST.LASTRICATO, rango >= secondaria) erano un
    # lastricato di pietra piatta - in gioco si legge come il pavimento di
    # una fabbrica, non come una piazza vera. Ora sono terra calpestata
    # esattamente come le strade extraurbane (ST.STRADA), con in piu' una
    # spolverata di ghiaia dove il passaggio e' piu' battuto: e' quello che
    # distingue un corso dal vicolo, non una lastra di pietra continua.
    sentiero = s.blocco("grass_path")
    ghiaia = s.blocco("gravel")
    assi = s.blocco("planks", material="oak")
    pilastro = s.blocco("log", axis="y", material="oak", stripped="true")
    aria = s.id_aria
    H = out.shape[1]

    def orlo_parapetto(lx: int, lz: int) -> bool:
        return int(tipo_orlo[lx, lz]) == ST.PARAPETTO

    for lx in range(16):
        for lz in range(16):
            t = int(tipo[lx, lz])
            if t == 0:
                continue
            y = int(quota[lx, lz]) - 1 - y0
            if not (0 <= y < H):
                continue
            if t in (ST.STRADA, ST.LASTRICATO):
                if t == ST.LASTRICATO:
                    # hash sulle coordinate DI MONDO: con quelle locali al
                    # chunk il motivo si ripeterebbe a ogni confine.
                    hv = ((ox + lx) * 73856093) ^ ((oz + lz) * 19349663)
                    out[lx, y, lz] = ghiaia if hv % 100 < 15 else sentiero
                else:
                    out[lx, y, lz] = sentiero
                out[lx, y + 1:min(y + 4, H), lz] = aria
                continue

            out[lx, y, lz] = assi                    # impalcato
            out[lx, y + 1:min(y + 5, H), lz] = aria  # luce sopra
            if t == ST.PARAPETTO and y + 1 < H:
                # Connesso come il recinto dei campi (vedi il commento in
                # _posa_campi): il vicino puo' stare nel chunk accanto, da
                # qui `tipo_orlo` con la sua cella di margine.
                staccionata = s.blocco(
                    "fence", material="oak",
                    north="true" if orlo_parapetto(lx + 1, lz) else "false",
                    south="true" if orlo_parapetto(lx + 1, lz + 2) else "false",
                    east="true" if orlo_parapetto(lx + 2, lz + 1) else "false",
                    west="true" if orlo_parapetto(lx, lz + 1) else "false")
                out[lx, y + 1, lz] = staccionata

            # pilastri a intervalli regolari, fino al terreno o al fondale.
            # Solo se il vuoto sotto e' abbastanza alto da leggersi come un
            # vero pilastro: sotto i due blocchi resta un moncone isolato in
            # mezzo al prato - il caso tipico e' un ponte quasi a raso, sopra
            # un laghetto poco profondo o un avvallamento lieve del terreno.
            gx, gz = ox + lx, oz + lz
            if gx % 5 == 0 and gz % 5 == 0:
                fondo = max(0, int(h_c[lx, lz]) - 1 - y0)
                if fondo < y - 1:
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

    imp = ImpostazioniMondo(nome=op.nome, modalita=1, spawn=(0, 150, 0),
                            versione=op.versione)
    os.makedirs(os.path.dirname(os.path.abspath(op.uscita)) or ".", exist_ok=True)
    primo = da == 0
    t0 = time.time()
    scritti = 0
    interrotto = False

    with ScrittoreMondo(op.uscita, imp, lato_blocchi=op.lato, crea=primo) as m:
        tav_ss = SS.Tavolozza(m)
        tav_mi = MI.Tavolozza(m) if piano.miniere else None
        tav_av = AP.Tavolozza(m) if piano.avamposti else None
        cat_ar = TM.Catalogo(piano.arredi_modelli, m) if piano.arredi else None
        # Il cimitero e il portale da template (vedi avamposti.py e il
        # commento gemello in motore.pianifica): stesso meccanismo delle case
        # da template qualche riga sotto, un catalogo a parte per ciascuno
        # perche' sono un'altra cartella e un'altra scelta (una struttura
        # sola, non una per lotto).
        cartelle_cim = [c.strip() for c in op.templates_cimitero.split(";") if c.strip()]
        modelli_cim = (AP.carica_cimiteri(cartelle_cim, versione=op.versione)
                      if cartelle_cim and piano.avamposti else [])
        cat_cim = TM.Catalogo(modelli_cim, m) if modelli_cim else None
        aria_cim = m.id_aria if cat_cim is not None else 0
        basamento_cim = m.blocco("cobblestone") if cat_cim is not None else 0
        cartelle_portale = [c.strip() for c in op.templates_portale.split(";") if c.strip()]
        modelli_portale = (AP.carica_portali(cartelle_portale, versione=op.versione)
                           if cartelle_portale and piano.avamposti else [])
        cat_portale = TM.Catalogo(modelli_portale, m) if modelli_portale else None
        aria_portale = m.id_aria if cat_portale is not None else 0
        basamento_portale = m.blocco("cobblestone") if cat_portale is not None else 0
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

        # La sponda: la fascia di terra che il fiume scava per restare in un
        # solco (vedi `fiumi.livella`, `scavo_rive`) e' spesso ripida quanto
        # bastava a farla leggere come "parete" da `superficie_montana` -
        # risultato, un bordo di pietra nuda intorno all'acqua, uno screenshot
        # da controllare. Il fiume e' terra scavata, non una cava: qui non
        # deve mai comparire la roccia di default, solo terra (vedi
        # `_posa_sponde`).
        fiume_m = a.cls == M.FIUME
        sponda_m = np.zeros(a.cls.shape, bool)
        if fiume_m.any():
            from scipy.ndimage import binary_dilation as _dil_sp
            sponda_m = _dil_sp(fiume_m, iterations=3) & ~fiume_m
            nuda_m &= ~sponda_m
        scarto_m = SG.piega(max(piano.h.shape),
                            seed=op.seed)[:piano.h.shape[0], :piano.h.shape[1]]
        manto_m = U.manto(a.cls, seed=op.seed)
        tav = V.Tavolozza(m) if piano.alberi is not None else None
        tav_ed = E.TavolozzaEdilizia(m) if (piano.edifici or piano.isolate) else None
        tav_ba = BA.Tavolozza(m) if piano.edifici else None

        # --- case da template ------------------------------------------
        # Il modello di ogni lotto lo ha deciso `pianifica()` (`piano.scelte`),
        # senza ripetizioni nel villaggio: qui si carica solo il catalogo, con
        # lo stesso ordine, perche' gli indici coincidano.
        cartelle_t = [c.strip() for c in op.templates.split(";") if c.strip()]
        modelli = (TM.carica_cartelle(cartelle_t, versione=op.versione)
                  if cartelle_t else [])
        cat = TM.Catalogo(modelli, m) if modelli else None
        scelte: dict[int, tuple[int, int]] = piano.scelte if cat is not None else {}
        if cat is not None:
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
                            if piano.scarpata is not None else None),
                sponda_c=fetta(sponda_m, sx, sz),
                campi_orlo_c=(fetta_orlo(piano.campi, sx, sz)
                             if piano.campi is not None else None),
                tipo_orlo_c=(fetta_orlo(piano.tipo_strada, sx, sz)
                            if piano.tipo_strada is not None else None))
            # Il sottosuolo si scava DOPO che `blocchi` ha gia' le strutture
            # in superficie (mura, strade, campi, case): le caverne devono
            # scavare nella roccia, non dentro una casa o sotto una strada
            # gia' posata - `protetto_c` e' quello che lo garantisce (vedi
            # il commento su `protetto` in `pianifica()`): senza, un
            # cunicolo puo' passare proprio sotto una cinta muraria e
            # lasciarla appesa sul vuoto.
            protetto_c = (fetta(piano.protetto, sx, sz)
                         if piano.protetto is not None else None)
            SS.posa(blocchi, tav_ss, fetta(piano.h, sx, sz), sx, sz, -64,
                    piano.caverne,
                    piano.indice_caverne.get((sx // 16, sz // 16), ()),
                    seed=19, protetto_c=protetto_c)
            # Le miniere DOPO le caverne, per lo stesso motivo: scavano
            # anche loro, e i filoni lungo il percorso devono sostituire
            # pietra vera, non l'aria di un cunicolo che ha gia' scavato
            # nello stesso punto. Stessa protezione delle caverne: le
            # gallerie evitano di NASCERE in un insediamento (`evita` in
            # `miniere.pianifica`) ma possono comunque attraversarne uno.
            if tav_mi is not None:
                MI.posa(blocchi, tav_mi, fetta(piano.h, sx, sz), sx, sz, -64,
                        piano.miniere,
                        piano.indice_miniere.get((sx // 16, sz // 16), ()),
                        tav_ss.scavabile, seed=53, protetto_c=protetto_c)
            # Accampamenti e cimiteri: SENZA `protetto_c` - a differenza
            # dello scavo delle caverne e delle miniere, qui non c'e' niente
            # da proteggere da questo disegno, ed e' proprio il loro stesso
            # ingombro (dentro `protetto`, vedi `pianifica()`) che ha appena
            # tenuto lontani i cunicoli sopra - la stessa ragione per cui
            # neanche `TM.costruisci()` (le case) lo controlla piu' sotto.
            if tav_av is not None:
                AP.posa(blocchi, tav_av, sx, sz, -64, piano.avamposti,
                       piano.indice_avamposti.get((sx // 16, sz // 16), ()),
                       cat_cimitero=cat_cim, aria_cimitero=aria_cim,
                       basamento_cimitero=basamento_cim,
                       cat_portale=cat_portale, aria_portale=aria_portale,
                       basamento_portale=basamento_portale)
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
                propr_c = (fetta(piano.proprietario, sx, sz)
                          if piano.proprietario is not None else None)
                for k in piano.indice_edifici.get((sx // 16, sz // 16), ()):
                    ed = piano.edifici[k]
                    if k in scelte:
                        mi, quarti = scelte[k]
                        ix, iz = modelli[mi].ingombro(quarti)
                        px = ed.x + (ed.larghezza - ix) // 2
                        pz = ed.z + (ed.profondita - iz) // 2
                        # `vietato_c` marca le colonne che appartengono al
                        # LOTTO di un ALTRO edificio (o a un banco): la
                        # gronda di questa casa non disegna sopra quella
                        # vicina, che apparirebbe "a meta'". Non e'
                        # `protetto_c`: qui non si protegge il sottosuolo, si
                        # protegge un edificio dall'altro.
                        vietato_c = (None if propr_c is None else
                                    (propr_c != -1) & (propr_c != k))
                        # molti template hanno uno strato di terra sotto il
                        # pavimento: interrato di uno, la loro erba prende il
                        # posto di quella del lotto invece di fare da zoccolo
                        base_c = ed.base - modelli[mi].affondo
                        TM.fondazione(blocchi, -64, sx, sz, cat, mi, quarti,
                                      px, pz, base_c,
                                      tav_ed.blocco[(ed.stile, "basamento")],
                                      tav_ed.aria, vietato=vietato_c)
                        TM.costruisci(blocchi, -64, sx, sz, cat, mi, quarti,
                                      px, pz, base_c, vietato=vietato_c)
                        if ed.mestiere:
                            E.posto_di_lavoro(blocchi, -64, sx, sz, ed, tav_ed)
                    # se il lotto non e' in `scelte` resta senza casa: se ne
                    # occupano gli arredi (`arredi.py`), non un generatore
                    # parametrico di riserva, che non esiste piu'.
                for k in piano.indice_banchi.get((sx // 16, sz // 16), ()):
                    E.costruisci_banco(blocchi, -64, sx, sz, piano.banchi[k],
                                       tav_ed)
            if cat is not None and tav_ed is not None:
                # le case isolate: stessa posa di quelle dei villaggi
                for k in piano.indice_isolate.get((sx // 16, sz // 16), ()):
                    ci = piano.isolate[k]
                    base_c = ci.base - modelli[ci.modello].affondo
                    TM.fondazione(blocchi, -64, sx, sz, cat, ci.modello, ci.quarti,
                                  ci.x, ci.z, base_c,
                                  tav_ed.blocco[(ci.stile, "basamento")],
                                  tav_ed.aria)
                    TM.costruisci(blocchi, -64, sx, sz, cat, ci.modello, ci.quarti,
                                  ci.x, ci.z, base_c)
            if cat_ar is not None:
                # gli arredi dopo le case e i banchi: un lampione o un giardino
                # occupano solo cio' che e' rimasto libero
                for k in piano.indice_arredi.get((sx // 16, sz // 16), ()):
                    ar = piano.arredi[k]
                    TM.costruisci(blocchi, -64, sx, sz, cat_ar, ar.modello, 0,
                                  ar.x, ar.z, ar.base)
            bio_c = (id_bioma[fetta(a.biomi, sx, sz)]
                     if id_bioma is not None else None)
            # Bottino dei forzieri: si cerca il blocco DOPO che le case sono
            # gia' state piazzate (vedi `bauli.trova()` sul perche' non si
            # segue l'elenco degli edifici), quindi va qui, a valle di tutto
            # cio' che puo' scrivere un forziere.
            #
            # ATTENZIONE alle coordinate, lo stesso avviso di `EN.scrivi_regioni`
            # qualche riga sotto: `sx, sz` sono coordinate di MAPPA (0..lato,
            # quello che usano `fetta()` e la generazione), non di GIOCO. Un
            # `BlockEntity` pero' si inserisce con la posizione assoluta nel
            # mondo, quindi qui ci vuole `ox, oz` (l'angolo del chunk gia' in
            # coordinate di gioco, calcolato sopra) - con `sx, sz` il forziere
            # finiva spostato di `meta` blocchi da quello vero, su una cella
            # qualunque senza nessun blocco `chest`: in gioco restava sempre
            # vuoto, e il controllo con un mondo di un solo chunk non lo vedeva
            # perche' a lato=16 l'offset `meta` e' zero.
            bauli_c = (BA.trova(blocchi, tav_ba, ox, oz, -64, seed=71)
                      if tav_ba is not None else None)
            m.scrivi_chunk(cx, cz, blocchi, biomi=bio_c, bauli=bauli_c)
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
        # `_Imp()` da solo darebbe sempre il DataVersion di 1.21.4 di
        # default, non quello scelto in `op.versione` - level.dat e le
        # entita' finirebbero con due DataVersion diversi.
        st = EN.scrivi_regioni(op.uscita, in_gioco,
                               _Imp(versione=op.versione).data_version)
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
