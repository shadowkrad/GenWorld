"""Scrittura dei chunk in piu' processi (#16).

Quasi tutto il tempo di un mondo grande e' nella scrittura dei chunk, e ogni
chunk si calcola da solo: dipende dal piano e dalle mappe, mai dagli altri
chunk. Il collo di bottiglia e' un processo solo.

Ogni processo scrive i SUOI chunk in un mondo di lavoro a parte (nessun
accesso concorrente agli stessi file, nessun `session.lock` da contendersi) e
alla fine i file `r.X.Z.mca` si spostano nel mondo vero. Le regioni sono
l'unita' di divisione, quindi due processi non scrivono mai la stessa regione.
Ogni chunk porta con se' il proprio palette, quindi gli id di palette diversi
fra processi non contano.

Cio' che non e' per chunk (abitanti, bordo naturale, elenco dei luoghi,
verifica del level.dat) lo fa il processo principale dopo, come prima.
"""
from __future__ import annotations

import dataclasses
import glob
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

REGIONE = 32            # chunk per lato di un file region

# Un mondo grande ha array enormi nel piano (centinaia di MB l'uno). Spedirli per pickle a ogni
# lavoratore vuol dire una copia a testa: con 4 processi sono 4 volte la memoria. Qui gli array
# sopra questa soglia vanno su disco in file `.npy` e i lavoratori li MAPPANO in sola lettura
# (sola lettura): il sistema ne tiene una copia sola in memoria, condivisa, e Windows non conta ogni mappatura
# nella memoria impegnata (con la copia-su-scrittura si: 6 lavoratori x 5 GB facevano esaurire il limite).
SOGLIA_ARRAY = 4 << 20


@dataclasses.dataclass
class RifArray:
    """Al posto di un array grande, dentro il pickle: dove ritrovarlo."""
    percorso: str


def su_disco(oggetto, cartella: str, visti: dict | None = None, livello: int = 0):
    """Una copia di `oggetto` con gli array grandi sostituiti da `RifArray`. Guarda dentro
    dataclass, dizionari, liste e tuple (due livelli: basta al piano e all'analisi)."""
    visti = {} if visti is None else visti
    if isinstance(oggetto, np.ndarray):
        if oggetto.nbytes < SOGLIA_ARRAY:
            return oggetto
        if id(oggetto) not in visti:
            nome = os.path.join(cartella, f"a{len(visti)}.npy")
            np.save(nome, np.ascontiguousarray(oggetto))
            visti[id(oggetto)] = RifArray(nome)
        return visti[id(oggetto)]
    if livello >= 3:
        return oggetto
    if dataclasses.is_dataclass(oggetto) and not isinstance(oggetto, type):
        cambi = {}
        for f in dataclasses.fields(oggetto):
            v = getattr(oggetto, f.name)
            n = su_disco(v, cartella, visti, livello + 1)
            if n is not v:
                cambi[f.name] = n
        return replace(oggetto, **cambi) if cambi else oggetto
    if isinstance(oggetto, dict):
        nuovo = {k: su_disco(v, cartella, visti, livello + 1) for k, v in oggetto.items()}
        return nuovo if any(nuovo[k] is not oggetto[k] for k in oggetto) else oggetto
    if isinstance(oggetto, (list, tuple)) and len(oggetto) < 64:
        nuovo = [su_disco(v, cartella, visti, livello + 1) for v in oggetto]
        if all(n is v for n, v in zip(nuovo, oggetto)):
            return oggetto
        return type(oggetto)(nuovo) if isinstance(oggetto, list) else tuple(nuovo)
    return oggetto


def da_disco(oggetto, livello: int = 0):
    """L'inverso di `su_disco`: gli `RifArray` tornano array, mappati dal file in sola lettura."""
    if isinstance(oggetto, RifArray):
        return np.load(oggetto.percorso, mmap_mode="r")
    if livello >= 3:
        return oggetto
    if dataclasses.is_dataclass(oggetto) and not isinstance(oggetto, type):
        cambi = {}
        for f in dataclasses.fields(oggetto):
            v = getattr(oggetto, f.name)
            n = da_disco(v, livello + 1)
            if n is not v:
                cambi[f.name] = n
        return replace(oggetto, **cambi) if cambi else oggetto
    if isinstance(oggetto, dict):
        nuovo = {k: da_disco(v, livello + 1) for k, v in oggetto.items()}
        return nuovo if any(nuovo[k] is not oggetto[k] for k in oggetto) else oggetto
    if isinstance(oggetto, (list, tuple)) and len(oggetto) < 64:
        nuovo = [da_disco(v, livello + 1) for v in oggetto]
        if all(n is v for n, v in zip(nuovo, oggetto)):
            return oggetto
        return type(oggetto)(nuovo) if isinstance(oggetto, list) else tuple(nuovo)
    return oggetto

_dati = None
_contatore = None


def coordinate_chunk(lato: int, altezza: int | None = None) -> list[tuple[int, int]]:
    """Le stesse coordinate di `ScrittoreMondo.chunk_coords()`, senza aprire un mondo."""
    meta = (lato // 16) // 2
    return [(cx, cz) for cx in range(-meta, meta)
            for cz in range(-meta, -meta + (altezza or lato) // 16)]


def ripartisci(coords: list[tuple[int, int]], compiti: int) -> list[list[int]]:
    """Divide gli indici dei chunk in al piu' `compiti` gruppi, ognuno fatto di
    regioni intere (carico bilanciato: le regioni piu' piene per prime)."""
    regioni: dict[tuple[int, int], list[int]] = {}
    for k, (cx, cz) in enumerate(coords):
        regioni.setdefault((cx // REGIONE, cz // REGIONE), []).append(k)
    gruppi: list[list[int]] = [[] for _ in range(max(1, min(compiti, len(regioni))))]
    for indici in sorted(regioni.values(), key=len, reverse=True):
        min(gruppi, key=len).extend(indici)
    return [sorted(g) for g in gruppi if g]


def _avvia(percorso: str, contatore) -> None:
    global _dati, _contatore
    with open(percorso, "rb") as f:
        _dati = da_disco(pickle.load(f))
    _contatore = contatore


def _lavora(k: int, indici: list[int], cartella: str) -> dict:
    from .motore import con_uscita, scrivi
    op, a, piano, precaricati = _dati
    sub = os.path.join(cartella, f"w{k}")
    visti = [0]

    def avanza(frazione: float, _testo: str) -> None:
        fatti = int(frazione * len(indici))
        with _contatore.get_lock():
            _contatore.value += fatti - visti[0]
        visti[0] = fatti

    st = scrivi(con_uscita(op, sub, op.nome), a, piano, avanza=avanza,
                indici=indici, lavoratore=True, precaricati=precaricati)
    with _contatore.get_lock():
        _contatore.value += len(indici) - visti[0]
    return {"k": k, "cartella": sub, "chunk": st["chunk"]}


def unisci(op, cartelle: list[str]) -> int:
    """Crea il mondo vero (vuoto) e ci sposta i file region dei lavoratori."""
    from .livello_dat import ImpostazioniMondo
    from .mondo import ScrittoreMondo
    imp = ImpostazioniMondo(nome=op.nome, modalita=1, spawn=op.spawn,
                            versione=op.versione, seed=op.seed,
                            fuori_mappa=op.fuori_mappa,
                            sottosopra=getattr(op, 'sottosopra', False))
    with ScrittoreMondo(op.uscita, imp, lato_blocchi=op.lato, altezza_blocchi=op.alt, crea=True):
        pass
    spostati = 0
    for sub in cartelle:
        for f in glob.glob(os.path.join(glob.escape(sub), "**", "r.*.mca"), recursive=True):
            rel = os.path.relpath(f, sub)
            dest = os.path.join(op.uscita, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                raise RuntimeError(f"due lavoratori hanno scritto la stessa regione: {rel}")
            shutil.move(f, dest)
            spostati += 1
    return spostati


def scrivi_parallelo(op, a, piano, processi: int = 0, avanza=None, ferma=None) -> dict:
    """Come `motore.scrivi()` per tutto il mondo, con `processi` processi
    (0 = tanti quanti i core, al massimo 8). Ritorna lo stesso dizionario."""
    from .motore import _nulla, carica_modelli, mappe_globali, scrivi
    avanza = avanza or _nulla
    n = processi or min(8, os.cpu_count() or 1)
    coords = coordinate_chunk(op.lato, op.alt)
    gruppi = ripartisci(coords, n * 2)
    if n <= 1 or len(gruppi) <= 1:
        return scrivi(op, a, piano, avanza, ferma=ferma)

    # `scrivi` usa piano.cls/livello al posto di quelli dell'analisi: lo si fa
    # qui una volta, per non portarsi dietro due copie
    if piano.cls is not None:
        a = replace(a, cls=piano.cls, livello=piano.livello)
    t0 = time.time()
    base = tempfile.mkdtemp(prefix="genworld_", dir=os.path.dirname(os.path.abspath(op.uscita)) or None)
    try:
        # i template si leggono UNA volta, qui: riletti da piu' processi insieme
        # (i fari si leggono da un mondo) a volte uscivano vuoti, e gli indici
        # dei modelli del piano non coincidevano piu'
        precaricati = carica_modelli(op, piano)
        # le mappe di tutto il mondo si calcolano qui, UNA volta, e i lavoratori le mappano
        precaricati["mappe"] = mappe_globali(op, a, piano)
        dati = os.path.join(base, "dati.pkl")
        pacchetto = su_disco((op, a, piano, precaricati), base)
        with open(dati, "wb") as f:
            pickle.dump(pacchetto, f, protocol=pickle.HIGHEST_PROTOCOL)
        ctx = mp.get_context("spawn")
        contatore = ctx.Value("i", 0)
        totale = len(coords)
        fatti = []
        with ProcessPoolExecutor(max_workers=min(n, len(gruppi)), mp_context=ctx,
                                 initializer=_avvia, initargs=(dati, contatore)) as ex:
            futuri = [ex.submit(_lavora, k, g, base) for k, g in enumerate(gruppi)]
            for fu in as_completed(futuri):
                fatti.append(fu.result())
                avanza(min(1.0, contatore.value / totale), f"chunk {contatore.value} di {totale}")
                if ferma is not None and ferma():
                    for g in futuri:
                        g.cancel()
                    raise InterruptedError("generazione annullata")
        avanza(1.0, "unisco le regioni")
        unisci(op, [r["cartella"] for r in sorted(fatti, key=lambda r: r["k"])])
        scritti = sum(r["chunk"] for r in fatti)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    # i passaggi finali, una volta sola e nel processo principale
    return _finali(op, a, piano, scritti, len(coords), time.time() - t0)


def _finali(op, a, piano, scritti: int, totale: int, secondi: float) -> dict:
    """Quello che `scrivi()` fa quando non resta nessun chunk."""
    from .motore import passaggi_finali
    return passaggi_finali(op, piano, {"chunk": scritti, "totale": totale, "restano": 0,
                                       "secondi": secondi, "interrotto": False})
