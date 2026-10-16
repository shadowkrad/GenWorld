"""Template compilati: la libreria personale di strutture, gia' pronta.

Leggere un template costa: aprire il file (un `.nbt`, uno schema, a volte un
intero mondo come per i fari), tradurre ogni blocco al formato universale,
ripararlo, ritagliarlo. Il risultato e' sempre lo stesso finche' il file e il
codice che lo legge non cambiano, quindi si fa UNA volta e si tiene su disco:
la prossima generazione rilegge il modello gia' pronto, in una frazione del
tempo.

Dove stanno: `templates/_compilati/` (cambia con GENWORLD_COMPILATI). Sono
derivati dei template di terzi, quindi restano SOLO sul disco, mai nel
repository (vedi `.gitignore`), come i template stessi.

Quando si rifanno: cambia il file (o ne arriva uno nuovo, o sparisce), cambia
`meta.json`/`stili.json` accanto, cambia la versione di gioco o cambia il
codice dei lettori. GENWORLD_SENZA_COMPILATI=1 salta la cache.

Il punto di ingresso per "migliorare" un modello e' `MIGLIORAMENTI`: funzioni
`f(modello) -> modello` applicate PRIMA di salvare, una volta sola. Quello che
si corregge li' (un blocco da sostituire, una porta da raddrizzare) resta
corretto in ogni mondo, senza ripetere il lavoro a ogni generazione.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Callable

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# i lettori il cui codice decide il contenuto di un modello: se cambiano, i
# compilati non valgono piu'
LETTORI = ("template.py", "monumenti.py", "avamposti.py", "compilati.py")
VERSIONE_FORMATO = 2

def varia_arenaria(m):
    """L'arenaria liscia di una costruzione del deserto e' una parete piatta:
    qui una parte dei blocchi diventa arenaria tagliata (la maggior parte),
    liscia o scolpita, scelti dalla posizione. Vale per `sandstone` e
    `red_sandstone` normali; le varianti gia' presenti non si toccano."""
    import numpy as np
    nuove = list(m.tavolozza)
    celle = m.celle.copy()
    cambiato = False
    for base in ("sandstone", "red_sandstone"):
        idx = [i for i, (n, p) in enumerate(m.tavolozza)
               if n == base and p.get("variant", "normal") == "normal"]
        if not idx:
            continue
        # le tre varianti si aggiungono in coda alla tavolozza
        varianti = {}
        for v in ("cut", "smooth", "chiseled"):
            nuove.append((base, {"variant": v}))
            varianti[v] = len(nuove) - 1
        x, y, z = np.indices(celle.shape)
        dado = ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) % 100
        for i in idx:
            dentro = celle == i
            celle[dentro & (dado < 24)] = varianti["cut"]
            celle[dentro & (dado >= 24) & (dado < 30)] = varianti["smooth"]
            celle[dentro & (dado >= 30) & (dado < 33)] = varianti["chiseled"]
        cambiato = True
    if not cambiato:
        return m
    import dataclasses
    return dataclasses.replace(m, celle=celle, tavolozza=nuove)


# funzioni modello -> modello, applicate prima di salvare (vedi sopra)
MIGLIORAMENTI: list[Callable] = [varia_arenaria]


def cartella() -> str:
    return os.environ.get("GENWORLD_COMPILATI") or os.path.join(RADICE, "templates", "_compilati")


def attiva() -> bool:
    return not os.environ.get("GENWORLD_SENZA_COMPILATI")


def _codice() -> str:
    h = hashlib.sha256(str(VERSIONE_FORMATO).encode())
    qui = os.path.dirname(os.path.abspath(__file__))
    for nome in LETTORI:
        with open(os.path.join(qui, nome), "rb") as f:
            h.update(nome.encode() + hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def impronta(cartelle: list[str]) -> str:
    """Impronta del contenuto delle cartelle: ogni file (anche dentro i mondi
    dei fari) con percorso relativo, dimensione e data di modifica."""
    h = hashlib.sha256()
    for c in cartelle:
        h.update(f"[{os.path.abspath(c)}]".encode())
        if not os.path.isdir(c):
            h.update(b"assente")
            continue
        for radice, dirs, files in os.walk(c):
            dirs.sort()
            for nome in sorted(files):
                f = os.path.join(radice, nome)
                try:
                    st = os.stat(f)
                except OSError:
                    continue
                h.update(f"{os.path.relpath(f, c)}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()


def _migliora(risultato):
    if not MIGLIORAMENTI:
        return risultato

    def uno(m):
        for f in MIGLIORAMENTI:
            m = f(m)
        return m
    if isinstance(risultato, tuple):
        return tuple([uno(m) for m in parte] for parte in risultato)
    return [uno(m) for m in risultato]


def con_cache(genere: str, cartelle: list[str], versione, produci: Callable):
    """`produci()` ritorna i modelli (una lista, o una tupla di liste). Se i file
    e il codice sono gli stessi dell'ultima volta si rilegge il risultato salvato.
    Ogni chiamata riceve oggetti NUOVI (i chiamanti li modificano)."""
    if not attiva():
        return produci()
    # lo stesso genere con cartelle diverse (case, cimiteri, portali) sono file
    # diversi: il prefisso porta i percorsi, la chiave il contenuto
    prefisso = f"{genere}-" + hashlib.sha256(
        "|".join(os.path.abspath(c) for c in cartelle).encode()).hexdigest()[:8] + "-"
    chiave = hashlib.sha256(
        f"{tuple(versione)}|{_codice()}|{impronta(cartelle)}".encode()).hexdigest()[:32]
    percorso = os.path.join(cartella(), f"{prefisso}{chiave}.pkl")
    if os.path.exists(percorso):
        try:
            with open(percorso, "rb") as f:
                return pickle.load(f)
        except Exception:                                  # noqa: BLE001 - file rovinato: si rifa'
            pass
    risultato = _migliora(produci())
    try:
        os.makedirs(cartella(), exist_ok=True)
        # una versione vecchia dello stesso genere non serve piu'
        for vecchio in os.listdir(cartella()):
            if vecchio.startswith(prefisso) and vecchio.endswith(".pkl"):
                try:
                    os.remove(os.path.join(cartella(), vecchio))
                except OSError:
                    pass
        tmp = percorso + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(risultato, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, percorso)
    except OSError:
        pass                                               # sola lettura: si usa il risultato e basta
    return risultato


def svuota() -> int:
    """Cancella i compilati. Ritorna quanti file."""
    n = 0
    if os.path.isdir(cartella()):
        for f in os.listdir(cartella()):
            if f.endswith(".pkl"):
                os.remove(os.path.join(cartella(), f))
                n += 1
    return n
