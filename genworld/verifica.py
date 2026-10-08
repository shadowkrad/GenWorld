"""Verifica automatica di un mondo gia' scritto.

I difetti segnalati finora sono stati trovati a occhio, in gioco: banchi
sospesi in aria, mura appese al vuoto, acqua che trabocca dal fosso. Qui si
rilegge il mondo dal disco e si cercano a macchina, a finestre di 64x64 blocchi
per non tenere in memoria tutto.

Controlli:

* **completezza**: ci sono tutti i chunk che il lato dichiara;
* **sospesi**: pezzi di costruzione (blocchi non d'aria e non d'acqua) che non
  toccano ne' il suolo ne' il resto del mondo - un banco o una trave
  appesa. Un pezzo che tocca il bordo della finestra si ignora (continua nella
  finestra accanto), quindi i falsi positivi sono possibili solo per
  strutture lunghe piu' di 64 blocchi e staccate da tutto;
* **acqua che trabocca**: una cella d'acqua con l'aria accanto allo stesso
  livello, che in gioco scorrerebbe via.

Non sostituisce una visita in gioco: dice dove guardare.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label

FINESTRA = 64
SOSPESO_MIN = 3           # un blocco o due isolati sono minerali in una caverna
SOSPESO_MAX = 4000        # sopra questa dimensione un pezzo e' un'isola, non un detrito
Y_INIZIO = -64


@dataclass
class Rapporto:
    chunk_attesi: int = 0
    chunk_trovati: int = 0
    sospesi: list[dict] = field(default_factory=list)
    acqua: list[dict] = field(default_factory=list)

    @property
    def completo(self) -> bool:
        return self.chunk_trovati >= self.chunk_attesi

    @property
    def pulito(self) -> bool:
        return self.completo and not self.sospesi and not self.acqua

    def righe(self) -> list[str]:
        r = [f"chunk: {self.chunk_trovati} su {self.chunk_attesi}"]
        r.append(f"pezzi sospesi: {len(self.sospesi)}")
        r += [f"  ({s['x']}, {s['y']}, {s['z']}) {s['blocchi']} blocchi" for s in self.sospesi[:20]]
        r.append(f"acqua che trabocca: {len(self.acqua)}")
        r += [f"  ({s['x']}, {s['y']}, {s['z']})" for s in self.acqua[:20]]
        return r


def pezzi_sospesi(solido: np.ndarray, massimo: int = SOSPESO_MAX,
                  minimo: int = SOSPESO_MIN) -> list[tuple[int, int, int, int]]:
    """Componenti connesse (6 vicini) di `solido` [x, y, z] che non toccano
    ne' il fondo ne' nessun lato della finestra. Ritorna (x, y, z, n) di un
    blocco per componente, quello piu' in basso."""
    etichette, n = label(solido)
    if n == 0:
        return []
    toccano = set(np.unique(etichette[0])) | set(np.unique(etichette[-1]))
    toccano |= set(np.unique(etichette[:, 0])) | set(np.unique(etichette[:, -1]))
    toccano |= set(np.unique(etichette[:, :, 0])) | set(np.unique(etichette[:, :, -1]))
    fuori = []
    dimensioni = np.bincount(etichette.ravel())
    for k in range(1, n + 1):
        if k in toccano or not minimo <= dimensioni[k] <= massimo:
            continue
        xs, ys, zs = np.nonzero(etichette == k)
        i = int(np.argmin(ys))
        fuori.append((int(xs[i]), int(ys[i]), int(zs[i]), int(dimensioni[k])))
    return fuori


def acqua_che_trabocca(acqua: np.ndarray, aria: np.ndarray
                       ) -> list[tuple[int, int, int]]:
    """Celle d'acqua [x, y, z] con aria a fianco (stesso y) o sotto."""
    esposta = np.zeros_like(acqua)
    for asse in (0, 2):
        for s in (1, -1):
            esposta |= acqua & np.roll(aria, s, axis=asse)
    # i bordi della finestra non contano: il vicino e' fuori campo
    esposta[0] = esposta[-1] = False
    esposta[:, :, 0] = esposta[:, :, -1] = False
    return [tuple(int(v) for v in p) for p in np.argwhere(esposta)]


def verifica_mondo(percorso: str, lato: int, avanza=None) -> Rapporto:
    """Rilegge il mondo in `percorso` (lato in blocchi) e lo controlla."""
    import amulet

    from .mondo import DIMENSIONE

    meta = lato // 2
    rap = Rapporto(chunk_attesi=(lato // 16) ** 2)
    lv = amulet.load_level(percorso)
    try:
        pal = lv.block_palette
        cache: dict[int, str] = {}

        def nome(i: int) -> str:
            if i not in cache:
                cache[i] = pal[i].base_name
            return cache[i]

        presenti = set(lv.all_chunk_coords(DIMENSIONE))
        rap.chunk_trovati = len(presenti)
        passo = FINESTRA // 16
        cx_min, cz_min = -meta // 16, -meta // 16
        cx_max, cz_max = cx_min + lato // 16, cz_min + lato // 16
        totale = ((cx_max - cx_min) // passo + 1) * ((cz_max - cz_min) // passo + 1)
        fatto = 0
        for wx in range(cx_min, cx_max, passo):
            for wz in range(cz_min, cz_max, passo):
                fatto += 1
                if avanza:
                    avanza(fatto / totale)
                blocchi, assente = _leggi_finestra(lv, presenti, wx, wz, passo, DIMENSIONE)
                if blocchi is None:
                    continue
                ids = np.unique(blocchi)
                nomi = {int(i): nome(int(i)) for i in ids}
                vuoto = np.array([nomi.get(i, "") in ("air", "cave_air", "void_air")
                                  for i in range(int(ids.max()) + 1)])
                liquido = np.array([nomi.get(i, "") == "water"
                                    for i in range(int(ids.max()) + 1)])
                aria, acqua = vuoto[blocchi], liquido[blocchi]
                aria &= ~assente[:, None, :]     # fuori dal mondo non e' aria: e' ignoto
                solido = ~aria & ~acqua
                solido[:, 0] = True      # il fondo del mondo e' terra
                ox, oz = wx * 16, wz * 16
                for x, y, z, n in pezzi_sospesi(solido):
                    rap.sospesi.append({"x": ox + x, "y": y + Y_INIZIO, "z": oz + z,
                                        "blocchi": n})
                for x, y, z in acqua_che_trabocca(acqua, aria):
                    rap.acqua.append({"x": ox + x, "y": y + Y_INIZIO, "z": oz + z})
    finally:
        lv.close()
    return rap


def _leggi_finestra(lv, presenti, wx, wz, passo, dimensione):
    """(blocchi [x, y, z], colonne assenti [x, z]) di una finestra di chunk."""
    fuori = None
    assente = np.ones((passo * 16, passo * 16), bool)
    for i in range(passo):
        for j in range(passo):
            if (wx + i, wz + j) not in presenti:
                continue
            ch = lv.get_chunk(wx + i, wz + j, dimensione)
            if fuori is None:
                fuori = np.zeros((passo * 16, 384, passo * 16), np.int64)
            arr = np.asarray(ch.blocks[:, Y_INIZIO:320, :])
            fuori[i * 16:(i + 1) * 16, :arr.shape[1], j * 16:(j + 1) * 16] = arr
            assente[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = False
    return fuori, assente
