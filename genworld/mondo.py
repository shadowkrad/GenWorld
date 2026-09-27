"""Scrittura di mondi Minecraft Java Edition (formato Anvil).

Incapsula amulet-core e ne corregge il difetto trovato nello spike: il
level.dat prodotto da `create_and_open` e' un moncone, quindi lo riscriviamo
noi (vedi `livello_dat`).

Regola di performance stabilita in architettura: si lavora **per chunk in
streaming**, costruendo un array (16, H, 16) alla volta. Non si tiene mai
l'intero mondo in memoria.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterator

import numpy as np

import amulet
from amulet.api.block import Block
from amulet_nbt import StringTag
from amulet.api.chunk import Chunk
from amulet.api.selection import SelectionBox, SelectionGroup
from amulet.level.formats.anvil_world import AnvilFormat

from .livello_dat import ImpostazioniMondo
from . import livello_dat

Y_MIN = -64
Y_MAX = 320
DIMENSIONE = "minecraft:overworld"
ARIA = "air"


class ScrittoreMondo:
    """Crea un mondo e ci scrive dentro chunk per chunk.

    Uso:
        with ScrittoreMondo("/percorso/mondo", imp, lato_blocchi=256) as m:
            for cx, cz in m.chunk_coords():
                m.scrivi_chunk(cx, cz, array_id_blocchi, y0=-64)
    """

    def __init__(
        self,
        percorso: str,
        impostazioni: ImpostazioniMondo | None = None,
        lato_blocchi: int = 256,
        sovrascrivi: bool = True,
        crea: bool = True,
    ) -> None:
        self.percorso = percorso
        self.imp = impostazioni or ImpostazioniMondo()
        self.lato = lato_blocchi
        self.sovrascrivi = sovrascrivi
        # crea=False riapre un mondo gia' esistente: serve a generare una mappa
        # grande in piu' esecuzioni, un lotto di chunk alla volta.
        self.crea = crea
        self._livello = None
        self._cache_blocchi: dict[tuple, int] = {}
        self._cache_biomi: dict[str, int] = {}
        self._chunk_scritti = 0

    # -- ciclo di vita ------------------------------------------------------

    def __enter__(self) -> "ScrittoreMondo":
        if not self.crea:
            if not os.path.exists(self.percorso):
                raise FileNotFoundError(f"non esiste: {self.percorso}")
            self._livello = amulet.load_level(self.percorso)
            return self

        if os.path.exists(self.percorso):
            if not self.sovrascrivi:
                raise FileExistsError(f"esiste gia': {self.percorso}")
            shutil.rmtree(self.percorso)

        meta = self.lato // 2
        fmt = AnvilFormat(self.percorso)
        fmt.create_and_open(
            "java",
            self.imp.versione,
            bounds=SelectionGroup(SelectionBox((-meta, Y_MIN, -meta), (meta, Y_MAX, meta))),
            overwrite=True,
        )
        fmt.close()

        # Il level.dat va rifatto ADESSO, non alla chiusura.
        #
        # Quello che scrive amulet ha un `WorldGenSettings` che amulet stesso
        # non sa rileggere ("...["type"] was not a StringTag or CompoundTag"),
        # e quando non lo sa rileggere ripiega sui limiti di prima della 1.18:
        # y da 0 a 256. Il livello aperto qui sotto ereditava quei limiti e
        # **buttava via in silenzio tutto quello che stava sotto y=0**: niente
        # bedrock a -64, niente ardesia, niente caverne profonde, niente
        # minerali del fondo. In memoria i sub-chunk da -4 a -1 c'erano, sul
        # disco no, e rileggendo il mondo finito i limiti erano di nuovo
        # giusti - perche' nel frattempo il level.dat l'avevamo rifatto noi -
        # quindi il controllo di andata e ritorno non vedeva niente.
        #
        # Scrivendolo prima, `load_level` legge i limiti veri (-64..320).
        livello_dat.scrivi(os.path.join(self.percorso, "level.dat"), self.imp)

        self._livello = amulet.load_level(self.percorso)
        limiti = self._livello.bounds(DIMENSIONE).min
        if limiti[1] > Y_MIN:
            raise RuntimeError(
                f"il livello si e' aperto con y minimo {limiti[1]} invece di "
                f"{Y_MIN}: tutto il sottosuolo andrebbe perso")
        return self

    def __exit__(self, *exc) -> None:
        if self._livello is not None:
            self._livello.save()
            self._livello.close()
            self._livello = None
        # amulet scrive un level.dat incompleto: lo rifacciamo da zero.
        livello_dat.scrivi(os.path.join(self.percorso, "level.dat"), self.imp)
        self._butta_le_entita()
        lock = os.path.join(self.percorso, "session.lock")
        if os.path.exists(lock):
            os.remove(lock)

    def _butta_le_entita(self) -> None:
        """Toglie la cartella `entities/` che amulet scrive rotta.

        Per ogni chunk amulet produce un chunk-entita' di 22 byte con dentro
        solo `DataVersion: 0`: niente `Position`, niente `Entities`. Minecraft
        legge la posizione da quell'array e trova un array vuoto, quindi

            Failed to parse chunk [x, z] position info
            java.lang.ArrayIndexOutOfBoundsException: Index 0 out of bounds
                                                      for length 0

        e poi "Failed to load chunk x,z", a ripetizione, in sovrimpressione
        sullo schermo. Il terreno si vede lo stesso - sono due archivi
        separati - ed e' per questo che il difetto e' sopravvissuto a tutti i
        controlli fatti finora: rileggendo i file region tornava tutto, e i
        file region non erano il problema.

        I nostri mondi non contengono entita', quindi la cartella giusta e'
        nessuna cartella: Minecraft la ricrea lui alla prima entita' che
        salva. Undici megabyte in meno, per giunta.
        """
        cartella = os.path.join(self.percorso, "entities")
        if os.path.isdir(cartella):
            shutil.rmtree(cartella)

    # -- palette ------------------------------------------------------------

    def blocco(self, nome: str, **proprieta: str) -> int:
        """Id nel palette del livello per un blocco, con le sue proprieta'.

        Le proprieta' NON sono un optional. Un tronco scritto come
        `universal_minecraft:oak_log` viene salvato senza errori e si rilegge
        identico dal file region, ma PyMCTranslate non sa tradurlo verso
        Minecraft: in gioco non compare. La forma giusta e'
        `log[axis="y", material="oak", stripped="false"]`, e lo stesso vale
        per foglie, erba e cespugli. E' un errore che si scopre solo aprendo
        il mondo, perche' il controllo di andata e ritorno lo supera.
        """
        chiave = (nome, tuple(sorted(proprieta.items())))
        if chiave not in self._cache_blocchi:
            self._cache_blocchi[chiave] = self._livello.block_palette.get_add_block(
                Block("universal_minecraft", nome,
                      {k: StringTag(v) for k, v in proprieta.items()})
            )
        return self._cache_blocchi[chiave]

    @property
    def id_aria(self) -> int:
        return self.blocco(ARIA)

    def bioma(self, nome: str) -> int:
        """Id nel palette dei biomi. Stessa trappola dei blocchi: il nome va
        dato nella forma UNIVERSALE (`snowy_tundra`, non `snowy_plains`), e
        un nome sconosciuto non da' errore, semplicemente non arriva in
        gioco. `genworld.biomi.verifica_traduzioni()` fa da guardia."""
        if nome not in self._cache_biomi:
            self._cache_biomi[nome] = self._livello.biome_palette.get_add_biome(
                f"universal_minecraft:{nome}")
        return self._cache_biomi[nome]

    # -- geometria ----------------------------------------------------------

    @property
    def chunk_per_lato(self) -> int:
        return self.lato // 16

    def chunk_coords(self) -> Iterator[tuple[int, int]]:
        """Coordinate chunk del mondo, centrate sull'origine."""
        meta = self.chunk_per_lato // 2
        for cx in range(-meta, meta):
            for cz in range(-meta, meta):
                yield cx, cz

    def origine_chunk(self, cx: int, cz: int) -> tuple[int, int]:
        """Coordinate blocco dell'angolo del chunk."""
        return cx * 16, cz * 16

    # -- scrittura ----------------------------------------------------------

    def scrivi_chunk(
        self,
        cx: int,
        cz: int,
        blocchi: np.ndarray,
        y0: int = Y_MIN,
        biomi: np.ndarray | None = None,
    ) -> None:
        """Scrive un chunk da un array (16, H, 16) di id di palette.

        `biomi` e' un array (16, 16) di id del palette dei biomi, nello
        stesso ordine [x, z] dei blocchi. Minecraft memorizza i biomi a
        blocchi di 4, quindi amulet ne tiene uno ogni quattro colonne: il
        dettaglio sotto i 4 blocchi si perde comunque.
        """
        if blocchi.shape[0] != 16 or blocchi.shape[2] != 16:
            raise ValueError(f"atteso array (16, H, 16), ricevuto {blocchi.shape}")

        chunk = Chunk(cx, cz)
        chunk.blocks[:, y0:y0 + blocchi.shape[1], :] = blocchi.astype(np.uint32)

        if biomi is not None:
            if biomi.shape != (16, 16):
                raise ValueError(f"atteso array (16, 16), ricevuto {biomi.shape}")
            chunk.biomes = np.ascontiguousarray(biomi, dtype=np.uint32)
            # il formato su disco e' 3D: senza questa conversione amulet
            # salverebbe la forma 2D dei mondi vecchi
            chunk.biomes.convert_to_3d()

        chunk.changed = True
        self._livello.put_chunk(chunk, DIMENSIONE)
        self._chunk_scritti += 1

        # scarico periodico: non teniamo tutto in RAM
        if self._chunk_scritti % 192 == 0:
            self._livello.save()

    @property
    def chunk_scritti(self) -> int:
        return self._chunk_scritti


def colonne_da_altezze(
    altezze: np.ndarray,
    scrittore: ScrittoreMondo,
    livello_mare: int = 62,
    y0: int = Y_MIN,
    y1: int = 200,
) -> np.ndarray:
    """Da una heightmap (16, 16) all'array di blocchi (16, H, 16).

    Vettoriale: niente cicli per colonna. Le regole di superficie sono ancora
    quelle dello spike, non la legenda vera: servono a verificare la scrittura.
    """
    h = altezze.astype(np.int32)[:, None, :]          # (16, 1, 16)
    ys = np.arange(y0, y1, dtype=np.int32)[None, :, None]   # (1, H, 1)

    aria = scrittore.id_aria
    out = np.full((16, y1 - y0, 16), aria, dtype=np.uint32)

    solido = ys < h
    out[solido] = scrittore.blocco("stone")

    # bedrock
    out[:, 0, :] = scrittore.blocco("bedrock")

    # strato superficiale: i 4 blocchi sotto la cima
    sottosuolo = solido & (ys >= h - 4)
    out[sottosuolo] = scrittore.blocco("dirt")

    cima = ys == h - 1
    sopra_mare = cima & (h > livello_mare + 1)
    spiaggia = cima & (h <= livello_mare + 1)
    roccia = cima & (h > livello_mare + 70)
    out[sopra_mare] = scrittore.blocco("grass_block")
    out[spiaggia] = scrittore.blocco("sand")
    out[roccia] = scrittore.blocco("stone")

    # acqua fino al livello del mare
    acqua = (~solido) & (ys <= livello_mare)
    out[acqua] = scrittore.blocco("water")

    return out
