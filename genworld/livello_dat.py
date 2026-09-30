"""Costruzione di un level.dat che Minecraft accetti davvero.

Motivo di esistere: amulet-core crea il mondo e scrive i file region
correttamente, ma il level.dat che produce contiene solo quattro campi
(DataVersion, LastPlayed, LevelName, version). Mancano WorldGenSettings,
GameType, Version, lo spawn e i gamerule, e Minecraft rifiuta o rigenera il
mondo. Questo modulo lo riscrive completo dopo la creazione.

Il generatore dell'overworld e' un superflat di sola aria: il terreno lo
scriviamo noi, e cosi' i chunk che non tocchiamo restano vuoti invece di
riempirsi di terreno vanilla a caso.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from amulet_nbt import (ByteTag, CompoundTag, DoubleTag, FloatTag, IntTag,
                        ListTag, LongTag, NamedTag, StringTag)

# DataVersion: NON scriverla a mano. Una tabella compilata a memoria aveva
# gia' due valori sbagliati di un'unita' (1.21.5 e 1.21.8), e un DataVersion
# errato e' il tipo di bug che si manifesta come "il mondo non si apre" senza
# dire perche'. La fonte autorevole e' PyMCTranslate, che arriva con amulet.
_CACHE_DV: dict[tuple, int] = {}


def data_version_di(versione: tuple[int, int, int]) -> int:
    """DataVersion Java per una versione, chiesta a PyMCTranslate."""
    if versione in _CACHE_DV:
        return _CACHE_DV[versione]
    import PyMCTranslate
    tm = PyMCTranslate.new_translation_manager()
    try:
        dv = tm.get_version("java", versione).data_version
    except Exception:
        # ricade sulla piu' alta versione conosciuta non superiore a quella chiesta
        note = sorted(v for v in tm.version_numbers("java") if v <= versione)
        if not note:
            raise ValueError(f"versione Java non supportata: {versione}") from None
        dv = tm.get_version("java", note[-1]).data_version
    _CACHE_DV[versione] = dv
    return dv


def versioni_supportate() -> list[tuple[int, int, int]]:
    import PyMCTranslate
    return sorted(PyMCTranslate.new_translation_manager().version_numbers("java"))

GAMERULE_DEFAULT = {
    "doDaylightCycle": "true",
    "doWeatherCycle": "true",
    "doMobSpawning": "true",
    "keepInventory": "false",
    "mobGriefing": "true",
    "doFireTick": "true",
    "randomTickSpeed": "3",
    "spawnRadius": "10",
}


@dataclass
class ImpostazioniMondo:
    nome: str = "GenWorld"
    versione: tuple[int, int, int] = (1, 21, 4)   # versione di gioco di riferimento
    seed: int = 0
    modalita: int = 1               # 0 sopravvivenza, 1 creativa
    difficolta: int = 2
    comandi: bool = True
    spawn: tuple[int, int, int] = (0, 80, 0)
    gamerule: dict[str, str] = field(default_factory=lambda: dict(GAMERULE_DEFAULT))

    @property
    def data_version(self) -> int:
        return data_version_di(self.versione)

    @property
    def nome_versione(self) -> str:
        return ".".join(str(n) for n in self.versione)


def _generatore_vuoto() -> CompoundTag:
    """Superflat senza strati: il mondo nasce vuoto e lo riempiamo noi."""
    return CompoundTag({
        "type": StringTag("minecraft:flat"),
        "settings": CompoundTag({
            "biome": StringTag("minecraft:plains"),
            "lakes": ByteTag(0),
            "features": ByteTag(0),
            "layers": ListTag([
                CompoundTag({
                    "block": StringTag("minecraft:air"),
                    "height": IntTag(1),
                })
            ]),
            "structure_overrides": ListTag([]),
        }),
    })


def _dimensioni(seed: int) -> CompoundTag:
    return CompoundTag({
        "minecraft:overworld": CompoundTag({
            "type": StringTag("minecraft:overworld"),
            "generator": _generatore_vuoto(),
        }),
        "minecraft:the_nether": CompoundTag({
            "type": StringTag("minecraft:the_nether"),
            "generator": CompoundTag({
                "type": StringTag("minecraft:noise"),
                "settings": StringTag("minecraft:nether"),
                "biome_source": CompoundTag({
                    "type": StringTag("minecraft:multi_noise"),
                    "preset": StringTag("minecraft:nether"),
                }),
            }),
        }),
        "minecraft:the_end": CompoundTag({
            "type": StringTag("minecraft:the_end"),
            "generator": CompoundTag({
                "type": StringTag("minecraft:noise"),
                "settings": StringTag("minecraft:end"),
                "biome_source": CompoundTag({"type": StringTag("minecraft:the_end")}),
            }),
        }),
    })


def costruisci(imp: ImpostazioniMondo) -> NamedTag:
    """Compone il level.dat completo."""
    sx, sy, sz = imp.spawn
    dv = imp.data_version
    ora_ms = int(time.time() * 1000)

    data = CompoundTag({
        # --- identita' e versione -------------------------------------
        "DataVersion": IntTag(dv),
        "version": IntTag(19133),                 # formato Anvil
        "Version": CompoundTag({
            "Id": IntTag(dv),
            "Name": StringTag(imp.nome_versione),
            "Series": StringTag("main"),
            "Snapshot": ByteTag(0),
        }),
        "LevelName": StringTag(imp.nome),
        "initialized": ByteTag(1),
        "WasModded": ByteTag(0),
        "DataPacks": CompoundTag({
            "Enabled": ListTag([StringTag("vanilla")]),
            "Disabled": ListTag([]),
        }),

        # --- partita --------------------------------------------------
        "GameType": IntTag(imp.modalita),
        "Difficulty": ByteTag(imp.difficolta),
        "DifficultyLocked": ByteTag(0),
        "hardcore": ByteTag(0),
        "allowCommands": ByteTag(1 if imp.comandi else 0),
        "raining": ByteTag(0),
        "thundering": ByteTag(0),
        "rainTime": IntTag(12000),
        "thunderTime": IntTag(120000),
        "clearWeatherTime": IntTag(0),
        "Time": LongTag(0),
        "DayTime": LongTag(1000),
        "LastPlayed": LongTag(ora_ms),

        # --- spawn ----------------------------------------------------
        "SpawnX": IntTag(sx),
        "SpawnY": IntTag(sy),
        "SpawnZ": IntTag(sz),
        "SpawnAngle": FloatTag(0.0),

        # --- confini del mondo ---------------------------------------
        "BorderCenterX": DoubleTag(0.0),
        "BorderCenterZ": DoubleTag(0.0),
        "BorderSize": DoubleTag(59999968.0),
        "BorderSafeZone": DoubleTag(5.0),
        "BorderWarningBlocks": DoubleTag(5.0),
        "BorderWarningTime": DoubleTag(15.0),
        "BorderDamagePerBlock": DoubleTag(0.2),
        "BorderSizeLerpTarget": DoubleTag(59999968.0),
        "BorderSizeLerpTime": LongTag(0),

        # --- generazione ---------------------------------------------
        "WorldGenSettings": CompoundTag({
            "seed": LongTag(imp.seed),
            "generate_features": ByteTag(0),
            "bonus_chest": ByteTag(0),
            "dimensions": _dimensioni(imp.seed),
        }),

        # Minecraft lo cerca all'apertura anche se l'End non l'hai mai visto,
        # e senza logga "key missing: DragonFight". Non rompe niente, ma un
        # avviso al caricamento di un mondo appena generato e' comunque un
        # difetto del mondo, non del gioco.
        "DragonFight": CompoundTag({
            "NeedsStateScanning": ByteTag(1),
            "DragonKilled": ByteTag(0),
            "PreviouslyKilled": ByteTag(0),
            "Gateways": ListTag([IntTag(i) for i in range(20)]),
        }),

        "GameRules": CompoundTag({k: StringTag(v) for k, v in imp.gamerule.items()}),
        "ServerBrands": ListTag([StringTag("GenWorld")]),
    })

    return NamedTag(CompoundTag({"Data": data}), "")


def scrivi(percorso_level_dat: str, imp: ImpostazioniMondo) -> None:
    costruisci(imp).save_to(percorso_level_dat, compressed=True)


# --------------------------------------------------------------------------
# Verifica strutturale: nessuno qui puo' lanciare Minecraft, ma possiamo
# controllare che ci sia tutto cio' che Minecraft legge all'apertura.
# --------------------------------------------------------------------------

OBBLIGATORI = {
    "DataVersion": IntTag, "version": IntTag, "Version": CompoundTag,
    "LevelName": StringTag, "initialized": ByteTag, "GameType": IntTag,
    "Difficulty": ByteTag, "hardcore": ByteTag, "allowCommands": ByteTag,
    "LastPlayed": LongTag, "SpawnX": IntTag, "SpawnY": IntTag, "SpawnZ": IntTag,
    "Time": LongTag, "DayTime": LongTag, "WorldGenSettings": CompoundTag,
    "GameRules": CompoundTag, "DataPacks": CompoundTag,
    "DragonFight": CompoundTag,
}


def verifica(percorso_level_dat: str) -> list[str]:
    """Ritorna la lista dei problemi trovati. Lista vuota = struttura a posto."""
    from amulet_nbt import load as nbt_load

    problemi: list[str] = []
    nt = nbt_load(percorso_level_dat)
    root = nt.compound
    if "Data" not in root:
        return ["manca il tag radice 'Data'"]
    d = root["Data"]

    for chiave, tipo in OBBLIGATORI.items():
        if chiave not in d:
            problemi.append(f"manca {chiave}")
        elif not isinstance(d[chiave], tipo):
            problemi.append(f"{chiave} e' {type(d[chiave]).__name__}, atteso {tipo.__name__}")

    wgs = d.get("WorldGenSettings")
    if isinstance(wgs, CompoundTag):
        if "dimensions" not in wgs:
            problemi.append("WorldGenSettings senza 'dimensions'")
        else:
            dims = wgs["dimensions"]
            for nome in ("minecraft:overworld", "minecraft:the_nether", "minecraft:the_end"):
                if nome not in dims:
                    problemi.append(f"dimensione mancante: {nome}")
                    continue
                dim = dims[nome]
                if not isinstance(dim.get("type"), (StringTag, CompoundTag)):
                    problemi.append(f"{nome}: 'type' non e' StringTag/CompoundTag")
                if not isinstance(dim.get("generator"), CompoundTag):
                    problemi.append(f"{nome}: 'generator' mancante o non CompoundTag")
    return problemi
