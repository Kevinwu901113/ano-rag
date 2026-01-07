from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

from relrag.config.attributes_loader import load_attributes_config


_DATA_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_vocab() -> Dict[str, Dict[str, Dict]]:
    path = _DATA_DIR / "vocab.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_alias_overrides() -> Dict[str, Dict[str, str]]:
    path = _DATA_DIR / "aliases.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
        lowered: Dict[str, Dict[str, str]] = {}
        for slot, mapping in data.items():
            lowered[slot] = {k.lower(): v for k, v in mapping.items()}
        return lowered


@lru_cache(maxsize=1)
def _compiled_slot_maps() -> Dict[str, Dict[str, str]]:
    compiled: Dict[str, Dict[str, str]] = {}
    vocab = load_vocab()
    for slot, entries in vocab.items():
        slot_map: Dict[str, str] = {}
        for canonical, meta in entries.items():
            canon_value = (meta.get("canonical") or canonical).strip()
            if not canon_value:
                continue
            slot_map[canonical.lower()] = canon_value
            for alias in meta.get("aliases", []):
                alias_norm = alias.strip().lower()
                if alias_norm:
                    slot_map[alias_norm] = canon_value
        compiled[slot] = slot_map

    overrides = load_alias_overrides()
    for slot, mapping in overrides.items():
        slot_map = compiled.setdefault(slot, {})
        for alias, canonical in mapping.items():
            alias_norm = alias.strip().lower()
            if not alias_norm:
                continue
            slot_map[alias_norm] = canonical

    # Inject dynamic attribute config (value lexicon + aliases)
    attr_cfg = load_attributes_config()
    for slot, payload in attr_cfg.items():
        if not isinstance(payload, dict):
            continue
        slot_map = compiled.setdefault(slot, {})
        lexicon = payload.get("value_lexicon") or []
        for entry in lexicon:
            if not isinstance(entry, str):
                continue
            canon = entry.strip()
            if not canon:
                continue
            slot_map[canon.lower()] = canon
        aliases = payload.get("value_aliases") or {}
        if isinstance(aliases, dict):
            for alias, canonical in aliases.items():
                if not isinstance(alias, str) or not isinstance(canonical, str):
                    continue
                alias_norm = alias.strip().lower()
                canon = canonical.strip()
                if alias_norm and canon:
                    slot_map[alias_norm] = canon
    return compiled


def normalize_slot_value(slot: str, value: str) -> Tuple[str, bool]:
    """Return canonical form for a slot value and whether an alias rule fired."""
    if not value:
        return value, False
    slot_map = _compiled_slot_maps().get(slot, {})
    lowered = value.strip().lower()
    canonical = slot_map.get(lowered)
    if canonical:
        return canonical, True
    return value.strip(), False


def normalize_entity_name(name: str) -> Tuple[str, bool]:
    """Canonicalise entity surface forms using the `entity` alias slot when possible."""
    if not name:
        return name, False
    overrides = load_alias_overrides().get("entity", {})
    lowered = name.strip().lower()
    canonical = overrides.get(lowered)
    if canonical:
        return canonical, True
    return name.strip(), False
