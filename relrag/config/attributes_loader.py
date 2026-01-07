from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_ATTR_PATH = Path(__file__).resolve().parent / "attributes.yaml"


@functools.lru_cache(maxsize=1)
def load_attributes_config() -> Dict[str, Dict[str, Any]]:
    if not _ATTR_PATH.exists():
        return {}
    with open(_ATTR_PATH, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    attrs = data.get("attributes")
    if not isinstance(attrs, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, payload in attrs.items():
        if not isinstance(payload, dict):
            continue
        normalized[name.strip().lower()] = payload
    return normalized


def get_attribute_config(name: Optional[str]) -> Dict[str, Any]:
    if not name:
        return {}
    return load_attributes_config().get(name.strip().lower(), {})


def get_selection_priority(name: Optional[str]) -> List[str]:
    cfg = get_attribute_config(name)
    values = cfg.get("selection_priority")
    if isinstance(values, list):
        return [str(v).strip().lower() for v in values if isinstance(v, str) and v.strip()]
    return []


def allowed_values(name: Optional[str]) -> List[str]:
    cfg = get_attribute_config(name)
    lexicon = cfg.get("value_lexicon")
    if isinstance(lexicon, list):
        return [str(v).strip() for v in lexicon if isinstance(v, str) and v.strip()]
    return []


def alias_map(name: Optional[str]) -> Dict[str, str]:
    cfg = get_attribute_config(name)
    aliases = cfg.get("value_aliases")
    if isinstance(aliases, dict):
        return {str(k).strip().lower(): str(v).strip() for k, v in aliases.items() if isinstance(k, str) and isinstance(v, str)}
    return {}


def negative_verbs(name: Optional[str]) -> List[str]:
    cfg = get_attribute_config(name)
    verbs = cfg.get("negative_verbs")
    if isinstance(verbs, list):
        return [str(v).strip().lower() for v in verbs if isinstance(v, str) and v.strip()]
    return []
