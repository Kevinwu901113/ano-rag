from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

from .aliases import ALIASES, DEPRECATED, LATEST_VERSION
from .utils import delete_path, get_path, set_path

Path = Tuple[str, ...]


def migrate_to_latest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    migrated = deepcopy(cfg)
    version = int(migrated.get("config_version", 1))

    while version < LATEST_VERSION:
        version += 1
        # Future version specific migrations can be inserted here.

    migrated["config_version"] = LATEST_VERSION
    _move_aliases(migrated)
    _record_deprecated_hits(migrated)
    return migrated


def _move_aliases(cfg: Dict[str, Any]) -> None:
    for target, sources in ALIASES.items():
        current = get_path(cfg, target)
        if current is None:
            for alias in sources:
                alias_value = get_path(cfg, alias)
                if alias_value is not None:
                    set_path(cfg, target, deepcopy(alias_value))
                    break
        # Remove legacy aliases so downstream only sees canonical keys.
        for alias in sources:
            delete_path(cfg, alias)


def _record_deprecated_hits(cfg: Dict[str, Any]) -> None:
    hits: list[str] = []
    for path in DEPRECATED:
        if get_path(cfg, path) is not None:
            hits.append(".".join(path))
            delete_path(cfg, path)
    if hits:
        diagnostics = cfg.setdefault("_diagnostics", {})
        deprecated = set(diagnostics.get("deprecated_paths", []))
        diagnostics["deprecated_paths"] = sorted(deprecated | set(hits))
