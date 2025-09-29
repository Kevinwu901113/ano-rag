from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from .freeze import FrozenConfig, freeze
from .migrator import migrate_to_latest
from .normalizer import normalize
from .printer import format_diagnostics
from .schema import RootConfig, DEFAULT_CONFIG
from .utils import deep_merge, expand_env
from .validator import validate_and_check_unknowns

PRIORITY: Tuple[str, ...] = (
    "configs/base.yaml",
    "configs/profiles/{PROFILE}.yaml",
    "configs/local.yaml",
)


def load_config(
    profile: str | None = None,
    extra_layers: Iterable[str] | None = None,
    inline_overrides: Dict[str, Any] | None = None,
) -> Tuple[FrozenConfig, RootConfig, Dict[str, Any], List[str]]:
    profile = profile or "default"
    merged: Dict[str, Any] = {}
    resolved_layers: List[str] = []

    for template in PRIORITY:
        path = Path(template.format(PROFILE=profile))
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            layer = yaml.safe_load(handle) or {}
        layer = expand_env(layer)
        merged = deep_merge(merged, layer)
        resolved_layers.append(str(path))

    for extra in extra_layers or []:
        extra_path = Path(extra)
        if not extra_path.exists():
            continue
        with extra_path.open("r", encoding="utf-8") as handle:
            layer = yaml.safe_load(handle) or {}
        layer = expand_env(layer)
        merged = deep_merge(merged, layer)
        resolved_layers.append(str(extra_path))

    if inline_overrides:
        merged = deep_merge(merged, inline_overrides)

    raw = deepcopy(merged)

    # Ensure defaults are always present before migrations/normalization.
    merged = deep_merge(DEFAULT_CONFIG, merged)

    migrated = migrate_to_latest(merged)
    normalized = normalize(migrated, raw)
    validate_and_check_unknowns(normalized)
    model = RootConfig.model_validate(normalized)
    frozen = freeze(model)
    return frozen, model, normalized, resolved_layers


def print_diagnostics(model: RootConfig) -> str:
    diagnostics = model.model_dump().get("_diagnostics", {})
    return format_diagnostics(diagnostics)
