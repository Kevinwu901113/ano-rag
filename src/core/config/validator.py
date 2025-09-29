from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, Mapping, Tuple

from .schema import DEFAULT_CONFIG
from .utils import iter_paths

Path = Tuple[str, ...]

ALLOWED_PATHS: set[Path] = set(iter_paths(DEFAULT_CONFIG)) | {("config_version",)}
WILDCARD_PREFIXES: Tuple[Path, ...] = (
    ("_diagnostics",),
    ("hybrid_search", "weights"),
    ("hybrid_search", "rrf"),
    ("retrieval", "hybrid"),
    ("retrieval", "hybrid", "weights"),
    ("retrieval", "bm25", "must_have_terms"),
    ("retrieval", "bm25", "boost_entities"),
    ("retrieval", "bm25", "boost_predicates"),
    ("retrieval", "json_parsing"),
    ("retrieval", "performance"),
    ("llm", "providers"),
    ("llm", "hybrid_llm"),
    ("rerank",),
)


def validate_and_check_unknowns(cfg: Dict[str, Any], *, strict: bool | None = None) -> None:
    unknown_paths = sorted({".".join(path) for path in _find_unknown_paths(cfg)})
    diagnostics = cfg.setdefault("_diagnostics", {})
    strict_mode = _determine_strict_mode(strict)
    if unknown_paths:
        diagnostics["unknown_keys"] = unknown_paths
        message = f"Unknown configuration keys detected: {unknown_paths}"
        if strict_mode:
            raise ValueError(message)
        logging.getLogger(__name__).warning(message)
    else:
        diagnostics.setdefault("unknown_keys", [])


ALLOWED_PATHS.add(("hybrid_search", "rrf_k"))


def _find_unknown_paths(data: Mapping[str, Any], prefix: Path = ()) -> Iterable[Path]:
    for key, value in data.items():
        current = prefix + (key,)
        if _is_wildcard_prefix(current):
            if isinstance(value, Mapping):
                yield from _find_unknown_paths(value, current)
            continue
        if current not in ALLOWED_PATHS:
            yield current
            continue
        if isinstance(value, Mapping):
            yield from _find_unknown_paths(value, current)


def _is_wildcard_prefix(path: Path) -> bool:
    return any(path[: len(prefix)] == prefix for prefix in WILDCARD_PREFIXES)


def _determine_strict_mode(strict: bool | None) -> bool:
    if strict is not None:
        return strict

    env_override = os.getenv("ANO_RAG_CONFIG_STRICT")
    if env_override is not None:
        value = env_override.strip().lower()
        if value in {"0", "false", "no", "off", "lenient", "warn", "warning"}:
            return False
        if value in {"1", "true", "yes", "on", "strict", "fail", "error"}:
            return True

    runtime_env = (os.getenv("ANO_RAG_ENV") or os.getenv("ENVIRONMENT") or "development").lower()
    if runtime_env in {"production", "prod", "release"}:
        return False
    return True
