from __future__ import annotations

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
)


def validate_and_check_unknowns(cfg: Dict[str, Any]) -> None:
    unknown_paths = sorted({".".join(path) for path in _find_unknown_paths(cfg)})
    diagnostics = cfg.setdefault("_diagnostics", {})
    if unknown_paths:
        diagnostics["unknown_keys"] = unknown_paths
        raise ValueError(f"Unknown configuration keys detected: {unknown_paths}")
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
