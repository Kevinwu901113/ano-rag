from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from .schema import DEFAULT_CONFIG
from .utils import deep_merge, get_path, set_path


def normalize(cfg: Dict[str, Any], raw: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    data = deepcopy(cfg)
    raw_data: Mapping[str, Any] = raw or {}
    diagnostics = data.setdefault("_diagnostics", {})
    diagnostics.setdefault("deprecated_paths", [])
    diagnostics.setdefault("unknown_keys", [])
    diagnostics.setdefault("missing_required", [])

    _synchronize_dispatcher(data, raw_data)
    _synchronize_hybrid_branch(data, raw_data)
    _synchronize_multi_hop(data, raw_data)
    _ensure_storage_defaults(data)
    return data


def _synchronize_dispatcher(config_data: Dict[str, Any], raw_data: Mapping[str, Any]) -> None:
    dispatcher_default = deepcopy(DEFAULT_CONFIG.get("dispatcher", {}))
    context_default = deepcopy(DEFAULT_CONFIG.get("context_dispatcher", {}))

    dispatcher = get_path(config_data, ("dispatcher",)) or {}
    context = get_path(config_data, ("context_dispatcher",)) or {}

    merged = deep_merge(dispatcher_default, dispatcher)
    context_merged = deep_merge(context_default, context)
    final = deep_merge(merged, context_merged)

    set_path(config_data, ("dispatcher",), deepcopy(final))
    set_path(config_data, ("context_dispatcher",), deepcopy(final))


def _synchronize_hybrid_branch(config_data: Dict[str, Any], raw_data: Mapping[str, Any]) -> None:
    hybrid_default = deepcopy(DEFAULT_CONFIG.get("hybrid_search", {}))
    hybrid = get_path(config_data, ("hybrid_search",)) or {}
    retrieval_hybrid = get_path(config_data, ("retrieval", "hybrid")) or {}

    merged = deep_merge(hybrid_default, hybrid)
    merged = deep_merge(merged, retrieval_hybrid)

    normalized = deepcopy(merged)
    linear = normalized.get("linear", {}) or {}
    weights = normalized.get("weights", {}) or {}

    if not linear and weights:
        linear = {
            "vector_weight": weights.get("dense", 1.0),
            "bm25_weight": weights.get("bm25", 0.5),
            "path_weight": weights.get("path", 0.1),
        }
    else:
        weights = dict(weights)
        if "vector_weight" in linear:
            weights["dense"] = linear["vector_weight"]
        if "bm25_weight" in linear:
            weights["bm25"] = linear["bm25_weight"]
        if "path_weight" in linear:
            weights["path"] = linear["path_weight"]
        if "graph" not in weights:
            weights["graph"] = DEFAULT_CONFIG["hybrid_search"].get("weights", {}).get("graph", 0.5)

    normalized["linear"] = linear
    normalized["weights"] = weights

    rrf = normalized.get("rrf", {}) or {}
    if "k" not in rrf:
        rrf["k"] = normalized.get("rrf_k", DEFAULT_CONFIG["hybrid_search"].get("rrf", {}).get("k", 60))
    normalized["rrf"] = rrf
    normalized["rrf_k"] = rrf.get("k")

    set_path(config_data, ("hybrid_search",), deepcopy(normalized))
    retrieval = get_path(config_data, ("retrieval",)) or {}
    retrieval = deep_merge(retrieval, {"hybrid": normalized})
    set_path(config_data, ("retrieval",), retrieval)


def _synchronize_multi_hop(config_data: Dict[str, Any], raw_data: Mapping[str, Any]) -> None:
    default_multi = deepcopy(DEFAULT_CONFIG.get("multi_hop", {}))
    retrieval_multi_default = deepcopy(DEFAULT_CONFIG.get("retrieval", {}).get("multi_hop", {}))

    legacy_multi = get_path(config_data, ("multi_hop",)) or {}
    retrieval_multi = get_path(config_data, ("retrieval", "multi_hop")) or {}

    raw_legacy = get_path(raw_data, ("multi_hop",)) or {}
    raw_retrieval = get_path(raw_data, ("retrieval", "multi_hop")) or {}

    merged = deep_merge(default_multi, retrieval_multi_default)
    merged = deep_merge(merged, retrieval_multi)
    merged = deep_merge(merged, legacy_multi)
    merged = deep_merge(merged, raw_legacy)
    merged = deep_merge(merged, raw_retrieval)

    set_path(config_data, ("multi_hop",), deepcopy(merged))
    retrieval = get_path(config_data, ("retrieval",)) or {}
    retrieval = deep_merge(retrieval, {"multi_hop": merged})
    set_path(config_data, ("retrieval",), retrieval)


def _ensure_storage_defaults(config_data: Dict[str, Any]) -> None:
    storage_default = deepcopy(DEFAULT_CONFIG.get("storage", {}))
    storage = get_path(config_data, ("storage",)) or {}
    merged = deep_merge(storage_default, storage)
    set_path(config_data, ("storage",), merged)
