"""Migrate legacy configuration files to the new unified skeleton."""
import sys
import yaml
import json
from pathlib import Path
from typing import Any, Dict

# Ensure repository root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config_loader import DEFAULT_CONFIG

OLD_NEW_MAP = {
    "system.project_name": "system.project_name",
    "system.seed": "system.seed",
    "system.device": "system.device",
    "document.chunk_size": "document.chunk_size",
    "document.overlap": "document.overlap",
    "embedding.model_name": "embedding.model_name",
    "embedding.batch_size": "embedding.batch_size",
    "embedding.max_length": "embedding.max_length",
    "embedding.normalize": "embedding.normalize",
    "hybrid_search.enabled": "retrieval.hybrid.enabled",
    "hybrid_search.fusion_method": "retrieval.hybrid.fusion_method",
    "hybrid_search.vector_weight": "retrieval.hybrid.weights.dense",
    "hybrid_search.bm25_weight": "retrieval.hybrid.weights.bm25",
    "hybrid_search.path_weight": "retrieval.hybrid.weights.path",
    "hybrid_search.rrf_k": "retrieval.hybrid.rrf_k",
    "hybrid_search.bm25_k1": "retrieval.bm25.k1",
    "hybrid_search.bm25_b": "retrieval.bm25.b",
    "hybrid_search.corpus_field": "retrieval.bm25.text_field",
    "graph.k_hop": "retrieval.graph.k_hop",
    "graph.expand_top_m": "retrieval.graph.expand_top_m",
    "path_aware_ranker.min_path_score": "path_aware.min_path_score",
    "path_aware_ranker.enabled": "path_aware.enabled",
    "context_dispatcher.final_semantic_count": "dispatcher.final_semantic_count",
    "context_dispatcher.final_graph_count": "dispatcher.final_graph_count",
    "context_dispatcher.bridge_policy": "dispatcher.bridge_policy",
    "context_dispatcher.bridge_boost_epsilon": "dispatcher.bridge_boost_epsilon",
    "context_dispatcher.debug_log": "dispatcher.debug_log",
    "llm.provider": "llm.provider",
    "llm.temperature": "llm.temperature",
    "llm.max_output_tokens": "llm.max_output_tokens",
    "retrieval_guardrail.enabled": "guardrail.enabled",
    "retrieval_guardrail.min_results": "guardrail.min_results",
    "retrieval_guardrail.min_score": "guardrail.min_score",
    "retrieval_guardrail.timeout_seconds": "guardrail.timeout_seconds",
}


def get_path(data: Dict[str, Any], parts: list[str]) -> Any:
    cur = data
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def set_path(data: Dict[str, Any], parts: list[str], value: Any):
    cur = data
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def flatten(data: Dict[str, Any], prefix: str = "") -> list[str]:
    paths = []
    for k, v in data.items():
        new_prefix = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            paths.extend(flatten(v, new_prefix))
        else:
            paths.append(new_prefix)
    return paths


def migrate(old_path: Path, new_path: Path):
    with open(old_path, "r", encoding="utf-8") as f:
        old_cfg = yaml.safe_load(f) or {}
    new_cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    report = {"migrated": {}, "deprecated": [], "defaulted": []}

    for old_key, new_key in OLD_NEW_MAP.items():
        val = get_path(old_cfg, old_key.split("."))
        if val is not None:
            set_path(new_cfg, new_key.split("."), val)
            report["migrated"][old_key] = new_key

    old_paths = flatten(old_cfg)
    for p in old_paths:
        if p not in OLD_NEW_MAP:
            report["deprecated"].append(p)

    default_paths = flatten(DEFAULT_CONFIG)
    for p in default_paths:
        val = get_path(new_cfg, p.split("."))
        default_val = get_path(DEFAULT_CONFIG, p.split("."))
        if val == default_val and p not in report["migrated"].values():
            report["defaulted"].append(p)

    with open(new_path, "w", encoding="utf-8") as f:
        yaml.dump(new_cfg, f, allow_unicode=True)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/migrate_config.py <old_config> <output_config>")
        sys.exit(1)
    migrate(Path(sys.argv[1]), Path(sys.argv[2]))
