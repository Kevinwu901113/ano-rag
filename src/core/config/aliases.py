from typing import Dict, Iterable, Tuple

Path = Tuple[str, ...]

LATEST_VERSION = 3

# Canonical keys -> legacy alias paths。系统会读取 canonical
# 的键，如果不存在则 fallback 到 alias 列表中的键。
ALIASES: Dict[Path, Iterable[Path]] = {
    ("hybrid_search",): [
        ("retrieval", "hybrid"),
    ],
    ("retrieval", "hybrid"): [
        ("hybrid_search",),
    ],
    ("hybrid_search", "bm25"): [
        ("retrieval", "bm25"),
    ],
    ("retrieval", "bm25"): [
        ("hybrid_search", "bm25"),
    ],
    ("retrieval", "multi_hop"): [
        ("multi_hop",),
    ],
    ("safety", "per_hop_keep_top_m"): [
        ("hybrid_search", "multi_hop", "per_hop_keep_top_m"),
    ],
    ("safety", "lower_threshold"): [
        ("hybrid_search", "multi_hop", "lower_threshold"),
    ],
    ("safety",): [
        ("hybrid_search", "safety"),
    ],
    ("vector_store", "top_k"): [
        ("retrieval", "vector", "top_k"),
    ],
    ("retrieval", "vector", "top_k"): [
        ("vector_store", "top_k"),
    ],
    # 新增：统一调度及旧路径别名
    ("scheduler", "coverage_guard"): [
        ("dispatcher", "scheduler", "coverage_guard"),
    ],
    ("context",): [
        ("retrieval", "context"),
    ],
    ("rerank",): [
        ("retrieval", "rerank"),
        ("hybrid_search", "rerank"),
    ],
    ("multi_hop",): [
        ("calibration", "multi_hop"),
    ],
    ("hybrid_search", "fusion_method"): [
        ("hybrid_search", "fusion", "method"),
    ],
    ("hybrid_search", "linear", "vector_weight"): [
        ("hybrid_search", "fusion", "dense_weight"),
    ],
    ("hybrid_search", "linear", "bm25_weight"): [
        ("hybrid_search", "fusion", "bm25_weight"),
    ],
    ("hybrid_search", "linear", "path_weight"): [
        ("hybrid_search", "fusion", "path_weight"),
        ("hybrid_search", "features", "focused_weight_hop2"),
    ],
}

# 需要提示用户迁移的旧路径（系统在加载后会删除这些键并记录诊断）
DEPRECATED: Tuple[Path, ...] = (
    ("hybrid_search", "multi_hop", "per_hop_keep_top_m"),
    ("hybrid_search", "multi_hop", "lower_threshold"),
    ("hybrid_search", "safety"),
    ("retrieval", "vector", "top_k"),
    ("dispatcher", "scheduler", "coverage_guard"),
    ("retrieval", "context"),
    ("retrieval", "rerank"),
    ("hybrid_search", "rerank"),
    ("calibration", "multi_hop"),
    ("hybrid_search", "fusion", "method"),
    ("hybrid_search", "fusion", "dense_weight"),
    ("hybrid_search", "fusion", "bm25_weight"),
    ("hybrid_search", "fusion", "path_weight"),
    ("hybrid_search", "features", "focused_weight_hop2"),
)
