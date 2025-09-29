from __future__ import annotations

from typing import Dict, Iterable, Tuple

Path = Tuple[str, ...]

LATEST_VERSION = 3

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
}

DEPRECATED: Tuple[Path, ...] = (
    ("hybrid_search", "multi_hop", "per_hop_keep_top_m"),
    ("hybrid_search", "multi_hop", "lower_threshold"),
    ("hybrid_search", "safety"),
    ("retrieval", "vector", "top_k"),
)
