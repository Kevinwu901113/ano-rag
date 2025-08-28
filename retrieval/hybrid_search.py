"""Simplified hybrid retrieval and fusion layer.

This module computes a single final_similarity score by fusing scores
from dense, bm25 and graph retrievers. Optional path scores are injected
additively. All weights and fusion strategy are read from
``retrieval.hybrid`` in the configuration.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any


class HybridSearcher:
    """Fuse scores from multiple retrievers according to configuration."""

    def __init__(self, config: Dict[str, Any]):
        cfg = config if isinstance(config, dict) else config.load_config()
        r_cfg = cfg.get("retrieval", {})
        h_cfg = r_cfg.get("hybrid", {})
        self.candidate_pool = r_cfg.get("candidate_pool", 50)
        self.enabled = h_cfg.get("enabled", True)
        self.fusion_method = h_cfg.get("fusion_method", "linear")
        self.weights = h_cfg.get("weights", {})
        self.rrf_k = h_cfg.get("rrf_k", 60)

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values())
        if max_score == 0:
            return {k: 0.0 for k in scores}
        return {k: v / max_score for k, v in scores.items()}

    def fuse(
        self,
        dense: List[Tuple[str, float]] | None = None,
        bm25: List[Tuple[str, float]] | None = None,
        graph: List[Tuple[str, float]] | None = None,
        path: List[Tuple[str, float]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Fuse scores and produce final similarity.

        Each argument is a list of ``(note_id, score)`` tuples. Missing lists
        are treated as empty.
        """
        if not self.enabled:
            return []

        dense = dense or []
        bm25 = bm25 or []
        graph = graph or []
        path = path or []

        sources = {
            "dense": {nid: s for nid, s in dense},
            "bm25": {nid: s for nid, s in bm25},
            "graph": {nid: s for nid, s in graph},
            "path": {nid: s for nid, s in path},
        }

        note_ids = set().union(*[set(d.keys()) for d in sources.values()])

        results: List[Dict[str, Any]] = []
        if self.fusion_method == "rrf":
            ranks: Dict[str, float] = {}
            for key in ["dense", "bm25", "graph"]:
                sorted_ids = sorted(sources[key].items(), key=lambda x: x[1], reverse=True)
                for rank, (nid, _) in enumerate(sorted_ids, start=1):
                    weight = self.weights.get(key, 0.0)
                    ranks.setdefault(nid, 0.0)
                    ranks[nid] += weight / (self.rrf_k + rank)
            for nid, score in ranks.items():
                final = score + self.weights.get("path", 0.0) * sources["path"].get(nid, 0.0)
                results.append({
                    "note_id": nid,
                    "scores": {k: sources[k].get(nid) for k in ["dense", "bm25", "graph", "path"]},
                    "final_similarity": final,
                    "tags": {
                        "source": "graph" if nid in sources["graph"] else "semantic",
                        "is_bridge": nid in sources["path"],
                    },
                })
        else:
            normed = {k: self._normalize(v) if k != "path" else v for k, v in sources.items()}
            for nid in note_ids:
                final = (
                    self.weights.get("dense", 0.0) * normed["dense"].get(nid, 0.0)
                    + self.weights.get("bm25", 0.0) * normed["bm25"].get(nid, 0.0)
                    + self.weights.get("graph", 0.0) * normed["graph"].get(nid, 0.0)
                    + self.weights.get("path", 0.0) * normed["path"].get(nid, 0.0)
                )
                results.append({
                    "note_id": nid,
                    "scores": {k: sources[k].get(nid) for k in ["dense", "bm25", "graph", "path"]},
                    "final_similarity": final,
                    "tags": {
                        "source": "graph" if nid in sources["graph"] else "semantic",
                        "is_bridge": nid in sources["path"],
                    },
                })

        results.sort(key=lambda x: x["final_similarity"], reverse=True)
        return results[: self.candidate_pool]
