"""Lightweight context dispatcher that performs quota merge and deduplication.

The dispatcher receives candidates that already contain ``final_similarity``
produced by the hybrid fusion layer. It separates semantic and graph
candidates, applies configured quotas, deduplicates by ``note_id`` (semantic
results take priority), and applies optional bridge policies.
"""
from __future__ import annotations

from typing import List, Dict, Any


class ContextDispatcher:
    def __init__(self, config):
        cfg = config if isinstance(config, dict) else config.load_config()
        d_cfg = cfg.get("dispatcher", {})
        self.final_semantic_count = d_cfg.get("final_semantic_count", 8)
        self.final_graph_count = d_cfg.get("final_graph_count", 5)
        self.bridge_policy = d_cfg.get("bridge_policy", "keepalive")
        self.bridge_boost_epsilon = d_cfg.get("bridge_boost_epsilon", 0.02)
        self.debug_log = d_cfg.get("debug_log", True)

    def dispatch(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        semantic = [c for c in candidates if c.get("tags", {}).get("source") != "graph"]
        graph = [c for c in candidates if c.get("tags", {}).get("source") == "graph"]

        semantic.sort(key=lambda x: x.get("final_similarity", 0), reverse=True)
        graph.sort(key=lambda x: x.get("final_similarity", 0), reverse=True)

        selected_semantic = semantic[: self.final_semantic_count]
        selected_graph = graph[: self.final_graph_count]

        merged: Dict[str, Dict[str, Any]] = {}
        for cand in selected_semantic:
            merged[cand["note_id"]] = cand
        for cand in selected_graph:
            nid = cand["note_id"]
            if nid in merged:
                merged[nid]["tags"]["is_bridge"] = merged[nid]["tags"].get("is_bridge") or cand["tags"].get("is_bridge")
                merged[nid]["scores"].update({k: v for k, v in cand["scores"].items() if v is not None})
            else:
                merged[nid] = cand

        results = list(merged.values())

        if self.bridge_policy == "boost":
            for cand in results:
                if cand.get("tags", {}).get("is_bridge"):
                    cand["final_similarity"] += self.bridge_boost_epsilon

        results.sort(key=lambda x: x.get("final_similarity", 0), reverse=True)
        limit = self.final_semantic_count + self.final_graph_count
        trimmed = results[:limit]

        if self.bridge_policy == "keepalive":
            bridges = [c for c in results if c.get("tags", {}).get("is_bridge") and c not in trimmed]
            trimmed.extend(bridges)

        return trimmed
