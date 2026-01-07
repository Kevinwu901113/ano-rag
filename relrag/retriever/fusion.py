from __future__ import annotations

from typing import Any, Dict, List


def fuse_rankings(
    rankings: Dict[str, List[Dict[str, float]]],
    *,
    weights: Dict[str, float],
    rrf_k: int,
) -> List[Dict[str, float]]:
    aggregated: Dict[str, float] = {}
    provenance: Dict[str, Dict[str, float]] = {}
    for source, items in rankings.items():
        if not items:
            continue
        weight = float(weights.get(source, 1.0))
        for item in items:
            note_id = item.get("note_id")
            if not note_id:
                continue
            rank = int(item.get("rank") or 0)
            if rank <= 0:
                rank = items.index(item) + 1
            contribution = weight / (rrf_k + rank)
            aggregated[note_id] = aggregated.get(note_id, 0.0) + contribution
            provenance.setdefault(note_id, {})[source] = contribution
    fused = [
        {"note_id": note_id, "score": score, "sources": provenance.get(note_id, {})}
        for note_id, score in aggregated.items()
    ]
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused


def deduplicate_by_triplet(
    candidates: List[Dict[str, Any]],
    *,
    topk: int,
) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    ordered: List[Dict[str, Any]] = []
    for cand in candidates:
        note = cand.get("note") or {}
        key = "|".join(
            [
                str(note.get("subj") or "").strip().lower(),
                str(note.get("pred") or "").strip().lower(),
                str(note.get("obj") or "").strip().lower(),
            ]
        )
        prev = seen.get(key)
        if prev is None or cand.get("final_score", 0.0) > prev.get("final_score", 0.0):
            seen[key] = cand
    ordered = sorted(seen.values(), key=lambda item: item.get("final_score", 0.0), reverse=True)
    return ordered[:topk]
