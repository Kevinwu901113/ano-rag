from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

def _normalize_token(text: str) -> str:
    value = (text or "").strip().lower()
    value = value.replace("-", " ").replace(".", " ")
    return " ".join(value.split())


def subject_match(subject: str | None, seeds: Optional[Sequence[str]], alias_lookup: Optional[Dict[str, str]] = None) -> float:
    """Score how well a note subject matches the query seeds."""
    if not subject:
        return 0.0
    norm_subj = _normalize_token(subject)
    if not norm_subj:
        return 0.0
    aliases = alias_lookup or {}
    subj_canon = _normalize_token(aliases.get(norm_subj, subject))
    best = 0.0
    for seed in seeds or []:
        seed_norm = _normalize_token(seed)
        if not seed_norm:
            continue
        if subj_canon == seed_norm:
            return 1.0
        if subj_canon in seed_norm or seed_norm in subj_canon:
            best = max(best, 0.5)
    return best


def score_path(
    path: Sequence[Dict[str, Any]],
    *,
    notes: Optional[Sequence[Dict[str, Any]]] = None,
    doc_name: Optional[str] = None,
    seeds: Optional[Sequence[str]] = None,
    query_ir: Any = None,
    alias_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[float, Dict[str, float]]:
    edges = list(path or [])
    if not edges:
        return 0.0, {"len_penalty": 0.0, "conf_score": 0.0, "entity_score": 0.0, "pred_score": 0.0, "entity_avg": 0.0}

    hop_penalty = -0.05 * max(0, len(edges) - 1)
    conf_values = []
    subjects = []
    preds = []
    for edge in edges:
        preds.append(_normalize_token(edge.get("pred")))
        conf_values.append(float(edge.get("conf", 0.0)))
        subjects.append(edge.get("subj"))
    conf_score = sum(conf_values) / len(conf_values) if conf_values else 0.0

    note_records = list(notes or [])
    if not note_records:
        note_records = [{"subj": subj} for subj in subjects]

    subj_scores = [subject_match(note.get("subj"), seeds, alias_lookup) for note in note_records]
    entity_min = min(subj_scores) if subj_scores else 0.0
    entity_avg = sum(subj_scores) / len(subj_scores) if subj_scores else 0.0

    expected_preds = []
    if query_ir and getattr(query_ir, "pred_chain", None):
        expected_preds = [_normalize_token(step.pred) for step in query_ir.pred_chain[: len(edges)]]
    pred_score = 0.0
    if expected_preds:
        pred_score = 1.0 if expected_preds == preds else -1.0

    metrics = {
        "len_penalty": hop_penalty,
        "conf_score": conf_score,
        "entity_score": entity_min,
        "entity_avg": entity_avg,
        "pred_score": pred_score,
    }
    total = hop_penalty + conf_score + 1.5 * entity_avg + 0.5 * entity_min + 2.0 * pred_score
    return total, metrics
