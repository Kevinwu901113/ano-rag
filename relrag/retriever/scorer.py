from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


def _normalize_token(text: str) -> str:
    value = (text or "").strip().lower()
    value = value.replace("-", " ").replace(".", " ")
    return " ".join(value.split())


def _record_subject_fail(diagnostics: Optional[Dict[str, Any]], reason: str) -> None:
    if diagnostics is None:
        return
    fail = diagnostics.setdefault("subject_match_fail_reason", {})
    fail[reason] = int(fail.get(reason, 0)) + 1


def subject_match(
    subject: str | None,
    seeds: Optional[Sequence[str]],
    alias_lookup: Optional[Dict[str, str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> float:
    """Score how well a note subject matches the query seeds."""
    if diagnostics is not None:
        diagnostics["subject_match_calls"] = int(diagnostics.get("subject_match_calls", 0)) + 1
    if not subject:
        _record_subject_fail(diagnostics, "empty_subject")
        return 0.0
    norm_subj = _normalize_token(subject)
    if not norm_subj:
        _record_subject_fail(diagnostics, "empty_subject")
        return 0.0
    aliases = alias_lookup or {}
    mapped = aliases.get(norm_subj, subject)
    subj_canon = _normalize_token(mapped)
    if diagnostics is not None and norm_subj in aliases and subj_canon and subj_canon != norm_subj:
        diagnostics["alias_lookup_hit_count"] = int(diagnostics.get("alias_lookup_hit_count", 0)) + 1
    if not seeds:
        _record_subject_fail(diagnostics, "no_seeds")
        return 0.0
    best = 0.0
    matched = False
    for seed in seeds or []:
        seed_norm = _normalize_token(seed)
        if not seed_norm:
            continue
        if subj_canon == seed_norm:
            matched = True
            return 1.0
        # Reduce false positives from prefix/suffix containment (e.g., "John Dawson Mayne" vs "John Mayne")
        subj_tokens = [t for t in subj_canon.split() if t]
        seed_tokens = [t for t in seed_norm.split() if t]
        if subj_tokens and seed_tokens:
            if len(subj_tokens) > len(seed_tokens):
                if subj_tokens[: len(seed_tokens)] == seed_tokens:
                    best = max(best, 0.3)
            elif len(seed_tokens) > len(subj_tokens):
                if seed_tokens[: len(subj_tokens)] == subj_tokens:
                    best = max(best, 0.3)
    if not matched and best <= 0.0:
        _record_subject_fail(diagnostics, "no_match")
    return best


def score_path(
    path: Sequence[Dict[str, Any]],
    *,
    notes: Optional[Sequence[Dict[str, Any]]] = None,
    doc_name: Optional[str] = None,
    seeds: Optional[Sequence[str]] = None,
    query_ir: Any = None,
    alias_lookup: Optional[Dict[str, str]] = None,
    scoring_stats: Optional[Dict[str, Any]] = None,
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

    subj_scores = [
        subject_match(note.get("subj"), seeds, alias_lookup, diagnostics=scoring_stats)
        for note in note_records
    ]
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
