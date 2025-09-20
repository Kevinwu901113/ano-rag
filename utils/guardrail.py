"""Guardrail filters for retrieval candidates."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger as default_logger


Candidate = Dict[str, Any]
Report = Dict[str, Any]


def _candidate_text(candidate: Candidate) -> str:
    """Return the textual content associated with a candidate."""
    content = candidate.get("content")
    if content:
        return str(content)
    title = candidate.get("title") or ""
    text = candidate.get("text") or ""
    if title and text:
        return f"{title}\n{text}"
    if title:
        return str(title)
    if text:
        return str(text)
    metadata = candidate.get("metadata") or {}
    return str(metadata.get("content", ""))


def _candidate_score(candidate: Candidate) -> float:
    info = candidate.get("retrieval_info") or {}
    score = info.get("similarity")
    if score is None:
        score = info.get("score")
    if score is None:
        score = info.get("fusion_score")
    if score is None:
        score = info.get("rerank_score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _cluster_id(candidate: Candidate) -> Any:
    cluster = candidate.get("cluster_id")
    if cluster is None:
        metadata = candidate.get("metadata") or {}
        cluster = metadata.get("cluster_id") or metadata.get("cluster")
    if cluster is None:
        note_id = candidate.get("note_id")
        if isinstance(note_id, str) and "__" in note_id:
            cluster = note_id.split("__")[0]
        else:
            cluster = note_id
    return cluster


def _emit_log(logger, report: Report, query_id: Optional[Any]) -> None:
    if not report:
        return
    payload = {k: v for k, v in report.items() if v not in (None, "")}
    if query_id is not None:
        payload.setdefault("query_id", query_id)
    if logger is not None:
        try:
            logger.info("guardrail_filter", **payload)
            return
        except Exception:
            pass
    default_logger.info(f"guardrail_filter {payload}")


def _extract_entities(text: str, *, stopwords: Optional[Sequence[str]] = None,
                      min_length: int = 3) -> List[str]:
    if not text:
        return []
    stop_set = {w.lower() for w in stopwords or []}
    entities: List[str] = []
    # Capture capitalised tokens and numeric spans
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9_\-']+|\d{3,})\b", text):
        token = match.group(0)
        if len(token) < min_length:
            continue
        if token.lower() in stop_set:
            continue
        entities.append(token)
    # Fallback to generic words when casing is unavailable
    if not entities:
        for match in re.finditer(r"\b([A-Za-z0-9_\-']{3,})\b", text):
            token = match.group(0)
            if token.lower() in stop_set:
                continue
            entities.append(token)
    return entities


def _restore_minimum(kept: List[Candidate], dropped: List[Dict[str, Any]],
                     min_keep: int) -> None:
    if min_keep <= 0 or len(kept) >= min_keep:
        return
    ranked = sorted(dropped, key=lambda item: _candidate_score(item["candidate"]), reverse=True)
    while len(kept) < min_keep and ranked:
        entry = ranked.pop(0)
        entry["restored"] = True
        kept.append(entry["candidate"])


def numeric_date_filter(candidates: Sequence[Candidate], question: str,
                        config: Dict[str, Any]) -> Tuple[List[Candidate], Report]:
    report: Report = {
        "filter": "numeric_date",
        "before": len(candidates),
    }
    if not candidates:
        report["after"] = 0
        report["skipped"] = True
        report["skip_reason"] = "no_candidates"
        return list(candidates), report

    question_lower = (question or "").lower()
    trigger_keywords = [
        "how many",
        "how much",
        "number",
        "amount",
        "total",
        "when",
        "what year",
        "date",
        "year",
        "age",
    ]
    trigger_keywords = list(config.get("question_triggers", trigger_keywords))
    requires_numeric = any(keyword in question_lower for keyword in trigger_keywords) or bool(re.search(r"\d", question_lower))
    if not requires_numeric:
        report["after"] = len(candidates)
        report["skipped"] = True
        report["skip_reason"] = "no_numeric_intent"
        return list(candidates), report

    month_keywords = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    month_keywords = list(config.get("month_keywords", month_keywords))

    kept: List[Candidate] = []
    dropped: List[Dict[str, Any]] = []
    for candidate in candidates:
        text = _candidate_text(candidate).lower()
        has_digit = bool(re.search(r"\d", text))
        has_date = bool(re.search(r"\b(?:19|20)\d{2}\b", text)) or any(month in text for month in month_keywords)
        if has_digit or has_date:
            kept.append(candidate)
        else:
            dropped.append({
                "candidate": candidate,
                "reason": "missing_numeric_or_date_evidence",
            })

    min_keep = max(int(config.get("min_keep", 1)), 0)
    _restore_minimum(kept, dropped, min_keep)

    report["after"] = len(kept)
    report["dropped"] = [
        {"note_id": entry["candidate"].get("note_id"), "reason": entry["reason"]}
        for entry in dropped if not entry.get("restored")
    ]
    report["restored"] = [
        {"note_id": entry["candidate"].get("note_id"), "reason": entry["reason"]}
        for entry in dropped if entry.get("restored")
    ]
    return kept, report


def entity_block_filter(candidates: Sequence[Candidate], question: str,
                        config: Dict[str, Any]) -> Tuple[List[Candidate], Report]:
    report: Report = {
        "filter": "entity_block",
        "before": len(candidates),
    }
    if not candidates:
        report["after"] = 0
        report["skipped"] = True
        report["skip_reason"] = "no_candidates"
        return list(candidates), report

    stopwords = config.get("stopwords") or [
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "which",
    ]
    min_overlap = max(int(config.get("min_overlap", 1)), 1)
    question_entities = set(_extract_entities(question, stopwords=stopwords))
    if len(question_entities) < min_overlap:
        report["after"] = len(candidates)
        report["skipped"] = True
        report["skip_reason"] = "insufficient_query_entities"
        return list(candidates), report

    kept: List[Candidate] = []
    dropped: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_entities = set(_extract_entities(_candidate_text(candidate), stopwords=stopwords))
        overlap = question_entities & candidate_entities
        if len(overlap) >= min_overlap:
            kept.append(candidate)
        else:
            dropped.append({
                "candidate": candidate,
                "reason": "no_entity_overlap",
            })

    min_keep = max(int(config.get("min_keep", 1)), 0)
    _restore_minimum(kept, dropped, min_keep)

    report["after"] = len(kept)
    report["dropped"] = [
        {"note_id": entry["candidate"].get("note_id"), "reason": entry["reason"]}
        for entry in dropped if not entry.get("restored")
    ]
    report["restored"] = [
        {"note_id": entry["candidate"].get("note_id"), "reason": entry["reason"]}
        for entry in dropped if entry.get("restored")
    ]
    report["overlap_sample"] = sorted(list(question_entities))[:5]
    return kept, report


def atomicity_filter(candidates: Sequence[Candidate], question: str,
                     config: Dict[str, Any]) -> Tuple[List[Candidate], Report]:
    report: Report = {
        "filter": "atomicity_check",
        "before": len(candidates),
    }
    if not candidates:
        report["after"] = 0
        report["skipped"] = True
        report["skip_reason"] = "no_candidates"
        return list(candidates), report

    max_sentences = max(int(config.get("max_sentences", 3)), 1)
    max_length = max(int(config.get("max_length", 600)), 1)
    connectors = [
        "however",
        "meanwhile",
        "whereas",
        "on the other hand",
        "additionally",
        "moreover",
        "furthermore",
    ]
    connectors = [c.lower() for c in config.get("disallowed_connectors", connectors)]

    kept: List[Candidate] = []
    dropped: List[Dict[str, Any]] = []
    for candidate in candidates:
        text = _candidate_text(candidate)
        lower_text = text.lower()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        long_content = len(text) > max_length
        too_many_sentences = len(sentences) > max_sentences
        has_connector = any(connector in lower_text for connector in connectors)
        if long_content or too_many_sentences or has_connector:
            reasons: List[str] = []
            if long_content:
                reasons.append(f"length>{max_length}")
            if too_many_sentences:
                reasons.append(f"sentences>{max_sentences}")
            if has_connector:
                reasons.append("multi_clause")
            dropped.append({
                "candidate": candidate,
                "reason": ",".join(reasons) or "non_atomic",
            })
        else:
            kept.append(candidate)

    min_keep = max(int(config.get("min_keep", 1)), 0)
    _restore_minimum(kept, dropped, min_keep)

    report["after"] = len(kept)
    report["dropped"] = [
        {"note_id": entry["candidate"].get("note_id"), "reason": entry["reason"]}
        for entry in dropped if not entry.get("restored")
    ]
    report["restored"] = [
        {"note_id": entry["candidate"].get("note_id"), "reason": entry["reason"]}
        for entry in dropped if entry.get("restored")
    ]
    return kept, report


def diversify_by_cluster(candidates: Sequence[Candidate], question: str,
                         config: Dict[str, Any]) -> Tuple[List[Candidate], Report]:
    report: Report = {
        "filter": "diversify",
        "before": len(candidates),
    }
    if not candidates:
        report["after"] = 0
        report["skipped"] = True
        report["skip_reason"] = "no_candidates"
        return list(candidates), report

    max_per_cluster = int(config.get("max_per_cluster", 1))
    if max_per_cluster <= 0:
        report["after"] = len(candidates)
        report["skipped"] = True
        report["skip_reason"] = "max_per_cluster<=0"
        return list(candidates), report

    kept: List[Candidate] = []
    dropped: List[Dict[str, Any]] = []
    cluster_counts: Dict[Any, int] = {}
    for candidate in candidates:
        cluster = _cluster_id(candidate)
        count = cluster_counts.get(cluster, 0)
        if count < max_per_cluster:
            cluster_counts[cluster] = count + 1
            kept.append(candidate)
        else:
            dropped.append({
                "candidate": candidate,
                "reason": f"cluster_quota_{cluster}",
            })

    min_clusters = max(int(config.get("min_clusters", 0)), 0)
    if min_clusters:
        kept_clusters = { _cluster_id(candidate) for candidate in kept }
        for entry in dropped:
            if len(kept_clusters) >= min_clusters:
                break
            candidate = entry["candidate"]
            cluster = _cluster_id(candidate)
            if cluster in kept_clusters:
                continue
            kept.append(candidate)
            kept_clusters.add(cluster)
            entry["restored"] = True

    min_keep = max(int(config.get("min_keep", 1)), 0)
    _restore_minimum(kept, dropped, min_keep)

    report["after"] = len(kept)
    report["dropped"] = [
        {"note_id": entry["candidate"].get("note_id"), "reason": entry["reason"]}
        for entry in dropped if not entry.get("restored")
    ]
    report["restored"] = [
        {"note_id": entry["candidate"].get("note_id"), "reason": entry["reason"]}
        for entry in dropped if entry.get("restored")
    ]
    report["kept_clusters"] = list({ _cluster_id(candidate) for candidate in kept })
    return kept, report


FILTERS = (
    ("numeric_date", numeric_date_filter),
    ("entity_block", entity_block_filter),
    ("atomicity_check", atomicity_filter),
    ("diversify", diversify_by_cluster),
)


def apply_guardrails(
    candidates: Sequence[Candidate],
    question: str,
    guardrail_config: Optional[Dict[str, Any]],
    *,
    logger=None,
    query_id: Optional[Any] = None,
) -> Tuple[List[Candidate], List[Report]]:
    """Apply configured guardrail filters to candidates."""

    if not candidates:
        return list(candidates), []

    cfg = guardrail_config or {}
    if not bool(cfg.get("enabled", True)):
        return list(candidates), []

    working: List[Candidate] = list(candidates)
    reports: List[Report] = []

    for key, fn in FILTERS:
        settings = cfg.get(key) or {}
        enabled = settings.get("enabled")
        if enabled is None:
            # default to False for disabled filters unless explicitly true
            enabled = False
        if not enabled:
            continue
        working, report = fn(working, question, settings)
        reports.append(report)
        _emit_log(logger, report, query_id)
        if not working:
            # Stop early if all candidates were removed
            break

    return working, reports


__all__ = [
    "apply_guardrails",
    "numeric_date_filter",
    "entity_block_filter",
    "atomicity_filter",
    "diversify_by_cluster",
]
