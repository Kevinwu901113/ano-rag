"""Validator for final answer outputs with verbatim evidence constraints."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from config import config


def _extract_json(text: str) -> Tuple[bool, Dict[str, Any]]:
    """Extract the first JSON object substring from text."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return False, {}
        obj = json.loads(text[start : end + 1])
        return True, obj
    except Exception:
        return False, {}


def _context_index(lines: List[str]) -> Dict[int, str]:
    return {i + 1: (ln or "").strip() for i, ln in enumerate(lines)}


def _span_in_context(span: str, ctx_idx: Dict[int, str]) -> bool:
    """Check that the span references a valid line tag and quotes verbatim text."""
    span = (span or "").strip()
    if not span.startswith("[L"):
        return False
    try:
        rbr = span.find("]")
        if rbr == -1:
            return False
        ln = int(span[2:rbr])
        quoted = span[rbr + 1 :].strip()
        base = ctx_idx.get(ln, "")
        return bool(quoted) and quoted.lower() in base.lower() and len(quoted) >= 4
    except Exception:
        return False


def _answer_in_spans(ans: str, spans: List[str]) -> bool:
    al = (ans or "").strip().lower()
    if not al or not spans:
        return False
    for span in spans:
        if al in (span or "").lower():
            return True
    return False


def validate_final_answer(
    raw_text: str,
    context_lines: List[str],
    question: str,
    candidate_answer: Optional[str],
) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    """Validate final answer JSON structure and evidence alignment.

    Returns (ok, parsed_obj, report).
    """

    ok_json, obj = _extract_json(raw_text)
    report: Dict[str, Any] = {"ok_json": ok_json}

    if not ok_json or not isinstance(obj, dict):
        report["error"] = "json_parse_failed"
        return False, {}, report

    spans = obj.get("evidence_spans") or []
    if not isinstance(spans, list):
        spans = []

    ctx_idx = _context_index(context_lines or [])
    span_hits = [_span_in_context(span, ctx_idx) for span in spans]
    valid_spans = [span for span, hit in zip(spans, span_hits) if hit]

    answering_cfg = config.get("answering", {}) or {}
    require_spans = bool(answering_cfg.get("require_verbatim_spans", True))

    report.update(
        {
            "span_count": len(spans),
            "valid_span_count": len(valid_spans),
            "valid_span_ratio": 0 if not spans else len(valid_spans) / len(spans),
        }
    )

    if require_spans and len(valid_spans) == 0:
        report["error"] = "no_valid_evidence_spans"
        return False, obj, report

    used_candidate = bool(obj.get("used_candidate"))
    if used_candidate:
        if not candidate_answer:
            report["error"] = "used_candidate_without_candidate_answer"
            return False, obj, report

        candidate_parts = [part.strip() for part in str(candidate_answer).split("|") if part.strip()]
        any_hit = any(_answer_in_spans(part, valid_spans) for part in candidate_parts)
        if not any_hit:
            report["error"] = "used_candidate_not_supported_by_spans"
            return False, obj, report

    return True, obj, report
