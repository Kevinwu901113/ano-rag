"""Validator for atomic note outputs enforcing schema and semantic rules."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

try:
    from jsonschema import Draft7Validator, ValidationError
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Draft7Validator = None

    class ValidationError(Exception):
        """Fallback validation error when jsonschema is unavailable."""

        pass

from .note_schema import NOTE_SCHEMA

_person_like = re.compile(
    r"(?:[A-Z][a-z]+(?:[-\s][A-Z][a-z]+){1,3}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[\u4E00-\u9FFF]{2,4})",
    re.UNICODE,
)


def _normalize(value: str) -> str:
    """Normalize whitespace for robust comparisons."""
    return (value or "").strip()


def _extract_json_payload(raw_text: str) -> str:
    """Best-effort extraction of a JSON array from noisy model output."""

    if not raw_text:
        return raw_text

    text = raw_text.strip()

    # Remove optional Markdown fences such as ```json ... ```
    if text.startswith("```"):
        fence_match = re.match(r"^```(?:json)?\s*", text, flags=re.IGNORECASE)
        if fence_match:
            text = text[fence_match.end():]
        fence_end = text.rfind("```")
        if fence_end != -1:
            text = text[:fence_end]

    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        text = text[first_bracket : last_bracket + 1]

    return text.strip()


def validate_notes(
    raw_text: str,
    *,
    allow_partial: bool = True,
) -> Tuple[bool, Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Validate raw LLM output against schema and semantic constraints.

    Args:
        raw_text: The raw string returned by the LLM.
        allow_partial: Whether to retain valid notes when some fail checks.

    Returns:
        A tuple ``(is_valid, partitioned_notes, metrics)`` where
        ``partitioned_notes`` contains ``{"valid": [...], "invalid": [...]}`` and
        metrics capture violation details and coverage ratios.
    """

    violations: Dict[str, Any] = {}
    partitioned: Dict[str, List[Dict[str, Any]]] = {"valid": [], "invalid": []}

    try:
        notes = json.loads(raw_text)
    except Exception:  # noqa: BLE001 - propagate parse errors as metrics only
        cleaned = _extract_json_payload(raw_text)
        try:
            notes = json.loads(cleaned)
        except Exception as exc:
            metrics = {
                "violations": {"json_parse": str(exc)},
                "person_coverage": 0.0,
                "valid_ratio": 0.0,
                "schema_validation_skipped": True,
            }
            return False, partitioned, metrics

    schema_checked = Draft7Validator is not None
    if Draft7Validator is not None:
        try:
            Draft7Validator(NOTE_SCHEMA).validate(notes)
        except ValidationError as exc:
            metrics = {
                "violations": {"schema": str(exc)},
                "person_coverage": 0.0,
                "valid_ratio": 0.0,
                "schema_validation_skipped": False,
            }
            return False, partitioned, metrics

    bad_idx: List[Tuple[int, str]] = []
    ok_count = 0

    for idx, note in enumerate(notes):
        content = _normalize(note.get("content", ""))
        persons: List[str] = note.get("entities", {}).get("PERSON", []) or []

        if not persons:
            bad_idx.append((idx, "empty_person"))
            partitioned["invalid"].append({"index": idx, "note": note, "reason": "empty_person"})
            continue

        content_low = content.lower()
        name_hit = any(name and name.lower() in content_low for name in persons)
        has_fullname_shape = bool(_person_like.search(content))

        if not (name_hit or has_fullname_shape):
            bad_idx.append((idx, "person_not_in_content"))
            partitioned["invalid"].append({"index": idx, "note": note, "reason": "person_not_in_content"})
            continue

        ok_count += 1
        partitioned["valid"].append(note)

    if bad_idx:
        violations["person_rules"] = bad_idx

    if not allow_partial and bad_idx:
        partitioned["valid"] = []

    total_notes = len(notes)
    valid_ratio = 0.0 if total_notes == 0 else ok_count / total_notes
    metrics = {
        "violations": violations,
        "person_coverage": valid_ratio,
        "valid_ratio": valid_ratio,
        "notes_total": total_notes,
        "notes_valid": ok_count,
        "notes_invalid": len(bad_idx),
        "schema_validation_skipped": not schema_checked,
    }

    is_valid = len(bad_idx) == 0
    return is_valid, partitioned, metrics
