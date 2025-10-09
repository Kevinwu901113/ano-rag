"""Validator for atomic note outputs enforcing schema and semantic rules."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from jsonschema import Draft7Validator, ValidationError

from .note_schema import NOTE_SCHEMA

_person_like = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+")


def _normalize(value: str) -> str:
    """Normalize whitespace for robust comparisons."""
    return (value or "").strip()


def validate_notes(raw_text: str) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    """Validate raw LLM output against schema and semantic constraints.

    Args:
        raw_text: The raw string returned by the LLM.

    Returns:
        A tuple of (is_valid, notes, metrics) where metrics contains violation
        details and coverage ratios.
    """

    violations: Dict[str, Any] = {}
    try:
        notes = json.loads(raw_text)
    except Exception as exc:  # noqa: BLE001 - propagate parse errors as metrics only
        return False, [], {"violations": {"json_parse": str(exc)}, "person_coverage": 0.0}

    try:
        Draft7Validator(NOTE_SCHEMA).validate(notes)
    except ValidationError as exc:
        return False, [], {"violations": {"schema": exc.message}, "person_coverage": 0.0}

    bad_idx: List[Tuple[int, str]] = []
    ok_count = 0
    for idx, note in enumerate(notes):
        content = _normalize(note.get("content", ""))
        persons: List[str] = note.get("entities", {}).get("PERSON", []) or []

        if not persons:
            bad_idx.append((idx, "empty_person"))
            continue

        content_low = content.lower()
        name_hit = any(name and name.lower() in content_low for name in persons)
        has_fullname_shape = bool(_person_like.search(content))

        if not (name_hit or has_fullname_shape):
            bad_idx.append((idx, "person_not_in_content"))
            continue

        ok_count += 1

    if bad_idx:
        violations["person_rules"] = bad_idx

    is_valid = ok_count == len(notes)
    metrics = {
        "violations": violations,
        "person_coverage": 0.0 if not notes else ok_count / len(notes),
    }
    return is_valid, notes if is_valid else [], metrics
