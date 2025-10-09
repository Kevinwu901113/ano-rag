"""Retry middleware helpers for atomic note generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from validators.note_validator import validate_notes
from .prompts.atomic_note import ATOMIC_NOTE_SYSTEM_PROMPT, ATOMIC_NOTE_USER_PROMPT


def retry_if_invalid_person(
    chunk_text: str,
    entity_card: Optional[Dict[str, Any]],
    llm: Any,
    first_result: Dict[str, Any],
    max_retry: int = 1,
    *,
    allow_partial: bool = True,
    strict_person: bool = True,
) -> Dict[str, Any]:
    """Retry generation when the validator flags missing full names."""

    if not first_result.get("invalid_person"):
        return first_result

    if max_retry <= 0:
        return first_result

    stronger_card = {
        "persons": (entity_card or {}).get("persons", [])[:5],
        "aliases": (entity_card or {}).get("aliases", {}),
    }
    retry_prompt = (
        ATOMIC_NOTE_USER_PROMPT
        + "\nIMPORTANT: At least one FULL NAME from entity_card.persons must appear literally in each note's `content`."
    ).format(
        chunk_text=chunk_text,
        entity_card_json=json.dumps(stronger_card, ensure_ascii=False),
    )

    raw_retry = llm.chat(system=ATOMIC_NOTE_SYSTEM_PROMPT, user=retry_prompt)
    is_valid, partitioned, metrics = validate_notes(raw_retry, allow_partial=allow_partial)
    notes_valid = partitioned.get("valid", []) if partitioned else []
    notes_invalid = partitioned.get("invalid", []) if partitioned else []

    if is_valid:
        return {
            "raw": raw_retry,
            "valid": True,
            "notes": notes_valid,
            "notes_valid": notes_valid,
            "notes_invalid": notes_invalid,
            "qc": metrics,
            "invalid_person": False,
            "retry": {"attempted": True, "success": True},
        }

    current_invalid_flag = bool(first_result.get("invalid_person"))
    combined_invalid_flag = current_invalid_flag or bool(notes_invalid)

    return {
        **first_result,
        "notes": notes_valid or first_result.get("notes", []),
        "notes_valid": notes_valid or first_result.get("notes_valid", []),
        "notes_invalid": notes_invalid or first_result.get("notes_invalid", []),
        "qc": metrics if metrics else first_result.get("qc", {}),
        "invalid_person": strict_person and combined_invalid_flag,
        "retry": {"attempted": True, "success": False},
    }
