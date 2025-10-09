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
    is_valid, notes, metrics = validate_notes(raw_retry)

    if is_valid:
        return {
            "raw": raw_retry,
            "valid": True,
            "notes": notes,
            "qc": metrics,
            "invalid_person": False,
            "retry": {"attempted": True, "success": True},
        }

    return {
        **first_result,
        "retry": {"attempted": True, "success": False},
    }
