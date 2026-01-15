from __future__ import annotations

import re
from typing import Dict, List, Optional

from relrag.utils import TextUtils
from relrag.utils.number_utils import NUM_WORD_PATTERN, parse_int


_COUNT_TOKEN = rf"(?:\d{{1,3}}(?:,\d{{3}})*|\d+|(?:{NUM_WORD_PATTERN})(?:[\s-](?:{NUM_WORD_PATTERN}))*)"
_SUBJ_TOKEN = r"(?:[Tt]he\s+)?[A-Z][A-Za-z0-9&'()./\-]*(?:\s+[A-Z][A-Za-z0-9&'()./\-]*)*"

_COUNT_PATTERNS = [
    (
        "has_member_count",
        re.compile(
            rf"(?P<subj>{_SUBJ_TOKEN})\s+(?:has|have|had|contains|contain|consists of|consisting of|includes|include)\s+(?P<count>{_COUNT_TOKEN})\s+members?\b",
            re.I,
        ),
    ),
    (
        "has_species_count",
        re.compile(
            rf"(?P<subj>{_SUBJ_TOKEN})\s+(?:has|have|had|contains|contain|consists of|consisting of|includes|include)\s+(?P<count>{_COUNT_TOKEN})\s+species\b",
            re.I,
        ),
    ),
    (
        "has_species_count",
        re.compile(
            rf"(?P<subj>{_SUBJ_TOKEN})\s+is\s+(?:a\s+)?genus\s+of\s+(?P<count>{_COUNT_TOKEN})\s+species\b",
            re.I,
        ),
    ),
]

_GENERIC_SUBJECTS = {
    "it",
    "this",
    "that",
    "the genus",
    "the species",
    "the group",
    "the organization",
    "the committee",
    "the senate",
}


def _normalize_subject(subject: str, doc_title: Optional[str]) -> Optional[str]:
    cleaned = (subject or "").strip(" ,.;:()")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if TextUtils.is_pronoun(cleaned) or lowered in _GENERIC_SUBJECTS:
        return doc_title.strip() if doc_title else None
    return cleaned


def extract_count_notes(
    chunk: Dict[str, str],
    *,
    doc_title: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Extract deterministic count facts (members/species) from chunk text."""
    text = (chunk.get("text") or "").strip()
    if not text:
        return []
    doc_id = (chunk.get("doc_id") or "").strip()
    chunk_id = (chunk.get("chunk_id") or "").strip()
    source = f"{doc_id}#{chunk_id}" if doc_id and chunk_id else doc_id or chunk_id
    notes: List[Dict[str, str]] = []
    sentences = TextUtils.split_with_spans(text)
    if not sentences:
        sentences = [{"text": text}]

    for span in sentences:
        sentence = (span.get("text") or "").strip()
        if not sentence:
            continue
        for pred, pattern in _COUNT_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            subj = _normalize_subject(match.group("subj"), doc_title)
            if not subj:
                continue
            count_val = parse_int(match.group("count"))
            if count_val is None:
                continue
            subj_type = TextUtils.guess_entity_type(subj) or "CONCEPT"
            notes.append(
                {
                    "subj": subj,
                    "pred": pred,
                    "obj": str(count_val),
                    "subj_type": subj_type,
                    "obj_type": "CONCEPT",
                    "evidence": sentence,
                    "meta": {"source": source, "confidence": 0.9},
                }
            )
    return notes
