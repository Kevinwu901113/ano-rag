from __future__ import annotations

import re
from typing import Dict, List, Optional

from relrag.utils import TextUtils


_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}

_NUM_WORD_PATTERN = "|".join(sorted(_NUM_WORDS.keys(), key=len, reverse=True))
_COUNT_TOKEN = rf"(?:\d{{1,3}}(?:,\d{{3}})*|\d+|(?:{_NUM_WORD_PATTERN})(?:[\s-](?:{_NUM_WORD_PATTERN}))*)"
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


def _parse_number(text: str) -> Optional[int]:
    raw = (text or "").strip().lower().replace(",", "")
    if not raw:
        return None
    if raw.isdigit():
        try:
            return int(raw)
        except ValueError:
            return None
    tokens = raw.replace("-", " ").split()
    total = 0
    current = 0
    for tok in tokens:
        if tok not in _NUM_WORDS:
            return None
        value = _NUM_WORDS[tok]
        if value in (100, 1000):
            if current == 0:
                current = 1
            current *= value
            if value == 1000:
                total += current
                current = 0
        else:
            current += value
    return total + current if (total + current) > 0 else None


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
            count_val = _parse_number(match.group("count"))
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
