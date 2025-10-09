from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AnswerSelectorResult:
    answer: str
    confidence: float
    evidence_note_ids: List[str]
    missing_entities: Optional[List[str]] = None
    covered_entities: Optional[List[str]] = None


def run_answer_selector(question: str, candidate_note_ids: List[str]) -> AnswerSelectorResult:
    """Minimal placeholder that surfaces retrieval outputs to downstream logic."""

    del question  # 当前占位实现不依赖问题内容

    return AnswerSelectorResult(
        answer="",
        confidence=0.0,
        evidence_note_ids=candidate_note_ids[:5],
        missing_entities=["bridge_entity_placeholder"] if candidate_note_ids else [],
        covered_entities=[],
    )


__all__ = ["AnswerSelectorResult", "run_answer_selector"]
