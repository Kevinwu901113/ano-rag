from __future__ import annotations

"""Answer selection using atomic-note graph search."""

import re
from typing import Any, Dict, Iterable, List

from config import config
from graph.index import NoteGraph
from graph.search import beam_search


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def extract_rel_chain(question: str) -> List[str]:
    """Infer an expected relation chain via configurable regex rules."""

    answer_cfg = config.get("answering", {}) or {}
    rules = answer_cfg.get("rel_chain_rules") or []
    question_lower = (question or "").lower()

    for rule in rules:
        pattern = rule.get("pattern")
        chain = rule.get("chain")
        if not (pattern and chain):
            continue
        try:
            if re.search(pattern, question_lower):
                if isinstance(chain, (list, tuple)):
                    return [str(rel) for rel in chain if rel]
        except re.error:
            continue

    return []


def answer_from_notes(
    question: str,
    notes: List[Dict[str, Any]],
    anchor_keys: List[str],
) -> Dict[str, Any]:
    """Build a note graph and attempt to answer via multi-hop search."""

    if not notes:
        return {"answer": "", "support_note_ids": [], "rels": [], "path_score": 0.0}

    graph = NoteGraph()
    for note in notes:
        graph.add_note(note)

    anchors = _dedupe_preserve_order(anchor_keys or [])
    if not anchors:
        anchors = _dedupe_preserve_order(
            [note.get("head_key") for note in notes if note.get("head_key")]
        )

    if not anchors:
        return {"answer": "", "support_note_ids": [], "rels": [], "path_score": 0.0}

    rel_chain = extract_rel_chain(question)
    paths = beam_search(graph, anchors, rel_chain or None)
    if not paths and rel_chain:
        paths = beam_search(graph, anchors, None)

    if not paths:
        return {"answer": "", "support_note_ids": [], "rels": [], "path_score": 0.0}

    path = paths[0]
    answer = path.keys[-1] if path.keys else ""

    return {
        "answer": answer,
        "support_note_ids": path.notes,
        "rels": path.rels,
        "path_score": path.score,
        "path_keys": path.keys,
    }


def answer_question(question: str, notes: List[Dict[str, Any]], anchors: List[str]) -> Dict[str, Any]:
    """Backward-compatible entry point used by the pipeline."""

    return answer_from_notes(question, notes, anchors)
