from typing import Any, Dict, List

from graph.index import NoteGraph
from graph.search import Path, beam_search


def extract_rel_chain(question: str) -> List[str]:
    """Placeholder for relation-chain extraction logic."""

    _ = question
    return []


def select_answer_from_paths(paths: List[Path], target_rel: str) -> Dict[str, Any]:
    if not paths:
        return {}

    for path in paths:
        if not path.rels:
            continue
        if target_rel and path.rels[-1] != target_rel:
            continue
        return {
            "answer": path.keys[-1],
            "support_note_ids": path.notes,
            "rels": path.rels,
            "path_score": path.score,
            "path_keys": path.keys,
        }

    best = paths[0]
    return {
        "answer": best.keys[-1],
        "support_note_ids": best.notes,
        "rels": best.rels,
        "path_score": best.score,
        "path_keys": best.keys,
    }


def answer_question(question: str, notes: List[Dict[str, Any]], anchors: List[str]) -> Dict[str, Any]:
    graph = NoteGraph()
    for note in notes:
        graph.add_note(note)

    anchor_list = anchors or []
    if not anchor_list:
        anchor_list = [note.get("head_key") for note in notes if note.get("head_key")]
    anchor_list = [a for a in anchor_list if a]
    # Deduplicate anchors while preserving order
    seen = set()
    deduped_anchors = []
    for anchor in anchor_list:
        if anchor in seen:
            continue
        seen.add(anchor)
        deduped_anchors.append(anchor)

    if not deduped_anchors:
        return {}

    rel_chain = extract_rel_chain(question)
    paths = beam_search(graph, deduped_anchors, rel_chain)
    if not paths:
        return {}

    target_rel = rel_chain[-1] if rel_chain else ""
    return select_answer_from_paths(paths, target_rel)
