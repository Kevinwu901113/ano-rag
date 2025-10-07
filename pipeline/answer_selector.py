from typing import Any, Dict, Iterable, List, Sequence

from config import config
from graph.index import NoteGraph
from graph.search import Path, beam_search


def extract_rel_chain(question: str) -> List[str]:
    """Return a planned relation chain for the given question if available."""

    _ = question  # Reserved for future question understanding logic
    answer_cfg = config.get("answering", {}) or {}
    chains = answer_cfg.get("rel_chains") or []
    if not chains:
        return []
    first_chain = chains[0]
    return list(first_chain) if isinstance(first_chain, (list, tuple)) else []


def _path_to_answer(path: Path, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.keys:
        return {}

    payload: Dict[str, Any] = {
        "answer": path.keys[-1],
        "support_note_ids": path.notes,
        "rels": path.rels,
        "path_score": path.score,
        "path_keys": path.keys,
    }
    if extra:
        payload.update(extra)
    return payload


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def answer_question(question: str, notes: List[Dict[str, Any]], anchors: List[str]) -> Dict[str, Any]:
    graph = NoteGraph()
    for note in notes:
        graph.add_note(note)

    anchor_list = anchors or []
    if not anchor_list:
        anchor_list = [note.get("head_key") for note in notes if note.get("head_key")]
    deduped_anchors = _dedupe_preserve_order(anchor_list)

    if not deduped_anchors:
        return {}

    answer_cfg = config.get("answering", {}) or {}
    configured_chains = answer_cfg.get("rel_chains") or []
    candidate_chains: List[Sequence[str]] = []

    primary_chain = extract_rel_chain(question)
    if primary_chain:
        candidate_chains.append(tuple(primary_chain))

    for chain in configured_chains:
        if isinstance(chain, (list, tuple)):
            chain_tuple = tuple(chain)
            if chain_tuple not in candidate_chains:
                candidate_chains.append(chain_tuple)

    for chain in candidate_chains:
        paths = beam_search(graph, deduped_anchors, list(chain))
        if paths:
            return _path_to_answer(paths[0], {
                "relation_chain": list(chain),
                "strategy": "planned",
            })

    relax_specs = answer_cfg.get("relax_last_hop") or []
    if candidate_chains and relax_specs:
        for chain in candidate_chains:
            if not chain:
                continue
            base = list(chain)
            for relax in relax_specs:
                relaxed_chain = base[:-1] + [relax]
                paths = beam_search(graph, deduped_anchors, relaxed_chain)
                if paths:
                    return _path_to_answer(paths[0], {
                        "relation_chain": relaxed_chain,
                        "strategy": "relaxed",
                    })

    fallback_paths = beam_search(graph, deduped_anchors, None)
    if fallback_paths:
        return _path_to_answer(fallback_paths[0], {
            "relation_chain": [],
            "strategy": "fallback",
        })

    return {}
