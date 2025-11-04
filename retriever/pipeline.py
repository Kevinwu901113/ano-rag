from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .ir import PredicateStep, QueryIR, Seed
from .note_store import NoteStore
from .operators import BIND, EXPAND_from, Indexes
from .parser import parse_question
from .scorer import score_path


@dataclass
class Candidate:
    answer: Optional[str]
    path: List[Dict[str, Any]]
    note_ids: List[str]
    score: float


def retrieve_answer(question: str, indexes: Indexes, note_store: NoteStore) -> Dict[str, Any]:
    ir = parse_question(question)
    if ir is None or not ir.is_valid:
        return {
            "ir": ir.to_dict() if ir else None,
            "answer": None,
            "paths": [],
            "support_note_ids": [],
            "evidence": [],
            "reason": "parse_failed",
        }

    seed_entities = _bind_seeds(ir.seeds, indexes, ir.fanout)
    if not seed_entities:
        return {
            "ir": ir.to_dict(),
            "answer": None,
            "paths": [],
            "support_note_ids": [],
            "evidence": [],
            "reason": "no_seed_match",
        }

    candidates = _walk_chain(seed_entities, ir, indexes, note_store)
    if not candidates:
        return {
            "ir": ir.to_dict(),
            "answer": None,
            "paths": [],
            "support_note_ids": [],
            "evidence": [],
            "reason": "no_path",
        }

    candidates.sort(key=lambda c: c.score, reverse=True)
    top_candidates = candidates[: ir.fanout]

    support_note_ids = []
    for cand in top_candidates:
        for nid in cand.note_ids:
            if nid not in support_note_ids:
                support_note_ids.append(nid)

    evidences = [
        {"note_id": note["note_id"], "evidence": note.get("evidence", "")}
        for note in note_store.get_many(support_note_ids)
    ]

    return {
        "ir": ir.to_dict(),
        "answer": top_candidates[0].answer,
        "paths": [cand.path for cand in top_candidates],
        "support_note_ids": support_note_ids,
        "evidence": evidences,
        "reason": None,
    }


def _bind_seeds(seeds: Sequence[Seed], indexes: Indexes, limit: int) -> List[str]:
    collected: List[str] = []
    seen = set()
    for seed in seeds:
        type_candidates = [seed.type_hint] if seed.type_hint else []
        matches = BIND(indexes, seed.text, type_candidates, limit=limit)
        for entity in matches:
            if entity in seen:
                continue
            collected.append(entity)
            seen.add(entity)
            if len(collected) >= limit:
                return collected
    return collected


def _walk_chain(
    entities: Sequence[str],
    ir: QueryIR,
    indexes: Indexes,
    note_store: NoteStore,
) -> List[Candidate]:
    if not ir.pred_chain:
        return _collect_entity_mentions(entities, indexes, note_store, ir)

    states = [{"entity": entity, "path": []} for entity in entities]
    for step_idx, step in enumerate(ir.pred_chain[: ir.max_hops]):
        next_states: List[Dict[str, Any]] = []
        for state in states:
            expanded = EXPAND_from(
                indexes,
                state["entity"],
                predicate=step.pred,
                direction=step.direction,
                limit=ir.fanout,
            )
            for obj, note_id in expanded:
                new_path = state["path"] + [
                    {
                        "subj": state["entity"],
                        "pred": step.pred,
                        "obj": obj,
                        "note_id": note_id,
                    }
                ]
                next_states.append({"entity": obj, "path": new_path})
        if not next_states:
            return []
        states = next_states[: ir.fanout]
        if not states:
            break

    candidates: List[Candidate] = []
    for state in states:
        path = state.get("path", [])
        if not path:
            continue
        note_ids = [edge["note_id"] for edge in path if edge.get("note_id")]
        score = score_path(path)
        final_note = note_store.get(note_ids[-1]) if note_ids else None
        if final_note and ir.target_type:
            obj_type = final_note.get("obj_type")
            if obj_type and obj_type.upper() == ir.target_type:
                score += 0.1
        answer = path[-1]["obj"] if path else None
        candidates.append(Candidate(answer=answer, path=path, note_ids=note_ids, score=score))
    return candidates


def _collect_entity_mentions(
    entities: Sequence[str], indexes: Indexes, note_store: NoteStore, ir: QueryIR
) -> List[Candidate]:
    candidates: List[Candidate] = []
    for entity in entities:
        note_ids = indexes.entity_to_notes.get(entity, [])[: ir.fanout]
        for nid in note_ids:
            path = [
                {
                    "subj": entity,
                    "pred": "__mention__",
                    "obj": entity,
                    "note_id": nid,
                }
            ]
            score = score_path(path)
            candidates.append(Candidate(answer=None, path=path, note_ids=[nid], score=score))
    return candidates
