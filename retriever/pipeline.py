from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from schema.note_schema_v1 import PRED_SYNONYM_SETS

from .ir import PredicateStep, QueryIR, Seed
from .intent_detector import AnswerIntent, AnswerIntentDetector
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


INTENT_DETECTOR = AnswerIntentDetector()


def retrieve_answer(question: str, indexes: Indexes, note_store: NoteStore) -> Dict[str, Any]:
    intent = INTENT_DETECTOR.detect(question)
    ir = parse_question(question)
    if ir is None or not ir.is_valid:
        return _fallback_lookup(intent, indexes, note_store, None, "parse_failed")

    seed_entities = _bind_seeds(ir.seeds, indexes, ir.fanout)
    if not seed_entities:
        return _fallback_lookup(intent, indexes, note_store, ir, "no_seed_match")

    candidates = _walk_chain(seed_entities, ir, indexes, note_store)
    if not candidates:
        return _fallback_lookup(intent, indexes, note_store, ir, "no_path")

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
        "fallback": {
            "used": False,
            "stage": None,
            "intent": intent.to_dict(),
            "status": "structured_hit",
        },
        "intent": intent.to_dict(),
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


def _fallback_lookup(
    intent: AnswerIntent,
    indexes: Indexes,
    note_store: NoteStore,
    ir: Optional[QueryIR],
    trigger: str,
) -> Dict[str, Any]:
    entity_queries: List[str] = []
    if ir:
        entity_queries.extend(seed.text for seed in ir.seeds if seed.text)
    if intent.entity and intent.entity not in entity_queries:
        entity_queries.append(intent.entity)

    type_candidates: List[str] = []
    if ir and ir.target_type:
        type_candidates.append(ir.target_type)
    if intent.entity_type and intent.entity_type not in type_candidates:
        type_candidates.append(intent.entity_type)

    attr_hint = intent.attribute
    if ir and ir.pred_chain:
        attr_hint = attr_hint or ir.pred_chain[0].pred
    canonical_attr = _canonical_predicate(attr_hint)

    entity_candidates: List[str] = []
    for query in entity_queries:
        lowered = (query or "").strip().lower()
        if not lowered:
            continue
        direct = [name for name in indexes.entity_to_notes.keys() if name.lower() == lowered]
        for name in direct:
            if name not in entity_candidates:
                entity_candidates.append(name)
        bound = BIND(indexes, query, type_candidates, limit=25)
        for entity in bound:
            if entity not in entity_candidates:
                entity_candidates.append(entity)
        if len(entity_candidates) >= 25:
            break

    status = "ok"
    if not entity_candidates:
        status = "entity_not_found"
        return {
            "ir": ir.to_dict() if ir else None,
            "answer": None,
            "paths": [],
            "support_note_ids": [],
            "evidence": [],
            "reason": trigger,
            "fallback": {
                "used": True,
                "stage": trigger,
                "status": status,
                "intent": intent.to_dict(),
                "candidates": [],
            },
            "intent": intent.to_dict(),
        }

    if not canonical_attr:
        status = "attribute_not_detected"
        return {
            "ir": ir.to_dict() if ir else None,
            "answer": None,
            "paths": [],
            "support_note_ids": [],
            "evidence": [],
            "reason": trigger,
            "fallback": {
                "used": True,
                "stage": trigger,
                "status": status,
                "intent": intent.to_dict(),
                "candidates": [
                    {"entity": entity_candidates[0], "attribute": None, "score": 0.0}
                ],
            },
            "intent": intent.to_dict(),
        }

    matched_notes: List[Dict[str, Any]] = []
    for entity in entity_candidates:
        matched_notes.extend(_collect_attribute_notes(entity, canonical_attr, indexes, note_store))

    if not matched_notes:
        status = "no_attribute_match"
        return {
            "ir": ir.to_dict() if ir else None,
            "answer": None,
            "paths": [],
            "support_note_ids": [],
            "evidence": [],
            "reason": trigger,
            "fallback": {
                "used": True,
                "stage": trigger,
                "status": status,
                "intent": intent.to_dict(),
                "candidates": [
                    {"entity": ent, "attribute": canonical_attr, "score": 0.0}
                    for ent in entity_candidates[:3]
                ],
            },
            "intent": intent.to_dict(),
        }

    scored_candidates: List[Dict[str, Any]] = []
    for note in matched_notes:
        value = _extract_answer_value(note)
        scored_candidates.append(
            {
                "entity": note.get("subj"),
                "attribute": canonical_attr,
                "value": value,
                "note_id": note.get("note_id"),
                "score": _score_note(note, canonical_attr),
                "confidence": (note.get("meta", {}) or {}).get("final_conf"),
            }
        )

    scored_candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    top_candidate = scored_candidates[0]
    support_note_ids = [item["note_id"] for item in scored_candidates[:5] if item.get("note_id")]

    support_notes = note_store.get_many(support_note_ids)
    evidences = [
        {
            "note_id": note.get("note_id"),
            "evidence": note.get("evidence", ""),
            "quality": (note.get("meta", {}) or {}).get("quality_score"),
        }
        for note in support_notes
    ]

    primary_note = note_store.get(top_candidate["note_id"]) if top_candidate.get("note_id") else None
    paths = []
    if primary_note:
        paths = [
            [
                {
                    "subj": primary_note.get("subj"),
                    "pred": canonical_attr,
                    "obj": _extract_answer_value(primary_note),
                    "note_id": primary_note.get("note_id"),
                    "fallback": True,
                }
            ]
        ]

    answer_value = top_candidate.get("value")
    result_reason = None if answer_value else trigger
    status = "ok" if answer_value else "no_match"

    return {
        "ir": ir.to_dict() if ir else None,
        "answer": answer_value,
        "paths": paths,
        "support_note_ids": support_note_ids,
        "evidence": evidences,
        "reason": result_reason,
        "fallback": {
            "used": True,
            "stage": trigger,
            "status": status,
            "intent": intent.to_dict(),
            "candidates": scored_candidates[:5],
        },
        "intent": intent.to_dict(),
    }


def _collect_attribute_notes(
    entity: str, attribute: str, indexes: Indexes, note_store: NoteStore
) -> List[Dict[str, Any]]:
    note_ids = indexes.entity_to_notes.get(entity, [])
    matched: List[Dict[str, Any]] = []
    for note_id in note_ids:
        note = note_store.get(note_id)
        if not note:
            continue
        meta = note.get("meta", {}) or {}
        attr = (meta.get("attribute") or {}).get("name")
        if attr != attribute:
            continue
        matched.append(note)
    return matched


def _extract_answer_value(note: Dict[str, Any]) -> Optional[str]:
    meta = note.get("meta", {}) or {}
    attribute = meta.get("attribute") or {}
    values = attribute.get("values")
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                normalized = item.get("normalized")
                if isinstance(normalized, str) and normalized.strip():
                    return normalized.strip()
                value = item.get("value")
                if isinstance(value, str) and value.strip():
                    return value.strip()
            elif isinstance(item, str) and item.strip():
                return item.strip()
    obj = note.get("obj")
    if isinstance(obj, str):
        return obj.strip()
    return None


def _score_note(note: Dict[str, Any], attribute: str) -> float:
    meta = note.get("meta", {}) or {}
    score = float(meta.get("final_conf", 0.0))
    quality = meta.get("quality_score")
    if isinstance(quality, (float, int)):
        score += 0.3 * float(quality)
    attr_name = (meta.get("attribute") or {}).get("name")
    if attr_name == attribute:
        score += 0.05
    return round(score, 4)


def _canonical_predicate(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = value.strip().lower()
    if not lowered:
        return None
    for canon, synonyms in PRED_SYNONYM_SETS.items():
        if lowered == canon or lowered in synonyms:
            return canon
    return lowered
