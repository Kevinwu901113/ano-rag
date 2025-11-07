from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from loguru import logger

from schema.note_schema_v1 import PRED_SYNONYM_SETS

from .ir import PredicateStep, QueryIR, Seed
from .intent_detector import AnswerIntent, AnswerIntentDetector
from .note_store import NoteStore
from .operators import BIND, EXPAND_from, Indexes
from utils.vector_search import VectorSearcher
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
    # 逐层诊断日志（定位常见失败点）
    try:
        logger.info("Q: {}", question)
        logger.info(
            "IR.pred_chain={}  seeds={}",
            [(s.pred, s.direction) for s in (ir.pred_chain or [])] if ir else None,
            [seed.text for seed in (ir.seeds or [])] if ir else None,
        )
    except Exception:
        pass
    if ir is None or not ir.is_valid:
        return _fallback_lookup(intent, indexes, note_store, None, "parse_failed")

    seed_entities = _bind_seeds(ir.seeds, indexes, ir.fanout)
    # 注入 doc_name 别名约束：若检测到实体名称，作为强别名参与绑定
    try:
        if intent.entity and isinstance(intent.entity, str) and intent.entity.strip():
            extra = BIND(indexes, intent.entity.strip(), [], limit=ir.fanout)
            for ent in extra:
                if ent not in seed_entities:
                    seed_entities.append(ent)
    except Exception:
        pass
    try:
        logger.info("seed_candidates={}", len(seed_entities))
    except Exception:
        pass
    if not seed_entities:
        # 结构化优先兜底：尝试限制在别名索引范围内的弱信号补全（向量-only）
        return _fallback_lookup(intent, indexes, note_store, ir, "no_seed_match")

    # 传递 doc_name 用于路径别名加权
    doc_name = intent.entity if isinstance(intent.entity, str) else None
    candidates = _walk_chain(seed_entities, ir, indexes, note_store, doc_name)
    try:
        logger.info(
            "paths_found={}  notes_collected={}",
            len(candidates or []),
            sum(len(c.note_ids or []) for c in (candidates or [])),
        )
    except Exception:
        pass
    if not candidates:
        # 结构化兜底：在绑定实体范围内做向量-only检索补全
        structured = _structured_fallback(seed_entities, intent, indexes, note_store)
        if structured:
            return structured
        return _fallback_lookup(intent, indexes, note_store, ir, "no_path")

    candidates.sort(key=lambda c: c.score, reverse=True)
    top_candidates = candidates[: ir.fanout]

    support_note_ids = []
    for cand in top_candidates:
        for nid in cand.note_ids:
            if nid not in support_note_ids:
                support_note_ids.append(nid)

    # 轻量调度：放宽阈值并确保留底，避免全清空
    evidences = _schedule_evidences(note_store, support_note_ids, keep_at_least=max(3, ir.fanout // 2))
    try:
        logger.info("evidence_kept={}  after_scheduler", len(evidences))
    except Exception:
        pass

    return {
        "ir": ir.to_dict(),
        "answer": top_candidates[0].answer,
        # occupation优先重排：仅在候选内部调分
        "paths": [cand.path for cand in _rerank_candidates(top_candidates, intent.attribute)],
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
    doc_name: Optional[str] = None,
) -> List[Candidate]:
    if not ir.pred_chain:
        return _collect_entity_mentions(entities, indexes, note_store, ir, doc_name)

    states = [{"entity": entity, "path": []} for entity in entities]
    for step_idx, step in enumerate(ir.pred_chain[: ir.max_hops]):
        next_states: List[Dict[str, Any]] = []
        for state in states:
            # 关系同义归一：确保检索入口与图关系名对齐
            canon_pred = _canonical_predicate(step.pred) or step.pred
            expanded = EXPAND_from(
                indexes,
                state["entity"],
                predicate=canon_pred,
                direction=step.direction,
                limit=ir.fanout,
            )
            for obj, note_id in expanded:
                new_path = state["path"] + [
                    {
                        "subj": state["entity"],
                        "pred": canon_pred,
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
        # 路径打分加入 doc_name（由上层传入的页面/标题别名约束）
        score = score_path(path, doc_name=doc_name)
        final_note = note_store.get(note_ids[-1]) if note_ids else None
        if final_note and ir.target_type:
            obj_type = final_note.get("obj_type")
            if obj_type and obj_type.upper() == ir.target_type:
                score += 0.1
        answer = path[-1]["obj"] if path else None
        candidates.append(Candidate(answer=answer, path=path, note_ids=note_ids, score=score))
    return candidates


def _collect_entity_mentions(
    entities: Sequence[str], indexes: Indexes, note_store: NoteStore, ir: QueryIR, doc_name: Optional[str] = None
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
            score = score_path(path, doc_name=doc_name)
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
        # Neighbor expansion: bring notes where pronoun unresolved near the entity mentions
        neighbor_ids: List[str] = []
        # Collect note_ids that corefer to the entity or mention it
        for ent in entity_candidates[:10]:
            for nid in indexes.entity_to_notes.get(ent, [])[:50]:
                neighbor_ids.append(nid)
        # Also include notes that mention the entity by surface forms
        mention_hits = []
        for nid, mentions in indexes.mentions_edges.items():
            if any(ent.lower() == m.lower() for ent in entity_candidates for m in mentions):
                mention_hits.append(nid)
        neighbor_ids.extend(mention_hits[:50])
        # Deduplicate
        neighbor_ids = list({nid for nid in neighbor_ids})
        neighbor_notes = note_store.get_many(neighbor_ids)
        # Apply discount score and return as tolerance evidences
        evidences = []
        for note in neighbor_notes:
            meta = (note.get("meta", {}) or {})
            if not meta.get("has_unresolved_pronoun"):
                continue
            evidences.append(
                {
                    "note_id": note.get("note_id"),
                    "evidence": note.get("evidence", ""),
                    "canonical": meta.get("evidence_canonical") or note.get("evidence", ""),
                    "quality": meta.get("quality_score"),
                    "discount": 0.5,
                    # 检索时间回填：若证据主语是代词，携带 anchor_entity / lead_in_note_id（来自meta）
                    "anchor_entity": meta.get("anchor_entity"),
                    "lead_in_note_id": meta.get("lead_in_note_id"),
                }
            )
        return {
            "ir": ir.to_dict() if ir else None,
            "answer": None,
            "paths": [],
            "support_note_ids": [e["note_id"] for e in evidences],
            "evidence": evidences,
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

    evidences = _schedule_evidences(note_store, support_note_ids, keep_at_least=3)

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
    # occupation/title（职业相关）优先加权
    if attribute == "occupation" and attr_name in {"occupation", "title"}:
        score += 0.05
    # entity_presence_score：若 subj/obj 是代词或破碎 token，扣分
    def _is_broken(token: str | None) -> bool:
        if not token:
            return True
        t = token.strip()
        if len(t) <= 1:
            return True
        # 禁止 Cone（专名）误匹配到 cone（普通词）：这里不做小写化
        if t.islower():
            return True
        return False
    if _is_broken(note.get("subj")) or _is_broken(note.get("obj")):
        score -= 0.1
    # 同名/别名加分：若 evidence 中包含 anchor_entity 或实体同名，轻微加分
    anchor = meta.get("anchor_entity")
    ev = (note.get("evidence") or "")
    if isinstance(anchor, str) and anchor and anchor in ev:
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


def _structured_fallback(entities: List[str], intent: AnswerIntent, indexes: Indexes, note_store: NoteStore) -> Optional[Dict[str, Any]]:
    # 仅在结构化实体范围内进行弱信号补全（Vector-only）
    canonical_attr = intent.attribute
    if not canonical_attr:
        return None
    scoped_notes: List[Dict[str, Any]] = []
    for ent in entities[:10]:
        for nid in indexes.entity_to_notes.get(ent, [])[:200]:
            note = note_store.get(nid)
            if not note:
                continue
            scoped_notes.append(note)
    if not scoped_notes:
        return None
    # 向量-only 在 scoped_notes 内检索（预筛与兜底）
    try:
        vs = VectorSearcher()
        ranked = vs.search_in_notes(intent.entity or "", scoped_notes, top_k=16)
    except Exception:
        ranked = []
    if not ranked:
        return None
    # occupation 优先：在兜底中仅作为轻微过滤，不改变向量分数排序
    if canonical_attr == "occupation":
        ranked = [item for item in ranked if ((item[0].get("meta", {}) or {}).get("attribute", {}) .get("name") in {"occupation", "title"})] or ranked
    top_note = ranked[0][0] if ranked else None
    if not top_note:
        return None
    answer_value = _extract_answer_value(top_note)
    paths = [[{"subj": top_note.get("subj"), "pred": canonical_attr, "obj": answer_value, "note_id": top_note.get("note_id"), "fallback": True}]] if answer_value else []
    support_note_ids = [note.get("note_id") for note, _ in ranked[:5] if note.get("note_id")]
    evidences = []
    for note, _ in ranked[:5]:
        meta = (note.get("meta", {}) or {})
        evidences.append({
            "note_id": note.get("note_id"),
            "evidence": note.get("evidence", ""),
            "canonical": meta.get("evidence_canonical") or note.get("evidence", ""),
            "quality": meta.get("quality_score"),
        })
    return {
        "ir": None,
        "answer": answer_value,
        "paths": paths,
        "support_note_ids": support_note_ids,
        "evidence": evidences,
        "reason": None if answer_value else "structured_fallback_no_match",
        "fallback": {"used": True, "stage": "structured_fallback", "status": "ok" if answer_value else "no_match", "intent": intent.to_dict(), "candidates": [{"entity": top_note.get("subj"), "attribute": canonical_attr, "note_id": top_note.get("note_id"), "score": 0.0}]},
        "intent": intent.to_dict(),
    }


def _rerank_candidates(candidates: List[Candidate], attribute: Optional[str]) -> List[Candidate]:
    if attribute != "occupation":
        return candidates
    def bonus(c: Candidate) -> float:
        last_note_id = c.note_ids[-1] if c.note_ids else None
        attr_name = None
        # We cannot load note_store here; rely on path pred hint
        if c.path:
            attr_name = c.path[-1].get("pred")
        return c.score + (0.05 if attr_name in {"occupation", "title"} else 0.0)
    return sorted(candidates, key=bonus, reverse=True)


def _schedule_evidences(note_store: NoteStore, note_ids: List[str], keep_at_least: int = 3) -> List[Dict[str, Any]]:
    # 放宽置信阈值、轻度去重，并设置留底下限，避免全清空
    raw_notes = note_store.get_many(note_ids)
    kept: List[Dict[str, Any]] = []
    seen_entities: set[str] = set()
    for note in raw_notes:
        meta = (note.get("meta", {}) or {})
        conf = meta.get("confidence")
        # min_confidence=0.3；None 视为通过
        if conf is not None and float(conf) < 0.3:
            continue
        subj = (note.get("subj") or "").strip()
        # 轻度按主体去重（非激进）
        if subj and subj in seen_entities:
            # 保留少量重复，避免过度去重
            if len(kept) >= 2:
                continue
        seen_entities.add(subj)
        kept.append({
            "note_id": note.get("note_id"),
            "evidence": note.get("evidence", ""),
            "canonical": meta.get("evidence_canonical") or note.get("evidence", ""),
            "quality": meta.get("quality_score"),
            "lead_in_note_id": meta.get("lead_in_note_id"),
        })
    # 留底：若数量不足，补齐到 keep_at_least
    if len(kept) < keep_at_least:
        for note in raw_notes:
            nid = note.get("note_id")
            if any(ev.get("note_id") == nid for ev in kept):
                continue
            kept.append({
                "note_id": nid,
                "evidence": note.get("evidence", ""),
                "canonical": (note.get("meta", {}) or {}).get("evidence_canonical") or note.get("evidence", ""),
                "quality": (note.get("meta", {}) or {}).get("quality_score"),
                "lead_in_note_id": (note.get("meta", {}) or {}).get("lead_in_note_id"),
            })
            if len(kept) >= keep_at_least:
                break
    return kept
