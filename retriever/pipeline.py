from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from loguru import logger

from schema.note_schema_v1 import PRED_SYNONYM_SETS
from schema.vocabulary import normalize_slot_value
from config.attributes_loader import get_selection_priority, allowed_values
from telemetry.metrics import record_binding_strength, record_anchor_usage, record_weak_ratio

from .ir import PredicateStep, QueryIR, Seed
from .intent_detector import AnswerIntent, AnswerIntentDetector
from .note_store import NoteStore
from .operators import BIND, EXPAND_from, Indexes
from utils.vector_search import VectorSearcher
from .parser import parse_question
from .scorer import score_path
from config import config as config_loader

if TYPE_CHECKING:
    from .hybrid import HybridRetriever


@dataclass
class Candidate:
    answer: Optional[str]
    path: List[Dict[str, Any]]
    note_ids: List[str]
    score: float
    match_strength: str = "weak"


INTENT_DETECTOR = AnswerIntentDetector()


def retrieve_answer(
    question: str,
    indexes: Indexes,
    note_store: NoteStore,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    hybrid: Optional["HybridRetriever"] = None,
    doc_hint: Optional[str] = None,
    attribute_hint: Optional[str] = None,
) -> Dict[str, Any]:
    intent = INTENT_DETECTOR.detect(question)
    ir = parse_question(question)
    normalized_doc_hint = _normalize_doc_hint(doc_hint)
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
        return _fallback_lookup(intent, indexes, note_store, None, "parse_failed", normalized_doc_hint)

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
        return _fallback_lookup(intent, indexes, note_store, ir, "no_seed_match", normalized_doc_hint)

    # 传递 doc_name 用于路径别名加权
    if attribute_hint:
        intent.attribute = attribute_hint
    doc_name = intent.entity if isinstance(intent.entity, str) else None
    normalized_doc_hint = _normalize_doc_hint(doc_hint)
    candidates = _walk_chain(seed_entities, ir, indexes, note_store, doc_name)
    candidates = _filter_candidates_by_doc(candidates, normalized_doc_hint)
    try:
        logger.info(
            "paths_found={}  notes_collected={}",
            len(candidates or []),
            sum(len(c.note_ids or []) for c in (candidates or [])),
        )
    except Exception:
        pass
    # Hybrid retrieval path (structured + embedding + BM25)
    hybrid_result = _maybe_run_hybrid(
        question,
        ir,
        intent,
        candidates,
        note_store,
        cfg=cfg,
        hybrid=hybrid,
    )
    if hybrid_result is not None:
        if normalized_doc_hint:
            _apply_doc_filter_to_result(hybrid_result, note_store, normalized_doc_hint)
        return hybrid_result

    if not candidates:
        # 结构化兜底：在绑定实体范围内做向量-only检索补全
        structured = _structured_fallback(seed_entities, intent, indexes, note_store, doc_hint=normalized_doc_hint)
        if structured:
            return structured
        return _fallback_lookup(intent, indexes, note_store, ir, "no_path", normalized_doc_hint)

    candidates.sort(key=lambda c: c.score, reverse=True)
    top_candidates = candidates[: ir.fanout]

    support_note_ids: List[str] = []
    for cand in top_candidates:
        for nid in cand.note_ids:
            if nid not in support_note_ids:
                support_note_ids.append(nid)
    support_note_ids = _filter_note_ids_by_doc(support_note_ids, normalized_doc_hint)

    # 轻量调度：放宽阈值并确保留底，避免全清空
    evidences = _schedule_evidences(
        note_store,
        support_note_ids,
        keep_at_least=max(3, ir.fanout // 2),
        doc_hint=normalized_doc_hint,
    )
    try:
        logger.info("evidence_kept={}  after_scheduler", len(evidences))
    except Exception:
        pass
    weak_evidences: List[Dict[str, Any]] = []
    if not support_note_ids or len(evidences) < max(3, ir.fanout // 2):
        predicate_hints = [step.pred for step in (ir.pred_chain or [])] if ir and ir.pred_chain else []
        weak_evidences = _collect_weak_evidences(
            seed_entities,
            predicate_hints,
            indexes,
            note_store,
            normalized_doc_hint,
        )
    evidences, weak_ids, weak_ratio = _merge_evidence_with_weak(evidences, weak_evidences, max_ratio=0.3)
    if weak_ratio > 0:
        record_weak_ratio(weak_ratio)
    support_note_ids = _merge_support_ids(support_note_ids, weak_ids)

    result = {
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
    if normalized_doc_hint:
        _apply_doc_filter_to_result(result, note_store, normalized_doc_hint)
    return result


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
    doc_hint: Optional[str],
) -> Dict[str, Any]:
    doc_hint_norm = _normalize_doc_hint(doc_hint)
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
        result = {
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
        _apply_doc_filter_to_result(result, note_store, doc_hint_norm)
        return result

    if not canonical_attr:
        status = "attribute_not_detected"
        result = {
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
        _apply_doc_filter_to_result(result, note_store, doc_hint_norm)
        return result

    matched_notes: List[Dict[str, Any]] = []
    for entity in entity_candidates:
        matched_notes.extend(_collect_attribute_notes(entity, canonical_attr, indexes, note_store))
    if doc_hint_norm:
        matched_notes = [note for note in matched_notes if _note_matches_source(note, doc_hint_norm)]

    if not matched_notes:
        status = "no_attribute_match"
        weak_evidences = _collect_weak_evidences(
            entity_candidates,
            [canonical_attr] if canonical_attr else None,
            indexes,
            note_store,
            doc_hint_norm,
        )
        if weak_evidences:
            result = {
                "ir": ir.to_dict() if ir else None,
                "answer": None,
                "paths": [],
                "support_note_ids": [ev["note_id"] for ev in weak_evidences if ev.get("note_id")],
                "evidence": weak_evidences,
                "reason": trigger,
                "fallback": {
                    "used": True,
                    "stage": trigger,
                    "status": "weak_index",
                    "intent": intent.to_dict(),
                    "candidates": [{"entity": ent, "attribute": canonical_attr, "score": 0.0} for ent in entity_candidates[:3]],
                },
                "intent": intent.to_dict(),
            }
            _apply_doc_filter_to_result(result, note_store, doc_hint_norm)
            return result
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
        if doc_hint_norm:
            neighbor_notes = [note for note in neighbor_notes if _note_matches_source(note, doc_hint_norm)]
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
        result = {
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
        _apply_doc_filter_to_result(result, note_store, doc_hint_norm)
        return result

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

    support_note_ids = _filter_note_ids_by_doc(support_note_ids, doc_hint_norm)
    evidences = _schedule_evidences(note_store, support_note_ids, keep_at_least=3, doc_hint=doc_hint_norm)

    primary_note = note_store.get(top_candidate["note_id"]) if top_candidate.get("note_id") else None
    paths = []
    if primary_note and (not doc_hint_norm or _note_matches_source(primary_note, doc_hint_norm)):
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

    result = {
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
    _apply_doc_filter_to_result(result, note_store, doc_hint_norm)
    return result


def _collect_weak_evidences(
    entities: Optional[List[str]],
    predicates: Optional[List[str]],
    indexes: Indexes,
    note_store: NoteStore,
    doc_hint: Optional[str],
    limit: int = 15,
) -> List[Dict[str, Any]]:
    weak_entities = getattr(indexes, "weak_entity_to_notes", {}) or {}
    weak_preds = getattr(indexes, "weak_predicate_to_notes", {}) or {}
    if not weak_entities and not weak_preds:
        return []
    evidences: List[Dict[str, Any]] = []
    normalized_hint = _normalize_doc_hint(doc_hint)
    scores: Dict[str, float] = {}
    entities = entities or []
    for ent in entities[:10]:
        refs = weak_entities.get(ent) or []
        for nid, weight in _iter_weak_refs(refs):
            if not nid:
                continue
            score = max(weight, 0.25)
            if score > scores.get(nid, 0.0):
                scores[nid] = score
    for pred in (predicates or [])[:5]:
        refs = weak_preds.get(pred) or []
        for nid, weight in _iter_weak_refs(refs):
            if not nid:
                continue
            score = max(weight + 0.1, 0.3)
            if score > scores.get(nid, 0.0):
                scores[nid] = score

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for nid, score in ordered:
        note = note_store.get_weak(nid)
        if not note:
            continue
        if normalized_hint and not _note_matches_source(note, normalized_hint):
            continue
        meta = (note.get("meta", {}) or {})
        evidences.append(
            {
                "note_id": nid,
                "evidence": note.get("evidence", ""),
                "canonical": meta.get("evidence_canonical") or note.get("evidence", ""),
                "quality": meta.get("quality_score"),
                "anchor_entity": meta.get("anchor_entity"),
                "weak": True,
                "score": round(score, 3),
            }
        )
        if len(evidences) >= limit:
            break
    return evidences


def _iter_weak_refs(refs: Any):
    if isinstance(refs, dict):
        for nid, weight in refs.items():
            try:
                yield nid, float(weight)
            except (TypeError, ValueError):
                yield nid, 0.0
        return
    if not isinstance(refs, list):
        return
    for ref in refs:
        if isinstance(ref, dict):
            nid = ref.get("note_id")
            try:
                weight = float(ref.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            yield nid, weight
        else:
            yield ref, 0.3


def _merge_evidence_with_weak(
    strong: List[Dict[str, Any]],
    weak: List[Dict[str, Any]],
    max_ratio: float = 0.3,
) -> Tuple[List[Dict[str, Any]], List[str], float]:
    if not weak:
        return strong, [], 0.0
    strong = strong or []
    strong_ids = {ev.get("note_id") for ev in strong if ev.get("note_id")}
    filtered = [ev for ev in weak if ev.get("note_id") not in strong_ids]
    if not strong:
        allowed = min(len(filtered), 10)
        if allowed == 1 and len(filtered) >= 2:
            allowed = 2
    else:
        allowed = max(1, int(len(strong) * max_ratio))
    selected = filtered[:allowed]
    merged = strong + selected
    weak_ids = [ev.get("note_id") for ev in selected if ev.get("note_id")]
    ratio = len(selected) / max(1, len(merged))
    return merged, weak_ids, ratio


def _merge_support_ids(primary: List[str], extra: List[str]) -> List[str]:
    merged = list(primary) if primary else []
    for nid in extra:
        if nid and nid not in merged:
            merged.append(nid)
    return merged


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


def _structured_fallback(
    entities: List[str],
    intent: AnswerIntent,
    indexes: Indexes,
    note_store: NoteStore,
    doc_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # 仅在结构化实体范围内进行弱信号补全（Vector-only）
    canonical_attr = intent.attribute
    if not canonical_attr:
        return None
    scoped_notes: List[Dict[str, Any]] = []
    normalized_doc_hint = _normalize_doc_hint(doc_hint)
    for ent in entities[:10]:
        for nid in indexes.entity_to_notes.get(ent, [])[:200]:
            note = note_store.get(nid)
            if not note:
                continue
            if normalized_doc_hint and not _note_matches_source(note, normalized_doc_hint):
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
    if not top_note or (normalized_doc_hint and not _note_matches_source(top_note, normalized_doc_hint)):
        return None
    answer_value = _extract_answer_value(top_note)
    paths = [[{"subj": top_note.get("subj"), "pred": canonical_attr, "obj": answer_value, "note_id": top_note.get("note_id"), "fallback": True}]] if answer_value else []
    support_note_ids = [note.get("note_id") for note, _ in ranked[:5] if note.get("note_id")]
    support_note_ids = _filter_note_ids_by_doc(support_note_ids, normalized_doc_hint)
    evidences = []
    for note, _ in ranked[:5]:
        if normalized_doc_hint and not _note_matches_source(note, normalized_doc_hint):
            continue
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


def _maybe_run_hybrid(
    question,
    ir,
    intent,
    candidates,
    note_store,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    hybrid: Optional["HybridRetriever"] = None,
):
    cfg_obj = cfg or getattr(hybrid, "cfg", None) or config_loader.load_config()
    retr_cfg = cfg_obj.get("retriever") or {}
    embedding_on = bool((retr_cfg.get("embedding") or {}).get("enabled"))
    bm25_on = bool((retr_cfg.get("bm25") or {}).get("enabled"))
    rerank_on = bool((cfg_obj.get("reranker") or {}).get("enabled"))
    if not (embedding_on or bm25_on or rerank_on):
        return None
    hybrid_inst = hybrid
    if hybrid_inst is None:
        try:
            from .hybrid import HybridRetriever
        except Exception as exc:
            logger.error("Hybrid retriever unavailable: {}", exc)
            return None
        hybrid_inst = HybridRetriever(cfg_obj)
    return hybrid_inst.retrieve(question, ir, intent, candidates, note_store)


def _rerank_candidates(candidates: List[Candidate], attribute: Optional[str]) -> List[Candidate]:
    if not attribute or not candidates:
        return candidates
    priorities = [val for val in get_selection_priority(attribute) if val]
    if not priorities:
        return candidates
    priority_index = {val: idx for idx, val in enumerate(priorities)}

    def sort_key(c: Candidate) -> tuple[int, float]:
        label = _candidate_label(c, attribute)
        rank = priority_index.get(label, len(priority_index))
        return (rank, -c.score)

    return sorted(candidates, key=sort_key)


def _schedule_evidences(
    note_store: NoteStore,
    note_ids: List[str],
    keep_at_least: int = 3,
    doc_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # 放宽置信阈值、轻度去重，并设置留底下限，避免全清空
    raw_notes = note_store.get_many(note_ids)
    normalized_doc_hint = _normalize_doc_hint(doc_hint)
    if normalized_doc_hint:
        raw_notes = [note for note in raw_notes if _note_matches_source(note, normalized_doc_hint)]
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
        record_anchor_usage(bool(meta.get("anchor")))
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
            if normalized_doc_hint and not _note_matches_source(note, normalized_doc_hint):
                continue
            nid = note.get("note_id")
            if any(ev.get("note_id") == nid for ev in kept):
                continue
            meta = (note.get("meta", {}) or {})
            record_anchor_usage(bool(meta.get("anchor")))
            kept.append({
                "note_id": nid,
                "evidence": note.get("evidence", ""),
                "canonical": meta.get("evidence_canonical") or note.get("evidence", ""),
                "quality": meta.get("quality_score"),
                "lead_in_note_id": meta.get("lead_in_note_id"),
            })
            if len(kept) >= keep_at_least:
                break
    return kept


def _normalize_doc_hint(doc_hint: Optional[str]) -> Optional[str]:
    if not doc_hint:
        return None
    return doc_hint.strip().lower()


def _note_id_matches_doc(note_id: Optional[str], doc_hint: Optional[str]) -> bool:
    if not doc_hint or not note_id:
        return True
    doc_part = note_id.split("#", 1)[0].lower()
    return doc_hint in doc_part


def _note_matches_source(note: Optional[Dict[str, Any]], doc_hint: Optional[str]) -> bool:
    if not doc_hint or not note:
        return True
    meta = (note.get("meta", {}) or {})
    source = (meta.get("source") or "").strip().lower()
    if source:
        return doc_hint in source
    return _note_id_matches_doc(note.get("note_id"), doc_hint)


def _filter_note_ids_by_doc(note_ids: List[str], doc_hint: Optional[str]) -> List[str]:
    if not doc_hint:
        return note_ids
    return [nid for nid in note_ids if _note_id_matches_doc(nid, doc_hint)]


def _filter_candidates_by_doc(candidates: List[Candidate], doc_hint: Optional[str]) -> List[Candidate]:
    if not doc_hint:
        return candidates
    filtered: List[Candidate] = []
    for cand in candidates:
        if not cand.note_ids:
            record_binding_strength("unknown")
            filtered.append(cand)
            continue
        if all(_note_id_matches_doc(nid, doc_hint) for nid in cand.note_ids if nid):
            cand.match_strength = "strong"
            record_binding_strength("strong")
            filtered.append(cand)
        else:
            record_binding_strength("weak_drop")
    return filtered


def _apply_doc_filter_to_result(result: Dict[str, Any], note_store: NoteStore, doc_hint: Optional[str]) -> None:
    if not doc_hint or not result:
        return
    doc_hint_norm = _normalize_doc_hint(doc_hint)
    paths = result.get("paths") or []
    filtered_paths: List[List[Dict[str, Any]]] = []
    for path in paths:
        if not path:
            continue
        note_ids = [edge.get("note_id") for edge in path if edge.get("note_id")]
        if not note_ids or all(_note_id_matches_doc(nid, doc_hint_norm) for nid in note_ids):
            filtered_paths.append(path)
    result["paths"] = filtered_paths
    result["support_note_ids"] = _filter_note_ids_by_doc(result.get("support_note_ids") or [], doc_hint_norm)
    evidences = result.get("evidence") or []
    result["evidence"] = [ev for ev in evidences if _note_id_matches_doc(ev.get("note_id"), doc_hint_norm)]
    if filtered_paths:
        # sync answer with first valid path
        first = filtered_paths[0]
        if first:
            result["answer"] = first[-1].get("obj")
    elif result.get("support_note_ids"):
        # attempt to derive answer from surviving support notes
        surviving = note_store.get(result["support_note_ids"][0])
        result["answer"] = _extract_answer_value(surviving) if surviving else result.get("answer")
    else:
        result["answer"] = None


def _candidate_label(candidate: Candidate, attribute: str) -> str:
    if not candidate.path:
        return ""
    obj = candidate.path[-1].get("obj")
    if not isinstance(obj, str):
        return ""
    normalized, _ = normalize_slot_value(attribute, obj)
    text = (normalized or obj or "").strip().lower()
    return text
