from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from loguru import logger

from relrag.schema.note_schema_v1 import PRED_SYNONYM_SETS
from relrag.schema.vocabulary import normalize_slot_value
from relrag.config.attributes_loader import get_selection_priority, allowed_values
from relrag.telemetry.metrics import (
    record_binding_strength,
    record_anchor_usage,
    record_weak_ratio,
    record_retrieval_total,
    record_retrieval_no_path,
    record_retrieval_empty_context,
    export_metrics,
)

from .ir import PredicateStep, QueryIR, Seed
from .intent_detector import AnswerIntent, AnswerIntentDetector
from .note_store import NoteStore
from .chunk_store import ChunkStore
from .operators import BIND, EXPAND_from, Indexes
from relrag.utils.vector_search import VectorSearcher
from .parser import parse_question
from .scorer import score_path
from relrag.config import config as config_loader

if TYPE_CHECKING:
    from .hybrid import HybridRetriever


@dataclass
class Candidate:
    answer: Optional[str]
    path: List[Dict[str, Any]]
    note_ids: List[str]
    score: float
    match_strength: str = "weak"
    path_metrics: Dict[str, float] = field(default_factory=dict)


INTENT_DETECTOR = AnswerIntentDetector()


def _normalize_name(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def _build_seed_alias_lookup(
    indexes: Indexes,
    entities: Sequence[str],
    seed_texts: Sequence[str],
    doc_name: Optional[str],
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for entity in entities:
        norm = _normalize_name(entity)
        if norm:
            lookup.setdefault(norm, entity)
    alias_index = getattr(indexes, "alias_to_entities", {}) or {}
    for alias, mapped_entities in alias_index.items():
        if not isinstance(mapped_entities, list):
            continue
        if any(entity in entities for entity in mapped_entities):
            lookup.setdefault(alias.lower(), mapped_entities[0])
    for seed in seed_texts:
        norm = _normalize_name(seed)
        if norm:
            lookup.setdefault(norm, seed)
    if doc_name:
        norm = _normalize_name(doc_name)
        if norm:
            lookup.setdefault(norm, doc_name)
    return lookup


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
    cfg = cfg or config_loader.load_config()
    
    # Apply top_k override if present in config
    if cfg:
        structured_cfg = (cfg.get("retriever") or {}).get("structured") or {}
        top_k = structured_cfg.get("top_k")
        if top_k is not None and isinstance(top_k, int) and ir:
            ir.fanout = top_k

    retr_cfg = cfg.get("retriever") or {}
    structured_cfg = retr_cfg.get("structured") or {}
    embedding_cfg = retr_cfg.get("embedding") or {}
    structured_enabled = bool(structured_cfg.get("enabled", True))
    walk_enabled = bool(structured_cfg.get("walk_enabled", True))
    multihop_rescue_enabled = bool(structured_cfg.get("multihop_rescue_enabled", True))
    entity_match_threshold = float(structured_cfg.get("entity_match_threshold", 0.5))
    path_consistency_threshold = float(structured_cfg.get("path_consistency_threshold", 0.9))
    vector_fallback_enabled = bool(structured_cfg.get("vector_fallback_enabled", True))
    predicate_mode = str(structured_cfg.get("predicate_mode", "on") or "on").strip().lower()
    if predicate_mode not in {"on", "off", "random"}:
        predicate_mode = "on"
    if not bool(structured_cfg.get("predicate_constraint_enabled", True)) and predicate_mode == "on":
        predicate_mode = "off"
    try:
        predicate_random_seed = int(structured_cfg.get("random_predicate_seed", 2026))
    except (TypeError, ValueError):
        predicate_random_seed = 2026
    normalized_doc_hint = _normalize_doc_hint(doc_hint)
    relaxed_path_used = False
    rescue_used = False

    def _finalize_result(
        result: Dict[str, Any],
        *,
        ir_override: Optional[QueryIR] = None,
        intent_override: Optional[AnswerIntent] = None,
    ) -> Dict[str, Any]:
        active_ir = ir_override if ir_override is not None else ir
        active_intent = intent_override if intent_override is not None else intent
        result_meta = result.setdefault("meta", {})
        result_meta["multihop_rescue"] = rescue_used
        result_meta["walk_enabled"] = walk_enabled
        result_meta["multihop_rescue_enabled"] = multihop_rescue_enabled
        result_meta["predicate_mode"] = predicate_mode
        result_meta["predicate_constraint_enabled"] = predicate_mode != "off"
        result_meta["predicate_random_seed"] = predicate_random_seed
        if normalized_doc_hint:
            _apply_doc_filter_to_result(result, note_store, normalized_doc_hint)
        result = _apply_chunk_fallback(
            result,
            question=question,
            ir=active_ir,
            intent=active_intent,
            note_store=note_store,
            doc_hint=normalized_doc_hint,
            cfg=cfg,
        )
        _record_retrieval_metrics(result)
        return result
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
        result = _fallback_lookup(intent, indexes, note_store, None, "parse_failed", normalized_doc_hint, cfg=cfg)
        result.setdefault("meta", {})["relaxed_path_retry"] = False
        return _finalize_result(result, ir_override=ir, intent_override=intent)

    walk_ir = ir
    if structured_enabled and not walk_enabled and ir.pred_chain:
        walk_ir = QueryIR(
            intent=ir.intent,
            seeds=list(ir.seeds or []),
            pred_chain=[],
            target_type=ir.target_type,
            question_type=ir.question_type,
            max_hops=0,
            fanout=ir.fanout,
            raw=ir.raw,
            fallback=ir.fallback,
        )

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
    if not seed_entities and structured_enabled:
        try:
            logger.info("no seed entities bound; trigger fallback (doc_hint={})", normalized_doc_hint)
        except Exception:
            pass
        # 结构化优先兜底：尝试限制在别名索引范围内的弱信号补全（向量-only）
        result = _fallback_lookup(intent, indexes, note_store, ir, "no_seed_match", normalized_doc_hint, cfg=cfg)
        result.setdefault("meta", {})["relaxed_path_retry"] = False
        return _finalize_result(result, ir_override=ir, intent_override=intent)

    # 传递 doc_name 用于路径别名加权
    if attribute_hint:
        intent.attribute = attribute_hint
    doc_name = intent.entity if isinstance(intent.entity, str) else None
    normalized_doc_hint = _normalize_doc_hint(doc_hint)
    seed_texts = [seed.text for seed in ir.seeds if seed.text]
    alias_lookup = _build_seed_alias_lookup(indexes, seed_entities, seed_texts, doc_name)
    candidates: List[Candidate] = []
    if structured_enabled:
        candidates = _walk_chain(
            seed_entities,
            walk_ir,
            indexes,
            note_store,
            doc_name,
            attribute=intent.attribute,
            seed_texts=seed_texts,
            alias_lookup=alias_lookup,
            entity_match_threshold=entity_match_threshold,
            path_match_threshold=path_consistency_threshold if walk_ir.pred_chain else -1.0,
            predicate_mode=predicate_mode,
            predicate_random_seed=predicate_random_seed,
        )
        pre_doc_candidates = len(candidates or [])
        candidates = _filter_candidates_by_doc(candidates, normalized_doc_hint)
        try:
            if pre_doc_candidates and not candidates:
                logger.info("all {} structured candidates dropped by doc_hint filter", pre_doc_candidates)
            elif pre_doc_candidates != len(candidates or []):
                logger.info("structured candidates filtered by doc_hint: {} -> {}", pre_doc_candidates, len(candidates or []))
        except Exception:
            pass
        if not candidates and walk_ir.pred_chain:
            try:
                logger.info("no structured path; retrying with relaxed thresholds")
            except Exception:
                pass
            relaxed_candidates = _walk_chain(
                seed_entities,
                walk_ir,
                indexes,
                note_store,
                doc_name,
                attribute=intent.attribute,
                seed_texts=seed_texts,
                alias_lookup=alias_lookup,
                entity_match_threshold=max(0.25, entity_match_threshold * 0.6),
                path_match_threshold=0.25,
                predicate_mode=predicate_mode,
                predicate_random_seed=predicate_random_seed,
            )
            relaxed_candidates = _filter_candidates_by_doc(relaxed_candidates, normalized_doc_hint)
            if relaxed_candidates:
                candidates = relaxed_candidates
                relaxed_path_used = True
                try:
                    logger.info("relaxed retry yielded {} candidates", len(relaxed_candidates))
                except Exception:
                    pass
        if (
            multihop_rescue_enabled
            and not candidates
            and walk_ir.pred_chain
            and len(walk_ir.pred_chain) >= 2
        ):
            rescued = _rescue_multihop(
                seed_entities,
                walk_ir,
                indexes,
                note_store,
                seed_texts=seed_texts,
                alias_lookup=alias_lookup,
                attribute=intent.attribute,
                entity_match_threshold=0.0,
                path_match_threshold=path_consistency_threshold if walk_ir.pred_chain else -1.0,
                predicate_mode=predicate_mode,
                predicate_random_seed=predicate_random_seed,
            )
            rescued = _filter_candidates_by_doc(rescued, normalized_doc_hint)
            if rescued:
                candidates = rescued
                rescue_used = True
                try:
                    logger.info("multihop rescue yielded {} candidates", len(rescued))
                except Exception:
                    pass
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
        walk_ir,
        intent,
        candidates,
        note_store,
        cfg=cfg,
        hybrid=hybrid,
        alias_lookup=alias_lookup,
    )
    if hybrid_result is not None:
        hybrid_result.setdefault("meta", {"path_consistency": 0.0, "entity_consistency": 0.0})
        hybrid_result["meta"]["relaxed_path_retry"] = relaxed_path_used
        return _finalize_result(hybrid_result, ir_override=ir, intent_override=intent)

    if not candidates:
        # 结构化兜底：在绑定实体范围内做向量-only检索补全
        structured = (
            _structured_fallback(
                seed_entities,
                intent,
                indexes,
                note_store,
                doc_hint=normalized_doc_hint,
                embedding_cfg=embedding_cfg,
            )
            if structured_enabled and vector_fallback_enabled
            else None
        )
        if structured:
            structured.setdefault("meta", {})["relaxed_path_retry"] = relaxed_path_used
            return _finalize_result(structured, ir_override=ir, intent_override=intent)
        result = _fallback_lookup(intent, indexes, note_store, ir, "no_path", normalized_doc_hint, cfg=cfg)
        result.setdefault("meta", {})["relaxed_path_retry"] = relaxed_path_used
        return _finalize_result(result, ir_override=ir, intent_override=intent)

    candidates.sort(key=lambda c: c.score, reverse=True)
    top_candidates = _select_top_candidates(candidates, ir.fanout, seed_entities)

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
        cfg=cfg,
    )
    try:
        logger.info("evidence_kept={}  after_scheduler", len(evidences))
    except Exception:
        pass
    weak_evidences: List[Dict[str, Any]] = []
    if not support_note_ids or len(evidences) < max(3, ir.fanout // 2):
        predicate_hints = (
            [step.pred for step in (walk_ir.pred_chain or [])]
            if walk_ir and walk_ir.pred_chain
            else []
        )
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
    _attach_meta(result, top_candidates[0].path_metrics if top_candidates else None)
    result.setdefault("meta", {})["relaxed_path_retry"] = relaxed_path_used
    return _finalize_result(result, ir_override=ir, intent_override=intent)


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


def _resolve_expand_predicate(
    requested_predicate: str,
    indexes: Indexes,
    *,
    predicate_mode: str,
    predicate_random_seed: int,
    step_idx: int,
    entity: str,
) -> Optional[str]:
    mode = str(predicate_mode or "on").strip().lower()
    if mode == "off":
        return None
    if mode != "random":
        return requested_predicate
    predicates = [
        str(name).strip()
        for name in (indexes.predicate_to_notes or {}).keys()
        if str(name).strip()
    ]
    if not predicates:
        return requested_predicate
    predicates = sorted(dict.fromkeys(predicates))
    if len(predicates) == 1:
        return predicates[0]
    material = f"{predicate_random_seed}|{step_idx}|{entity}|{requested_predicate}"
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()
    selected_idx = int(digest[:12], 16) % len(predicates)
    selected = predicates[selected_idx]
    if selected == requested_predicate:
        selected = predicates[(selected_idx + 1) % len(predicates)]
    return selected


def _walk_chain(
    entities: Sequence[str],
    ir: QueryIR,
    indexes: Indexes,
    note_store: NoteStore,
    doc_name: Optional[str] = None,
    attribute: Optional[str] = None,
    *,
    seed_texts: Optional[Sequence[str]] = None,
    alias_lookup: Optional[Dict[str, str]] = None,
    entity_match_threshold: float = 0.0,
    path_match_threshold: float = -1.0,
    predicate_mode: str = "on",
    predicate_random_seed: int = 2026,
) -> List[Candidate]:
    if not ir.pred_chain:
        return _collect_entity_mentions(
            entities,
            indexes,
            note_store,
            ir,
            doc_name,
            seed_texts=seed_texts,
            alias_lookup=alias_lookup,
            entity_match_threshold=entity_match_threshold,
        )

    states = [{"entity": entity, "path": []} for entity in entities]
    for step_idx, step in enumerate(ir.pred_chain[: ir.max_hops]):
        next_states: List[Dict[str, Any]] = []
        for state in states:
            # 关系同义归一：确保检索入口与图关系名对齐
            canon_pred = _canonical_predicate(step.pred) or step.pred
            expand_pred = _resolve_expand_predicate(
                canon_pred,
                indexes,
                predicate_mode=predicate_mode,
                predicate_random_seed=predicate_random_seed,
                step_idx=step_idx,
                entity=str(state["entity"]),
            )
            expanded = EXPAND_from(
                indexes,
                state["entity"],
                predicate=expand_pred,
                direction=step.direction,
                limit=ir.fanout,
            )
            for obj, note_id, edge_conf in expanded:
                new_path = state["path"] + [
                    {
                        "subj": state["entity"],
                        "pred": canon_pred,
                        "obj": obj,
                        "note_id": note_id,
                        "conf": edge_conf,
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
        notes = note_store.get_many(note_ids)
        score, metrics = score_path(
            path,
            notes=notes,
            doc_name=doc_name,
            seeds=seed_texts,
            query_ir=ir,
            alias_lookup=alias_lookup,
        )
        if metrics.get("entity_score", 0.0) < entity_match_threshold:
            continue
        if ir.pred_chain and metrics.get("pred_score", 0.0) < path_match_threshold:
            continue
        final_note = note_store.get(note_ids[-1]) if note_ids else None
        score += _attribute_note_bonus(final_note, attribute, indexes)
        if final_note and ir.target_type:
            obj_type = final_note.get("obj_type")
            if obj_type and obj_type.upper() == ir.target_type:
                score += 0.1
        answer = path[-1]["obj"] if path else None
        candidates.append(Candidate(answer=answer, path=path, note_ids=note_ids, score=score, path_metrics=metrics))
    return candidates


def _rescue_multihop(
    entities: Sequence[str],
    ir: QueryIR,
    indexes: Indexes,
    note_store: NoteStore,
    *,
    seed_texts: Optional[Sequence[str]] = None,
    alias_lookup: Optional[Dict[str, str]] = None,
    attribute: Optional[str] = None,
    entity_match_threshold: float = 0.0,
    path_match_threshold: float = -1.0,
    predicate_mode: str = "on",
    predicate_random_seed: int = 2026,
) -> List[Candidate]:
    if not ir.pred_chain or len(ir.pred_chain) < 2:
        return []
    hop1 = ir.pred_chain[0]
    hop2 = ir.pred_chain[1]
    hop1_pred = _canonical_predicate(hop1.pred) or hop1.pred
    hop2_pred = _canonical_predicate(hop2.pred) or hop2.pred

    intermediates: List[Tuple[str, str, str, float]] = []
    seen_intermediate = set()
    for subj in entities:
        hop1_expand_pred = _resolve_expand_predicate(
            hop1_pred,
            indexes,
            predicate_mode=predicate_mode,
            predicate_random_seed=predicate_random_seed,
            step_idx=0,
            entity=str(subj),
        )
        expanded = EXPAND_from(indexes, subj, hop1_expand_pred, direction=hop1.direction, limit=ir.fanout)
        for mid, nid, conf in expanded:
            key = (subj, mid, nid)
            if key in seen_intermediate:
                continue
            seen_intermediate.add(key)
            intermediates.append((subj, mid, nid, conf))

    if not intermediates:
        return []
    max_intermediate = max(10, min(20, ir.fanout * 2))
    intermediates = intermediates[:max_intermediate]

    candidates: List[Candidate] = []
    for subj0, mid, nid1, conf1 in intermediates:
        hop2_expand_pred = _resolve_expand_predicate(
            hop2_pred,
            indexes,
            predicate_mode=predicate_mode,
            predicate_random_seed=predicate_random_seed,
            step_idx=1,
            entity=str(mid),
        )
        expanded2 = EXPAND_from(indexes, mid, hop2_expand_pred, direction=hop2.direction, limit=ir.fanout)
        for obj2, nid2, conf2 in expanded2:
            path = [
                {"subj": subj0, "pred": hop1_pred, "obj": mid, "note_id": nid1, "conf": conf1},
                {"subj": mid, "pred": hop2_pred, "obj": obj2, "note_id": nid2, "conf": conf2},
            ]
            note_ids = [nid1, nid2]
            notes = note_store.get_many(note_ids)
            score, metrics = score_path(
                path,
                notes=notes,
                doc_name=None,
                seeds=seed_texts,
                query_ir=ir,
                alias_lookup=alias_lookup,
            )
            if metrics.get("entity_score", 0.0) < entity_match_threshold:
                continue
            if ir.pred_chain and metrics.get("pred_score", 0.0) < path_match_threshold:
                continue
            final_note = note_store.get(nid2) if nid2 else None
            score += _attribute_note_bonus(final_note, attribute, indexes)
            answer = path[-1]["obj"] if path else None
            candidates.append(
                Candidate(
                    answer=answer,
                    path=path,
                    note_ids=note_ids,
                    score=score,
                    path_metrics=metrics,
                )
            )
    return candidates


def _collect_entity_mentions(
    entities: Sequence[str],
    indexes: Indexes,
    note_store: NoteStore,
    ir: QueryIR,
    doc_name: Optional[str] = None,
    *,
    seed_texts: Optional[Sequence[str]] = None,
    alias_lookup: Optional[Dict[str, str]] = None,
    entity_match_threshold: float = 0.0,
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
                    "conf": 0.6,
                }
            ]
            note = note_store.get(nid)
            score, metrics = score_path(
                path,
                notes=[note] if note else None,
                doc_name=doc_name,
                seeds=seed_texts,
                alias_lookup=alias_lookup,
            )
            if metrics.get("entity_score", 0.0) < entity_match_threshold:
                continue
            candidates.append(
                Candidate(answer=None, path=path, note_ids=[nid], score=score, path_metrics=metrics)
            )
    return candidates


def _fallback_lookup(
    intent: AnswerIntent,
    indexes: Indexes,
    note_store: NoteStore,
    ir: Optional[QueryIR],
    trigger: str,
    doc_hint: Optional[str],
    cfg: Optional[Dict[str, Any]] = None,
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
        _attach_meta(result, None)
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
        _attach_meta(result, None)
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
            _attach_meta(result, None)
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
        _attach_meta(result, None)
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
    evidences = _schedule_evidences(
        note_store,
        support_note_ids,
        keep_at_least=3,
        doc_hint=doc_hint_norm,
        cfg=cfg,
    )

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
    _attach_meta(result, None)
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
                "subj": note.get("subj"),
                "pred": note.get("pred"),
                "obj": note.get("obj"),
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


def _attribute_note_bonus(note: Optional[Dict[str, Any]], attribute: Optional[str], indexes: Optional[Indexes]) -> float:
    if not note or not attribute:
        return 0.0
    attr = attribute.strip().lower()
    meta = note.get("meta", {}) or {}
    attr_name = ((meta.get("attribute") or {}).get("name") or "").strip().lower()
    quality = meta.get("quality_score")
    bonus = 0.0

    if attr == "occupation":
        if isinstance(quality, (float, int)):
            bonus += 0.25 * float(quality)
        quality_flags = meta.get("quality") or {}
        if isinstance(quality_flags, dict) and quality_flags.get("has_definition"):
            bonus += 0.05
        if attr_name == "occupation":
            bonus += 0.08
        elif attr_name == "title":
            bonus += 0.04
        else:
            bonus -= 0.08
        bonus += _field_hit_bias(note, "occupation", indexes)
    else:
        if isinstance(quality, (float, int)):
            bonus += 0.1 * float(quality)
        if attr_name == attr:
            bonus += 0.03
    return bonus


def _field_hit_bias(note: Dict[str, Any], attribute: str, indexes: Optional[Indexes]) -> float:
    if not indexes or not attribute:
        return 0.0
    field_index = getattr(indexes, "field_index", {}) or {}
    bucket = field_index.get(attribute) or {}
    if not bucket:
        return 0.0
    nid = note.get("note_id")
    if not nid:
        return 0.0
    hit = any(isinstance(ids, list) and nid in ids for ids in bucket.values())
    if hit:
        return 0.05
    return -0.05


def _score_note(note: Dict[str, Any], attribute: str) -> float:
    meta = note.get("meta", {}) or {}
    score = float(meta.get("final_conf", 0.0))
    quality = meta.get("quality_score")
    attr = (attribute or "").strip().lower()
    qual_weight = 0.4 if attr == "occupation" else 0.3
    if isinstance(quality, (float, int)):
        score += qual_weight * float(quality)
    attr_name = ((meta.get("attribute") or {}).get("name") or "").strip().lower()
    if attr == "occupation":
        if attr_name in {"occupation", "title"}:
            score += 0.1
        else:
            score -= 0.05
        quality_flags = meta.get("quality") or {}
        if isinstance(quality_flags, dict) and quality_flags.get("has_definition"):
            score += 0.05
    elif attr_name == attr:
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
    if meta.get("weak"):
        score *= 0.6
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
    embedding_cfg: Optional[Dict[str, Any]] = None,
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
        vs = VectorSearcher(embedding_cfg=embedding_cfg)
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
    result = {
        "ir": None,
        "answer": answer_value,
        "paths": paths,
        "support_note_ids": support_note_ids,
        "evidence": evidences,
        "reason": None if answer_value else "structured_fallback_no_match",
        "fallback": {"used": True, "stage": "structured_fallback", "status": "ok" if answer_value else "no_match", "intent": intent.to_dict(), "candidates": [{"entity": top_note.get("subj"), "attribute": canonical_attr, "note_id": top_note.get("note_id"), "score": 0.0}]},
        "intent": intent.to_dict(),
    }
    _attach_meta(result, None)
    return result


def _maybe_run_hybrid(
    question,
    ir,
    intent,
    candidates,
    note_store,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    hybrid: Optional["HybridRetriever"] = None,
    alias_lookup: Optional[Dict[str, str]] = None,
):
    cfg_obj = cfg or getattr(hybrid, "cfg", None) or config_loader.load_config()
    retr_cfg = cfg_obj.get("retriever") or {}
    hybrid_cfg = retr_cfg.get("hybrid") or {}
    if not bool(hybrid_cfg.get("enabled", True)):
        return None
    hybrid_inst = hybrid
    if hybrid_inst is None:
        try:
            from .hybrid import HybridRetriever
        except Exception as exc:
            logger.error("Hybrid retriever unavailable: {}", exc)
            return None
        hybrid_inst = HybridRetriever(cfg_obj)
    embedding_on = bool(getattr(hybrid_inst.embedding_client, "enabled", False))
    bm25_on = bool(getattr(hybrid_inst.bm25_client, "enabled", False))
    rerank_on = bool(getattr(hybrid_inst.reranker, "enabled", False))
    if not (embedding_on or bm25_on or rerank_on):
        return None
    return hybrid_inst.retrieve(question, ir, intent, candidates, note_store, alias_lookup=alias_lookup)


def _candidate_root_entity(candidate: Candidate) -> Optional[str]:
    if not candidate.path:
        return None
    first = candidate.path[0]
    if not isinstance(first, dict):
        return None
    subj = first.get("subj")
    return str(subj) if subj else None


def _select_top_candidates(
    candidates: List[Candidate],
    fanout: int,
    seed_entities: Sequence[str],
) -> List[Candidate]:
    if fanout <= 0 or not candidates:
        return []
    if len(seed_entities) <= 1:
        return candidates[:fanout]

    selected: List[Candidate] = []
    used_indices: set[int] = set()
    seed_set = {seed for seed in seed_entities if seed}

    # Ensure each seed entity can contribute at least one top candidate when available.
    for seed in seed_entities:
        if not seed:
            continue
        for idx, cand in enumerate(candidates):
            if idx in used_indices:
                continue
            if _candidate_root_entity(cand) == seed:
                selected.append(cand)
                used_indices.add(idx)
                break
        if len(selected) >= fanout:
            return selected[:fanout]

    # Fill the remaining slots with global best-scoring candidates.
    for idx, cand in enumerate(candidates):
        if idx in used_indices:
            continue
        root = _candidate_root_entity(cand)
        if root and seed_set and root not in seed_set and len(selected) < min(len(seed_set), fanout):
            continue
        selected.append(cand)
        if len(selected) >= fanout:
            break
    return selected[:fanout]


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
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    # 放宽置信阈值、轻度去重，并设置留底下限，避免全清空
    sched_cfg = (cfg or {}).get("retriever", {}).get("scheduler") or {}
    if sched_cfg.get("keep_at_least") is not None:
        try:
            keep_at_least = max(keep_at_least, int(sched_cfg.get("keep_at_least")))
        except (TypeError, ValueError):
            pass
    min_conf = sched_cfg.get("min_confidence", 0.3)
    try:
        min_conf_val = float(min_conf) if min_conf is not None else None
    except (TypeError, ValueError):
        min_conf_val = 0.3
    if min_conf_val is not None and min_conf_val <= 0:
        min_conf_val = None
    dedup_subject = bool(sched_cfg.get("dedup_subject", True))
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
        if min_conf_val is not None and conf is not None:
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                conf_val = None
            if conf_val is not None and conf_val < min_conf_val:
                continue
        subj = (note.get("subj") or "").strip()
        # 轻度按主体去重（非激进）
        if dedup_subject and subj and subj in seen_entities:
            # 保留少量重复，避免过度去重
            if len(kept) >= 2:
                continue
        if subj:
            seen_entities.add(subj)
        record_anchor_usage(bool(meta.get("anchor")))
        kept.append({
            "note_id": note.get("note_id"),
            "evidence": note.get("evidence", ""),
            "canonical": meta.get("evidence_canonical") or note.get("evidence", ""),
            "quality": meta.get("quality_score"),
            "lead_in_note_id": meta.get("lead_in_note_id"),
            "subj": note.get("subj"),
            "pred": note.get("pred"),
            "obj": note.get("obj"),
            "weak": bool(meta.get("weak")),
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
                "subj": note.get("subj"),
                "pred": note.get("pred"),
                "obj": note.get("obj"),
                "weak": bool(meta.get("weak")),
            })
            if len(kept) >= keep_at_least:
                break
    return kept


def _normalize_doc_hint(doc_hint: Optional[str]) -> Optional[str]:
    if not doc_hint:
        return None
    return doc_hint.strip().lower()


def _attach_meta(result: Dict[str, Any], metrics: Optional[Dict[str, float]]) -> None:
    payload = metrics or {}
    result["meta"] = {
        "path_consistency": float(payload.get("pred_score", 0.0)),
        "entity_consistency": float(payload.get("entity_score", 0.0)),
    }


def _note_id_matches_doc(note_id: Optional[str], doc_hint: Optional[str]) -> bool:
    if not doc_hint or not note_id:
        return True
    doc_part = note_id.split("#", 1)[0].lower()
    hint = (doc_hint or "").strip().lower()
    if not hint:
        return True
    if hint in doc_part:
        return True
    if "/" in hint:
        ds, qid = hint.split("/", 1)
        if not ds or not qid:
            return False
        if not doc_part.startswith(ds + "/"):
            return False
        suffix = doc_part.split("/", 1)[1]
        if "__" in suffix:
            return suffix.split("__")[-1] == qid
        return suffix == qid
    return False


def _note_matches_source(note: Optional[Dict[str, Any]], doc_hint: Optional[str]) -> bool:
    if not doc_hint or not note:
        return True
    meta = (note.get("meta", {}) or {})
    source = (meta.get("source") or "").strip().lower()
    if source:
        return _note_id_matches_doc(source, doc_hint)
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
    intent = result.get("intent") or {}
    entity_name = (intent.get("entity") or "").strip()
    desired_prefix: Optional[str] = None
    if entity_name and "/" in doc_hint_norm:
        ds, _ = doc_hint_norm.split("/", 1)
        slug = "".join(ch.lower() if ch.isalnum() else ("_" if ch.isspace() or ch in "-_" else "") for ch in entity_name).strip("_")
        if slug:
            desired_prefix = f"{ds}/{slug}__"
    paths = result.get("paths") or []
    filtered_paths: List[List[Dict[str, Any]]] = []
    for path in paths:
        if not path:
            continue
        note_ids = [edge.get("note_id") for edge in path if edge.get("note_id")]
        def _ok(nid: Optional[str]) -> bool:
            if not _note_id_matches_doc(nid, doc_hint_norm):
                return False
            if desired_prefix and isinstance(nid, str):
                nid_lower = nid.lower()
                if not nid_lower.startswith(desired_prefix):
                    return False
            return True
        if not note_ids or all(_ok(nid) for nid in note_ids):
            filtered_paths.append(path)
    result["paths"] = filtered_paths
    support_ids = result.get("support_note_ids") or []
    support_ids = _filter_note_ids_by_doc(support_ids, doc_hint_norm)
    if desired_prefix:
        support_ids = [nid for nid in support_ids if isinstance(nid, str) and nid.lower().startswith(desired_prefix)]
    result["support_note_ids"] = support_ids
    evidences = result.get("evidence") or []
    filtered_evs = [ev for ev in evidences if _note_id_matches_doc(ev.get("note_id"), doc_hint_norm)]
    if desired_prefix:
        filtered_evs = [ev for ev in filtered_evs if isinstance(ev.get("note_id"), str) and ev.get("note_id").lower().startswith(desired_prefix)]
    result["evidence"] = filtered_evs
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
    hybrid = result.get("hybrid") or {}
    consensus_candidates = hybrid.get("consensus") or []
    if consensus_candidates:
        filtered_consensus: list[dict[str, Any]] = []
        for cand in consensus_candidates:
            support_notes = cand.get("support_notes") or []
            if not support_notes:
                continue
            if any(_note_id_matches_doc(nid, doc_hint_norm) for nid in support_notes if nid):
                filtered_consensus.append(cand)
        hybrid["consensus"] = filtered_consensus
        best_meta = hybrid.get("best") or {}
        consensus_label = best_meta.get("consensus_label")
        if consensus_label and not any(c.get("label") == consensus_label for c in filtered_consensus):
            best_meta.pop("consensus_label", None)
            best_meta.pop("consensus_agreement", None)
        hybrid["best"] = best_meta
        result["hybrid"] = hybrid


_CHUNK_STORE_CACHE: Dict[str, ChunkStore] = {}


def _resolve_chunks_path(note_store: NoteStore) -> Optional[str]:
    notes_path = getattr(note_store, "notes_path", None)
    if not notes_path:
        return None
    base = Path(notes_path)
    candidate = base.parent / "chunks.jsonl"
    return str(candidate) if candidate.exists() else None


def _get_chunk_store(path: str) -> ChunkStore:
    store = _CHUNK_STORE_CACHE.get(path)
    if store is None:
        store = ChunkStore(path)
        _CHUNK_STORE_CACHE[path] = store
    return store


def _merge_evidence_items(
    existing: List[Dict[str, Any]],
    extras: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for ev in existing:
        key = (ev.get("note_id"), ev.get("evidence"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(ev)
    added = 0
    for ev in extras:
        key = (ev.get("note_id"), ev.get("evidence"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(ev)
        added += 1
    return merged, added


def _load_all_notes(note_store: NoteStore) -> List[Dict[str, Any]]:
    try:
        note_store._ensure_loaded()  # type: ignore[attr-defined]
    except Exception:
        return []
    cache = getattr(note_store, "_cache", None) or {}
    if not isinstance(cache, dict):
        return []
    return [note for note in cache.values() if isinstance(note, dict)]


def _vector_fallback_search(
    *,
    question: str,
    note_store: NoteStore,
    doc_hint: Optional[str],
    embedding_cfg: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    notes = _load_all_notes(note_store)
    if not notes:
        return []
    normalized_hint = _normalize_doc_hint(doc_hint)
    if normalized_hint:
        notes = [note for note in notes if _note_matches_source(note, normalized_hint)]
    if not notes:
        return []
    try:
        ranked = VectorSearcher(embedding_cfg=embedding_cfg).search_in_notes(
            question,
            notes,
            top_k=top_k,
        )
    except Exception as exc:
        logger.warning("vector fallback search failed: {}", exc)
        return []
    if not ranked:
        return []
    evidences: List[Dict[str, Any]] = []
    for note, score in ranked:
        meta = (note.get("meta", {}) or {})
        evidences.append(
            {
                "note_id": note.get("note_id"),
                "evidence": note.get("evidence", ""),
                "canonical": meta.get("evidence_canonical") or note.get("evidence", ""),
                "quality": meta.get("quality_score"),
                "subj": note.get("subj"),
                "pred": note.get("pred"),
                "obj": note.get("obj"),
                "weak": bool(meta.get("weak", False)),
                "score": float(score),
                "source": "vector_fallback",
            }
        )
    return evidences


def _should_chunk_fallback(result: Dict[str, Any], intent: AnswerIntent, ir: Optional[QueryIR]) -> bool:
    evidence = result.get("evidence") or []
    reason = result.get("reason")
    fallback = result.get("fallback") or {}
    status = fallback.get("status")
    if not evidence:
        return True
    if reason in {"no_path", "attribute_not_detected", "parse_failed"}:
        return True
    if status in {"no_path", "attribute_not_detected", "no_attribute_match"}:
        return True
    if intent.attribute is None and ir and not ir.pred_chain:
        return True
    if all(ev.get("weak") or ev.get("discount") for ev in evidence):
        return True
    qualities = [
        float(ev.get("quality"))
        for ev in evidence
        if isinstance(ev.get("quality"), (int, float))
    ]
    if qualities and max(qualities) < 0.2:
        return True
    return False


def _evidence_doc_key(ev: Dict[str, Any]) -> str:
    for key in ("doc_id", "source_doc_id"):
        value = str(ev.get(key) or "").strip()
        if value:
            return value
    note_id = str(ev.get("note_id") or "").strip()
    if note_id:
        if "#c" in note_id:
            return note_id.split("#c", 1)[0].strip()
        if "#" in note_id:
            return note_id.split("#", 1)[0].strip()
        return note_id
    subj = str(ev.get("subj") or "").strip()
    if subj:
        return subj
    return ""


def _unique_doc_count(evidence: List[Dict[str, Any]], probe_k: int) -> int:
    if probe_k <= 0:
        return 0
    seen: set[str] = set()
    for ev in evidence[:probe_k]:
        doc_key = _evidence_doc_key(ev)
        if doc_key:
            seen.add(doc_key)
    return len(seen)


def _apply_chunk_fallback(
    result: Dict[str, Any],
    *,
    question: str,
    ir: Optional[QueryIR],
    intent: AnswerIntent,
    note_store: NoteStore,
    doc_hint: Optional[str],
    cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not _should_chunk_fallback(result, intent, ir):
        return result
    retr_cfg = (cfg or {}).get("retriever") or {}
    chunk_cfg = retr_cfg.get("chunk_fallback") or {}
    top_k = chunk_cfg.get("top_k")
    if top_k is None:
        structured_cfg = retr_cfg.get("structured") or {}
        top_k = structured_cfg.get("top_k")
    if top_k is None:
        top_k = retr_cfg.get("top_k")
    if top_k is None:
        return result
    top_k = int(top_k)
    if top_k <= 0:
        return result
    chunks_path = _resolve_chunks_path(note_store)
    seeds: List[str] = []
    if ir:
        seeds.extend(seed.text for seed in ir.seeds if seed.text)
    if intent.entity and intent.entity not in seeds:
        seeds.append(intent.entity)
    existing = result.get("evidence") or []
    merged = list(existing)
    chunk_evs: List[Dict[str, Any]] = []
    if chunks_path:
        chunk_store = _get_chunk_store(chunks_path)
        chunk_evs = chunk_store.search(question, seeds=seeds, top_k=top_k, doc_hint=doc_hint)
    if chunk_evs:
        merged, _ = _merge_evidence_items(merged, chunk_evs)
        result.setdefault("chunk_fallback", {})["used"] = True
        result["chunk_fallback"]["hits"] = len(chunk_evs)
    else:
        result.setdefault("chunk_fallback", {})["used"] = False
        result["chunk_fallback"]["hits"] = 0

    vector_cfg = chunk_cfg.get("vector") or {}
    vector_enabled = bool(vector_cfg.get("enabled", True))
    vector_hits = 0
    vector_reason = "none"
    need_vector = len(merged) < top_k
    if need_vector:
        vector_reason = "insufficient_count"
    elif vector_enabled:
        probe_k = int(vector_cfg.get("diversity_probe_k", min(top_k, 5)))
        probe_k = max(2, probe_k)
        min_unique_docs = int(vector_cfg.get("min_unique_docs", 2 if top_k >= 5 else 1))
        min_unique_docs = max(1, min_unique_docs)
        if min_unique_docs > 1 and _unique_doc_count(merged, probe_k) < min_unique_docs:
            need_vector = True
            vector_reason = "low_doc_diversity"

    if vector_enabled and need_vector:
        vector_top_k = int(vector_cfg.get("top_k", top_k))
        vector_top_k = max(vector_top_k, top_k)
        query_for_vector = question
        if seeds:
            query_for_vector = f"{question}\nSeeds: {'; '.join(seeds[:6])}"
        vector_evs = _vector_fallback_search(
            question=query_for_vector,
            note_store=note_store,
            doc_hint=doc_hint,
            embedding_cfg=(retr_cfg.get("embedding") or {}),
            top_k=vector_top_k,
        )
        if vector_evs:
            merged, vector_hits = _merge_evidence_items(merged, vector_evs)
    result.setdefault("vector_fallback", {})["used"] = vector_hits > 0
    result["vector_fallback"]["hits"] = vector_hits
    result["vector_fallback"]["enabled"] = vector_enabled
    result["vector_fallback"]["trigger_reason"] = vector_reason

    if merged:
        keep_cap = max(len(existing), top_k)
        if len(merged) > keep_cap and (chunk_evs or vector_hits > 0):
            keep_cap = len(merged)
        result["evidence"] = merged[:keep_cap]
        if not existing:
            result["reason"] = None
        support_ids = list(result.get("support_note_ids") or [])
        seen_ids = set(support_ids)
        for ev in result["evidence"]:
            nid = ev.get("note_id")
            if nid and nid not in seen_ids:
                support_ids.append(nid)
                seen_ids.add(nid)
        if support_ids:
            result["support_note_ids"] = support_ids
    return result


def _record_retrieval_metrics(result: Dict[str, Any]) -> None:
    record_retrieval_total()
    reason = result.get("reason")
    status = (result.get("fallback") or {}).get("status")
    if reason == "no_path" or status == "no_path":
        record_retrieval_no_path()
    evidences = result.get("evidence") or []
    if not evidences:
        record_retrieval_empty_context()

    metrics = export_metrics()
    total = max(1, metrics.get("retrieval.total", 0))
    no_path = metrics.get("retrieval.no_path", 0)
    empty_ctx = metrics.get("retrieval.empty_context", 0)
    if total == 1 or total % 20 == 0 or not evidences or reason == "no_path":
        logger.info(
            "retrieval_rates no_path={:.2%} empty_context={:.2%} total={}",
            no_path / total,
            empty_ctx / total,
            total,
        )


def _candidate_label(candidate: Candidate, attribute: str) -> str:
    if not candidate.path:
        return ""
    obj = candidate.path[-1].get("obj")
    if not isinstance(obj, str):
        return ""
    normalized, _ = normalize_slot_value(attribute, obj)
    text = (normalized or obj or "").strip().lower()
    return text
