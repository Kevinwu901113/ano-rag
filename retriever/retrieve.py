from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import re

from config import config
from .operators import Indexes, BIND
from .note_store import NoteStore
from .intent_detector import AnswerIntentDetector
from utils.vector_search import VectorSearcher


@dataclass
class RetrieveWeights:
    corefers: float = 1.0
    mentions: float = 0.9
    unresolved_pronoun: float = 0.8
    neighbor_bonus: float = 0.9
    alias_penalty: float = 0.95
    fallback_contains_entity_boost: float = 1.05


def _get_weights() -> RetrieveWeights:
    cfg = config.load_config()
    wcfg = cfg.get("retrieval", {}) or {}
    weights = (wcfg or {}).get("weights", {}) or {}
    return RetrieveWeights(
        corefers=float(weights.get("corefers", 1.0) or 1.0),
        mentions=float(weights.get("mentions", 0.9) or 0.9),
        unresolved_pronoun=float(weights.get("unresolved_pronoun", 0.8) or 0.8),
        neighbor_bonus=float(weights.get("neighbor_bonus", 0.9) or 0.9),
        alias_penalty=float(weights.get("alias_penalty", 0.95) or 0.95),
        fallback_contains_entity_boost=float(weights.get("fallback_contains_entity_boost", 1.05) or 1.05),
    )


def retrieve(
    question: str,
    indexes: Indexes,
    note_store: NoteStore,
    *,
    title: Optional[str] = None,
    recent_entities: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """实体优先检索管线：COREFERS/MENTIONS、别名扩展、邻域容错，及 Vector-only 回退。

    注意：不再使用 BM25；回退仅以向量检索缩小候选集，不参与最终排序或与结构结果融合。
    """
    weights = _get_weights()
    detector = AnswerIntentDetector()
    intent = detector.detect(question)

    # 1) 收集实体候选：标题、用户输入检测、会话实体
    raw_candidates: List[str] = []
    if title and (title.strip()):
        raw_candidates.append(title.strip())
    if intent.entity:
        raw_candidates.append(intent.entity.strip())
    for ent in recent_entities or []:
        if ent and ent.strip():
            raw_candidates.append(ent.strip())

    raw_candidates = _unique_preserve_order(raw_candidates)

    # 2) 规范化（大小写折叠的精确匹配）→ 实体倒排取候选
    canonical_entities = _direct_canonical_entities(raw_candidates, indexes)
    candidates = _collect_candidates_for_entities(
        canonical_entities,
        indexes,
        note_store,
        weights,
        alias_hit=False,
        only_unresolved=False,
        neighbor=False,
        limit=top_k,
    )

    stage = "direct_hit" if candidates else None

    # 3) 别名扩展：候选为空→对实体做别名扩展（alias_map）
    if not candidates:
        alias_bound = _bind_entities(raw_candidates, indexes, limit=50)
        canonical_entities = _unique_preserve_order(alias_bound)
        candidates = _collect_candidates_for_entities(
            canonical_entities,
            indexes,
            note_store,
            weights,
            alias_hit=True,
            only_unresolved=False,
            neighbor=False,
            limit=top_k,
        )
        stage = "alias_expansion" if candidates else None

    # 4) 邻域容错扩展：仍为空→图1-hop邻居 + 未决代词笔记
    if not candidates:
        # 若尚未绑定到规范实体，尝试以问题文本再绑定一次
        if not canonical_entities:
            canonical_entities = _bind_entities([question], indexes, limit=25)
        neighbor_entities = _one_hop_neighbors(canonical_entities, indexes)
        candidates = _collect_candidates_for_entities(
            neighbor_entities,
            indexes,
            note_store,
            weights,
            alias_hit=False,
            only_unresolved=True,
            neighbor=True,
            limit=top_k,
        )
        stage = "neighbor_tolerance" if candidates else None

    # 5) 全文回退：向量-only检索；实体重排行（不引入词面/关键词额外评分）
    if not candidates:
        vec_results = _vector_fallback(question, canonical_entities, indexes, note_store, weights, top_k)
        candidates = vec_results
        stage = "vector_fallback"

    # 排序与去重
    candidates.sort(key=lambda c: c["score"], reverse=True)
    unique = {}
    ordered: List[Dict[str, Any]] = []
    for cand in candidates:
        nid = cand.get("note_id")
        if not nid:
            continue
        if nid in unique:
            # 保留最高分
            if cand["score"] > unique[nid]["score"]:
                unique[nid] = cand
            continue
        unique[nid] = cand
        ordered.append(cand)

    ordered = sorted(unique.values(), key=lambda c: c["score"], reverse=True)[:top_k]
    evidences: List[Dict[str, Any]] = []
    for item in ordered:
        note = note_store.get(item["note_id"]) or {}
        evidences.append(
            {
                "note_id": item["note_id"],
                "evidence": note.get("evidence", ""),
                "subj": note.get("subj"),
                "pred": note.get("pred"),
                "obj": note.get("obj"),
                "score": round(item["score"], 4),
                "flags": item.get("flags", []),
            }
        )

    return {
        "stage": stage,
        "entities": canonical_entities,
        "notes": evidences,
    }


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        key = (it or "").strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _direct_canonical_entities(candidates: List[str], indexes: Indexes) -> List[str]:
    direct: List[str] = []
    for q in candidates:
        lowered = (q or "").strip().lower()
        if not lowered:
            continue
        for name in indexes.entity_to_notes.keys():
            if name.lower() == lowered and name not in direct:
                direct.append(name)
                break
    return direct


def _bind_entities(candidates: List[str], indexes: Indexes, limit: int) -> List[str]:
    bound: List[str] = []
    for q in candidates:
        for ent in BIND(indexes, q, [], limit=limit):
            if ent not in bound:
                bound.append(ent)
        if len(bound) >= limit:
            break
    return bound


def _collect_candidates_for_entities(
    entities: List[str],
    indexes: Indexes,
    note_store: NoteStore,
    weights: RetrieveWeights,
    *,
    alias_hit: bool,
    only_unresolved: bool,
    neighbor: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for entity in entities:
        note_ids = indexes.entity_to_notes.get(entity, [])
        for nid in note_ids:
            note = note_store.get(nid)
            if not note:
                continue
            meta = (note.get("meta", {}) or {})
            if only_unresolved and not bool(meta.get("has_unresolved_pronoun")):
                continue
            base = _base_score(note)
            w = 1.0
            flags: List[str] = []
            # 命中 COREFERS_TO vs MENTIONS
            if entity in indexes.corefers_edges.get(nid, []):
                w *= weights.corefers
                flags.append("COREFERS_TO")
            elif entity in indexes.mentions_edges.get(nid, []):
                w *= weights.mentions
                flags.append("MENTIONS")
            # 未决代词加权
            if bool(meta.get("has_unresolved_pronoun")):
                w *= weights.unresolved_pronoun
                flags.append("UNRESOLVED_PRONOUN")
            # 邻域带出额外权重
            if neighbor:
                w *= weights.neighbor_bonus
                flags.append("NEIGHBOR")
            # 别名命中而非canonical
            if alias_hit:
                w *= weights.alias_penalty
                flags.append("ALIAS_HIT")

            output.append(
                {
                    "note_id": nid,
                    "entity": entity,
                    "score": float(base) * float(w),
                    "flags": flags,
                }
            )
            if len(output) >= limit:
                break
        if len(output) >= limit:
            break
    return output


def _one_hop_neighbors(entities: List[str], indexes: Indexes) -> List[str]:
    collected: List[str] = []
    seen = set()
    for ent in entities:
        for pred, obj, _ in indexes.graph_edges.get(ent, []):
            if obj in seen:
                continue
            seen.add(obj)
            collected.append(obj)
    return collected


def _vector_fallback(
    question: str,
    canonical_entities: List[str],
    indexes: Indexes,
    note_store: NoteStore,
    weights: RetrieveWeights,
    top_k: int,
) -> List[Dict[str, Any]]:
    # 聚合所有 note_id
    all_note_ids: List[str] = []
    for ids in indexes.entity_to_notes.values():
        for nid in ids:
            if nid not in all_note_ids:
                all_note_ids.append(nid)
    notes = note_store.get_many(all_note_ids)
    if not notes:
        return []
    # 向量-only 排序（不做额外词面加权）
    try:
        vs = VectorSearcher()
        ranked = vs.search_in_notes(question or "", notes, top_k=top_k)
    except Exception:
        ranked = []
    candidates: List[Dict[str, Any]] = []
    for note, score in ranked:
        nid = note.get("note_id")
        if not nid:
            continue
        candidates.append({
            "note_id": nid,
            "entity": None,
            "score": float(score),
            "flags": ["VECTOR"],
        })
    return candidates


def _base_score(note: Dict[str, Any]) -> float:
    meta = (note.get("meta", {}) or {})
    score = float(meta.get("final_conf", 0.0))
    quality = meta.get("quality_score")
    if isinstance(quality, (float, int)):
        score += 0.3 * float(quality)
    return max(score, 0.01)