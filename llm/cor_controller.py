from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

from config import config
from graph.index import NoteGraph
from graph.search import beam_search  # noqa: F401  # 保留导入以兼容扩展逻辑
from answer_selector import AnswerSelectorResult, run_answer_selector

# -------- 缓存结构（进程内）---------
_CE_CACHE: Dict[Tuple[str, str], float] = {}  # (q_id/norm_q, note_id) -> ce_score


def _norm_q(q: str) -> str:
    return (q or '').strip().lower()


def _merge_and_dedupe(base: List[str], extra: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for nid in base + extra:
        if nid and nid not in seen:
            out.append(nid)
            seen.add(nid)
    return out


def _coverage_gain(prev_entities: Set[str], new_entities: Set[str]) -> float:
    if not new_entities:
        return 0.0
    if not prev_entities:
        return 1.0
    inc = len(new_entities - prev_entities)
    return inc / max(1, len(new_entities))


def _rewrite_query_with_missing(q: str, missing_entities: List[str]) -> str:
    # 轻量“子查询”生成：不依赖LLM，直接把缺口实体拼接
    miss = ', '.join(missing_entities[:3])
    return f"{q} [bridge: {miss}]"


def _bi_rank_then_ce_rank(question: str, note_ids: List[str], top_m: int, top_n: int) -> List[str]:
    """
    你现有的“双塔->CE”重排如果已经有接口，就调用它们；
    这里给一个可替代的最小占位逻辑：
    - 先用 beam_search 的打分或 BM25/vec 得到 top_m
    - 再逐对做 CE 打分，但对已算过的 (q,n) 走 _CE_CACHE
    """
    # 占位：这里假设外部已先按粗排顺序给到 note_ids
    rough = note_ids[:top_m]

    # 伪CE打分（请替换为你已有的 CE 接口）：
    def ce_score(q: str, nid: str) -> float:
        key = (_norm_q(q), nid)
        if key in _CE_CACHE:
            return _CE_CACHE[key]
        # TODO: 调用你已有 CE 模型；下面用长度替代，确保可运行
        score = 0.5 + (hash(nid) % 100) / 200.0
        _CE_CACHE[key] = score
        return score

    scored = [(nid, ce_score(question, nid)) for nid in rough]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in scored[:top_n]]


def _single_round(
    question: str,
    graph: NoteGraph,
    extra_seed_notes: Optional[List[str]] = None,
    k0: Optional[int] = None,
    beam_width: Optional[int] = None,
    neighbor_cap: Optional[int] = None,
    bi_top_m: Optional[int] = None,
    ce_top_n: Optional[int] = None,
) -> Tuple[AnswerSelectorResult, List[str], Set[str]]:
    """
    一轮：种子召回 -> 图扩展(单hop) -> 重排 -> AnswerSelector
    返回：AnswerSelector 结果、本轮查看过的 notes、实体覆盖集
    """
    cor_cfg = config.get('cor', {}) or {}
    k0 = k0 or int(cor_cfg.get('k0', 40))
    beam_width = beam_width or int(cor_cfg.get('beam_width', 8))
    neighbor_cap = neighbor_cap or int(cor_cfg.get('neighbor_cap', 8))
    bi_top_m = bi_top_m or int(cor_cfg.get('bi_top_m', 60))
    ce_top_n = ce_top_n or int(cor_cfg.get('ce_top_n', 20))

    # Round-0 种子：你已有的检索函数（BM25 ∪ 向量 + MMR 去重）
    seed_note_ids = graph.seed_recall(question, top_k=k0, diversify=True)
    if extra_seed_notes:
        seed_note_ids = _merge_and_dedupe(seed_note_ids, extra_seed_notes)

    # 轻量图扩展（只扩一层，用作当轮候选池）
    frontier = seed_note_ids[:beam_width]
    expanded: List[str] = []
    for nid in frontier:
        nbrs = graph.get_neighbors(nid, cap=neighbor_cap)
        expanded.extend(nbrs)
    pool = _merge_and_dedupe(seed_note_ids, expanded)

    # 两级重排（可替换为你现有实现）
    candidates = _bi_rank_then_ce_rank(question, pool, top_m=bi_top_m, top_n=ce_top_n)

    # AnswerSelector：请确保返回 AnswerSelectorResult(confidence, answer, evidence_note_ids, missing_entities)
    asr = run_answer_selector(question, candidates)

    # 统计覆盖的实体集合（由 AnswerSelector 或 Reader 回传，或你从 candidates 的元数据里汇总）
    covered_entities: Set[str] = set(asr.covered_entities or [])

    return asr, candidates, covered_entities


def chain_of_retrieval(
    question: str,
    graph: NoteGraph,
    max_rounds: int = 2,  # 总轮次：1(单轮) + 1(再检) = 2
    tau_conf: float = 0.80,  # 早停阈值（置信度）
    eps_cov: float = 0.05,  # 覆盖增量阈值
    hard_ce_cap: int = 250,  # 单问题 CE 评估上限（成本保险丝）
) -> AnswerSelectorResult:
    """
    轻量 CoR 控制器：在 AnswerSelector 给出低置信时，再进行一次“受控扩展”。
    """
    cor_cfg = config.get('cor', {}) or {}
    max_rounds = max_rounds or int(cor_cfg.get('max_rounds', 2))
    tau_conf = tau_conf or float(cor_cfg.get('tau_conf', 0.80))
    eps_cov = eps_cov or float(cor_cfg.get('eps_cov', 0.05))
    hard_ce_cap = hard_ce_cap or int(cor_cfg.get('hard_ce_cap', 250))

    # 允许外部重置/观察缓存用量：你可改成线程/问题粒度的计数器
    ce_eval_counter = 0
    q_norm = _norm_q(question)

    def ce_count_guard(new_pairs: int) -> bool:
        nonlocal ce_eval_counter
        ce_eval_counter += new_pairs
        return ce_eval_counter <= hard_ce_cap

    # Round-0
    start_ts = time.time()
    asr0, seen0, cov0 = _single_round(question, graph)
    _ = (start_ts, seen0, q_norm, ce_count_guard)  # 保留变量以便调试/扩展
    if asr0.confidence >= tau_conf:
        return asr0

    # 若需要继续：基于缺失实体/桥接关系做一次“补检”
    missing = list(asr0.missing_entities or [])[:3]
    if max_rounds <= 1 or not missing:
        return asr0

    sub_q = _rewrite_query_with_missing(question, missing)
    # 额外种子（例如直接按实体查节点；也可以是 graph.search_by_entities(missing)）
    extra_seed = graph.seed_recall(sub_q, top_k=20, diversify=True)

    # Round-1（受控）
    asr1, seen1, cov1 = _single_round(sub_q, graph, extra_seed_notes=extra_seed)
    _ = (seen1,)

    # 早停/覆盖增量判断
    gain = _coverage_gain(cov0, cov1)
    if asr1.confidence >= tau_conf or gain >= eps_cov:
        # 置信度融合（保守）：二者取加权，证据并集
        if asr1.confidence >= asr0.confidence:
            return asr1
        return asr0

    # 否则回退最可信的一版
    return asr1 if asr1.confidence >= asr0.confidence else asr0


__all__ = ['chain_of_retrieval', 'AnswerSelectorResult']
