"""
Entity-Focused Scoring and Answer selection (EFSA) Algorithm

核心思想：在最终入围的原子笔记小集合里，不去猜"关系/谓词"，而是把出现的实体当作候选答案来聚合打分；
排除桥接实体，选证据分最高的另一个实体作为短答案。

用到的现成信号（都来自原子笔记）：
- note.entities：每条笔记抽到的实体集合
- note.final_score：重排后的综合分
- note.hop_no、bridge_entity / bridge_path：路径信息
- 覆盖率/一致性（Coverage/Consistency）：可现场计算
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from loguru import logger
import os
from pathlib import Path


def compute_cov_cons(note: Dict[str, Any], path_entities: List[str]) -> Tuple[float, int]:
    """
    计算覆盖率和一致性（复用现有逻辑）
    
    Args:
        note: 原子笔记
        path_entities: 路径实体列表
        
    Returns:
        (coverage, consistency) 元组
    """
    ne = set((note.get("entities") or []))
    pe = set(e.lower() for e in (path_entities or []))
    if not pe:
        return 0.0, 0
    
    # 覆盖率：路径实体在笔记实体中的占比
    cov = len({e.lower() for e in ne} & pe) / max(1, len(pe))
    
    # 一致性：候选文本是否提到路径实体名（0/1）
    text = f"{note.get('title','')} {note.get('content','')}".lower()
    cons = 1 if any(e in text for e in pe) else 0
    
    return float(cov), int(cons)


def efsa_answer(candidates: List[Dict[str, Any]],
                query: str,
                bridge_entity: Optional[str] = None,
                path_entities: Optional[List[str]] = None,
                topN: int = 20) -> Tuple[Optional[str], List[int], float]:
    """
    EFSA核心算法：实体聚合打分和答案选择
    
    Args:
        candidates: 候选笔记列表（已重排和过滤）
        query: 查询问题
        bridge_entity: 桥接实体（需要排除的实体）
        path_entities: 路径实体列表
        topN: 取前N个候选进行处理
        
    Returns:
        (predicted_answer, predicted_support_idxs, score) 元组
        如果没有找到合适的实体答案，返回 (None, [], 0.0)
    """
    logger.info(f"EFSA processing {len(candidates)} candidates with topN={topN}")
    
    # 1) 取最终入围的小集合
    C = candidates[:topN]
    pe = set([*(path_entities or [])])
    be = (bridge_entity or "").lower()
    
    if not C:
        logger.warning("No candidates provided to EFSA")
        return None, [], 0.0
    
    # 2) 聚合实体证据
    score = defaultdict(float)
    docset = defaultdict(set)
    contrib = defaultdict(list)  # 记录贡献明细以便返回支持段
    
    logger.debug(f"Processing {len(C)} candidates, bridge_entity='{be}', path_entities={list(pe)}")
    
    for n in C:
        hop = int(n.get("hop_no", 1))
        hop_decay = 0.85
        
        # 计算覆盖率和一致性
        cov, cons = compute_cov_cons(n, list(pe))
        
        # 计算权重
        base_score = float(n.get("final_score", 0.0))
        w = base_score * (hop_decay ** (hop - 1)) * (1 + 0.10*cov + 0.05*cons)
        
        # 为每个实体累计证据分
        entities = n.get("entities") or []
        for e in entities:
            if be and e.lower() == be:
                logger.debug(f"Skipping bridge entity: {e}")
                continue  # 排除桥接实体
            
            score[e] += w
            docset[e].add(n.get("doc_id"))
            contrib[e].append((w, n))  # 用于挑选支持段
            
        logger.debug(f"Note {n.get('note_id', 'unknown')}: hop={hop}, base_score={base_score:.3f}, "
                    f"cov={cov:.3f}, cons={cons}, weight={w:.3f}, entities={entities}")
    
    if not score:
        logger.warning("No valid entities found after filtering bridge entities")
        return None, [], 0.0  # 回退到原有的句子型答案
    
    # 3) 文档多样性轻奖励
    for e in score:
        diversity_bonus = 1 + 0.03 * min(max(len(docset[e]) - 1, 0), 3)
        original_score = score[e]
        score[e] *= diversity_bonus
        logger.debug(f"Entity '{e}': original_score={original_score:.3f}, "
                    f"diversity_bonus={diversity_bonus:.3f}, final_score={score[e]:.3f}")
    
    # 4) 选答案 + 支持段
    ans, final_score = max(score.items(), key=lambda kv: kv[1])
    logger.info(f"Selected answer: '{ans}' with score {final_score:.3f}")
    
    # 按贡献排序，选择最能代表的支持段
    contrib[ans].sort(key=lambda t: t[0], reverse=True)
    support_idxs = []
    
    for _, note in contrib[ans][:2]:  # 取前2个贡献最大的笔记
        # 取最能代表的段落（笔记里已带 paragraph_idxs）
        paragraph_idxs = note.get("paragraph_idxs")
        if paragraph_idxs:
            support_idxs.append(paragraph_idxs[0])
        else:
            # 如果没有paragraph_idxs，使用note_id作为fallback
            note_id = note.get("note_id")
            if note_id:
                support_idxs.append(note_id)
    
    # 去重支持段
    support_idxs = list(dict.fromkeys(support_idxs))  # 保持顺序的去重
    
    score_value = float(final_score)
    logger.info(f"EFSA result: answer='{ans}', support_idxs={support_idxs}, score={score_value:.3f}")
    return ans, support_idxs, score_value


def efsa_answer_with_fallback(candidates: List[Dict[str, Any]] = None,
                             query: str = "",
                             bridge_entity: Optional[str] = None,
                             path_entities: Optional[List[str]] = None,
                             topN: int = 20,
                             fallback_func: Optional[callable] = None,
                             final_recall_path: Optional[str] = None) -> Tuple[str, List[int], float]:
    """
    带回退机制的EFSA答案生成
    
    Args:
        candidates: 候选笔记列表（可选，如果提供final_recall_path则从文件读取）
        query: 查询问题
        bridge_entity: 桥接实体
        path_entities: 路径实体列表
        topN: 取前N个候选
        fallback_func: 回退函数，当EFSA无法找到实体答案时调用
        final_recall_path: final_recall.jsonl文件路径，如果提供则从此文件读取candidates
        
    Returns:
        (predicted_answer, predicted_support_idxs, score) 元组
    """
    # 如果提供了final_recall_path，从文件读取candidates
    if final_recall_path and Path(final_recall_path).exists():
        logger.info(f"Loading candidates from final_recall_path: {final_recall_path}")
        try:
            from utils.file_utils import FileUtils
            candidates = FileUtils.read_jsonl(final_recall_path)
            logger.info(f"Loaded {len(candidates)} candidates from {final_recall_path}")
        except Exception as e:
            logger.error(f"Failed to load candidates from {final_recall_path}: {e}")
            if candidates is None:
                candidates = []
    
    # 如果仍然没有candidates，返回默认答案
    if not candidates:
        logger.warning("No candidates available for EFSA processing")
        return None, [], 0.0
    
    # 尝试EFSA
    answer, support_idxs, score = efsa_answer(candidates, query, bridge_entity, path_entities, topN)

    if answer is not None:
        return answer, support_idxs, score
    
    # 回退到原有逻辑
    logger.info("EFSA failed to find entity answer, falling back to original method")
    if fallback_func:
        fb_answer, fb_support = fallback_func(candidates, query)
        return fb_answer, fb_support, 0.0
    else:
        # 简单回退：返回第一个候选的内容片段
        if candidates:
            first_candidate = candidates[0]
            content = first_candidate.get('content', '')
            # 简单截取前50个字符作为答案
            fallback_answer = content[:50].strip() if content else None
            paragraph_idxs = first_candidate.get('paragraph_idxs', [])
            fallback_support = paragraph_idxs[:1] if paragraph_idxs else []
            return fallback_answer, fallback_support, 0.0
        else:
            return None, [], 0.0


def extract_bridge_info_from_candidates(candidates: List[Dict[str, Any]]) -> Tuple[Optional[str], List[str]]:
    """
    从候选笔记中提取桥接实体和路径实体信息
    
    Args:
        candidates: 候选笔记列表
        
    Returns:
        (bridge_entity, path_entities) 元组
    """
    bridge_entity = None
    path_entities = []
    
    for candidate in candidates:
        # 提取桥接实体
        if not bridge_entity:
            bridge_entity = candidate.get('bridge_entity')
        
        # 提取路径实体
        bridge_path = candidate.get('bridge_path', [])
        if bridge_path:
            path_entities.extend(bridge_path)
    
    # 去重路径实体
    path_entities = list(set(path_entities))
    
    logger.debug(f"Extracted bridge_entity='{bridge_entity}', path_entities={path_entities}")
    return bridge_entity, path_entities