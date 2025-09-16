#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
答案一致性检查器 - 使用原子笔记特征验证关键信息映射

基于原子笔记的事实信息，验证生成答案的一致性，确保关键信息能够从原子事实映射到原文span。
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ConsistencyResult:
    """一致性检查结果"""
    is_consistent: bool
    confidence_score: float
    mapped_facts: List[Dict[str, Any]]
    unmapped_claims: List[str]
    evidence_coverage: float
    fact_support_ratio: float
    reason: str

class AnswerConsistencyChecker:
    """
    答案一致性检查器
    
    使用原子笔记特征验证答案的一致性：
    1. 提取答案中的关键声明
    2. 将声明映射到原子事实
    3. 验证事实到原文span的映射
    4. 计算一致性分数
    """
    
    def __init__(self, 
                 atomic_notes: List[Dict[str, Any]] = None,
                 config: Dict[str, Any] = None,
                 min_fact_support_ratio: float = 0.6,
                 min_evidence_coverage: float = 0.4,
                 min_confidence_threshold: float = 0.5):
        """
        初始化一致性检查器
        
        Args:
            atomic_notes: 原子笔记列表（可选）
            config: 配置字典（可选）
            min_fact_support_ratio: 最小事实支持比例
            min_evidence_coverage: 最小证据覆盖度
            min_confidence_threshold: 最小置信度阈值
        """
        # 存储原子笔记和配置
        self.atomic_notes = atomic_notes or []
        self.config = config or {}
        self.min_fact_support_ratio = min_fact_support_ratio
        self.min_evidence_coverage = min_evidence_coverage
        self.min_confidence_threshold = min_confidence_threshold
    
    def check_consistency(self, 
                         answer: str, 
                         atomic_notes: List[Dict[str, Any]], 
                         evidence_spans: List[Dict[str, Any]] = None) -> ConsistencyResult:
        """
        检查答案与原子笔记的一致性
        
        Args:
            answer: 生成的答案
            atomic_notes: 原子笔记列表
            evidence_spans: 证据span列表
            
        Returns:
            一致性检查结果
        """
        try:
            # 1. 提取答案中的关键声明
            answer_claims = self._extract_answer_claims(answer)
            logger.info(f"提取到 {len(answer_claims)} 个答案声明")
            
            # 2. 收集所有原子事实
            atomic_facts = self._collect_atomic_facts(atomic_notes)
            logger.info(f"收集到 {len(atomic_facts)} 个原子事实")
            
            # 3. 将声明映射到事实
            mapped_facts, unmapped_claims = self._map_claims_to_facts(answer_claims, atomic_facts)
            logger.info(f"映射了 {len(mapped_facts)} 个事实，{len(unmapped_claims)} 个声明未映射")
            
            # 4. 验证事实到原文的映射
            evidence_coverage = self._calculate_evidence_coverage(mapped_facts, atomic_notes, evidence_spans)
            
            # 5. 计算事实支持比例
            fact_support_ratio = len(mapped_facts) / max(len(answer_claims), 1)
            
            # 6. 计算综合置信度分数
            confidence_score = self._calculate_confidence_score(
                fact_support_ratio, evidence_coverage, mapped_facts
            )
            
            # 7. 判断是否一致
            is_consistent = (
                fact_support_ratio >= self.min_fact_support_ratio and
                evidence_coverage >= self.min_evidence_coverage and
                confidence_score >= self.min_confidence_threshold
            )
            
            # 8. 生成原因说明
            reason = self._generate_consistency_reason(
                is_consistent, fact_support_ratio, evidence_coverage, confidence_score
            )
            
            return ConsistencyResult(
                is_consistent=is_consistent,
                confidence_score=confidence_score,
                mapped_facts=mapped_facts,
                unmapped_claims=unmapped_claims,
                evidence_coverage=evidence_coverage,
                fact_support_ratio=fact_support_ratio,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"一致性检查失败: {e}")
            return ConsistencyResult(
                is_consistent=False,
                confidence_score=0.0,
                mapped_facts=[],
                unmapped_claims=[],
                evidence_coverage=0.0,
                fact_support_ratio=0.0,
                reason=f"检查过程出错: {str(e)}"
            )
    
    def _extract_answer_claims(self, answer: str) -> List[str]:
        """
        从答案中提取关键声明
        
        Args:
            answer: 答案文本
            
        Returns:
            声明列表
        """
        claims = []
        
        # 按句子分割
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        
        for sentence in sentences:
            # 过滤掉太短的句子
            if len(sentence.split()) < 3:
                continue
                
            # 提取包含关键信息的句子
            if self._contains_factual_content(sentence):
                claims.append(sentence)
        
        return claims
    
    def _contains_factual_content(self, sentence: str) -> bool:
        """
        判断句子是否包含事实性内容
        
        Args:
            sentence: 句子文本
            
        Returns:
            是否包含事实性内容
        """
        # 检查是否包含数字、日期、专有名词等
        factual_patterns = [
            r'\b\d+\b',  # 数字
            r'\b\d{4}\b',  # 年份
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 专有名词
            r'\b(?:is|was|are|were|has|have|had)\b',  # 事实性动词
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, sentence):
                return True
        
        return False
    
    def _collect_atomic_facts(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        收集所有原子事实
        
        Args:
            atomic_notes: 原子笔记列表
            
        Returns:
            原子事实列表
        """
        all_facts = []
        
        for note in atomic_notes:
            atomic_facts = note.get('atomic_facts', [])
            note_id = note.get('note_id', note.get('id', ''))
            
            for fact in atomic_facts:
                fact_with_source = fact.copy()
                fact_with_source['source_note_id'] = note_id
                fact_with_source['source_content'] = note.get('content', '')
                all_facts.append(fact_with_source)
        
        return all_facts
    
    def _map_claims_to_facts(self, claims: List[str], facts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        将答案声明映射到原子事实
        
        Args:
            claims: 答案声明列表
            facts: 原子事实列表
            
        Returns:
            (映射的事实列表, 未映射的声明列表)
        """
        mapped_facts = []
        unmapped_claims = []
        
        for claim in claims:
            best_fact = None
            best_similarity = 0.0
            
            for fact in facts:
                fact_text = fact.get('text', '')
                similarity = self._calculate_semantic_similarity(claim, fact_text)
                
                if similarity > best_similarity and similarity > 0.3:  # 相似度阈值
                    best_similarity = similarity
                    best_fact = fact.copy()
                    best_fact['claim'] = claim
                    best_fact['similarity'] = similarity
            
            if best_fact:
                mapped_facts.append(best_fact)
            else:
                unmapped_claims.append(claim)
        
        return mapped_facts, unmapped_claims
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        # 简单的词汇重叠相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # 考虑长度因素
        length_factor = min(len(words1), len(words2)) / max(len(words1), len(words2))
        
        return jaccard_similarity * (0.7 + 0.3 * length_factor)
    
    def _calculate_evidence_coverage(self, 
                                   mapped_facts: List[Dict[str, Any]], 
                                   atomic_notes: List[Dict[str, Any]], 
                                   evidence_spans: List[Dict[str, Any]] = None) -> float:
        """
        计算证据覆盖度
        
        Args:
            mapped_facts: 映射的事实列表
            atomic_notes: 原子笔记列表
            evidence_spans: 证据span列表
            
        Returns:
            证据覆盖度 (0-1)
        """
        if not mapped_facts:
            return 0.0
        
        covered_facts = 0
        
        for fact in mapped_facts:
            source_note_id = fact.get('source_note_id', '')
            fact_text = fact.get('text', '')
            
            # 检查是否有对应的原文span
            has_span_coverage = False
            
            if evidence_spans:
                for span in evidence_spans:
                    span_text = span.get('text', '')
                    if self._text_overlap(fact_text, span_text) > 0.3:
                        has_span_coverage = True
                        break
            
            # 检查是否能在原文中找到支持
            has_source_coverage = False
            for note in atomic_notes:
                if note.get('note_id', note.get('id', '')) == source_note_id:
                    content = note.get('content', '')
                    if self._text_overlap(fact_text, content) > 0.2:
                        has_source_coverage = True
                        break
            
            if has_span_coverage or has_source_coverage:
                covered_facts += 1
        
        return covered_facts / len(mapped_facts)
    
    def _text_overlap(self, text1: str, text2: str) -> float:
        """
        计算两个文本的重叠度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            重叠度 (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        return intersection / len(words1)
    
    def _calculate_confidence_score(self, 
                                  fact_support_ratio: float, 
                                  evidence_coverage: float, 
                                  mapped_facts: List[Dict[str, Any]]) -> float:
        """
        计算综合置信度分数
        
        Args:
            fact_support_ratio: 事实支持比例
            evidence_coverage: 证据覆盖度
            mapped_facts: 映射的事实列表
            
        Returns:
            置信度分数 (0-1)
        """
        # 基础分数：事实支持比例和证据覆盖度的加权平均
        base_score = 0.6 * fact_support_ratio + 0.4 * evidence_coverage
        
        # 重要性加权：考虑映射事实的重要性
        if mapped_facts:
            avg_importance = np.mean([fact.get('importance', 0.5) for fact in mapped_facts])
            importance_boost = 0.1 * avg_importance
        else:
            importance_boost = 0.0
        
        # 相似度加权：考虑映射质量
        if mapped_facts:
            avg_similarity = np.mean([fact.get('similarity', 0.0) for fact in mapped_facts])
            similarity_boost = 0.1 * avg_similarity
        else:
            similarity_boost = 0.0
        
        final_score = base_score + importance_boost + similarity_boost
        return min(final_score, 1.0)
    
    def _generate_consistency_reason(self, 
                                   is_consistent: bool, 
                                   fact_support_ratio: float, 
                                   evidence_coverage: float, 
                                   confidence_score: float) -> str:
        """
        生成一致性判断的原因说明
        
        Args:
            is_consistent: 是否一致
            fact_support_ratio: 事实支持比例
            evidence_coverage: 证据覆盖度
            confidence_score: 置信度分数
            
        Returns:
            原因说明
        """
        if is_consistent:
            return f"答案一致性良好：事实支持比例 {fact_support_ratio:.2f}，证据覆盖度 {evidence_coverage:.2f}，置信度 {confidence_score:.2f}"
        else:
            issues = []
            if fact_support_ratio < self.min_fact_support_ratio:
                issues.append(f"事实支持不足 ({fact_support_ratio:.2f} < {self.min_fact_support_ratio})")
            if evidence_coverage < self.min_evidence_coverage:
                issues.append(f"证据覆盖不足 ({evidence_coverage:.2f} < {self.min_evidence_coverage})")
            if confidence_score < self.min_confidence_threshold:
                issues.append(f"置信度过低 ({confidence_score:.2f} < {self.min_confidence_threshold})")
            
            return f"答案一致性问题：{'; '.join(issues)}"

def create_consistency_checker(**kwargs) -> AnswerConsistencyChecker:
    """
    创建答案一致性检查器实例
    
    Args:
        **kwargs: 初始化参数
        
    Returns:
        AnswerConsistencyChecker实例
    """
    return AnswerConsistencyChecker(**kwargs)