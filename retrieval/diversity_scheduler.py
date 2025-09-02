#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
候选多样性与证据覆盖调度模块

实现功能：
1. 候选多样性控制
2. 证据覆盖调度
3. 去重策略管理
4. 候选拼装优化
5. 多样性评估
"""

import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
from abc import ABC, abstractmethod

# 导入增强日志功能
from utils.logging_utils import StructuredLogger, log_performance, log_diversity_metrics

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("numpy not available, some features may be limited")

logger = logging.getLogger(__name__)

class DiversityStrategy(Enum):
    """多样性策略"""
    SEMANTIC = "semantic"          # 语义多样性
    TOPICAL = "topical"            # 主题多样性
    TEMPORAL = "temporal"          # 时间多样性
    SOURCE = "source"              # 来源多样性
    ENTITY = "entity"              # 实体多样性
    HYBRID = "hybrid"              # 混合多样性

class DeduplicationMethod(Enum):
    """去重方法"""
    EXACT = "exact"                # 精确匹配
    FUZZY = "fuzzy"                # 模糊匹配
    SEMANTIC = "semantic"          # 语义相似性
    HASH = "hash"                  # 哈希去重
    HYBRID = "hybrid"              # 混合去重

class CoverageMetric(Enum):
    """覆盖度指标"""
    ENTITY_COVERAGE = "entity_coverage"        # 实体覆盖度
    TOPIC_COVERAGE = "topic_coverage"          # 主题覆盖度
    TEMPORAL_COVERAGE = "temporal_coverage"    # 时间覆盖度
    SOURCE_COVERAGE = "source_coverage"        # 来源覆盖度
    SEMANTIC_COVERAGE = "semantic_coverage"    # 语义覆盖度

@dataclass
class DiversityConfig:
    """多样性配置"""
    strategy: DiversityStrategy = DiversityStrategy.HYBRID
    max_candidates: int = 50                    # 最大候选数
    diversity_threshold: float = 0.7            # 多样性阈值
    deduplication_method: DeduplicationMethod = DeduplicationMethod.HYBRID
    similarity_threshold: float = 0.85          # 相似度阈值
    coverage_weights: Dict[str, float] = field(
        default_factory=lambda: {
            'entity': 0.3,
            'topic': 0.3,
            'temporal': 0.2,
            'source': 0.2
        }
    )
    enable_clustering: bool = True              # 启用聚类
    cluster_threshold: float = 0.6              # 聚类阈值
    # 证据类型配额管理
    evidence_type_quotas: Dict[str, int] = field(
        default_factory=lambda: {
            'vector': 15,      # 向量检索证据配额
            'graph': 10,       # 图检索证据配额
            'path_aware': 8,   # 路径感知证据配额
            'semantic': 12,    # 语义相似证据配额
            'entity_rich': 6   # 实体丰富证据配额
        }
    )
    enable_evidence_quota: bool = True          # 启用证据类型配额
    quota_balance_factor: float = 0.8           # 配额平衡因子

@dataclass
class CandidateItem:
    """候选项"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    timestamp: Optional[float] = None
    source: Optional[str] = None
    embedding: Optional[List[float]] = None
    content_hash: Optional[str] = None
    # 证据类型相关字段
    evidence_types: Set[str] = field(default_factory=set)  # 证据类型集合
    path_score: float = 0.0                               # 路径分数
    entity_density: float = 0.0                           # 实体密度
    source_types: Set[str] = field(default_factory=set)   # 来源类型集合
    
    def __post_init__(self):
        """计算内容哈希和推断证据类型"""
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
        
        # 自动推断证据类型
        self._infer_evidence_types()
    
    def _infer_evidence_types(self):
        """推断证据类型"""
        # 基于来源类型推断
        if 'vector' in self.source_types:
            self.evidence_types.add('vector')
        if 'graph' in self.source_types:
            self.evidence_types.add('graph')
        
        # 基于路径分数推断
        if self.path_score > 0.3:
            self.evidence_types.add('path_aware')
        
        # 基于实体密度推断
        if self.entity_density > 0.5:
            self.evidence_types.add('entity_rich')
        
        # 基于语义分数推断
        if self.score > 0.7:
            self.evidence_types.add('semantic')
        
        # 默认类型
        if not self.evidence_types:
            self.evidence_types.add('general')

@dataclass
class DiversityResult:
    """多样性调度结果"""
    selected_candidates: List[CandidateItem]
    diversity_score: float
    coverage_metrics: Dict[str, float]
    deduplication_stats: Dict[str, int]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class DiversityEvaluator(ABC):
    """多样性评估器抽象基类"""
    
    @abstractmethod
    def evaluate_diversity(self, candidates: List[CandidateItem]) -> float:
        """评估候选集的多样性"""
        pass
    
    @abstractmethod
    def calculate_similarity(self, item1: CandidateItem, item2: CandidateItem) -> float:
        """计算两个候选项的相似度"""
        pass

class SemanticDiversityEvaluator(DiversityEvaluator):
    """语义多样性评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('semantic_diversity', {})
        self.embedding_dim = self.config.get('embedding_dim', 768)
    
    def evaluate_diversity(self, candidates: List[CandidateItem]) -> float:
        """评估语义多样性"""
        if len(candidates) <= 1:
            return 1.0
        
        # 计算平均相似度
        total_similarity = 0.0
        pair_count = 0
        
        for i, item1 in enumerate(candidates):
            for item2 in candidates[i+1:]:
                similarity = self.calculate_similarity(item1, item2)
                total_similarity += similarity
                pair_count += 1
        
        if pair_count == 0:
            return 1.0
        
        avg_similarity = total_similarity / pair_count
        diversity = 1.0 - avg_similarity
        
        return max(0.0, min(1.0, diversity))
    
    def calculate_similarity(self, item1: CandidateItem, item2: CandidateItem) -> float:
        """计算语义相似度"""
        if not item1.embedding or not item2.embedding:
            # 使用简单的文本相似度
            return self._text_similarity(item1.content, item2.content)
        
        if NUMPY_AVAILABLE:
            emb1 = np.array(item1.embedding)
            emb2 = np.array(item2.embedding)
            
            # 计算余弦相似度
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))
        
        return self._text_similarity(item1.content, item2.content)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

class TopicalDiversityEvaluator(DiversityEvaluator):
    """主题多样性评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('topical_diversity', {})
    
    def evaluate_diversity(self, candidates: List[CandidateItem]) -> float:
        """评估主题多样性"""
        if len(candidates) <= 1:
            return 1.0
        
        # 收集所有主题
        all_topics = set()
        for candidate in candidates:
            all_topics.update(candidate.topics)
        
        if not all_topics:
            return 0.5  # 没有主题信息时返回中等多样性
        
        # 计算主题分布的均匀性
        topic_counts = defaultdict(int)
        for candidate in candidates:
            for topic in candidate.topics:
                topic_counts[topic] += 1
        
        # 计算熵作为多样性指标
        total_count = sum(topic_counts.values())
        if total_count == 0:
            return 0.5
        
        entropy = 0.0
        for count in topic_counts.values():
            if count > 0:
                p = count / total_count
                entropy -= p * np.log2(p) if NUMPY_AVAILABLE else p * (count / total_count)
        
        # 归一化熵
        max_entropy = np.log2(len(all_topics)) if NUMPY_AVAILABLE and len(all_topics) > 0 else 1.0
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return max(0.0, min(1.0, diversity))
    
    def calculate_similarity(self, item1: CandidateItem, item2: CandidateItem) -> float:
        """计算主题相似度"""
        topics1 = set(item1.topics)
        topics2 = set(item2.topics)
        
        if not topics1 or not topics2:
            return 0.0
        
        intersection = len(topics1.intersection(topics2))
        union = len(topics1.union(topics2))
        
        return intersection / union if union > 0 else 0.0

class DeduplicationProcessor:
    """去重处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('deduplication', {})
        self.method = DeduplicationMethod(
            self.config.get('method', 'hybrid')
        )
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.hash_cache: Dict[str, str] = {}
    
    def deduplicate(self, candidates: List[CandidateItem]) -> Tuple[List[CandidateItem], Dict[str, int]]:
        """
        去重处理
        
        Args:
            candidates: 候选项列表
            
        Returns:
            去重后的候选项列表和统计信息
        """
        if not candidates:
            return [], {'total': 0, 'removed': 0, 'kept': 0}
        
        original_count = len(candidates)
        
        if self.method == DeduplicationMethod.EXACT:
            deduplicated = self._exact_deduplication(candidates)
        elif self.method == DeduplicationMethod.FUZZY:
            deduplicated = self._fuzzy_deduplication(candidates)
        elif self.method == DeduplicationMethod.SEMANTIC:
            deduplicated = self._semantic_deduplication(candidates)
        elif self.method == DeduplicationMethod.HASH:
            deduplicated = self._hash_deduplication(candidates)
        else:  # HYBRID
            deduplicated = self._hybrid_deduplication(candidates)
        
        stats = {
            'total': original_count,
            'removed': original_count - len(deduplicated),
            'kept': len(deduplicated)
        }
        
        return deduplicated, stats
    
    def _exact_deduplication(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        """精确去重"""
        seen_content = set()
        deduplicated = []
        
        for candidate in candidates:
            if candidate.content not in seen_content:
                seen_content.add(candidate.content)
                deduplicated.append(candidate)
        
        return deduplicated
    
    def _hash_deduplication(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        """哈希去重"""
        seen_hashes = set()
        deduplicated = []
        
        for candidate in candidates:
            if candidate.content_hash not in seen_hashes:
                seen_hashes.add(candidate.content_hash)
                deduplicated.append(candidate)
        
        return deduplicated
    
    def _fuzzy_deduplication(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        """模糊去重"""
        deduplicated = []
        
        for candidate in candidates:
            is_duplicate = False
            
            for existing in deduplicated:
                similarity = self._calculate_text_similarity(candidate.content, existing.content)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    # 保留分数更高的候选项
                    if candidate.score > existing.score:
                        deduplicated.remove(existing)
                        deduplicated.append(candidate)
                    break
            
            if not is_duplicate:
                deduplicated.append(candidate)
        
        return deduplicated
    
    def _semantic_deduplication(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        """语义去重"""
        if not NUMPY_AVAILABLE:
            return self._fuzzy_deduplication(candidates)
        
        deduplicated = []
        
        for candidate in candidates:
            is_duplicate = False
            
            for existing in deduplicated:
                similarity = self._calculate_semantic_similarity(candidate, existing)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    # 保留分数更高的候选项
                    if candidate.score > existing.score:
                        deduplicated.remove(existing)
                        deduplicated.append(candidate)
                    break
            
            if not is_duplicate:
                deduplicated.append(candidate)
        
        return deduplicated
    
    def _hybrid_deduplication(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        """混合去重"""
        # 先进行哈希去重
        candidates = self._hash_deduplication(candidates)
        
        # 再进行语义去重
        candidates = self._semantic_deduplication(candidates)
        
        return candidates
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_semantic_similarity(self, item1: CandidateItem, item2: CandidateItem) -> float:
        """计算语义相似度"""
        if not item1.embedding or not item2.embedding:
            return self._calculate_text_similarity(item1.content, item2.content)
        
        if NUMPY_AVAILABLE:
            emb1 = np.array(item1.embedding)
            emb2 = np.array(item2.embedding)
            
            # 计算余弦相似度
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))
        
        return self._calculate_text_similarity(item1.content, item2.content)

class DiversityScheduler:
    """多样性调度器"""
    
    def __init__(self, config: Dict[str, Any]):
        from utils.config_loader_helper import load_config_with_fallback
        
        # 尝试从外部配置文件加载，如果失败则使用内联配置
        config_file_path = config.get('diversity_scheduler_config_file')
        
        # 默认配置
        default_config = {
            'strategy': 'hybrid',
            'max_candidates': 50,
            'diversity_threshold': 0.7,
            'deduplication_method': 'hybrid',
            'similarity_threshold': 0.85
        }
        
        # 加载配置（外部文件优先，回退到内联配置，最后使用默认配置）
        inline_config = config.get('diversity_scheduler', {})
        self.config = load_config_with_fallback(config_file_path, inline_config, default_config)
        
        # 多样性配置
        self.diversity_config = DiversityConfig(
            strategy=DiversityStrategy(self.config.get('strategy', default_config['strategy'])),
            max_candidates=self.config.get('max_candidates', default_config['max_candidates']),
            diversity_threshold=self.config.get('diversity_threshold', default_config['diversity_threshold']),
            deduplication_method=DeduplicationMethod(
                self.config.get('deduplication_method', default_config['deduplication_method'])
            ),
            similarity_threshold=self.config.get('similarity_threshold', default_config['similarity_threshold'])
        )
        
        # 初始化组件
        self.deduplication_processor = DeduplicationProcessor(self.config)
        self.diversity_evaluators = self._initialize_evaluators()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'candidates_processed': 0,
            'candidates_selected': 0,
            'duplicates_removed': 0,
            'average_diversity_score': 0.0
        }
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 初始化结构化日志记录器
        self.structured_logger = StructuredLogger("DiversityScheduler")
        
        # 记录初始化完成
        self.structured_logger.info("DiversityScheduler initialized",
                                  strategy=self.diversity_config.strategy.value,
                                  max_candidates=self.diversity_config.max_candidates,
                                  diversity_threshold=self.diversity_config.diversity_threshold,
                                  deduplication_method=self.diversity_config.deduplication_method.value,
                                  evidence_quota_enabled=self.config.get('enable_evidence_quota', False))
        
        logger.info(f"DiversityScheduler initialized with strategy: {self.diversity_config.strategy.value}")
    
    def _initialize_evaluators(self) -> Dict[str, DiversityEvaluator]:
        """初始化多样性评估器"""
        evaluators = {}
        
        # 语义多样性评估器
        evaluators['semantic'] = SemanticDiversityEvaluator(self.config)
        
        # 主题多样性评估器
        evaluators['topical'] = TopicalDiversityEvaluator(self.config)
        
        return evaluators
    
    @log_performance("DiversityScheduler.schedule_candidates")
    def schedule_candidates(self, candidates: List[Dict[str, Any]], 
                          query: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None) -> DiversityResult:
        """
        调度候选项
        
        Args:
            candidates: 候选项列表
            query: 查询字符串
            context: 上下文信息
            
        Returns:
            多样性调度结果
        """
        start_time = time.time()
        context = context or {}
        
        # 记录调度开始
        self.structured_logger.debug("Starting candidate scheduling",
                                   candidates_count=len(candidates),
                                   query_length=len(query) if query else 0,
                                   strategy=self.diversity_config.strategy.value,
                                   evidence_quota_enabled=self.config.get('enable_evidence_quota', False))
        
        with self._lock:
            self.stats['total_requests'] += 1
            self.stats['candidates_processed'] += len(candidates)
        
        # 转换为CandidateItem
        candidate_items = self._convert_to_candidate_items(candidates)
        
        # 去重处理
        deduplicated_items, dedup_stats = self.deduplication_processor.deduplicate(candidate_items)
        
        # 多样性选择
        selected_items = self._select_diverse_candidates(deduplicated_items, query, context)
        
        # 评估多样性
        diversity_score = self._evaluate_overall_diversity(selected_items)
        
        # 计算覆盖度指标
        coverage_metrics = self._calculate_coverage_metrics(selected_items)
        
        execution_time = time.time() - start_time
        
        # 更新统计信息
        with self._lock:
            self.stats['candidates_selected'] += len(selected_items)
            self.stats['duplicates_removed'] += dedup_stats['removed']
            
            # 更新平均多样性分数
            total_requests = self.stats['total_requests']
            current_avg = self.stats['average_diversity_score']
            self.stats['average_diversity_score'] = (
                (current_avg * (total_requests - 1) + diversity_score) / total_requests
            )
        
        # 记录多样性调度指标
        log_diversity_metrics(
            candidates_count=len(candidates),
            selected_count=len(selected_items),
            diversity_score=diversity_score,
            evidence_quota_enabled=self.config.get('enable_evidence_quota', False),
            scheduler_strategy=self.diversity_config.strategy.value
        )
        
        # 记录调度完成
        self.structured_logger.info("Candidate scheduling completed",
                                   selected_count=len(selected_items),
                                   diversity_score=f"{diversity_score:.3f}",
                                   duplicates_removed=dedup_stats['removed'],
                                   execution_time_ms=f"{execution_time*1000:.2f}")
        
        return DiversityResult(
            selected_candidates=selected_items,
            diversity_score=diversity_score,
            coverage_metrics=coverage_metrics,
            deduplication_stats=dedup_stats,
            execution_time=execution_time,
            metadata={
                'query': query,
                'context': context,
                'strategy': self.diversity_config.strategy.value
            }
        )
    
    def _convert_to_candidate_items(self, candidates: List[Dict[str, Any]]) -> List[CandidateItem]:
        """转换为CandidateItem"""
        items = []
        
        for i, candidate in enumerate(candidates):
            # 提取证据类型相关信息
            source_types = set()
            if 'source_info' in candidate:
                source_type = candidate['source_info'].get('source_type', '')
                if source_type:
                    source_types.add(source_type)
            if 'source_types' in candidate:
                source_types.update(candidate['source_types'])
            
            # 计算实体密度
            entities = candidate.get('entities', [])
            content = candidate.get('content', '')
            entity_density = len(entities) / max(len(content.split()), 1) if content else 0.0
            
            # 处理 metadata，将 paragraph_idxs 放入其中
            meta = dict(candidate.get('metadata', {}))
            meta['paragraph_idxs'] = candidate.get('paragraph_idxs', [])
            
            item = CandidateItem(
                id=candidate.get('id', f'candidate_{i}'),
                content=content,
                score=candidate.get('score', 0.0),
                metadata=meta,
                entities=entities,
                topics=candidate.get('topics', []),
                timestamp=candidate.get('timestamp'),
                source=candidate.get('source'),
                embedding=candidate.get('embedding'),
                # 证据类型相关字段
                path_score=candidate.get('path_score', 0.0),
                entity_density=entity_density,
                source_types=source_types
            )
            items.append(item)
        
        return items
    
    def _select_diverse_candidates(self, candidates: List[CandidateItem],
                                  query: Optional[str],
                                  context: Dict[str, Any]) -> List[CandidateItem]:
        """选择多样化的候选项"""
        if len(candidates) <= self.diversity_config.max_candidates:
            return candidates
        
        # 按分数排序
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # 如果启用证据类型配额管理
        if self.diversity_config.enable_evidence_quota:
            return self._select_with_evidence_quota(candidates, query, context)
        
        # 传统贪心选择多样化候选项
        selected = [candidates[0]]  # 选择分数最高的
        remaining = candidates[1:]
        
        while len(selected) < self.diversity_config.max_candidates and remaining:
            best_candidate = None
            best_diversity_gain = -1.0
            
            for candidate in remaining:
                # 计算添加该候选项后的多样性增益
                diversity_gain = self._calculate_diversity_gain(selected, candidate)
                
                if diversity_gain > best_diversity_gain:
                    best_diversity_gain = diversity_gain
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _select_with_evidence_quota(self, candidates: List[CandidateItem],
                                   query: Optional[str],
                                   context: Dict[str, Any]) -> List[CandidateItem]:
        """基于证据类型配额选择候选项"""
        selected = []
        evidence_type_counts = defaultdict(int)
        quotas = self.diversity_config.evidence_type_quotas.copy()
        
        # 第一轮：按配额分配高质量候选
        for candidate in candidates:
            if len(selected) >= self.diversity_config.max_candidates:
                break
            
            # 检查候选的证据类型
            candidate_types = candidate.evidence_types
            
            # 找到未满配额的证据类型
            available_types = []
            for evidence_type in candidate_types:
                if evidence_type_counts[evidence_type] < quotas.get(evidence_type, 0):
                    available_types.append(evidence_type)
            
            # 如果有可用类型，选择该候选
            if available_types:
                selected.append(candidate)
                # 更新计数
                for evidence_type in candidate_types:
                    evidence_type_counts[evidence_type] += 1
                
                logger.debug(f"Selected candidate with types {candidate_types}, "
                           f"current counts: {dict(evidence_type_counts)}")
        
        # 第二轮：填充剩余位置，考虑多样性
        remaining_candidates = [c for c in candidates if c not in selected]
        
        while (len(selected) < self.diversity_config.max_candidates and 
               remaining_candidates):
            
            best_candidate = None
            best_score = -1.0
            
            for candidate in remaining_candidates:
                # 计算综合分数：原始分数 + 多样性增益 + 配额平衡奖励
                diversity_gain = self._calculate_diversity_gain(selected, candidate)
                quota_bonus = self._calculate_quota_balance_bonus(
                    candidate, evidence_type_counts, quotas
                )
                
                combined_score = (
                    candidate.score * 0.5 + 
                    diversity_gain * 0.3 + 
                    quota_bonus * 0.2
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
                # 更新计数
                for evidence_type in best_candidate.evidence_types:
                    evidence_type_counts[evidence_type] += 1
            else:
                break
        
        logger.info(f"Evidence quota selection completed: {len(selected)} candidates, "
                   f"type distribution: {dict(evidence_type_counts)}")
        
        return selected
    
    def _calculate_quota_balance_bonus(self, candidate: CandidateItem,
                                     current_counts: Dict[str, int],
                                     quotas: Dict[str, int]) -> float:
        """计算配额平衡奖励分数"""
        bonus = 0.0
        balance_factor = self.diversity_config.quota_balance_factor
        
        for evidence_type in candidate.evidence_types:
            quota = quotas.get(evidence_type, 0)
            current_count = current_counts.get(evidence_type, 0)
            
            if quota > 0:
                # 计算该类型的填充率
                fill_rate = current_count / quota
                # 奖励未充分填充的类型
                if fill_rate < balance_factor:
                    bonus += (balance_factor - fill_rate) * 0.5
        
        return bonus
    
    def _calculate_diversity_gain(self, current_selected: List[CandidateItem],
                                 new_candidate: CandidateItem) -> float:
        """计算添加新候选项的多样性增益"""
        if not current_selected:
            return 1.0
        
        # 计算与已选择候选项的平均相似度
        total_similarity = 0.0
        
        for selected in current_selected:
            similarity = self._calculate_candidate_similarity(selected, new_candidate)
            total_similarity += similarity
        
        avg_similarity = total_similarity / len(current_selected)
        diversity_gain = 1.0 - avg_similarity
        
        # 结合候选项的分数
        score_weight = 0.3
        diversity_weight = 0.7
        
        normalized_score = min(1.0, new_candidate.score)
        combined_gain = (diversity_weight * diversity_gain + 
                        score_weight * normalized_score)
        
        return combined_gain
    
    def _calculate_candidate_similarity(self, item1: CandidateItem, item2: CandidateItem) -> float:
        """计算候选项相似度"""
        # 使用主要的多样性评估器
        if self.diversity_config.strategy == DiversityStrategy.SEMANTIC:
            evaluator = self.diversity_evaluators.get('semantic')
        elif self.diversity_config.strategy == DiversityStrategy.TOPICAL:
            evaluator = self.diversity_evaluators.get('topical')
        else:  # HYBRID或其他
            # 使用语义评估器作为默认
            evaluator = self.diversity_evaluators.get('semantic')
        
        if evaluator:
            return evaluator.calculate_similarity(item1, item2)
        
        return 0.0
    
    def _evaluate_overall_diversity(self, candidates: List[CandidateItem]) -> float:
        """评估整体多样性"""
        if not candidates:
            return 0.0
        
        diversity_scores = []
        
        # 使用所有评估器
        for name, evaluator in self.diversity_evaluators.items():
            score = evaluator.evaluate_diversity(candidates)
            diversity_scores.append(score)
        
        if not diversity_scores:
            return 0.0
        
        # 计算加权平均
        return sum(diversity_scores) / len(diversity_scores)
    
    def _calculate_coverage_metrics(self, candidates: List[CandidateItem]) -> Dict[str, float]:
        """计算覆盖度指标"""
        if not candidates:
            return {}
        
        metrics = {}
        
        # 实体覆盖度 - 计算平均每个候选项的实体数量，归一化到0-1
        all_entities = set()
        total_entity_count = 0
        for candidate in candidates:
            all_entities.update(candidate.entities)
            total_entity_count += len(candidate.entities)
        
        if total_entity_count > 0:
            # 使用独特实体数与总实体数的比例作为覆盖度
            metrics['entity_coverage'] = min(1.0, len(all_entities) / max(1, total_entity_count))
        else:
            metrics['entity_coverage'] = 0.0
        
        # 主题覆盖度 - 计算平均每个候选项的主题数量，归一化到0-1
        all_topics = set()
        total_topic_count = 0
        for candidate in candidates:
            all_topics.update(candidate.topics)
            total_topic_count += len(candidate.topics)
        
        if total_topic_count > 0:
            # 使用独特主题数与总主题数的比例作为覆盖度
            metrics['topic_coverage'] = min(1.0, len(all_topics) / max(1, total_topic_count))
        else:
            metrics['topic_coverage'] = 0.0
        
        # 来源覆盖度 - 计算来源多样性比例
        all_sources = set()
        source_count = 0
        for candidate in candidates:
            if candidate.source:
                all_sources.add(candidate.source)
                source_count += 1
        
        if source_count > 0:
            # 使用独特来源数与有来源的候选项数的比例
            metrics['source_coverage'] = len(all_sources) / source_count
        else:
            metrics['source_coverage'] = 0.0
        
        # 时间覆盖度 - 归一化时间跨度到0-1范围
        timestamps = [c.timestamp for c in candidates if c.timestamp]
        if len(timestamps) > 1:
            time_span = max(timestamps) - min(timestamps)
            # 假设一天(86400秒)为最大合理时间跨度，归一化到0-1
            max_reasonable_span = 86400  # 1 day in seconds
            metrics['temporal_coverage'] = min(1.0, time_span / max_reasonable_span)
        else:
            metrics['temporal_coverage'] = 0.0
        
        return metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._lock:
            self.stats = {
                'total_requests': 0,
                'candidates_processed': 0,
                'candidates_selected': 0,
                'duplicates_removed': 0,
                'average_diversity_score': 0.0
            }
        logger.info("Statistics reset")

# 便利函数
def create_diversity_scheduler(config: Dict[str, Any]) -> DiversityScheduler:
    """创建多样性调度器"""
    return DiversityScheduler(config)

def create_diversity_config(strategy: str = "hybrid", **kwargs) -> DiversityConfig:
    """
    创建多样性配置的便利函数
    
    Args:
        strategy: 多样性策略
        **kwargs: 其他配置参数
        
    Returns:
        多样性配置实例
    """
    return DiversityConfig(
        strategy=DiversityStrategy(strategy),
        max_candidates=kwargs.get('max_candidates', 50),
        diversity_threshold=kwargs.get('diversity_threshold', 0.7),
        deduplication_method=DeduplicationMethod(
            kwargs.get('deduplication_method', 'hybrid')
        ),
        similarity_threshold=kwargs.get('similarity_threshold', 0.85)
    )

def schedule_diverse_candidates(scheduler: DiversityScheduler,
                              candidates: List[Dict[str, Any]],
                              query: Optional[str] = None) -> DiversityResult:
    """
    调度多样化候选项的便利函数
    
    Args:
        scheduler: 多样性调度器
        candidates: 候选项列表
        query: 查询字符串
        
    Returns:
        多样性调度结果
    """
    return scheduler.schedule_candidates(candidates, query)