#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索模块 - 支持BM25/SPLADE与向量检索的融合

实现功能：
1. BM25稀疏检索与向量检索融合
2. SPLADE稀疏检索支持（可选，自动回退到BM25）
3. RRF（Reciprocal Rank Fusion）和线性融合策略
4. 命名空间限定的候选池
5. 可配置的融合权重和参数
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from collections import defaultdict
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    logging.warning("rank_bm25 not available, BM25 functionality will be disabled")

try:
    # SPLADE相关导入（如果可用）
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    SPLADE_AVAILABLE = True
except ImportError:
    SPLADE_AVAILABLE = False
    logging.warning("SPLADE dependencies not available, will fallback to BM25")

logger = logging.getLogger(__name__)

class FusionStrategy:
    """融合策略枚举"""
    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR = "linear"  # 线性加权融合
    WEIGHTED_SUM = "weighted_sum"  # 加权求和

class HybridSearcher:
    """混合检索器 - 融合稀疏检索和向量检索"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化混合检索器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.hybrid_config = config.get('hybrid_search', {})
        
        # 融合策略配置
        self.fusion_strategy = self.hybrid_config.get('fusion_strategy', FusionStrategy.RRF)
        self.sparse_weight = self.hybrid_config.get('sparse_weight', 0.5)
        self.dense_weight = self.hybrid_config.get('dense_weight', 0.5)
        self.rrf_k = self.hybrid_config.get('rrf_k', 60)  # RRF参数
        
        # 稀疏检索配置
        self.enable_splade = self.hybrid_config.get('enable_splade', False) and SPLADE_AVAILABLE
        self.bm25_k1 = self.hybrid_config.get('bm25_k1', 1.2)
        self.bm25_b = self.hybrid_config.get('bm25_b', 0.75)
        
        # 命名空间限定
        self.enable_namespace_filtering = self.hybrid_config.get('enable_namespace_filtering', True)
        
        # 初始化组件
        self.bm25_index = None
        self.splade_model = None
        self.splade_tokenizer = None
        self.corpus_texts = []
        self.corpus_metadata = []
        
        # 初始化SPLADE（如果启用）
        if self.enable_splade:
            self._init_splade()
        
        logger.info(f"HybridSearcher initialized with strategy={self.fusion_strategy}, "
                   f"sparse_weight={self.sparse_weight}, dense_weight={self.dense_weight}")
    
    def _init_splade(self):
        """初始化SPLADE模型"""
        try:
            splade_model_name = self.hybrid_config.get('splade_model', 'naver/splade-cocondenser-ensembledistil')
            self.splade_tokenizer = AutoTokenizer.from_pretrained(splade_model_name)
            self.splade_model = AutoModelForMaskedLM.from_pretrained(splade_model_name)
            self.splade_model.eval()
            logger.info(f"SPLADE model loaded: {splade_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load SPLADE model: {e}, falling back to BM25")
            self.enable_splade = False
    
    def build_sparse_index(self, documents: List[Dict[str, Any]]):
        """
        构建稀疏检索索引
        
        Args:
            documents: 文档列表，每个文档包含text和metadata
        """
        if not documents:
            logger.warning("No documents provided for sparse index building")
            return
        
        # 提取文本和元数据
        self.corpus_texts = []
        self.corpus_metadata = []
        
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get('content', '') or doc.get('text', '')
                metadata = doc.get('metadata', {})
            else:
                text = str(doc)
                metadata = {}
            
            self.corpus_texts.append(text)
            self.corpus_metadata.append(metadata)
        
        # 构建BM25索引
        if BM25Okapi is not None:
            tokenized_corpus = [text.split() for text in self.corpus_texts]
            self.bm25_index = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
            logger.info(f"BM25 index built with {len(self.corpus_texts)} documents")
        else:
            logger.error("BM25Okapi not available, cannot build sparse index")
    
    def _sparse_search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        执行稀疏检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            
        Returns:
            (文档索引, 分数) 的列表
        """
        if self.enable_splade and self.splade_model is not None:
            return self._splade_search(query, top_k)
        elif self.bm25_index is not None:
            return self._bm25_search(query, top_k)
        else:
            logger.warning("No sparse search method available")
            return []
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25检索"""
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # 获取top_k结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        
        logger.debug(f"BM25 search returned {len(results)} results for query: '{query[:50]}...'")
        return results
    
    def _splade_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """SPLADE检索（占位符实现）"""
        # 这里应该实现SPLADE的具体检索逻辑
        # 由于SPLADE实现较复杂，这里提供一个简化版本
        logger.warning("SPLADE search not fully implemented, falling back to BM25")
        return self._bm25_search(query, top_k)
    
    def _apply_namespace_filter(self, candidates: List[Tuple[int, float]], 
                               dataset: Optional[str] = None, 
                               qid: Optional[str] = None) -> List[Tuple[int, float]]:
        """
        应用命名空间过滤
        
        Args:
            candidates: 候选结果列表
            dataset: 数据集名称
            qid: 问题ID
            
        Returns:
            过滤后的候选结果
        """
        if not self.enable_namespace_filtering or (not dataset and not qid):
            return candidates
        
        filtered_candidates = []
        for idx, score in candidates:
            if idx < len(self.corpus_metadata):
                metadata = self.corpus_metadata[idx]
                source_info = metadata.get('source_info', {})
                
                # 检查命名空间匹配
                dataset_match = not dataset or source_info.get('dataset') == dataset
                qid_match = not qid or source_info.get('qid') == qid
                
                if dataset_match and qid_match:
                    filtered_candidates.append((idx, score))
        
        logger.debug(f"Namespace filtering: {len(candidates)} -> {len(filtered_candidates)} candidates")
        return filtered_candidates
    
    def _rrf_fusion(self, sparse_results: List[Tuple[int, float]], 
                   dense_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        RRF (Reciprocal Rank Fusion) 融合策略
        
        Args:
            sparse_results: 稀疏检索结果
            dense_results: 向量检索结果
            
        Returns:
            融合后的结果
        """
        # 构建排名字典
        sparse_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_results)}
        dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}
        
        # 计算RRF分数
        rrf_scores = defaultdict(float)
        
        # 稀疏检索贡献
        for idx, rank in sparse_ranks.items():
            rrf_scores[idx] += self.sparse_weight / (self.rrf_k + rank)
        
        # 向量检索贡献
        for idx, rank in dense_ranks.items():
            rrf_scores[idx] += self.dense_weight / (self.rrf_k + rank)
        
        # 排序并返回
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_results]
    
    def _linear_fusion(self, sparse_results: List[Tuple[int, float]], 
                      dense_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        线性加权融合策略
        
        Args:
            sparse_results: 稀疏检索结果
            dense_results: 向量检索结果
            
        Returns:
            融合后的结果
        """
        # 归一化分数
        def normalize_scores(results):
            if not results:
                return {}
            scores = [score for _, score in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            return {idx: (score - min_score) / score_range for idx, score in results}
        
        sparse_norm = normalize_scores(sparse_results)
        dense_norm = normalize_scores(dense_results)
        
        # 线性融合
        fused_scores = defaultdict(float)
        
        for idx, score in sparse_norm.items():
            fused_scores[idx] += self.sparse_weight * score
        
        for idx, score in dense_norm.items():
            fused_scores[idx] += self.dense_weight * score
        
        # 排序并返回
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_results]
    
    def hybrid_search(self, query: str, 
                     dense_results: List[Dict[str, Any]], 
                     top_k: int = 50,
                     dataset: Optional[str] = None,
                     qid: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        执行混合检索
        
        Args:
            query: 查询字符串
            dense_results: 向量检索结果
            top_k: 返回结果数量
            dataset: 数据集名称（用于命名空间过滤）
            qid: 问题ID（用于命名空间过滤）
            
        Returns:
            融合后的检索结果
        """
        if not self.bm25_index:
            logger.warning("Sparse index not built, returning dense results only")
            return dense_results[:top_k]
        
        # 执行稀疏检索
        sparse_candidates = self._sparse_search(query, top_k * 2)
        
        # 应用命名空间过滤
        if self.enable_namespace_filtering:
            sparse_candidates = self._apply_namespace_filter(sparse_candidates, dataset, qid)
        
        # 转换向量检索结果为索引-分数格式
        dense_candidates = []
        for i, result in enumerate(dense_results[:top_k * 2]):
            # 假设结果中包含原始索引信息
            original_idx = result.get('index', i)
            similarity = result.get('similarity', result.get('score', 0.0))
            dense_candidates.append((original_idx, similarity))
        
        # 执行融合
        if self.fusion_strategy == FusionStrategy.RRF:
            fused_results = self._rrf_fusion(sparse_candidates, dense_candidates)
        elif self.fusion_strategy == FusionStrategy.LINEAR:
            fused_results = self._linear_fusion(sparse_candidates, dense_candidates)
        else:
            logger.warning(f"Unknown fusion strategy: {self.fusion_strategy}, using RRF")
            fused_results = self._rrf_fusion(sparse_candidates, dense_candidates)
        
        # 构建最终结果
        final_results = []
        for idx, fused_score in fused_results[:top_k]:
            if idx < len(self.corpus_texts):
                result = {
                    'content': self.corpus_texts[idx],
                    'metadata': self.corpus_metadata[idx],
                    'fused_score': fused_score,
                    'fusion_strategy': self.fusion_strategy,
                    'index': idx
                }
                
                # 添加原始分数信息
                sparse_score = next((s for i, s in sparse_candidates if i == idx), 0.0)
                dense_score = next((s for i, s in dense_candidates if i == idx), 0.0)
                
                result['scores'] = {
                    'sparse': sparse_score,
                    'dense': dense_score,
                    'fused': fused_score
                }
                
                final_results.append(result)
        
        logger.info(f"Hybrid search completed: {len(sparse_candidates)} sparse + {len(dense_candidates)} dense -> {len(final_results)} fused results")
        return final_results
    
    def update_fusion_weights(self, sparse_weight: float, dense_weight: float):
        """
        动态更新融合权重
        
        Args:
            sparse_weight: 稀疏检索权重
            dense_weight: 向量检索权重
        """
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        logger.info(f"Fusion weights updated: sparse={sparse_weight}, dense={dense_weight}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取混合检索统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'fusion_strategy': self.fusion_strategy,
            'sparse_weight': self.sparse_weight,
            'dense_weight': self.dense_weight,
            'rrf_k': self.rrf_k,
            'enable_splade': self.enable_splade,
            'splade_available': SPLADE_AVAILABLE,
            'bm25_available': BM25Okapi is not None,
            'corpus_size': len(self.corpus_texts),
            'namespace_filtering': self.enable_namespace_filtering
        }

# 便利函数
def create_hybrid_searcher(config: Dict[str, Any]) -> HybridSearcher:
    """创建混合检索器实例"""
    return HybridSearcher(config)

def hybrid_search_with_fallback(query: str, 
                               dense_results: List[Dict[str, Any]], 
                               hybrid_searcher: Optional[HybridSearcher] = None,
                               **kwargs) -> List[Dict[str, Any]]:
    """
    带回退机制的混合检索
    
    Args:
        query: 查询字符串
        dense_results: 向量检索结果
        hybrid_searcher: 混合检索器实例
        **kwargs: 其他参数
        
    Returns:
        检索结果
    """
    if hybrid_searcher is None:
        logger.warning("No hybrid searcher provided, returning dense results only")
        return dense_results
    
    try:
        return hybrid_searcher.hybrid_search(query, dense_results, **kwargs)
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}, falling back to dense results")
        return dense_results