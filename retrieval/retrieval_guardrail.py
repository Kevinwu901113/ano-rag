#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索最低保障与失败兜底策略模块

实现功能：
1. 多层兜底机制
2. 检索质量评估
3. 失败恢复策略
4. 最低保障检索
5. 降级策略管理
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from abc import ABC, abstractmethod

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("numpy not available, some features may be limited")

logger = logging.getLogger(__name__)

class GuardrailLevel(Enum):
    """保障级别"""
    STRICT = "strict"          # 严格模式：必须满足所有条件
    MODERATE = "moderate"      # 适中模式：满足基本条件即可
    PERMISSIVE = "permissive"  # 宽松模式：尽力而为
    EMERGENCY = "emergency"    # 紧急模式：最低保障

class FallbackStrategy(Enum):
    """兜底策略"""
    KEYWORD_SEARCH = "keyword_search"      # 关键词搜索
    FUZZY_MATCH = "fuzzy_match"            # 模糊匹配
    RANDOM_SAMPLE = "random_sample"        # 随机采样
    CACHED_RESULTS = "cached_results"      # 缓存结果
    DEFAULT_RESPONSE = "default_response"  # 默认响应

class RetrievalStatus(Enum):
    """检索状态"""
    SUCCESS = "success"        # 成功
    PARTIAL = "partial"        # 部分成功
    FALLBACK = "fallback"      # 兜底
    FAILED = "failed"          # 失败

@dataclass
class GuardrailConfig:
    """保障配置"""
    level: GuardrailLevel = GuardrailLevel.MODERATE
    min_results: int = 1                    # 最少结果数
    min_score: float = 0.1                  # 最低分数
    max_retries: int = 3                    # 最大重试次数
    timeout_seconds: float = 30.0           # 超时时间
    fallback_strategies: List[FallbackStrategy] = field(
        default_factory=lambda: [FallbackStrategy.KEYWORD_SEARCH, FallbackStrategy.FUZZY_MATCH]
    )
    enable_caching: bool = True             # 启用缓存
    cache_ttl: int = 3600                   # 缓存TTL

@dataclass
class RetrievalResult:
    """检索结果"""
    status: RetrievalStatus
    results: List[Dict[str, Any]]
    score: float
    strategy_used: str
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    fallback_applied: bool = False
    error_message: Optional[str] = None

class FallbackHandler(ABC):
    """兜底处理器抽象基类"""
    
    @abstractmethod
    def can_handle(self, query: str, context: Dict[str, Any]) -> bool:
        """检查是否可以处理该查询"""
        pass
    
    @abstractmethod
    def handle(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理查询"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """获取优先级（数字越小优先级越高）"""
        pass

class KeywordSearchHandler(FallbackHandler):
    """关键词搜索处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('keyword_search', {})
        self.min_keyword_length = self.config.get('min_keyword_length', 2)
        self.max_keywords = self.config.get('max_keywords', 10)
    
    def can_handle(self, query: str, context: Dict[str, Any]) -> bool:
        """检查是否可以处理"""
        keywords = self._extract_keywords(query)
        return len(keywords) > 0
    
    def handle(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理关键词搜索"""
        keywords = self._extract_keywords(query)
        
        # 模拟关键词搜索结果
        results = []
        for i, keyword in enumerate(keywords[:3]):  # 取前3个关键词
            results.append({
                'id': f'keyword_{i}',
                'content': f'Content related to {keyword}',
                'score': 0.5 - i * 0.1,
                'source': 'keyword_search',
                'keyword': keyword
            })
        
        return results
    
    def get_priority(self) -> int:
        return 1
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = query.lower().split()
        keywords = [w for w in words if len(w) >= self.min_keyword_length]
        return keywords[:self.max_keywords]

class FuzzyMatchHandler(FallbackHandler):
    """模糊匹配处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('fuzzy_match', {})
        self.similarity_threshold = self.config.get('similarity_threshold', 0.3)
    
    def can_handle(self, query: str, context: Dict[str, Any]) -> bool:
        """检查是否可以处理"""
        return len(query.strip()) > 0
    
    def handle(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理模糊匹配"""
        # 模拟模糊匹配结果
        results = [
            {
                'id': 'fuzzy_1',
                'content': f'Fuzzy match result for: {query[:50]}...',
                'score': 0.4,
                'source': 'fuzzy_match',
                'similarity': self.similarity_threshold + 0.1
            },
            {
                'id': 'fuzzy_2',
                'content': f'Alternative fuzzy match for: {query[:50]}...',
                'score': 0.3,
                'source': 'fuzzy_match',
                'similarity': self.similarity_threshold
            }
        ]
        
        return results
    
    def get_priority(self) -> int:
        return 2

class CachedResultsHandler(FallbackHandler):
    """缓存结果处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('cached_results', {})
        self.cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.ttl = self.config.get('ttl', 3600)
    
    def can_handle(self, query: str, context: Dict[str, Any]) -> bool:
        """检查是否有缓存结果"""
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache:
            # 检查是否过期
            if time.time() - self.cache_timestamps[cache_key] < self.ttl:
                return True
            else:
                # 清理过期缓存
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
        return False
    
    def handle(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """返回缓存结果"""
        cache_key = self._get_cache_key(query)
        return self.cache.get(cache_key, [])
    
    def get_priority(self) -> int:
        return 0  # 最高优先级
    
    def cache_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """缓存结果"""
        cache_key = self._get_cache_key(query)
        self.cache[cache_key] = results
        self.cache_timestamps[cache_key] = time.time()
    
    def _get_cache_key(self, query: str) -> str:
        """生成缓存键"""
        return f"query_{hash(query.lower().strip())}"

class RetrievalGuardrail:
    """检索保障器"""
    
    def __init__(self, config: Dict[str, Any]):
        from utils.config_loader_helper import load_config_with_fallback
        
        # 尝试从外部配置文件加载，如果失败则使用内联配置
        config_file_path = config.get('retrieval_guardrail_config_file')
        
        # 默认配置
        default_config = {
            'level': 'moderate',
            'min_results': 1,
            'min_score': 0.1,
            'max_retries': 3,
            'timeout_seconds': 30.0,
            'enable_caching': True
        }
        
        # 加载配置（外部文件优先，回退到内联配置，最后使用默认配置）
        inline_config = config.get('retrieval_guardrail', {})
        self.config = load_config_with_fallback(config_file_path, inline_config, default_config)
        
        # 保障配置
        self.guardrail_config = GuardrailConfig(
            level=GuardrailLevel(self.config.get('level', default_config['level'])),
            min_results=self.config.get('min_results', default_config['min_results']),
            min_score=self.config.get('min_score', default_config['min_score']),
            max_retries=self.config.get('max_retries', default_config['max_retries']),
            timeout_seconds=self.config.get('timeout_seconds', default_config['timeout_seconds']),
            enable_caching=self.config.get('enable_caching', default_config['enable_caching'])
        )
        
        # 兜底处理器
        self.fallback_handlers: List[FallbackHandler] = []
        self._initialize_handlers()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'fallback_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0
        }
        
        # 线程锁
        self._lock = threading.Lock()
        
        logger.info(f"RetrievalGuardrail initialized with level: {self.guardrail_config.level.value}")
    
    def _initialize_handlers(self) -> None:
        """初始化兜底处理器"""
        # 添加缓存处理器
        if self.guardrail_config.enable_caching:
            self.fallback_handlers.append(CachedResultsHandler(self.config))
        
        # 添加关键词搜索处理器
        self.fallback_handlers.append(KeywordSearchHandler(self.config))
        
        # 添加模糊匹配处理器
        self.fallback_handlers.append(FuzzyMatchHandler(self.config))
        
        # 按优先级排序
        self.fallback_handlers.sort(key=lambda h: h.get_priority())
    
    def retrieve_with_guardrail(self, query: str, 
                               primary_retriever: Callable[[str], List[Dict[str, Any]]],
                               context: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """
        带保障的检索
        
        Args:
            query: 查询字符串
            primary_retriever: 主检索器函数
            context: 上下文信息
            
        Returns:
            检索结果
        """
        start_time = time.time()
        context = context or {}
        
        with self._lock:
            self.stats['total_requests'] += 1
        
        try:
            # 尝试主检索器
            result = self._try_primary_retrieval(query, primary_retriever, context)
            
            if self._is_result_acceptable(result):
                # 缓存成功结果
                self._cache_result(query, result.results)
                
                with self._lock:
                    self.stats['successful_requests'] += 1
                
                return result
            
            # 主检索失败，尝试兜底策略
            return self._apply_fallback_strategies(query, context, start_time)
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return self._apply_fallback_strategies(query, context, start_time, str(e))
    
    def _try_primary_retrieval(self, query: str, 
                              primary_retriever: Callable[[str], List[Dict[str, Any]]],
                              context: Dict[str, Any]) -> RetrievalResult:
        """尝试主检索"""
        start_time = time.time()
        
        try:
            results = primary_retriever(query)
            execution_time = time.time() - start_time
            
            # 计算平均分数
            avg_score = 0.0
            if results:
                scores = [r.get('score', 0.0) for r in results]
                avg_score = sum(scores) / len(scores) if scores else 0.0
            
            return RetrievalResult(
                status=RetrievalStatus.SUCCESS if results else RetrievalStatus.FAILED,
                results=results,
                score=avg_score,
                strategy_used="primary_retriever",
                execution_time=execution_time,
                metadata={'query': query, 'context': context}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return RetrievalResult(
                status=RetrievalStatus.FAILED,
                results=[],
                score=0.0,
                strategy_used="primary_retriever",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _apply_fallback_strategies(self, query: str, context: Dict[str, Any],
                                  start_time: float, error_message: Optional[str] = None) -> RetrievalResult:
        """应用兜底策略"""
        logger.warning(f"Applying fallback strategies for query: {query[:50]}...")
        
        with self._lock:
            self.stats['fallback_requests'] += 1
        
        # 尝试每个兜底处理器
        for handler in self.fallback_handlers:
            try:
                if handler.can_handle(query, context):
                    results = handler.handle(query, context)
                    
                    if results:
                        execution_time = time.time() - start_time
                        
                        # 计算平均分数
                        scores = [r.get('score', 0.0) for r in results]
                        avg_score = sum(scores) / len(scores) if scores else 0.0
                        
                        return RetrievalResult(
                            status=RetrievalStatus.FALLBACK,
                            results=results,
                            score=avg_score,
                            strategy_used=handler.__class__.__name__,
                            execution_time=execution_time,
                            fallback_applied=True,
                            metadata={'query': query, 'context': context}
                        )
                        
            except Exception as e:
                logger.error(f"Fallback handler {handler.__class__.__name__} failed: {e}")
                continue
        
        # 所有策略都失败
        execution_time = time.time() - start_time
        
        with self._lock:
            self.stats['failed_requests'] += 1
        
        return RetrievalResult(
            status=RetrievalStatus.FAILED,
            results=[],
            score=0.0,
            strategy_used="none",
            execution_time=execution_time,
            fallback_applied=True,
            error_message=error_message or "All fallback strategies failed"
        )
    
    def _is_result_acceptable(self, result: RetrievalResult) -> bool:
        """检查结果是否可接受"""
        if result.status == RetrievalStatus.FAILED:
            return False
        
        # 检查最少结果数
        if len(result.results) < self.guardrail_config.min_results:
            return False
        
        # 检查最低分数
        if result.score < self.guardrail_config.min_score:
            return False
        
        return True
    
    def _cache_result(self, query: str, results: List[Dict[str, Any]]) -> None:
        """缓存结果"""
        if not self.guardrail_config.enable_caching:
            return
        
        # 找到缓存处理器
        for handler in self.fallback_handlers:
            if isinstance(handler, CachedResultsHandler):
                handler.cache_results(query, results)
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total = self.stats['total_requests']
            if total > 0:
                success_rate = self.stats['successful_requests'] / total
                fallback_rate = self.stats['fallback_requests'] / total
                failure_rate = self.stats['failed_requests'] / total
            else:
                success_rate = fallback_rate = failure_rate = 0.0
            
            return {
                'total_requests': total,
                'successful_requests': self.stats['successful_requests'],
                'fallback_requests': self.stats['fallback_requests'],
                'failed_requests': self.stats['failed_requests'],
                'success_rate': success_rate,
                'fallback_rate': fallback_rate,
                'failure_rate': failure_rate,
                'guardrail_level': self.guardrail_config.level.value
            }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._lock:
            self.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'fallback_requests': 0,
                'failed_requests': 0,
                'average_response_time': 0.0
            }
        logger.info("Statistics reset")

# 便利函数
def create_retrieval_guardrail(config: Dict[str, Any]) -> RetrievalGuardrail:
    """创建检索保障器"""
    return RetrievalGuardrail(config)

def create_guardrail_config(level: str = "moderate", **kwargs) -> GuardrailConfig:
    """
    创建保障配置的便利函数
    
    Args:
        level: 保障级别
        **kwargs: 其他配置参数
        
    Returns:
        保障配置实例
    """
    return GuardrailConfig(
        level=GuardrailLevel(level),
        min_results=kwargs.get('min_results', 1),
        min_score=kwargs.get('min_score', 0.1),
        max_retries=kwargs.get('max_retries', 3),
        timeout_seconds=kwargs.get('timeout_seconds', 30.0),
        enable_caching=kwargs.get('enable_caching', True)
    )