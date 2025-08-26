#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索模块 - 统一的检索和重排接口

包含以下组件：
- hybrid_search: 混合检索（BM25/SPLADE + 向量检索融合）
- query_planner: 查询规划和子问题分解
- path_aware_ranker: 图扩展和路径感知重排
- diversity_scheduler: 候选多样性调度
- retrieval_guardrail: 检索保障和兜底策略
"""

from .hybrid_search import (
    HybridSearcher,
    FusionStrategy,
    create_hybrid_searcher,
    hybrid_search_with_fallback
)

# 查询规划模块
from .query_planner import (
    QueryPlanner,
    SubQuery,
    QueryPlan,
    RuleBasedDecomposer,
    LLMBasedRewriter,
    create_query_planner,
    plan_and_execute_query
)

# 路径感知重排模块
from .path_aware_ranker import (
    PathAwareRanker,
    LightweightGraph,
    GraphExtractor,
    create_path_aware_ranker,
    rerank_with_path_awareness
)

# 检索保障模块
from .retrieval_guardrail import (
    RetrievalGuardrail,
    GuardrailConfig,
    RetrievalResult,
    GuardrailLevel,
    FallbackStrategy,
    RetrievalStatus,
    FallbackHandler,
    create_retrieval_guardrail,
    create_guardrail_config
)

# 多样性调度模块
from .diversity_scheduler import (
    DiversityScheduler,
    DiversityConfig,
    CandidateItem,
    DiversityResult,
    DiversityStrategy,
    DeduplicationMethod,
    CoverageMetric,
    DiversityEvaluator,
    DeduplicationProcessor,
    create_diversity_scheduler,
    create_diversity_config
)

__all__ = [
    'HybridSearcher',
    'FusionStrategy', 
    'create_hybrid_searcher',
    'hybrid_search_with_fallback',
    'QueryPlanner',
    'SubQuery',
    'QueryPlan',
    'RuleBasedDecomposer',
    'LLMBasedRewriter',
    'create_query_planner',
    'plan_and_execute_query',
    'PathAwareRanker',
    'LightweightGraph',
    'GraphExtractor',
    'create_path_aware_ranker',
    'rerank_with_path_awareness',
    'RetrievalGuardrail',
    'GuardrailConfig',
    'RetrievalResult',
    'GuardrailLevel',
    'FallbackStrategy',
    'RetrievalStatus',
    'FallbackHandler',
    'create_retrieval_guardrail',
    'create_guardrail_config',
    'DiversityScheduler',
    'DiversityConfig',
    'CandidateItem',
    'DiversityResult',
    'DiversityStrategy',
    'DeduplicationMethod',
    'CoverageMetric',
    'DiversityEvaluator',
    'DeduplicationProcessor',
    'create_diversity_scheduler',
    'create_diversity_config'
]

__version__ = '1.0.0'