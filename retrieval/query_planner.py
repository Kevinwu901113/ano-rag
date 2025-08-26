#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询规划与子问题分解模块

实现功能：
1. 原问题自动分解为若干子问题
2. 基于规则与语言模型的查询改写器
3. 结构化子查询列表生成
4. 并行执行和结果合并
5. 重复实体聚类与证据去重
"""

import logging
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available, LLM-based query rewriting will be disabled")

# 导入配置加载辅助函数
from ..utils.config_loader_helper import load_external_config, load_config_with_fallback

logger = logging.getLogger(__name__)

@dataclass
class SubQuery:
    """子查询结构"""
    text: str                           # 查询文本
    target_entities: List[str] = field(default_factory=list)  # 目标实体类型
    target_predicates: List[str] = field(default_factory=list)  # 目标谓词
    constraint_keywords: List[str] = field(default_factory=list)  # 约束关键词
    priority: float = 1.0               # 优先级权重
    query_type: str = "general"         # 查询类型：general, entity, relation, temporal
    original_query: str = ""            # 原始查询
    decomposition_method: str = ""      # 分解方法
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'text': self.text,
            'target_entities': self.target_entities,
            'target_predicates': self.target_predicates,
            'constraint_keywords': self.constraint_keywords,
            'priority': self.priority,
            'query_type': self.query_type,
            'original_query': self.original_query,
            'decomposition_method': self.decomposition_method
        }

@dataclass
class QueryPlan:
    """查询计划"""
    original_query: str
    sub_queries: List[SubQuery] = field(default_factory=list)
    execution_strategy: str = "parallel"  # parallel, sequential, adaptive
    merge_strategy: str = "weighted"      # weighted, ranked, clustered
    max_results_per_subquery: int = 20
    total_max_results: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'original_query': self.original_query,
            'sub_queries': [sq.to_dict() for sq in self.sub_queries],
            'execution_strategy': self.execution_strategy,
            'merge_strategy': self.merge_strategy,
            'max_results_per_subquery': self.max_results_per_subquery,
            'total_max_results': self.total_max_results
        }

class RuleBasedDecomposer:
    """基于规则的查询分解器"""
    
    def __init__(self, config: Dict[str, Any]):
        # 获取查询规划器配置
        query_planner_config = config.get('query_planner', {})
        
        # 默认配置
        default_decomposer_config = {
            'conjunction_patterns': [
                r'和|与|以及|还有|同时|并且',
                r'\s+and\s+',
                r'，|；|;'
            ],
            'question_patterns': [
                r'什么是|什么叫|如何|怎么|为什么|哪些|哪个|谁是|何时|何地',
                r'what\s+is|what\s+are|how\s+to|why\s+|which\s+|who\s+is|when\s+|where\s+'
            ],
            'entity_patterns': [
                r'关于(.+?)的',
                r'(.+?)相关',
                r'(.+?)方面'
            ],
            'predicate_patterns': [
                r'(.+?)的(.+?)是什么',
                r'(.+?)如何(.+?)',
                r'(.+?)与(.+?)的关系'
            ]
        }
        
        # 从外部配置文件加载配置
        config_file_path = query_planner_config.get('query_decomposer_config_file', '')
        if config_file_path:
            self.config = load_config_with_fallback(
                config_file_path, 
                default_decomposer_config, 
                config_key='rule_based_decomposer'
            )
        else:
            # 使用内联配置或默认配置
            inline_config = query_planner_config.get('rule_based_decomposer', {})
            self.config = {**default_decomposer_config, **inline_config}
        
        # 分解规则配置
        self.conjunction_patterns = self.config.get('conjunction_patterns', default_decomposer_config['conjunction_patterns'])
        self.question_patterns = self.config.get('question_patterns', default_decomposer_config['question_patterns'])
        self.entity_patterns = self.config.get('entity_patterns', default_decomposer_config['entity_patterns'])
        self.predicate_patterns = self.config.get('predicate_patterns', default_decomposer_config['predicate_patterns'])
        
        logger.info("RuleBasedDecomposer initialized with external config support")
    
    def decompose(self, query: str) -> List[SubQuery]:
        """
        基于规则分解查询
        
        Args:
            query: 原始查询
            
        Returns:
            子查询列表
        """
        sub_queries = []
        
        # 1. 连接词分解
        conjunction_subqueries = self._decompose_by_conjunction(query)
        sub_queries.extend(conjunction_subqueries)
        
        # 2. 实体提取分解
        entity_subqueries = self._decompose_by_entities(query)
        sub_queries.extend(entity_subqueries)
        
        # 3. 谓词关系分解
        predicate_subqueries = self._decompose_by_predicates(query)
        sub_queries.extend(predicate_subqueries)
        
        # 4. 如果没有分解出子查询，返回原查询
        if not sub_queries:
            sub_queries.append(SubQuery(
                text=query,
                original_query=query,
                decomposition_method="no_decomposition",
                query_type="general"
            ))
        
        logger.debug(f"Rule-based decomposition: {query} -> {len(sub_queries)} sub-queries")
        return sub_queries
    
    def _decompose_by_conjunction(self, query: str) -> List[SubQuery]:
        """基于连接词分解"""
        sub_queries = []
        
        for pattern in self.conjunction_patterns:
            parts = re.split(pattern, query)
            if len(parts) > 1:
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        sub_queries.append(SubQuery(
                            text=part,
                            original_query=query,
                            decomposition_method="conjunction",
                            priority=1.0 / len(parts),
                            query_type="general"
                        ))
                break  # 只使用第一个匹配的模式
        
        return sub_queries
    
    def _decompose_by_entities(self, query: str) -> List[SubQuery]:
        """基于实体提取分解"""
        sub_queries = []
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                entity = match.strip()
                if entity:
                    sub_queries.append(SubQuery(
                        text=f"关于{entity}的信息",
                        target_entities=[entity],
                        original_query=query,
                        decomposition_method="entity_extraction",
                        query_type="entity"
                    ))
        
        return sub_queries
    
    def _decompose_by_predicates(self, query: str) -> List[SubQuery]:
        """基于谓词关系分解"""
        sub_queries = []
        
        for pattern in self.predicate_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if len(match) >= 2:
                    entity1, predicate = match[0].strip(), match[1].strip()
                    sub_queries.append(SubQuery(
                        text=f"{entity1}的{predicate}",
                        target_entities=[entity1],
                        target_predicates=[predicate],
                        original_query=query,
                        decomposition_method="predicate_extraction",
                        query_type="relation"
                    ))
        
        return sub_queries

class LLMBasedRewriter:
    """基于语言模型的查询改写器"""
    
    def __init__(self, config: Dict[str, Any]):
        # 获取查询规划器配置
        query_planner_config = config.get('query_planner', {})
        
        # 默认配置
        default_rewriter_config = {
            'enabled': False,
            'model': 'gpt-3.5-turbo',
            'max_tokens': 500,
            'temperature': 0.3,
            'api_key': '',
            'prompt_template': ''
        }
        
        # 从外部配置文件加载配置
        config_file_path = query_planner_config.get('query_decomposer_config_file', '')
        if config_file_path:
            self.config = load_config_with_fallback(
                config_file_path, 
                default_rewriter_config, 
                config_key='llm_based_rewriter'
            )
        else:
            # 使用内联配置或默认配置
            inline_config = query_planner_config.get('llm_based_rewriter', {})
            self.config = {**default_rewriter_config, **inline_config}
        
        self.enabled = self.config.get('enabled', False) and OPENAI_AVAILABLE
        
        if self.enabled:
            self.api_key = self.config.get('api_key', '')
            self.model = self.config.get('model', 'gpt-3.5-turbo')
            self.max_tokens = self.config.get('max_tokens', 500)
            self.temperature = self.config.get('temperature', 0.3)
            self.prompt_template = self.config.get('prompt_template', '')
            
            if self.api_key:
                openai.api_key = self.api_key
            
            logger.info(f"LLMBasedRewriter initialized with model: {self.model} (external config support)")
        else:
            logger.info("LLMBasedRewriter disabled")
    
    def rewrite_query(self, query: str, context: Optional[str] = None) -> List[SubQuery]:
        """
        使用LLM改写查询
        
        Args:
            query: 原始查询
            context: 上下文信息
            
        Returns:
            改写后的子查询列表
        """
        if not self.enabled:
            return [SubQuery(text=query, original_query=query, decomposition_method="no_rewrite")]
        
        try:
            prompt = self._build_rewrite_prompt(query, context)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的查询分解和改写助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            result_text = response.choices[0].message.content.strip()
            return self._parse_llm_response(result_text, query)
            
        except Exception as e:
            logger.error(f"LLM rewriting failed: {e}")
            return [SubQuery(text=query, original_query=query, decomposition_method="llm_failed")]
    
    def _build_rewrite_prompt(self, query: str, context: Optional[str] = None) -> str:
        """构建LLM提示词"""
        prompt = f"""
请将以下查询分解为多个子查询，每个子查询应该：
1. 专注于一个特定的信息需求
2. 包含明确的实体和谓词
3. 便于检索系统处理

原始查询：{query}

请以JSON格式返回结果，包含以下字段：
- text: 子查询文本
- target_entities: 目标实体列表
- target_predicates: 目标谓词列表
- constraint_keywords: 约束关键词列表
- query_type: 查询类型（general/entity/relation/temporal）

示例格式：
[
  {{
    "text": "什么是人工智能",
    "target_entities": ["人工智能"],
    "target_predicates": ["定义", "概念"],
    "constraint_keywords": [],
    "query_type": "entity"
  }}
]
"""
        
        if context:
            prompt += f"\n\n上下文信息：{context}"
        
        return prompt
    
    def _parse_llm_response(self, response_text: str, original_query: str) -> List[SubQuery]:
        """解析LLM响应"""
        try:
            # 尝试解析JSON
            data = json.loads(response_text)
            
            sub_queries = []
            for item in data:
                sub_query = SubQuery(
                    text=item.get('text', ''),
                    target_entities=item.get('target_entities', []),
                    target_predicates=item.get('target_predicates', []),
                    constraint_keywords=item.get('constraint_keywords', []),
                    query_type=item.get('query_type', 'general'),
                    original_query=original_query,
                    decomposition_method="llm_rewrite"
                )
                sub_queries.append(sub_query)
            
            return sub_queries if sub_queries else [SubQuery(
                text=original_query, 
                original_query=original_query, 
                decomposition_method="llm_parse_failed"
            )]
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {response_text[:100]}...")
            return [SubQuery(
                text=original_query, 
                original_query=original_query, 
                decomposition_method="llm_parse_failed"
            )]

class QueryPlanner:
    """查询规划器 - 统一的查询分解和改写接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('query_planner', {})
        
        # 初始化组件
        self.rule_decomposer = RuleBasedDecomposer(config)
        self.llm_rewriter = LLMBasedRewriter(config)
        
        # 配置参数
        self.enable_rule_decomposition = self.config.get('enable_rule_decomposition', True)
        self.enable_llm_rewriting = self.config.get('enable_llm_rewriting', False)
        self.max_sub_queries = self.config.get('max_sub_queries', 5)
        self.min_query_length = self.config.get('min_query_length', 3)
        
        # 执行配置
        self.execution_strategy = self.config.get('execution_strategy', 'parallel')
        self.merge_strategy = self.config.get('merge_strategy', 'weighted')
        self.max_workers = self.config.get('max_workers', 4)
        
        logger.info(f"QueryPlanner initialized: rule={self.enable_rule_decomposition}, "
                   f"llm={self.enable_llm_rewriting}, strategy={self.execution_strategy}")
    
    def plan_query(self, query: str, context: Optional[str] = None) -> QueryPlan:
        """
        规划查询执行
        
        Args:
            query: 原始查询
            context: 上下文信息
            
        Returns:
            查询计划
        """
        logger.info(f"Planning query: {query[:50]}...")
        
        sub_queries = []
        
        # 1. 基于规则的分解
        if self.enable_rule_decomposition:
            rule_subqueries = self.rule_decomposer.decompose(query)
            sub_queries.extend(rule_subqueries)
            logger.debug(f"Rule decomposition produced {len(rule_subqueries)} sub-queries")
        
        # 2. 基于LLM的改写
        if self.enable_llm_rewriting:
            llm_subqueries = self.llm_rewriter.rewrite_query(query, context)
            sub_queries.extend(llm_subqueries)
            logger.debug(f"LLM rewriting produced {len(llm_subqueries)} sub-queries")
        
        # 3. 去重和过滤
        sub_queries = self._deduplicate_subqueries(sub_queries)
        sub_queries = self._filter_subqueries(sub_queries)
        
        # 4. 限制数量
        if len(sub_queries) > self.max_sub_queries:
            sub_queries = sorted(sub_queries, key=lambda x: x.priority, reverse=True)[:self.max_sub_queries]
        
        # 5. 创建查询计划
        plan = QueryPlan(
            original_query=query,
            sub_queries=sub_queries,
            execution_strategy=self.execution_strategy,
            merge_strategy=self.merge_strategy
        )
        
        logger.info(f"Query plan created with {len(sub_queries)} sub-queries")
        return plan
    
    def execute_plan(self, plan: QueryPlan, retrieval_fn: Callable) -> List[Dict[str, Any]]:
        """
        执行查询计划
        
        Args:
            plan: 查询计划
            retrieval_fn: 检索函数
            
        Returns:
            合并后的检索结果
        """
        logger.info(f"Executing query plan with {len(plan.sub_queries)} sub-queries")
        
        if plan.execution_strategy == 'parallel':
            results = self._execute_parallel(plan, retrieval_fn)
        elif plan.execution_strategy == 'sequential':
            results = self._execute_sequential(plan, retrieval_fn)
        else:
            logger.warning(f"Unknown execution strategy: {plan.execution_strategy}, using parallel")
            results = self._execute_parallel(plan, retrieval_fn)
        
        # 合并结果
        merged_results = self._merge_results(results, plan)
        
        logger.info(f"Query execution completed: {len(merged_results)} final results")
        return merged_results
    
    def _deduplicate_subqueries(self, sub_queries: List[SubQuery]) -> List[SubQuery]:
        """去重子查询"""
        seen_texts = set()
        unique_subqueries = []
        
        for sq in sub_queries:
            if sq.text not in seen_texts:
                seen_texts.add(sq.text)
                unique_subqueries.append(sq)
        
        logger.debug(f"Deduplication: {len(sub_queries)} -> {len(unique_subqueries)} sub-queries")
        return unique_subqueries
    
    def _filter_subqueries(self, sub_queries: List[SubQuery]) -> List[SubQuery]:
        """过滤子查询"""
        filtered = []
        
        for sq in sub_queries:
            # 过滤太短的查询
            if len(sq.text.strip()) < self.min_query_length:
                continue
            
            # 过滤空查询
            if not sq.text.strip():
                continue
            
            filtered.append(sq)
        
        logger.debug(f"Filtering: {len(sub_queries)} -> {len(filtered)} sub-queries")
        return filtered
    
    def _execute_parallel(self, plan: QueryPlan, retrieval_fn: Callable) -> List[Tuple[SubQuery, List[Dict[str, Any]]]]:
        """并行执行子查询"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_subquery = {
                executor.submit(retrieval_fn, sq.text, plan.max_results_per_subquery): sq
                for sq in plan.sub_queries
            }
            
            # 收集结果
            for future in as_completed(future_to_subquery):
                sub_query = future_to_subquery[future]
                try:
                    result = future.result()
                    results.append((sub_query, result))
                    logger.debug(f"Sub-query completed: {sub_query.text[:30]}... -> {len(result)} results")
                except Exception as e:
                    logger.error(f"Sub-query failed: {sub_query.text[:30]}... -> {e}")
                    results.append((sub_query, []))
        
        return results
    
    def _execute_sequential(self, plan: QueryPlan, retrieval_fn: Callable) -> List[Tuple[SubQuery, List[Dict[str, Any]]]]:
        """顺序执行子查询"""
        results = []
        
        for sq in plan.sub_queries:
            try:
                result = retrieval_fn(sq.text, plan.max_results_per_subquery)
                results.append((sq, result))
                logger.debug(f"Sub-query completed: {sq.text[:30]}... -> {len(result)} results")
            except Exception as e:
                logger.error(f"Sub-query failed: {sq.text[:30]}... -> {e}")
                results.append((sq, []))
        
        return results
    
    def _merge_results(self, results: List[Tuple[SubQuery, List[Dict[str, Any]]]], 
                      plan: QueryPlan) -> List[Dict[str, Any]]:
        """
        合并子查询结果
        
        Args:
            results: 子查询结果列表
            plan: 查询计划
            
        Returns:
            合并后的结果
        """
        if plan.merge_strategy == 'weighted':
            return self._merge_weighted(results, plan)
        elif plan.merge_strategy == 'ranked':
            return self._merge_ranked(results, plan)
        elif plan.merge_strategy == 'clustered':
            return self._merge_clustered(results, plan)
        else:
            logger.warning(f"Unknown merge strategy: {plan.merge_strategy}, using weighted")
            return self._merge_weighted(results, plan)
    
    def _merge_weighted(self, results: List[Tuple[SubQuery, List[Dict[str, Any]]]], 
                       plan: QueryPlan) -> List[Dict[str, Any]]:
        """加权合并策略"""
        merged_results = []
        seen_content = set()
        
        for sub_query, sub_results in results:
            for result in sub_results:
                content = result.get('content', '')
                
                # 去重
                if content in seen_content:
                    continue
                seen_content.add(content)
                
                # 添加子查询信息
                result_copy = result.copy()
                result_copy['sub_query_info'] = {
                    'sub_query': sub_query.to_dict(),
                    'weighted_score': result.get('similarity', 0.0) * sub_query.priority
                }
                
                merged_results.append(result_copy)
        
        # 按加权分数排序
        merged_results.sort(
            key=lambda x: x['sub_query_info']['weighted_score'], 
            reverse=True
        )
        
        return merged_results[:plan.total_max_results]
    
    def _merge_ranked(self, results: List[Tuple[SubQuery, List[Dict[str, Any]]]], 
                     plan: QueryPlan) -> List[Dict[str, Any]]:
        """排名合并策略"""
        # 简化实现：按原始相似度排序
        all_results = []
        seen_content = set()
        
        for sub_query, sub_results in results:
            for result in sub_results:
                content = result.get('content', '')
                if content not in seen_content:
                    seen_content.add(content)
                    result_copy = result.copy()
                    result_copy['sub_query_info'] = {'sub_query': sub_query.to_dict()}
                    all_results.append(result_copy)
        
        all_results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        return all_results[:plan.total_max_results]
    
    def _merge_clustered(self, results: List[Tuple[SubQuery, List[Dict[str, Any]]]], 
                        plan: QueryPlan) -> List[Dict[str, Any]]:
        """聚类合并策略"""
        # 简化实现：按实体聚类
        entity_clusters = defaultdict(list)
        
        for sub_query, sub_results in results:
            for result in sub_results:
                # 提取实体（简化版本）
                entities = result.get('metadata', {}).get('entities', [])
                cluster_key = tuple(sorted(entities)) if entities else 'general'
                
                result_copy = result.copy()
                result_copy['sub_query_info'] = {'sub_query': sub_query.to_dict()}
                entity_clusters[cluster_key].append(result_copy)
        
        # 从每个聚类中选择最佳结果
        merged_results = []
        for cluster_results in entity_clusters.values():
            cluster_results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
            merged_results.extend(cluster_results[:2])  # 每个聚类最多2个结果
        
        merged_results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        return merged_results[:plan.total_max_results]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'enable_rule_decomposition': self.enable_rule_decomposition,
            'enable_llm_rewriting': self.enable_llm_rewriting,
            'max_sub_queries': self.max_sub_queries,
            'execution_strategy': self.execution_strategy,
            'merge_strategy': self.merge_strategy,
            'max_workers': self.max_workers,
            'llm_available': OPENAI_AVAILABLE
        }

# 便利函数
def create_query_planner(config: Dict[str, Any]) -> QueryPlanner:
    """创建查询规划器实例"""
    return QueryPlanner(config)

def plan_and_execute_query(query: str, 
                          retrieval_fn: Callable,
                          planner: Optional[QueryPlanner] = None,
                          context: Optional[str] = None,
                          **kwargs) -> List[Dict[str, Any]]:
    """
    规划并执行查询的便利函数
    
    Args:
        query: 查询字符串
        retrieval_fn: 检索函数
        planner: 查询规划器实例
        context: 上下文信息
        **kwargs: 其他参数
        
    Returns:
        检索结果
    """
    if planner is None:
        logger.warning("No query planner provided, executing original query")
        return retrieval_fn(query, kwargs.get('top_k', 50))
    
    try:
        plan = planner.plan_query(query, context)
        return planner.execute_plan(plan, retrieval_fn)
    except Exception as e:
        logger.error(f"Query planning and execution failed: {e}")
        return retrieval_fn(query, kwargs.get('top_k', 50))