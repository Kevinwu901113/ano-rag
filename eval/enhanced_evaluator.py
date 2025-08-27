#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强评测模块

实现功能：
1. 路径召回率评测
2. 谓词一致率评测
3. 实体覆盖率评测
4. 传统评测指标（精确率、召回率、F1等）
5. 综合评测报告生成
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

# 导入相关模块
try:
    from query.query_processor import QueryProcessor
except ImportError:
    # 如果无法导入，使用模拟版本
    QueryProcessor = None
    
try:
    from llm import LocalLLM
except ImportError:
    LocalLLM = None
    
try:
    from utils.entity_predicate_normalizer import EntityNormalizer, PredicateNormalizer
except ImportError:
    EntityNormalizer = None
    PredicateNormalizer = None
    
try:
    from utils.enhanced_ner import EnhancedNER
except ImportError:
    EnhancedNER = None
    
try:
    from config import config
except ImportError:
    config = {}

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """评测结果"""
    query_id: str
    query: str
    ground_truth: Dict[str, Any]
    prediction: Dict[str, Any]
    
    # 传统指标
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # 新增指标
    path_recall_rate: float = 0.0
    predicate_consistency_rate: float = 0.0
    entity_coverage_rate: float = 0.0
    
    # 二跳检索指标
    two_hop_efficiency: float = 0.0
    bridge_entity_quality: float = 0.0
    candidate_pool_coverage: float = 0.0
    
    # 详细分析
    path_analysis: Dict[str, Any] = field(default_factory=dict)
    predicate_analysis: Dict[str, Any] = field(default_factory=dict)
    entity_analysis: Dict[str, Any] = field(default_factory=dict)
    two_hop_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # 执行信息
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationSummary:
    """评测汇总"""
    total_queries: int
    avg_precision: float
    avg_recall: float
    avg_f1_score: float
    avg_path_recall_rate: float
    avg_predicate_consistency_rate: float
    avg_entity_coverage_rate: float
    
    # 二跳检索指标
    avg_two_hop_efficiency: float = 0.0
    avg_bridge_entity_quality: float = 0.0
    avg_candidate_pool_coverage: float = 0.0
    
    # 分布统计
    score_distribution: Dict[str, List[float]] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    
    # 详细报告
    detailed_results: List[EvaluationResult] = field(default_factory=list)

class EnhancedEvaluator:
    """增强评测器"""
    
    def __init__(self, atomic_notes: List[Dict], embeddings=None, llm: Optional[LocalLLM] = None):
        self.processor = QueryProcessor(atomic_notes, embeddings, llm=llm)
        self.atomic_notes = atomic_notes
        
        # 配置参数
        eval_config = config.get('eval', {})
        self.batch_size = eval_config.get('batch_size', 16)
        self.metrics = eval_config.get('metrics', ["precision", "recall", "f1", "path_recall", "predicate_consistency", "entity_coverage"])
        
        # 初始化标准化器
        self.entity_normalizer = EntityNormalizer()
        self.predicate_normalizer = PredicateNormalizer()
        self.ner = EnhancedNER()
        
        # 统计信息
        self.stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'avg_execution_time': 0.0
        }
        
        logger.info(f"Enhanced evaluator initialized with metrics: {self.metrics}")
    
    def evaluate_batch(self, test_data: List[Dict[str, Any]]) -> EvaluationSummary:
        """
        批量评测
        
        Args:
            test_data: 测试数据，每个元素包含query, ground_truth等字段
            
        Returns:
            评测汇总结果
        """
        logger.info(f"Starting batch evaluation with {len(test_data)} queries")
        
        results = []
        total_time = 0.0
        
        for i, test_item in enumerate(test_data):
            try:
                start_time = time.time()
                
                # 执行查询
                query = test_item['query']
                ground_truth = test_item.get('ground_truth', {})
                query_id = test_item.get('id', f"query_{i}")
                
                prediction = self.processor.process(query)
                
                # 计算评测指标
                result = self._evaluate_single_query(
                    query_id=query_id,
                    query=query,
                    ground_truth=ground_truth,
                    prediction=prediction
                )
                
                execution_time = time.time() - start_time
                result.execution_time = execution_time
                total_time += execution_time
                
                results.append(result)
                self.stats['successful_evaluations'] += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_data)} queries")
                    
            except Exception as e:
                logger.error(f"Evaluation failed for query {i}: {e}")
                self.stats['failed_evaluations'] += 1
                continue
        
        self.stats['total_evaluations'] = len(test_data)
        self.stats['avg_execution_time'] = total_time / len(results) if results else 0.0
        
        # 生成汇总报告
        summary = self._generate_summary(results)
        
        logger.info(f"Batch evaluation completed. Success rate: {len(results)}/{len(test_data)}")
        return summary
    
    def _evaluate_single_query(self, query_id: str, query: str, 
                             ground_truth: Dict[str, Any], 
                             prediction: Dict[str, Any]) -> EvaluationResult:
        """
        单个查询评测
        
        Args:
            query_id: 查询ID
            query: 查询文本
            ground_truth: 标准答案
            prediction: 预测结果
            
        Returns:
            评测结果
        """
        result = EvaluationResult(
            query_id=query_id,
            query=query,
            ground_truth=ground_truth,
            prediction=prediction
        )
        
        # 1. 计算传统指标
        if "precision" in self.metrics or "recall" in self.metrics or "f1" in self.metrics:
            precision, recall, f1 = self._calculate_traditional_metrics(ground_truth, prediction)
            result.precision = precision
            result.recall = recall
            result.f1_score = f1
        
        # 2. 计算路径召回率
        if "path_recall" in self.metrics:
            path_recall, path_analysis = self._calculate_path_recall_rate(ground_truth, prediction)
            result.path_recall_rate = path_recall
            result.path_analysis = path_analysis
        
        # 3. 计算谓词一致率
        if "predicate_consistency" in self.metrics:
            predicate_consistency, predicate_analysis = self._calculate_predicate_consistency_rate(ground_truth, prediction)
            result.predicate_consistency_rate = predicate_consistency
            result.predicate_analysis = predicate_analysis
        
        # 4. 计算实体覆盖率
        if "entity_coverage" in self.metrics:
            entity_coverage, entity_analysis = self._calculate_entity_coverage_rate(ground_truth, prediction)
            result.entity_coverage_rate = entity_coverage
            result.entity_analysis = entity_analysis
        
        # 5. 计算二跳检索指标
        if "two_hop_efficiency" in self.metrics:
            two_hop_efficiency, two_hop_analysis = self._calculate_two_hop_metrics(ground_truth, prediction)
            result.two_hop_efficiency = two_hop_efficiency.get('efficiency', 0.0)
            result.bridge_entity_quality = two_hop_efficiency.get('bridge_quality', 0.0)
            result.candidate_pool_coverage = two_hop_efficiency.get('pool_coverage', 0.0)
            result.two_hop_analysis = two_hop_analysis
        
        return result
    
    def _calculate_traditional_metrics(self, ground_truth: Dict[str, Any], 
                                     prediction: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        计算传统评测指标（精确率、召回率、F1）
        
        Args:
            ground_truth: 标准答案
            prediction: 预测结果
            
        Returns:
            (precision, recall, f1_score)
        """
        try:
            # 提取相关文档ID或内容
            gt_docs = set(ground_truth.get('relevant_docs', []))
            pred_docs = set()
            
            # 从预测结果中提取文档
            if 'results' in prediction:
                for result in prediction['results']:
                    doc_id = result.get('id') or result.get('note_id')
                    if doc_id:
                        pred_docs.add(doc_id)
            
            if not gt_docs and not pred_docs:
                return 1.0, 1.0, 1.0
            
            if not pred_docs:
                return 0.0, 0.0, 0.0
            
            if not gt_docs:
                return 0.0, 1.0, 0.0
            
            # 计算交集
            intersection = gt_docs & pred_docs
            
            precision = len(intersection) / len(pred_docs) if pred_docs else 0.0
            recall = len(intersection) / len(gt_docs) if gt_docs else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return precision, recall, f1_score
            
        except Exception as e:
            logger.error(f"Error calculating traditional metrics: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_path_recall_rate(self, ground_truth: Dict[str, Any], 
                                  prediction: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算路径召回率
        
        Args:
            ground_truth: 标准答案，应包含expected_paths字段
            prediction: 预测结果
            
        Returns:
            (path_recall_rate, analysis)
        """
        try:
            # 提取标准路径
            expected_paths = ground_truth.get('expected_paths', [])
            if not expected_paths:
                return 1.0, {'message': 'No expected paths provided'}
            
            # 提取预测路径
            predicted_paths = []
            if 'results' in prediction:
                for result in prediction['results']:
                    # 从结果中提取路径信息
                    if 'path_score' in result and result['path_score'] > 0:
                        # 提取实体和关系
                        entities = self._extract_entities_from_content(result.get('content', ''))
                        relations = self._extract_relations_from_content(result.get('content', ''))
                        
                        if entities and relations:
                            predicted_paths.append({
                                'entities': entities,
                                'relations': relations,
                                'score': result.get('path_score', 0.0)
                            })
            
            if not predicted_paths:
                return 0.0, {
                    'expected_count': len(expected_paths),
                    'predicted_count': 0,
                    'matched_count': 0
                }
            
            # 计算路径匹配
            matched_paths = 0
            for expected_path in expected_paths:
                for predicted_path in predicted_paths:
                    if self._paths_match(expected_path, predicted_path):
                        matched_paths += 1
                        break
            
            recall_rate = matched_paths / len(expected_paths)
            
            analysis = {
                'expected_count': len(expected_paths),
                'predicted_count': len(predicted_paths),
                'matched_count': matched_paths,
                'recall_rate': recall_rate,
                'expected_paths': expected_paths[:3],  # 只保留前3个用于分析
                'predicted_paths': predicted_paths[:3]
            }
            
            return recall_rate, analysis
            
        except Exception as e:
            logger.error(f"Error calculating path recall rate: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_predicate_consistency_rate(self, ground_truth: Dict[str, Any], 
                                            prediction: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算谓词一致率
        
        Args:
            ground_truth: 标准答案，应包含expected_predicates字段
            prediction: 预测结果
            
        Returns:
            (predicate_consistency_rate, analysis)
        """
        try:
            # 提取标准谓词
            expected_predicates = ground_truth.get('expected_predicates', [])
            if not expected_predicates:
                return 1.0, {'message': 'No expected predicates provided'}
            
            # 标准化期望谓词
            normalized_expected = set()
            for predicate in expected_predicates:
                normalized_pred, _ = self.predicate_normalizer.normalize(predicate)
                normalized_expected.add(normalized_pred)
            
            # 提取预测谓词
            predicted_predicates = set()
            if 'results' in prediction:
                for result in prediction['results']:
                    content = result.get('content', '')
                    relations = self._extract_relations_from_content(content)
                    for relation in relations:
                        normalized_pred, _ = self.predicate_normalizer.normalize(relation)
                        predicted_predicates.add(normalized_pred)
            
            if not predicted_predicates:
                return 0.0, {
                    'expected_count': len(normalized_expected),
                    'predicted_count': 0,
                    'matched_count': 0
                }
            
            # 计算一致性
            matched_predicates = normalized_expected & predicted_predicates
            consistency_rate = len(matched_predicates) / len(normalized_expected)
            
            analysis = {
                'expected_count': len(normalized_expected),
                'predicted_count': len(predicted_predicates),
                'matched_count': len(matched_predicates),
                'consistency_rate': consistency_rate,
                'expected_predicates': list(normalized_expected),
                'predicted_predicates': list(predicted_predicates)[:10],  # 限制数量
                'matched_predicates': list(matched_predicates)
            }
            
            return consistency_rate, analysis
            
        except Exception as e:
            logger.error(f"Error calculating predicate consistency rate: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_entity_coverage_rate(self, ground_truth: Dict[str, Any], 
                                      prediction: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算实体覆盖率
        
        Args:
            ground_truth: 标准答案，应包含expected_entities字段
            prediction: 预测结果
            
        Returns:
            (entity_coverage_rate, analysis)
        """
        try:
            # 提取标准实体
            expected_entities = ground_truth.get('expected_entities', [])
            if not expected_entities:
                return 1.0, {'message': 'No expected entities provided'}
            
            # 标准化期望实体
            normalized_expected = set()
            for entity in expected_entities:
                normalized_ent, _ = self.entity_normalizer.normalize(entity)
                normalized_expected.add(normalized_ent)
            
            # 提取预测实体
            predicted_entities = set()
            if 'results' in prediction:
                for result in prediction['results']:
                    content = result.get('content', '')
                    entities = self._extract_entities_from_content(content)
                    for entity in entities:
                        normalized_ent, _ = self.entity_normalizer.normalize(entity)
                        predicted_entities.add(normalized_ent)
            
            if not predicted_entities:
                return 0.0, {
                    'expected_count': len(normalized_expected),
                    'predicted_count': 0,
                    'covered_count': 0
                }
            
            # 计算覆盖率
            covered_entities = normalized_expected & predicted_entities
            coverage_rate = len(covered_entities) / len(normalized_expected)
            
            analysis = {
                'expected_count': len(normalized_expected),
                'predicted_count': len(predicted_entities),
                'covered_count': len(covered_entities),
                'coverage_rate': coverage_rate,
                'expected_entities': list(normalized_expected),
                'predicted_entities': list(predicted_entities)[:10],  # 限制数量
                'covered_entities': list(covered_entities)
            }
            
            return coverage_rate, analysis
            
        except Exception as e:
            logger.error(f"Error calculating entity coverage rate: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_two_hop_metrics(self, ground_truth: Dict[str, Any], 
                                 prediction: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        计算二跳检索相关指标
        
        Args:
            ground_truth: 标准答案，应包含expected_bridge_entities、expected_two_hop_results等字段
            prediction: 预测结果
            
        Returns:
            (metrics_dict, analysis)
        """
        try:
            metrics = {
                'efficiency': 0.0,
                'bridge_quality': 0.0,
                'pool_coverage': 0.0
            }
            
            # 1. 计算二跳检索效率
            # 基于候选池大小与最终结果质量的比值
            two_hop_stats = prediction.get('two_hop_stats', {})
            candidate_pool_size = two_hop_stats.get('candidate_pool_size', 0)
            final_results_count = len(prediction.get('results', []))
            
            if candidate_pool_size > 0 and final_results_count > 0:
                # 效率 = 有效结果数 / 候选池大小，值越高表示检索越精准
                metrics['efficiency'] = min(final_results_count / candidate_pool_size, 1.0)
            
            # 2. 计算桥接实体质量
            expected_bridge_entities = set(ground_truth.get('expected_bridge_entities', []))
            predicted_bridge_entities = set(two_hop_stats.get('bridge_entities', []))
            
            if expected_bridge_entities:
                # 桥接实体质量 = 正确桥接实体数 / 期望桥接实体数
                correct_bridges = expected_bridge_entities & predicted_bridge_entities
                metrics['bridge_quality'] = len(correct_bridges) / len(expected_bridge_entities)
            else:
                # 如果没有期望的桥接实体，基于桥接实体的多样性评估
                if predicted_bridge_entities:
                    metrics['bridge_quality'] = min(len(predicted_bridge_entities) / 5.0, 1.0)  # 假设5个是理想数量
            
            # 3. 计算候选池覆盖率
            expected_results = set(ground_truth.get('relevant_docs', []))
            candidate_pool_docs = set(two_hop_stats.get('candidate_pool_docs', []))
            
            if expected_results:
                # 覆盖率 = 候选池中包含的相关文档数 / 总相关文档数
                covered_docs = expected_results & candidate_pool_docs
                metrics['pool_coverage'] = len(covered_docs) / len(expected_results)
            
            # 详细分析
            analysis = {
                'candidate_pool_size': candidate_pool_size,
                'final_results_count': final_results_count,
                'expected_bridge_count': len(expected_bridge_entities),
                'predicted_bridge_count': len(predicted_bridge_entities),
                'correct_bridge_count': len(expected_bridge_entities & predicted_bridge_entities) if expected_bridge_entities else 0,
                'expected_docs_count': len(expected_results),
                'candidate_pool_docs_count': len(candidate_pool_docs),
                'covered_docs_count': len(expected_results & candidate_pool_docs) if expected_results else 0,
                'fallback_triggered': two_hop_stats.get('fallback_triggered', False),
                'namespace_filtered_count': two_hop_stats.get('namespace_filtered_count', 0)
            }
            
            return metrics, analysis
            
        except Exception as e:
            logger.error(f"Error calculating two-hop metrics: {e}")
            return {'efficiency': 0.0, 'bridge_quality': 0.0, 'pool_coverage': 0.0}, {'error': str(e)}
    
    def _extract_entities_from_content(self, content: str) -> List[str]:
        """从内容中提取实体"""
        try:
            entities = self.ner.extract_entities(content)
            return [entity['text'] for entity in entities]
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_relations_from_content(self, content: str) -> List[str]:
        """从内容中提取关系（简化实现）"""
        try:
            # 简化的关系提取，基于常见的关系词
            relation_patterns = [
                r'(创建|建立|成立|创立)',
                r'(拥有|持有|具有|包含)',
                r'(位于|在|处于|坐落)',
                r'(属于|隶属|归属)',
                r'(导致|引起|造成)',
                r'(影响|作用|改变)'
            ]
            
            relations = []
            for pattern in relation_patterns:
                import re
                matches = re.findall(pattern, content)
                relations.extend(matches)
            
            return relations
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return []
    
    def _paths_match(self, expected_path: Dict[str, Any], predicted_path: Dict[str, Any]) -> bool:
        """判断两个路径是否匹配"""
        try:
            # 简化的路径匹配逻辑
            expected_entities = set(expected_path.get('entities', []))
            predicted_entities = set(predicted_path.get('entities', []))
            
            expected_relations = set(expected_path.get('relations', []))
            predicted_relations = set(predicted_path.get('relations', []))
            
            # 计算实体和关系的重叠度
            entity_overlap = len(expected_entities & predicted_entities) / max(len(expected_entities), 1)
            relation_overlap = len(expected_relations & predicted_relations) / max(len(expected_relations), 1)
            
            # 如果实体和关系重叠度都超过阈值，认为匹配
            return entity_overlap >= 0.5 and relation_overlap >= 0.5
            
        except Exception as e:
            logger.error(f"Error matching paths: {e}")
            return False
    
    def _generate_summary(self, results: List[EvaluationResult]) -> EvaluationSummary:
        """
        生成评测汇总
        
        Args:
            results: 评测结果列表
            
        Returns:
            评测汇总
        """
        if not results:
            return EvaluationSummary(
                total_queries=0,
                avg_precision=0.0,
                avg_recall=0.0,
                avg_f1_score=0.0,
                avg_path_recall_rate=0.0,
                avg_predicate_consistency_rate=0.0,
                avg_entity_coverage_rate=0.0
            )
        
        # 计算平均值
        avg_precision = sum(r.precision for r in results) / len(results)
        avg_recall = sum(r.recall for r in results) / len(results)
        avg_f1_score = sum(r.f1_score for r in results) / len(results)
        avg_path_recall_rate = sum(r.path_recall_rate for r in results) / len(results)
        avg_predicate_consistency_rate = sum(r.predicate_consistency_rate for r in results) / len(results)
        avg_entity_coverage_rate = sum(r.entity_coverage_rate for r in results) / len(results)
        
        # 二跳检索指标平均值
        avg_two_hop_efficiency = sum(r.two_hop_efficiency for r in results) / len(results)
        avg_bridge_entity_quality = sum(r.bridge_entity_quality for r in results) / len(results)
        avg_candidate_pool_coverage = sum(r.candidate_pool_coverage for r in results) / len(results)
        
        # 分布统计
        score_distribution = {
            'precision': [r.precision for r in results],
            'recall': [r.recall for r in results],
            'f1_score': [r.f1_score for r in results],
            'path_recall_rate': [r.path_recall_rate for r in results],
            'predicate_consistency_rate': [r.predicate_consistency_rate for r in results],
            'entity_coverage_rate': [r.entity_coverage_rate for r in results],
            'two_hop_efficiency': [r.two_hop_efficiency for r in results],
            'bridge_entity_quality': [r.bridge_entity_quality for r in results],
            'candidate_pool_coverage': [r.candidate_pool_coverage for r in results]
        }
        
        # 性能统计
        execution_times = [r.execution_time for r in results]
        performance_stats = {
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'total_execution_time': sum(execution_times)
        }
        
        return EvaluationSummary(
            total_queries=len(results),
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1_score=avg_f1_score,
            avg_path_recall_rate=avg_path_recall_rate,
            avg_predicate_consistency_rate=avg_predicate_consistency_rate,
            avg_entity_coverage_rate=avg_entity_coverage_rate,
            avg_two_hop_efficiency=avg_two_hop_efficiency,
            avg_bridge_entity_quality=avg_bridge_entity_quality,
            avg_candidate_pool_coverage=avg_candidate_pool_coverage,
            score_distribution=score_distribution,
            performance_stats=performance_stats,
            detailed_results=results
        )
    
    def save_results(self, summary: EvaluationSummary, output_path: str) -> None:
        """
        保存评测结果
        
        Args:
            summary: 评测汇总
            output_path: 输出路径
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备输出数据
            output_data = {
                'summary': {
                    'total_queries': summary.total_queries,
                    'avg_precision': summary.avg_precision,
                    'avg_recall': summary.avg_recall,
                    'avg_f1_score': summary.avg_f1_score,
                    'avg_path_recall_rate': summary.avg_path_recall_rate,
                    'avg_predicate_consistency_rate': summary.avg_predicate_consistency_rate,
                    'avg_entity_coverage_rate': summary.avg_entity_coverage_rate,
                    'avg_two_hop_efficiency': summary.avg_two_hop_efficiency,
                    'avg_bridge_entity_quality': summary.avg_bridge_entity_quality,
                    'avg_candidate_pool_coverage': summary.avg_candidate_pool_coverage,
                    'performance_stats': summary.performance_stats
                },
                'score_distribution': summary.score_distribution,
                'detailed_results': []
            }
            
            # 添加详细结果（限制数量以避免文件过大）
            for result in summary.detailed_results[:100]:  # 只保存前100个
                output_data['detailed_results'].append({
                    'query_id': result.query_id,
                    'query': result.query,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'path_recall_rate': result.path_recall_rate,
                    'predicate_consistency_rate': result.predicate_consistency_rate,
                    'entity_coverage_rate': result.entity_coverage_rate,
                    'two_hop_efficiency': result.two_hop_efficiency,
                    'bridge_entity_quality': result.bridge_entity_quality,
                    'candidate_pool_coverage': result.candidate_pool_coverage,
                    'execution_time': result.execution_time,
                    'path_analysis': result.path_analysis,
                    'predicate_analysis': result.predicate_analysis,
                    'entity_analysis': result.entity_analysis,
                    'two_hop_analysis': result.two_hop_analysis
                })
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_summary(self, summary: EvaluationSummary) -> None:
        """
        打印评测汇总
        
        Args:
            summary: 评测汇总
        """
        print("\n" + "="*60)
        print("增强评测结果汇总")
        print("="*60)
        
        print(f"总查询数: {summary.total_queries}")
        print(f"平均执行时间: {summary.performance_stats.get('avg_execution_time', 0):.3f}s")
        
        print("\n传统指标:")
        print(f"  精确率 (Precision): {summary.avg_precision:.3f}")
        print(f"  召回率 (Recall): {summary.avg_recall:.3f}")
        print(f"  F1分数: {summary.avg_f1_score:.3f}")
        
        print("\n新增指标:")
        print(f"  路径召回率: {summary.avg_path_recall_rate:.3f}")
        print(f"  谓词一致率: {summary.avg_predicate_consistency_rate:.3f}")
        print(f"  实体覆盖率: {summary.avg_entity_coverage_rate:.3f}")
        
        print("\n二跳检索指标:")
        print(f"  二跳检索效率: {summary.avg_two_hop_efficiency:.3f}")
        print(f"  桥接实体质量: {summary.avg_bridge_entity_quality:.3f}")
        print(f"  候选池覆盖率: {summary.avg_candidate_pool_coverage:.3f}")
        
        print("\n性能统计:")
        perf_stats = summary.performance_stats
        print(f"  最小执行时间: {perf_stats.get('min_execution_time', 0):.3f}s")
        print(f"  最大执行时间: {perf_stats.get('max_execution_time', 0):.3f}s")
        print(f"  总执行时间: {perf_stats.get('total_execution_time', 0):.3f}s")
        
        print("="*60)

# 便捷函数
def create_enhanced_evaluator(atomic_notes: List[Dict], embeddings=None, llm: Optional[LocalLLM] = None) -> EnhancedEvaluator:
    """创建增强评测器"""
    return EnhancedEvaluator(atomic_notes, embeddings, llm)

def evaluate_with_enhanced_metrics(test_data: List[Dict[str, Any]], 
                                 atomic_notes: List[Dict], 
                                 embeddings=None, 
                                 llm: Optional[LocalLLM] = None,
                                 output_path: Optional[str] = None) -> EvaluationSummary:
    """
    使用增强指标进行评测
    
    Args:
        test_data: 测试数据
        atomic_notes: 原子笔记
        embeddings: 嵌入向量
        llm: LLM实例
        output_path: 输出路径
        
    Returns:
        评测汇总
    """
    evaluator = create_enhanced_evaluator(atomic_notes, embeddings, llm)
    summary = evaluator.evaluate_batch(test_data)
    
    # 打印结果
    evaluator.print_summary(summary)
    
    # 保存结果
    if output_path:
        evaluator.save_results(summary, output_path)
    
    return summary