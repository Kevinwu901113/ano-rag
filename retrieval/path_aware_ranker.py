#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径感知重排序模块

实现功能：
1. 对检索候选中出现的实体构建轻量邻接图
2. 从关键实体出发进行k-hop扩展
3. 对满足谓词链条的候选赋予路径分
4. 路径分与语义分、稀疏分进行融合
5. 支持任意谓词标签和配置化映射
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import math

# 导入增强日志功能
from utils.logging_utils import StructuredLogger, log_performance, log_path_aware_metrics

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体节点"""
    name: str                           # 实体名称
    normalized_name: str = ""           # 标准化名称
    entity_type: str = "general"        # 实体类型
    aliases: List[str] = field(default_factory=list)  # 别名列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def __hash__(self):
        return hash(self.normalized_name or self.name)
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (self.normalized_name or self.name) == (other.normalized_name or other.name)

@dataclass
class Relation:
    """关系边"""
    predicate: str                      # 谓词
    normalized_predicate: str = ""      # 标准化谓词
    source: Entity = None               # 源实体
    target: Entity = None               # 目标实体
    confidence: float = 1.0             # 置信度
    evidence: str = ""                  # 证据文本
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def __hash__(self):
        return hash((self.source, self.target, self.normalized_predicate or self.predicate))

@dataclass
class Path:
    """路径"""
    entities: List[Entity] = field(default_factory=list)     # 路径上的实体
    relations: List[Relation] = field(default_factory=list)  # 路径上的关系
    length: int = 0                     # 路径长度
    score: float = 0.0                  # 路径得分
    evidence_texts: List[str] = field(default_factory=list)  # 证据文本
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'entities': [e.name for e in self.entities],
            'relations': [r.predicate for r in self.relations],
            'length': self.length,
            'score': self.score,
            'evidence_texts': self.evidence_texts
        }

class LightweightGraph:
    """轻量级邻接图"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}           # 实体索引
        self.adjacency: Dict[str, List[Relation]] = defaultdict(list)  # 邻接表
        self.reverse_adjacency: Dict[str, List[Relation]] = defaultdict(list)  # 反向邻接表
        self.entity_count = 0
        self.relation_count = 0
    
    def add_entity(self, entity: Entity) -> None:
        """添加实体"""
        key = entity.normalized_name or entity.name
        if key not in self.entities:
            self.entities[key] = entity
            self.entity_count += 1
        else:
            # 合并别名
            existing = self.entities[key]
            existing.aliases.extend(entity.aliases)
            existing.aliases = list(set(existing.aliases))
    
    def add_relation(self, relation: Relation) -> None:
        """添加关系"""
        if relation.source and relation.target:
            source_key = relation.source.normalized_name or relation.source.name
            target_key = relation.target.normalized_name or relation.target.name
            
            # 添加实体
            self.add_entity(relation.source)
            self.add_entity(relation.target)
            
            # 添加关系
            self.adjacency[source_key].append(relation)
            self.reverse_adjacency[target_key].append(relation)
            self.relation_count += 1
    
    def get_neighbors(self, entity_name: str, direction: str = "out") -> List[Tuple[Entity, Relation]]:
        """
        获取邻居实体
        
        Args:
            entity_name: 实体名称
            direction: 方向 ("out", "in", "both")
            
        Returns:
            邻居实体和关系的列表
        """
        neighbors = []
        
        if direction in ["out", "both"]:
            for relation in self.adjacency.get(entity_name, []):
                if relation.target:
                    neighbors.append((relation.target, relation))
        
        if direction in ["in", "both"]:
            for relation in self.reverse_adjacency.get(entity_name, []):
                if relation.source:
                    neighbors.append((relation.source, relation))
        
        return neighbors
    
    def find_paths(self, start_entity: str, end_entity: str, max_hops: int = 2) -> List[Path]:
        """
        查找两个实体之间的路径
        
        Args:
            start_entity: 起始实体
            end_entity: 目标实体
            max_hops: 最大跳数
            
        Returns:
            路径列表
        """
        if start_entity not in self.entities or end_entity not in self.entities:
            return []
        
        paths = []
        queue = deque([(start_entity, [self.entities[start_entity]], [], 0)])
        visited = set()
        
        while queue:
            current_entity, path_entities, path_relations, hops = queue.popleft()
            
            if hops > max_hops:
                continue
            
            if current_entity == end_entity and hops > 0:
                # 找到路径
                path = Path(
                    entities=path_entities.copy(),
                    relations=path_relations.copy(),
                    length=hops,
                    score=self._calculate_path_score(path_relations)
                )
                paths.append(path)
                continue
            
            # 避免循环
            state = (current_entity, tuple(e.name for e in path_entities))
            if state in visited:
                continue
            visited.add(state)
            
            # 扩展邻居
            for neighbor_entity, relation in self.get_neighbors(current_entity, "out"):
                neighbor_key = neighbor_entity.normalized_name or neighbor_entity.name
                if neighbor_key not in [e.name for e in path_entities]:  # 避免循环
                    new_path_entities = path_entities + [neighbor_entity]
                    new_path_relations = path_relations + [relation]
                    queue.append((neighbor_key, new_path_entities, new_path_relations, hops + 1))
        
        # 按分数排序
        paths.sort(key=lambda p: p.score, reverse=True)
        return paths
    
    def k_hop_expansion(self, start_entities: List[str], k: int = 2) -> Set[str]:
        """
        k-hop扩展
        
        Args:
            start_entities: 起始实体列表
            k: 扩展跳数
            
        Returns:
            扩展后的实体集合
        """
        expanded_entities = set(start_entities)
        current_entities = set(start_entities)
        
        for hop in range(k):
            next_entities = set()
            
            for entity in current_entities:
                if entity in self.entities:
                    neighbors = self.get_neighbors(entity, "both")
                    for neighbor_entity, _ in neighbors:
                        neighbor_key = neighbor_entity.normalized_name or neighbor_entity.name
                        next_entities.add(neighbor_key)
            
            expanded_entities.update(next_entities)
            current_entities = next_entities
            
            if not current_entities:
                break
        
        return expanded_entities
    
    def _calculate_path_score(self, relations: List[Relation]) -> float:
        """计算路径得分"""
        if not relations:
            return 0.0
        
        # 基于关系置信度和路径长度计算得分
        confidence_product = 1.0
        for relation in relations:
            confidence_product *= relation.confidence
        
        # 路径长度惩罚
        length_penalty = 1.0 / (1.0 + len(relations))
        
        return confidence_product * length_penalty
    
    def get_stats(self) -> Dict[str, Any]:
        """获取图统计信息"""
        return {
            'entity_count': self.entity_count,
            'relation_count': self.relation_count,
            'avg_degree': self.relation_count / max(self.entity_count, 1) * 2
        }

class GraphExtractor:
    """图提取器 - 从文本中提取实体和关系"""
    
    def __init__(self, config: Dict[str, Any]):
        from utils.config_loader_helper import load_config_with_fallback
        
        # 尝试从外部配置文件加载，如果失败则使用内联配置
        config_file_path = config.get('path_aware_ranker', {}).get('graph_extractor_config_file')
        
        # 默认配置
        default_config = {
            'entity_patterns': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 英文实体
                r'[\u4e00-\u9fff]+(?:公司|集团|大学|学院|医院|银行)',  # 中文机构
                r'[\u4e00-\u9fff]{2,4}(?:市|省|县|区|国)',  # 中文地名
            ],
            'predicate_mapping': {
                '创立': 'founded_by',
                '成立': 'founded_by', 
                '建立': 'founded_by',
                '发行': 'distributed_by',
                '分发': 'distributed_by',
                '位于': 'located_in',
                '在': 'located_in',
                '属于': 'member_of',
                '隶属': 'member_of',
                '拥有': 'owns',
                '持有': 'owns'
            },
            'relation_patterns': [
                r'(.+?)(?:由|被)(.+?)(?:创立|成立|建立)',
                r'(.+?)(?:位于|在)(.+?)(?:市|省|县|区|国)',
                r'(.+?)(?:属于|隶属于)(.+?)(?:公司|集团|组织)',
                r'(.+?)(?:拥有|持有)(.+?)(?:股份|股权)'
            ]
        }
        
        # 加载配置（外部文件优先，回退到内联配置，最后使用默认配置）
        inline_config = config.get('graph_extractor', {})
        self.config = load_config_with_fallback(config_file_path, inline_config, default_config)
        
        # 实体提取模式
        self.entity_patterns = self.config.get('entity_patterns', default_config['entity_patterns'])
        
        # 谓词映射
        self.predicate_mapping = self.config.get('predicate_mapping', default_config['predicate_mapping'])
        
        # 关系提取模式
        self.relation_patterns = self.config.get('relation_patterns', default_config['relation_patterns'])
        
        logger.info("GraphExtractor initialized")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        seen_entities = set()
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entity_name = match.strip()
                if entity_name and entity_name not in seen_entities:
                    entity = Entity(
                        name=entity_name,
                        normalized_name=self._normalize_entity(entity_name)
                    )
                    entities.append(entity)
                    seen_entities.add(entity_name)
        
        return entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        从文本中提取关系
        
        Args:
            text: 输入文本
            entities: 已提取的实体列表
            
        Returns:
            关系列表
        """
        relations = []
        entity_dict = {e.name: e for e in entities}
        
        for pattern in self.relation_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:
                    source_name, target_name = match[0].strip(), match[1].strip()
                    
                    # 查找对应的实体
                    source_entity = self._find_entity(source_name, entity_dict)
                    target_entity = self._find_entity(target_name, entity_dict)
                    
                    if source_entity and target_entity:
                        # 推断谓词
                        predicate = self._infer_predicate(text, source_name, target_name)
                        
                        relation = Relation(
                            predicate=predicate,
                            normalized_predicate=self.predicate_mapping.get(predicate, predicate),
                            source=source_entity,
                            target=target_entity,
                            evidence=text,
                            confidence=0.8  # 默认置信度
                        )
                        relations.append(relation)
        
        return relations
    
    def _normalize_entity(self, entity_name: str) -> str:
        """标准化实体名称"""
        # 简单的标准化：去除空格，转换为小写
        return entity_name.strip().lower()
    
    def _find_entity(self, name: str, entity_dict: Dict[str, Entity]) -> Optional[Entity]:
        """查找实体"""
        # 精确匹配
        if name in entity_dict:
            return entity_dict[name]
        
        # 模糊匹配
        for entity_name, entity in entity_dict.items():
            if name in entity_name or entity_name in name:
                return entity
        
        return None
    
    def _infer_predicate(self, text: str, source: str, target: str) -> str:
        """推断谓词"""
        # 简单的谓词推断
        for predicate, normalized in self.predicate_mapping.items():
            if predicate in text:
                return predicate
        
        return "related_to"  # 默认关系

class PathAwareRanker:
    """路径感知重排序器"""
    
    def __init__(self, config: Dict[str, Any]):
        from utils.config_loader_helper import load_config_with_fallback
        
        # 尝试从外部配置文件加载，如果失败则使用内联配置
        config_file_path = config.get('path_aware_ranker_config_file')
        
        # 默认配置
        default_config = {
            'k_hop': 2,
            'path_weight': 0.3,
            'semantic_weight': 0.5,
            'sparse_weight': 0.2,
            'max_paths_per_candidate': 5,
            'min_path_score': 0.1,
            'graph_extractor_config_file': None
        }
        
        # 加载配置（外部文件优先，回退到内联配置，最后使用默认配置）
        inline_config = config.get('path_aware_ranker', {})
        self.config = load_config_with_fallback(config_file_path, inline_config, default_config)
        
        # 初始化组件
        # 为GraphExtractor传递完整的配置，包括path_aware_ranker部分
        graph_extractor_config = config.copy()
        graph_extractor_config['path_aware_ranker'] = self.config
        self.graph_extractor = GraphExtractor(graph_extractor_config)
        self.graph = LightweightGraph()
        
        # 配置参数
        self.k_hop = self.config.get('k_hop', default_config['k_hop'])
        self.path_weight = self.config.get('path_weight', default_config['path_weight'])
        self.semantic_weight = self.config.get('semantic_weight', default_config['semantic_weight'])
        self.sparse_weight = self.config.get('sparse_weight', default_config['sparse_weight'])
        self.max_paths_per_candidate = self.config.get('max_paths_per_candidate', default_config['max_paths_per_candidate'])
        self.min_path_score = self.config.get('min_path_score', default_config['min_path_score'])
        
        # 缓存
        self.entity_cache = {}
        self.path_cache = {}
        
        # 初始化结构化日志记录器
        self.structured_logger = StructuredLogger("PathAwareRanker")
        
        # 记录初始化完成
        self.structured_logger.info("PathAwareRanker initialized successfully",
                                   k_hop=self.k_hop,
                                   path_weight=self.path_weight,
                                   semantic_weight=self.semantic_weight,
                                   sparse_weight=self.sparse_weight,
                                   max_paths_per_candidate=self.max_paths_per_candidate,
                                   min_path_score=self.min_path_score)
        
        logger.info(f"PathAwareRanker initialized: k_hop={self.k_hop}, "
                   f"weights=({self.path_weight}, {self.semantic_weight}, {self.sparse_weight})")
    
    def build_graph_from_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        """
        从候选结果构建图
        
        Args:
            candidates: 候选结果列表
        """
        logger.info(f"Building graph from {len(candidates)} candidates")
        
        # 清空现有图
        self.graph = LightweightGraph()
        
        for candidate in candidates:
            content = candidate.get('content', '')
            
            # 提取实体和关系
            entities = self.graph_extractor.extract_entities(content)
            relations = self.graph_extractor.extract_relations(content, entities)
            
            # 添加到图中
            for entity in entities:
                self.graph.add_entity(entity)
            
            for relation in relations:
                self.graph.add_relation(relation)
        
        stats = self.graph.get_stats()
        logger.info(f"Graph built: {stats['entity_count']} entities, {stats['relation_count']} relations")
    
    def extract_key_entities(self, query: str, candidates: List[Dict[str, Any]]) -> List[str]:
        """
        从查询和候选中提取关键实体
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            
        Returns:
            关键实体列表
        """
        key_entities = set()
        
        # 从查询中提取实体
        query_entities = self.graph_extractor.extract_entities(query)
        for entity in query_entities:
            key_entities.add(entity.normalized_name or entity.name)
        
        # 从高分候选中提取实体
        top_candidates = sorted(candidates, key=lambda x: x.get('similarity', 0.0), reverse=True)[:5]
        for candidate in top_candidates:
            content = candidate.get('content', '')
            entities = self.graph_extractor.extract_entities(content)
            for entity in entities:
                key_entities.add(entity.normalized_name or entity.name)
        
        return list(key_entities)
    
    @log_performance
    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        重排序候选结果
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            
        Returns:
            重排序后的候选结果
        """
        if not candidates:
            return candidates
        
        # 记录重排序开始
        self.structured_logger.debug("Starting path-aware reranking",
                                   candidates_count=len(candidates),
                                   query_length=len(query),
                                   k_hop=self.k_hop,
                                   path_weight=self.path_weight)
        
        logger.info(f"Reranking {len(candidates)} candidates")
        
        # 1. 构建图
        self.build_graph_from_candidates(candidates)
        
        # 2. 提取关键实体
        key_entities = self.extract_key_entities(query, candidates)
        logger.debug(f"Key entities: {key_entities[:5]}...")  # 只显示前5个
        
        # 3. k-hop扩展
        expanded_entities = self.graph.k_hop_expansion(key_entities, self.k_hop)
        logger.debug(f"Expanded to {len(expanded_entities)} entities")
        
        # 4. 计算路径分数
        reranked_candidates = []
        for candidate in candidates:
            path_score = self._calculate_path_score_for_candidate(
                candidate, key_entities, expanded_entities
            )
            
            # 融合分数
            semantic_score = candidate.get('similarity', 0.0)
            sparse_score = candidate.get('sparse_score', 0.0)
            
            final_score = (
                self.semantic_weight * semantic_score +
                self.sparse_weight * sparse_score +
                self.path_weight * path_score
            )
            
            candidate_copy = candidate.copy()
            candidate_copy['path_score'] = path_score
            candidate_copy['final_score'] = final_score
            candidate_copy['score_breakdown'] = {
                'semantic': semantic_score,
                'sparse': sparse_score,
                'path': path_score
            }
            
            reranked_candidates.append(candidate_copy)
        
        # 5. 按最终分数排序
        reranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 计算统计信息
        avg_path_score = sum(c['path_score'] for c in reranked_candidates) / len(reranked_candidates)
        enhanced_count = sum(1 for c in reranked_candidates if c['path_score'] > 0)
        
        # 记录路径感知指标
        log_path_aware_metrics(
            candidates_count=len(candidates),
            enhanced_count=enhanced_count,
            avg_path_score=avg_path_score,
            path_weight=self.path_weight
        )
        
        # 记录重排序完成
        self.structured_logger.info("Path-aware reranking completed",
                                  candidates_count=len(candidates),
                                  enhanced_count=enhanced_count,
                                  avg_path_score=f"{avg_path_score:.3f}",
                                  key_entities_count=len(key_entities),
                                  expanded_entities_count=len(expanded_entities))
        
        logger.info(f"Reranking completed: avg path score = {avg_path_score:.3f}")
        return reranked_candidates
    
    def _calculate_path_score_for_candidate(self, candidate: Dict[str, Any], 
                                          key_entities: List[str], 
                                          expanded_entities: Set[str]) -> float:
        """
        计算候选的路径分数
        
        Args:
            candidate: 候选结果
            key_entities: 关键实体列表
            expanded_entities: 扩展实体集合
            
        Returns:
            路径分数
        """
        content = candidate.get('content', '')
        
        # 提取候选中的实体
        candidate_entities = self.graph_extractor.extract_entities(content)
        candidate_entity_names = [e.normalized_name or e.name for e in candidate_entities]
        
        # 计算实体覆盖度
        key_entity_coverage = len(set(candidate_entity_names) & set(key_entities)) / max(len(key_entities), 1)
        expanded_entity_coverage = len(set(candidate_entity_names) & expanded_entities) / max(len(expanded_entities), 1)
        
        # 查找路径
        path_scores = []
        for key_entity in key_entities:
            for candidate_entity in candidate_entity_names:
                if key_entity != candidate_entity:
                    paths = self.graph.find_paths(key_entity, candidate_entity, self.k_hop)
                    for path in paths[:self.max_paths_per_candidate]:
                        if path.score >= self.min_path_score:
                            path_scores.append(path.score)
        
        # 计算最终路径分数
        if path_scores:
            avg_path_score = sum(path_scores) / len(path_scores)
        else:
            avg_path_score = 0.0
        
        # 综合分数
        final_path_score = (
            0.4 * key_entity_coverage +
            0.3 * expanded_entity_coverage +
            0.3 * avg_path_score
        )
        
        return final_path_score
    
    def get_path_explanations(self, candidate: Dict[str, Any], 
                             key_entities: List[str]) -> List[Dict[str, Any]]:
        """
        获取路径解释
        
        Args:
            candidate: 候选结果
            key_entities: 关键实体列表
            
        Returns:
            路径解释列表
        """
        explanations = []
        content = candidate.get('content', '')
        
        # 提取候选中的实体
        candidate_entities = self.graph_extractor.extract_entities(content)
        candidate_entity_names = [e.normalized_name or e.name for e in candidate_entities]
        
        # 查找路径并生成解释
        for key_entity in key_entities:
            for candidate_entity in candidate_entity_names:
                if key_entity != candidate_entity:
                    paths = self.graph.find_paths(key_entity, candidate_entity, self.k_hop)
                    for path in paths[:3]:  # 最多3条路径
                        if path.score >= self.min_path_score:
                            explanation = {
                                'from_entity': key_entity,
                                'to_entity': candidate_entity,
                                'path': path.to_dict(),
                                'explanation': self._generate_path_explanation(path)
                            }
                            explanations.append(explanation)
        
        return explanations
    
    def _generate_path_explanation(self, path: Path) -> str:
        """生成路径解释文本"""
        if not path.entities or not path.relations:
            return "直接关联"
        
        explanation_parts = []
        for i, relation in enumerate(path.relations):
            if i < len(path.entities) - 1:
                source = path.entities[i].name
                target = path.entities[i + 1].name
                predicate = relation.predicate
                explanation_parts.append(f"{source} {predicate} {target}")
        
        return " -> ".join(explanation_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        graph_stats = self.graph.get_stats()
        return {
            'k_hop': self.k_hop,
            'weights': {
                'path': self.path_weight,
                'semantic': self.semantic_weight,
                'sparse': self.sparse_weight
            },
            'graph_stats': graph_stats,
            'cache_sizes': {
                'entity_cache': len(self.entity_cache),
                'path_cache': len(self.path_cache)
            }
        }

# 便利函数
def create_path_aware_ranker(config: Dict[str, Any]) -> PathAwareRanker:
    """创建路径感知重排序器实例"""
    return PathAwareRanker(config)

def rerank_with_path_awareness(query: str, 
                              candidates: List[Dict[str, Any]],
                              ranker: Optional[PathAwareRanker] = None,
                              **kwargs) -> List[Dict[str, Any]]:
    """
    使用路径感知重排序的便利函数
    
    Args:
        query: 查询字符串
        candidates: 候选结果列表
        ranker: 路径感知重排序器实例
        **kwargs: 其他参数
        
    Returns:
        重排序后的候选结果
    """
    if ranker is None:
        logger.warning("No path aware ranker provided, returning original candidates")
        return candidates
    
    try:
        return ranker.rerank_candidates(query, candidates)
    except Exception as e:
        logger.error(f"Path-aware reranking failed: {e}")
        return candidates