"""Enhanced multi-hop reasoning path planner based on atomic note features."""

from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from collections import defaultdict, deque
from loguru import logger
import networkx as nx
from datetime import datetime
import dateutil.parser as date_parser
import re

class EnhancedPathPlanner:
    """增强的多跳推理路径规划器，基于原子笔记特征优化路径选择"""
    
    def __init__(self, graph_index, atomic_notes: List[Dict[str, Any]]):
        self.graph_index = graph_index
        self.graph = graph_index.graph
        self.atomic_notes = atomic_notes
        
        # 构建笔记ID到笔记的映射
        self.note_id_to_note = {note.get('note_id', ''): note for note in atomic_notes}
        
        # 时间序列权重
        self.temporal_weights = {
            'chronological': 1.0,  # 时间顺序一致
            'reverse_chronological': 0.8,  # 时间顺序相反
            'contemporary': 0.9,  # 同时期
            'temporal_gap': 0.6,  # 时间间隔较大
            'no_temporal': 0.5   # 无时间信息
        }
        
        # 实体关系权重
        self.entity_relation_weights = {
            'direct_mention': 1.0,  # 直接提及
            'co_occurrence': 0.8,   # 共现
            'hierarchical': 0.9,    # 层次关系
            'causal': 0.95,         # 因果关系
            'temporal_sequence': 0.85,  # 时间序列
            'semantic_similarity': 0.7   # 语义相似
        }
        
        # 路径质量阈值
        self.min_path_quality = 0.4
        self.max_path_length = 4
        self.max_paths_per_query = 15
        
        logger.info("Enhanced path planner initialized with atomic note features")
    
    def plan_reasoning_paths(self, query_entities: List[str], 
                           query_temporal: List[str] = None,
                           initial_candidates: List[str] = None) -> List[Dict[str, Any]]:
        """基于原子笔记特征规划推理路径"""
        
        if not initial_candidates:
            initial_candidates = self._find_entity_candidates(query_entities)
        
        if not initial_candidates:
            logger.warning("No initial candidates found for path planning")
            return []
        
        # 发现所有可能的路径
        all_paths = self._discover_enhanced_paths(initial_candidates, query_entities, query_temporal)
        
        # 基于原子笔记特征评分路径
        scored_paths = self._score_paths_with_atomic_features(all_paths, query_entities, query_temporal)
        
        # 选择多样化的高质量路径
        selected_paths = self._select_diverse_quality_paths(scored_paths)
        
        logger.info(f"Planned {len(selected_paths)} enhanced reasoning paths")
        return selected_paths
    
    def _discover_enhanced_paths(self, initial_candidates: List[str], 
                               query_entities: List[str],
                               query_temporal: List[str] = None) -> List[List[str]]:
        """发现增强的推理路径"""
        all_paths = []
        
        for start_node in initial_candidates:
            if start_node not in self.graph:
                continue
            
            # 使用广度优先搜索发现路径，考虑原子笔记特征
            paths = self._bfs_with_atomic_features(start_node, query_entities, query_temporal)
            all_paths.extend(paths)
        
        # 去重
        unique_paths = self._deduplicate_paths(all_paths)
        return unique_paths
    
    def _bfs_with_atomic_features(self, start_node: str, 
                                query_entities: List[str],
                                query_temporal: List[str] = None) -> List[List[str]]:
        """基于原子笔记特征的广度优先搜索"""
        paths = []
        queue = deque([(start_node, [start_node])])
        visited = set()
        
        while queue and len(paths) < self.max_paths_per_query:
            current_node, current_path = queue.popleft()
            
            if len(current_path) > self.max_path_length:
                continue
            
            # 添加当前路径
            if len(current_path) >= 2:
                paths.append(current_path.copy())
            
            # 探索邻居节点
            if current_node in self.graph:
                neighbors = list(self.graph.neighbors(current_node))
                
                # 基于原子笔记特征排序邻居
                scored_neighbors = self._score_neighbors_by_atomic_features(
                    current_node, neighbors, query_entities, query_temporal, current_path
                )
                
                # 选择最佳邻居继续探索
                for neighbor, score in scored_neighbors[:5]:  # 限制分支数
                    if neighbor not in current_path:  # 避免循环
                        new_path = current_path + [neighbor]
                        queue.append((neighbor, new_path))
        
        return paths
    
    def _score_neighbors_by_atomic_features(self, current_node: str, 
                                          neighbors: List[str],
                                          query_entities: List[str],
                                          query_temporal: List[str],
                                          current_path: List[str]) -> List[Tuple[str, float]]:
        """基于原子笔记特征为邻居节点评分"""
        scored_neighbors = []
        
        current_note = self.note_id_to_note.get(current_node)
        if not current_note:
            return [(neighbor, 0.5) for neighbor in neighbors]
        
        for neighbor in neighbors:
            neighbor_note = self.note_id_to_note.get(neighbor)
            if not neighbor_note:
                scored_neighbors.append((neighbor, 0.3))
                continue
            
            # 计算综合分数
            entity_score = self._calculate_entity_relation_score(current_note, neighbor_note, query_entities)
            temporal_score = self._calculate_temporal_coherence_score(current_note, neighbor_note, query_temporal)
            path_coherence_score = self._calculate_path_coherence_score(current_path + [neighbor])
            semantic_score = self._calculate_semantic_relevance_score(neighbor_note, query_entities)
            
            # 加权组合
            total_score = (
                entity_score * 0.3 +
                temporal_score * 0.25 +
                path_coherence_score * 0.25 +
                semantic_score * 0.2
            )
            
            scored_neighbors.append((neighbor, total_score))
        
        # 按分数排序
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        return scored_neighbors
    
    def _calculate_entity_relation_score(self, note1: Dict[str, Any], 
                                       note2: Dict[str, Any],
                                       query_entities: List[str]) -> float:
        """计算实体关系分数"""
        entities1 = set(note1.get('entities', []) + note1.get('normalized_entities', []))
        entities2 = set(note2.get('entities', []) + note2.get('normalized_entities', []))
        query_entities_set = set([e.lower() for e in query_entities])
        
        # 1. 直接实体重叠
        entity_overlap = len(entities1 & entities2)
        overlap_score = min(entity_overlap / max(len(entities1), len(entities2), 1), 1.0)
        
        # 2. 查询实体相关性
        query_relevance1 = len(entities1 & query_entities_set) / max(len(query_entities_set), 1)
        query_relevance2 = len(entities2 & query_entities_set) / max(len(query_entities_set), 1)
        query_score = (query_relevance1 + query_relevance2) / 2
        
        # 3. 关系类型权重
        relations1 = note1.get('extracted_relations', [])
        relations2 = note2.get('extracted_relations', [])
        
        relation_score = 0.0
        if relations1 and relations2:
            # 检查是否有共同的关系模式
            relation_types1 = set([r.get('relation_type', '') for r in relations1])
            relation_types2 = set([r.get('relation_type', '') for r in relations2])
            common_relations = relation_types1 & relation_types2
            
            if common_relations:
                relation_score = 0.3  # 有共同关系类型
        
        return (overlap_score * 0.4 + query_score * 0.4 + relation_score * 0.2)
    
    def _calculate_temporal_coherence_score(self, note1: Dict[str, Any], 
                                          note2: Dict[str, Any],
                                          query_temporal: List[str] = None) -> float:
        """计算时间连贯性分数"""
        timestamp1 = note1.get('timestamp', '')
        timestamp2 = note2.get('timestamp', '')
        
        # 如果没有时间信息，返回中性分数
        if not timestamp1 or not timestamp2:
            return self.temporal_weights['no_temporal']
        
        try:
            # 解析时间戳
            time1 = self._parse_timestamp(timestamp1)
            time2 = self._parse_timestamp(timestamp2)
            
            if not time1 or not time2:
                return self.temporal_weights['no_temporal']
            
            # 计算时间差
            time_diff = abs((time1 - time2).total_seconds())
            
            # 根据时间差分类
            if time_diff < 86400:  # 1天内
                return self.temporal_weights['contemporary']
            elif time_diff < 2592000:  # 1个月内
                return self.temporal_weights['chronological']
            elif time_diff < 31536000:  # 1年内
                return self.temporal_weights['temporal_gap']
            else:
                return self.temporal_weights['temporal_gap'] * 0.8
        
        except Exception as e:
            logger.debug(f"Error parsing timestamps: {e}")
            return self.temporal_weights['no_temporal']
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """解析时间戳字符串"""
        if not timestamp_str:
            return None
        
        try:
            # 尝试ISO格式
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            try:
                # 尝试dateutil解析
                return date_parser.parse(timestamp_str)
            except:
                return None
    
    def _calculate_path_coherence_score(self, path: List[str]) -> float:
        """计算路径连贯性分数"""
        if len(path) < 2:
            return 1.0
        
        coherence_scores = []
        
        for i in range(len(path) - 1):
            note1 = self.note_id_to_note.get(path[i])
            note2 = self.note_id_to_note.get(path[i + 1])
            
            if not note1 or not note2:
                coherence_scores.append(0.5)
                continue
            
            # 计算相邻笔记的连贯性
            entity_coherence = self._calculate_entity_coherence(note1, note2)
            topic_coherence = self._calculate_topic_coherence(note1, note2)
            
            coherence = (entity_coherence + topic_coherence) / 2
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_entity_coherence(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> float:
        """计算实体连贯性"""
        entities1 = set(note1.get('entities', []) + note1.get('normalized_entities', []))
        entities2 = set(note2.get('entities', []) + note2.get('normalized_entities', []))
        
        if not entities1 or not entities2:
            return 0.3
        
        overlap = len(entities1 & entities2)
        union = len(entities1 | entities2)
        
        return overlap / union if union > 0 else 0.0
    
    def _calculate_topic_coherence(self, note1: Dict[str, Any], note2: Dict[str, Any]) -> float:
        """计算主题连贯性"""
        topics1 = set(note1.get('topics', []))
        topics2 = set(note2.get('topics', []))
        
        if not topics1 or not topics2:
            return 0.3
        
        overlap = len(topics1 & topics2)
        union = len(topics1 | topics2)
        
        return overlap / union if union > 0 else 0.0
    
    def _calculate_semantic_relevance_score(self, note: Dict[str, Any], 
                                          query_entities: List[str]) -> float:
        """计算语义相关性分数"""
        content = note.get('content', '').lower()
        title = note.get('title', '').lower()
        full_text = f"{title} {content}"
        
        # 计算查询实体在笔记中的覆盖度
        entity_matches = 0
        for entity in query_entities:
            if entity.lower() in full_text:
                entity_matches += 1
        
        entity_coverage = entity_matches / max(len(query_entities), 1)
        
        # 考虑重要性分数
        importance = note.get('importance_score', 0.5)
        
        return (entity_coverage * 0.7 + importance * 0.3)
    
    def _score_paths_with_atomic_features(self, paths: List[List[str]], 
                                        query_entities: List[str],
                                        query_temporal: List[str] = None) -> List[Dict[str, Any]]:
        """基于原子笔记特征为路径评分"""
        scored_paths = []
        
        for path in paths:
            if len(path) < 2:
                continue
            
            # 计算路径的各项分数
            entity_score = self._calculate_path_entity_score(path, query_entities)
            temporal_score = self._calculate_path_temporal_score(path, query_temporal)
            coherence_score = self._calculate_path_coherence_score(path)
            diversity_score = self._calculate_path_diversity_score(path)
            reasoning_value = self._calculate_path_reasoning_value(path)
            
            # 加权组合总分
            total_score = (
                entity_score * 0.25 +
                temporal_score * 0.2 +
                coherence_score * 0.25 +
                diversity_score * 0.15 +
                reasoning_value * 0.15
            )
            
            if total_score >= self.min_path_quality:
                scored_paths.append({
                    'path': path,
                    'score': total_score,
                    'entity_score': entity_score,
                    'temporal_score': temporal_score,
                    'coherence_score': coherence_score,
                    'diversity_score': diversity_score,
                    'reasoning_value': reasoning_value,
                    'length': len(path)
                })
        
        # 按分数排序
        scored_paths.sort(key=lambda x: x['score'], reverse=True)
        return scored_paths
    
    def _calculate_path_entity_score(self, path: List[str], query_entities: List[str]) -> float:
        """计算路径实体分数"""
        path_entities = set()
        query_entities_set = set([e.lower() for e in query_entities])
        
        for node_id in path:
            note = self.note_id_to_note.get(node_id)
            if note:
                entities = note.get('entities', []) + note.get('normalized_entities', [])
                path_entities.update([e.lower() for e in entities])
        
        # 计算查询实体覆盖度
        covered_entities = path_entities & query_entities_set
        coverage = len(covered_entities) / max(len(query_entities_set), 1)
        
        return min(coverage, 1.0)
    
    def _calculate_path_temporal_score(self, path: List[str], query_temporal: List[str] = None) -> float:
        """计算路径时间分数"""
        if not query_temporal:
            return 0.5
        
        timestamps = []
        for node_id in path:
            note = self.note_id_to_note.get(node_id)
            if note and note.get('timestamp'):
                timestamp = self._parse_timestamp(note['timestamp'])
                if timestamp:
                    timestamps.append(timestamp)
        
        if len(timestamps) < 2:
            return 0.5
        
        # 检查时间顺序
        is_chronological = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        
        if is_chronological:
            return self.temporal_weights['chronological']
        else:
            return self.temporal_weights['reverse_chronological']
    
    def _calculate_path_diversity_score(self, path: List[str]) -> float:
        """计算路径多样性分数"""
        if len(path) < 2:
            return 1.0
        
        # 计算笔记类型多样性
        note_types = set()
        topics = set()
        
        for node_id in path:
            note = self.note_id_to_note.get(node_id)
            if note:
                note_types.add(note.get('note_type', 'fact'))
                topics.update(note.get('topics', []))
        
        type_diversity = len(note_types) / max(len(path), 1)
        topic_diversity = min(len(topics) / max(len(path), 1), 1.0)
        
        return (type_diversity + topic_diversity) / 2
    
    def _calculate_path_reasoning_value(self, path: List[str]) -> float:
        """计算路径推理价值"""
        reasoning_indicators = 0
        total_notes = len(path)
        
        for node_id in path:
            note = self.note_id_to_note.get(node_id)
            if note:
                # 检查是否包含推理指示词
                content = note.get('content', '').lower()
                
                reasoning_keywords = ['because', 'therefore', 'thus', 'hence', 'consequently', 
                                    'as a result', 'due to', 'caused by', 'leads to']
                
                if any(keyword in content for keyword in reasoning_keywords):
                    reasoning_indicators += 1
                
                # 检查关系类型
                relations = note.get('extracted_relations', [])
                causal_relations = [r for r in relations if r.get('relation_type') == 'causal']
                if causal_relations:
                    reasoning_indicators += 0.5
        
        return min(reasoning_indicators / max(total_notes, 1), 1.0)
    
    def _select_diverse_quality_paths(self, scored_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """选择多样化的高质量路径"""
        if not scored_paths:
            return []
        
        selected_paths = []
        used_nodes = set()
        
        for path_info in scored_paths:
            path = path_info['path']
            
            # 检查路径多样性
            path_nodes = set(path)
            overlap_ratio = len(path_nodes & used_nodes) / len(path_nodes)
            
            # 如果重叠度太高，跳过
            if overlap_ratio > 0.7 and len(selected_paths) > 0:
                continue
            
            selected_paths.append(path_info)
            used_nodes.update(path_nodes)
            
            # 限制路径数量
            if len(selected_paths) >= self.max_paths_per_query:
                break
        
        return selected_paths
    
    def _find_entity_candidates(self, query_entities: List[str]) -> List[str]:
        """查找实体候选节点"""
        candidates = set()
        
        for entity in query_entities:
            entity_lower = entity.lower()
            
            for note_id, note in self.note_id_to_note.items():
                note_entities = [e.lower() for e in note.get('entities', []) + note.get('normalized_entities', [])]
                
                if entity_lower in note_entities:
                    candidates.add(note_id)
                
                # 也检查内容中的提及
                content = note.get('content', '').lower()
                if entity_lower in content:
                    candidates.add(note_id)
        
        return list(candidates)
    
    def _deduplicate_paths(self, paths: List[List[str]]) -> List[List[str]]:
        """去重路径"""
        unique_paths = []
        seen_paths = set()
        
        for path in paths:
            path_tuple = tuple(path)
            if path_tuple not in seen_paths:
                unique_paths.append(path)
                seen_paths.add(path_tuple)
        
        return unique_paths