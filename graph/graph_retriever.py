import heapq
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, deque
from loguru import logger
import networkx as nx
from .graph_index import GraphIndex
from .enhanced_path_planner import EnhancedPathPlanner
from .dynamic_threshold_adjuster import DynamicThresholdAdjuster
from config import config

class GraphRetriever:
    """统一的图谱检索器，支持基础k-hop搜索和增强的多跳推理"""
    def __init__(self, graph_index: GraphIndex, k_hop: int = 2, atomic_notes: List[Dict[str, Any]] = None):
        self.index = graph_index
        self.graph = graph_index.graph
        self.k_hop = k_hop
        self.atomic_notes = atomic_notes or []
        
        # 多跳推理配置
        self.multi_hop_config = config.get('multi_hop', {})
        self.max_hops = self.multi_hop_config.get('max_hops', 3)
        self.max_paths = self.multi_hop_config.get('max_paths', 10)
        self.min_path_score = self.multi_hop_config.get('min_path_score', 0.3)
        self.min_path_score_floor = self.multi_hop_config.get('min_path_score_floor', 0.1)
        self.min_path_score_step = self.multi_hop_config.get('min_path_score_step', 0.05)
        self.path_diversity_threshold = self.multi_hop_config.get('path_diversity_threshold', 0.7)
        
        # 推理路径权重
        self.path_weights = {
            'direct': 1.0,
            'two_hop': 0.8,
            'three_hop': 0.6,
            'multi_hop': 0.4
        }
        
        # 关系类型权重（用于路径评分）
        self.relation_weights = {
            'causal': 1.0,
            'definition': 0.95,
            'reference': 0.9,
            'temporal': 0.85,
            'instance_of': 0.8,
            'part_of': 0.8,
            'comparison': 0.75,
            'entity_coexistence': 0.7,
            'support': 0.7,
            'topic_relation': 0.6,
            'context_relation': 0.5,
            'semantic_similarity': 0.4,
            'contradiction': 0.3
        }
        
        # 初始化增强路径规划器
        self.enhanced_path_planner = None
        if self.atomic_notes:
            try:
                self.enhanced_path_planner = EnhancedPathPlanner(graph_index, self.atomic_notes)
                logger.info("Enhanced path planner initialized with atomic note features")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced path planner: {e}")
        
        # 初始化动态阈值调整器
        self.dynamic_threshold_adjuster = None
        if self.atomic_notes:
            try:
                self.dynamic_threshold_adjuster = DynamicThresholdAdjuster(self.atomic_notes)
                logger.info("Dynamic threshold adjuster initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize dynamic threshold adjuster: {e}")
        
        logger.info("GraphRetriever initialized with unified k-hop and multi-hop reasoning support")

    def retrieve(self, seed_note_ids: List[str]) -> List[Dict[str, Any]]:
        G = self.index.graph
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty")
            return []
        results = []
        visited = set()
        for seed in seed_note_ids:
            if seed not in G:
                continue
            try:
                nodes = nx.single_source_dijkstra_path_length(
                    G, seed, cutoff=self.k_hop, weight="weight"
                )
            except Exception as e:
                logger.error(f"Weighted traversal failed: {e}")
                nodes = nx.single_source_shortest_path_length(G, seed, cutoff=self.k_hop)
            for node_id, dist in nodes.items():
                if node_id == seed or node_id in visited:
                    continue
                visited.add(node_id)
                data = G.nodes[node_id].copy()
                data["graph_distance"] = dist
                centrality = self.index.get_centrality(node_id)
                data["centrality"] = centrality
                importance = data.get("importance_score", 1.0)
                data["graph_score"] = (centrality / (dist + 1e-5)) * importance
                data["paragraph_idxs"] = G.nodes[node_id].get('paragraph_idxs', [])
                results.append(data)
        results.sort(key=lambda x: x.get("graph_score", 0), reverse=True)
        logger.info(f"Graph retrieval returned {len(results)} notes")
        return results
    
    def retrieve_with_reasoning_paths(self, query_embedding: np.ndarray, 
                                    top_k: int = 10,
                                    query_keywords: List[str] = None,
                                    query_entities: List[str] = None) -> List[Dict[str, Any]]:
        """基于推理路径的检索"""
        logger.info(f"Starting reasoning path retrieval for top-{top_k} results")
        
        # 使用动态阈值调整器调整参数
        if self.dynamic_threshold_adjuster:
            adjusted_params = self.dynamic_threshold_adjuster.adjust_retrieval_params(
                query_keywords=query_keywords,
                query_entities=query_entities,
                current_top_k=top_k
            )
            top_k = adjusted_params.get('top_k', top_k)
            # 动态调整路径评分阈值
            dynamic_min_path_score = adjusted_params.get('min_path_score', 0.5)
            logger.info(f"Dynamic threshold adjuster updated top_k to {top_k}, min_path_score to {dynamic_min_path_score}")
        else:
            dynamic_min_path_score = 0.5
        
        # 1. 找到初始候选节点
        initial_candidates = self._find_initial_candidates(
            query_embedding, query_keywords, query_entities
        )
        
        if not initial_candidates:
            logger.warning("No initial candidates found")
            return []
        
        # 2. 发现推理路径（优先使用增强路径规划器）
        if self.enhanced_path_planner and query_entities:
            # 提取查询中的时间信息
            query_temporal = self._extract_temporal_from_query(query_keywords or [])
            
            # 使用增强路径规划器
            scored_paths = self.enhanced_path_planner.plan_reasoning_paths(
                query_entities, query_temporal, initial_candidates
            )
            logger.info(f"Enhanced path planner found {len(scored_paths)} paths")
        else:
            # 使用传统方法
            reasoning_paths = self._discover_reasoning_paths(initial_candidates)
            scored_paths = self._score_reasoning_paths(reasoning_paths, dynamic_min_path_score)
            logger.info(f"Traditional path discovery found {len(scored_paths)} paths")
        
        # 3. 检查是否有有效路径，避免在空集上继续处理
        if not scored_paths:
            logger.warning("No valid reasoning paths found after scoring, returning empty results")
            logger.info("Graph paths result: 0, fallback: semantic_only")
            return []
        
        # 4. 选择多样化的路径（如果使用增强路径规划器，路径已经评分）
        if self.enhanced_path_planner and query_entities:
            selected_paths = scored_paths[:self.max_paths]  # 增强规划器已经选择了最佳路径
        else:
            selected_paths = self._select_diverse_paths(scored_paths)
        
        # 5. 从路径中提取最终结果
        final_results = self._extract_results_from_paths(selected_paths)
        
        # 6. 应用top_k限制
        final_results = final_results[:top_k]
        
        logger.info(f"Retrieved {len(final_results)} results from {len(selected_paths)} reasoning paths")
        return final_results
    
    def _find_initial_candidates(self, query_embedding: np.ndarray,
                               query_keywords: List[str] = None,
                               query_entities: List[str] = None) -> List[str]:
        """找到初始候选节点"""
        candidates = set()
        
        # 1. 基于嵌入相似度的候选
        if query_embedding is not None:
            embedding_candidates = self._find_embedding_candidates(query_embedding)
            candidates.update(embedding_candidates)
        
        # 2. 基于关键词的候选
        if query_keywords:
            keyword_candidates = self._find_keyword_candidates(query_keywords)
            candidates.update(keyword_candidates)
        
        # 3. 基于实体的候选
        if query_entities:
            entity_candidates = self._find_entity_candidates(query_entities)
            candidates.update(entity_candidates)
        
        # 限制候选数量
        max_initial_candidates = self.multi_hop_config.get('max_initial_candidates', 20)
        return list(candidates)[:max_initial_candidates]
    
    def _find_embedding_candidates(self, query_embedding: np.ndarray, top_k: int = 15) -> List[str]:
        """基于嵌入相似度找到候选节点"""
        if not hasattr(self.index, 'embeddings') or self.index.embeddings is None:
            return []
        
        # 计算相似度
        similarities = np.dot(self.index.embeddings, query_embedding)
        
        # 获取top-k最相似的节点
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        candidates = []
        for idx in top_indices:
            if idx < len(self.index.note_id_to_index):
                note_id = list(self.index.note_id_to_index.keys())[idx]
                candidates.append(note_id)
        
        return candidates
    
    def _find_keyword_candidates(self, query_keywords: List[str]) -> List[str]:
        """基于关键词找到候选节点"""
        candidates = set()
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            node_keywords = set(node_data.get('keywords', []))
            
            # 计算关键词重叠
            overlap = len(set(query_keywords) & node_keywords)
            if overlap > 0:
                candidates.add(node_id)
        
        return list(candidates)
    
    def _find_entity_candidates(self, query_entities: List[str]) -> List[str]:
        """基于实体找到候选节点"""
        candidates = set()
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            node_entities = set(node_data.get('entities', []))
            
            # 计算实体重叠
            overlap = len(set(query_entities) & node_entities)
            if overlap > 0:
                candidates.add(node_id)
        
        return list(candidates)
    
    def _discover_reasoning_paths(self, initial_candidates: List[str]) -> List[List[str]]:
        """发现推理路径"""
        all_paths = []
        
        for start_node in initial_candidates:
            if start_node not in self.graph:
                continue
            
            # 使用广度优先搜索发现路径
            paths_from_node = self._bfs_reasoning_paths(start_node)
            all_paths.extend(paths_from_node)
        
        # 去重和过滤
        unique_paths = self._deduplicate_paths(all_paths)
        
        logger.info(f"Discovered {len(unique_paths)} unique reasoning paths")
        return unique_paths
    
    def _extract_temporal_from_query(self, query_keywords: List[str]) -> List[str]:
        """从查询关键词中提取时间信息"""
        temporal_keywords = []
        
        # 时间相关的关键词模式
        temporal_patterns = [
            r'\b\d{4}\b',  # 年份
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\b',
            r'\b(today|yesterday|tomorrow|now|recently|currently|before|after|during|since|until)\b',
            r'\b\d+\s+(years?|months?|days?|hours?|minutes?)\s+(ago|later|before|after)\b'
        ]
        
        import re
        query_text = ' '.join(query_keywords).lower()
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            temporal_keywords.extend(matches)
        
        return list(set(temporal_keywords))
    
    def _score_reasoning_paths(self, paths: List[List[str]], dynamic_min_path_score: float = None) -> List[Dict]:
        """为推理路径评分，支持自适应阈值调整"""
        if not paths:
            logger.warning("No paths to score")
            return []
        
        # 首先计算所有路径的分数
        all_scored_paths = []
        for path in paths:
            score = self._calculate_path_score(path)
            all_scored_paths.append({
                'path': path,
                'score': score,
                'length': len(path),
                'relations': self._get_path_relations(path)
            })
        
        # 按分数排序
        all_scored_paths.sort(key=lambda x: x['score'], reverse=True)
        
        # 确定初始阈值
        base_threshold = dynamic_min_path_score if dynamic_min_path_score is not None else self.min_path_score
        
        # 应用初始阈值过滤
        scored_paths = [p for p in all_scored_paths if p['score'] >= base_threshold]
        
        # 自适应阈值调整：当有路径但符合阈值的为0时，逐步降低阈值
        if len(all_scored_paths) >= 1 and len(scored_paths) == 0:
            logger.info(f"Found {len(all_scored_paths)} paths but 0 above threshold {base_threshold:.3f}, applying adaptive threshold")
            
            # 定义阈值降级序列
            threshold_steps = [
                base_threshold * 0.7,  # 第一档：降到70%
                base_threshold * 0.45, # 第二档：降到45%
                base_threshold * 0.30  # 第三档：降到30%
            ]
            
            for step_idx, lower_threshold in enumerate(threshold_steps):
                scored_paths = [p for p in all_scored_paths if p['score'] >= lower_threshold]
                if len(scored_paths) > 0:
                    logger.info(f"Adaptive threshold step {step_idx + 1}: lowered to {lower_threshold:.3f}, found {len(scored_paths)} paths")
                    break
                else:
                    logger.info(f"Adaptive threshold step {step_idx + 1}: threshold {lower_threshold:.3f} still yields 0 paths")
            
            # 如果所有阈值都无效，返回空集并记录日志
            if len(scored_paths) == 0:
                logger.warning(f"Adaptive threshold failed: all {len(all_scored_paths)} paths below minimum threshold {threshold_steps[-1]:.3f}")
                logger.info(f"Graph paths result: 0, fallback: semantic_only")
                return []
        
        logger.info(f"Scored {len(scored_paths)} paths above threshold (from {len(all_scored_paths)} total paths)")
        return scored_paths
    
    def _calculate_path_score(self, path: List[str]) -> float:
        """计算路径分数"""
        if len(path) < 2:
            return 0.0
        
        # 1. 路径长度分数（较短路径得分更高）
        length_score = 1.0 / len(path)
        
        # 2. 关系质量分数
        relation_score = self._calculate_relation_score(path)
        
        # 3. 节点质量分数
        node_score = self._calculate_node_score(path)
        
        # 4. 路径连贯性分数
        coherence_score = self._calculate_path_coherence(path)
        
        # 5. 推理价值分数
        reasoning_score = self._calculate_path_reasoning_value(path)
        
        # 加权组合
        total_score = (
            length_score * 0.15 +
            relation_score * 0.25 +
            node_score * 0.25 +
            coherence_score * 0.20 +
            reasoning_score * 0.15
        )
        
        return total_score
    
    def _calculate_relation_score(self, path: List[str]) -> float:
        """计算关系质量分数"""
        if len(path) < 2:
            return 0.0
        
        total_score = 0.0
        relation_count = 0
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            
            if self.graph.has_edge(node1, node2):
                edge_data = self.graph[node1][node2]
                relation_type = edge_data.get('relation_type', 'unknown')
                
                # 根据关系类型权重评分
                type_weight = self.relation_weights.get(relation_type, 0.5)
                
                # 考虑关系强度
                relation_strength = edge_data.get('weight', 0.5)
                
                relation_score = type_weight * relation_strength
                total_score += relation_score
                relation_count += 1
        
        return total_score / relation_count if relation_count > 0 else 0.0
    
    def _calculate_node_score(self, path: List[str]) -> float:
        """计算节点质量分数"""
        total_score = 0.0
        
        for node_id in path:
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                
                # 1. 相似度分数
                similarity = node_data.get('similarity', 0.0)
                
                # 2. 中心性分数
                centrality = self.index.get_centrality(node_id)
                
                # 3. 重要性分数
                importance = node_data.get('importance', 0.5)
                
                node_score = similarity * 0.4 + centrality * 0.3 + importance * 0.3
                total_score += node_score
        
        return total_score / len(path) if path else 0.0
    
    def _calculate_path_coherence(self, path: List[str]) -> float:
        """计算路径连贯性分数"""
        if len(path) < 2:
            return 0.0
        
        coherence_scores = []
        
        # 1. 三元组连贯性
        for i in range(len(path) - 2):
            triplet = path[i:i+3]
            triplet_coherence = self._calculate_triplet_coherence(triplet)
            coherence_scores.append(triplet_coherence)
        
        # 2. 主题一致性
        topic_consistency = self._calculate_topic_consistency(path)
        coherence_scores.append(topic_consistency)
        
        # 3. 关键词重叠
        keyword_overlap = self._calculate_keyword_overlap(path)
        coherence_scores.append(keyword_overlap)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_triplet_coherence(self, triplet: List[str]) -> float:
        """计算三元组连贯性"""
        if len(triplet) != 3:
            return 0.0
        
        # 检查三个节点之间的关系强度
        coherence = 0.0
        connections = 0
        
        for i in range(len(triplet)):
            for j in range(i + 1, len(triplet)):
                if self.graph.has_edge(triplet[i], triplet[j]):
                    edge_data = self.graph[triplet[i]][triplet[j]]
                    coherence += edge_data.get('weight', 0.5)
                    connections += 1
        
        return coherence / connections if connections > 0 else 0.0
    
    def _calculate_topic_consistency(self, path: List[str]) -> float:
        """计算主题一致性"""
        all_topics = set()
        node_topics = []
        
        for node_id in path:
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                topics = set(node_data.get('topics', []))
                node_topics.append(topics)
                all_topics.update(topics)
        
        if not all_topics:
            return 0.0
        
        # 计算主题重叠度
        common_topics = set.intersection(*node_topics) if node_topics else set()
        consistency = len(common_topics) / len(all_topics)
        
        return consistency
    
    def _calculate_keyword_overlap(self, path: List[str]) -> float:
        """计算关键词重叠度"""
        all_keywords = set()
        node_keywords = []
        
        for node_id in path:
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                keywords = set(node_data.get('keywords', []))
                node_keywords.append(keywords)
                all_keywords.update(keywords)
        
        if not all_keywords:
            return 0.0
        
        # 计算关键词重叠度
        overlap_count = 0
        total_pairs = 0
        
        for i in range(len(node_keywords)):
            for j in range(i + 1, len(node_keywords)):
                overlap = len(node_keywords[i] & node_keywords[j])
                total_keywords = len(node_keywords[i] | node_keywords[j])
                if total_keywords > 0:
                    overlap_count += overlap / total_keywords
                total_pairs += 1
        
        return overlap_count / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_path_reasoning_value(self, path: List[str]) -> float:
        """计算路径推理价值"""
        reasoning_value = 0.0
        
        # 基于路径中的关系类型计算推理价值
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            
            if self.graph.has_edge(node1, node2):
                edge_data = self.graph[node1][node2]
                relation_type = edge_data.get('relation_type', 'unknown')
                
                # 某些关系类型具有更高的推理价值
                reasoning_weights = {
                    'causal': 1.0,
                    'temporal': 0.8,
                    'definition': 0.7,
                    'semantic_similarity': 0.6,
                    'context': 0.5,
                    'reference': 0.4,
                    'entity_coexistence': 0.3
                }
                
                weight = reasoning_weights.get(relation_type, 0.5)
                reasoning_value += weight
        
        return reasoning_value / (len(path) - 1) if len(path) > 1 else 0.0
    
    def _get_path_relations(self, path: List[str]) -> List[str]:
        """获取路径中的关系类型"""
        relations = []
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            
            if self.graph.has_edge(node1, node2):
                edge_data = self.graph[node1][node2]
                relation_type = edge_data.get('relation_type', 'unknown')
                relations.append(relation_type)
        
        return relations
    
    def _select_diverse_paths(self, scored_paths: List[Dict]) -> List[Dict]:
        """选择多样化的路径"""
        if not scored_paths:
            return []
        
        selected_paths = [scored_paths[0]]  # 总是选择最高分的路径
        
        for path_info in scored_paths[1:]:
            if len(selected_paths) >= self.max_paths:
                break
            
            # 检查与已选择路径的多样性
            is_diverse = True
            for selected_info in selected_paths:
                diversity = self._calculate_path_diversity(path_info['path'], selected_info['path'])
                if diversity < self.path_diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_paths.append(path_info)
        
        logger.info(f"Selected {len(selected_paths)} diverse paths")
        return selected_paths
    
    def _calculate_path_diversity(self, path1: List[str], path2: List[str]) -> float:
        """计算两个路径之间的多样性"""
        # 基于节点重叠计算多样性
        set1, set2 = set(path1), set(path2)
        
        if not set1 or not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        # 多样性 = 1 - 重叠度
        overlap = intersection / union if union > 0 else 0
        diversity = 1.0 - overlap
        
        return diversity
    
    def _extract_results_from_paths(self, selected_paths: List[Dict]) -> List[Dict]:
        """从选择的路径中提取最终结果"""
        node_scores = {}
        node_paths = {}
        
        # 收集所有节点及其分数
        for path_info in selected_paths:
            path = path_info['path']
            path_score = path_info['score']
            
            for i, node_id in enumerate(path):
                if node_id not in node_scores:
                    node_scores[node_id] = 0.0
                    node_paths[node_id] = []
                
                # 基于路径分数和位置计算节点分数
                position_weight = 1.0 / (i + 1)  # 路径开始的节点权重更高
                node_score = path_score * position_weight
                
                node_scores[node_id] += node_score
                node_paths[node_id].append(path_info)
        
        # 构建最终结果
        results = []
        for node_id, total_score in node_scores.items():
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                
                result = {
                    'node_id': node_id,
                    'note_id': node_id,  # 添加 note_id 键
                    'content': node_data.get('content', ''),
                    'title': node_data.get('title', ''),
                    'score': total_score,
                    'graph_score': total_score,  # 保持向后兼容
                    'reasoning_paths': [p['path'] for p in node_paths[node_id]],
                    'path_count': len(node_paths[node_id]),
                    'paragraph_idxs': node_data.get('paragraph_idxs', []),
                    'retrieval_info': {
                        'similarity': total_score,
                        'score': total_score,
                        'rank': len(results),
                        'retrieval_method': 'graph_search'
                    },
                    'metadata': {
                        'centrality': self.index.get_centrality(node_id),
                        'similarity': node_data.get('similarity', 0.0),
                        'keywords': node_data.get('keywords', []),
                        'entities': node_data.get('entities', []),
                        'topics': node_data.get('topics', [])
                    }
                }
                results.append(result)
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def get_reasoning_explanation(self, results: List[Dict], selected_paths: List[Dict]) -> Dict:
        """获取推理解释"""
        explanation = {
            'total_paths': len(selected_paths),
            'max_hops': self.max_hops,
            'relation_types_used': set(),
            'path_details': []
        }
        
        # 收集使用的关系类型
        for path_info in selected_paths:
            relations = path_info.get('relations', [])
            explanation['relation_types_used'].update(relations)
        
        explanation['relation_types_used'] = list(explanation['relation_types_used'])
        
        # 添加顶级路径的详细解释
        top_paths = selected_paths[:3]  # 只解释前3个路径
        explanation['path_details'] = self._get_top_path_explanations(top_paths)
        
        return explanation
    
    def _get_top_path_explanations(self, top_paths: List[Dict]) -> List[Dict]:
        """获取顶级路径的详细解释"""
        explanations = []
        
        for i, path_info in enumerate(top_paths):
            path = path_info['path']
            score = path_info['score']
            relations = path_info.get('relations', [])
            
            explanation = {
                'rank': i + 1,
                'path': path,
                'score': score,
                'length': len(path),
                'relations': relations,
                'reasoning_steps': self._explain_reasoning_steps(path, relations)
            }
            
            explanations.append(explanation)
        
        return explanations
    
    def _explain_reasoning_steps(self, path: List[str], relations: List[str]) -> List[str]:
        """解释推理步骤"""
        steps = []
        
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            relation = relations[i] if i < len(relations) else 'unknown'
            
            # 获取节点标题用于解释
            title1 = self.graph.nodes[node1].get('title', node1) if node1 in self.graph else node1
            title2 = self.graph.nodes[node2].get('title', node2) if node2 in self.graph else node2
            
            step = f"从 '{title1}' 通过 '{relation}' 关系到达 '{title2}'"
            steps.append(step)
        
        return steps
    
    def _bfs_reasoning_paths(self, start_node: str) -> List[List[str]]:
        """使用BFS发现从起始节点的推理路径"""
        paths = []
        queue = deque([(start_node, [start_node])])
        visited_paths = set()
        
        while queue and len(paths) < self.max_paths:
            current_node, current_path = queue.popleft()
            
            # 如果路径已经足够长，添加到结果中
            if len(current_path) >= 2:
                path_key = tuple(current_path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    paths.append(current_path.copy())
            
            # 如果还没有达到最大跳数，继续扩展
            if len(current_path) < self.max_hops + 1:
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in current_path:  # 避免循环
                        new_path = current_path + [neighbor]
                        queue.append((neighbor, new_path))
        
        return paths
    
    def _deduplicate_paths(self, paths: List[List[str]]) -> List[List[str]]:
        """去重路径"""
        unique_paths = []
        seen_paths = set()
        
        for path in paths:
            # 创建路径的标准化表示（考虑双向性）
            forward_key = tuple(path)
            backward_key = tuple(reversed(path))
            
            if forward_key not in seen_paths and backward_key not in seen_paths:
                seen_paths.add(forward_key)
                unique_paths.append(path)
        
        return unique_paths
