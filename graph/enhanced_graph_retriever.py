import heapq
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, deque
from loguru import logger
import networkx as nx
from config import config

class EnhancedGraphRetriever:
    """增强的图谱检索器，专门针对多跳推理优化"""
    
    def __init__(self, graph_index):
        self.graph_index = graph_index
        self.graph = graph_index.graph
        
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
        
        logger.info("EnhancedGraphRetriever initialized with multi-hop reasoning support")
    
    def retrieve_with_reasoning_paths(self, query_embedding: np.ndarray, 
                                    top_k: int = 10,
                                    query_keywords: List[str] = None,
                                    query_entities: List[str] = None) -> List[Dict[str, Any]]:
        """基于推理路径的检索"""
        logger.info(f"Starting reasoning path retrieval for top-{top_k} results")
        
        # 1. 找到初始候选节点
        initial_candidates = self._find_initial_candidates(
            query_embedding, query_keywords, query_entities
        )
        
        if not initial_candidates:
            logger.warning("No initial candidates found")
            return []
        
        # 2. 发现推理路径
        reasoning_paths = self._discover_reasoning_paths(initial_candidates)
        
        # 3. 评估和排序路径
        scored_paths = self._score_reasoning_paths(reasoning_paths, query_embedding)
        
        # 4. 选择多样化的路径
        selected_paths = self._select_diverse_paths(scored_paths)
        
        # 5. 从路径中提取最终结果
        final_results = self._extract_results_from_paths(
            selected_paths, query_embedding, top_k
        )
        
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
        if not hasattr(self.graph_index, 'embeddings') or self.graph_index.embeddings is None:
            return []
        
        # 计算相似度
        similarities = np.dot(self.graph_index.embeddings, query_embedding)
        
        # 获取top-k最相似的节点
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        candidates = []
        for idx in top_indices:
            if idx < len(self.graph_index.note_id_to_index):
                note_id = list(self.graph_index.note_id_to_index.keys())[idx]
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
    
    def _score_reasoning_paths(self, paths: List[List[str]], 
                             query_embedding: np.ndarray) -> List[Tuple[List[str], float]]:
        """评估推理路径的得分"""
        scored_paths = []
        threshold = self.min_path_score

        while True:
            scored_paths = []
            for path in paths:
                score = self._calculate_path_score(path, query_embedding)
                if score >= threshold:
                    scored_paths.append((path, score))

            if scored_paths or threshold <= self.min_path_score_floor:
                break

            threshold = max(threshold - self.min_path_score_step, self.min_path_score_floor)
            logger.debug(f"Lowering path score threshold to {threshold:.2f} due to sparse results")

        scored_paths.sort(key=lambda x: x[1], reverse=True)

        return scored_paths
    
    def _calculate_path_score(self, path: List[str], query_embedding: np.ndarray) -> float:
        """计算路径得分"""
        if len(path) < 2:
            return 0.0
        
        # 1. 路径长度权重
        path_length = len(path) - 1
        if path_length == 1:
            length_weight = self.path_weights['direct']
        elif path_length == 2:
            length_weight = self.path_weights['two_hop']
        elif path_length == 3:
            length_weight = self.path_weights['three_hop']
        else:
            length_weight = self.path_weights['multi_hop']
        
        # 2. 关系质量得分
        relation_score = self._calculate_relation_score(path)
        
        # 3. 节点质量得分
        node_score = self._calculate_node_score(path, query_embedding)
        
        # 4. 路径连贯性得分
        coherence_score = self._calculate_path_coherence(path)
        
        # 5. 推理价值得分
        reasoning_value = self._calculate_path_reasoning_value(path)
        
        # 综合得分
        total_score = (
            0.2 * length_weight +
            0.3 * relation_score +
            0.2 * node_score +
            0.15 * coherence_score +
            0.15 * reasoning_value
        )
        
        return min(total_score, 1.0)
    
    def _calculate_relation_score(self, path: List[str]) -> float:
        """计算路径中关系的质量得分"""
        if len(path) < 2:
            return 0.0
        
        relation_scores = []
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            if self.graph.has_edge(source, target):
                edge_data = self.graph[source][target]
                relation_type = edge_data.get('relation_type', 'unknown')
                relation_weight = edge_data.get('weight', 0.5)
                
                # 基于关系类型的权重
                type_weight = self.relation_weights.get(relation_type, 0.5)
                
                # 推理价值
                reasoning_value = edge_data.get('reasoning_value', 0.5)
                
                # 关系得分
                relation_score = type_weight * relation_weight * reasoning_value
                relation_scores.append(relation_score)
            else:
                relation_scores.append(0.1)  # 惩罚缺失的边
        
        return np.mean(relation_scores) if relation_scores else 0.0
    
    def _calculate_node_score(self, path: List[str], query_embedding: np.ndarray) -> float:
        """计算路径中节点的质量得分"""
        if not hasattr(self.graph_index, 'embeddings') or self.graph_index.embeddings is None:
            return 0.5
        
        node_scores = []
        
        for node_id in path:
            if node_id in self.graph_index.note_id_to_index:
                node_idx = self.graph_index.note_id_to_index[node_id]
                node_embedding = self.graph_index.embeddings[node_idx]
                
                # 与查询的相似度
                similarity = np.dot(node_embedding, query_embedding)
                
                # 节点中心性
                centrality = self.graph_index.centrality_scores.get(node_id, 0.0)
                
                # 节点重要性
                importance = self.graph.nodes[node_id].get("importance_score", 1.0)
                # 节点得分
                node_score = (0.7 * similarity + 0.3 * centrality) * importance
                node_scores.append(node_score)
            else:
                node_scores.append(0.1)
        
        return np.mean(node_scores) if node_scores else 0.0
    
    def _calculate_path_coherence(self, path: List[str]) -> float:
        """计算路径的连贯性得分"""
        if len(path) < 3:
            return 1.0  # 短路径默认连贯
        
        coherence_scores = []
        
        # 检查相邻节点对的连贯性
        for i in range(len(path) - 2):
            node1 = path[i]
            node2 = path[i + 1]
            node3 = path[i + 2]
            
            # 计算三元组的连贯性
            triplet_coherence = self._calculate_triplet_coherence(node1, node2, node3)
            coherence_scores.append(triplet_coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_triplet_coherence(self, node1: str, node2: str, node3: str) -> float:
        """计算三元组的连贯性"""
        # 获取节点数据
        node1_data = self.graph.nodes.get(node1, {})
        node2_data = self.graph.nodes.get(node2, {})
        node3_data = self.graph.nodes.get(node3, {})
        
        # 主题一致性
        topics = []
        for node_data in [node1_data, node2_data, node3_data]:
            topic = node_data.get('topic', '')
            if topic:
                topics.append(topic)
        
        topic_consistency = len(set(topics)) / len(topics) if topics else 0.5
        topic_consistency = 1.0 - topic_consistency  # 转换为一致性得分
        
        # 关键词重叠
        keywords1 = set(node1_data.get('keywords', []))
        keywords2 = set(node2_data.get('keywords', []))
        keywords3 = set(node3_data.get('keywords', []))
        
        all_keywords = keywords1 | keywords2 | keywords3
        common_keywords = keywords1 & keywords2 & keywords3
        
        keyword_overlap = len(common_keywords) / len(all_keywords) if all_keywords else 0.0
        
        # 综合连贯性
        coherence = 0.6 * topic_consistency + 0.4 * keyword_overlap
        
        return coherence
    
    def _calculate_path_reasoning_value(self, path: List[str]) -> float:
        """计算路径的推理价值"""
        if len(path) < 2:
            return 0.0
        
        reasoning_values = []
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            if self.graph.has_edge(source, target):
                edge_data = self.graph[source][target]
                reasoning_value = edge_data.get('reasoning_value', 0.5)
                reasoning_values.append(reasoning_value)
            else:
                reasoning_values.append(0.1)
        
        # 路径的推理价值是所有边推理价值的几何平均
        if reasoning_values:
            geometric_mean = np.exp(np.mean(np.log(np.array(reasoning_values) + 1e-8)))
            return geometric_mean
        
        return 0.0
    
    def _select_diverse_paths(self, scored_paths: List[Tuple[List[str], float]]) -> List[Tuple[List[str], float]]:
        """选择多样化的路径"""
        if not scored_paths:
            return []
        
        selected_paths = [scored_paths[0]]  # 总是选择得分最高的路径
        
        for path, score in scored_paths[1:]:
            # 检查与已选择路径的多样性
            is_diverse = True
            for selected_path, _ in selected_paths:
                diversity = self._calculate_path_diversity(path, selected_path)
                if diversity < self.path_diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_paths.append((path, score))
            
            # 限制选择的路径数量
            if len(selected_paths) >= self.max_paths:
                break
        
        return selected_paths
    
    def _calculate_path_diversity(self, path1: List[str], path2: List[str]) -> float:
        """计算两个路径的多样性"""
        # 节点重叠度
        nodes1 = set(path1)
        nodes2 = set(path2)
        
        if not nodes1 or not nodes2:
            return 1.0
        
        overlap = len(nodes1 & nodes2)
        total = len(nodes1 | nodes2)
        
        # 多样性 = 1 - 重叠度
        diversity = 1.0 - (overlap / total) if total > 0 else 0.0
        
        return diversity
    
    def _extract_results_from_paths(self, selected_paths: List[Tuple[List[str], float]],
                                  query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """从选择的路径中提取最终结果"""
        # 收集所有路径中的节点及其得分
        node_scores = defaultdict(list)
        node_path_info = defaultdict(list)
        
        for path, path_score in selected_paths:
            for i, node_id in enumerate(path):
                # 节点在路径中的位置权重
                position_weight = 1.0 / (i + 1)  # 路径开始的节点权重更高
                
                # 节点得分 = 路径得分 * 位置权重
                node_score = path_score * position_weight
                node_scores[node_id].append(node_score)
                
                # 记录路径信息
                node_path_info[node_id].append({
                    'path': path,
                    'path_score': path_score,
                    'position_in_path': i,
                    'position_weight': position_weight
                })
        
        # 计算每个节点的最终得分
        final_node_scores = {}
        for node_id, scores in node_scores.items():
            # 使用最大值和平均值的组合
            max_score = max(scores)
            avg_score = np.mean(scores)
            final_score = 0.7 * max_score + 0.3 * avg_score
            final_node_scores[node_id] = final_score
        
        # 排序并选择top-k
        sorted_nodes = sorted(final_node_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = sorted_nodes[:top_k]
        
        # 构建最终结果
        results = []
        for node_id, final_score in top_nodes:
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id].copy()
                
                # 添加推理信息
                reasoning_info = {
                    'final_score': final_score,
                    'reasoning_paths': node_path_info[node_id],
                    'path_count': len(node_path_info[node_id]),
                    'max_path_score': max(info['path_score'] for info in node_path_info[node_id]),
                    'retrieval_method': 'reasoning_path'
                }
                
                node_data.update(reasoning_info)
                results.append(node_data)
        
        return results
    
    def get_reasoning_explanation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取推理过程的解释"""
        if not results:
            return {'explanation': 'No reasoning paths found'}
        
        # 分析推理路径
        all_paths = []
        relation_types_used = set()
        max_hops = 0
        
        for result in results:
            reasoning_paths = result.get('reasoning_paths', [])
            for path_info in reasoning_paths:
                path = path_info['path']
                all_paths.append(path)
                max_hops = max(max_hops, len(path) - 1)
                
                # 收集使用的关系类型
                for i in range(len(path) - 1):
                    source = path[i]
                    target = path[i + 1]
                    if self.graph.has_edge(source, target):
                        relation_type = self.graph[source][target].get('relation_type', 'unknown')
                        relation_types_used.add(relation_type)
        
        # 生成解释
        explanation = {
            'total_paths': len(all_paths),
            'max_reasoning_hops': max_hops,
            'relation_types_used': list(relation_types_used),
            'reasoning_strategy': 'multi_hop_path_discovery',
            'top_paths': self._get_top_path_explanations(results[:3])
        }
        
        return explanation
    
    def _get_top_path_explanations(self, top_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取顶部结果的路径解释"""
        explanations = []
        
        for result in top_results:
            reasoning_paths = result.get('reasoning_paths', [])
            if reasoning_paths:
                best_path_info = max(reasoning_paths, key=lambda x: x['path_score'])
                path = best_path_info['path']
                
                # 构建路径解释
                path_explanation = {
                    'target_note': result.get('note_id', 'unknown'),
                    'reasoning_path': path,
                    'path_length': len(path) - 1,
                    'path_score': best_path_info['path_score'],
                    'reasoning_steps': self._explain_reasoning_steps(path)
                }
                
                explanations.append(path_explanation)
        
        return explanations
    
    def _explain_reasoning_steps(self, path: List[str]) -> List[Dict[str, Any]]:
        """解释推理步骤"""
        steps = []
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            step_info = {
                'step': i + 1,
                'from_note': source,
                'to_note': target,
                'relation_type': 'unknown',
                'relation_weight': 0.0,
                'reasoning': 'No relation found'
            }
            
            if self.graph.has_edge(source, target):
                edge_data = self.graph[source][target]
                relation_type = edge_data.get('relation_type', 'unknown')
                relation_weight = edge_data.get('weight', 0.0)
                reasoning = edge_data.get('metadata', {}).get('reasoning', '')
                
                step_info.update({
                    'relation_type': relation_type,
                    'relation_weight': relation_weight,
                    'reasoning': reasoning or f'Connected via {relation_type} relation'
                })
            
            steps.append(step_info)
        
        return steps