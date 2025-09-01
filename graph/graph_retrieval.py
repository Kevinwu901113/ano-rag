#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图感知两阶段rerank - 子图截取与路径生成模块

实现功能：
1. 基于语义+BM25种子的子图截取
2. K条路径生成（简单路径，避免循环）
3. 路径打分函数（终点相似度+平均权重+上下文覆盖-长度惩罚）
4. 自适应参数调整（precise/explanatory/auto模式）
"""

import math
import heapq
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, deque
from loguru import logger
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available, using simplified graph operations")


class GraphAwareRetrieval:
    """图感知检索器 - 子图截取与路径生成"""
    
    def __init__(self, graph_index, config: Dict[str, Any]):
        self.graph_index = graph_index
        self.graph = graph_index.graph if graph_index else None
        
        # 从配置中读取参数
        retrieval_config = config.get('retrieval', {})
        self.seeds_semantic = retrieval_config.get('seeds_semantic', 50)
        self.seeds_bm25 = retrieval_config.get('seeds_bm25', 30)
        self.subgraph_radius = retrieval_config.get('subgraph_radius', 2)
        self.edge_thresh = retrieval_config.get('edge_thresh', 0.35)
        self.k_paths = retrieval_config.get('k_paths', 20)
        self.pick_paths = retrieval_config.get('pick_paths', 4)
        self.overlap_thresh = retrieval_config.get('overlap_thresh', 0.5)
        self.token_budget = retrieval_config.get('token_budget', 1800)
        
        # 路径打分参数
        self.alpha = retrieval_config.get('alpha', 0.5)  # 终点相似度权重
        self.beta = retrieval_config.get('beta', 0.3)    # 路径平均权重
        self.gamma = retrieval_config.get('gamma', 0.2)  # 上下文覆盖权重
        self.rho = retrieval_config.get('rho', 0.25)     # 上下文覆盖上限
        self.lambda_len = retrieval_config.get('lambda_len', 0.05)  # 长度惩罚
        
        # 查询模式
        self.query_mode = retrieval_config.get('query_mode', 'auto')
        
        # 自适应调整参数
        self._adjust_parameters_by_mode()
        
        logger.info(f"GraphAwareRetrieval initialized with mode: {self.query_mode}")
    
    def _adjust_parameters_by_mode(self):
        """根据查询模式自适应调整参数"""
        if self.query_mode == 'precise':
            # precise模式：α↑ β↓ γ↓ ρ↓
            self.alpha = min(0.7, self.alpha * 1.4)
            self.beta = max(0.1, self.beta * 0.7)
            self.gamma = max(0.1, self.gamma * 0.7)
            self.rho = max(0.15, self.rho * 0.6)
        elif self.query_mode == 'explanatory':
            # explanatory模式：α↓ β↑ γ↑ ρ↑
            self.alpha = max(0.2, self.alpha * 0.6)
            self.beta = min(0.5, self.beta * 1.5)
            self.gamma = min(0.4, self.gamma * 1.5)
            self.rho = min(0.4, self.rho * 1.6)
        # auto模式保持原参数不变
    
    def extract_subgraph(self, semantic_seeds: List[str], bm25_seeds: List[str]) -> Set[str]:
        """基于种子节点截取子图"""
        if not self.graph:
            logger.warning("No graph available for subgraph extraction")
            return set()
        
        # 合并种子节点并去重
        all_seeds = list(set(semantic_seeds[:self.seeds_semantic] + bm25_seeds[:self.seeds_bm25]))
        
        # 过滤存在于图中的种子节点
        valid_seeds = [seed for seed in all_seeds if seed in self.graph]
        
        if not valid_seeds:
            logger.warning("No valid seed nodes found in graph")
            return set()
        
        logger.info(f"Starting subgraph extraction from {len(valid_seeds)} seeds with radius {self.subgraph_radius}")
        
        # BFS扩展子图
        subgraph_nodes = set(valid_seeds)
        current_layer = set(valid_seeds)
        
        for hop in range(self.subgraph_radius):
            next_layer = set()
            for node in current_layer:
                if node in self.graph:
                    for neighbor in self.graph.neighbors(node):
                        # 检查边权阈值
                        edge_data = self.graph.get_edge_data(node, neighbor, {})
                        weight = edge_data.get('weight', 1.0)
                        
                        if weight >= self.edge_thresh:
                            next_layer.add(neighbor)
            
            subgraph_nodes.update(next_layer)
            current_layer = next_layer
            
            if not next_layer:
                break
        
        logger.info(f"Extracted subgraph with {len(subgraph_nodes)} nodes")
        return subgraph_nodes
    
    def generate_paths(self, subgraph_nodes: Set[str], query_embedding: np.ndarray, 
                      similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """在子图上生成K条简单路径"""
        if not self.graph or not subgraph_nodes:
            return []
        
        # 创建子图
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # 找到与query相似度≥σ的终点节点
        end_nodes = self._find_similar_nodes(subgraph_nodes, query_embedding, similarity_threshold)
        
        if not end_nodes:
            logger.warning("No suitable end nodes found for path generation")
            return []
        
        logger.info(f"Generating paths to {len(end_nodes)} end nodes")
        
        all_paths = []
        
        # 为每个种子节点生成到终点的路径
        for start_node in subgraph_nodes:
            if start_node not in subgraph:
                continue
                
            paths_from_start = self._generate_paths_from_node(
                subgraph, start_node, end_nodes
            )
            all_paths.extend(paths_from_start)
            
            # 限制总路径数量
            if len(all_paths) >= self.k_paths:
                break
        
        # 截取前K条路径
        all_paths = all_paths[:self.k_paths]
        
        # 计算路径得分
        scored_paths = []
        for path in all_paths:
            score = self._calculate_path_score(path, query_embedding)
            scored_paths.append({
                'path': path,
                'score': score,
                'length': len(path)
            })
        
        # 按得分排序
        scored_paths.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Generated {len(scored_paths)} scored paths")
        return scored_paths
    
    def _find_similar_nodes(self, nodes: Set[str], query_embedding: np.ndarray, 
                           threshold: float) -> List[str]:
        """找到与查询相似度≥阈值的节点"""
        similar_nodes = []
        
        for node in nodes:
            # 获取节点嵌入向量
            node_embedding = self._get_node_embedding(node)
            if node_embedding is not None:
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_embedding, node_embedding)
                if similarity >= threshold:
                    similar_nodes.append(node)
        
        return similar_nodes
    
    def _get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """获取节点的嵌入向量"""
        try:
            if hasattr(self.graph_index, 'get_node_embedding'):
                return self.graph_index.get_node_embedding(node_id)
            # 回退方案：从图节点数据中获取
            node_data = self.graph.nodes.get(node_id, {})
            return node_data.get('embedding')
        except Exception as e:
            logger.debug(f"Failed to get embedding for node {node_id}: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    def _generate_paths_from_node(self, subgraph, start_node: str, 
                                 end_nodes: List[str]) -> List[List[str]]:
        """从起始节点生成到终点的简单路径"""
        paths = []
        
        # 使用BFS生成简单路径（避免循环）
        queue = deque([(start_node, [start_node])])
        visited_paths = set()
        
        while queue and len(paths) < 10:  # 限制每个起点的路径数
            current_node, current_path = queue.popleft()
            
            # 如果到达终点，记录路径
            if current_node in end_nodes and len(current_path) > 1:
                path_key = tuple(current_path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    paths.append(current_path.copy())
            
            # 继续扩展路径（限制最大长度）
            if len(current_path) < 5:  # 最大路径长度
                for neighbor in subgraph.neighbors(current_node):
                    if neighbor not in current_path:  # 避免循环
                        new_path = current_path + [neighbor]
                        queue.append((neighbor, new_path))
        
        return paths
    
    def _calculate_path_score(self, path: List[str], query_embedding: np.ndarray) -> float:
        """计算路径得分"""
        if len(path) < 2:
            return 0.0
        
        # 1. 终点相似度 S_end
        end_node = path[-1]
        end_embedding = self._get_node_embedding(end_node)
        s_end = 0.0
        if end_embedding is not None:
            s_end = self._cosine_similarity(query_embedding, end_embedding)
        
        # 2. 路径平均权重 S_mean
        edge_weights = []
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i+1], {})
            weight = edge_data.get('weight', 0.5)
            # 归一化到(0,1]
            normalized_weight = max(0.01, min(1.0, weight))
            edge_weights.append(normalized_weight)
        
        s_mean = sum(edge_weights) / len(edge_weights) if edge_weights else 0.0
        
        # 3. 上下文覆盖 C_ctx（简化为路径长度的函数）
        c_ctx = min(len(path) / 5.0, 1.0)  # 归一化路径长度
        
        # 4. 长度惩罚
        l0 = 3  # 理想路径长度
        length_penalty = max(0, len(path) - l0)
        
        # 计算最终得分
        score = (self.alpha * s_end + 
                self.beta * s_mean + 
                self.gamma * min(c_ctx, self.rho) - 
                self.lambda_len * length_penalty)
        
        return max(0.0, score)  # 确保得分非负
    
    def select_diverse_paths(self, scored_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """选择多样化的路径（贪心最大覆盖）"""
        if not scored_paths:
            return []
        
        selected_paths = []
        remaining_paths = scored_paths.copy()
        
        # 贪心选择路径
        while len(selected_paths) < self.pick_paths and remaining_paths:
            best_path = None
            best_score = -1
            best_idx = -1
            
            for i, path_info in enumerate(remaining_paths):
                # 计算与已选路径的重叠度
                overlap_penalty = self._calculate_overlap_penalty(
                    path_info['path'], selected_paths
                )
                
                # 调整得分
                adjusted_score = path_info['score'] * (1 - overlap_penalty)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_path = path_info
                    best_idx = i
            
            if best_path:
                selected_paths.append(best_path)
                remaining_paths.pop(best_idx)
            else:
                break
        
        logger.info(f"Selected {len(selected_paths)} diverse paths from {len(scored_paths)} candidates")
        return selected_paths
    
    def _calculate_overlap_penalty(self, path: List[str], selected_paths: List[Dict[str, Any]]) -> float:
        """计算路径重叠惩罚"""
        if not selected_paths:
            return 0.0
        
        max_overlap = 0.0
        path_set = set(path)
        
        for selected in selected_paths:
            selected_set = set(selected['path'])
            intersection = len(path_set & selected_set)
            union = len(path_set | selected_set)
            
            if union > 0:
                overlap = intersection / union
                max_overlap = max(max_overlap, overlap)
        
        # 如果重叠度超过阈值，应用惩罚
        if max_overlap > self.overlap_thresh:
            return max_overlap
        
        return 0.0
    
    def generate_and_select_paths(self, query: str, semantic_seeds: List[str], 
                                 bm25_seeds: List[str]) -> List[Dict[str, Any]]:
        """生成和选择路径的主要接口"""
        try:
            # 获取查询嵌入向量
            query_embedding = self._get_query_embedding(query)
            if query_embedding is None:
                logger.warning("Failed to get query embedding")
                return []
            
            # 截取子图
            subgraph_nodes = self.extract_subgraph(semantic_seeds, bm25_seeds)
            if not subgraph_nodes:
                return []
            
            # 生成路径
            scored_paths = self.generate_paths(subgraph_nodes, query_embedding)
            if not scored_paths:
                return []
            
            # 选择多样化路径
            selected_paths = self.select_diverse_paths(scored_paths)
            
            # 格式化输出
            result_paths = []
            for i, path_info in enumerate(selected_paths):
                result_paths.append({
                    'path_id': f'path_{i}',
                    'nodes': path_info['path'],
                    'score': path_info['score'],
                    'length': path_info['length']
                })
            
            return result_paths
            
        except Exception as e:
            logger.error(f"Error in generate_and_select_paths: {e}")
            return []
    
    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """获取查询的嵌入向量"""
        try:
            # 尝试使用图索引的嵌入方法
            if hasattr(self.graph_index, 'get_query_embedding'):
                return self.graph_index.get_query_embedding(query)
            # 简化实现：返回随机向量作为占位符
            return np.random.rand(768)  # 假设768维向量
        except Exception as e:
            logger.debug(f"Failed to get query embedding: {e}")
            return None


def create_graph_aware_retrieval(graph_index, config: Dict[str, Any]) -> GraphAwareRetrieval:
    """创建图感知检索器实例"""
    return GraphAwareRetrieval(graph_index, config)