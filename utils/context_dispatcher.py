"""Enhanced context dispatcher with graph-aware two-stage rerank.

The dispatcher receives candidates that already contain ``final_similarity``
produced by the hybrid fusion layer. It can operate in two modes:
1. Legacy mode: quota merge and deduplication
2. Graph-aware mode: two-stage rerank (path selection + node selection)
"""
from __future__ import annotations

import math
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict
from loguru import logger

try:
    from graph.graph_retrieval import create_graph_aware_retrieval
    GRAPH_RETRIEVAL_AVAILABLE = True
except ImportError:
    GRAPH_RETRIEVAL_AVAILABLE = False
    logger.warning("Graph retrieval module not available, using legacy mode only")


class ContextDispatcher:
    def __init__(self, config, graph_index=None, vector_retriever=None):
        cfg = config if isinstance(config, dict) else config.load_config()
        d_cfg = cfg.get("dispatcher", {})
        r_cfg = cfg.get("retrieval", {})
        
        # 原有配置
        self.final_semantic_count = d_cfg.get("final_semantic_count", 8)
        self.final_graph_count = d_cfg.get("final_graph_count", 5)
        self.bridge_policy = d_cfg.get("bridge_policy", "keepalive")
        self.bridge_boost_epsilon = d_cfg.get("bridge_boost_epsilon", 0.02)
        self.debug_log = d_cfg.get("debug_log", True)
        
        # 图感知rerank配置
        self.use_graph_rerank = r_cfg.get("use_graph_rerank", False)
        self.token_budget = r_cfg.get("token_budget", 1800)
        
        # 初始化图感知检索器
        self.graph_retriever = None
        if self.use_graph_rerank and GRAPH_RETRIEVAL_AVAILABLE and graph_index:
            try:
                self.graph_retriever = create_graph_aware_retrieval(graph_index, cfg)
                logger.info("Graph-aware rerank enabled")
            except Exception as e:
                logger.error(f"Failed to initialize graph-aware retrieval: {e}")
                self.use_graph_rerank = False
        
        # 存储依赖组件
        self.vector_retriever = vector_retriever
        
        if self.use_graph_rerank and not self.graph_retriever:
            logger.warning("Graph-aware rerank requested but not available, falling back to legacy mode")
            self.use_graph_rerank = False

    def dispatch(self, candidates: List[Dict[str, Any]], query: str = None) -> List[Dict[str, Any]]:
        """主调度方法，根据配置选择legacy或graph-aware模式"""
        if self.use_graph_rerank and self.graph_retriever and query:
            try:
                return self._graph_aware_dispatch(candidates, query)
            except Exception as e:
                logger.error(f"Graph-aware dispatch failed: {e}, falling back to legacy mode")
                return self._legacy_dispatch(candidates)
        else:
            return self._legacy_dispatch(candidates)
    
    def _legacy_dispatch(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """原有的调度逻辑"""
        semantic = [c for c in candidates if c.get("tags", {}).get("source") != "graph"]
        graph = [c for c in candidates if c.get("tags", {}).get("source") == "graph"]

        semantic.sort(key=lambda x: x.get("final_similarity", 0), reverse=True)
        graph.sort(key=lambda x: x.get("final_similarity", 0), reverse=True)

        selected_semantic = semantic[: self.final_semantic_count]
        selected_graph = graph[: self.final_graph_count]

        merged: Dict[str, Dict[str, Any]] = {}
        for cand in selected_semantic:
            merged[cand["note_id"]] = cand
        for cand in selected_graph:
            nid = cand["note_id"]
            if nid in merged:
                merged[nid]["tags"]["is_bridge"] = merged[nid]["tags"].get("is_bridge") or cand["tags"].get("is_bridge")
                merged[nid]["scores"].update({k: v for k, v in cand["scores"].items() if v is not None})
            else:
                merged[nid] = cand

        results = list(merged.values())

        if self.bridge_policy == "boost":
            for cand in results:
                if cand.get("tags", {}).get("is_bridge"):
                    cand["final_similarity"] += self.bridge_boost_epsilon

        results.sort(key=lambda x: x.get("final_similarity", 0), reverse=True)
        limit = self.final_semantic_count + self.final_graph_count
        trimmed = results[:limit]

        if self.bridge_policy == "keepalive":
            bridges = [c for c in results if c.get("tags", {}).get("is_bridge") and c not in trimmed]
            trimmed.extend(bridges)

        return trimmed
    
    def _graph_aware_dispatch(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """图感知两阶段rerank调度"""
        # 第一阶段：路径选择
        selected_paths = self._select_paths(candidates, query)
        
        # 第二阶段：节点选择（token预算内最大覆盖+去冗余）
        final_nodes = self._select_nodes_within_budget(selected_paths, query)
        
        return final_nodes
    
    def _select_paths(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """第一阶段：路径选择"""
        # 提取种子节点
        semantic_seeds = [c["note_id"] for c in candidates 
                         if c.get("tags", {}).get("source") != "graph"]
        bm25_seeds = [c["note_id"] for c in candidates 
                     if c.get("tags", {}).get("source") == "bm25"]
        
        # 生成和选择路径
        paths = self.graph_retriever.generate_and_select_paths(
            query, semantic_seeds, bm25_seeds
        )
        
        # 将路径转换为候选节点，保留路径信息
        path_candidates = []
        for path_info in paths:
            for node_id in path_info["nodes"]:
                # 查找原始候选中的节点信息
                original_cand = next(
                    (c for c in candidates if c["note_id"] == node_id), None
                )
                if original_cand:
                    cand = original_cand.copy()
                    cand["path_info"] = path_info
                    path_candidates.append(cand)
        
        return path_candidates
    
    def _select_nodes_within_budget(self, path_candidates: List[Dict[str, Any]], 
                                   query: str) -> List[Dict[str, Any]]:
        """第二阶段：token预算内节点选择（最大覆盖+去冗余）"""
        if not path_candidates:
            return []
        
        # 按路径分组
        paths_dict = defaultdict(list)
        for cand in path_candidates:
            path_id = cand.get("path_info", {}).get("path_id", "default")
            paths_dict[path_id].append(cand)
        
        # 贪心选择节点，最大化覆盖度并控制冗余
        selected_nodes = []
        selected_ids = set()
        current_tokens = 0
        
        # 按路径得分排序
        sorted_paths = sorted(paths_dict.items(), 
                            key=lambda x: x[1][0].get("path_info", {}).get("score", 0), 
                            reverse=True)
        
        for path_id, nodes in sorted_paths:
            # 按节点在路径中的重要性排序
            nodes.sort(key=lambda x: x.get("final_similarity", 0), reverse=True)
            
            for node in nodes:
                node_id = node["note_id"]
                if node_id in selected_ids:
                    continue
                
                # 估算token消耗
                estimated_tokens = self._estimate_tokens(node)
                if current_tokens + estimated_tokens > self.token_budget:
                    continue
                
                # 计算覆盖增益和冗余惩罚
                coverage_gain = self._calculate_coverage_gain(node, selected_nodes, query)
                redundancy_penalty = self._calculate_redundancy_penalty(node, selected_nodes)
                
                net_gain = coverage_gain - redundancy_penalty
                if net_gain > 0.1:  # 阈值可配置
                    selected_nodes.append(node)
                    selected_ids.add(node_id)
                    current_tokens += estimated_tokens
        
        # 按最终得分排序
        selected_nodes.sort(key=lambda x: x.get("final_similarity", 0), reverse=True)
        
        return selected_nodes
    
    def _estimate_tokens(self, node: Dict[str, Any]) -> int:
        """估算节点的token消耗"""
        # 简单估算：假设平均每个字符0.25个token
        content = node.get("content", "")
        return max(50, int(len(content) * 0.25))
    
    def _calculate_coverage_gain(self, node: Dict[str, Any], 
                               selected_nodes: List[Dict[str, Any]], 
                               query: str) -> float:
        """计算节点的覆盖增益"""
        base_score = node.get("final_similarity", 0)
        path_score = node.get("path_info", {}).get("score", 0)
        
        # 结合基础得分和路径得分
        coverage_gain = 0.7 * base_score + 0.3 * path_score
        
        # 如果是路径中的关键节点，增加权重
        if node.get("path_info", {}).get("is_endpoint", False):
            coverage_gain *= 1.2
        
        return coverage_gain
    
    def _calculate_redundancy_penalty(self, node: Dict[str, Any], 
                                    selected_nodes: List[Dict[str, Any]]) -> float:
        """计算冗余惩罚"""
        if not selected_nodes:
            return 0.0
        
        # 简单的基于相似度的冗余计算
        max_similarity = 0.0
        node_content = node.get("content", "")
        
        for selected in selected_nodes:
            selected_content = selected.get("content", "")
            # 简单的词汇重叠度计算
            similarity = self._calculate_content_similarity(node_content, selected_content)
            max_similarity = max(max_similarity, similarity)
        
        # 冗余惩罚随最大相似度增加
        return max_similarity * 0.5
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度（简单实现）"""
        if not content1 or not content2:
            return 0.0
        
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
