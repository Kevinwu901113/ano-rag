#!/usr/bin/env python3
"""
结构增强的上下文调度器
实现三阶段检索流程：语义召回 -> 图谱扩展 -> 上下文调度
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from loguru import logger
from collections import defaultdict
import numpy as np

from config import config
from vector_store import VectorRetriever
from graph.graph_retriever import GraphRetriever


class ContextDispatcher:
    """结构增强的上下文调度器
    
    实现三阶段检索流程：
    1. 语义召回：使用嵌入向量余弦相似度召回top-n原子笔记
    2. 图谱扩展：对top-p个召回项进行k-hop扩展
    3. 上下文调度：从两个集合中选取前x个语义结果和前y个图谱结果
    """
    
    def __init__(self, 
                 vector_retriever: VectorRetriever,
                 graph_retriever: GraphRetriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        
        # 从配置加载参数
        dispatcher_config = config.get('context_dispatcher', {})
        
        # 阶段1：语义召回参数
        self.semantic_top_n = dispatcher_config.get('semantic_top_n', 50)  # n
        
        # 阶段2：图谱扩展参数
        self.graph_expand_top_p = dispatcher_config.get('graph_expand_top_p', 20)  # p < n
        self.k_hop = dispatcher_config.get('k_hop', 2)  # k
        
        # 阶段3：上下文调度参数
        self.final_semantic_count = dispatcher_config.get('final_semantic_count', 8)  # x
        self.final_graph_count = dispatcher_config.get('final_graph_count', 5)  # y
        
        # 上下文模板
        self.context_template = dispatcher_config.get(
            'context_template',
            "Note {note_id}: {content}\n"
        )
        
        # 相似度阈值
        self.semantic_threshold = dispatcher_config.get('semantic_threshold', 0.1)
        self.graph_threshold = dispatcher_config.get('graph_threshold', 0.0)
        
        logger.info(f"ContextDispatcher initialized with params: "
                   f"n={self.semantic_top_n}, p={self.graph_expand_top_p}, "
                   f"k={self.k_hop}, x={self.final_semantic_count}, y={self.final_graph_count}")
    
    def dispatch(self, query: str, rewritten_queries: List[str] = None) -> Dict[str, Any]:
        """执行三阶段上下文调度
        
        Args:
            query: 原始查询
            rewritten_queries: 重写后的查询列表，如果为None则使用原始查询
            
        Returns:
            包含最终上下文和相关信息的字典
        """
        if rewritten_queries is None:
            rewritten_queries = [query]
            
        logger.info(f"Starting context dispatch for query: {query}")
        
        # 阶段1：语义召回
        semantic_results = self._semantic_recall(rewritten_queries)
        logger.info(f"Stage 1 - Semantic recall: {len(semantic_results)} notes")
        
        # 阶段2：图谱扩展召回
        graph_results = self._graph_expansion_recall(semantic_results)
        logger.info(f"Stage 2 - Graph expansion: {len(graph_results)} notes")
        
        # 阶段3：上下文调度
        final_context, selected_notes = self._context_scheduling(
            query,
            semantic_results,
            graph_results
        )
        logger.info(f"Stage 3 - Context scheduling: {len(selected_notes)} final notes")
        
        return {
            'context': final_context,
            'selected_notes': selected_notes,
            'semantic_results': semantic_results,
            'graph_results': graph_results,
            'stage_info': {
                'semantic_count': len(semantic_results),
                'graph_count': len(graph_results),
                'final_count': len(selected_notes)
            }
        }
    
    def _semantic_recall(self, queries: List[str]) -> List[Dict[str, Any]]:
        """阶段1：语义召回
        
        使用嵌入向量余弦相似度召回top-n原子笔记
        """
        try:
            # 执行向量检索
            search_results = self.vector_retriever.search(
                queries, 
                top_k=self.semantic_top_n,
                similarity_threshold=self.semantic_threshold
            )
            
            # 合并多个查询的结果并去重
            semantic_notes = []
            seen_note_ids = set()
            
            for query_results in search_results:
                for note in query_results:
                    note_id = note.get('note_id')
                    if note_id and note_id not in seen_note_ids:
                        # 标记为语义召回结果
                        note['retrieval_stage'] = 'semantic'
                        note['stage_rank'] = len(semantic_notes) + 1
                        semantic_notes.append(note)
                        seen_note_ids.add(note_id)
            
            # 按相似度排序
            semantic_notes.sort(
                key=lambda x: x.get('retrieval_info', {}).get('similarity', 0),
                reverse=True
            )
            
            return semantic_notes[:self.semantic_top_n]
            
        except Exception as e:
            logger.error(f"Semantic recall failed: {e}")
            return []
    
    def _graph_expansion_recall(self, semantic_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """阶段2：图谱扩展召回
        
        对top-p个语义召回结果进行k-hop图谱扩展
        """
        if not semantic_results:
            return []
            
        try:
            # 选择top-p个语义结果作为种子节点
            seed_notes = semantic_results[:self.graph_expand_top_p]
            seed_note_ids = [note.get('note_id') for note in seed_notes if note.get('note_id')]
            
            if not seed_note_ids:
                logger.warning("No valid seed note IDs for graph expansion")
                return []
            
            logger.info(f"Graph expansion from {len(seed_note_ids)} seed notes")
            
            # 执行k-hop图谱检索
            graph_notes = self.graph_retriever.retrieve(seed_note_ids)
            
            # 过滤已在语义结果中的笔记
            semantic_note_ids = {note.get('note_id') for note in semantic_results}
            filtered_graph_notes = []
            
            for note in graph_notes:
                note_id = note.get('note_id')
                if note_id and note_id not in semantic_note_ids:
                    # 标记为图谱扩展结果
                    note['retrieval_stage'] = 'graph_expansion'
                    note['stage_rank'] = len(filtered_graph_notes) + 1
                    
                    # 确保有基本的相似度信息
                    if 'retrieval_info' not in note:
                        note['retrieval_info'] = {}
                    if 'similarity' not in note['retrieval_info']:
                        note['retrieval_info']['similarity'] = note.get('graph_score', 0.0)
                    
                    filtered_graph_notes.append(note)
            
            # 按图谱分数排序
            filtered_graph_notes.sort(
                key=lambda x: x.get('graph_score', 0),
                reverse=True
            )
            
            return filtered_graph_notes
            
        except Exception as e:
            logger.error(f"Graph expansion recall failed: {e}")
            return []
    
    def _context_scheduling(self,
                           query: str,
                           semantic_results: List[Dict[str, Any]],
                           graph_results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """阶段3：上下文调度

        从语义和图谱结果中分别选取前x和前y个笔记，合并去重后生成最终上下文
        """
        selected_notes = []

        # 选择前x个语义结果
        semantic_selected = semantic_results[:self.final_semantic_count]
        for note in semantic_selected:
            note['selection_reason'] = 'semantic_top'
            selected_notes.append(note)

        # 选择前y个图谱结果（去重）
        semantic_note_ids = {note.get('note_id') for note in semantic_selected}
        graph_selected = []

        for note in graph_results:
            if len(graph_selected) >= self.final_graph_count:
                break

            note_id = note.get('note_id')
            if note_id and note_id not in semantic_note_ids:
                note['selection_reason'] = 'graph_expansion'
                graph_selected.append(note)
                selected_notes.append(note)

        logger.info(f"Selected {len(semantic_selected)} semantic + {len(graph_selected)} graph notes")

        # 计算查询嵌入
        try:
            query_embedding = self.vector_retriever.embedding_manager.encode_queries([query])[0]
        except Exception as e:
            logger.error(f"Failed to encode query embedding: {e}")
            query_embedding = np.array([])

        # 计算每个笔记与查询的最终相似度
        for note in selected_notes:
            note_id = note.get('note_id')
            note_embedding = None

            if note_id is not None:
                idx = self.vector_retriever.note_id_to_index.get(note_id)
                if (idx is not None and self.vector_retriever.note_embeddings is not None and
                        idx < len(self.vector_retriever.note_embeddings)):
                    note_embedding = self.vector_retriever.note_embeddings[idx]

            if note_embedding is None:
                try:
                    note_embedding = self.vector_retriever.embedding_manager.encode_atomic_notes([note])[0]
                except Exception as e:
                    logger.error(f"Failed to encode note {note_id}: {e}")
                    note_embedding = np.array([])

            if query_embedding.size > 0 and note_embedding.size > 0:
                similarity = float(
                    self.vector_retriever.embedding_manager.compute_similarity(
                        query_embedding, note_embedding
                    )[0][0]
                )
            else:
                similarity = 0.0

            if 'retrieval_info' not in note:
                note['retrieval_info'] = {}
            note['retrieval_info']['final_similarity'] = similarity

        # 按最终相似度排序
        selected_notes.sort(
            key=lambda n: n.get('retrieval_info', {}).get('final_similarity', 0.0),
            reverse=True
        )

        # 生成最终上下文并标注排名
        context_parts = []
        for i, note in enumerate(selected_notes, 1):
            note['final_rank'] = i
            note['retrieval_info']['final_rank'] = i

            note_id = note.get('note_id', f'note_{i}')
            content = note.get('content', '')

            # 使用模板格式化上下文
            formatted_content = self.context_template.format(
                note_id=note_id,
                content=content
            )
            context_parts.append(formatted_content)

        final_context = '\n'.join(context_parts)

        return final_context, selected_notes
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取当前配置摘要"""
        return {
            'semantic_top_n': self.semantic_top_n,
            'graph_expand_top_p': self.graph_expand_top_p,
            'k_hop': self.k_hop,
            'final_semantic_count': self.final_semantic_count,
            'final_graph_count': self.final_graph_count,
            'semantic_threshold': self.semantic_threshold,
            'graph_threshold': self.graph_threshold
        }
    
    def update_config(self, **kwargs):
        """动态更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")