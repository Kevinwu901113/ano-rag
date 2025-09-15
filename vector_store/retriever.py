import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from .embedding_manager import EmbeddingManager
from .vector_index import VectorIndex
from utils import BatchProcessor, FileUtils
from config import config

# 导入混合检索模块
try:
    from retrieval.hybrid_search import HybridSearcher, create_hybrid_searcher
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False
    logger.warning("Hybrid search module not available")

try:
    from retrieval.retrieval_guardrail import RetrievalGuardrail, create_retrieval_guardrail
    GUARDRAIL_AVAILABLE = True
except ImportError:
    GUARDRAIL_AVAILABLE = False
    logger.warning("Retrieval guardrail module not available")

class VectorRetriever:
    """向量检索器，整合嵌入管理和向量索引功能"""
    
    def __init__(self):
        # 初始化组件
        self.embedding_manager = EmbeddingManager()
        self.vector_index = VectorIndex(self.embedding_manager.embedding_dim)
        
        # 配置参数
        self.top_k = config.get('vector_store.top_k', 20)
        self.similarity_threshold = config.get('vector_store.similarity_threshold', 0.5)
        self.batch_size = config.get('vector_store.batch_size', 32)
        
        # 检索器扩展配置
        enhancement_config = config.get('vector_store.retriever_enhancement', {})
        self.default_topk_multiplier = enhancement_config.get('topk_multiplier', 3.0)
        self.must_have_terms_penalty = enhancement_config.get('must_have_terms_penalty', 0.6)
        self.entity_boost_factor = enhancement_config.get('entity_boost_factor', 1.2)
        self.predicate_boost_factor = enhancement_config.get('predicate_boost_factor', 1.15)
        self.enable_filter_logging = enhancement_config.get('enable_filter_logging', True)
        
        # 默认参数
        self.default_must_have_terms = enhancement_config.get('default_must_have_terms', [])
        self.default_boost_entities = enhancement_config.get('default_boost_entities', [])
        self.default_boost_predicates = enhancement_config.get('default_boost_predicates', [])
        
        # 数据存储
        self.atomic_notes = []  # 存储原子笔记
        self.note_embeddings = None  # 存储嵌入
        self.note_id_to_index = {}  # 笔记ID到索引的映射
        self.index_to_note_id = {}  # 索引到笔记ID的映射
        
        # 存储路径
        default_path = config.get('storage.vector_store_path')
        if not default_path:
            # 使用临时目录避免在项目根目录创建data文件夹
            import tempfile
            default_path = os.path.join(tempfile.gettempdir(), 'anorag_vector_store')
        self.data_dir = default_path
        FileUtils.ensure_dir(self.data_dir)
        
        # 批处理器
        self.batch_processor = BatchProcessor(
            batch_size=self.batch_size,
            use_gpu=config.get('performance.use_gpu', True)
        )
        
        # BM25 回退机制组件
        self.bm25_enabled = config.get('vector_store.bm25_fallback.enabled', True)
        self.bm25_k1 = config.get('vector_store.bm25_fallback.k1', 1.2)
        self.bm25_b = config.get('vector_store.bm25_fallback.b', 0.75)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.processed_texts = []
        
        # 混合检索器（延迟初始化）
        self.hybrid_searcher = None
        self.hybrid_config = config.get('vector_store.hybrid_search', {})
        self.enable_hybrid_search = self.hybrid_config.get('enable_hybrid_search', False) and HYBRID_SEARCH_AVAILABLE
        
        if self.enable_hybrid_search:
            try:
                self.hybrid_searcher = create_hybrid_searcher(config)
                logger.info("Hybrid searcher initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid searcher: {e}")
                self.enable_hybrid_search = False
        
        # 检索保障器（延迟初始化）
        self.retrieval_guardrail = None
        self.guardrail_config = config.get('hybrid_search.retrieval_guardrail', {})
        self.enable_guardrail = self.guardrail_config.get('enabled', True) and GUARDRAIL_AVAILABLE
        
        if self.enable_guardrail:
            try:
                self.retrieval_guardrail = create_retrieval_guardrail(config)
                logger.info("Retrieval guardrail initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize retrieval guardrail: {e}")
                self.enable_guardrail = False
        
        # 模型一致性检查
        self._validate_embedding_consistency()
        
        logger.info("VectorRetriever initialized with BM25 fallback support")
        logger.info(f"Enhancement config: topk_multiplier={self.default_topk_multiplier}, entity_boost={self.entity_boost_factor}, predicate_boost={self.predicate_boost_factor}")
        if self.enable_guardrail:
            logger.info(f"Guardrail enabled with level: {self.guardrail_config.get('level', 'moderate')}")
    
    def build_index(self, atomic_notes: List[Dict[str, Any]], 
                   force_rebuild: bool = False,
                   save_index: bool = True) -> bool:
        """构建向量索引"""
        if not atomic_notes:
            logger.warning("No atomic notes provided for indexing")
            return False
        
        try:
            logger.info(f"Building vector index for {len(atomic_notes)} atomic notes")
            
            # 检查是否需要重建
            if not force_rebuild and self._can_load_existing_index(atomic_notes):
                logger.info("Loading existing index")
                return True
            
            # 存储原子笔记
            self.atomic_notes = atomic_notes
            self._build_id_mappings()
            
            # 生成嵌入
            logger.info("Generating embeddings for atomic notes")
            self.note_embeddings = self.embedding_manager.encode_atomic_notes(
                atomic_notes, 
                include_metadata=True
            )
            
            if self.note_embeddings.size == 0:
                logger.error("Failed to generate embeddings")
                return False
            
            # 创建向量索引
            logger.info("Creating vector index")
            if not self.vector_index.create_index():
                logger.error("Failed to create vector index")
                return False
            
            # 添加向量到索引
            note_ids = np.array([i for i in range(len(atomic_notes))], dtype=np.int64)
            if not self.vector_index.add_vectors(self.note_embeddings, note_ids):
                logger.error("Failed to add vectors to index")
                return False
            
            # 构建 BM25 索引（如果启用）
            if self.bm25_enabled:
                logger.info("Building BM25 index for fallback retrieval")
                self._build_bm25_index(atomic_notes)
            
            # 构建混合检索索引（如果启用）
            if self.enable_hybrid_search and self.hybrid_searcher:
                logger.info("Building hybrid search index")
                try:
                    self.hybrid_searcher.build_index(atomic_notes)
                except Exception as e:
                    logger.warning(f"Failed to build hybrid search index: {e}")
                    self.enable_hybrid_search = False
            
            # 保存索引和数据
            if save_index:
                self._save_index_data()
            
            logger.info(f"Vector index built successfully with {len(atomic_notes)} notes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            return False
    
    def search(self, queries: List[str], 
              top_k: Optional[int] = None,
              similarity_threshold: Optional[float] = None,
              include_metadata: bool = True) -> List[List[Dict[str, Any]]]:
        """搜索相似的原子笔记"""
        if not queries:
            return []
        
        if not self.atomic_notes or self.vector_index.total_vectors == 0:
            logger.warning("Vector index is empty")
            return [[] for _ in queries]
        
        # 参数验证和默认值设置
        top_k = top_k or self.top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        try:
            logger.info(f"Searching for {len(queries)} queries, top_k={top_k}")
            
            # 生成查询嵌入
            query_embeddings = self.embedding_manager.encode_queries(queries)
            
            if query_embeddings.size == 0:
                logger.error("Failed to generate query embeddings")
                return [[] for _ in queries]
            
            # 执行向量搜索
            search_results = self.vector_index.search(
                query_embeddings, 
                top_k=top_k
            )
            
            # 处理搜索结果
            final_results = []
            
            # 确保search_results是列表的列表
            if len(queries) == 1 and isinstance(search_results, list) and \
               len(search_results) > 0 and isinstance(search_results[0], dict):
                search_results = [search_results]
            
            for query_idx, query in enumerate(queries):
                query_results = []
                
                if query_idx < len(search_results):
                    for result in search_results[query_idx]:
                        # 过滤低相似度结果
                        if result.get('similarity', 0) < similarity_threshold:
                            continue
                        
                        # 获取原子笔记
                        note_index = result['index']
                        if note_index < len(self.atomic_notes):
                            note = self.atomic_notes[note_index].copy()
                            
                            # 添加检索信息
                            retrieval_info = {
                                'similarity': result['similarity'],
                                'score': result['score'],
                                'rank': result['rank'],
                                'query': query,
                                'retrieval_method': 'vector_search'
                            }
                            
                            if include_metadata:
                                note['retrieval_info'] = retrieval_info
                            else:
                                # 只保留核心信息，但保留paragraph_idxs字段
                                note = {
                                    'note_id': note.get('note_id'),
                                    'content': note.get('content'),
                                    'paragraph_idxs': note.get('paragraph_idxs', []),
                                    'retrieval_info': retrieval_info
                                }
                            
                            query_results.append(note)
                
                final_results.append(query_results)
            
            # 记录搜索统计
            total_results = sum(len(results) for results in final_results)
            logger.info(f"Search completed: {total_results} results for {len(queries)} queries")
            
            return final_results
        
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return [[] for _ in queries]
    
    def hybrid_search(self, queries: List[str],
                    top_k: Optional[int] = None,
                    similarity_threshold: Optional[float] = None,
                    include_metadata: bool = True,
                    **kwargs) -> List[List[Dict[str, Any]]]:
        """使用混合检索进行搜索"""
        if not self.enable_hybrid_search or not self.hybrid_searcher:
            logger.warning("Hybrid search not available, falling back to vector search")
            return self.search(queries, top_k, similarity_threshold, include_metadata)
        
        try:
            logger.info(f"Performing hybrid search for {len(queries)} queries")
            
            # 使用混合检索器
            results = self.hybrid_searcher.search(
                queries=queries,
                documents=self.atomic_notes,
                top_k=top_k or self.top_k,
                similarity_threshold=similarity_threshold or self.similarity_threshold,
                **kwargs
            )
            
            # 格式化结果
            formatted_results = []
            for query_idx, query_results in enumerate(results):
                formatted_query_results = []
                for result in query_results:
                    if include_metadata:
                        formatted_query_results.append(result)
                    else:
                        # 只保留核心信息，但保留paragraph_idxs字段
                        core_result = {
                            'note_id': result.get('note_id'),
                            'content': result.get('content'),
                            'paragraph_idxs': result.get('paragraph_idxs', []),
                            'retrieval_info': result.get('retrieval_info')
                        }
                        formatted_query_results.append(core_result)
                formatted_results.append(formatted_query_results)
            
            logger.info(f"Hybrid search completed: {sum(len(r) for r in formatted_results)} total results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            logger.info("Falling back to vector search")
            return self.search(queries, top_k, similarity_threshold, include_metadata)
    
    def hybrid_search_single(self, query: str,
                           top_k: Optional[int] = None,
                           similarity_threshold: Optional[float] = None,
                           include_metadata: bool = True,
                           **kwargs) -> List[Dict[str, Any]]:
        """单个查询的混合检索"""
        results = self.hybrid_search([query], top_k, similarity_threshold, include_metadata, **kwargs)
        return results[0] if results else []
    
    def search_single(self, query: str, 
                     top_k: Optional[int] = None,
                     similarity_threshold: Optional[float] = None,
                     include_metadata: bool = True) -> List[Dict[str, Any]]:
        """搜索单个查询"""
        results = self.search([query], top_k, similarity_threshold, include_metadata)
        return results[0] if results else []
    
    def retrieve(self, query: str,
                top_k: Optional[int] = None,
                similarity_threshold: Optional[float] = None,
                filter_fn: Optional[callable] = None,
                must_have_terms: Optional[List[str]] = None,
                boost_entities: Optional[List[str]] = None,
                boost_predicates: Optional[List[str]] = None,
                topk_multiplier: float = 3.0,
                include_metadata: bool = True,
                enable_guardrail: Optional[bool] = None) -> List[Dict[str, Any]]:
        """增强的向量检索器，支持过滤和相关性提升功能
        
        Args:
            query: 查询字符串
            top_k: 最终返回的结果数量
            similarity_threshold: 相似度阈值
            filter_fn: 自定义过滤函数，用于筛选候选结果
            must_have_terms: 必含关键词列表，未包含的候选将降权处理
            boost_entities: 需要提升权重的实体列表，匹配的候选将获得加分
            boost_predicates: 需要提升权重的谓词列表，匹配的候选将获得加分
            topk_multiplier: 初始检索倍数，用于扩大候选池
            include_metadata: 是否包含元数据
            enable_guardrail: 是否启用检索保障器，None时使用默认配置
            
        Returns:
            增强后的检索结果列表
        """
        if not query:
            return []
            
        if not self.atomic_notes or self.vector_index.total_vectors == 0:
            logger.warning("Vector index is empty")
            return []
            
        top_k = top_k or self.top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        try:
            logger.info(f"Enhanced retrieval for query: '{query[:50]}...', top_k={top_k}")
            
            # 第1阶段：向量检索 - 获取 top_k*topk_multiplier 数量的候选结果
            effective_multiplier = topk_multiplier if topk_multiplier is not None else self.default_topk_multiplier
            expanded_top_k = int(top_k * effective_multiplier)
            candidates = self.search_single(
                query=query,
                top_k=expanded_top_k,
                similarity_threshold=0.0,  # 暂时不过滤，后续统一处理
                include_metadata=include_metadata
            )
            
            if not candidates:
                logger.info("No candidates found in vector search")
                return []
                
            logger.info(f"Found {len(candidates)} candidates from vector search")
            
            # 第2阶段：过滤阶段 - 应用 filter_fn 函数
            if filter_fn:
                filtered_candidates = []
                for candidate in candidates:
                    try:
                        if filter_fn(candidate):
                            filtered_candidates.append(candidate)
                    except Exception as e:
                        logger.warning(f"Filter function failed for candidate: {e}")
                        # 过滤函数出错时保留候选
                        filtered_candidates.append(candidate)
                candidates = filtered_candidates
                logger.info(f"After filtering: {len(candidates)} candidates remain")
            
            # 第3阶段：相关性调整
            enhanced_candidates = []
            
            for candidate in candidates:
                # 获取候选文本内容
                content = candidate.get('content', '')
                if isinstance(content, dict):
                    # 如果content是字典，尝试获取文本字段
                    content = content.get('text', '') or content.get('content', '') or str(content)
                elif not isinstance(content, str):
                    content = str(content)
                
                content_lower = content.lower()
                
                # 获取原始相似度分数
                original_similarity = candidate.get('retrieval_info', {}).get('similarity', 0.0)
                adjusted_similarity = original_similarity
                
                # 处理必含关键词 - 降权处理
                if must_have_terms:
                    has_required_terms = any(
                        term.lower() in content_lower 
                        for term in must_have_terms
                    )
                    if not has_required_terms:
                        adjusted_similarity *= self.must_have_terms_penalty
                        if self.enable_filter_logging:
                            logger.debug(f"Downweighted candidate (missing required terms): {adjusted_similarity:.3f}")
                
                # 处理提升实体 - 加分处理
                entity_boost_applied = False
                if boost_entities:
                    matched_entities = [
                        entity for entity in boost_entities
                        if entity.lower() in content_lower
                    ]
                    if matched_entities:
                        adjusted_similarity *= self.entity_boost_factor
                        entity_boost_applied = True
                        if self.enable_filter_logging:
                            logger.debug(f"Boosted candidate (matched entities: {matched_entities}): {adjusted_similarity:.3f}")
                
                # 处理提升谓词 - 加分处理
                predicate_boost_applied = False
                if boost_predicates:
                    matched_predicates = [
                        predicate for predicate in boost_predicates
                        if predicate.lower() in content_lower
                    ]
                    if matched_predicates:
                        adjusted_similarity *= self.predicate_boost_factor
                        predicate_boost_applied = True
                        if self.enable_filter_logging:
                            logger.debug(f"Boosted candidate (matched predicates: {matched_predicates}): {adjusted_similarity:.3f}")
                
                # 更新候选的相似度分数
                candidate_copy = candidate.copy()
                if 'retrieval_info' in candidate_copy:
                    candidate_copy['retrieval_info'] = candidate_copy['retrieval_info'].copy()
                    candidate_copy['retrieval_info']['similarity'] = adjusted_similarity
                    candidate_copy['retrieval_info']['original_similarity'] = original_similarity
                    
                    # 记录调整信息
                    adjustments = []
                    if must_have_terms:
                        has_terms = any(term.lower() in content_lower for term in must_have_terms)
                        if not has_terms:
                            adjustments.append('downweighted_missing_terms')
                    if entity_boost_applied:
                        matched = [e for e in boost_entities if e.lower() in content_lower]
                        adjustments.append(f'boosted_entities_{len(matched)}')
                    if predicate_boost_applied:
                        matched = [p for p in boost_predicates if p.lower() in content_lower]
                        adjustments.append(f'boosted_predicates_{len(matched)}')
                    
                    candidate_copy['retrieval_info']['adjustments'] = adjustments
                
                enhanced_candidates.append(candidate_copy)
            
            # 第4阶段：最终处理 - 按相似度排序并应用阈值过滤
            # 过滤低相似度结果
            filtered_final = [
                candidate for candidate in enhanced_candidates
                if candidate.get('retrieval_info', {}).get('similarity', 0.0) >= similarity_threshold
            ]
            
            # 按调整后的相似度排序
            filtered_final.sort(
                key=lambda x: x.get('retrieval_info', {}).get('similarity', 0.0),
                reverse=True
            )
            
            # 返回 top_k 个结果
            final_results = filtered_final[:top_k]
            
            logger.info(f"Enhanced retrieval completed: {len(final_results)} results returned")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            # 回退到基础搜索
            logger.info("Falling back to basic search")
            return self.search_single(query, top_k, similarity_threshold, include_metadata)
    
    def add_notes(self, new_notes: List[Dict[str, Any]], 
                 rebuild_index: bool = False) -> bool:
        """添加新的原子笔记"""
        if not new_notes:
            return True
        
        try:
            logger.info(f"Adding {len(new_notes)} new notes to index")
            
            if rebuild_index or not self.atomic_notes:
                # 重建整个索引
                all_notes = self.atomic_notes + new_notes
                return self.build_index(all_notes, force_rebuild=True)
            else:
                # 增量添加
                start_index = len(self.atomic_notes)
                self.atomic_notes.extend(new_notes)
                
                # 更新ID映射
                self._build_id_mappings()
                
                # 生成新笔记的嵌入
                new_embeddings = self.embedding_manager.encode_atomic_notes(
                    new_notes, 
                    include_metadata=True
                )
                
                if new_embeddings.size == 0:
                    logger.error("Failed to generate embeddings for new notes")
                    return False
                
                # 添加到索引
                new_ids = np.array([i for i in range(start_index, start_index + len(new_notes))], dtype=np.int64)
                if not self.vector_index.add_vectors(new_embeddings, new_ids):
                    logger.error("Failed to add new vectors to index")
                    return False
                
                # 更新嵌入矩阵
                if self.note_embeddings is not None:
                    self.note_embeddings = np.vstack([self.note_embeddings, new_embeddings])
                else:
                    self.note_embeddings = new_embeddings
                
                logger.info(f"Successfully added {len(new_notes)} notes to index")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add notes: {e}")
            return False
    
    def remove_notes(self, note_ids: List[str]) -> bool:
        """移除指定的原子笔记"""
        if not note_ids:
            return True
        
        try:
            logger.info(f"Removing {len(note_ids)} notes from index")
            
            # 找到要移除的索引
            indices_to_remove = []
            for note_id in note_ids:
                if note_id in self.note_id_to_index:
                    indices_to_remove.append(self.note_id_to_index[note_id])
            
            if not indices_to_remove:
                logger.warning("No matching notes found to remove")
                return True
            
            # 由于FAISS索引的限制，我们需要重建索引
            logger.info("Rebuilding index after note removal")
            
            # 移除笔记
            indices_to_remove = sorted(indices_to_remove, reverse=True)
            for idx in indices_to_remove:
                if idx < len(self.atomic_notes):
                    del self.atomic_notes[idx]
            
            # 重建索引
            return self.build_index(self.atomic_notes, force_rebuild=True)
            
        except Exception as e:
            logger.error(f"Failed to remove notes: {e}")
            return False
    
    def update_note(self, note_id: str, updated_note: Dict[str, Any]) -> bool:
        """更新指定的原子笔记"""
        try:
            if note_id not in self.note_id_to_index:
                logger.warning(f"Note {note_id} not found")
                return False
            
            # 更新笔记
            note_index = self.note_id_to_index[note_id]
            self.atomic_notes[note_index] = updated_note
            
            # 重新生成该笔记的嵌入
            new_embedding = self.embedding_manager.encode_atomic_notes(
                [updated_note], 
                include_metadata=True
            )
            
            if new_embedding.size == 0:
                logger.error("Failed to generate embedding for updated note")
                return False
            
            # 更新嵌入矩阵
            if self.note_embeddings is not None:
                self.note_embeddings[note_index] = new_embedding[0]
            
            # 由于FAISS的限制，需要重建索引
            logger.info(f"Rebuilding index after updating note {note_id}")
            return self.build_index(self.atomic_notes, force_rebuild=True)
            
        except Exception as e:
            logger.error(f"Failed to update note {note_id}: {e}")
            return False
    
    def get_note_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取原子笔记"""
        if note_id in self.note_id_to_index:
            note_index = self.note_id_to_index[note_id]
            if note_index < len(self.atomic_notes):
                return self.atomic_notes[note_index]
        return None
    
    def get_notes_by_ids(self, note_ids: List[str]) -> List[Dict[str, Any]]:
        """根据ID列表获取原子笔记"""
        notes = []
        for note_id in note_ids:
            note = self.get_note_by_id(note_id)
            if note:
                notes.append(note)
        return notes
    
    def get_similar_notes(self, note_id: str, 
                         top_k: int = 10,
                         exclude_self: bool = True) -> List[Dict[str, Any]]:
        """获取与指定笔记相似的其他笔记"""
        note = self.get_note_by_id(note_id)
        if not note:
            return []
        
        # 使用笔记内容进行搜索
        content = note.get('content', '')
        if not content:
            return []
        
        results = self.search_single(content, top_k=top_k + (1 if exclude_self else 0))
        
        # 排除自身
        if exclude_self:
            results = [r for r in results if r.get('note_id') != note_id]
        
        return results[:top_k]
    
    def _build_id_mappings(self):
        """构建ID映射"""
        self.note_id_to_index = {}
        self.index_to_note_id = {}
        
        for idx, note in enumerate(self.atomic_notes):
            note_id = note.get('note_id')
            if note_id:
                self.note_id_to_index[note_id] = idx
                self.index_to_note_id[idx] = note_id
    
    def _can_load_existing_index(self, atomic_notes: List[Dict[str, Any]]) -> bool:
        """检查是否可以加载现有索引"""
        # 检查索引文件是否存在
        index_files = [f for f in os.listdir(self.data_dir) if f.endswith('.faiss')]
        if not index_files:
            return False
        
        # 检查数据文件是否存在
        data_file = os.path.join(self.data_dir, 'atomic_notes.json')
        if not os.path.exists(data_file):
            return False
        
        # 加载现有数据
        try:
            existing_notes = FileUtils.read_json(data_file)
            
            # 简单比较：检查笔记数量和第一个笔记的ID
            if len(existing_notes) != len(atomic_notes):
                return False
            
            if existing_notes and atomic_notes:
                if existing_notes[0].get('note_id') != atomic_notes[0].get('note_id'):
                    return False
            
            # 加载索引
            index_file = index_files[0]
            if self.vector_index.load_index(index_file):
                self.atomic_notes = existing_notes
                self._build_id_mappings()
                
                # 加载嵌入
                embedding_file = os.path.join(self.data_dir, 'note_embeddings.npz')
                if os.path.exists(embedding_file):
                    data = np.load(embedding_file)
                    self.note_embeddings = data['embeddings']
                
                logger.info(f"Loaded existing index with {len(self.atomic_notes)} notes")
                return True
            
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
        
        return False
    
    def _save_index_data(self):
        """保存索引和数据"""
        try:
            # 保存向量索引
            self.vector_index.save_index()
            
            # 保存原子笔记
            notes_file = os.path.join(self.data_dir, 'atomic_notes.json')
            FileUtils.write_json(self.atomic_notes, notes_file)
            
            # 保存嵌入
            if self.note_embeddings is not None:
                embedding_file = os.path.join(self.data_dir, 'note_embeddings.npz')
                np.savez_compressed(embedding_file, embeddings=self.note_embeddings)
            
            # 保存ID映射
            mapping_file = os.path.join(self.data_dir, 'id_mappings.json')
            FileUtils.write_json({
                'note_id_to_index': self.note_id_to_index,
                'index_to_note_id': self.index_to_note_id
            }, mapping_file)
            
            logger.info("Index data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save index data: {e}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        stats = {
            'total_notes': len(self.atomic_notes),
            'embedding_dim': self.embedding_manager.embedding_dim,
            'model_name': self.embedding_manager.model_name,
            'index_stats': self.vector_index.get_index_stats(),
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold
        }
        
        if self.note_embeddings is not None:
            stats['embedding_stats'] = self.embedding_manager.get_embedding_stats(self.note_embeddings)
        
        return stats
    
    def optimize_retrieval(self, test_queries: List[str], 
                          ground_truth: List[List[str]],
                          target_recall: float = 0.9) -> Dict[str, Any]:
        """优化检索参数"""
        if not test_queries or not ground_truth:
            logger.warning("No test data provided for optimization")
            return {}
        
        logger.info(f"Optimizing retrieval parameters with {len(test_queries)} test queries")
        
        # 转换ground truth为索引
        gt_indices = []
        for gt_note_ids in ground_truth:
            indices = []
            for note_id in gt_note_ids:
                if note_id in self.note_id_to_index:
                    indices.append(self.note_id_to_index[note_id])
            gt_indices.append(indices)
        
        # 生成测试查询的嵌入
        query_embeddings = self.embedding_manager.encode_queries(test_queries)
        
        # 优化向量索引参数
        index_optimization = self.vector_index.optimize_search_params(
            query_embeddings, 
            np.array(gt_indices, dtype=object),
            target_recall
        )
        
        # 测试不同的相似度阈值
        best_threshold = self.similarity_threshold
        best_f1 = 0.0
        
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # 执行搜索
            results = self.search(test_queries, similarity_threshold=threshold)
            
            # 计算F1分数
            f1_score = self._calculate_f1_score(results, ground_truth)
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = threshold
        
        # 更新最佳参数
        self.similarity_threshold = best_threshold
        
        optimization_result = {
            'index_optimization': index_optimization,
            'best_similarity_threshold': best_threshold,
            'best_f1_score': best_f1,
            'target_recall': target_recall
        }
        
        logger.info(f"Optimization completed: threshold={best_threshold}, F1={best_f1:.3f}")
        return optimization_result
    
    def _calculate_f1_score(self, search_results: List[List[Dict[str, Any]]], 
                           ground_truth: List[List[str]]) -> float:
        """计算F1分数"""
        if not search_results or not ground_truth:
            return 0.0
        
        total_f1 = 0.0
        valid_queries = 0
        
        for i, (results, gt) in enumerate(zip(search_results, ground_truth)):
            if not gt:
                continue
            
            retrieved_ids = set(result.get('note_id') for result in results)
            true_ids = set(gt)
            
            if len(retrieved_ids) == 0 and len(true_ids) == 0:
                f1 = 1.0
            elif len(retrieved_ids) == 0 or len(true_ids) == 0:
                f1 = 0.0
            else:
                precision = len(retrieved_ids & true_ids) / len(retrieved_ids)
                recall = len(retrieved_ids & true_ids) / len(true_ids)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
            
            total_f1 += f1
            valid_queries += 1
        
        return total_f1 / valid_queries if valid_queries > 0 else 0.0
    
    def clear_index(self):
        """清空索引"""
        self.atomic_notes = []
        self.note_embeddings = None
        self.note_id_to_index = {}
        self.index_to_note_id = {}
        
        if self.vector_index:
            self.vector_index.reset_index()
        
        logger.info("Vector index cleared")
    
    def cleanup(self):
        """清理资源"""
        if self.embedding_manager:
            self.embedding_manager.cleanup()
        
        if self.vector_index:
            self.vector_index.cleanup()
        
        if self.hybrid_searcher:
            try:
                self.hybrid_searcher.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup hybrid searcher: {e}")
        
        if self.retrieval_guardrail:
            try:
                # RetrievalGuardrail没有cleanup方法，但可以清理统计信息
                self.retrieval_guardrail.stats.clear()
            except Exception as e:
                logger.warning(f"Failed to cleanup retrieval guardrail: {e}")
        
        logger.info("VectorRetriever cleanup completed")
    
    def _validate_embedding_consistency(self):
        """验证嵌入模型一致性"""
        try:
            # 检查EmbeddingManager的模型一致性
            is_consistent, details = self.embedding_manager.validate_model_consistency()
            
            if is_consistent:
                logger.info("Embedding model consistency validated")
            else:
                logger.warning(f"Embedding model consistency check failed: {details}")
                
                # 根据配置决定是否严格处理
                model_config = config.get('model_consistency', {})
                if model_config.get('violation_handling', {}).get('strict_mode', False):
                    raise RuntimeError(f"Model consistency violation: {details}")
                    
        except Exception as e:
            logger.error(f"Error during embedding consistency validation: {e}")
            # 根据配置决定是否继续
            model_config = config.get('model_consistency', {})
            if model_config.get('violation_handling', {}).get('strict_mode', False):
                raise
    
    def get_embedding_model_info(self) -> Dict[str, Any]:
        """获取嵌入模型信息，包括一致性检查结果"""
        try:
            return self.embedding_manager.get_model_info()
        except Exception as e:
            logger.error(f"Error getting embedding model info: {e}")
            return {'error': str(e)}
    
    def _build_bm25_index(self, atomic_notes: List[Dict[str, Any]]) -> None:
        """构建 BM25 索引用于回退检索"""
        try:
            # 提取文本内容
            texts = []
            for note in atomic_notes:
                content = note.get('content', '')
                if content:
                    # 简单的文本预处理
                    processed_text = self._preprocess_text(content)
                    texts.append(processed_text)
                else:
                    texts.append('')
            
            self.processed_texts = texts
            
            # 构建 TF-IDF 矩阵（用于 BM25 计算）
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=10000,
                ngram_range=(1, 2)
            )
            
            if texts:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                logger.info(f"BM25 index built with {len(texts)} documents")
            else:
                logger.warning("No valid texts found for BM25 indexing")
                
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_enabled = False
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本用于 BM25 索引"""
        # 移除特殊字符，保留字母数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def _bm25_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """使用 BM25 进行回退检索"""
        if not self.bm25_enabled or not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []
        
        try:
            # 预处理查询
            processed_query = self._preprocess_text(query)
            
            # 将查询转换为 TF-IDF 向量
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # 计算余弦相似度
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # 获取 top_k 结果
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.atomic_notes) and similarities[idx] > 0:
                    note = self.atomic_notes[idx].copy()
                    note['retrieval_info'] = {
                        'similarity': float(similarities[idx]),
                        'score': float(similarities[idx]),
                        'rank': len(results) + 1,
                        'query': query,
                        'retrieval_method': 'bm25_fallback'
                    }
                    results.append(note)
            
            logger.info(f"BM25 fallback search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def search_with_namespace_fallback(self, queries: List[str], dataset: str, qid: str,
                                     top_k: Optional[int] = None,
                                     similarity_threshold: Optional[float] = None,
                                     include_metadata: bool = True) -> List[List[Dict[str, Any]]]:
        """带命名空间过滤和 BM25 回退的搜索方法"""
        from utils.dataset_guard import filter_notes_by_namespace
        
        # 首先进行常规向量搜索
        vector_results = self.search(queries, top_k, similarity_threshold, include_metadata)
        
        final_results = []
        for query_idx, (query, query_results) in enumerate(zip(queries, vector_results)):
            # 应用命名空间过滤
            filtered_results = filter_notes_by_namespace(query_results, dataset, qid)
            
            # 如果命名空间过滤后没有结果，尝试 BM25 回退
            if not filtered_results and self.bm25_enabled:
                logger.info(f"No namespace matches for query {query_idx + 1}, trying BM25 fallback")
                bm25_results = self._bm25_search(query, top_k or self.top_k)
                
                # 对 BM25 结果也应用命名空间过滤
                filtered_bm25_results = filter_notes_by_namespace(bm25_results, dataset, qid)
                
                if filtered_bm25_results:
                    logger.info(f"BM25 fallback found {len(filtered_bm25_results)} namespace matches")
                    filtered_results = filtered_bm25_results
                else:
                    logger.warning(f"No results found for query '{query}' in namespace {dataset}/{qid}")
            
            final_results.append(filtered_results)
        
        return final_results