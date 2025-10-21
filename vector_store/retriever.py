import os
import copy
import numpy as np
from typing import List, Dict, Any, Optional
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

    def __init__(self,
                 embedding_manager: Optional[EmbeddingManager] = None,
                 vector_index: Optional[VectorIndex] = None,
                 retrieval_mode: str = "auto"):
        # 初始化组件
        self.embedding_manager: Optional[EmbeddingManager] = embedding_manager
        self.vector_index = vector_index or VectorIndex(
            getattr(self.embedding_manager, 'embedding_dim', None)
        )

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
        self._bm25_config_enabled = self.bm25_enabled
        self.bm25_k1 = config.get('vector_store.bm25_fallback.k1', 1.2)
        self.bm25_b = config.get('vector_store.bm25_fallback.b', 0.75)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.processed_texts = []

        # 检索模式
        self.retrieval_mode = "auto"
        self._vector_search_enabled = True
        
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
        self.guardrail_config = config.get('retrieval_guardrail', {})
        self.enable_guardrail = self.guardrail_config.get('enabled', True) and GUARDRAIL_AVAILABLE
        
        if self.enable_guardrail:
            try:
                self.retrieval_guardrail = create_retrieval_guardrail(config)
                logger.info("Retrieval guardrail initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize retrieval guardrail: {e}")
                self.enable_guardrail = False
        
        # 模型一致性检查
        self.configure_mode(retrieval_mode)
        self._validate_embedding_consistency()
        
        logger.info("VectorRetriever initialized with BM25 fallback support")
        logger.info(
            "Enhancement config: topk_multiplier={}, entity_boost={}, predicate_boost={}",
            self.default_topk_multiplier,
            self.entity_boost_factor,
            self.predicate_boost_factor,
        )
        if self.enable_guardrail:
            logger.info(f"Guardrail enabled with level: {self.guardrail_config.get('level', 'moderate')}")
    
    def configure_mode(self, mode: str) -> None:
        """配置检索模式（bm25 / dense / hybrid / auto）"""
        if not mode:
            mode = "auto"
        mode = mode.lower()
        if mode not in {"auto", "bm25", "dense", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        self.retrieval_mode = mode

        if mode == "bm25":
            self._vector_search_enabled = False
            self.bm25_enabled = True
            self.enable_hybrid_search = False
        elif mode == "dense":
            self._vector_search_enabled = True
            self.bm25_enabled = False
            self.enable_hybrid_search = False
        elif mode == "hybrid":
            self._vector_search_enabled = True
            self.bm25_enabled = True
            if HYBRID_SEARCH_AVAILABLE and self.hybrid_searcher is None:
                try:
                    self.hybrid_searcher = create_hybrid_searcher(config)
                except Exception as exc:
                    logger.warning(f"Failed to configure hybrid searcher: {exc}")
                    self.hybrid_searcher = None
            self.enable_hybrid_search = HYBRID_SEARCH_AVAILABLE and self.hybrid_searcher is not None
        else:  # auto
            self._vector_search_enabled = True
            self.bm25_enabled = self._bm25_config_enabled
            # 保持初始化时的混合检索配置

    def set_embedding_model(self, model_name: Optional[str]) -> None:
        """覆盖嵌入模型配置"""
        if not model_name:
            return
        manager = self._get_embedding_manager()
        if hasattr(manager, 'set_model'):
            manager.set_model(model_name)
        else:
            manager.model_name = model_name
        self._validate_embedding_consistency()

    def _get_embedding_manager(self) -> EmbeddingManager:
        if self.embedding_manager is None:
            self.embedding_manager = EmbeddingManager()
            self._validate_embedding_consistency()
        return self.embedding_manager

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

            if self.retrieval_mode == "bm25":
                logger.info("Retriever configured for BM25-only mode")
                self.note_embeddings = np.array([])
                self._build_bm25_index(atomic_notes)
                if save_index:
                    self._save_index_data()
                logger.info(f"BM25 index built successfully for {len(atomic_notes)} notes")
                return True

            # 生成嵌入，失败则回退到BM25/TF-IDF
            if self._vector_search_enabled:
                try:
                    logger.info("Generating embeddings for atomic notes")
                    manager = self._get_embedding_manager()
                    self.note_embeddings = manager.encode_atomic_notes(
                        atomic_notes,
                        include_metadata=True
                    )
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}. Falling back to BM25/TF-IDF only mode")
                    self.note_embeddings = np.array([])
            else:
                self.note_embeddings = np.array([])

            if self.note_embeddings.size == 0:
                if self.bm25_enabled:
                    logger.info("Using BM25/TF-IDF index only (no vector index)")
                    self._build_bm25_index(atomic_notes)
                    if save_index:
                        # 仍然保存必要的数据以便后续加载
                        self._save_index_data()
                    logger.info(f"BM25/TF-IDF index built successfully for {len(atomic_notes)} notes")
                    return True
                else:
                    logger.error("Failed to generate embeddings and BM25 fallback disabled")
                    return False

            # 创建向量索引
            logger.info("Creating vector index")
            if not self.vector_index.create_index():
                logger.error("Failed to create vector index")
                # 回退到BM25
                if self.bm25_enabled:
                    logger.info("Falling back to BM25/TF-IDF index only")
                    self._build_bm25_index(atomic_notes)
                    if save_index:
                        self._save_index_data()
                    return True
                return False

            # 添加向量到索引
            note_ids = np.array([i for i in range(len(atomic_notes))], dtype=np.int64)
            if not self.vector_index.add_vectors(self.note_embeddings, note_ids):
                logger.error("Failed to add vectors to index")
                # 回退到BM25
                if self.bm25_enabled:
                    logger.info("Falling back to BM25/TF-IDF index only")
                    self._build_bm25_index(atomic_notes)
                    if save_index:
                        self._save_index_data()
                    return True
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
            # 顶层异常仍尝试BM25回退
            if self.bm25_enabled:
                try:
                    self._build_bm25_index(atomic_notes)
                    if save_index:
                        self._save_index_data()
                    logger.info("Recovered by building BM25/TF-IDF index")
                    return True
                except Exception as e2:
                    logger.error(f"BM25/TF-IDF fallback also failed: {e2}")
            return False

    # 新增：BM25/TF-IDF 索引构建
    def _build_bm25_index(self, atomic_notes: List[Dict[str, Any]]):
        self.processed_texts = []
        for note in atomic_notes:
            content = note.get('content', '')
            if isinstance(content, dict):
                content = content.get('text', '') or content.get('content', '') or str(content)
            elif not isinstance(content, str):
                content = str(content)
            # 简单预处理
            self.processed_texts.append(content)
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_texts)
        logger.info(f"TF-IDF matrix built: shape={self.tfidf_matrix.shape}")
    
    # 新增：BM25/TF-IDF 搜索回退
    def _bm25_search(self, queries: List[str],
                     top_k: Optional[int],
                     similarity_threshold: Optional[float],
                     include_metadata: bool) -> List[List[Dict[str, Any]]]:
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            logger.warning("BM25/TF-IDF index not available")
            return [[] for _ in queries]
        
        top_k = top_k or self.top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        results_all: List[List[Dict[str, Any]]] = []
        for q in queries:
            try:
                q_vec = self.tfidf_vectorizer.transform([q])
                sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
                ranked = np.argsort(sims)[::-1]
                query_results: List[Dict[str, Any]] = []
                rank = 0
                for idx in ranked[:top_k]:
                    score = float(sims[idx])
                    if score < similarity_threshold:
                        continue
                    if idx < len(self.atomic_notes):
                        note = self.atomic_notes[idx].copy()
                        retrieval_info = {
                            'similarity': score,
                            'score': score,
                            'rank': rank,
                            'query': q,
                            'retrieval_method': 'bm25_tfidf'
                        }
                        rank += 1
                        if include_metadata:
                            note['retrieval_info'] = retrieval_info
                            query_results.append(note)
                        else:
                            query_results.append({
                                'note_id': note.get('note_id'),
                                'content': note.get('content'),
                                'paragraph_idxs': note.get('paragraph_idxs', []),
                                'retrieval_info': retrieval_info
                            })
                results_all.append(query_results)
            except Exception as e:
                logger.warning(f"BM25/TF-IDF search failed for query: {e}")
                results_all.append([])
        logger.info(f"BM25/TF-IDF search completed: {sum(len(r) for r in results_all)} total results")
        return results_all
    
    def _bm25_available(self) -> bool:
        return self.bm25_enabled and self.tfidf_vectorizer is not None and self.tfidf_matrix is not None

    def search(self, queries: List[str],
              top_k: Optional[int] = None,
              similarity_threshold: Optional[float] = None,
              include_metadata: bool = True) -> List[List[Dict[str, Any]]]:
        """搜索相似的原子笔记"""
        if not queries:
            return []

        if not self.atomic_notes:
            logger.warning("No atomic notes available")
            return [[] for _ in queries]

        # 参数验证和默认值设置
        top_k = top_k or self.top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold

        if self.retrieval_mode == "bm25":
            if self._bm25_available():
                return self._bm25_search(queries, top_k, similarity_threshold, include_metadata)
            logger.warning("BM25 index not available for BM25 mode")
            return [[] for _ in queries]

        if self.retrieval_mode == "hybrid":
            return self._hybrid_search(queries, top_k, similarity_threshold, include_metadata)

        return self._vector_search(
            queries,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            include_metadata=include_metadata,
            allow_bm25_fallback=self.bm25_enabled and self.retrieval_mode != "dense"
        )

    def _vector_search(self,
                       queries: List[str],
                       top_k: int,
                       similarity_threshold: float,
                       include_metadata: bool,
                       allow_bm25_fallback: bool) -> List[List[Dict[str, Any]]]:
        # 当向量索引不可用时，根据需要回退BM25
        if not self._vector_search_enabled or self.vector_index.total_vectors == 0:
            if allow_bm25_fallback and self._bm25_available():
                logger.info("Vector index unavailable; using BM25/TF-IDF fallback search")
                return self._bm25_search(queries, top_k, similarity_threshold, include_metadata)
            logger.warning("Vector index is empty and vector search disabled")
            return [[] for _ in queries]

        try:
            logger.info(f"Searching for {len(queries)} queries, top_k={top_k}")

            # 生成查询嵌入
            manager = self._get_embedding_manager()
            query_embeddings = manager.encode_queries(queries)

            if query_embeddings.size == 0:
                logger.error("Failed to generate query embeddings")
                if allow_bm25_fallback and self._bm25_available():
                    logger.info("Falling back to BM25/TF-IDF search for queries")
                    return self._bm25_search(queries, top_k, similarity_threshold, include_metadata)
                return [[] for _ in queries]

            # 执行向量搜索
            search_results = self.vector_index.search(
                query_embeddings,
                top_k=top_k
            )

            # 处理搜索结果
            final_results = []

            # 确保search_results是列表的列表
            if (len(queries) == 1 and isinstance(search_results, list) and
                len(search_results) > 0 and isinstance(search_results[0], dict)):
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
            # 失败时回退到BM25/TF-IDF
            if allow_bm25_fallback and self._bm25_available():
                logger.info("Falling back to BM25/TF-IDF search after vector search failure")
                return self._bm25_search(queries, top_k, similarity_threshold, include_metadata)
            return [[] for _ in queries]

    def _hybrid_search(self,
                       queries: List[str],
                       top_k: int,
                       similarity_threshold: float,
                       include_metadata: bool) -> List[List[Dict[str, Any]]]:
        vector_results = self._vector_search(
            queries,
            top_k=top_k,
            similarity_threshold=0.0,
            include_metadata=True,
            allow_bm25_fallback=False
        )

        bm25_results: List[List[Dict[str, Any]]] = [[] for _ in queries]
        if self._bm25_available():
            bm25_results = self._bm25_search(
                queries,
                top_k=top_k,
                similarity_threshold=0.0,
                include_metadata=True
            )
        elif not self._vector_search_enabled:
            logger.warning("Hybrid mode requested but BM25 index unavailable")

        combined_results: List[List[Dict[str, Any]]] = []

        for idx, query in enumerate(queries):
            dense_list = vector_results[idx] if idx < len(vector_results) else []
            bm25_list = bm25_results[idx] if idx < len(bm25_results) else []
            merged = self._merge_hybrid_results(
                dense_list,
                bm25_list,
                top_k,
                similarity_threshold,
                include_metadata,
                query
            )
            combined_results.append(merged)

        return combined_results

    def _merge_hybrid_results(self,
                              dense_results: List[Dict[str, Any]],
                              bm25_results: List[Dict[str, Any]],
                              top_k: int,
                              similarity_threshold: float,
                              include_metadata: bool,
                              query: str) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}

        def _note_id(entry: Dict[str, Any]) -> Optional[str]:
            return entry.get('note_id') or entry.get('chunk_id')

        def _copy_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
            return copy.deepcopy(entry)

        def _extract_similarity(entry: Dict[str, Any]) -> float:
            info = entry.get('retrieval_info', {})
            if isinstance(info, dict):
                return float(info.get('similarity', info.get('score', 0.0)))
            return 0.0

        for entry in dense_results:
            note_id = _note_id(entry)
            if note_id is None:
                continue
            cloned = _copy_entry(entry)
            info = cloned.setdefault('retrieval_info', {})
            base_sim = _extract_similarity(cloned)
            info['similarity'] = base_sim
            info['sources'] = {'vector': base_sim}
            info['retrieval_method'] = 'hybrid'
            info['query'] = info.get('query', query)
            merged[note_id] = cloned

        for entry in bm25_results:
            note_id = _note_id(entry)
            if note_id is None:
                continue
            bm25_sim = _extract_similarity(entry)
            if note_id in merged:
                info = merged[note_id].setdefault('retrieval_info', {})
                sources = info.setdefault('sources', {})
                sources['bm25'] = bm25_sim
                info['similarity'] = max(info.get('similarity', 0.0), bm25_sim)
                info['query'] = info.get('query', query)
            else:
                cloned = _copy_entry(entry)
                info = cloned.setdefault('retrieval_info', {})
                info['similarity'] = bm25_sim
                info['sources'] = {'bm25': bm25_sim}
                info['retrieval_method'] = 'hybrid'
                info['query'] = info.get('query', query)
                merged[note_id] = cloned

        filtered = [note for note in merged.values()
                    if note.get('retrieval_info', {}).get('similarity', 0.0) >= similarity_threshold]

        filtered.sort(key=lambda x: x.get('retrieval_info', {}).get('similarity', 0.0), reverse=True)

        if not include_metadata:
            compact_results = []
            for note in filtered[:top_k]:
                compact_results.append({
                    'note_id': note.get('note_id'),
                    'content': note.get('content'),
                    'paragraph_idxs': note.get('paragraph_idxs', []),
                    'retrieval_info': note.get('retrieval_info')
                })
            return compact_results

        return filtered[:top_k]

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
        """增强的向量检索器，支持过滤和相关性提升功能"""
        if not query:
            return []
        
        if not self.atomic_notes:
            logger.warning("No atomic notes available")
            return []
        
        top_k = top_k or self.top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        try:
            logger.info(f"Enhanced retrieval for query: '{query[:50]}...', top_k={top_k}")
            
            # 第1阶段：检索（向量或BM25回退） - 获取 top_k*topk_multiplier 数量的候选结果
            effective_multiplier = topk_multiplier if topk_multiplier is not None else self.default_topk_multiplier
            expanded_top_k = int(top_k * effective_multiplier)
            candidates = self.search_single(
                query=query,
                top_k=expanded_top_k,
                similarity_threshold=0.0,  # 暂时不过滤，后续统一处理
                include_metadata=include_metadata
            )
            
            if not candidates:
                logger.info("No candidates found in initial search")
                return []
            
            logger.info(f"Found {len(candidates)} candidates from initial search")
            
            # ... existing code ...
        except Exception as e:
            logger.error(f"Enhanced retrieval failed for query '{query[:50]}...': {e}")
            return []

    def _validate_embedding_consistency(self) -> None:
        """校验嵌入维度与索引设置一致性，避免运行时错误"""
        try:
            emb_dim = getattr(self.embedding_manager, 'embedding_dim', None)
            if emb_dim is None:
                emb_dim = config.get('vector_store.dimension', 768)
            index_dim = getattr(self.vector_index, 'embedding_dim', None)
            if index_dim is None:
                index_dim = emb_dim
            if emb_dim != index_dim:
                logger.warning(f"Embedding dim ({emb_dim}) != index dim ({index_dim}); aligning index.")
                self.vector_index.embedding_dim = emb_dim
        except Exception as e:
            logger.warning(f"Embedding consistency check failed: {e}")

    def _build_id_mappings(self) -> None:
        """建立 note_id 与内部索引的双向映射"""
        self.note_id_to_index = {}
        self.index_to_note_id = {}
        for idx, note in enumerate(self.atomic_notes):
            note_id = note.get('note_id') or idx
            self.note_id_to_index[note_id] = idx
            self.index_to_note_id[idx] = note_id

    def _index_meta_path(self) -> str:
        return os.path.join(self.data_dir, 'vector_index_meta.json')

    def _save_index_data(self) -> None:
        """持久化必要的检索数据用于后续加载（原子笔记+映射+索引元数据）"""
        try:
            FileUtils.ensure_dir(self.data_dir)
            notes_to_save = []
            for note in self.atomic_notes:
                notes_to_save.append({
                    'note_id': note.get('note_id'),
                    'content': note.get('content'),
                    'paragraph_idxs': note.get('paragraph_idxs', []),
                    'doc_name': note.get('doc_name'),
                    'chunk_id': note.get('chunk_id')
                })
            FileUtils.write_json(notes_to_save, os.path.join(self.data_dir, 'atomic_notes.json'))
            FileUtils.write_json({
                'note_id_to_index': self.note_id_to_index,
                'index_to_note_id': self.index_to_note_id
            }, os.path.join(self.data_dir, 'note_mappings.json'))
            try:
                self.vector_index.save_index()
            except Exception as e:
                logger.warning(f"Saving FAISS index failed: {e}")
            meta = {
                'total_notes': len(self.atomic_notes),
                'bm25_enabled': self.bm25_enabled,
                'has_tfidf': self.tfidf_matrix is not None,
            }
            FileUtils.write_json(meta, self._index_meta_path())
        except Exception as e:
            logger.warning(f"Failed to save index data: {e}")

    def _can_load_existing_index(self, atomic_notes: List[Dict[str, Any]]) -> bool:
        """若磁盘上已有兼容数据，则加载以避免重建"""
        try:
            meta_path = self._index_meta_path()
            if not os.path.exists(meta_path):
                return False
            meta = FileUtils.read_json(meta_path)
            if meta.get('total_notes') != len(atomic_notes):
                return False
            loaded = self.vector_index.load_index()
            if loaded:
                notes_path = os.path.join(self.data_dir, 'atomic_notes.json')
                maps_path = os.path.join(self.data_dir, 'note_mappings.json')
                if os.path.exists(notes_path):
                    self.atomic_notes = FileUtils.read_json(notes_path) or atomic_notes
                else:
                    self.atomic_notes = atomic_notes
                if os.path.exists(maps_path):
                    maps = FileUtils.read_json(maps_path) or {}
                    self.note_id_to_index = maps.get('note_id_to_index', {})
                    self.index_to_note_id = maps.get('index_to_note_id', {})
                else:
                    self._build_id_mappings()
                if meta.get('bm25_enabled'):
                    try:
                        self._build_bm25_index(self.atomic_notes)
                    except Exception as e:
                        logger.warning(f"Rebuilding TF-IDF failed: {e}")
                logger.info("Existing index and data loaded successfully")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load existing index data: {e}")
            return False

    def search_single(self, query: str, top_k: int = None, similarity_threshold: float = None,
                      include_metadata: bool = True) -> List[Dict[str, Any]]:
        """对单一查询执行检索，便于增强检索流水线使用"""
        results_list = self.search([query], top_k=top_k, similarity_threshold=similarity_threshold,
                                   include_metadata=include_metadata)
        if not results_list:
            return []
        return results_list[0]