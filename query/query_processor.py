from typing import List, Dict, Any, Optional
from loguru import logger
import os
import numpy as np
import concurrent.futures
import threading

# 导入增强的日志功能
from utils.logging_utils import (
    StructuredLogger, log_performance, log_operation,
    log_retrieval_metrics, log_diversity_metrics, log_path_aware_metrics
)

from llm import OllamaClient, LocalLLM
from vector_store import VectorRetriever, EnhancedRecallOptimizer
from graph.graph_builder import GraphBuilder
from graph.graph_index import GraphIndex
from graph.graph_retriever import GraphRetriever
from utils.context_scheduler import ContextScheduler, MultiHopContextScheduler
from utils.context_dispatcher import ContextDispatcher
from utils.dataset_guard import filter_notes_by_namespace, assert_namespace_or_raise, DatasetNamespaceError
from utils.bm25_search import build_bm25_corpus, bm25_scores
from retrieval.path_aware_ranker import PathAwareRanker, create_path_aware_ranker
from config import config

# 导入多跳推理组件
from graph.multi_hop_query_processor import MultiHopQueryProcessor

# 导入子问题分解组件
from .subquestion_planner import SubQuestionPlanner
from .evidence_merger import EvidenceMerger
from retrieval.query_planner import LLMBasedRewriter
from retrieval.diversity_scheduler import DiversityScheduler
from graph.entity_inverted_index import EntityInvertedIndex

class QueryProcessor:
    """High level query processing pipeline."""

    def __init__(
        self,
        atomic_notes: List[Dict[str, Any]],
        embeddings=None,
        graph_file: Optional[str] = None,
        vector_index_file: Optional[str] = None,
        llm: Optional[LocalLLM] = None,
    ):
        # Query rewriter functionality has been removed
        self.vector_retriever = VectorRetriever()
        if vector_index_file and os.path.exists(vector_index_file):
            try:
                # adjust storage directories
                dir_path = os.path.dirname(vector_index_file)
                self.vector_retriever.data_dir = dir_path
                self.vector_retriever.vector_index.index_dir = dir_path
                # load index directly
                self.vector_retriever.vector_index.load_index(os.path.basename(vector_index_file))
                self.vector_retriever.atomic_notes = atomic_notes
                self.vector_retriever._build_id_mappings()
                # load stored embeddings if available
                embed_file = os.path.join(dir_path, "note_embeddings.npz")
                if os.path.exists(embed_file):
                    try:
                        loaded = np.load(embed_file)
                        self.vector_retriever.note_embeddings = loaded["embeddings"]
                    except Exception as e:
                        logger.warning(f"Failed to load stored embeddings: {e}")
                logger.info(f"Loaded vector index from {vector_index_file}")
            except Exception as e:
                logger.error(f"Failed to load vector index: {e}, rebuilding")
                self.vector_retriever.build_index(atomic_notes)
        else:
            self.vector_retriever.build_index(atomic_notes)
        if embeddings is None:
            embeddings = self.vector_retriever.note_embeddings

        builder = GraphBuilder(llm=llm)
        graph = None
        if graph_file and os.path.exists(graph_file):
            self.graph_index = GraphIndex()
            try:
                self.graph_index.load_index(graph_file)
                logger.info(f"Loaded graph from {graph_file}")
            except Exception as e:
                logger.error(f"Failed to load graph index: {e}, rebuilding")
                graph = builder.build_graph(atomic_notes, embeddings)
                self.graph_index.build_index(graph, atomic_notes, embeddings)
        else:
            graph = builder.build_graph(atomic_notes, embeddings)
            self.graph_index = GraphIndex()
            self.graph_index.build_index(graph, atomic_notes, embeddings)

        self.multi_hop_enabled = config.get('multi_hop.enabled', False)
        if self.multi_hop_enabled:
            self.multi_hop_processor = MultiHopQueryProcessor(
                atomic_notes,
                embeddings,
                graph_file=graph_file if graph_file and os.path.exists(graph_file) else None,
                graph_index=self.graph_index,
            )

        # 初始化图谱检索器（无论是否使用multi_hop都需要）
        self.graph_retriever = GraphRetriever(self.graph_index, k_hop=config.get('context_dispatcher.k_hop', 2))
        
        # 初始化调度器
        self.use_context_dispatcher = config.get('context_dispatcher.enabled', True)
        
        if self.use_context_dispatcher:
            # 使用新的结构增强上下文调度器
            self.context_dispatcher = ContextDispatcher(config)
            logger.info("Using ContextDispatcher for structure-enhanced retrieval")
        else:
            # 使用原有的调度器
            if self.multi_hop_enabled:
                self.scheduler = MultiHopContextScheduler()
            else:
                self.scheduler = ContextScheduler()
            logger.info("Using legacy ContextScheduler")

        self.recall_optimization_enabled = config.get('vector_store.recall_optimization.enabled', True)
        if self.multi_hop_enabled:
            self.recall_optimizer = EnhancedRecallOptimizer(self.vector_retriever, self.multi_hop_processor)
        else:
            self.recall_optimizer = EnhancedRecallOptimizer(self.vector_retriever, self.graph_retriever)

        self.ollama = OllamaClient()
        self.atomic_notes = atomic_notes
        
        # 初始化子问题分解组件
        self.use_subquestion_decomposition = config.get('query.use_subquestion_decomposition', False)
        if self.use_subquestion_decomposition:
            self.subquestion_planner = SubQuestionPlanner(llm_client=self.ollama)
            self.evidence_merger = EvidenceMerger()
            self.parallel_retrieval = config.get('query.subquestion.parallel_retrieval', True)
            logger.info("Sub-question decomposition enabled")
        else:
            self.subquestion_planner = None
            self.evidence_merger = None
            self.parallel_retrieval = False
            logger.info("Sub-question decomposition disabled")
        
        # 初始化命名空间守卫配置
        self.namespace_guard_enabled = config.get('dataset_guard.enabled', True)
        self.bm25_fallback_enabled = config.get('dataset_guard.bm25_fallback', True)
        logger.info(f"Dataset namespace guard: {'enabled' if self.namespace_guard_enabled else 'disabled'}")
        logger.info(f"BM25 fallback: {'enabled' if self.bm25_fallback_enabled else 'disabled'}")
        
        # 初始化混合检索配置（强制启用）
        self.hybrid_search_enabled = config.get('hybrid_search.enabled', True)  # 强制默认启用
        self.fusion_method = config.get('hybrid_search.fusion_method', 'linear')
        
        # 获取融合权重配置
        if self.fusion_method == 'linear':
            linear_config = config.get('hybrid_search.linear', {})
            self.vector_weight = linear_config.get('vector_weight', 0.7)
            self.bm25_weight = linear_config.get('bm25_weight', 0.3)
            self.path_weight = linear_config.get('path_weight', 0.3)
        else:  # rrf
            rrf_config = config.get('hybrid_search.rrf', {})
            self.rrf_k = rrf_config.get('k', 60)
            self.vector_weight = rrf_config.get('vector_weight', 1.0)
            self.bm25_weight = rrf_config.get('bm25_weight', 1.0)
            self.path_weight = rrf_config.get('path_weight', 1.0)
        
        # BM25配置
        bm25_config = config.get('hybrid_search.bm25', {})
        self.bm25_k1 = bm25_config.get('k1', 1.2)
        self.bm25_b = bm25_config.get('b', 0.75)
        self.bm25_corpus_field = bm25_config.get('corpus_field', 'title_raw_span')
        
        # 初始化PathAwareRanker
        path_config = config.get('hybrid_search.path_aware', {})
        self.path_aware_enabled = path_config.get('enabled', True)
        if self.path_aware_enabled:
            try:
                ranker_config = config.get('path_aware_ranker', {})
                self.path_aware_ranker = create_path_aware_ranker(ranker_config)
                logger.info("PathAwareRanker initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PathAwareRanker: {e}")
                self.path_aware_enabled = False
                self.path_aware_ranker = None
        else:
            self.path_aware_ranker = None
        
        # 检索守卫配置
        guardrail_config = config.get('hybrid_search.retrieval_guardrail', {})
        self.retrieval_guardrail_enabled = guardrail_config.get('enabled', True)
        self.must_have_terms_config = guardrail_config.get('must_have_terms', {})
        self.boost_entities_config = guardrail_config.get('boost_entities', {})
        self.boost_predicates_config = guardrail_config.get('boost_predicates', {})
        self.predicate_mappings = guardrail_config.get('predicate_mappings', {})
        
        # 失败兜底策略配置
        fallback_config = config.get('hybrid_search.fallback', {})
        self.fallback_enabled = fallback_config.get('enabled', True)
        self.fallback_sparse_boost = fallback_config.get('sparse_boost_factor', 1.5)
        self.fallback_query_rewrite_enabled = fallback_config.get('query_rewrite_enabled', True)
        self.fallback_max_retries = fallback_config.get('max_retries', 2)
        
        # 初始化LLM查询改写器（用于失败兜底）
        if self.fallback_query_rewrite_enabled:
            try:
                self.llm_rewriter = LLMBasedRewriter(config)
                logger.info(f"LLM query rewriter initialized for fallback strategy: {'enabled' if self.llm_rewriter.enabled else 'disabled'}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM rewriter: {e}")
                self.llm_rewriter = None
        else:
            self.llm_rewriter = None
        
        # 初始化DiversityScheduler（PathAware + Diversity联动）
        diversity_config = config.get('diversity_scheduler', {})
        self.diversity_scheduler_enabled = diversity_config.get('enabled', True)
        if self.diversity_scheduler_enabled:
            try:
                self.diversity_scheduler = DiversityScheduler(diversity_config)
                logger.info(f"DiversityScheduler initialized with evidence quota: {self.diversity_scheduler.config.get('enable_evidence_quota', False)}")
            except Exception as e:
                logger.error(f"Failed to initialize DiversityScheduler: {e}")
                self.diversity_scheduler_enabled = False
                self.diversity_scheduler = None
        else:
            self.diversity_scheduler = None
        
        # 预构建 BM25 语料库（强制启用）
        self.bm25_corpus = None
        try:
            # 根据配置选择语料字段
            if self.bm25_corpus_field == 'title_raw_span':
                corpus_func = lambda note: f"{note.get('title', '')} {note.get('raw_span', '')}".strip()
            elif self.bm25_corpus_field == 'content':
                corpus_func = lambda note: note.get('content', '')
            else:  # summary
                corpus_func = lambda note: note.get('summary', note.get('content', ''))
            
            self.bm25_corpus = build_bm25_corpus(atomic_notes, corpus_func)
            logger.info(f"Built BM25 corpus for hybrid search with {len(atomic_notes)} notes using field: {self.bm25_corpus_field}")
        except Exception as e:
            logger.error(f"Failed to build BM25 corpus: {e}")
            self.hybrid_search_enabled = False
        
        logger.info(f"Hybrid search: {'enabled' if self.hybrid_search_enabled else 'disabled'}")
        logger.info(f"Fusion method: {self.fusion_method}")
        if self.hybrid_search_enabled:
            logger.info(f"Weights - Vector: {self.vector_weight}, BM25: {self.bm25_weight}, Path: {self.path_weight}")
            logger.info(f"PathAware ranking: {'enabled' if self.path_aware_enabled else 'disabled'}")
            logger.info(f"Retrieval guardrail: {'enabled' if self.retrieval_guardrail_enabled else 'disabled'}")
        
        # 加载新的配置项
        # 二跳扩展配置
        two_hop_config = config.get('hybrid_search.two_hop_expansion', {})
        self.two_hop_enabled = two_hop_config.get('enabled', True)
        self.top_m_candidates = two_hop_config.get('top_m_candidates', 20)
        self.entity_extraction_method = two_hop_config.get('entity_extraction_method', 'rule_based')
        self.target_predicates = two_hop_config.get('target_predicates', ['founded_by', 'located_in', 'member_of', 'works_for', 'part_of', 'instance_of'])
        self.max_second_hop_candidates = two_hop_config.get('max_second_hop_candidates', 15)
        self.merge_strategy = two_hop_config.get('merge_strategy', 'weighted')
        
        # 段域过滤配置
        section_filter_config = config.get('hybrid_search.section_filtering', {})
        self.section_filtering_enabled = section_filter_config.get('enabled', True)
        self.section_filter_rule = section_filter_config.get('filter_rule', 'main_entity_related')
        self.fallback_to_lexical = section_filter_config.get('fallback_to_lexical', True)
        
        # 词面保底配置
        lexical_fallback_config = config.get('hybrid_search.lexical_fallback', {})
        self.lexical_fallback_enabled = lexical_fallback_config.get('enabled', True)
        self.must_have_terms_sources = lexical_fallback_config.get('must_have_terms_sources', ['main_entity', 'predicate_stems'])
        self.miss_penalty = lexical_fallback_config.get('miss_penalty', 0.6)
        self.blacklist_penalty = lexical_fallback_config.get('blacklist_penalty', 0.5)
        self.noise_threshold = lexical_fallback_config.get('noise_threshold', 0.20)
        
        # 命名空间过滤配置（四阶段）
        namespace_config = config.get('hybrid_search.namespace_filtering', {})
        self.namespace_filtering_enabled = namespace_config.get('enabled', True)
        self.namespace_filter_stages = namespace_config.get('stages', ['initial_recall', 'post_fusion', 'post_two_hop', 'final_scheduling'])
        self.same_namespace_bm25_fallback = namespace_config.get('same_namespace_bm25_fallback', True)
        self.strict_mode = namespace_config.get('strict_mode', True)
        
        logger.info(f"Two-hop expansion: {'enabled' if self.two_hop_enabled else 'disabled'}")
        logger.info(f"Section filtering: {'enabled' if self.section_filtering_enabled else 'disabled'}")
        logger.info(f"Lexical fallback: {'enabled' if self.lexical_fallback_enabled else 'disabled'}")
        logger.info(f"Namespace filtering stages: {self.namespace_filter_stages}")
        
        # 初始化实体倒排索引
        self.entity_inverted_index = EntityInvertedIndex()
        try:
            self.entity_inverted_index.build_index(atomic_notes)
            logger.info(f"Built entity inverted index with {len(self.entity_inverted_index.entity_to_notes)} entities")
        except Exception as e:
            logger.error(f"Failed to build entity inverted index: {e}")
            self.entity_inverted_index = None
        
        # 初始化结构化日志记录器
        self.structured_logger = StructuredLogger("QueryProcessor")
        self.structured_logger.info("QueryProcessor initialized successfully", 
                                  notes_count=len(atomic_notes),
                                  hybrid_search=self.hybrid_search_enabled,
                                  path_aware=self.path_aware_enabled,
                                  diversity_scheduler=self.diversity_scheduler_enabled,
                                  multi_hop=self.multi_hop_enabled)

    @log_performance("QueryProcessor.process")
    def process(self, query: str, dataset: Optional[str] = None, qid: Optional[str] = None) -> Dict[str, Any]:
        # 记录查询开始
        self.structured_logger.info("Starting query processing", 
                                   query_length=len(query),
                                   dataset=dataset,
                                   qid=qid,
                                   subquestion_enabled=self.use_subquestion_decomposition)
        
        # Check if sub-question decomposition is enabled
        if self.use_subquestion_decomposition:
            result = self._process_with_subquestion_decomposition(query, dataset, qid)
        else:
            result = self._process_traditional(query, dataset, qid)
        
        # 记录查询结果
        self.structured_logger.info("Query processing completed",
                                   results_count=len(result.get('selected_notes', [])),
                                   processing_method="subquestion" if self.use_subquestion_decomposition else "traditional")
        return result
    
    def _fix_entity_extraction_flow(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """修复实体抽取流程，对raw_span再次执行NER与标准化。"""
        try:
            for candidate in candidates:
                raw_span = candidate.get('raw_span', '')
                if raw_span:
                    # 对raw_span执行NER
                    extracted_entities = self._perform_ner_on_text(raw_span)
                    
                    # 标准化实体
                    normalized_entities = self._normalize_entities(extracted_entities)
                    candidate['normalized_entities'] = normalized_entities
                    
                    # 提取并标准化谓词
                    extracted_predicates = self._extract_predicates_from_text(raw_span)
                    normalized_predicates = self._normalize_predicates(extracted_predicates)
                    candidate['normalized_predicates'] = normalized_predicates
                    
                    logger.debug(f"Fixed entity extraction for candidate: entities={len(normalized_entities)}, predicates={len(normalized_predicates)}")
            
            logger.info(f"Entity extraction flow fixed for {len(candidates)} candidates")
            
        except Exception as e:
            logger.error(f"Failed to fix entity extraction flow: {e}")
        
        return candidates
    
    def _perform_ner_on_text(self, text: str) -> List[str]:
        """对文本执行命名实体识别。"""
        import re
        entities = []
        
        # 基于规则的NER（可以后续替换为更高级的NER模型）
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 人名、地名等
            r'\b[A-Z]{2,}\b',  # 缩写组织名
            r'\b\d{4}\b',  # 年份
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 日期
            r'\b[A-Z][a-z]*\s+(?:Inc|Corp|Ltd|LLC|Co)\b',  # 公司名
            r'\b(?:University|College|Institute)\s+of\s+[A-Z][a-z]+\b',  # 大学名
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # 去重并过滤
        entities = list(set(entities))
        entities = [e for e in entities if len(e.strip()) > 2]  # 过滤太短的实体
        
        return entities
    
    def _normalize_entities(self, entities: List[str]) -> List[str]:
        """标准化实体名称。"""
        normalized = []
        
        for entity in entities:
            # 基本的标准化处理
            normalized_entity = entity.strip()
            
            # 移除多余的空格
            normalized_entity = ' '.join(normalized_entity.split())
            
            # 统一大小写（保持首字母大写）
            if normalized_entity:
                words = normalized_entity.split()
                normalized_entity = ' '.join([word.capitalize() for word in words])
                normalized.append(normalized_entity)
        
        return list(set(normalized))  # 去重
    
    def _extract_predicates_from_text(self, text: str) -> List[str]:
        """从文本中提取谓词。"""
        import re
        predicates = []
        
        # 常见的关系谓词模式
        predicate_patterns = [
            r'\b(?:founded|established|created|built)\s+(?:by|in)\b',
            r'\b(?:located|situated|based)\s+(?:in|at|on)\b',
            r'\b(?:member|part)\s+of\b',
            r'\b(?:worked|employed)\s+(?:at|for|in)\b',
            r'\b(?:born|died)\s+(?:in|on|at)\b',
            r'\b(?:graduated|studied)\s+(?:from|at)\b',
            r'\b(?:married|divorced)\s+(?:to|from)\b',
            r'\b(?:owns|owned|operates)\b',
            r'\b(?:leads|led|manages|managed)\b',
            r'\b(?:instance|type|kind)\s+of\b',
        ]
        
        text_lower = text.lower()
        for pattern in predicate_patterns:
            matches = re.findall(pattern, text_lower)
            predicates.extend(matches)
        
        # 提取动词作为潜在谓词
        verb_pattern = r'\b\w+(?:ed|ing|s)\b'  # 简单的动词识别
        verbs = re.findall(verb_pattern, text_lower)
        
        # 过滤常见动词
        common_verbs = {'is', 'was', 'are', 'were', 'has', 'have', 'had', 'will', 'would', 'could', 'should'}
        filtered_verbs = [v for v in verbs if v not in common_verbs and len(v) > 3]
        
        predicates.extend(filtered_verbs[:5])  # 只取前5个动词
        
        return list(set(predicates))
    
    def _normalize_predicates(self, predicates: List[str]) -> List[str]:
        """标准化谓词。"""
        normalized = []
        
        # 谓词映射表
        predicate_mapping = {
            'founded': 'founded_by',
            'established': 'founded_by',
            'created': 'created_by',
            'located': 'located_in',
            'situated': 'located_in',
            'based': 'located_in',
            'member': 'member_of',
            'part': 'part_of',
            'worked': 'worked_at',
            'employed': 'worked_at',
            'born': 'born_in',
            'died': 'died_in',
            'graduated': 'graduated_from',
            'studied': 'studied_at',
            'married': 'married_to',
            'owns': 'owns',
            'owned': 'owns',
            'operates': 'operates',
            'leads': 'leads',
            'led': 'leads',
            'manages': 'manages',
            'managed': 'manages',
            'instance': 'instance_of',
            'type': 'instance_of',
        }
        
        for predicate in predicates:
            # 清理谓词
            clean_predicate = predicate.strip().lower()
            
            # 移除介词和冠词
            clean_predicate = re.sub(r'\s+(?:of|in|at|on|by|to|from|for)\b', '', clean_predicate)
            clean_predicate = clean_predicate.strip()
            
            # 应用映射
            if clean_predicate in predicate_mapping:
                normalized.append(predicate_mapping[clean_predicate])
            elif clean_predicate and len(clean_predicate) > 2:
                normalized.append(clean_predicate)
        
        return list(set(normalized))  # 去重
    
    def _apply_noise_threshold_filtering(self, candidates: List[Dict[str, Any]], must_have_terms: List[str] = None) -> List[Dict[str, Any]]:
        """应用噪声阈值过滤，丢弃相似度小于0.20且不满足must_have_terms的候选。"""
        if not candidates:
            return candidates
        
        filtered_candidates = []
        noise_threshold = self.noise_threshold  # 0.20
        
        for candidate in candidates:
            # 获取候选的相似度分数
            similarity = candidate.get('final_score', candidate.get('final_base_score', candidate.get('similarity', 0.0)))
            
            # 检查是否满足噪声阈值
            if similarity >= noise_threshold:
                # 相似度足够高，直接保留
                filtered_candidates.append(candidate)
            elif must_have_terms and self._satisfies_must_have_terms(candidate, must_have_terms):
                # 相似度较低但满足必需词汇要求，也保留
                filtered_candidates.append(candidate)
                logger.debug(f"Kept low-similarity candidate due to must_have_terms: {similarity:.3f}")
            else:
                # 相似度低且不满足必需词汇，过滤掉
                logger.debug(f"Filtered out noisy candidate: similarity={similarity:.3f}, noise_threshold={noise_threshold}")
        
        logger.info(f"Noise threshold filtering: {len(candidates)} -> {len(filtered_candidates)} candidates (threshold={noise_threshold})")
        return filtered_candidates
    
    def _generate_enhanced_guardrail_params(self, query: str) -> tuple:
        """生成增强的检索守卫参数，包含段域过滤和词面保底功能。"""
        must_have_terms = []
        boost_entities = []
        boost_predicates = []
        
        try:
            # 基础的must_have_terms生成
            if 'main_entity' in self.must_have_terms_sources:
                # 从查询中提取主实体词元
                main_entities = self._extract_main_entities_from_query(query)
                must_have_terms.extend(main_entities)
            
            if 'predicate_stems' in self.must_have_terms_sources:
                # 提取谓词词干
                predicate_stems = self._extract_predicate_stems_from_query(query)
                must_have_terms.extend(predicate_stems)
            
            # 生成boost参数
            boost_entities = self._extract_boost_entities_from_query(query)
            boost_predicates = self._extract_boost_predicates_from_query(query)
            
            logger.debug(f"Enhanced guardrail params - must_have: {must_have_terms}, boost_entities: {boost_entities}, boost_predicates: {boost_predicates}")
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced guardrail params: {e}")
        
        return must_have_terms, boost_entities, boost_predicates
    
    def _extract_main_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取主实体词元。"""
        # 简单的实体提取逻辑，可以后续增强
        import re
        # 提取大写开头的词作为潜在实体
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        return [entity.lower() for entity in entities]
    
    def _extract_predicate_stems_from_query(self, query: str) -> List[str]:
        """从查询中提取谓词词干。"""
        # 简单的谓词词干提取
        predicate_keywords = ['founded', 'located', 'member', 'work', 'part', 'instance', 'born', 'died', 'created']
        stems = []
        query_lower = query.lower()
        for keyword in predicate_keywords:
            if keyword in query_lower:
                stems.append(keyword)
        return stems
    
    def _extract_boost_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取需要boost的实体。"""
        return self._extract_main_entities_from_query(query)
    
    def _extract_boost_predicates_from_query(self, query: str) -> List[str]:
        """从查询中提取需要boost的谓词。"""
        return self._extract_predicate_stems_from_query(query)
    
    @log_performance("QueryProcessor._enhanced_hybrid_search_v2")
    def _enhanced_hybrid_search_v2(self, query: str, candidates: List[Dict[str, Any]], 
                                  must_have_terms: List[str] = None, 
                                  boost_entities: List[str] = None, 
                                  boost_predicates: List[str] = None) -> List[Dict[str, Any]]:
        """增强的混合检索v2，使用新融合公式：final_base = 1.0 * dense + 0.6 * sparse。"""
        if not candidates:
            return []
        
        try:
            # 计算向量相似度
            vector_similarities = self._calculate_vector_similarities(query, candidates)
            
            # 计算BM25分数
            bm25_similarities = self._calculate_bm25_similarities(query, candidates)
            
            # 应用新的融合公式：final_base = 1.0 * dense + 0.6 * sparse
            for i, candidate in enumerate(candidates):
                dense_score = vector_similarities[i] if i < len(vector_similarities) else 0.0
                sparse_score = bm25_similarities[i] if i < len(bm25_similarities) else 0.0
                
                # 新融合公式
                final_base = 1.0 * dense_score + 0.6 * sparse_score
                
                # 应用段域过滤
                if self.section_filtering_enabled:
                    section_penalty = self._apply_section_filtering(candidate, query)
                    final_base *= section_penalty
                
                # 应用词面保底
                if self.lexical_fallback_enabled and must_have_terms:
                    lexical_penalty = self._apply_lexical_fallback(candidate, must_have_terms)
                    final_base *= lexical_penalty
                
                # 应用噪声阈值过滤
                if final_base < self.noise_threshold and not self._satisfies_must_have_terms(candidate, must_have_terms):
                    final_base = 0.0  # 标记为过滤
                
                # 应用boost
                if boost_entities:
                    entity_boost = self._calculate_entity_boost(candidate, boost_entities)
                    final_base *= entity_boost
                
                if boost_predicates:
                    predicate_boost = self._calculate_predicate_boost(candidate, boost_predicates)
                    final_base *= predicate_boost
                
                candidate['final_base_score'] = final_base
                candidate['dense_score'] = dense_score
                candidate['sparse_score'] = sparse_score
            
            # 过滤掉分数为0的候选
            candidates = [c for c in candidates if c.get('final_base_score', 0) > 0]
            
            # 按final_base_score排序
            candidates.sort(key=lambda x: x.get('final_base_score', 0), reverse=True)
            
            logger.debug(f"Enhanced hybrid search v2 completed: {len(candidates)} candidates after filtering and ranking")
            
        except Exception as e:
            logger.error(f"Enhanced hybrid search v2 failed: {e}")
        
        return candidates
    
    def _calculate_bm25_similarities(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """计算BM25相似度分数。"""
        if not self.bm25_corpus:
            return [0.0] * len(candidates)
        
        try:
            # 构建候选文档的语料
            if self.bm25_corpus_field == 'title_raw_span':
                docs = [f"{c.get('title', '')} {c.get('raw_span', '')}".strip() for c in candidates]
            elif self.bm25_corpus_field == 'content':
                docs = [c.get('content', '') for c in candidates]
            else:  # summary
                docs = [c.get('summary', c.get('content', '')) for c in candidates]
            
            # 计算BM25分数
            scores = bm25_scores(query, docs, k1=self.bm25_k1, b=self.bm25_b)
            return scores
            
        except Exception as e:
            logger.error(f"BM25 calculation failed: {e}")
            return [0.0] * len(candidates)
    
    def _apply_section_filtering(self, candidate: Dict[str, Any], query: str) -> float:
        """应用段域过滤规则。"""
        if not self.section_filtering_enabled:
            return 1.0
        
        # 检查是否有section_role标注
        section_role = candidate.get('section_role')
        if section_role:
            # 如果有section_role，检查是否与主实体相关
            if self.section_filter_rule == 'main_entity_related':
                # 简单的相关性检查
                if 'main' in section_role.lower() or 'primary' in section_role.lower():
                    return 1.0
                else:
                    return 0.8  # 轻微惩罚
        elif self.fallback_to_lexical:
            # 如果没有section_role且启用词面保底，返回1.0让词面保底处理
            return 1.0
        else:
            # 没有section_role且不启用词面保底，轻微惩罚
            return 0.9
        
        return 1.0
    
    def _apply_lexical_fallback(self, candidate: Dict[str, Any], must_have_terms: List[str]) -> float:
        """应用词面保底策略。"""
        if not self.lexical_fallback_enabled or not must_have_terms:
            return 1.0
        
        # 检查候选是否包含必需词汇
        content = candidate.get('content', '').lower()
        title = candidate.get('title', '').lower()
        raw_span = candidate.get('raw_span', '').lower()
        
        full_text = f"{title} {content} {raw_span}"
        
        hit_count = 0
        for term in must_have_terms:
            if term.lower() in full_text:
                hit_count += 1
        
        if hit_count == 0:
            return self.miss_penalty  # 未命中任何必需词汇
        elif hit_count < len(must_have_terms) / 2:
            return 0.8  # 命中较少
        else:
            return 1.0  # 命中足够多
    
    def _satisfies_must_have_terms(self, candidate: Dict[str, Any], must_have_terms: List[str]) -> bool:
        """检查候选是否满足必需词汇要求。"""
        if not must_have_terms:
            return True
        
        content = candidate.get('content', '').lower()
        title = candidate.get('title', '').lower()
        raw_span = candidate.get('raw_span', '').lower()
        
        full_text = f"{title} {content} {raw_span}"
        
        for term in must_have_terms:
            if term.lower() in full_text:
                return True
        
        return False
    
    def _calculate_entity_boost(self, candidate: Dict[str, Any], boost_entities: List[str]) -> float:
        """计算实体boost分数。"""
        if not boost_entities:
            return 1.0
        
        content = candidate.get('content', '').lower()
        boost_factor = 1.0
        
        for entity in boost_entities:
            if entity.lower() in content:
                boost_factor *= 1.2  # 每个匹配的实体增加20%
        
        return min(boost_factor, 2.0)  # 最大boost 2倍
    
    def _calculate_predicate_boost(self, candidate: Dict[str, Any], boost_predicates: List[str]) -> float:
        """计算谓词boost分数。"""
        if not boost_predicates:
            return 1.0
        
        content = candidate.get('content', '').lower()
        boost_factor = 1.0
        
        for predicate in boost_predicates:
            if predicate.lower() in content:
                boost_factor *= 1.15  # 每个匹配的谓词增加15%
        
        return min(boost_factor, 1.8)  # 最大boost 1.8倍
    
    def _extract_entities_from_candidates(self, candidates: List[Dict[str, Any]]) -> List[str]:
        """从候选中抽取实体。"""
        entities = set()
        
        try:
            for candidate in candidates:
                # 优先使用entities字段
                if 'entities' in candidate and candidate['entities']:
                    entities.update(candidate['entities'])
                else:
                    # 从raw_span用规则抽取
                    raw_span = candidate.get('raw_span', '')
                    if raw_span:
                        extracted = self._rule_based_entity_extraction(raw_span)
                        entities.update(extracted)
                
                # 从normalized_entities字段抽取（如果存在）
                if 'normalized_entities' in candidate and candidate['normalized_entities']:
                    entities.update(candidate['normalized_entities'])
            
            logger.debug(f"Extracted {len(entities)} unique entities from {len(candidates)} candidates")
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
        
        return list(entities)
    
    def _rule_based_entity_extraction(self, text: str) -> List[str]:
        """基于规则的实体抽取。"""
        import re
        entities = []
        
        # 提取大写开头的词组
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 标准实体模式
            r'\b[A-Z]{2,}\b',  # 缩写
            r'\b\d{4}\b',  # 年份
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return entities
    
    def _perform_second_hop_retrieval(self, entities: List[str], query: str, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        """执行二跳检索 - 图优先策略。"""
        second_hop_notes = []
        bridge_entities = entities[:10]  # 限制桥接实体数量
        
        logger.info(f"Starting second hop retrieval with bridge entities: {bridge_entities}")
        
        try:
            # 1. 使用倒排索引获取二跳候选池
            candidate_pool = self._get_second_hop_candidate_pool(bridge_entities)
            logger.info(f"Second hop candidate pool size: {len(candidate_pool)}")
            
            if not candidate_pool:
                # 2. 触发fallback机制
                logger.info("Second hop candidate pool is empty, triggering fallback")
                candidate_pool = self._second_hop_fallback(bridge_entities, query, dataset)
                logger.info(f"Fallback candidate pool size: {len(candidate_pool)}")
            
            if not candidate_pool:
                logger.warning("No candidates found even after fallback")
                return []
            
            # 3. 命名空间过滤
            if dataset and self.namespace_filtering_enabled:
                filtered_pool = filter_notes_by_namespace(candidate_pool, dataset)
                logger.info(f"After namespace filtering: {len(filtered_pool)} candidates (dropped: {len(candidate_pool) - len(filtered_pool)})")
                candidate_pool = filtered_pool
            
            # 4. 对候选池进行重排
            second_hop_notes = self._rerank_second_hop_candidates(
                candidate_pool, query, bridge_entities
            )
            
            # 5. 限制最终数量
            second_hop_notes = second_hop_notes[:self.max_second_hop_candidates]
            
            logger.info(f"Final second hop results: {len(second_hop_notes)} candidates")
            
        except Exception as e:
            logger.error(f"Second hop retrieval failed: {e}")
        
        return second_hop_notes
    
    def _get_second_hop_candidate_pool(self, bridge_entities: List[str]) -> List[Dict[str, Any]]:
        """使用倒排索引获取二跳候选池。"""
        if not self.entity_inverted_index:
            logger.warning("Entity inverted index not available")
            return []
        
        candidate_pool = []
        seen_ids = set()
        entity_stats = {}
        
        for entity in bridge_entities:
            # 从倒排索引获取候选笔记
            candidates = self.entity_inverted_index.get_candidate_notes(
                [entity], fuzzy_match=True
            )
            
            entity_candidates = 0
            for candidate in candidates:
                note_id = candidate.get('note_id')
                if note_id and note_id not in seen_ids:
                    candidate['bridge_entity'] = entity
                    candidate_pool.append(candidate)
                    seen_ids.add(note_id)
                    entity_candidates += 1
            
            entity_stats[entity] = entity_candidates
        
        # 记录每个桥接实体的贡献
        stats_str = ", ".join([f"{entity}: {count}" for entity, count in entity_stats.items()])
        logger.info(f"Entity contributions to candidate pool: {stats_str}")
        
        return candidate_pool
    
    def _second_hop_fallback(self, bridge_entities: List[str], query: str, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        """二跳检索fallback机制：以桥接实体为查询在全库检索。"""
        logger.info(f"Triggering fallback mechanism for {len(bridge_entities)} bridge entities")
        
        fallback_candidates = []
        seen_ids = set()
        entity_fallback_stats = {}
        
        for entity in bridge_entities:
            # 以桥接实体为查询进行向量检索
            entity_candidates = 0
            try:
                results = self.vector_retriever.search([entity])
                if results and results[0]:
                    for candidate in results[0][:5]:  # 每个实体取前5个结果
                        # 检查候选是否包含该实体
                        if self._candidate_contains_entity(candidate, entity):
                            note_id = candidate.get('note_id')
                            if note_id and note_id not in seen_ids:
                                candidate['bridge_entity'] = entity
                                candidate['fallback_source'] = 'vector_search'
                                fallback_candidates.append(candidate)
                                seen_ids.add(note_id)
                                entity_candidates += 1
                entity_fallback_stats[entity] = entity_candidates
            except Exception as e:
                logger.error(f"Fallback search failed for entity {entity}: {e}")
                entity_fallback_stats[entity] = 0
        
        # 记录fallback统计信息
        total_found = sum(entity_fallback_stats.values())
        successful_entities = sum(1 for count in entity_fallback_stats.values() if count > 0)
        logger.info(f"Fallback completed: {total_found} candidates from {successful_entities}/{len(bridge_entities)} entities")
        
        return fallback_candidates
    
    def _candidate_contains_entity(self, candidate: Dict[str, Any], entity: str) -> bool:
        """检查候选笔记是否包含指定实体。"""
        entity_lower = entity.lower()
        note_id = candidate.get('note_id', 'unknown')
        
        # 检查各个字段
        fields_to_check = ['title', 'content', 'summary', 'raw_span', 'raw_span_evidence']
        for field in fields_to_check:
            if field in candidate and candidate[field]:
                if entity_lower in candidate[field].lower():
                    logger.debug(f"Entity '{entity}' found in {field} of note {note_id}")
                    return True
        
        # 检查实体列表
        entities = candidate.get('entities', [])
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, str) and entity_lower in ent.lower():
                    logger.debug(f"Entity '{entity}' matched with '{ent}' in entities list of note {note_id}")
                    return True
        
        logger.debug(f"Entity '{entity}' not found in note {note_id}")
        return False
    
    def _rerank_second_hop_candidates(self, candidates: List[Dict[str, Any]], query: str, bridge_entities: List[str]) -> List[Dict[str, Any]]:
        """对二跳候选池进行重排。"""
        if not candidates:
            return []
        
        logger.info(f"Reranking {len(candidates)} second hop candidates with {len(bridge_entities)} bridge entities")
        
        # 计算向量相似度
        vector_scores = self._calculate_vector_similarities(query, candidates)
        
        # 计算BM25分数
        bm25_scores_list = self._calculate_bm25_similarities(query, candidates)
        
        # 应用重排逻辑
        scored_candidates = []
        for i, candidate in enumerate(candidates):
            dense_score = vector_scores[i] if i < len(vector_scores) else 0.0
            sparse_score = bm25_scores_list[i] if i < len(bm25_scores_list) else 0.0
            
            # found谓词软过滤（降分而非硬过滤）
            predicate_penalty = self._calculate_predicate_penalty(candidate)
            
            # 桥接实体加分
            bridge_bonus = self._calculate_bridge_entity_bonus(candidate, bridge_entities)
            
            # 应用调整
            dense_score *= predicate_penalty * bridge_bonus
            sparse_score *= predicate_penalty * bridge_bonus
            
            # 路径分数（暂时设为0，后续在path_aware_reranking中计算）
            path_score = 0.0
            
            # 最终打分：final = 1.0 * dense + 0.6 * sparse + 0.3 * path_score
            final_score = 1.0 * dense_score + 0.6 * sparse_score + 0.3 * path_score
            
            candidate['dense_score'] = dense_score
            candidate['sparse_score'] = sparse_score
            candidate['path_score'] = path_score
            candidate['final_score'] = final_score
            candidate['predicate_penalty'] = predicate_penalty
            candidate['bridge_bonus'] = bridge_bonus
            
            scored_candidates.append(candidate)
        
        # 按最终分数排序
        scored_candidates.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        
        # 统计重排结果
        penalty_count = sum(1 for c in scored_candidates if c.get('predicate_penalty', 1.0) < 1.0)
        bonus_count = sum(1 for c in scored_candidates if c.get('bridge_bonus', 1.0) > 1.0)
        avg_final_score = sum(c.get('final_score', 0.0) for c in scored_candidates) / len(scored_candidates) if scored_candidates else 0.0
        
        logger.info(f"Reranking completed: {penalty_count} candidates penalized, {bonus_count} candidates boosted, avg_score: {avg_final_score:.3f}")
        
        return scored_candidates
    
    def _calculate_predicate_penalty(self, candidate: Dict[str, Any]) -> float:
        """计算谓词惩罚分数（found等谓词软过滤）。"""
        penalty_predicates = ['found', 'co founded', 'founded']
        note_id = candidate.get('note_id', 'unknown')
        
        # 检查候选中是否包含这些谓词
        text_fields = [candidate.get('title', ''), candidate.get('content', ''), 
                      candidate.get('summary', ''), candidate.get('raw_span_evidence', '')]
        full_text = ' '.join(text_fields).lower()
        
        for predicate in penalty_predicates:
            if predicate in full_text:
                logger.debug(f"Predicate penalty applied: note {note_id} penalized for containing '{predicate}' (penalty: 0.7)")
                return 0.7  # 命中found等谓词时降分
        
        logger.debug(f"No predicate penalty for note {note_id}")
        return 1.0  # 未命中时不惩罚
    
    def _calculate_bridge_entity_bonus(self, candidate: Dict[str, Any], bridge_entities: List[str]) -> float:
        """计算桥接实体加分。"""
        bridge_entity = candidate.get('bridge_entity', '')
        note_id = candidate.get('note_id', 'unknown')
        
        if bridge_entity and bridge_entity in bridge_entities:
            logger.debug(f"Bridge entity bonus applied: note {note_id} gets 1.25x boost for entity '{bridge_entity}'")
            return 1.25  # 命中桥接实体时加分
        
        logger.debug(f"No bridge entity bonus for note {note_id} (bridge_entity: '{bridge_entity}')")
        return 1.0
    
    def _apply_path_aware_reranking(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """应用路径感知重排，计算path_score。"""
        if not self.path_aware_enabled or not self.path_aware_ranker:
            return candidates
        
        try:
            # 识别第一跳和第二跳证据
            first_hop_candidates = [c for c in candidates if c.get('hop_type') != 'second_hop']
            second_hop_candidates = [c for c in candidates if c.get('hop_type') == 'second_hop']
            
            # 计算路径分数
            for candidate in candidates:
                path_score = 0.0
                
                # 只有当集合中同时覆盖第一跳证据与第二跳证据时才加分
                if first_hop_candidates and second_hop_candidates:
                    # 检查是否存在路径连接
                    if self._has_path_connection(candidate, first_hop_candidates, second_hop_candidates):
                        path_score = 0.5  # 基础路径分数
                        
                        # 根据路径质量调整分数
                        path_quality = self._calculate_path_quality(candidate, query)
                        path_score *= path_quality
                
                candidate['path_score'] = path_score
                
                # 计算最终分数：final = final_base + 0.3 * path_score
                final_base = candidate.get('final_base_score', candidate.get('similarity', 0.0))
                final_score = final_base + 0.3 * path_score
                candidate['final_score'] = final_score
            
            # 按最终分数重新排序
            candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Path-aware reranking failed: {e}")
        
        return candidates
    
    def _has_path_connection(self, candidate: Dict[str, Any], first_hop: List[Dict[str, Any]], second_hop: List[Dict[str, Any]]) -> bool:
        """检查是否存在路径连接。"""
        # 简单的路径连接检查
        candidate_entities = set(candidate.get('entities', []))
        candidate_entities.update(candidate.get('normalized_entities', []))
        
        # 检查与第一跳的连接
        first_hop_entities = set()
        for fh in first_hop:
            first_hop_entities.update(fh.get('entities', []))
            first_hop_entities.update(fh.get('normalized_entities', []))
        
        # 检查与第二跳的连接
        second_hop_entities = set()
        for sh in second_hop:
            second_hop_entities.update(sh.get('entities', []))
            second_hop_entities.update(sh.get('normalized_entities', []))
        
        # 如果候选与两跳都有实体重叠，认为存在路径连接
        has_first_connection = bool(candidate_entities & first_hop_entities)
        has_second_connection = bool(candidate_entities & second_hop_entities)
        
        return has_first_connection and has_second_connection
    
    def _calculate_path_quality(self, candidate: Dict[str, Any], query: str) -> float:
        """计算路径质量分数。"""
        # 基于候选的相关性和实体密度计算路径质量
        base_score = candidate.get('similarity', 0.0)
        entity_count = len(candidate.get('entities', [])) + len(candidate.get('normalized_entities', []))
        
        # 实体密度越高，路径质量越好
        entity_density = min(entity_count / 5.0, 1.0)  # 最多5个实体为满分
        
        quality = 0.7 * base_score + 0.3 * entity_density
        return min(quality, 1.0)
    
    def _log_retrieval_details(self, query: str, dataset: Optional[str], qid: Optional[str], selected_notes: List[Dict[str, Any]]):
        """记录详细的检索日志。"""
        try:
            # 获取索引版本
            index_version = getattr(self.vector_retriever.vector_index, 'version', 'unknown')
            
            # 收集TopN候选的详细信息
            top_n_details = []
            for i, note in enumerate(selected_notes[:10]):  # 只记录前10个
                detail = {
                    'rank': i + 1,
                    'file_name': note.get('file_name', 'unknown'),
                    'paragraph_idxs': note.get('paragraph_idxs', []),
                    'entities': note.get('entities', []),
                    'normalized_entities': note.get('normalized_entities', []),
                    'predicates': note.get('predicates', []),
                    'normalized_predicates': note.get('normalized_predicates', []),
                    'final_score': note.get('final_score', note.get('similarity', 0.0)),
                    'path_score': note.get('path_score', 0.0),
                    'hop_type': note.get('hop_type', 'first_hop')
                }
                top_n_details.append(detail)
            
            # 记录结构化日志
            self.structured_logger.info(
                "Detailed retrieval results",
                dataset=dataset,
                qid=qid,
                query_length=len(query),
                index_version=index_version,
                fusion_method=self.fusion_method,
                two_hop_enabled=self.two_hop_enabled,
                section_filtering_enabled=self.section_filtering_enabled,
                lexical_fallback_enabled=self.lexical_fallback_enabled,
                namespace_filtering_stages=self.namespace_filter_stages,
                total_selected=len(selected_notes),
                top_n_details=top_n_details
            )
            
        except Exception as e:
            logger.error(f"Failed to log retrieval details: {e}")
    
    @log_performance("QueryProcessor._process_traditional")
    def _process_traditional(self, query: str, dataset: Optional[str] = None, qid: Optional[str] = None) -> Dict[str, Any]:
        """Traditional query processing without sub-question decomposition."""
        # 记录传统处理开始
        self.structured_logger.debug("Starting traditional query processing",
                                   query_length=len(query),
                                   dataset=dataset,
                                   qid=qid,
                                   context_dispatcher=self.use_context_dispatcher)
        
        # Query rewriting functionality has been removed - using original query directly
        queries = [query]
        rewrite = {
            'original_query': query,
            'rewritten_queries': queries,
            'query_type': 'simple',
            'enhancements': []
        }
        
        if self.use_context_dispatcher:
            # 使用新的结构增强上下文调度器
            # 首先进行向量检索获取候选结果
            vector_results = self.vector_retriever.search(queries)
            
            # 合并结果并去重
            candidate_notes = []
            seen_note_ids = set()
            for sub in vector_results:
                for note in sub:
                    note_id = note.get('note_id')
                    if note_id and note_id not in seen_note_ids:
                        candidate_notes.append(note)
                        seen_note_ids.add(note_id)
                    elif not note_id:  # 如果没有note_id，基于内容去重
                        content = note.get('content', '')
                        content_hash = hash(content)
                        if content_hash not in seen_note_ids:
                            candidate_notes.append(note)
                            seen_note_ids.add(content_hash)
            
            logger.info(f"Initial vector recall for dispatcher: {len(candidate_notes)} unique notes")
            
            # 调用 ContextDispatcher 处理候选结果
            selected_notes = self.context_dispatcher.dispatch(candidate_notes)
            
            # 构建上下文
            context = "\n".join(n.get('content','') for n in selected_notes)
            
            # 构建调度结果信息
            dispatch_result = {
                'context': context,
                'selected_notes': selected_notes,
                'stage_info': {
                    'semantic_count': len([n for n in selected_notes if n.get('tags', {}).get('source') != 'graph']),
                    'graph_count': len([n for n in selected_notes if n.get('tags', {}).get('source') == 'graph']),
                    'final_count': len(selected_notes)
                }
            }
            
            # 在向量召回完成但尚未进行融合重排时进行命名空间校验
            if self.namespace_guard_enabled and dataset and qid:
                try:
                    # 获取索引版本号（如果可用）
                    index_version = getattr(self.vector_retriever.vector_index, 'version', None)
                    
                    # 调用命名空间断言检查
                    assert_namespace_or_raise(selected_notes, dataset, qid, index_version)
                    
                except DatasetNamespaceError as e:
                    # 记录错误日志并触发BM25回退逻辑
                    logger.error(f"dataset/qid mismatch")
                    
                    if self.bm25_fallback_enabled:
                        logger.warning(f"Triggering BM25 fallback for namespace {dataset}/{qid}")
                        try:
                            # 使用BM25进行回退搜索
                            fallback_notes = self._bm25_fallback_search(query, dataset, qid)
                            if fallback_notes:
                                selected_notes = fallback_notes
                                context = "\n".join(n.get('content','') for n in selected_notes)
                                logger.info(f"BM25 fallback retrieved {len(selected_notes)} notes")
                            else:
                                logger.error(f"BM25 fallback also failed for namespace {dataset}/{qid}")
                                raise
                        except Exception as fallback_error:
                            logger.error(f"BM25 fallback failed: {fallback_error}")
                            raise e
                    else:
                        raise
            
            logger.info(
                f"ContextDispatcher processed {dispatch_result['stage_info']['semantic_count']} semantic + "
                f"{dispatch_result['stage_info']['graph_count']} graph notes, "
                f"selected {dispatch_result['stage_info']['final_count']} final notes"
            )
            
        else:
            # 新的混合检索主流程
            # 1. 首轮向量召回
            vector_results = self.vector_retriever.search(queries)

            # 合并结果并去重
            candidate_notes = []
            seen_note_ids = set()
            for sub in vector_results:
                for note in sub:
                    note_id = note.get('note_id')
                    if note_id and note_id not in seen_note_ids:
                        candidate_notes.append(note)
                        seen_note_ids.add(note_id)
                    elif not note_id:  # 如果没有note_id，基于内容去重
                        content = note.get('content', '')
                        content_hash = hash(content)
                        if content_hash not in seen_note_ids:
                            candidate_notes.append(note)
                            seen_note_ids.add(content_hash)
            
            logger.info(f"Initial vector recall: {len(candidate_notes)} unique notes from {sum(len(sub) for sub in vector_results)} total results")
            
            # 第一阶段命名空间过滤：首轮召回后
            if self.namespace_filtering_enabled and 'initial_recall' in self.namespace_filter_stages and dataset and qid:
                try:
                    candidate_notes = filter_notes_by_namespace(candidate_notes, dataset, qid)
                    logger.info(f"Stage 1 - After initial recall namespace filtering: {len(candidate_notes)} notes")
                except Exception as e:
                    if self.same_namespace_bm25_fallback:
                        logger.warning(f"Stage 1 namespace filtering failed, trying BM25 fallback: {e}")
                        fallback_notes = self._bm25_fallback_search(query, dataset, qid)
                        if fallback_notes:
                            candidate_notes = fallback_notes
                            logger.info(f"Stage 1 BM25 fallback retrieved {len(candidate_notes)} notes")
                        elif self.strict_mode:
                            raise
                    elif self.strict_mode:
                        raise
            
            # 2. 对同一候选池执行BM25融合（使用新融合公式）
            if self.hybrid_search_enabled and candidate_notes:
                try:
                    # 生成检索守卫参数（包含段域过滤和词面保底）
                    must_have_terms, boost_entities, boost_predicates = self._generate_enhanced_guardrail_params(query)
                    
                    # 调用增强的混合检索（使用新融合公式）
                    candidate_notes = self._enhanced_hybrid_search_v2(
                        query, candidate_notes, 
                        must_have_terms=must_have_terms,
                        boost_entities=boost_entities, 
                        boost_predicates=boost_predicates
                    )
                    logger.info(f"After hybrid fusion (final_base = 1.0*dense + 0.6*sparse): {len(candidate_notes)} notes")
                except Exception as e:
                    logger.error(f"Hybrid search failed: {e}, continuing with vector results")
            
            # 修复实体抽取流程
            if self.fix_entity_extraction_enabled:
                logger.info("Fixing entity extraction flow after fusion")
                candidate_notes = self._fix_entity_extraction_flow(candidate_notes)
            
            # 应用噪声阈值过滤
            must_have_terms, _, _ = self._generate_enhanced_guardrail_params(query)
            candidate_notes = self._apply_noise_threshold_filtering(candidate_notes, must_have_terms)
            
            # 第二阶段命名空间过滤：融合后
            if self.namespace_filtering_enabled and 'post_fusion' in self.namespace_filter_stages and dataset and qid:
                candidate_notes = filter_notes_by_namespace(candidate_notes, dataset, qid)
                logger.info(f"Stage 2 - After post-fusion namespace filtering: {len(candidate_notes)} notes")
            
            # 3. 从第一跳TopM候选中抽取实体并发起二跳检索
            second_hop_notes = []
            if self.two_hop_enabled and candidate_notes:
                try:
                    # 取TopM候选进行实体抽取
                    top_m_notes = candidate_notes[:self.top_m_candidates]
                    extracted_entities = self._extract_entities_from_candidates(top_m_notes)
                    
                    if extracted_entities:
                        # 使用抽取的实体发起二跳检索
                        second_hop_notes = self._perform_second_hop_retrieval(extracted_entities, query, dataset)
                        logger.info(f"Second hop retrieval: extracted {len(extracted_entities)} entities, retrieved {len(second_hop_notes)} notes")
                except Exception as e:
                    logger.error(f"Two-hop expansion failed: {e}")
            
            # 4. 合并第一跳和二跳候选
            if second_hop_notes:
                if self.merge_strategy == 'weighted':
                    # 为二跳候选添加权重标记
                    for note in second_hop_notes:
                        note['hop_type'] = 'second_hop'
                        note['original_score'] = note.get('similarity', 0.0)
                elif self.merge_strategy == 'ranked':
                    # 简单合并，保持原有排序
                    pass
                
                candidate_notes.extend(second_hop_notes)
                logger.info(f"After merging first and second hop: {len(candidate_notes)} total notes")
                
                # 对合并后的候选再次修复实体抽取
                if self.fix_entity_extraction_enabled:
                    logger.info("Fixing entity extraction flow for merged candidates")
                    candidate_notes = self._fix_entity_extraction_flow(candidate_notes)
                
                # 对合并后的候选应用噪声阈值过滤
                candidate_notes = self._apply_noise_threshold_filtering(candidate_notes, must_have_terms)
            
            # 第三阶段命名空间过滤：二跳合并后
            if self.namespace_filtering_enabled and 'post_two_hop' in self.namespace_filter_stages and dataset and qid:
                candidate_notes = filter_notes_by_namespace(candidate_notes, dataset, qid)
                logger.info(f"Stage 3 - After two-hop merge namespace filtering: {len(candidate_notes)} notes")
            
            # 5. 路径感知重排（计算path_score）
            if self.path_aware_enabled and self.path_aware_ranker:
                try:
                    candidate_notes = self._apply_path_aware_reranking(candidate_notes, query)
                    logger.info(f"After path-aware reranking: {len(candidate_notes)} notes")
                except Exception as e:
                    logger.error(f"Path-aware reranking failed: {e}")
            
            # 6. 召回优化
            query_emb = self.vector_retriever.embedding_manager.encode_queries([query])[0]
            if self.recall_optimization_enabled:
                candidate_notes = self.recall_optimizer.optimize_recall(candidate_notes, query, query_emb)

            # 7. 图扩展（如果启用多跳）
            reasoning_paths: List[Dict[str, Any]] = []
            if self.multi_hop_enabled:
                mh_result = self.multi_hop_processor.retrieve(query_emb)
                graph_notes = mh_result.get('notes', [])
                candidate_notes.extend(graph_notes)
                for n in graph_notes:
                    reasoning_paths.extend(n.get('reasoning_paths', []))
                selected_notes = self.scheduler.schedule_for_multi_hop(candidate_notes, reasoning_paths)
            else:
                seed_ids = [note.get('note_id') for note in candidate_notes]
                graph_notes = self.graph_retriever.retrieve(seed_ids)
                candidate_notes.extend(graph_notes)
                selected_notes = self.scheduler.schedule(candidate_notes)
            
            # 第四阶段命名空间过滤：最终调度后
            if self.namespace_filtering_enabled and 'final_scheduling' in self.namespace_filter_stages and dataset and qid:
                try:
                    # 获取索引版本号（如果可用）
                    index_version = getattr(self.vector_retriever.vector_index, 'version', None)
                    
                    # 调用命名空间断言检查
                    assert_namespace_or_raise(selected_notes, dataset, qid, index_version)
                    logger.info(f"Stage 4 - Final namespace validation passed: {len(selected_notes)} notes")
                    
                except DatasetNamespaceError as e:
                    # 记录错误日志并触发BM25回退逻辑
                    logger.error(f"Final stage dataset/qid mismatch: {e}")
                    
                    if self.same_namespace_bm25_fallback:
                        logger.warning(f"Triggering final stage BM25 fallback for namespace {dataset}/{qid}")
                        try:
                            fallback_notes = self._enhanced_fallback_search(query, dataset, qid)
                            if fallback_notes:
                                selected_notes = fallback_notes
                                logger.info(f"Final stage BM25 fallback retrieved {len(selected_notes)} notes")
                            else:
                                logger.error(f"Final stage BM25 fallback also failed for namespace {dataset}/{qid}")
                                if self.strict_mode:
                                    raise
                        except Exception as fallback_error:
                            logger.error(f"Final stage BM25 fallback failed: {fallback_error}")
                            if self.strict_mode:
                                raise e
                    elif self.strict_mode:
                        raise
            
            # 记录详细的检索日志
            self._log_retrieval_details(query, dataset, qid, selected_notes)
            
            logger.info(
                f"Final scheduling: {len(candidate_notes)} candidates -> {len(selected_notes)} selected: "
                f"{[n.get('note_id') for n in selected_notes]}"
            )
            context = "\n".join(n.get('content','') for n in selected_notes)
        
        # 生成答案和评分
        answer = self.ollama.generate_final_answer(context, query)
        scores = self.ollama.evaluate_answer(query, context, answer)

        # 收集所有相关的paragraph idx信息
        predicted_support_idxs = []
        for n in selected_notes:
            n['feedback_score'] = scores.get('relevance',0)
            # 从原子笔记中提取paragraph_idxs
            if 'paragraph_idxs' in n and n['paragraph_idxs']:
                predicted_support_idxs.extend(n['paragraph_idxs'])
        
        # 去重并排序
        predicted_support_idxs = sorted(list(set(predicted_support_idxs)))

        result = {
            'query': query,
            'rewrite': rewrite,
            'answer': answer,
            'scores': scores,
            'notes': selected_notes,
            'predicted_support_idxs': predicted_support_idxs,
        }
        
        # 添加调度器特定的信息
        if self.use_context_dispatcher:
            result['dispatch_info'] = dispatch_result['stage_info']
        else:
            result['reasoning'] = reasoning_paths if self.multi_hop_enabled else None
            
        return result
    
    def _process_with_subquestion_decomposition(self, query: str, dataset: Optional[str] = None, qid: Optional[str] = None) -> Dict[str, Any]:
        """Process query using sub-question decomposition and parallel retrieval."""
        try:
            # Step 1: Decompose query into sub-questions
            sub_questions = self.subquestion_planner.decompose(query)
            
            # Step 2: Parallel retrieval for each sub-question
            if self.parallel_retrieval and len(sub_questions) > 1:
                subquestion_results = self._parallel_retrieval(sub_questions, dataset, qid)
            else:
                subquestion_results = self._sequential_retrieval(sub_questions, dataset, qid)
            
            # Step 3: Merge evidence from all sub-questions
            query_emb = self.vector_retriever.embedding_manager.encode_queries([query])[0]
            merged_evidence = self.evidence_merger.merge_evidence(
                subquestion_results, 
                query, 
                query_emb
            )
            
            # Step 4: Apply context scheduling to merged evidence
            if self.use_context_dispatcher:
                # Use context dispatcher with merged evidence
                selected_notes = self._schedule_merged_evidence_with_dispatcher(merged_evidence, query)
            else:
                # Use traditional scheduler with merged evidence
                selected_notes = self._schedule_merged_evidence_traditional(merged_evidence, query)
            
            # Step 4.5: Apply namespace validation after context scheduling
            if self.namespace_guard_enabled and dataset and qid:
                try:
                    # 获取索引版本号（如果可用）
                    index_version = getattr(self.vector_retriever.vector_index, 'version', None)
                    
                    # 调用命名空间断言检查
                    assert_namespace_or_raise(selected_notes, dataset, qid, index_version)
                    
                except DatasetNamespaceError as e:
                    # 记录错误日志并触发BM25回退逻辑
                    logger.error(f"dataset/qid mismatch")
                    
                    if self.bm25_fallback_enabled:
                        logger.warning(f"Triggering enhanced fallback for namespace {dataset}/{qid} in subquestion processing")
                        try:
                            # 使用增强的失败兜底策略
                            fallback_notes = self._enhanced_fallback_search(query, dataset, qid)
                            if fallback_notes:
                                selected_notes = fallback_notes
                                logger.info(f"Enhanced fallback retrieved {len(selected_notes)} notes for subquestion processing")
                            else:
                                logger.error(f"Enhanced fallback also failed for namespace {dataset}/{qid} in subquestion processing")
                                raise
                        except Exception as fallback_error:
                            logger.error(f"BM25 fallback failed in subquestion processing: {fallback_error}")
                            raise e
                    else:
                        raise
            
            # Step 5: Generate final answer using original query
            context = "\n".join(n.get('content', '') for n in selected_notes)
            answer = self.ollama.generate_final_answer(context, query)
            scores = self.ollama.evaluate_answer(query, context, answer)
            
            # Step 6: Collect paragraph indices
            predicted_support_idxs = []
            for n in selected_notes:
                n['feedback_score'] = scores.get('relevance', 0)
                if 'paragraph_idxs' in n and n['paragraph_idxs']:
                    predicted_support_idxs.extend(n['paragraph_idxs'])
            
            predicted_support_idxs = sorted(list(set(predicted_support_idxs)))
            
            # Step 7: Prepare result with sub-question information
            rewrite = {
                'original_query': query,
                'rewritten_queries': sub_questions,
                'query_type': 'multi_hop_decomposed',
                'enhancements': ['subquestion_decomposition']
            }
            
            result = {
                'query': query,
                'rewrite': rewrite,
                'answer': answer,
                'scores': scores,
                'notes': selected_notes,
                'predicted_support_idxs': predicted_support_idxs,
                'subquestion_info': {
                    'sub_questions': sub_questions,
                    'subquestion_results': subquestion_results,
                    'merge_statistics': self.evidence_merger.get_merge_statistics(merged_evidence)
                }
            }
            
            logger.info(f"Sub-question decomposition completed: {len(sub_questions)} sub-questions, {len(merged_evidence)} merged evidence, {len(selected_notes)} final notes")
            return result
            
        except Exception as e:
            logger.error(f"Error in sub-question decomposition processing: {e}")
            # Fallback to traditional processing
            logger.info("Falling back to traditional processing")
            return self._process_traditional(query)
    
    def _parallel_retrieval(self, sub_questions: List[str], dataset: Optional[str] = None, qid: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform parallel retrieval for multiple sub-questions."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sub_questions), 4)) as executor:
            # Submit retrieval tasks
            future_to_question = {
                executor.submit(self._retrieve_for_subquestion, sq, i, dataset, qid): (sq, i) 
                for i, sq in enumerate(sub_questions)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_question):
                sub_question, index = future_to_question[future]
                try:
                    result = future.result()
                    results.append((index, result))
                except Exception as e:
                    logger.error(f"Error retrieving for sub-question '{sub_question}': {e}")
                    # Add empty result to maintain order
                    results.append((index, {
                        'sub_question': sub_question,
                        'vector_results': [],
                        'graph_results': []
                    }))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [result[1] for result in results]
    
    def _sequential_retrieval(self, sub_questions: List[str], dataset: Optional[str] = None, qid: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform sequential retrieval for multiple sub-questions."""
        results = []
        
        for i, sub_question in enumerate(sub_questions):
            try:
                result = self._retrieve_for_subquestion(sub_question, i, dataset, qid)
                results.append(result)
            except Exception as e:
                logger.error(f"Error retrieving for sub-question '{sub_question}': {e}")
                results.append({
                    'sub_question': sub_question,
                    'vector_results': [],
                    'graph_results': []
                })
        
        return results
    
    def _retrieve_for_subquestion(self, sub_question: str, index: int, dataset: Optional[str] = None, qid: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve evidence for a single sub-question."""
        # Vector retrieval
        vector_results = self.vector_retriever.search([sub_question])
        vector_notes = vector_results[0] if vector_results else []
        
        logger.info(f"Sub-question {index}: '{sub_question}' initial vector recall: {len(vector_notes)} notes")
        
        # 第一阶段命名空间守卫：子问题向量召回后
        if self.namespace_guard_enabled and dataset and qid:
            vector_notes = filter_notes_by_namespace(vector_notes, dataset, qid)
            logger.info(f"Sub-question {index}: After namespace filtering: {len(vector_notes)} notes")
        
        # Apply hybrid search if enabled
        if self.hybrid_search_enabled and vector_notes:
            try:
                # 生成检索守卫参数
                must_have_terms, boost_entities, boost_predicates = self._generate_guardrail_params(sub_question)
                
                # 调用增强混合检索
                vector_notes = self._enhanced_hybrid_search(
                    sub_question, vector_notes,
                    must_have_terms=must_have_terms,
                    boost_entities=boost_entities,
                    boost_predicates=boost_predicates
                )
                logger.info(f"Sub-question {index}: After hybrid search: {len(vector_notes)} notes")
            except Exception as e:
                logger.error(f"Hybrid search failed for sub-question {index}: {e}")
        
        # 第二阶段命名空间守卫：子问题混合检索后
        if self.namespace_guard_enabled and dataset and qid:
            vector_notes = filter_notes_by_namespace(vector_notes, dataset, qid)
            logger.info(f"Sub-question {index}: After post-hybrid namespace filtering: {len(vector_notes)} notes")
        
        # Graph retrieval
        seed_ids = [note.get('note_id') for note in vector_notes if note.get('note_id')]
        graph_notes = self.graph_retriever.retrieve(seed_ids) if seed_ids else []
        
        # 第三阶段命名空间守卫：子问题图扩展后
        if self.namespace_guard_enabled and dataset and qid:
            graph_notes = filter_notes_by_namespace(graph_notes, dataset, qid)
            logger.info(f"Sub-question {index}: After graph expansion namespace filtering: {len(graph_notes)} notes")
        
        logger.info(f"Sub-question {index}: '{sub_question}' final retrieved {len(vector_notes)} vector + {len(graph_notes)} graph notes")
        
        return {
            'sub_question': sub_question,
            'sub_question_index': index,
            'vector_results': vector_notes,
            'graph_results': graph_notes
        }
    
    def _schedule_merged_evidence_with_dispatcher(self, merged_evidence: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Schedule merged evidence using context dispatcher."""
        # Convert merged evidence to the format expected by context dispatcher
        # For now, treat merged evidence as semantic results
        dispatch_result = {
            'context': "\n".join(n.get('content', '') for n in merged_evidence),
            'selected_notes': merged_evidence[:config.get('dispatcher.final_semantic_count', 3) + 
                                           config.get('dispatcher.final_graph_count', 5)],
            'stage_info': {
                'semantic_count': len([n for n in merged_evidence if 'vector' in n.get('source_types', set())]),
                'graph_count': len([n for n in merged_evidence if 'graph' in n.get('source_types', set())]),
                'final_count': min(len(merged_evidence), 
                                 config.get('dispatcher.final_semantic_count', 3) + 
                                 config.get('dispatcher.final_graph_count', 5))
            }
        }
        
        return dispatch_result['selected_notes']
    
    @log_performance("QueryProcessor._schedule_merged_evidence_traditional")
    def _schedule_merged_evidence_traditional(self, merged_evidence: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Schedule merged evidence using traditional scheduler with PathAware + Diversity linkage."""
        # 记录调度开始
        self.structured_logger.debug("Starting evidence scheduling",
                                   evidence_count=len(merged_evidence),
                                   diversity_enabled=self.diversity_scheduler_enabled,
                                   recall_optimization=self.recall_optimization_enabled)
        
        # Apply recall optimization if enabled
        if self.recall_optimization_enabled:
            query_emb = self.vector_retriever.embedding_manager.encode_queries([query])[0]
            merged_evidence = self.recall_optimizer.optimize_recall(merged_evidence, query, query_emb)
        
        # 优先使用DiversityScheduler进行PathAware + Diversity联动调度
        if self.diversity_scheduler_enabled and self.diversity_scheduler:
            try:
                # 使用DiversityScheduler进行候选调度，支持证据类型配额管理
                diversity_result = self.diversity_scheduler.schedule_candidates(merged_evidence)
                selected_notes = diversity_result.selected_candidates
                
                # 记录多样性调度指标
                log_diversity_metrics(
                    candidates_count=len(merged_evidence),
                    selected_count=len(selected_notes),
                    diversity_score=diversity_result.diversity_score,
                    evidence_quota_enabled=self.diversity_scheduler.config.get('enable_evidence_quota', True),
                    scheduler_strategy=self.diversity_scheduler.config.get('strategy', 'default')
                )
                
                self.structured_logger.info("DiversityScheduler completed successfully",
                                           selected_count=len(selected_notes),
                                           diversity_score=f"{diversity_result.diversity_score:.3f}")
                return selected_notes
                
            except Exception as e:
                self.structured_logger.error("DiversityScheduler failed, using fallback",
                                           error=str(e))
        
        # 传统调度器作为兜底
        if self.multi_hop_enabled:
            # Extract reasoning paths from merged evidence
            reasoning_paths = []
            for evidence in merged_evidence:
                reasoning_paths.extend(evidence.get('reasoning_paths', []))
            
            selected_notes = self.scheduler.schedule_for_multi_hop(merged_evidence, reasoning_paths)
        else:
            selected_notes = self.scheduler.schedule(merged_evidence)
        
        self.structured_logger.debug("Traditional scheduler completed",
                                   selected_count=len(selected_notes),
                                   scheduler_type="multi_hop" if self.multi_hop_enabled else "standard")
        return selected_notes
    
    def _hybrid_search(self, query: str, notes_pool: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        混合检索：结合向量相似度和 BM25 分数
        
        Args:
            query: 查询字符串
            notes_pool: 候选笔记池
            top_k: 返回的结果数量
            
        Returns:
            按融合分数排序的 top_k 个结果
        """
        if not notes_pool:
            logger.warning("Empty notes pool for hybrid search")
            return []
            
        if not self.hybrid_search_enabled or not self.bm25_corpus:
            logger.warning("Hybrid search not enabled or BM25 corpus not available, falling back to vector search")
            # 回退到纯向量检索
            return self._fallback_vector_search(query, notes_pool, top_k)
        
        try:
            # 1. 计算向量相似度分数
            vector_scores = self._calculate_vector_similarities(query, notes_pool)
            
            # 2. 计算 BM25 分数
            bm25_scores_list = bm25_scores(self.bm25_corpus, notes_pool, query)
            
            # 3. 确保分数列表长度一致
            if len(vector_scores) != len(bm25_scores_list) or len(vector_scores) != len(notes_pool):
                logger.error(f"Score length mismatch: vector={len(vector_scores)}, bm25={len(bm25_scores_list)}, notes={len(notes_pool)}")
                return self._fallback_vector_search(query, notes_pool, top_k)
            
            # 4. 线性融合分数
            hybrid_results = []
            for i, note in enumerate(notes_pool):
                vector_score = vector_scores[i]
                bm25_score = bm25_scores_list[i]
                
                # 线性融合：score = vector_weight * vector_score + bm25_weight * bm25_score
                hybrid_score = self.vector_weight * vector_score + self.bm25_weight * bm25_score
                
                # 创建结果项
                result_note = note.copy()
                result_note['hybrid_score'] = hybrid_score
                result_note['vector_score'] = vector_score
                result_note['bm25_score'] = bm25_score
                result_note['search_method'] = 'hybrid'
                
                hybrid_results.append(result_note)
            
            # 5. 按融合分数排序并返回 top_k
            hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            top_results = hybrid_results[:top_k]
            
            logger.info(f"Hybrid search completed: {len(notes_pool)} candidates -> {len(top_results)} results")
            if top_results:
                logger.debug(f"Top hybrid score: {top_results[0]['hybrid_score']:.4f} (vector: {top_results[0]['vector_score']:.4f}, bm25: {top_results[0]['bm25_score']:.4f})")
            
            return top_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self._fallback_vector_search(query, notes_pool, top_k)
    
    def _calculate_vector_similarities(self, query: str, notes_pool: List[Dict[str, Any]]) -> List[float]:
        """
        计算查询与笔记池中每个笔记的向量相似度
        
        Args:
            query: 查询字符串
            notes_pool: 笔记池
            
        Returns:
            相似度分数列表
        """
        try:
            # 编码查询
            query_embedding = self.vector_retriever.embedding_manager.encode_queries([query])[0]
            
            similarities = []
            for note in notes_pool:
                note_id = note.get('note_id')
                if note_id and hasattr(self.vector_retriever, 'id_to_index') and note_id in self.vector_retriever.id_to_index:
                    # 使用预计算的嵌入
                    note_idx = self.vector_retriever.id_to_index[note_id]
                    if note_idx < len(self.vector_retriever.note_embeddings):
                        note_embedding = self.vector_retriever.note_embeddings[note_idx]
                        # 计算余弦相似度
                        similarity = np.dot(query_embedding, note_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(note_embedding)
                        )
                        similarities.append(max(0.0, similarity))  # 确保非负
                    else:
                        similarities.append(0.0)
                else:
                    # 动态计算嵌入（较慢）
                    note_text = note.get('content', '')
                    if note_text:
                        note_embedding = self.vector_retriever.embedding_manager.encode_queries([note_text])[0]
                        similarity = np.dot(query_embedding, note_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(note_embedding)
                        )
                        similarities.append(max(0.0, similarity))
                    else:
                        similarities.append(0.0)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating vector similarities: {e}")
            return [0.0] * len(notes_pool)
    
    def _fallback_vector_search(self, query: str, notes_pool: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        回退到纯向量检索
        
        Args:
            query: 查询字符串
            notes_pool: 笔记池
            top_k: 返回结果数量
            
        Returns:
            按向量相似度排序的结果
        """
        try:
            vector_scores = self._calculate_vector_similarities(query, notes_pool)
            
            # 按相似度排序并返回top_k结果
            scored_notes = list(zip(notes_pool, vector_scores))
            scored_notes.sort(key=lambda x: x[1], reverse=True)
            
            return [note for note, score in scored_notes[:top_k]]
            
        except Exception as e:
            logger.error(f"Vector fallback search failed: {e}")
            return []
    
    @log_performance("QueryProcessor._enhanced_hybrid_search")
    def _enhanced_hybrid_search(self, query: str, candidates: List[Dict[str, Any]], 
                               must_have_terms: List[str] = None, 
                               boost_entities: List[str] = None, 
                               boost_predicates: List[str] = None) -> List[Dict[str, Any]]:
        """
        增强混合搜索：集成PathAwareRanker的路径分数
        """
        if not candidates:
            return candidates
        
        # 记录检索开始
        self.structured_logger.debug("Starting enhanced hybrid search",
                                   candidates_count=len(candidates),
                                   path_aware_enabled=bool(self.path_aware_ranker),
                                   has_must_terms=bool(must_have_terms),
                                   has_boost_entities=bool(boost_entities),
                                   has_boost_predicates=bool(boost_predicates))
        
        try:
            # 首先进行标准混合检索
            candidates = self._hybrid_search(query, candidates, must_have_terms, boost_entities, boost_predicates)
            
            # 如果启用了PathAwareRanker，计算路径分数
            if self.path_aware_ranker:
                try:
                    # 使用PathAwareRanker重新排序
                    candidates = self.path_aware_ranker.rerank_candidates(query, candidates)
                    
                    # 统计路径增强信息
                    path_enhanced_count = sum(1 for c in candidates if c.get('path_score', 0) > 0)
                    avg_path_score = sum(c.get('path_score', 0) for c in candidates) / max(len(candidates), 1)
                    
                    # 记录路径感知指标
                    log_path_aware_metrics(
                        candidates_count=len(candidates),
                        path_enhanced_count=path_enhanced_count,
                        avg_path_score=avg_path_score,
                        path_weight=self.path_weight
                    )
                    
                    # 融合路径分数与混合分数
                    for candidate in candidates:
                        hybrid_score = candidate.get('hybrid_score', 0.0)
                        path_score = candidate.get('path_score', 0.0)
                        
                        # 最终分数：α * dense + β * sparse + γ * path_score
                        # 这里hybrid_score已经是dense+sparse的融合结果
                        final_score = (1 - self.path_weight) * hybrid_score + self.path_weight * path_score
                        candidate['final_score'] = final_score
                    
                    # 按最终分数重新排序
                    candidates.sort(key=lambda x: x.get('final_score', x.get('hybrid_score', 0)), reverse=True)
                    logger.info(f"PathAware reranking completed, path_weight={self.path_weight}")
                    
                except Exception as e:
                    logger.error(f"PathAware ranking failed: {e}, using hybrid scores only")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Enhanced hybrid search failed: {e}")
            return self._fallback_vector_search(query, candidates)
    
    @log_performance("QueryProcessor._hybrid_search")
    def _hybrid_search(self, query: str, candidates: List[Dict[str, Any]], 
                      must_have_terms: List[str] = None, 
                      boost_entities: List[str] = None, 
                      boost_predicates: List[str] = None) -> List[Dict[str, Any]]:
        """
        混合搜索：结合向量相似度和BM25分数
        """
        # 记录混合搜索开始
        self.structured_logger.debug("Starting hybrid search",
                                   candidates_count=len(candidates),
                                   must_have_terms_count=len(must_have_terms) if must_have_terms else 0,
                                   boost_entities_count=len(boost_entities) if boost_entities else 0,
                                   boost_predicates_count=len(boost_predicates) if boost_predicates else 0)
        
        if not self.hybrid_search_enabled or not self.bm25_corpus:
            self.structured_logger.warning("Hybrid search not available, using fallback",
                                         hybrid_enabled=self.hybrid_search_enabled,
                                         bm25_corpus_available=bool(self.bm25_corpus))
            return self._fallback_vector_search(query, candidates)
        
        try:
            # 计算向量相似度
            vector_scores = self._calculate_vector_similarities(query, candidates)
            
            # 计算BM25分数
            from utils.bm25_search import bm25_scores
            bm25_scores_list = bm25_scores(self.bm25_corpus, candidates, query)
            
            # 融合分数
            if self.fusion_method == 'linear':
                # 线性融合
                for i, candidate in enumerate(candidates):
                    vector_score = vector_scores[i] if i < len(vector_scores) else 0.0
                    bm25_score = bm25_scores_list[i] if i < len(bm25_scores_list) else 0.0
                    
                    # 应用检索守卫
                    if must_have_terms:
                        content = candidate.get('content', '').lower()
                        if not any(term.lower() in content for term in must_have_terms):
                            bm25_score *= 0.1  # 大幅降低分数
                    
                    if boost_entities:
                        content = candidate.get('content', '').lower()
                        for entity in boost_entities:
                            if entity.lower() in content:
                                vector_score *= 1.2
                    
                    if boost_predicates:
                        content = candidate.get('content', '').lower()
                        for predicate in boost_predicates:
                            if predicate.lower() in content:
                                bm25_score *= 1.3
                    
                    final_score = (self.vector_weight * vector_score + 
                                 self.bm25_weight * bm25_score)
                    candidate['hybrid_score'] = final_score
            
            elif self.fusion_method == 'rrf':
                # RRF融合
                vector_ranks = {i: rank for rank, i in enumerate(sorted(range(len(candidates)), 
                                                                       key=lambda x: vector_scores[x] if x < len(vector_scores) else 0, reverse=True))}
                bm25_ranks = {i: rank for rank, i in enumerate(sorted(range(len(candidates)), 
                                                                     key=lambda x: bm25_scores_list[x] if x < len(bm25_scores_list) else 0, reverse=True))}
                
                for i, candidate in enumerate(candidates):
                    vector_rank = vector_ranks.get(i, len(candidates))
                    bm25_rank = bm25_ranks.get(i, len(candidates))
                    
                    rrf_score = (self.vector_weight / (self.rrf_k + vector_rank) + 
                               self.bm25_weight / (self.rrf_k + bm25_rank))
                    
                    # 应用检索守卫
                    if must_have_terms:
                        content = candidate.get('content', '').lower()
                        if not any(term.lower() in content for term in must_have_terms):
                            rrf_score *= 0.1
                    
                    candidate['hybrid_score'] = rrf_score
            
            # 按融合分数排序
            candidates.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
            
            logger.info(f"Hybrid search completed with {self.fusion_method} fusion")
            return candidates
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._fallback_vector_search(query, candidates)
    
    def _generate_guardrail_params(self, query: str) -> tuple:
        """
        从查询中生成检索守卫参数
        """
        must_have_terms = []
        boost_entities = []
        boost_predicates = []
        
        try:
            # 从配置中获取模板
            if self.retrieval_guardrail_enabled:
                # 提取查询中的关键词作为must_have_terms
                query_lower = query.lower()
                
                # 添加配置中的must_have_terms模板
                template_terms = self.must_have_terms_config.get('templates', [])
                for term in template_terms:
                    if term.lower() in query_lower:
                        must_have_terms.append(term)
                
                # 添加标准谓词映射
                for standard_pred, variants in self.predicate_mappings.items():
                    for variant in variants:
                        if variant.lower() in query_lower:
                            must_have_terms.append(standard_pred)
                            boost_predicates.append(standard_pred)
                            break
                
                # 实体提升（这里可以集成NER或实体识别）
                entity_templates = self.boost_entities_config.get('templates', [])
                for entity in entity_templates:
                    if entity.lower() in query_lower:
                        boost_entities.append(entity)
                
                # 谓词提升
                predicate_templates = self.boost_predicates_config.get('templates', [])
                for predicate in predicate_templates:
                    if predicate.lower() in query_lower:
                        boost_predicates.append(predicate)
            
            logger.debug(f"Generated guardrail params - must_have: {must_have_terms}, boost_entities: {boost_entities}, boost_predicates: {boost_predicates}")
            
        except Exception as e:
            logger.error(f"Failed to generate guardrail params: {e}")
        
        return must_have_terms, boost_entities, boost_predicates
    
    def _enhanced_fallback_search(self, query: str, dataset: str, qid: str, top_k: int = 10, retry_count: int = 0) -> List[Dict[str, Any]]:
        """
        增强的失败兜底策略：二次改写 + 提升稀疏权重
        
        Args:
            query: 查询字符串
            dataset: 数据集名称
            qid: 问题ID
            top_k: 返回结果数量
            retry_count: 重试次数
            
        Returns:
            兜底搜索结果
        """
        try:
            # 首先尝试原始查询的BM25搜索
            fallback_notes = self._bm25_fallback_search(query, dataset, qid, top_k)
            
            # 如果结果不足且启用了查询改写，尝试改写查询
            if (len(fallback_notes) < top_k // 2 and 
                self.fallback_query_rewrite_enabled and 
                self.llm_rewriter and 
                self.llm_rewriter.enabled and 
                retry_count < self.fallback_max_retries):
                
                logger.info(f"Insufficient results ({len(fallback_notes)}), attempting query rewriting (retry {retry_count + 1})")
                
                try:
                    # 使用LLM改写查询
                    rewritten_subqueries = self.llm_rewriter.rewrite_query(query)
                    
                    if rewritten_subqueries:
                        # 尝试改写后的查询
                        for subquery in rewritten_subqueries[:2]:  # 限制最多2个改写查询
                            rewritten_query = subquery.text
                            if rewritten_query != query:  # 确保查询确实被改写了
                                logger.info(f"Trying rewritten query: {rewritten_query[:50]}...")
                                
                                # 使用改写后的查询进行BM25搜索，提升稀疏权重
                                rewritten_notes = self._bm25_fallback_search_with_boost(
                                    rewritten_query, dataset, qid, top_k, 
                                    boost_factor=self.fallback_sparse_boost
                                )
                                
                                # 合并结果，去重
                                combined_notes = self._merge_fallback_results(fallback_notes, rewritten_notes)
                                
                                if len(combined_notes) > len(fallback_notes):
                                    fallback_notes = combined_notes
                                    logger.info(f"Query rewriting improved results: {len(combined_notes)} notes")
                                    break
                                    
                except Exception as rewrite_error:
                    logger.error(f"Query rewriting failed: {rewrite_error}")
            
            # 如果仍然结果不足，尝试降低阈值的BM25搜索
            if len(fallback_notes) < top_k // 3:
                logger.info("Attempting relaxed BM25 search with lower threshold")
                relaxed_notes = self._bm25_fallback_search_relaxed(query, dataset, qid, top_k * 2)
                fallback_notes = self._merge_fallback_results(fallback_notes, relaxed_notes)
            
            logger.info(f"Enhanced fallback search completed: {len(fallback_notes)} notes found")
            return fallback_notes[:top_k]
            
        except Exception as e:
            logger.error(f"Enhanced fallback search failed: {e}")
            # 降级到基础BM25搜索
            return self._bm25_fallback_search(query, dataset, qid, top_k)
    
    def _bm25_fallback_search(self, query: str, dataset: str, qid: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        BM25回退搜索，当命名空间校验失败时使用
        
        Args:
            query: 查询字符串
            dataset: 数据集名称
            qid: 问题ID
            top_k: 返回结果数量
            
        Returns:
            符合命名空间要求的BM25搜索结果
        """
        try:
            # 首先过滤出符合命名空间的笔记
            namespace_notes = filter_notes_by_namespace(self.atomic_notes, dataset, qid)
            
            if not namespace_notes:
                logger.warning(f"No notes found in namespace {dataset}/{qid} for BM25 fallback")
                return []
            
            # 构建BM25语料库
            bm25_corpus = build_bm25_corpus(namespace_notes, lambda note: note.get('content', ''))
            
            # 计算BM25分数
            bm25_result = bm25_scores(query, bm25_corpus)
            
            # 按分数排序并返回top_k结果
            scored_notes = list(zip(namespace_notes, bm25_result))
            scored_notes.sort(key=lambda x: x[1], reverse=True)
            
            fallback_notes = [note for note, score in scored_notes[:top_k] if score > 0]
            
            logger.info(f"BM25 fallback found {len(fallback_notes)} relevant notes in namespace {dataset}/{qid}")
            return fallback_notes
            
        except Exception as e:
            logger.error(f"BM25 fallback search failed: {e}")
            return []
    
    def _bm25_fallback_search_with_boost(self, query: str, dataset: str, qid: str, top_k: int = 10, boost_factor: float = 1.5) -> List[Dict[str, Any]]:
        """
        带权重提升的BM25回退搜索
        
        Args:
            query: 查询字符串
            dataset: 数据集名称
            qid: 问题ID
            top_k: 返回结果数量
            boost_factor: 稀疏权重提升因子
            
        Returns:
            符合命名空间要求的BM25搜索结果
        """
        try:
            # 首先过滤出符合命名空间的笔记
            namespace_notes = filter_notes_by_namespace(self.atomic_notes, dataset, qid)
            
            if not namespace_notes:
                logger.warning(f"No notes found in namespace {dataset}/{qid} for boosted BM25 fallback")
                return []
            
            # 构建BM25语料库
            bm25_corpus = build_bm25_corpus(namespace_notes, lambda note: note.get('content', ''))
            
            # 计算BM25分数并应用提升因子
            bm25_result = bm25_scores(query, bm25_corpus)
            boosted_scores = [score * boost_factor for score in bm25_result]
            
            # 按分数排序并返回top_k结果
            scored_notes = list(zip(namespace_notes, boosted_scores))
            scored_notes.sort(key=lambda x: x[1], reverse=True)
            
            fallback_notes = [note for note, score in scored_notes[:top_k] if score > 0]
            
            logger.info(f"Boosted BM25 fallback (factor={boost_factor}) found {len(fallback_notes)} relevant notes")
            return fallback_notes
            
        except Exception as e:
            logger.error(f"Boosted BM25 fallback search failed: {e}")
            return []
    
    def _bm25_fallback_search_relaxed(self, query: str, dataset: str, qid: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        放宽阈值的BM25回退搜索
        
        Args:
            query: 查询字符串
            dataset: 数据集名称
            qid: 问题ID
            top_k: 返回结果数量
            
        Returns:
            符合命名空间要求的BM25搜索结果（放宽阈值）
        """
        try:
            # 首先过滤出符合命名空间的笔记
            namespace_notes = filter_notes_by_namespace(self.atomic_notes, dataset, qid)
            
            if not namespace_notes:
                return []
            
            # 构建BM25语料库
            bm25_corpus = build_bm25_corpus(namespace_notes, lambda note: note.get('content', ''))
            
            # 计算BM25分数
            bm25_result = bm25_scores(query, bm25_corpus)
            
            # 按分数排序，放宽阈值（接受更低分数）
            scored_notes = list(zip(namespace_notes, bm25_result))
            scored_notes.sort(key=lambda x: x[1], reverse=True)
            
            # 放宽阈值，接受分数 > 0.01 的结果
            fallback_notes = [note for note, score in scored_notes[:top_k] if score > 0.01]
            
            logger.info(f"Relaxed BM25 fallback found {len(fallback_notes)} relevant notes")
            return fallback_notes
            
        except Exception as e:
            logger.error(f"Relaxed BM25 fallback search failed: {e}")
            return []
    
    def _merge_fallback_results(self, primary_results: List[Dict[str, Any]], secondary_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并兜底搜索结果，去重并保持相关性排序
        
        Args:
            primary_results: 主要搜索结果
            secondary_results: 次要搜索结果
            
        Returns:
            合并后的去重结果
        """
        try:
            # 使用note_id进行去重
            seen_ids = set()
            merged_results = []
            
            # 首先添加主要结果
            for note in primary_results:
                note_id = note.get('note_id', note.get('id', ''))
                if note_id and note_id not in seen_ids:
                    seen_ids.add(note_id)
                    merged_results.append(note)
            
            # 然后添加次要结果中的新结果
            for note in secondary_results:
                note_id = note.get('note_id', note.get('id', ''))
                if note_id and note_id not in seen_ids:
                    seen_ids.add(note_id)
                    merged_results.append(note)
            
            logger.debug(f"Merged {len(primary_results)} + {len(secondary_results)} -> {len(merged_results)} unique results")
            return merged_results
            
        except Exception as e:
            logger.error(f"Failed to merge fallback results: {e}")
            return primary_results  # 返回主要结果作为兜底
