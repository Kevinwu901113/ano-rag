from typing import List, Dict, Any, Optional
from loguru import logger
import os
import numpy as np
import concurrent.futures
import threading
import time
import json

# 导入增强的日志功能
from utils.logging_utils import (
    StructuredLogger, log_performance, log_operation,
    log_retrieval_metrics, log_diversity_metrics, log_path_aware_metrics
)

from llm import OllamaClient, LocalLLM
from llm.multi_model_client import HybridLLMDispatcher
from llm.prompts import build_context_prompt, build_context_prompt_with_passages
from utils.robust_json_parser import extract_prediction_with_retry
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

# 导入新的无硬编码模块
from retrieval.learned_fusion import create_learned_fusion
from reasoning.qa_coverage import create_qa_coverage_scorer
from answer.span_picker import create_span_picker
from answer.verify_shell import create_answer_verifier
from context.packer import ContextPacker
from retrieval.listt5_reranker import ListT5Reranker, create_listt5_reranker, fuse_scores, sort_desc

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
                # 加载图数据
                from utils import FileUtils
                from networkx.readwrite import json_graph
                data = FileUtils.read_json(graph_file)
                graph = json_graph.node_link_graph(data, edges="links")
                # 重新构建索引以确保映射正确
                self.graph_index.build_index(graph, atomic_notes, embeddings)
                logger.info(f"Loaded and rebuilt graph index from {graph_file}")
            except Exception as e:
                logger.error(f"Failed to load graph from {graph_file}: {e}, rebuilding")
                graph = builder.build_graph(atomic_notes, embeddings)
                self.graph_index.build_index(graph, atomic_notes, embeddings)
        else:
            graph = builder.build_graph(atomic_notes, embeddings)
            self.graph_index = GraphIndex()
            self.graph_index.build_index(graph, atomic_notes, embeddings)

        self.multi_hop_enabled = config.get('retrieval.multi_hop.enabled', False)
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
            self.context_dispatcher = ContextDispatcher(config, graph_index=self.graph_index, vector_retriever=self.vector_retriever)
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

        # 初始化LLM客户端 - 支持混合模式
        llm_provider = config.get('llm.provider', 'ollama')
        if llm_provider == 'hybrid_llm':
            self.llm_client = HybridLLMDispatcher()
            logger.info("Using HybridLLMDispatcher for intelligent task routing")
        else:
            self.llm_client = OllamaClient()
            logger.info(f"Using single LLM provider: {llm_provider}")
        
        # 保持向后兼容性
        self.ollama = self.llm_client
        self.atomic_notes = atomic_notes
        
        # 初始化子问题分解组件
        self.use_subquestion_decomposition = config.get('query.use_subquestion_decomposition', False)
        if self.use_subquestion_decomposition:
            self.subquestion_planner = SubQuestionPlanner(llm_client=self.llm_client)
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
        
        # 初始化新的无硬编码模块
        calibration_path = config.get('calibration.path', 'calibration.json')
        
        # 初始化可学习融合器
        try:
            fusion_model_type = config.get('learned_fusion.model_type', 'linear')
            self.learned_fusion = create_learned_fusion(
                model_type=fusion_model_type,
                calibration_path=calibration_path
            )
            self.use_learned_fusion = config.get('learned_fusion.enabled', False)
            logger.info(f"Learned fusion initialized: {fusion_model_type} model")
        except Exception as e:
            logger.warning(f"Failed to initialize learned fusion: {e}")
            self.learned_fusion = None
            self.use_learned_fusion = False
        
        # 初始化QA覆盖度评估器
        try:
            self.qa_coverage_scorer = create_qa_coverage_scorer(calibration_path=calibration_path)
            logger.info("QA coverage scorer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize QA coverage scorer: {e}")
            self.qa_coverage_scorer = None
        
        # 初始化答案定位器
        try:
            span_model_type = config.get('span_picker.model_type', 'cross_encoder')
            self.span_picker = create_span_picker(
                model_type=span_model_type,
                calibration_path=calibration_path
            )
            logger.info(f"Span picker initialized: {span_model_type} model")
        except Exception as e:
            logger.warning(f"Failed to initialize span picker: {e}")
            self.span_picker = None
        
        # 初始化答案验证器
        try:
            self.answer_verifier = create_answer_verifier(
                span_picker=self.span_picker,
                calibration_path=calibration_path
            )
            self.use_answer_verification = config.get('answer_verification.enabled', False)
            logger.info("Answer verifier initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize answer verifier: {e}")
            self.answer_verifier = None
            self.use_answer_verification = False
        
        # 初始化上下文打包器（使用新的结构化打包）
        try:
            self.context_packer = ContextPacker(calibration_path=calibration_path)
            logger.info("Context packer initialized with structure-based packing")
        except Exception as e:
            logger.warning(f"Failed to initialize context packer: {e}")
            self.context_packer = None
        
        # 初始化ListT5重排序器
        try:
            self.listt5 = create_listt5_reranker(config) if config.get('rerank.use_listt5', False) else None
            if self.listt5:
                logger.info(f"ListT5 reranker initialized: {config.get('rerank.listt5_model', 'default')}")
            else:
                logger.info("ListT5 reranker disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize ListT5 reranker: {e}")
            self.listt5 = None
        
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
        
        # 图检索策略配置
        graph_config = config.get('retrieval.multi_hop', {})
        self.graph_strategy = graph_config.get('strategy', 'entity_extraction')
        
        # Top-K种子节点策略配置
        top_k_seed_config = graph_config.get('top_k_seed', {})
        self.top_k_seed_enabled = top_k_seed_config.get('enabled', False)
        self.seed_count = top_k_seed_config.get('seed_count', 5)
        self.fallback_to_entity = top_k_seed_config.get('fallback_to_entity', True)
        
        # 实体提取策略配置
        entity_extraction_config = graph_config.get('entity_extraction', {})
        self.entity_extraction_enabled = entity_extraction_config.get('enabled', True)
        self.max_entities = entity_extraction_config.get('max_entities', 10)
        
        # 混合策略配置
        hybrid_mode_config = graph_config.get('hybrid_mode', {})
        self.primary_strategy = hybrid_mode_config.get('primary_strategy', 'entity_extraction')
        self.fallback_strategy = hybrid_mode_config.get('fallback_strategy', 'top_k_seed')
        self.switch_threshold = hybrid_mode_config.get('switch_threshold', 3)
        
        # 新增：多跳扩展配置（支持3/4跳）
        multi_hop_config = config.get('hybrid_search.multi_hop', {})
        self.max_hops = multi_hop_config.get('max_hops', 4)
        self.beam_width = multi_hop_config.get('beam_width', 8)
        self.per_hop_keep_top_m = multi_hop_config.get('per_hop_keep_top_m', 5)
        self.focused_weight_by_hop = multi_hop_config.get('focused_weight_by_hop', {
            1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15
        })
        self.hop_decay = multi_hop_config.get('hop_decay', 0.85)
        self.multi_hop_lower_threshold = multi_hop_config.get('lower_threshold', 0.10)
        
        # 答案偏置配置
        answer_bias_config = config.get('hybrid_search.answer_bias', {})
        self.who_person_boost = answer_bias_config.get('who_person_boost', 1.10)
        
        logger.info(f"Graph retrieval strategy: {self.graph_strategy}")
        logger.info(f"Top-K seed: {'enabled' if self.top_k_seed_enabled else 'disabled'} (count: {self.seed_count})")
        logger.info(f"Entity extraction: {'enabled' if self.entity_extraction_enabled else 'disabled'} (max: {self.max_entities})")
        if self.graph_strategy == 'hybrid':
            logger.info(f"Hybrid mode - Primary: {self.primary_strategy}, Fallback: {self.fallback_strategy}, Threshold: {self.switch_threshold}")
        
        # 记录多跳配置信息
        logger.info(f"Multi-hop configuration - Max hops: {self.max_hops}, Beam width: {self.beam_width}")
        logger.info(f"Per-hop keep top-M: {self.per_hop_keep_top_m}, Hop decay: {self.hop_decay}")
        logger.info(f"Multi-hop lower threshold: {self.multi_hop_lower_threshold}")
        logger.info(f"Answer bias - Who person boost: {self.who_person_boost}")
        
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
    
    def _extract_relation_focus_phrase(self, query: str) -> str:
        """
        从原始问句中抽出"关系焦点短语"：移除实体、去停用词与疑问词，保留核心内容词。
        不依赖具体谓词词表，适配 spouse/author/birthplace 等各种关系。
        """
        q = (query or "").strip()
        if not q:
            return ""

        # 1) 去除命名实体（如果 EnhancedNER 可用）
        text_wo_entities = q
        try:
            if hasattr(self, "enhanced_ner") and self.enhanced_ner:
                spans = self.enhanced_ner.find_entities_with_spans(q)
                if spans:
                    # spans: [(start, end, label), ...] 假设有这样的方法；若没有，退化为原文
                    mask = [True] * len(q)
                    for s, e, _lab in spans:
                        for i in range(s, e):
                            if 0 <= i < len(mask):
                                mask[i] = False
                    text_wo_entities = "".join([ch for i, ch in enumerate(q) if mask[i]])
        except Exception:
            text_wo_entities = q

        # 2) 粗略 token 化 + 停用词/疑问词过滤（不引入新依赖）
        import re
        tokens = re.findall(r"[A-Za-z]+", text_wo_entities.lower())
        stop = {
            "a","an","the","of","in","on","at","for","to","by","from",
            "and","or","as","is","are","was","were","be","been","being",
            "with","that","this","those","these","it","its","their","his","her",
            "who","whom","what","when","where","why","how","which"
        }
        content = [t for t in tokens if t not in stop and len(t) >= 2]

        # 限制长度，避免太长；保留前 4 个内容词
        if not content:
            return ""
        return " ".join(content[:4])

    def _infer_qtype(self, query: str) -> str:
        """
        非侵入式问题类型推断，仅用于轻量偏置：
        返回 {'who','when','where','what','other'} 之一
        """
        if not query:
            return "other"
        head = query.strip().split()[0].lower()
        if head in {"who"}:
            return "who"
        if head in {"when"}:
            return "when"
        if head in {"where"}:
            return "where"
        if head in {"what"}:
            return "what"
        return "other"
    
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
        """应用噪声阈值过滤，丢弃相似度小于0.20且不满足must_have_terms的候选。
        新增：对二跳候选实施分桶+保活策略。"""
        if not candidates:
            return candidates
        
        cfg = (self.config or {}).get("hybrid_search", {})
        sec_cfg = cfg.get("second_hop_safety", {})
        keep_top_m = int(sec_cfg.get("keep_top_m", 5))
        lower_th = float(sec_cfg.get("lower_threshold", 0.10))
        
        primary_bucket, second_bucket = [], []
        for c in candidates:
            if c.get("hop_type") == "second_hop":
                second_bucket.append(c)
            else:
                primary_bucket.append(c)
        
        # === 原有：对一跳照常用你的阈值与 must-have 规则 ===
        filtered_primary = []
        for c in primary_bucket:
            if not self._passes_noise_rules(c, must_have_terms):
                continue
            filtered_primary.append(c)
        
        # === 新增：二跳安全网 ===
        # 先按 final_score 排序
        second_bucket.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        # Top-M 无条件保活
        safe = second_bucket[:keep_top_m]
        
        # 其余用"放宽阈值 + must-have"再筛一次
        tail = second_bucket[keep_top_m:]
        filtered_tail = []
        for c in tail:
            # 放宽阈值（如果你原本在 _passes_noise_rules 里面有阈值，这里覆盖一下）
            score_ok = (c.get("final_score", 0.0) >= lower_th)
            must_ok = self._contains_any_term(c, must_have_terms) if must_have_terms else True
            if score_ok and must_ok:
                filtered_tail.append(c)
        
        logger.info(f"Noise threshold filtering: {len(candidates)} -> {len(filtered_primary + safe + filtered_tail)} candidates")
        logger.info(f"  Primary hop: {len(primary_bucket)} -> {len(filtered_primary)}")
        logger.info(f"  Second hop: {len(second_bucket)} -> {len(safe + filtered_tail)} (safe: {len(safe)}, filtered: {len(filtered_tail)})")
        
        return filtered_primary + safe + filtered_tail
    
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
    
    def _contains_any_term(self, cand: Dict[str, Any], terms: List[str]) -> bool:
        """检查候选是否包含任何指定的词汇。"""
        if not terms:
            return False
        blob = f"{cand.get('title','')} {cand.get('content','')}".lower()
        return any(t.lower() in blob for t in terms)
    
    def _passes_noise_rules(self, cand: Dict[str, Any], must_have_terms: List[str] = None) -> bool:
        """检查候选是否通过噪声过滤规则。"""
        # 你原有的一跳过滤逻辑；若没有，这里给个温和的默认
        th = float(getattr(self, "noise_threshold", 0.20))
        score_ok = cand.get("final_score", 0.0) >= th
        must_ok = self._contains_any_term(cand, must_have_terms) if must_have_terms else True
        return score_ok and must_ok
    
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
            
            # 应用ListT5重排序（如果启用）
            if self.listt5 and len(candidates) > 0:
                try:
                    # 记录ListT5前的状态
                    pre_listt5_topk = min(len(candidates), config.get('rerank.listt5_input_topk', 24))
                    pre_listt5_candidates = candidates[:pre_listt5_topk]
                    
                    # 计算ListT5前的多样性指标
                    pre_listt5_titles = set(c.get('title', '') for c in pre_listt5_candidates)
                    pre_listt5_diversity = len(pre_listt5_titles) / max(len(pre_listt5_candidates), 1)
                    
                    logger.debug(f"Applying ListT5 reranking to top {pre_listt5_topk} candidates (diversity: {pre_listt5_diversity:.3f})")
                    
                    # 获取ListT5分数
                    list_scores = self.listt5.score(query, pre_listt5_candidates)
                    
                    # 融合分数
                    calibration_config = config.get('calibration', {})
                    fused_candidates = fuse_scores(
                        pre_listt5_candidates, 
                        list_scores, 
                        weights={'listt5_weight': calibration_config.get('listt5_weight', 0.35)}
                    )
                    
                    # 重新排序
                    fused_candidates = sort_desc(fused_candidates, 'fused_score')
                    
                    # 保留指定数量的候选
                    keep_after_listt5 = config.get('rerank.keep_after_listt5', 16)
                    final_candidates = fused_candidates[:keep_after_listt5]
                    
                    # 添加剩余的候选（如果有）
                    if len(candidates) > pre_listt5_topk:
                        final_candidates.extend(candidates[pre_listt5_topk:])
                    
                    candidates = final_candidates
                    
                    # 记录ListT5后的状态和性能指标
                    post_listt5_topk = len(final_candidates)
                    post_listt5_titles = set(c.get('title', '') for c in final_candidates[:keep_after_listt5])
                    post_listt5_diversity = len(post_listt5_titles) / max(keep_after_listt5, 1)
                    
                    # 计算排序变化
                    pre_order = [c.get('note_id', i) for i, c in enumerate(pre_listt5_candidates)]
                    post_order = [c.get('note_id', i) for i, c in enumerate(final_candidates[:keep_after_listt5])]
                    reorder_ratio = len(set(pre_order[:keep_after_listt5]) - set(post_order)) / max(keep_after_listt5, 1)
                    
                    logger.info(f"ListT5 reranking metrics - Input: {pre_listt5_topk}, Output: {post_listt5_topk}, "
                              f"Pre-diversity: {pre_listt5_diversity:.3f}, Post-diversity: {post_listt5_diversity:.3f}, "
                              f"Reorder ratio: {reorder_ratio:.3f}")
                    
                    # 记录分数分布
                    if list_scores:
                        avg_listt5_score = sum(list_scores) / len(list_scores)
                        max_listt5_score = max(list_scores)
                        min_listt5_score = min(list_scores)
                        logger.debug(f"ListT5 scores - Avg: {avg_listt5_score:.4f}, Max: {max_listt5_score:.4f}, Min: {min_listt5_score:.4f}")
                    
                except Exception as e:
                    logger.error(f"ListT5 reranking failed: {e}")
            
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
            second_hop_notes = self._rerank_khop_candidates(query, candidate_pool)
            
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
                    # === 数据结构约定：补充路径信息 ===
                    candidate['bridge_entity'] = entity
                    candidate['hop_type'] = 'second_hop'  # 标记为二跳
                    candidate['hop_no'] = 2  # 第二跳
                    candidate['bridge_path'] = [entity]  # 从起点到此的实体链（二跳只有一个桥接实体）
                    
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
                                # === 数据结构约定：补充路径信息 ===
                                candidate['bridge_entity'] = entity
                                candidate['hop_type'] = 'second_hop'  # 标记为二跳
                                candidate['hop_no'] = 2  # 第二跳
                                candidate['bridge_path'] = [entity]  # 从起点到此的实体链（二跳只有一个桥接实体）
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
        """
        对二跳候选池进行重排。
        
        注意：此函数已被更通用的 _rerank_khop_candidates 替代，保留此函数仅为向后兼容性。
        新代码应使用 _rerank_khop_candidates(query, candidates) 替代。
        
        保持原有的 dense/bm25/path 融合，在最后融合额外信号：
        - RFR：对每个候选，用 "bridge_entity + relation_focus" 再算一次相似度并融合
        - Who 答案类型偏置：候选文本含 PERSON 名称时小幅加乘
        """
        # 为向后兼容，调用新的通用函数
        return self._rerank_khop_candidates(query, candidates)
    
    def _rerank_khop_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对包含 hop_no 的多跳候选统一重排：base_score + relation-focused + answer-bias + hop_decay
        
        运行时开销控制：
        1. Beam/Top-M 限定：beam_width 控制每跳扩展的实体数，per_hop_keep_top_m 控制每跳进入后续步骤的候选数
        2. Focused 相似度计算规模：仅对"进入重排的候选"各算 1 次向量相似度
        3. 路径查询长度：focused_query 仅取路径末端2个实体 + 关系短语，避免 query 过长
        4. 提前剪枝：如果某一跳没有产生新实体/候选，立即停止扩展（early stop）
        """
        if not candidates:
            return []

        cfg = (self.config or {}).get("hybrid_search", {})
        mh_cfg = cfg.get("multi_hop", {})
        fw_by_hop = mh_cfg.get("focused_weight_by_hop", {})
        hop_decay = float(mh_cfg.get("hop_decay", 0.85))
        beam_width = int(mh_cfg.get("beam_width", 8))
        per_hop_keep_top_m = int(mh_cfg.get("per_hop_keep_top_m", 5))

        bias_cfg = cfg.get("answer_bias", {})
        who_person_boost = float(bias_cfg.get("who_person_boost", 1.10))

        # === 运行时开销控制：Beam/Top-M 限定 ===
        # 按跳数分组，每跳最多保留 beam_width 个候选进入重排
        from collections import defaultdict
        hop_buckets = defaultdict(list)
        for cand in candidates:
            hop_no = int(cand.get("hop_no", 1))
            hop_buckets[hop_no].append(cand)

        # 对每跳应用 beam_width 限制
        filtered_candidates = []
        for hop_no, bucket in hop_buckets.items():
            # 按现有分数排序（如果有的话）
            bucket.sort(key=lambda x: x.get("final_score", x.get("score", 0.0)), reverse=True)
            # 应用 beam_width 限制
            beam_limited = bucket[:beam_width]
            filtered_candidates.extend(beam_limited)
            
            if len(bucket) > beam_width:
                logger.info(f"Beam limiting hop {hop_no}: {len(bucket)} -> {len(beam_limited)} candidates")

        # === 提前剪枝检查 ===
        # 检查是否有新实体/候选产生，如果没有则可以提前停止
        unique_entities = set()
        for cand in filtered_candidates:
            if cand.get("bridge_entity"):
                unique_entities.add(cand["bridge_entity"])
            for entity in (cand.get("bridge_path") or []):
                unique_entities.add(entity)
        
        if len(unique_entities) == 0:
            logger.info("Early stopping: No new entities found in candidates")
            return []

        # 基础分（与原问句相似度），尽量复用你现有的 dense/bm25 逻辑
        dense_scores = self._calculate_vector_similarities(query, filtered_candidates)
        bm25_scores = self._calculate_bm25_similarities(query, filtered_candidates)

        relation_focus = self._extract_relation_focus_phrase(query)
        qtype = self._infer_qtype(query)

        # === Focused 相似度计算规模控制 ===
        # 仅对进入重排的候选计算 focused 相似度，且仅计算一次
        focused_scores = {}
        if relation_focus:
            focused_candidates_for_calc = []
            focused_queries = []
            
            for i, cand in enumerate(filtered_candidates):
                hop_no = int(cand.get("hop_no", 1))
                alpha = float(fw_by_hop.get(str(hop_no), fw_by_hop.get("2", 0.25)))
                
                # 深跳的 focused_weight 更小 + hop_decay 更重
                if hop_no > 2:
                    alpha *= 0.8  # 深跳衰减
                
                if alpha > 0.0:
                    path_entities = cand.get("bridge_path") or []
                    if not path_entities and cand.get("bridge_entity"):
                        path_entities = [cand["bridge_entity"]]
                    
                    if path_entities:
                        # === 路径查询长度控制：仅取路径末端2个实体 + 关系短语 ===
                        tail = path_entities[-2:]  # 只取最后 2 个，控制开销
                        focused_query = " ".join(tail + [relation_focus])
                        focused_candidates_for_calc.append(cand)
                        focused_queries.append(focused_query)
                        focused_scores[i] = {"alpha": alpha, "query": focused_query}
            
            # 批量计算 focused 相似度（如果有需要计算的）
            if focused_candidates_for_calc and focused_queries:
                try:
                    # 为了进一步控制开销，只对 Top-K（如32）候选计算 focused sim
                    max_focused_calc = min(32, len(focused_candidates_for_calc))
                    if len(focused_candidates_for_calc) > max_focused_calc:
                        # 按基础分数排序，只对前32个计算focused相似度
                        temp_scores = []
                        for i, cand in enumerate(focused_candidates_for_calc):
                            base_score = (dense_scores[filtered_candidates.index(cand)] if cand in filtered_candidates else 0.0) + \
                                       (bm25_scores[filtered_candidates.index(cand)] if cand in filtered_candidates else 0.0)
                            temp_scores.append((base_score, i, cand))
                        temp_scores.sort(reverse=True)
                        
                        top_focused_candidates = [item[2] for item in temp_scores[:max_focused_calc]]
                        top_focused_queries = [focused_queries[item[1]] for item in temp_scores[:max_focused_calc]]
                        
                        focused_sims = self._calculate_vector_similarities_batch(top_focused_queries, top_focused_candidates)
                        
                        # 更新 focused_scores
                        for j, (_, orig_i, _) in enumerate(temp_scores[:max_focused_calc]):
                            if orig_i < len(focused_candidates_for_calc):
                                cand_idx = filtered_candidates.index(focused_candidates_for_calc[orig_i])
                                focused_scores[cand_idx]["similarity"] = focused_sims[j] if j < len(focused_sims) else 0.0
                    else:
                        focused_sims = self._calculate_vector_similarities_batch(focused_queries, focused_candidates_for_calc)
                        for j, cand in enumerate(focused_candidates_for_calc):
                            cand_idx = filtered_candidates.index(cand)
                            focused_scores[cand_idx]["similarity"] = focused_sims[j] if j < len(focused_sims) else 0.0
                            
                except Exception as e:
                    logger.warning(f"Failed to calculate focused similarities: {e}")
                    focused_scores = {}

        reranked = []
        for i, cand in enumerate(filtered_candidates):
            dense = dense_scores[i] if i < len(dense_scores) else 0.0
            sparse = bm25_scores[i] if i < len(bm25_scores) else 0.0

            # 你已有的惩罚/加成（若无就当1.0）
            try:
                pred_pen = self._calculate_predicate_penalty(cand)
            except Exception:
                pred_pen = 1.0
            dense *= pred_pen
            sparse *= pred_pen

            base_final = 1.0 * dense + 0.6 * sparse   # 保持你当前融合

            # === 关系聚焦（多跳版，使用预计算的结果）===
            focused_final = base_final
            if i in focused_scores and "similarity" in focused_scores[i]:
                alpha = focused_scores[i]["alpha"]
                focused_sim = focused_scores[i]["similarity"]
                focused_final = (1.0 - alpha) * base_final + alpha * focused_sim

            final_score = focused_final

            # === 答案类型偏置（轻量通用）===
            if qtype == "who" and who_person_boost > 1.0:
                text = f"{cand.get('content','')} {cand.get('title','')}"
                try:
                    has_person = False
                    if hasattr(self, "enhanced_ner") and self.enhanced_ner:
                        has_person = self.enhanced_ner.contains_person_name(text)
                    else:
                        import re
                        has_person = bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text))
                    if has_person:
                        final_score *= who_person_boost
                except Exception:
                    pass

            # === 跳数惩罚（越远越弱，避免噪声）===
            hop_no = int(cand.get("hop_no", 1))
            if hop_no > 1 and 0.0 < hop_decay < 1.0:
                final_score *= (hop_decay ** (hop_no - 1))

            # === 路径一致性与协同打分（可选但推荐，仍然通用）===
            consistency_bonus = 1.0
            be = cand.get("bridge_entity", "")
            text = f"{cand.get('title','')} {cand.get('content','')}".lower()
            if be and be.lower() in text:
                consistency_bonus *= 1.05   # 5% 小奖励
            if relation_focus:
                rf_tokens = relation_focus.split()
                if any(tok in text for tok in rf_tokens):
                    consistency_bonus *= 1.03
            final_score *= consistency_bonus

            cand["final_score"] = float(final_score)
            reranked.append(cand)

        # 最终排序
        reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        # === 应用 per_hop_keep_top_m 限制 ===
        # 重新按跳数分组，每跳保留 top_m
        final_hop_buckets = defaultdict(list)
        for cand in reranked:
            hop_no = int(cand.get("hop_no", 1))
            final_hop_buckets[hop_no].append(cand)

        final_candidates = []
        for hop_no, bucket in final_hop_buckets.items():
            kept = bucket[:per_hop_keep_top_m]
            final_candidates.extend(kept)
            if len(bucket) > per_hop_keep_top_m:
                logger.info(f"Per-hop top-M limiting hop {hop_no}: {len(bucket)} -> {len(kept)} candidates")

        # 重新排序最终结果
        final_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        logger.info(f"Multi-hop reranking completed: {len(candidates)} -> {len(filtered_candidates)} -> {len(final_candidates)} candidates, "
                   f"relation_focus='{relation_focus}', qtype={qtype}, hop_decay={hop_decay}, "
                   f"beam_width={beam_width}, per_hop_keep_top_m={per_hop_keep_top_m}")
        
        return final_candidates
    
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
    
    def _log_subquestion_debug_info(self, sub_question: str, index: int, must_have_entities: List[str], 
                                   constraint_words: List[str], expanded_query: str, 
                                   vector_notes: List[Dict[str, Any]], graph_notes: List[Dict[str, Any]], 
                                   hit_entities: Dict[str, int], dataset: Optional[str] = None, 
                                   qid: Optional[str] = None, fallback_triggered: bool = False, 
                                   fallback_candidates: List[Dict[str, Any]] = None):
        """记录子问题的详细调试信息。"""
        try:
            # 处理回补候选
            if fallback_candidates is None:
                fallback_candidates = []
            
            # 计算回补情况
            min_per_subq = config.get('retrieval.min_per_subq', 1)
            initial_candidates_count = len(vector_notes) + len(graph_notes)
            
            # 统计实体命中情况
            total_entities = len(must_have_entities)
            hit_entity_count = len([entity for entity, count in hit_entities.items() if count > 0])
            entity_coverage_rate = hit_entity_count / total_entities if total_entities > 0 else 1.0
            
            # 分析候选来源分布
            vector_sources = {}
            graph_sources = {}
            fallback_sources = {}
            
            for note in vector_notes:
                source = note.get('retrieval_source', 'vector_default')
                vector_sources[source] = vector_sources.get(source, 0) + 1
            
            for note in graph_notes:
                source = note.get('retrieval_source', 'graph_default')
                graph_sources[source] = graph_sources.get(source, 0) + 1
            
            for note in fallback_candidates:
                source = note.get('retrieval_source', 'fallback_default')
                fallback_sources[source] = fallback_sources.get(source, 0) + 1
            
            # 收集Top5候选的详细信息
            all_candidates = vector_notes + graph_notes + fallback_candidates
            top_candidates_details = []
            for i, note in enumerate(all_candidates[:5]):
                detail = {
                    'rank': i + 1,
                    'source': note.get('subq_source', 'unknown'),
                    'file_name': note.get('file_name', 'unknown'),
                    'entities': note.get('entities', [])[:3],  # 只记录前3个实体
                    'similarity': note.get('similarity', 0.0),
                    'final_score': note.get('final_score', 0.0),
                    'entity_overlap_score': note.get('entity_overlap_score', 0.0),
                    'path_consistency_score': note.get('path_consistency_score', 0.0)
                }
                top_candidates_details.append(detail)
            
            # 记录结构化调试日志
            self.structured_logger.info(
                "Sub-question retrieval debug info",
                dataset=dataset,
                qid=qid,
                sub_question_index=index,
                sub_question=sub_question,
                expanded_query=expanded_query,
                must_have_entities=must_have_entities,
                constraint_words=constraint_words,
                total_entities=total_entities,
                hit_entity_count=hit_entity_count,
                entity_coverage_rate=round(entity_coverage_rate, 3),
                entity_hits=hit_entities,
                vector_candidates_count=len(vector_notes),
                graph_candidates_count=len(graph_notes),
                fallback_candidates_count=len(fallback_candidates),
                total_candidates_count=len(all_candidates),
                fallback_triggered=fallback_triggered,
                min_per_subq_threshold=min_per_subq,
                vector_sources=vector_sources,
                graph_sources=graph_sources,
                fallback_sources=fallback_sources,
                top_candidates_details=top_candidates_details
            )
            
            # 如果实体覆盖率低，记录警告
            if entity_coverage_rate < 0.5 and total_entities > 0:
                logger.warning(f"Sub-question {index}: Low entity coverage rate {entity_coverage_rate:.2f}, "
                             f"only {hit_entity_count}/{total_entities} entities found in candidates")
            
            # 如果触发了回补，记录详细信息
            if fallback_triggered:
                logger.info(f"Sub-question {index}: Fallback retrieval triggered, "
                           f"initial candidates: {initial_candidates_count}, threshold: {min_per_subq}, "
                           f"fallback retrieved: {len(fallback_candidates)}")
            
        except Exception as e:
            logger.error(f"Failed to log sub-question debug info for index {index}: {e}")
    
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
                    # 为向量检索结果添加subq_source标记
                    if 'tags' not in note:
                        note['tags'] = {}
                    note['tags']['subq_source'] = 'vector'
                    
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
            
            # 如果启用了多跳检索，添加图检索结果
            logger.info(f"Multi-hop enabled: {self.multi_hop_enabled}")
            if self.multi_hop_enabled:
                try:
                    # 获取查询的嵌入向量
                    query_embeddings = self.vector_retriever.embedding_manager.encode_queries([query])
                    query_emb = query_embeddings[0] if len(query_embeddings) > 0 else None
                    
                    if query_emb is not None:
                        # 调用多跳处理器进行图检索
                        graph_results = self.multi_hop_processor.retrieve(query_emb)
                    else:
                        logger.warning("Failed to generate query embedding for graph retrieval")
                        graph_results = {'notes': []}
                    graph_notes = graph_results.get('notes', [])
                    
                    # 为图检索结果添加标记
                    for note in graph_notes:
                        if 'tags' not in note:
                            note['tags'] = {}
                        note['tags']['source'] = 'graph'
                        note['tags']['subq_source'] = 'graph'  # 添加subq_source标记
                        
                        # 为图检索候选添加hop_no和bridge_path字段
                        # 图检索可能是多跳的，需要根据实际情况设置
                        if 'hop_no' not in note:
                            note['hop_type'] = 'graph_hop'
                            note['hop_no'] = 1  # 图检索默认为1跳，多跳信息由multi_hop_processor提供
                            note['bridge_entity'] = None
                            note['bridge_path'] = []
                        
                        # 确保有final_similarity字段
                        if 'final_similarity' not in note:
                            note['final_similarity'] = note.get('retrieval_info', {}).get('score', 0.0)
                    
                    # 将图检索结果添加到候选结果中
                    for note in graph_notes:
                        # 为图检索结果添加subq_source标记
                        if 'tags' not in note:
                            note['tags'] = {}
                        note['tags']['subq_source'] = 'graph'
                        
                        note_id = note.get('note_id')
                        if note_id and note_id not in seen_note_ids:
                            candidate_notes.append(note)
                            seen_note_ids.add(note_id)
                    
                    logger.info(f"Added {len(graph_notes)} graph retrieval results")
                    
                except Exception as e:
                    logger.error(f"Graph retrieval failed: {e}")
            
            # 调用 ContextDispatcher 处理候选结果
            selected_notes = self.context_dispatcher.dispatch(candidate_notes, query=query)
            
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
            
            # === 新增：把二跳候选中的桥接实体写进 must-have，防止被误杀 ===
            bridge_terms = []
            for n in second_hop_notes:  # 二跳候选集合
                be = n.get("bridge_entity")
                if be:
                    bridge_terms.append(be.lower())
            
            if bridge_terms:
                ml = set(t.lower() for t in (must_have_terms or []))
                ml.update(bridge_terms)
                must_have_terms = list(ml)
                logger.info(f"Added {len(bridge_terms)} bridge entities to must_have_terms: {bridge_terms}")
            
            candidate_notes = self._filter_with_multihop_safety(candidate_notes, query)
            
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
                # === 新增：把二跳候选中的桥接实体写进 must-have，防止被误杀 ===
                bridge_terms = []
                for n in second_hop_notes:  # 二跳候选集合
                    be = n.get("bridge_entity")
                    if be:
                        bridge_terms.append(be.lower())
                
                if bridge_terms:
                    ml = set(t.lower() for t in (must_have_terms or []))
                    ml.update(bridge_terms)
                    must_have_terms = list(ml)
                    logger.info(f"Added {len(bridge_terms)} bridge entities to must_have_terms for merged candidates: {bridge_terms}")
                
                candidate_notes = self._filter_with_multihop_safety(candidate_notes, query)
            
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

            # 7. 图扩展（混合策略）
            reasoning_paths: List[Dict[str, Any]] = []
            if self.multi_hop_enabled:
                mh_result = self.multi_hop_processor.retrieve(query_emb)
                graph_notes = mh_result.get('notes', [])
                
                # 为multi_hop_processor的图检索结果添加hop_no和bridge_path字段
                for note in graph_notes:
                    if 'hop_no' not in note:
                        note['hop_type'] = 'graph_hop'
                        note['hop_no'] = 1  # 默认为1跳，多跳信息由multi_hop_processor提供
                        note['bridge_entity'] = None
                        note['bridge_path'] = []
                
                candidate_notes.extend(graph_notes)
                for n in graph_notes:
                    reasoning_paths.extend(n.get('reasoning_paths', []))
                
                # 应用多跳安全网过滤，替代传统的噪声阈值过滤
                logger.info(f"Before multi-hop safety filtering: {len(candidate_notes)} candidates")
                candidate_notes = self._filter_with_multihop_safety(candidate_notes, query)
                logger.info(f"After multi-hop safety filtering: {len(candidate_notes)} candidates")
                
                selected_notes = self.scheduler.schedule_for_multi_hop(candidate_notes, reasoning_paths)
            else:
                # 使用混合图检索策略
                graph_notes = self._perform_hybrid_graph_retrieval(candidate_notes, query)
                
                # 为混合图检索结果添加hop_no和bridge_path字段
                for note in graph_notes:
                    if 'hop_no' not in note:
                        note['hop_type'] = 'graph_hop'
                        note['hop_no'] = 1  # 混合图检索默认为1跳
                        note['bridge_entity'] = None
                        note['bridge_path'] = []
                
                candidate_notes.extend(graph_notes)
                
                # 应用多跳安全网过滤，替代传统的噪声阈值过滤
                logger.info(f"Before multi-hop safety filtering (non-multi-hop): {len(candidate_notes)} candidates")
                candidate_notes = self._filter_with_multihop_safety(candidate_notes, query)
                logger.info(f"After multi-hop safety filtering (non-multi-hop): {len(candidate_notes)} candidates")
                
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
        # 生成答案和评分
        # 使用新的 build_context_prompt_with_passages 函数生成带有 [P{idx}] 标签的上下文和passages字典
        prompt, passages_by_idx, packed_order = build_context_prompt_with_passages(selected_notes, query)
        raw_answer = self.ollama.generate_final_answer(prompt)
        
        # 使用鲁棒的JSON解析器，带重试机制
        def retry_generate():
            return self.ollama.generate_final_answer(prompt)
        
        # 从配置获取重试参数
        json_parsing_config = config.get('retrieval.json_parsing', {})
        max_retries = json_parsing_config.get('max_retries', 3)
        
        answer, raw_support_idxs = extract_prediction_with_retry(
            raw_answer, passages_by_idx, retry_func=retry_generate, max_retries=max_retries
        )
        
        # 写盘前的最终合法性校验：再次过滤幽灵id并补齐
        # 第二道兜底：确保最终结果没有幽灵id
        filtered_support_idxs = [i for i in raw_support_idxs if i in passages_by_idx]
        
        # 若过滤后为空且上下文中存在任何包含答案子串的段，用其中一个合法id顶上第一位
        if not filtered_support_idxs and answer:
            for idx, content in passages_by_idx.items():
                if answer in content:
                    filtered_support_idxs = [idx]
                    break
        
        # 使用支持段落补齐模块对LLM输出进行结构化补齐/纠偏
        from utils.support_fill import fill_support_idxs_noid
        predicted_support_idxs = fill_support_idxs_noid(
            question=query,
            answer=answer, 
            raw_support_idxs=filtered_support_idxs,  # 使用过滤后的合法id
            passages_by_idx=passages_by_idx,
            packed_order=packed_order
        )
        
        # 评估答案质量
        context = "\n".join(n.get('content','') for n in selected_notes)
        scores = self.ollama.evaluate_answer(query, context, answer)
        
        # 为笔记添加反馈分数
        for n in selected_notes:
            n['feedback_score'] = scores.get('relevance', 0)

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
            # 使用新的 build_context_prompt_with_passages 函数生成带有 [P{idx}] 标签的上下文和passages字典
            prompt, passages = build_context_prompt_with_passages(selected_notes, query)
            raw_answer = self.ollama.generate_final_answer(prompt)
            
            # 使用鲁棒的JSON解析器，带重试机制
            def retry_generate():
                return self.ollama.generate_final_answer(prompt)
            
            answer, predicted_support_idxs = extract_prediction_with_retry(
            raw_answer, passages, retry_func=retry_generate, max_retries=3
        )
            
            # Step 6: Evaluate answer and collect feedback
            context = "\n".join(n.get('content', '') for n in selected_notes)
            scores = self.ollama.evaluate_answer(query, context, answer)
            
            # 为笔记添加反馈分数
            for n in selected_notes:
                n['feedback_score'] = scores.get('relevance', 0)
            
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
        """Retrieve evidence for a single sub-question with coverage-driven recall."""
        # Step 1: Extract key entities and constraint words from sub-question
        must_have_entities = self._extract_key_entities_from_subquestion(sub_question)
        constraint_words = self._extract_constraint_words_from_subquestion(sub_question)
        
        # Step 2: Query expansion - inject entities and constraints into search query
        expanded_query = self._expand_subquestion_query(sub_question, must_have_entities, constraint_words)
        
        logger.debug(f"Sub-question {index}: '{sub_question}' -> expanded: '{expanded_query}'")
        logger.debug(f"Sub-question {index}: must_have_entities: {must_have_entities}")
        logger.debug(f"Sub-question {index}: constraint_words: {constraint_words}")
        
        # Step 3: Initial vector retrieval with expanded query
        vector_results = self.vector_retriever.search([expanded_query])
        vector_notes = vector_results[0] if vector_results else []
        
        # 为一跳候选添加hop_no和bridge_path字段
        for note in vector_notes:
            note['hop_type'] = 'first_hop'
            note['hop_no'] = 1
            note['bridge_entity'] = None  # 一跳没有桥接实体
            note['bridge_path'] = []  # 一跳的路径为空
        
        logger.info(f"Sub-question {index}: '{sub_question}' initial vector recall: {len(vector_notes)} notes")
        
        # Step 4: Apply minimum candidate constraint (k_min=1)
        min_per_subq = config.get('retrieval.min_per_subq', 1)
        if len(vector_notes) < min_per_subq:
            logger.warning(f"Sub-question {index}: Insufficient candidates ({len(vector_notes)} < {min_per_subq}), triggering fallback retrieval")
            vector_notes = self._fallback_retrieval_for_subquestion(sub_question, expanded_query, must_have_entities, dataset, qid)
        
        # 第一阶段命名空间守卫：子问题向量召回后
        if self.namespace_guard_enabled and dataset and qid:
            vector_notes = filter_notes_by_namespace(vector_notes, dataset, qid)
            logger.info(f"Sub-question {index}: After namespace filtering: {len(vector_notes)} notes")
        
        # Apply hybrid search if enabled
        if self.hybrid_search_enabled and vector_notes:
            try:
                # 生成检索守卫参数，融合must_have_entities
                must_have_terms, boost_entities, boost_predicates = self._generate_enhanced_guardrail_params(sub_question, must_have_entities)
                
                # 调用增强混合检索
                vector_notes = self._enhanced_hybrid_search(
                    expanded_query, vector_notes,
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
        
        # Step 5: Graph retrieval with entity-anchored expansion
        seed_ids = [note.get('note_id') for note in vector_notes if note.get('note_id')]
        graph_notes = self._entity_anchored_graph_retrieval(seed_ids, must_have_entities) if seed_ids else []
        
        # 第三阶段命名空间守卫：子问题图扩展后
        if self.namespace_guard_enabled and dataset and qid:
            graph_notes = filter_notes_by_namespace(graph_notes, dataset, qid)
            logger.info(f"Sub-question {index}: After graph expansion namespace filtering: {len(graph_notes)} notes")
        
        # Step 6: Check if fallback retrieval is needed
        min_per_subq = config.get('retrieval.min_per_subq', 1)
        initial_candidates_count = len(vector_notes) + len(graph_notes)
        fallback_candidates = []
        fallback_triggered = False
        
        if initial_candidates_count < min_per_subq:
            fallback_triggered = True
            logger.info(f"Sub-question {index}: Triggering fallback retrieval, current: {initial_candidates_count}, required: {min_per_subq}")
            fallback_candidates = self._fallback_retrieval_for_subquestion(
                sub_question=sub_question,
                expanded_query=expanded_query,
                must_have_entities=must_have_entities,
                dataset=dataset,
                qid=qid
            )
            
            # 第四阶段命名空间守卫：回补检索后
            if self.namespace_guard_enabled and dataset and qid:
                fallback_candidates = filter_notes_by_namespace(fallback_candidates, dataset, qid)
                logger.info(f"Sub-question {index}: After fallback namespace filtering: {len(fallback_candidates)} notes")
        
        # Step 7: Tag candidates with subq_id for later integrity checking
        for note in vector_notes:
            note['subq_id'] = index
            note['subq_source'] = 'vector'
        for note in graph_notes:
            note['subq_id'] = index
            note['subq_source'] = 'graph'
        for note in fallback_candidates:
            note['subq_id'] = index
            note['subq_source'] = 'fallback'
        
        # Step 8: Log structured debug information
        all_candidates = vector_notes + graph_notes + fallback_candidates
        hit_entities = self._count_entity_hits_in_candidates(all_candidates, must_have_entities)
        
        # 记录详细的子问题调试信息
        self._log_subquestion_debug_info(
            sub_question=sub_question,
            index=index,
            must_have_entities=must_have_entities,
            constraint_words=constraint_words,
            expanded_query=expanded_query,
            vector_notes=vector_notes,
            graph_notes=graph_notes,
            hit_entities=hit_entities,
            dataset=dataset,
            qid=qid,
            fallback_triggered=fallback_triggered,
            fallback_candidates=fallback_candidates
        )
        
        logger.info(f"Sub-question {index}: '{sub_question}' final retrieved {len(vector_notes)} vector + {len(graph_notes)} graph + {len(fallback_candidates)} fallback notes")
        
        return {
            'sub_question': sub_question,
            'sub_question_index': index,
            'vector_results': vector_notes,
            'graph_results': graph_notes,
            'fallback_results': fallback_candidates,
            'must_have_entities': must_have_entities,
            'constraint_words': constraint_words,
            'expanded_query': expanded_query,
            'entity_hits': hit_entities,
            'fallback_triggered': fallback_triggered
        }
    
    def _extract_key_entities_from_subquestion(self, sub_question: str) -> List[str]:
        """Extract key entities from sub-question for must-have constraints."""
        # Use existing NER functionality
        entities = self._perform_ner_on_text(sub_question)
        # Normalize entities
        normalized_entities = self._normalize_entities(entities)
        # Filter for key entities (proper nouns, locations, organizations)
        key_entities = []
        for entity in normalized_entities:
            if len(entity) > 2 and entity[0].isupper():  # Basic proper noun filter
                key_entities.append(entity)
        return key_entities[:5]  # Limit to top 5 key entities
    
    def _extract_constraint_words_from_subquestion(self, sub_question: str) -> List[str]:
        """Extract constraint words (important terms) from sub-question."""
        # Extract predicates and important terms
        predicates = self._extract_predicates_from_text(sub_question)
        # Add some domain-specific constraint words
        constraint_patterns = ['boundary', 'dynasty', 'expelled', 'independence', 'defeated', 'regrouped']
        constraint_words = []
        
        # Add predicates
        constraint_words.extend(predicates[:3])  # Top 3 predicates
        
        # Add pattern matches
        sub_lower = sub_question.lower()
        for pattern in constraint_patterns:
            if pattern in sub_lower:
                constraint_words.append(pattern)
        
        return list(set(constraint_words))  # Remove duplicates
    
    def _expand_subquestion_query(self, sub_question: str, must_have_entities: List[str], constraint_words: List[str]) -> str:
        """Expand sub-question query by injecting key entities and constraint words."""
        # Start with original sub-question
        expanded_parts = [sub_question]
        
        # Add must-have entities
        if must_have_entities:
            expanded_parts.extend(must_have_entities)
        
        # Add constraint words
        if constraint_words:
            expanded_parts.extend(constraint_words)
        
        # Join with spaces, avoiding duplicates
        expanded_query = ' '.join(expanded_parts)
        return expanded_query
    
    def _fallback_retrieval_for_subquestion(self, sub_question: str, expanded_query: str, must_have_entities: List[str], dataset: Optional[str] = None, qid: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform fallback retrieval when initial retrieval fails to meet minimum threshold."""
        fallback_candidates = []
        # 从配置获取性能控制参数
        max_fallback_per_step = config.get('retrieval.performance.max_fallback_per_step', 3)
        max_total_fallback = config.get('retrieval.performance.max_total_fallback', 10)
        fallback_timeout_ms = config.get('retrieval.performance.fallback_timeout_ms', 1000)
        min_per_subq = config.get('retrieval.min_per_subq', 1)
        
        start_time = time.time() * 1000  # 转换为毫秒
        
        # 记录回补检索的详细步骤
        fallback_steps = []
        
        # Step 1: BM25 precise short query
        bm25_success = False
        bm25_count = 0
        
        # 检查超时
        if (time.time() * 1000 - start_time) > fallback_timeout_ms:
            logger.warning(f"Fallback retrieval timeout after {fallback_timeout_ms}ms")
            return fallback_candidates[:max_total_fallback]
        
        try:
            bm25_results = self._bm25_fallback_search(sub_question, dataset or '', qid or '', top_k=max_fallback_per_step)
            if bm25_results:
                bm25_count = len(bm25_results)
                # 限制总候选数量
                available_slots = max_total_fallback - len(fallback_candidates)
                bm25_results = bm25_results[:available_slots]
                bm25_count = len(bm25_results)
                fallback_candidates.extend(bm25_results)
                bm25_success = True
                # 标记候选来源
                for note in bm25_results:
                    note['retrieval_source'] = 'bm25_fallback'
                    # 为fallback候选添加hop_no和bridge_path字段
                    note['hop_type'] = 'first_hop'
                    note['hop_no'] = 1
                    note['bridge_entity'] = None  # fallback候选没有桥接实体
                    note['bridge_path'] = []  # fallback候选的路径为空
                logger.debug(f"Fallback step 1 (BM25): retrieved {bm25_count} candidates")
        except Exception as e:
            logger.error(f"BM25 fallback failed: {e}")
        
        fallback_steps.append({
            'step': 1,
            'method': 'BM25',
            'success': bm25_success,
            'candidates_retrieved': bm25_count,
            'total_after_step': len(fallback_candidates)
        })
        
        # Step 2: Embedding semantic completion (if still insufficient)
        vector_success = False
        vector_count = 0
        if len(fallback_candidates) < min_per_subq and len(fallback_candidates) < max_total_fallback:
            # 检查超时
            if (time.time() * 1000 - start_time) > fallback_timeout_ms:
                logger.warning(f"Fallback retrieval timeout after {fallback_timeout_ms}ms")
                return fallback_candidates[:max_total_fallback]
            
            try:
                vector_results = self.vector_retriever.search([expanded_query], top_k=max_fallback_per_step)
                vector_notes = vector_results[0] if vector_results else []
                # Filter out duplicates
                existing_ids = {note.get('note_id') for note in fallback_candidates}
                new_vector_notes = [note for note in vector_notes if note.get('note_id') not in existing_ids]
                # 限制总候选数量
                available_slots = max_total_fallback - len(fallback_candidates)
                new_vector_notes = new_vector_notes[:available_slots]
                vector_count = len(new_vector_notes)
                fallback_candidates.extend(new_vector_notes)
                vector_success = True
                # 标记候选来源
                for note in new_vector_notes:
                    note['retrieval_source'] = 'vector_fallback'
                    # 为fallback候选添加hop_no和bridge_path字段
                    note['hop_type'] = 'first_hop'
                    note['hop_no'] = 1
                    note['bridge_entity'] = None  # fallback候选没有桥接实体
                    note['bridge_path'] = []  # fallback候选的路径为空
                logger.debug(f"Fallback step 2 (Vector): retrieved {vector_count} new candidates")
            except Exception as e:
                logger.error(f"Vector fallback failed: {e}")
        
        fallback_steps.append({
            'step': 2,
            'method': 'Vector',
            'success': vector_success,
            'candidates_retrieved': vector_count,
            'total_after_step': len(fallback_candidates),
            'skipped': len(fallback_candidates) >= min_per_subq
        })
        
        # Step 3: Graph neighborhood expansion (if still insufficient)
        graph_success = False
        graph_count = 0
        if len(fallback_candidates) < min_per_subq and must_have_entities:
            # 检查超时
            elapsed_time = (time.time() - start_time) * 1000
            if elapsed_time >= fallback_timeout_ms:
                logger.warning(f"Fallback retrieval timeout ({elapsed_time:.1f}ms >= {fallback_timeout_ms}ms), skipping graph expansion")
                fallback_steps.append({
                    'step': 3,
                    'method': 'Graph',
                    'success': False,
                    'candidates_retrieved': 0,
                    'total_after_step': len(fallback_candidates),
                    'skipped': True,
                    'reason': 'timeout'
                })
            else:
                try:
                    # 计算剩余可用的候选数量
                    remaining_slots = max_total_fallback - len(fallback_candidates)
                    actual_max_candidates = min(max_fallback_per_step, remaining_slots)
                    
                    if actual_max_candidates > 0:
                        graph_candidates = self._graph_neighborhood_expansion_fallback(must_have_entities, actual_max_candidates)
                        # Filter out duplicates
                        existing_ids = {note.get('note_id') for note in fallback_candidates}
                        new_graph_candidates = [note for note in graph_candidates if note.get('note_id') not in existing_ids]
                        graph_count = len(new_graph_candidates)
                        fallback_candidates.extend(new_graph_candidates)
                        graph_success = True
                        # 标记候选来源
                        for note in new_graph_candidates:
                            note['retrieval_source'] = 'graph_fallback'
                        logger.debug(f"Fallback step 3 (Graph): retrieved {graph_count} new candidates")
                    else:
                        logger.debug(f"Fallback step 3 (Graph): skipped due to candidate limit ({len(fallback_candidates)}/{max_total_fallback})")
                except Exception as e:
                    logger.error(f"Graph fallback failed: {e}")
        
        fallback_steps.append({
            'step': 3,
            'method': 'Graph',
            'success': graph_success,
            'candidates_retrieved': graph_count,
            'total_after_step': len(fallback_candidates),
            'skipped': len(fallback_candidates) >= min_per_subq or not must_have_entities
        })
        
        # 记录回补检索的结构化日志
        self._log_fallback_retrieval_info(
            sub_question=sub_question,
            expanded_query=expanded_query,
            must_have_entities=must_have_entities,
            min_threshold=min_per_subq,
            fallback_steps=fallback_steps,
            final_candidates=fallback_candidates,
            dataset=dataset,
            qid=qid
        )
        
        logger.info(f"Fallback retrieval completed: {len(fallback_candidates)} total candidates")
        return fallback_candidates
    
    def _log_fallback_retrieval_info(self, sub_question: str, expanded_query: str, must_have_entities: List[str], 
                                   min_threshold: int, fallback_steps: List[Dict], final_candidates: List[Dict], 
                                   dataset: Optional[str] = None, qid: Optional[str] = None):
        """Log detailed fallback retrieval information for debugging."""
        try:
            # 统计最终候选的来源分布
            source_distribution = {}
            for candidate in final_candidates:
                source = candidate.get('retrieval_source', 'unknown')
                source_distribution[source] = source_distribution.get(source, 0) + 1
            
            # 检查是否满足最小阈值
            threshold_met = len(final_candidates) >= min_threshold
            
            # 构建结构化日志
            fallback_log = {
                'event': 'fallback_retrieval_debug',
                'sub_question': sub_question,
                'expanded_query': expanded_query,
                'must_have_entities': must_have_entities,
                'min_threshold': min_threshold,
                'threshold_met': threshold_met,
                'total_candidates': len(final_candidates),
                'source_distribution': source_distribution,
                'fallback_steps': fallback_steps,
                'dataset': dataset,
                'query_id': qid
            }
            
            # 添加Top3候选的详细信息
            if final_candidates:
                top_candidates = []
                for i, candidate in enumerate(final_candidates[:3]):
                    candidate_info = {
                        'rank': i + 1,
                        'note_id': candidate.get('note_id', 'unknown'),
                        'filename': candidate.get('filename', 'unknown'),
                        'retrieval_source': candidate.get('retrieval_source', 'unknown'),
                        'final_score': candidate.get('final_score', 0.0),
                        'entities': candidate.get('entities', []),
                        'predicates': candidate.get('predicates', [])
                    }
                    top_candidates.append(candidate_info)
                fallback_log['top_candidates'] = top_candidates
            
            # 记录结构化日志
            logger.info(f"FALLBACK_DEBUG: {json.dumps(fallback_log, ensure_ascii=False, indent=2)}")
            
            # 如果未满足阈值，记录警告
            if not threshold_met:
                logger.warning(f"Fallback retrieval failed to meet minimum threshold: {len(final_candidates)}/{min_threshold} for subquestion: {sub_question}")
                
        except Exception as e:
            logger.error(f"Error logging fallback retrieval info: {e}")
    
    def _entity_anchored_graph_retrieval(self, seed_ids: List[str], must_have_entities: List[str]) -> List[Dict[str, Any]]:
        """Perform graph retrieval with entity anchoring to prevent irrelevant expansion."""
        if not seed_ids:
            return []
        
        # Standard graph retrieval
        graph_notes = self.graph_retriever.retrieve(seed_ids)
        
        # Apply entity anchoring filter if must_have_entities exist
        if must_have_entities and graph_notes:
            expand_k = config.get('graph.expand_k', 1)
            filtered_notes = []
            
            for note in graph_notes:
                # Check if note contains any anchor entity or its aliases
                if self._note_contains_anchor_entities(note, must_have_entities):
                    filtered_notes.append(note)
                    if len(filtered_notes) >= expand_k:
                        break
            
            logger.debug(f"Entity-anchored graph filtering: {len(graph_notes)} -> {len(filtered_notes)} notes")
            return filtered_notes
        
        return graph_notes
    
    def _graph_neighborhood_expansion_fallback(self, must_have_entities: List[str], max_candidates: int) -> List[Dict[str, Any]]:
        """Perform limited graph neighborhood expansion as fallback."""
        candidates = []
        
        # 从配置获取性能控制参数
        max_entity_lookup = config.get('retrieval', {}).get('performance', {}).get('max_entity_lookup', 2)
        
        # Use entity inverted index if available
        if hasattr(self, 'entity_index') and self.entity_index:
            for entity in must_have_entities[:max_entity_lookup]:  # 使用配置的实体数量限制
                try:
                    entity_notes = self.entity_index.get_notes_by_entity(entity)
                    candidates.extend(entity_notes[:max_candidates//max_entity_lookup])
                except Exception as e:
                    logger.error(f"Entity index lookup failed for {entity}: {e}")
        
        return candidates[:max_candidates]
    
    def _note_contains_anchor_entities(self, note: Dict[str, Any], anchor_entities: List[str]) -> bool:
        """Check if note contains any anchor entity or its aliases."""
        content = note.get('content', '').lower()
        title = note.get('title', '').lower()
        text = f"{title} {content}"
        
        for entity in anchor_entities:
            if entity.lower() in text:
                return True
            # Check aliases (basic implementation)
            aliases = self._get_entity_aliases(entity)
            for alias in aliases:
                if alias.lower() in text:
                    return True
        
        return False
    
    def _get_entity_aliases(self, entity: str) -> List[str]:
        """Get aliases for an entity (basic implementation)."""
        # Basic alias mapping - can be extended
        alias_map = {
            'Portuguese': ['Portugal', 'Lusitanian'],
            'Myanmar': ['Burma', 'Burmese'],
            'Laos': ['Lao', 'Laotian'],
            'Thailand': ['Thai', 'Siam', 'Siamese']
        }
        return alias_map.get(entity, [])
    
    def _count_entity_hits_in_candidates(self, candidates: List[Dict[str, Any]], must_have_entities: List[str]) -> Dict[str, int]:
        """Count how many candidates contain each must-have entity."""
        entity_hits = {entity: 0 for entity in must_have_entities}
        
        for candidate in candidates:
            content = candidate.get('content', '').lower()
            title = candidate.get('title', '').lower()
            text = f"{title} {content}"
            
            for entity in must_have_entities:
                if entity.lower() in text:
                    entity_hits[entity] += 1
        
        return entity_hits
    
    def _generate_enhanced_guardrail_params(self, query: str, must_have_entities: List[str]) -> tuple:
        """Generate enhanced guardrail parameters incorporating must_have_entities."""
        # Get base parameters
        must_have_terms, boost_entities, boost_predicates = self._generate_guardrail_params(query)
        
        # Enhance with must_have_entities
        enhanced_boost_entities = list(set(boost_entities + must_have_entities))
        enhanced_must_have_terms = list(set(must_have_terms + must_have_entities))
        
        return enhanced_must_have_terms, enhanced_boost_entities, boost_predicates
    
    def _bm25_fallback_search(self, query: str, dataset: str, qid: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform BM25 fallback search for precise short query matching."""
        try:
            # Use hybrid retriever's BM25 component if available
            if hasattr(self, 'hybrid_retriever') and self.hybrid_retriever:
                # Try to get BM25 results directly
                bm25_results = self.hybrid_retriever.bm25_search(query, top_k=top_k)
                # 为BM25回退结果添加subq_source标记
                for note in bm25_results:
                    if 'tags' not in note:
                        note['tags'] = {}
                    note['tags']['subq_source'] = 'fallback'
                return bm25_results
            
            # Fallback to vector retriever with keyword emphasis
            vector_results = self.vector_retriever.search([query], top_k=top_k)
            fallback_results = vector_results[0] if vector_results else []
            # 为向量回退结果添加subq_source标记
            for note in fallback_results:
                if 'tags' not in note:
                    note['tags'] = {}
                note['tags']['subq_source'] = 'fallback'
            return fallback_results
            
        except Exception as e:
            logger.error(f"BM25 fallback search failed: {e}")
            return []
    
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
                
                # 将CandidateItem转换回字典格式，确保paragraph_idxs被正确传递
                selected_notes = []
                for candidate_item in diversity_result.selected_candidates:
                    note_dict = {
                        'note_id': candidate_item.id,
                        'content': candidate_item.content,
                        'score': candidate_item.score,
                        'paragraph_idxs': candidate_item.metadata.get('paragraph_idxs', []),
                        # 保留其他重要字段
                        'entities': candidate_item.metadata.get('entities', []),
                        'topics': candidate_item.metadata.get('topics', []),
                        'retrieval_info': candidate_item.metadata.get('retrieval_info', {}),
                        'reasoning_paths': candidate_item.metadata.get('reasoning_paths', [])
                    }
                    # 复制其他可能的metadata字段
                    for key, value in candidate_item.metadata.items():
                        if key not in note_dict:
                            note_dict[key] = value
                    selected_notes.append(note_dict)
                
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
    
    def _calculate_vector_similarities_batch(self, queries: List[str], candidates: List[Dict[str, Any]]) -> List[float]:
        """
        批量计算多个查询与候选的向量相似度，用于优化focused相似度计算
        
        Args:
            queries: 查询字符串列表
            candidates: 候选列表
            
        Returns:
            相似度分数列表
        """
        try:
            if not queries or not candidates or len(queries) != len(candidates):
                return [0.0] * len(candidates)
            
            # 批量编码查询
            query_embeddings = self.vector_retriever.embedding_manager.encode_queries(queries)
            
            similarities = []
            for i, (query_embedding, candidate) in enumerate(zip(query_embeddings, candidates)):
                note_id = candidate.get('note_id')
                if note_id and hasattr(self.vector_retriever, 'id_to_index') and note_id in self.vector_retriever.id_to_index:
                    # 使用预计算的嵌入
                    note_idx = self.vector_retriever.id_to_index[note_id]
                    if note_idx < len(self.vector_retriever.note_embeddings):
                        note_embedding = self.vector_retriever.note_embeddings[note_idx]
                        # 计算余弦相似度
                        similarity = np.dot(query_embedding, note_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(note_embedding)
                        )
                        similarities.append(max(0.0, similarity))
                    else:
                        similarities.append(0.0)
                else:
                    # 动态计算嵌入（较慢）
                    note_text = candidate.get('content', '')
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
            logger.error(f"Error calculating batch vector similarities: {e}")
            return [0.0] * len(candidates)

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
    
    def _perform_hybrid_graph_retrieval(self, candidate_notes: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """执行混合图检索策略"""
        try:
            if self.graph_strategy == "entity_extraction":
                return self._perform_entity_extraction_graph_retrieval(candidate_notes)
            elif self.graph_strategy == "top_k_seed":
                return self._perform_top_k_seed_graph_retrieval(candidate_notes)
            elif self.graph_strategy == "hybrid":
                return self._perform_hybrid_mode_graph_retrieval(candidate_notes, query)
            else:
                logger.warning(f"Unknown graph strategy: {self.graph_strategy}, falling back to entity_extraction")
                return self._perform_entity_extraction_graph_retrieval(candidate_notes)
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []
    
    def _perform_entity_extraction_graph_retrieval(self, candidate_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于实体提取的图检索"""
        try:
            # 从候选笔记中提取实体
            entities = self._extract_entities_from_candidates(candidate_notes)
            if self.entity_extraction_max_entities > 0:
                entities = entities[:self.entity_extraction_max_entities]
            
            if not entities:
                logger.debug("No entities extracted for graph retrieval")
                return []
            
            # 使用实体倒排索引获取种子节点
            seed_ids = []
            if self.entity_inverted_index:
                candidate_note_ids = self.entity_inverted_index.get_candidate_notes(entities)
                seed_ids = list(candidate_note_ids)
            
            # 去重
            seed_ids = list(set(seed_ids))
            
            if not seed_ids:
                logger.debug("No seed nodes found from entities")
                return []
            
            # 执行图检索
            graph_notes = self.graph_retriever.retrieve(seed_ids)
            
            # 为图检索结果添加retrieval_method标识
            for note in graph_notes:
                if 'retrieval_info' not in note:
                    note['retrieval_info'] = {}
                note['retrieval_info']['retrieval_method'] = 'graph_search'
                note['retrieval_info']['graph_strategy'] = 'entity_extraction'
                
                # 为图检索候选添加hop_no和bridge_path字段
                if 'hop_no' not in note:
                    note['hop_type'] = 'graph_hop'
                    note['hop_no'] = 1
                    note['bridge_entity'] = None
                    note['bridge_path'] = []
            
            logger.debug(f"Entity extraction graph retrieval: {len(entities)} entities -> {len(seed_ids)} seeds -> {len(graph_notes)} results")
            return graph_notes
            
        except Exception as e:
            logger.error(f"Entity extraction graph retrieval failed: {e}")
            return []
    
    def _perform_top_k_seed_graph_retrieval(self, candidate_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于Top-K种子节点的图检索"""
        try:
            # 获取Top-K候选作为种子节点
            top_k_candidates = candidate_notes[:self.top_k_seed_num_seeds]
            seed_ids = [note.get('note_id') for note in top_k_candidates if note.get('note_id')]
            
            if not seed_ids:
                logger.debug("No valid seed IDs found from top-k candidates")
                if self.top_k_seed_fallback_to_entity:
                    logger.debug("Falling back to entity extraction")
                    return self._perform_entity_extraction_graph_retrieval(candidate_notes)
                return []
            
            # 执行图检索
            graph_notes = self.graph_retriever.retrieve(seed_ids)
            
            # 为图检索结果添加retrieval_method标识
            for note in graph_notes:
                if 'retrieval_info' not in note:
                    note['retrieval_info'] = {}
                note['retrieval_info']['retrieval_method'] = 'graph_search'
                note['retrieval_info']['graph_strategy'] = 'top_k_seed'
                
                # 为图检索候选添加hop_no和bridge_path字段
                if 'hop_no' not in note:
                    note['hop_type'] = 'graph_hop'
                    note['hop_no'] = 1
                    note['bridge_entity'] = None
                    note['bridge_path'] = []
            
            logger.debug(f"Top-K seed graph retrieval: {len(seed_ids)} seeds -> {len(graph_notes)} results")
            return graph_notes
            
        except Exception as e:
            logger.error(f"Top-K seed graph retrieval failed: {e}")
            if self.top_k_seed_fallback_to_entity:
                logger.debug("Falling back to entity extraction due to error")
                return self._perform_entity_extraction_graph_retrieval(candidate_notes)
            return []
    
    def _perform_hybrid_mode_graph_retrieval(self, candidate_notes: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """混合模式图检索"""
        try:
            # 执行主策略
            if self.hybrid_mode_primary_strategy == "top_k_seed":
                primary_results = self._perform_top_k_seed_graph_retrieval(candidate_notes)
            else:
                primary_results = self._perform_entity_extraction_graph_retrieval(candidate_notes)
            
            # 检查是否需要切换到备用策略
            if len(primary_results) >= self.hybrid_mode_switch_threshold:
                logger.debug(f"Primary strategy returned {len(primary_results)} results, using primary only")
                return primary_results
            
            # 执行备用策略
            logger.debug(f"Primary strategy returned {len(primary_results)} results (< {self.hybrid_mode_switch_threshold}), trying fallback")
            if self.hybrid_mode_fallback_strategy == "top_k_seed":
                fallback_results = self._perform_top_k_seed_graph_retrieval(candidate_notes)
            else:
                fallback_results = self._perform_entity_extraction_graph_retrieval(candidate_notes)
            
            # 合并结果
            merged_results = self._merge_fallback_results(primary_results, fallback_results)
            
            # 为混合模式图检索结果添加retrieval_method标识
            for note in merged_results:
                if 'retrieval_info' not in note:
                    note['retrieval_info'] = {}
                note['retrieval_info']['retrieval_method'] = 'graph_search'
                note['retrieval_info']['graph_strategy'] = 'hybrid_mode'
                
                # 为图检索候选添加hop_no和bridge_path字段
                if 'hop_no' not in note:
                    note['hop_type'] = 'graph_hop'
                    note['hop_no'] = 1
                    note['bridge_entity'] = None
                    note['bridge_path'] = []
            
            logger.debug(f"Hybrid mode: {len(primary_results)} + {len(fallback_results)} -> {len(merged_results)} results")
            return merged_results
            
        except Exception as e:
            logger.error(f"Hybrid mode graph retrieval failed: {e}")
            return []

    def _filter_with_multihop_safety(self, all_candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        多跳安全网（每跳 Top-M 保活 + must-have 扩展）
        
        把"桥接实体并入 must-have"和"Top-M 保活"推广到每一跳：
        
        Args:
            all_candidates: 各跳合并后的候选（每个 cand 有 hop_no / bridge_entity / bridge_path）
            query: 原始查询
            
        规则：
          1) 每跳 Top-M 无条件保活
          2) 其余用放宽阈值 + must-have（包含所有路径上实体）再筛
        """
        if not all_candidates:
            return []

        cfg = (self.config or {}).get("hybrid_search", {})
        mh = cfg.get("multi_hop", {})
        per_hop_keep_top_m = int(mh.get("per_hop_keep_top_m", 5))
        lower_th = float(mh.get("lower_threshold", 0.10))

        # 1) 生成 must-have：原始问句 terms + 所有路径上的实体（去重）
        must_have_terms, _, _ = self._generate_guardrail_params(query)
        path_terms = set()
        for c in all_candidates:
            # 收集桥接路径上的所有实体
            for e in (c.get("bridge_path") or []):
                path_terms.add(e.lower())
            # 收集桥接实体
            if c.get("bridge_entity"):
                path_terms.add(c["bridge_entity"].lower())
        
        # 合并原始must_have_terms和路径实体
        mh_terms = set(t.lower() for t in (must_have_terms or []))
        mh_terms |= path_terms
        must_have_terms = list(mh_terms)

        # 2) 分 hop 分桶
        from collections import defaultdict
        buckets = defaultdict(list)
        for c in all_candidates:
            hop_no = int(c.get("hop_no", 1))
            buckets[hop_no].append(c)

        kept = []
        for hop_no, bucket in buckets.items():
            # 按final_score降序排序
            bucket.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            
            # Top-M 保活
            safe = bucket[:per_hop_keep_top_m]
            tail = bucket[per_hop_keep_top_m:]

            # 对剩余候选应用放宽阈值 + must-have 过滤
            filtered_tail = []
            for c in tail:
                score_ok = c.get("final_score", 0.0) >= lower_th
                must_ok = self._contains_any_term(c, must_have_terms)
                if score_ok and must_ok:
                    filtered_tail.append(c)

            kept.extend(safe + filtered_tail)
            
            logger.info(f"Multi-hop safety filter - Hop {hop_no}: {len(bucket)} -> {len(safe + filtered_tail)} "
                       f"(safe: {len(safe)}, filtered: {len(filtered_tail)})")

        logger.info(f"Multi-hop safety filtering: {len(all_candidates)} -> {len(kept)} candidates total")
        return kept
