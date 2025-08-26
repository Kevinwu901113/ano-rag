from typing import List, Dict, Any, Optional
from loguru import logger
import os
import numpy as np
import concurrent.futures
import threading

from llm import OllamaClient, LocalLLM
from vector_store import VectorRetriever, EnhancedRecallOptimizer
from graph.graph_builder import GraphBuilder
from graph.graph_index import GraphIndex
from graph.graph_retriever import GraphRetriever
from utils.context_scheduler import ContextScheduler, MultiHopContextScheduler
from utils.context_dispatcher import ContextDispatcher
from utils.dataset_guard import filter_notes_by_namespace, assert_namespace_or_raise, DatasetNamespaceError
from utils.bm25_search import build_bm25_corpus, bm25_scores
from config import config

# 导入多跳推理组件
from graph.multi_hop_query_processor import MultiHopQueryProcessor

# 导入子问题分解组件
from .subquestion_planner import SubQuestionPlanner
from .evidence_merger import EvidenceMerger

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
            self.context_dispatcher = ContextDispatcher(self.vector_retriever, self.graph_retriever)
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
        
        # 初始化混合检索配置
        self.hybrid_search_enabled = config.get('hybrid_search.enabled', False)
        self.bm25_weight = config.get('hybrid_search.bm25_weight', 0.6)
        self.vector_weight = config.get('hybrid_search.vector_weight', 1.0)
        
        # 预构建 BM25 语料库（如果启用混合检索）
        self.bm25_corpus = None
        if self.hybrid_search_enabled:
            try:
                self.bm25_corpus = build_bm25_corpus(atomic_notes, lambda note: note.get('content', ''))
                logger.info(f"Built BM25 corpus for hybrid search with {len(atomic_notes)} notes")
            except Exception as e:
                logger.error(f"Failed to build BM25 corpus: {e}")
                self.hybrid_search_enabled = False
        
        logger.info(f"Hybrid search: {'enabled' if self.hybrid_search_enabled else 'disabled'}")
        if self.hybrid_search_enabled:
            logger.info(f"Hybrid search weights - Vector: {self.vector_weight}, BM25: {self.bm25_weight}")

    def process(self, query: str, dataset: Optional[str] = None, qid: Optional[str] = None) -> Dict[str, Any]:
        # Check if sub-question decomposition is enabled
        if self.use_subquestion_decomposition:
            return self._process_with_subquestion_decomposition(query, dataset, qid)
        else:
            return self._process_traditional(query, dataset, qid)
    
    def _process_traditional(self, query: str, dataset: Optional[str] = None, qid: Optional[str] = None) -> Dict[str, Any]:
        """Traditional query processing without sub-question decomposition."""
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
            dispatch_result = self.context_dispatcher.dispatch(query, queries)
            
            context = dispatch_result['context']
            selected_notes = dispatch_result['selected_notes']
            
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
            # 使用原有的调度逻辑
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
            
            logger.info(f"After deduplication: {len(candidate_notes)} unique notes from {sum(len(sub) for sub in vector_results)} total results")
            reasoning_paths: List[Dict[str, Any]] = []

            query_emb = self.vector_retriever.embedding_manager.encode_queries([query])[0]
            if self.recall_optimization_enabled:
                candidate_notes = self.recall_optimizer.optimize_recall(candidate_notes, query, query_emb)

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
                f"Scheduling {len(candidate_notes)} notes yielded {len(selected_notes)} selected: "
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
                subquestion_results = self._parallel_retrieval(sub_questions)
            else:
                subquestion_results = self._sequential_retrieval(sub_questions)
            
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
                        logger.warning(f"Triggering BM25 fallback for namespace {dataset}/{qid} in subquestion processing")
                        try:
                            # 使用BM25进行回退搜索
                            fallback_notes = self._bm25_fallback_search(query, dataset, qid)
                            if fallback_notes:
                                selected_notes = fallback_notes
                                logger.info(f"BM25 fallback retrieved {len(selected_notes)} notes for subquestion processing")
                            else:
                                logger.error(f"BM25 fallback also failed for namespace {dataset}/{qid} in subquestion processing")
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
    
    def _parallel_retrieval(self, sub_questions: List[str]) -> List[Dict[str, Any]]:
        """Perform parallel retrieval for multiple sub-questions."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sub_questions), 4)) as executor:
            # Submit retrieval tasks
            future_to_question = {
                executor.submit(self._retrieve_for_subquestion, sq, i): (sq, i) 
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
    
    def _sequential_retrieval(self, sub_questions: List[str]) -> List[Dict[str, Any]]:
        """Perform sequential retrieval for multiple sub-questions."""
        results = []
        
        for i, sub_question in enumerate(sub_questions):
            try:
                result = self._retrieve_for_subquestion(sub_question, i)
                results.append(result)
            except Exception as e:
                logger.error(f"Error retrieving for sub-question '{sub_question}': {e}")
                results.append({
                    'sub_question': sub_question,
                    'vector_results': [],
                    'graph_results': []
                })
        
        return results
    
    def _retrieve_for_subquestion(self, sub_question: str, index: int) -> Dict[str, Any]:
        """Retrieve evidence for a single sub-question."""
        # Vector retrieval
        vector_results = self.vector_retriever.search([sub_question])
        vector_notes = vector_results[0] if vector_results else []
        
        # Graph retrieval
        seed_ids = [note.get('note_id') for note in vector_notes if note.get('note_id')]
        graph_notes = self.graph_retriever.retrieve(seed_ids) if seed_ids else []
        
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
            'selected_notes': merged_evidence[:config.get('context_dispatcher.final_semantic_count', 3) + 
                                           config.get('context_dispatcher.final_graph_count', 5)],
            'stage_info': {
                'semantic_count': len([n for n in merged_evidence if 'vector' in n.get('source_types', set())]),
                'graph_count': len([n for n in merged_evidence if 'graph' in n.get('source_types', set())]),
                'final_count': min(len(merged_evidence), 
                                 config.get('context_dispatcher.final_semantic_count', 3) + 
                                 config.get('context_dispatcher.final_graph_count', 5))
            }
        }
        
        return dispatch_result['selected_notes']
    
    def _schedule_merged_evidence_traditional(self, merged_evidence: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Schedule merged evidence using traditional scheduler."""
        # Apply recall optimization if enabled
        if self.recall_optimization_enabled:
            query_emb = self.vector_retriever.embedding_manager.encode_queries([query])[0]
            merged_evidence = self.recall_optimizer.optimize_recall(merged_evidence, query, query_emb)
        
        # Use appropriate scheduler
        if self.multi_hop_enabled:
            # Extract reasoning paths from merged evidence
            reasoning_paths = []
            for evidence in merged_evidence:
                reasoning_paths.extend(evidence.get('reasoning_paths', []))
            
            selected_notes = self.scheduler.schedule_for_multi_hop(merged_evidence, reasoning_paths)
        else:
            selected_notes = self.scheduler.schedule(merged_evidence)
        
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
    
    def _fallback_vector_search(self, query: str, notes_pool: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
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
