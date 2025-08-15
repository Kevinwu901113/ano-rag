from typing import List, Dict, Any, Optional
from loguru import logger
import os
import numpy as np

from llm import QueryRewriter, OllamaClient, LocalLLM
from vector_store import VectorRetriever, EnhancedRecallOptimizer
from graph.graph_builder import GraphBuilder
from graph.graph_index import GraphIndex
from graph.graph_retriever import GraphRetriever
from utils.context_scheduler import ContextScheduler, MultiHopContextScheduler
from utils.context_dispatcher import ContextDispatcher
from config import config

# 尝试导入增强的多跳推理组件
try:
    from graph.enhanced_relation_extractor import EnhancedRelationExtractor
    from graph.enhanced_graph_retriever import EnhancedGraphRetriever
    from graph.multi_hop_query_processor import MultiHopQueryProcessor
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False
    logger.warning("Enhanced multi-hop components not available, using standard components")

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
        self.rewriter = QueryRewriter(llm=llm)
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

        builder = GraphBuilder()
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

        self.multi_hop_enabled = config.get('multi_hop.enabled', False) and ENHANCED_COMPONENTS_AVAILABLE
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

    def process(self, query: str) -> Dict[str, Any]:
        rewrite = self.rewriter.rewrite_query(query)
        queries = rewrite['rewritten_queries']
        
        if self.use_context_dispatcher:
            # 使用新的结构增强上下文调度器
            dispatch_result = self.context_dispatcher.dispatch(query, queries)
            
            context = dispatch_result['context']
            selected_notes = dispatch_result['selected_notes']
            
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
