from typing import List, Dict, Any, Optional
from loguru import logger
import os
import numpy as np

from llm import QueryRewriter, OllamaClient
from vector_store import VectorRetriever
from graph.graph_builder import GraphBuilder
from graph.graph_index import GraphIndex
from graph.graph_retriever import GraphRetriever
from utils.context_scheduler import ContextScheduler, MultiHopContextScheduler
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
    ):
        self.rewriter = QueryRewriter()
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
            self.scheduler = MultiHopContextScheduler()
        else:
            self.graph_retriever = GraphRetriever(self.graph_index, k_hop=config.get('graph.k_hop', 2))
            self.scheduler = ContextScheduler()

        self.ollama = OllamaClient()
        self.atomic_notes = atomic_notes

    def process(self, query: str) -> Dict[str, Any]:
        rewrite = self.rewriter.rewrite_query(query)
        queries = rewrite['rewritten_queries']
        vector_results = self.vector_retriever.search(queries)
        candidate_notes = [note for sub in vector_results for note in sub]
        reasoning_paths: List[Dict[str, Any]] = []

        if self.multi_hop_enabled:
            query_emb = self.vector_retriever.embedding_manager.encode_queries([query])[0]
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

        return {
            'query': query,
            'rewrite': rewrite,
            'answer': answer,
            'scores': scores,
            'notes': selected_notes,
            'predicted_support_idxs': predicted_support_idxs,
            'reasoning': reasoning_paths if self.multi_hop_enabled else None
        }
