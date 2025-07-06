from typing import List, Dict, Any, Optional
from loguru import logger
import os

from llm import QueryRewriter, OllamaClient
from vector_store import VectorRetriever
from graph.graph_builder import GraphBuilder
from graph.graph_index import GraphIndex
from graph.graph_retriever import GraphRetriever
from utils.context_scheduler import ContextScheduler
from config import config

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
                logger.info(f"Loaded vector index from {vector_index_file}")
            except Exception as e:
                logger.error(f"Failed to load vector index: {e}, rebuilding")
                self.vector_retriever.build_index(atomic_notes)
        else:
            self.vector_retriever.build_index(atomic_notes)

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
                self.graph_index.build_index(graph)
        else:
            graph = builder.build_graph(atomic_notes, embeddings)
            self.graph_index = GraphIndex()
            self.graph_index.build_index(graph)
        self.graph_retriever = GraphRetriever(self.graph_index, k_hop=config.get('graph.k_hop',2))
        self.scheduler = ContextScheduler()
        self.ollama = OllamaClient()
        self.atomic_notes = atomic_notes

    def process(self, query: str) -> Dict[str, Any]:
        rewrite = self.rewriter.rewrite_query(query)
        queries = rewrite['rewritten_queries']
        vector_results = self.vector_retriever.search(queries)
        candidate_notes = [note for sub in vector_results for note in sub]
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
        logger.info(f"Evaluation scores returned: {scores}")
        for n in selected_notes:
            n['feedback_score'] = scores.get('relevance',0)
        logger.info(
            f"Applied feedback scores to {len(selected_notes)} notes: "
            f"{[n.get('note_id') for n in selected_notes]}"
        )
        return {
            'query': query,
            'rewrite': rewrite,
            'answer': answer,
            'scores': scores,
            'notes': selected_notes
        }
