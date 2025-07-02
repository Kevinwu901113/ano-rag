from typing import List, Dict, Any
from loguru import logger

from llm import QueryRewriter, OllamaClient
from vector_store import VectorRetriever
from graph.graph_builder import GraphBuilder
from graph.graph_index import GraphIndex
from graph.graph_retriever import GraphRetriever
from context_scheduler import ContextScheduler
from config import config

class QueryProcessor:
    """High level query processing pipeline."""
    def __init__(self, atomic_notes: List[Dict[str, Any]], embeddings=None):
        self.rewriter = QueryRewriter()
        self.vector_retriever = VectorRetriever()
        self.vector_retriever.build_index(atomic_notes)
        builder = GraphBuilder()
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
        context = "\n".join(n.get('content','') for n in selected_notes)
        answer = self.ollama.generate_final_answer(context, query)
        scores = self.ollama.evaluate_answer(query, context, answer)
        for n in selected_notes:
            n['feedback_score'] = scores.get('relevance',0)
        return {
            'query': query,
            'rewrite': rewrite,
            'answer': answer,
            'scores': scores,
            'notes': selected_notes
        }
