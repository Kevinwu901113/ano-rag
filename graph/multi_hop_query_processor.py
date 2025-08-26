"""Multi-hop graph query processing utilities."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from loguru import logger

import numpy as np

from config import config
from .graph_builder import GraphBuilder
from .graph_index import GraphIndex
from .graph_retriever import GraphRetriever


class MultiHopQueryProcessor:
    """Build and query the knowledge graph with multi-hop reasoning."""

    def __init__(
        self,
        atomic_notes: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        graph_file: Optional[str] = None,
        graph_index: Optional[GraphIndex] = None,
    ) -> None:
        builder = GraphBuilder()

        if graph_index is not None:
            self.graph_index = graph_index
        elif graph_file:
            self.graph_index = GraphIndex()
            try:
                self.graph_index.load_index(graph_file)
                logger.info(f"Loaded graph from {graph_file}")
            except Exception as exc:  # pragma: no cover - corrupted file
                logger.error(f"Failed to load graph index: {exc}, rebuilding")
                graph = builder.build_graph(atomic_notes, embeddings)
                self.graph_index.build_index(graph, atomic_notes, embeddings)
        else:
            graph = builder.build_graph(atomic_notes, embeddings)
            self.graph_index = GraphIndex(graph)
            self.graph_index.build_index(graph, atomic_notes, embeddings)

        self.retriever = GraphRetriever(self.graph_index)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 10,
        query_keywords: Optional[List[str]] = None,
        query_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve notes and reasoning paths from the graph."""

        notes = self.retriever.retrieve_with_reasoning_paths(
            query_embedding,
            top_k=top_k,
            query_keywords=query_keywords,
            query_entities=query_entities,
        )

        explanation = self.retriever.get_reasoning_explanation(notes)
        return {"notes": notes, "explanation": explanation}

