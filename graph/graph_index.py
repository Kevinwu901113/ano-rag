import networkx as nx
from typing import Dict, Any
from loguru import logger
from networkx.readwrite import json_graph
from utils import FileUtils

class GraphIndex:
    """Index wrapper for the knowledge graph."""

    def __init__(self, graph: nx.Graph = None):
        self.graph = graph or nx.Graph()
        self.node_centrality: Dict[str, float] = {}

    def build_index(self, graph: nx.Graph):
        """Prepare internal structures for fast retrieval."""
        self.graph = graph
        if self.graph.number_of_nodes() == 0:
            logger.warning("Empty graph for indexing")
            return
        self.node_centrality = nx.degree_centrality(self.graph)
        logger.info("Graph index built")

    def get_centrality(self, node_id: str) -> float:
        return self.node_centrality.get(node_id, 0.0)

    def load_index(self, filepath: str):
        """Load graph data from a JSON file and build the index."""
        try:
            data = FileUtils.read_json(filepath)
            graph = json_graph.node_link_graph(data)
            self.build_index(graph)
            logger.info(f"Graph loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load graph index: {e}")
