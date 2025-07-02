from typing import List, Dict, Any
from loguru import logger
import networkx as nx
from .graph_index import GraphIndex

class GraphRetriever:
    """Retrieve notes from the graph using k-hop search."""
    def __init__(self, graph_index: GraphIndex, k_hop: int = 2):
        self.index = graph_index
        self.k_hop = k_hop

    def retrieve(self, seed_note_ids: List[str]) -> List[Dict[str, Any]]:
        G = self.index.graph
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty")
            return []
        results = []
        for seed in seed_note_ids:
            if seed not in G:
                continue
            nodes = nx.single_source_shortest_path_length(G, seed, cutoff=self.k_hop)
            for node_id, dist in nodes.items():
                if node_id == seed:
                    continue
                data = G.nodes[node_id].copy()
                data['graph_distance'] = dist
                data['centrality'] = self.index.get_centrality(node_id)
                results.append(data)
        logger.info(f"Graph retrieval returned {len(results)} notes")
        return results
