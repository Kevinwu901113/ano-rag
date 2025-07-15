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
        visited = set()
        for seed in seed_note_ids:
            if seed not in G:
                continue
            try:
                nodes = nx.single_source_dijkstra_path_length(
                    G, seed, cutoff=self.k_hop, weight="weight"
                )
            except Exception as e:
                logger.error(f"Weighted traversal failed: {e}")
                nodes = nx.single_source_shortest_path_length(G, seed, cutoff=self.k_hop)
            for node_id, dist in nodes.items():
                if node_id == seed or node_id in visited:
                    continue
                visited.add(node_id)
                data = G.nodes[node_id].copy()
                data["graph_distance"] = dist
                centrality = self.index.get_centrality(node_id)
                data["centrality"] = centrality
                importance = data.get("importance_score", 1.0)
                data["graph_score"] = (centrality / (dist + 1e-5)) * importance
                results.append(data)
        results.sort(key=lambda x: x.get("graph_score", 0), reverse=True)
        logger.info(f"Graph retrieval returned {len(results)} notes")
        return results
