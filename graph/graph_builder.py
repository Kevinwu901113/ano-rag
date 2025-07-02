import networkx as nx
from typing import List, Dict, Any
from loguru import logger
from .relation_extractor import RelationExtractor

class GraphBuilder:
    """Builds a knowledge graph from atomic notes and relations."""
    def __init__(self):
        self.relation_extractor = RelationExtractor()

    def build_graph(self, atomic_notes: List[Dict[str, Any]], embeddings=None) -> nx.Graph:
        """Create a graph from notes and optional embeddings."""
        logger.info(f"Building graph with {len(atomic_notes)} notes")
        G = nx.Graph()
        for note in atomic_notes:
            node_id = note.get("note_id")
            if not node_id:
                continue
            G.add_node(node_id, **note)

        relations = self.relation_extractor.extract_all_relations(atomic_notes, embeddings)
        for rel in relations:
            src = rel.get("source_id")
            tgt = rel.get("target_id")
            weight = rel.get("weight", 1.0)
            rtype = rel.get("relation_type")
            if src and tgt:
                G.add_edge(src, tgt, weight=weight, relation_type=rtype, **rel.get("metadata", {}))
        logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
