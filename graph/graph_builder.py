import networkx as nx
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
from .relation_extractor import RelationExtractor
from config import config

try:
    from .graph_quality import compute_metrics
except Exception:  # pragma: no cover - optional dependency
    compute_metrics = None  # type: ignore
try:
    from .enhanced_relation_extractor import EnhancedRelationExtractor
    ENHANCED_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ENHANCED_AVAILABLE = False

class GraphBuilder:
    """Builds a knowledge graph from atomic notes and relations."""
    def __init__(self, llm=None):
        use_enhanced = config.get('multi_hop.enabled', False) and ENHANCED_AVAILABLE
        if use_enhanced:
            self.relation_extractor = EnhancedRelationExtractor(llm=llm)
        else:
            self.relation_extractor = RelationExtractor()

    def build_graph(self, atomic_notes: List[Dict[str, Any]], embeddings=None) -> nx.Graph:
        """Create a graph from notes and optional embeddings."""
        logger.info(f"Building graph with {len(atomic_notes)} notes")
        G = nx.Graph()
        for note in tqdm(atomic_notes, desc="Adding nodes"):
            node_id = note.get("note_id")
            if not node_id:
                continue
            G.add_node(node_id, **note)

        relations = self.relation_extractor.extract_all_relations(atomic_notes, embeddings)
        for rel in tqdm(relations, desc="Adding relations"):
            src = rel.get("source_id")
            tgt = rel.get("target_id")
            weight = rel.get("weight", 1.0)
            rtype = rel.get("relation_type")
            if src and tgt:
                G.add_edge(src, tgt, weight=weight, relation_type=rtype, **rel.get("metadata", {}))
        logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def build_graph_with_metrics(self, atomic_notes: List[Dict[str, Any]], embeddings=None):
        """Build the graph and compute quality metrics."""
        G = self.build_graph(atomic_notes, embeddings)
        metrics = {}
        if compute_metrics:
            try:
                metrics = compute_metrics(G)
                logger.info(f"Graph metrics: {metrics}")
            except Exception as exc:  # pragma: no cover - shouldn't break build
                logger.error(f"Failed to compute graph metrics: {exc}")
        return G, metrics


if __name__ == "__main__":
    import argparse
    import json
    import numpy as np
    from utils.file_utils import FileUtils

    parser = argparse.ArgumentParser(description="Build graph and report metrics")
    parser.add_argument("notes", help="Path to atomic notes JSON file")
    parser.add_argument("--embeddings", help="Optional embeddings .npy file")
    args = parser.parse_args()

    notes = FileUtils.read_json(args.notes)
    embeddings = None
    if args.embeddings:
        try:
            embeddings = np.load(args.embeddings)
        except Exception as exc:  # pragma: no cover - optional file
            logger.error(f"Failed to load embeddings: {exc}")

    builder = GraphBuilder()
    graph, metrics = builder.build_graph_with_metrics(notes, embeddings)
    print(json.dumps(metrics, indent=2))
