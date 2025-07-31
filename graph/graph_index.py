import os
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional
from loguru import logger
from networkx.readwrite import json_graph
from utils import FileUtils

class GraphIndex:
    """Index wrapper for the knowledge graph."""

    def __init__(self, graph: nx.Graph = None):
        self.graph = graph or nx.Graph()
        self.embeddings: Optional[np.ndarray] = None
        self.note_id_to_index: Dict[str, int] = {}
        self.index_to_note_id: Dict[int, str] = {}
        self.node_centrality: Dict[str, float] = {}

    def build_index(
        self,
        graph: nx.Graph,
        atomic_notes: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """Prepare internal structures for fast retrieval."""
        self.graph = graph

        if atomic_notes is not None:
            self.note_id_to_index = {}
            self.index_to_note_id = {}
            for idx, note in enumerate(atomic_notes):
                note_id = note.get("note_id")
                if note_id is not None:
                    self.note_id_to_index[note_id] = idx
                    self.index_to_note_id[idx] = note_id

        if embeddings is not None:
            self.embeddings = embeddings

        if self.graph.number_of_nodes() == 0:
            logger.warning("Empty graph for indexing")
            return
        # Use edge weights when calculating node importance. Fall back to
        # PageRank which supports weighted edges.
        try:
            self.node_centrality = nx.pagerank(self.graph, weight="weight")
        except Exception as e:
            logger.error(f"Failed weighted centrality computation: {e}")
            self.node_centrality = nx.pagerank(self.graph)
        logger.info("Graph index built")

    @property
    def centrality_scores(self) -> Dict[str, float]:
        return self.node_centrality

    def get_centrality(self, node_id: str) -> float:
        return self.node_centrality.get(node_id, 0.0)

    def get_embedding(self, note_id: str) -> Optional[np.ndarray]:
        """Return embedding vector for a note if available."""
        if self.embeddings is None:
            return None
        idx = self.note_id_to_index.get(note_id)
        if idx is None or idx >= len(self.embeddings):
            return None
        return self.embeddings[idx]

    def load_index(self, filepath: str):
        """Load graph data from a JSON file and build the index."""
        try:
            data = FileUtils.read_json(filepath)
            graph = json_graph.node_link_graph(data, edges="links")
            self.build_index(graph)

            base = os.path.splitext(filepath)[0]
            embed_file = base + "_embeddings.npz"
            mapping_file = base + "_mappings.json"

            if os.path.exists(embed_file):
                loaded = np.load(embed_file)
                self.embeddings = loaded["embeddings"]

            if os.path.exists(mapping_file):
                mapping = FileUtils.read_json(mapping_file)
                self.note_id_to_index = mapping.get("note_id_to_index", {})
                self.index_to_note_id = mapping.get("index_to_note_id", {})

            logger.info(f"Graph loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load graph index: {e}")

    def save_index(self, filepath: str) -> None:
        """Persist the graph and associated arrays."""
        try:
            data = json_graph.node_link_data(self.graph)
            FileUtils.write_json(data, filepath)

            base = os.path.splitext(filepath)[0]
            embed_file = base + "_embeddings.npz"
            mapping_file = base + "_mappings.json"

            if self.embeddings is not None:
                np.savez_compressed(embed_file, embeddings=self.embeddings)

            FileUtils.write_json({
                "note_id_to_index": self.note_id_to_index,
                "index_to_note_id": self.index_to_note_id,
            }, mapping_file)

            logger.info(f"Graph saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save graph index: {e}")

    def save_graphml(self, filepath: str) -> None:
        """Save the graph in GraphML format for visualization and analysis."""
        try:
            # 确保文件路径以.graphml结尾
            if not filepath.endswith('.graphml'):
                filepath = filepath + '.graphml'
            
            # 创建一个图的副本用于导出，处理节点和边的属性
            export_graph = self.graph.copy()
            
            # 为节点添加中心性分数
            for node_id in export_graph.nodes():
                centrality = self.get_centrality(node_id)
                export_graph.nodes[node_id]['centrality'] = centrality
            
            # 确保所有属性都是GraphML兼容的类型（字符串、数字、布尔值）
            for node_id, node_data in export_graph.nodes(data=True):
                for key, value in list(node_data.items()):
                    if isinstance(value, (list, dict, tuple)):
                        # 将复杂类型转换为字符串
                        export_graph.nodes[node_id][key] = str(value)
                    elif value is None:
                        # 移除None值
                        del export_graph.nodes[node_id][key]
            
            for u, v, edge_data in export_graph.edges(data=True):
                for key, value in list(edge_data.items()):
                    if isinstance(value, (list, dict, tuple)):
                        # 将复杂类型转换为字符串
                        export_graph.edges[u, v][key] = str(value)
                    elif value is None:
                        # 移除None值
                        del export_graph.edges[u, v][key]
            
            # 保存为GraphML格式
            nx.write_graphml(export_graph, filepath, encoding='utf-8')
            
            logger.info(f"Graph saved in GraphML format to {filepath}")
            logger.info(f"GraphML contains {export_graph.number_of_nodes()} nodes and {export_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Failed to save graph in GraphML format: {e}")
