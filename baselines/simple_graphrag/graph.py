import pickle
from typing import Dict, List, Set, Optional, Any

class Node:
    def __init__(self, node_id: str, name: str):
        self.id = node_id
        self.name = name
        self.chunk_ids: Set[str] = set()
        self.type = "entity"

    def add_chunk(self, chunk_id: str):
        self.chunk_ids.add(chunk_id)

    def __repr__(self):
        return f"Node(id={self.id}, name={self.name}, chunks={len(self.chunk_ids)})"

class Edge:
    def __init__(self, source_id: str, target_id: str, relation: str, chunk_id: str):
        self.source_id = source_id
        self.target_id = target_id
        self.relation = relation
        self.chunk_id = chunk_id

    def __repr__(self):
        return f"Edge({self.source_id} -> {self.target_id} [{self.relation}])"

class SimpleGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        # Adjacency list: source_id -> list of (target_id, edge)
        self.adjacency: Dict[str, List[Edge]] = {}

    def add_node(self, name: str, chunk_id: Optional[str] = None) -> Node:
        # Simple normalization: lowercase and strip
        node_id = name.strip().lower()
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, name.strip())
            self.adjacency[node_id] = []
        
        if chunk_id:
            self.nodes[node_id].add_chunk(chunk_id)
        
        return self.nodes[node_id]

    def add_edge(self, source_name: str, target_name: str, relation: str, chunk_id: str):
        source_node = self.add_node(source_name, chunk_id)
        target_node = self.add_node(target_name, chunk_id)
        
        edge = Edge(source_node.id, target_node.id, relation, chunk_id)
        self.adjacency[source_node.id].append(edge)

    def get_node(self, name: str) -> Optional[Node]:
        node_id = name.strip().lower()
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[Edge]:
        return self.adjacency.get(node_id, [])

    def build_chunk_map(self) -> Dict[str, Set[str]]:
        """
        Builds a reverse mapping from chunk_id to set of node_ids.
        """
        chunk_map: Dict[str, Set[str]] = {}
        for node_id, node in self.nodes.items():
            for cid in node.chunk_ids:
                if cid not in chunk_map:
                    chunk_map[cid] = set()
                chunk_map[cid].add(node_id)
        return chunk_map

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'SimpleGraph':
        with open(path, 'rb') as f:
            return pickle.load(f)
