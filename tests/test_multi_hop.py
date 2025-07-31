import os
import sys
import importlib.util

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODULE_PATH = os.path.join(ROOT_DIR, 'vector_store', 'enhanced_recall_optimizer.py')

sys.path.insert(0, ROOT_DIR)

spec = importlib.util.spec_from_file_location('enhanced_recall_optimizer', MODULE_PATH)
ero_module = importlib.util.module_from_spec(spec)
sys.modules['enhanced_recall_optimizer'] = ero_module
spec.loader.exec_module(ero_module)
EnhancedRecallOptimizer = ero_module.EnhancedRecallOptimizer


def set_mhp_class(cls):
    ero_module.MultiHopQueryProcessor = cls


class DummyEmbeddingManager:
    def __init__(self):
        self.encoded_queries = []
    def encode_queries(self, queries):
        self.encoded_queries.extend(queries)
        return [[0.1, 0.2] for _ in queries]

class DummyVectorRetriever:
    def __init__(self):
        self.embedding_manager = DummyEmbeddingManager()
        self.searched_queries = []
    def search(self, queries, top_k=3, **kwargs):
        self.searched_queries.extend(queries)
        return [[{"note_id": f"v_{i}", "similarity_score": 0.9}] for i in range(len(queries))]

class DummyGraphProcessor:
    def __init__(self):
        self.called_with = None
    def retrieve(self, embedding, **kwargs):
        self.called_with = embedding
        return {"notes": [{"note_id": "g1", "similarity_score": 0.95}]}


def test_multi_hop_uses_graph_processor():
    vector = DummyVectorRetriever()
    graph = DummyGraphProcessor()
    set_mhp_class(DummyGraphProcessor)
    opt = EnhancedRecallOptimizer(vector_retriever=vector, graph_retriever=graph)

    results = opt._execute_multi_hop_retrieval("test hop", set())

    assert graph.called_with is not None
    assert not vector.searched_queries
    assert results and results[0]["note_id"] == "g1"

class DummyReasoningRetriever:
    def __init__(self):
        self.called_with = None
    def retrieve_with_reasoning_paths(self, embedding, **kwargs):
        self.called_with = embedding
        return [{"note_id": "gp1", "similarity_score": 0.92}]

def test_multi_hop_reasoning_paths():
    vector = DummyVectorRetriever()
    graph = DummyReasoningRetriever()
    set_mhp_class(None)
    opt = EnhancedRecallOptimizer(vector_retriever=vector, graph_retriever=graph)

    results = opt._execute_multi_hop_retrieval("hop two", set())

    assert graph.called_with is not None
    assert not vector.searched_queries
    assert results and results[0]["note_id"] == "gp1"
