import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

if "GPUtil" not in sys.modules:
    gputil_stub = types.ModuleType("GPUtil")
    gputil_stub.getGPUs = lambda: []
    sys.modules["GPUtil"] = gputil_stub

from config import config as global_config
from query import query_processor as qp_mod


class _DummyVectorIndex:
    def __init__(self):
        self.index_dir = ""

    def load_index(self, filename):  # pragma: no cover - simple stub
        return None

    def embed(self, text):
        return np.ones(1, dtype=float)


class _DummyVectorRetriever:
    def __init__(self):
        self.data_dir = ""
        self.vector_index = _DummyVectorIndex()
        self.note_embeddings = np.zeros((0, 1), dtype=float)
        self.atomic_notes = []

    def build_index(self, atomic_notes):
        self.atomic_notes = list(atomic_notes)
        self.note_embeddings = np.ones((len(atomic_notes), 1), dtype=float)

    def _build_id_mappings(self):  # pragma: no cover - not used in test
        return None


class _DummyGraphBuilder:
    def __init__(self, llm=None):
        self.llm = llm

    def build_graph(self, atomic_notes, embeddings):
        import networkx as nx

        graph = nx.Graph()
        for note in atomic_notes:
            note_id = note.get("note_id")
            if note_id:
                graph.add_node(note_id)
        return graph


class _DummyGraphRetriever:
    def __init__(self, graph_index, k_hop=2):
        self.graph_index = graph_index
        self.k_hop = k_hop

    def retrieve(self, seed_ids):  # pragma: no cover - not used in test
        return []


class _DummyRecallOptimizer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _DummyContextPacker:
    def __init__(self, *args, **kwargs):
        pass


class _DummyEntityIndex:
    def __init__(self):
        self.entity_to_notes = {}

    def build_index(self, atomic_notes):
        self.entity_to_notes = {}


def test_filter_with_multihop_safety_respects_config(monkeypatch):
    test_config = {
        "retrieval": {"multi_hop": {"enabled": False}},
        "context_dispatcher": {"enabled": False},
        "vector_store": {"recall_optimization": {"enabled": False}},
        "hybrid_search": {
            "enabled": True,
            "fusion_method": "linear",
            "linear": {},
            "path_aware": {"enabled": False},
            "fallback": {"query_rewrite_enabled": False},
            "safety": {
                "per_hop_keep_top_m": 1,
                "lower_threshold": 0.25,
                "keep_one_per_doc": True,
            },
            "cluster_suppression": {"enabled": False},
        },
        "diversity_scheduler": {"enabled": False},
        "learned_fusion": {"enabled": False},
        "answer_verification": {"enabled": False},
        "rerank": {"use_listt5": False},
        "llm": {"provider": "ollama"},
    }

    def fake_load_config():
        return test_config

    monkeypatch.setattr(global_config, "load_config", fake_load_config)
    monkeypatch.setattr(global_config, "_config", test_config, raising=False)
    monkeypatch.setattr(global_config, "_raw_config", test_config, raising=False)

    monkeypatch.setattr(qp_mod, "VectorRetriever", _DummyVectorRetriever)
    monkeypatch.setattr(qp_mod, "GraphBuilder", _DummyGraphBuilder)
    monkeypatch.setattr(qp_mod, "GraphRetriever", _DummyGraphRetriever)
    monkeypatch.setattr(qp_mod, "EnhancedRecallOptimizer", _DummyRecallOptimizer)
    monkeypatch.setattr(qp_mod, "OllamaClient", lambda: object())
    monkeypatch.setattr(qp_mod, "create_learned_fusion", lambda *args, **kwargs: object())
    monkeypatch.setattr(qp_mod, "create_qa_coverage_scorer", lambda *args, **kwargs: object())
    monkeypatch.setattr(qp_mod, "create_span_picker", lambda *args, **kwargs: object())
    monkeypatch.setattr(qp_mod, "create_answer_verifier", lambda *args, **kwargs: object())
    monkeypatch.setattr(qp_mod, "ContextPacker", _DummyContextPacker)
    monkeypatch.setattr(qp_mod, "create_path_aware_ranker", lambda *args, **kwargs: object())
    monkeypatch.setattr(qp_mod, "EntityInvertedIndex", _DummyEntityIndex)

    atomic_notes = [
        {"note_id": "n1", "title": "Title 1", "content": "content"},
        {"note_id": "n2", "title": "Title 2", "content": "content"},
    ]

    processor = qp_mod.QueryProcessor(atomic_notes, embeddings=np.ones((2, 1)))

    assert processor.config is test_config

    candidates = [
        {"note_id": "n1", "doc_id": "doc1", "final_score": 0.9, "hop_no": 1, "title": "Note 1", "content": ""},
        {"note_id": "n2", "doc_id": "doc2", "final_score": 0.6, "hop_no": 2, "title": "Note 2", "content": "plain"},
        {"note_id": "n5", "doc_id": "doc5", "final_score": 0.55, "hop_no": 2, "title": "Note 5", "content": "plain"},
        {
            "note_id": "n3",
            "doc_id": "doc3",
            "final_score": 0.2,
            "hop_no": 2,
            "title": "Bridge note",
            "content": "contains bridge",
        },
        {
            "note_id": "n4",
            "doc_id": "doc_missing",
            "final_score": 0.1,
            "hop_no": 2,
            "title": "Bridge doc",
            "content": "bridge",
        },
    ]

    filtered = processor._filter_with_multihop_safety(
        candidates,
        query="test",
        path_entities=["Bridge"],
        first_hop_doc_id="doc_missing",
    )

    filtered_ids = [cand["note_id"] for cand in filtered]

    assert filtered_ids == ["n1", "n2", "n4"]
    assert "n5" not in filtered_ids
    assert "n3" not in filtered_ids
