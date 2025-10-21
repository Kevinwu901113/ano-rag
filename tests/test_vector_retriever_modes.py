import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_store import VectorRetriever


class FakeEmbeddingManager:
    def __init__(self):
        self.model_name = "default-model"
        self.embedding_dim = 3
        self.atomic_notes_calls = []
        self.query_calls = []
        self.set_model_calls = []

    def encode_atomic_notes(self, notes, include_metadata=True):
        self.atomic_notes_calls.append(notes)
        return np.ones((len(notes), self.embedding_dim))

    def encode_queries(self, queries):
        self.query_calls.append(queries)
        return np.ones((len(queries), self.embedding_dim))

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.set_model_calls.append(model_name)


class FakeVectorIndex:
    def __init__(self):
        self.embedding_dim = 3
        self.total_vectors = 0
        self.create_called = False
        self.add_called = False
        self.search_called = False
        self.search_results = None

    def create_index(self, index_type: str = None):
        self.create_called = True
        return True

    def add_vectors(self, vectors, ids=None):
        self.add_called = True
        self.total_vectors = len(vectors)
        return True

    def search(self, query_embeddings, top_k=5):
        self.search_called = True
        if self.search_results is not None:
            return self.search_results
        return [[{"index": 0, "similarity": 0.5, "score": 0.5, "rank": 0} for _ in range(top_k)]
                for _ in range(len(query_embeddings))]

    def save_index(self):
        return True

    def load_index(self):
        return False


ATOMIC_NOTES = [
    {
        "note_id": "note-0",
        "content": "Test content",
        "paragraph_idxs": [],
        "doc_name": "doc",
        "chunk_id": "doc#0",
    }
]


def _dense_results():
    manager = FakeEmbeddingManager()
    index = FakeVectorIndex()
    retriever = VectorRetriever(embedding_manager=manager, vector_index=index, retrieval_mode="dense")
    retriever.set_embedding_model("cli-model")
    assert manager.model_name == "cli-model"
    retriever.build_index(ATOMIC_NOTES, force_rebuild=True, save_index=False)
    index.search_results = [[{"index": 0, "similarity": 0.9, "score": 0.9, "rank": 0}]]
    results = retriever.search(["query"], top_k=1)
    return manager, index, results


def _bm25_results():
    manager = FakeEmbeddingManager()
    index = FakeVectorIndex()
    retriever = VectorRetriever(embedding_manager=manager, vector_index=index, retrieval_mode="bm25")
    retriever.build_index(ATOMIC_NOTES, force_rebuild=True, save_index=False)
    results = retriever.search([ATOMIC_NOTES[0]["content"]], top_k=1)
    return index, results


def _hybrid_results():
    manager = FakeEmbeddingManager()
    index = FakeVectorIndex()
    retriever = VectorRetriever(embedding_manager=manager, vector_index=index, retrieval_mode="hybrid")
    retriever.build_index(ATOMIC_NOTES, force_rebuild=True, save_index=False)

    index.search_results = [[{"index": 0, "similarity": 0.2, "score": 0.2, "rank": 0}]]

    def fake_bm25_search(queries, top_k, similarity_threshold, include_metadata):
        return [[{
            "note_id": ATOMIC_NOTES[0]["note_id"],
            "content": ATOMIC_NOTES[0]["content"],
            "paragraph_idxs": [],
            "retrieval_info": {
                "similarity": 0.4,
                "score": 0.4,
                "rank": 0,
                "query": queries[0],
                "retrieval_method": "bm25_tfidf",
            },
        }]]

    retriever._bm25_search = fake_bm25_search  # type: ignore[attr-defined]
    retriever.tfidf_vectorizer = object()
    retriever.tfidf_matrix = np.zeros((1, 1))

    results = retriever.search([ATOMIC_NOTES[0]["content"]], top_k=1)
    return results


def test_dense_mode_uses_vector_path():
    manager, index, results = _dense_results()
    assert index.create_called and index.add_called and index.search_called
    assert results[0][0]["retrieval_info"]["retrieval_method"] == "vector_search"
    assert manager.set_model_calls == ["cli-model"]


def test_bm25_mode_skips_vector_index():
    index, results = _bm25_results()
    assert not index.create_called and not index.add_called
    assert results[0][0]["retrieval_info"]["retrieval_method"] == "bm25_tfidf"


def test_hybrid_mode_merges_sources():
    results = _hybrid_results()
    info = results[0][0]["retrieval_info"]
    assert info["retrieval_method"] == "hybrid"
    assert set(info["sources"].keys()) == {"vector", "bm25"}
    assert info["similarity"] >= 0.4
