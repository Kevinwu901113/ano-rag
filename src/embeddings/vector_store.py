import os
import pickle
from typing import List, Sequence

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class VectorStore:
    """Simple FAISS-based vector store with optional persistence."""

    def __init__(self, dim: int, persist_path: str = "store.faiss"):
        if faiss is None:
            raise ImportError("faiss is required for VectorStore")
        self.dim = dim
        self.persist_path = persist_path
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []

    def add(self, embeddings: Sequence[np.ndarray], texts: Sequence[str]):
        vecs = np.vstack(embeddings).astype("float32")
        self.index.add(vecs)
        self.texts.extend(texts)

    def search(self, query: np.ndarray, top_k: int = 5):
        query = query.reshape(1, -1).astype("float32")
        distances, idx = self.index.search(query, top_k)
        results = []
        for i, dist in zip(idx[0], distances[0]):
            if i == -1:
                continue
            results.append((self.texts[i], float(dist)))
        return results

    def save(self):
        faiss.write_index(self.index, self.persist_path)
        meta_path = os.path.splitext(self.persist_path)[0] + ".pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self):
        if os.path.exists(self.persist_path):
            self.index = faiss.read_index(self.persist_path)
            meta_path = os.path.splitext(self.persist_path)[0] + ".pkl"
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.texts = pickle.load(f)
