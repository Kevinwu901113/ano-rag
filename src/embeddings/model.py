import json
from typing import Iterable, List, Optional

import numpy as np

try:
    import cudf
except ImportError:  # cudf is optional
    cudf = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import requests


class EmbeddingModel:
    """Utility class for embedding text using bge-m3 or nomic-embed models."""

    def __init__(self, model: str, provider: str = "huggingface", device: str = "cpu", use_cudf: bool = False):
        self.model_name = model
        self.provider = provider
        self.device = device
        self.use_cudf = use_cudf and cudf is not None
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.provider in {"huggingface", "local"}:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required for HuggingFace models")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        elif self.provider == "ollama":
            # Ollama requires a running instance accessible via HTTP
            self.model = None  # nothing to load
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def embed_batch(self, texts: Iterable[str], batch_size: int = 32) -> List[np.ndarray]:
        texts = list(texts)
        if self.provider == "ollama":
            return [self._embed_ollama(text) for text in texts]
        if self.model is None:
            raise RuntimeError("Model not loaded")
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        if self.use_cudf:
            return cudf.DataFrame(embeddings)
        return [emb for emb in embeddings]

    def _embed_ollama(self, text: str) -> np.ndarray:
        url = "http://localhost:11434/api/embeddings"
        payload = {"model": self.model_name, "prompt": text}
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return np.array(data.get("embedding", []), dtype=np.float32)

