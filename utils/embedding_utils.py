from __future__ import annotations

from typing import Sequence

import numpy as np
from loguru import logger


class EmbeddingEncoder:
    """Shared embedding encoder for offline/online stages."""

    def __init__(self, provider: str, model_name: str, max_length: int = 256) -> None:
        self.provider = provider
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        if self.provider == "qwen3":
            return self._encode_transformers(texts)
        if self.provider == "st":
            return self._encode_sentence_transformers(texts)
        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _encode_transformers(self, texts: Sequence[str]) -> np.ndarray:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers/torch are required for provider 'qwen3'") from exc
        if self._model is None or self._tokenizer is None:
            logger.info("Loading transformer embedding model: {}", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        assert self._model is not None and self._tokenizer is not None
        device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
        self._model.to(device)
        vectors = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                pooled = outputs.last_hidden_state.mean(dim=1)
            vectors.extend(pooled.cpu().numpy())
        return np.vstack(vectors).astype("float32")

    def _encode_sentence_transformers(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is required for provider 'st'") from exc
        if self._model is None:
            logger.info("Loading SentenceTransformer model: {}", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        embeddings = self._model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
