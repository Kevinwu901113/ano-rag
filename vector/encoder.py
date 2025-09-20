"""Embedding helpers wrapping configurable sentence transformers."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

__all__ = ["encode_notes", "encode_query"]

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_CACHE_LOCK = threading.Lock()


def _get_model(model_name: str) -> SentenceTransformer:
    """Load a sentence-transformer model with caching."""
    if not model_name:
        raise ValueError("model_name must be provided for encoding")
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name, trust_remote_code=True)
            _MODEL_CACHE[model_name] = model
    return model


def _normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32, copy=False)


def _prepare_note_text(note: Dict[str, Any]) -> str:
    title = (note.get("title") or "").strip()
    body = (note.get("text") or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def encode_notes(
    notes: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    normalize: bool = True,
    batch_size: int = 32,
    show_progress: bool = False,
    instruction: str | None = None,
) -> np.ndarray:
    """Encode notes into dense vectors using the configured embedding model.

    Args:
        notes: Sequence of note dictionaries containing ``title`` and ``text``.
        model_name: SentenceTransformer model name to load.
        normalize: Whether to L2-normalize vectors (cosine similarity).
        batch_size: Batch size fed to :meth:`SentenceTransformer.encode`.
        show_progress: Whether to display encode progress bar.
        instruction: Optional prefix instruction applied to each note.

    Returns:
        ``np.ndarray`` with shape ``(len(notes), embedding_dim)``.
    """
    if not notes:
        return np.empty((0, 0), dtype=np.float32)

    model = _get_model(model_name)

    texts: List[str] = []
    if instruction:
        prefix = instruction.strip()
    else:
        prefix = ""
    for note in notes:
        text = _prepare_note_text(note)
        if prefix:
            texts.append(f"{prefix} {text}".strip())
        else:
            texts.append(text)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if normalize:
        embeddings = _normalize_matrix(embeddings)
    return embeddings


def encode_query(
    query: str,
    *,
    model_name: str,
    normalize: bool = True,
    instruction: str | None = None,
) -> np.ndarray:
    """Encode a query/question into a dense vector."""
    model = _get_model(model_name)
    prepared = (query or "").strip()
    if instruction:
        prepared = f"{instruction.strip()} {prepared}".strip()
    vector = model.encode(
        [prepared],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    vector = np.asarray(vector[0], dtype=np.float32)
    if normalize:
        vector = _normalize_matrix(vector)[0]
    return vector
