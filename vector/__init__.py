"""Lightweight vector utilities for retrieval."""

from .encoder import encode_notes, encode_query  # noqa: F401
from .indexer import VectorIndexer  # noqa: F401

__all__ = ["encode_notes", "encode_query", "VectorIndexer"]
