"""Constraint-based retriever entry points."""

from typing import Any


def retrieve_answer(*args: Any, **kwargs: Any):
    # Lazy import keeps lightweight modules (e.g. parser tests) importable without heavy deps.
    from .pipeline import retrieve_answer as _retrieve_answer

    return _retrieve_answer(*args, **kwargs)


__all__ = ["retrieve_answer"]
