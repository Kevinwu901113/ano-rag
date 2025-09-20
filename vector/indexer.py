"""HNSW vector index wrapper for fast recall."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np

__all__ = ["VectorIndexer"]


class VectorIndexer:
    """Thin wrapper around a FAISS HNSW index."""

    def __init__(
        self,
        dimension: int,
        *,
        metric: str = "ip",
        index_path: str | Path | None = None,
        ann_params: Optional[Dict[str, int]] = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        self.dimension = int(dimension)
        self.metric = (metric or "ip").lower()
        if self.metric not in {"ip", "cosine", "l2"}:
            self.metric = "ip"
        self.index_path = Path(index_path) if index_path else None
        self._metadata_path = (
            self.index_path.with_suffix(".meta.json") if self.index_path else None
        )
        self.ann_params = ann_params or {}
        self.index: Optional[faiss.Index] = None
        self.note_version: Optional[str] = None
        self._id_to_label: Dict[int, str] = {}
        self._label_to_id: Dict[str, int] = {}
        self._is_built = False

    # ------------------------------------------------------------------
    # Internal helpers
    def _faiss_metric(self) -> int:
        if self.metric in {"ip", "cosine"}:
            return faiss.METRIC_INNER_PRODUCT
        return faiss.METRIC_L2

    def _hnsw_handle(self) -> Optional[faiss.IndexHNSWFlat]:  # type: ignore[name-defined]
        index = self.index
        while index is not None:
            if hasattr(index, "hnsw"):
                return index.hnsw  # type: ignore[return-value]
            index = getattr(index, "index", None)
        return None

    def _set_hnsw_params(self) -> None:
        hnsw = self._hnsw_handle()
        if hnsw is None:
            return
        M = self.ann_params.get("M")
        if M is not None and hasattr(hnsw, "M"):
            hnsw.M = int(M)
        ef_construction = self.ann_params.get("ef_construction")
        if ef_construction is not None and hasattr(hnsw, "efConstruction"):
            hnsw.efConstruction = int(ef_construction)
        ef_search = self.ann_params.get("ef_search")
        if ef_search is not None and hasattr(hnsw, "efSearch"):
            hnsw.efSearch = int(ef_search)

    def _create_index(self) -> faiss.IndexIDMap2:
        M = int(self.ann_params.get("M", 32))
        base = faiss.IndexHNSWFlat(self.dimension, M, self._faiss_metric())
        index = faiss.IndexIDMap2(base)
        self.index = index
        self._set_hnsw_params()
        return index

    def _ensure_metadata_dir(self) -> None:
        if not self.index_path:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def build(
        self,
        vectors: np.ndarray,
        ids: Sequence[str],
        *,
        note_version: Optional[str] = None,
    ) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-dimensional")
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")
        if vectors.shape[1] != self.dimension:
            raise ValueError("vector dimension mismatch")
        if len(vectors) == 0:
            raise ValueError("cannot build index with zero vectors")

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        index = self._create_index()
        internal_ids = np.arange(len(ids), dtype=np.int64)
        labels = [str(i) for i in ids]
        self._id_to_label = {
            int(internal): label for internal, label in zip(internal_ids, labels)
        }
        self._label_to_id = {label: internal for internal, label in self._id_to_label.items()}

        index.add_with_ids(vectors, internal_ids)
        self.note_version = note_version
        self._is_built = True

    def persist(self) -> bool:
        if not self.index_path or not self.index or not self._is_built:
            return False
        self._ensure_metadata_dir()
        faiss.write_index(self.index, str(self.index_path))
        if self._metadata_path:
            metadata = {
                "dimension": self.dimension,
                "metric": self.metric,
                "note_version": self.note_version,
                "id_to_label": {str(k): v for k, v in self._id_to_label.items()},
            }
            self._metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        return True

    def load(self, *, expected_version: Optional[str] = None) -> bool:
        if not self.index_path or not self.index_path.exists():
            return False
        if not self._metadata_path or not self._metadata_path.exists():
            return False
        index = faiss.read_index(str(self.index_path))
        metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("dimension", self.dimension)) != self.dimension:
            return False
        if expected_version and metadata.get("note_version") != expected_version:
            return False
        self.index = index
        self.metric = str(metadata.get("metric", self.metric)).lower()
        self._set_hnsw_params()
        id_map = metadata.get("id_to_label", {})
        self._id_to_label = {int(k): str(v) for k, v in id_map.items()}
        self._label_to_id = {v: k for k, v in self._id_to_label.items()}
        self.note_version = metadata.get("note_version")
        self._is_built = True
        return True

    def query(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if not self.index or not self._is_built:
            raise RuntimeError("index not built or loaded")
        if top_k <= 0:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != self.dimension:
            raise ValueError("query vector dimension mismatch")
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        hnsw = self._hnsw_handle()
        ef_search = self.ann_params.get("ef_search")
        if hnsw is not None and ef_search is not None and hasattr(hnsw, "efSearch"):
            hnsw.efSearch = max(int(ef_search), top_k)

        scores, ids = self.index.search(query_vector, top_k)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0:
                continue
            label = self._id_to_label.get(int(idx))
            if label is None:
                continue
            results.append((label, float(score)))
        return results

    def __len__(self) -> int:
        if not self.index or not self._is_built:
            return 0
        return int(self.index.ntotal)
