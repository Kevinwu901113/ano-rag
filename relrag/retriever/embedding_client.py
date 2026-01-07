from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    logger.warning("numpy unavailable: {}. Embedding search disabled.", exc)

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    logger.warning("pandas unavailable: {}. Embedding search disabled.", exc)

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    logger.warning("FAISS unavailable: {}", exc)


class EmbeddingClient:
    """Online FAISS search over note embeddings."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled"))
        self._index = None
        self._meta_df: Optional[pd.DataFrame] = None
        self._meta_by_id: Dict[int, Dict[str, Any]] = {}
        self._encoder: Optional[EmbeddingEncoder] = None
        if self.enabled:
            self._load_resources()

    def _load_resources(self) -> None:
        if faiss is None or pd is None or np is None:
            logger.warning("Embedding search disabled: faiss/pandas/numpy unavailable.")
            self.enabled = False
            return
        index_path = Path(self.cfg.get("offline_index_path", "indexes/faiss/notes.faiss"))
        meta_path = Path(self.cfg.get("meta_path", "indexes/faiss/notes.meta.parquet"))
        if not index_path.exists() or not meta_path.exists():
            logger.warning("Embedding artifacts missing ({} / {}); disabling embedding channel.", index_path, meta_path)
            self.enabled = False
            return
        self._index = faiss.read_index(str(index_path))
        self._meta_df = pd.read_parquet(meta_path).sort_values("vector_id")
        self._meta_by_id = {int(row["vector_id"]): row.to_dict() for _, row in self._meta_df.iterrows()}
        self.load_encoder()

    def load_encoder(self) -> None:
        """Explicitly load the embedding encoder based on config."""
        try:
            from relrag.utils.embedding_utils import EmbeddingEncoder
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Embedding encoder unavailable: {}. Disabling embedding.", exc)
            self.enabled = False
            return
        provider = self.cfg.get("provider", "qwen3")
        model = self._resolve_model_name()
        max_len = int(self.cfg.get("max_len_note", 256))
        cache_dir = self._clean_path(self.cfg.get("cache_dir"))
        device = self._resolve_device()
        dtype = self.cfg.get("dtype")
        self._encoder = EmbeddingEncoder(provider, model, max_len, cache_dir=cache_dir, device=device, dtype=dtype)

    def search(self, question: str, topn: int) -> List[Dict[str, Any]]:
        if not self.enabled or not question.strip():
            return []
        if self._index is None or self._encoder is None:
            self._load_resources()
        if self._index is None or self._encoder is None:
            return []
        query_vec = self._encoder.encode([question.strip()])
        if query_vec.size == 0:
            return []
        if bool(self.cfg.get("normalize", True)):
            faiss.normalize_L2(query_vec)
        index = self._index
        assert index is not None
        faiss_cfg = self.cfg.get("faiss") or {}
        if hasattr(index, "nprobe") and "nprobe" in faiss_cfg:
            index.nprobe = int(faiss_cfg["nprobe"])
        if hasattr(index, "hnsw") and "efSearch" in faiss_cfg:
            index.hnsw.efSearch = int(faiss_cfg["efSearch"])
        limit = min(topn, index.ntotal)
        if limit <= 0:
            return []
        distances, indices = index.search(query_vec.astype("float32"), limit)
        ranked: List[Dict[str, Any]] = []
        for rank, (score, vec_id) in enumerate(zip(distances[0], indices[0]), start=1):
            if vec_id < 0:
                continue
            meta = self._meta_by_id.get(int(vec_id))
            if not meta:
                continue
            ranked.append(
                {
                    "note_id": meta.get("note_id"),
                    "score": float(score),
                    "rank": rank,
                    "source": "emb",
                }
            )
        return ranked

    def _resolve_model_name(self) -> str:
        override = self.cfg.get("model_path_override")
        base = self.cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        candidate = str(override or base).strip()
        if not candidate:
            raise ValueError("Embedding model name is not configured")
        if override:
            logger.info("Embedding model override detected: {}", candidate)
        return candidate

    def _clean_path(self, value: Any) -> Optional[str]:
        if not value:
            return None
        return str(Path(str(value)).expanduser())

    def _resolve_device(self) -> Optional[str]:
        device = self.cfg.get("device")
        if device:
            return str(device)
        return None
