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
        self._auto_build = bool(self.cfg.get("auto_build", False))
        self._auto_build_attempted = False
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
        if (not index_path.exists() or not meta_path.exists()) and self._auto_build and not self._auto_build_attempted:
            self._auto_build_attempted = True
            try:
                from relrag.indexer.embedding_index import EmbeddingIndexBuilder
                logger.info("Embedding artifacts missing; auto_build enabled, attempting build.")
                EmbeddingIndexBuilder(self.cfg).build()
            except Exception as exc:
                logger.warning("Embedding auto_build failed: {}", exc)
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
            from relrag.utils.embedding_utils import get_shared_encoder
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
        endpoint = self.cfg.get("endpoint")
        api_key = self.cfg.get("api_key")
        timeout_s = self.cfg.get("timeout_s")
        self._encoder = get_shared_encoder(
            provider,
            model,
            max_length=max_len,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            endpoint=endpoint,
            api_key=api_key,
            request_timeout_s=timeout_s,
        )

    def _uses_l2_distance(self, index: Any, faiss_cfg: Dict[str, Any]) -> bool:
        kind = str(faiss_cfg.get("kind", "")).upper()
        if kind.startswith("HNSW"):
            return True
        metric_type = getattr(index, "metric_type", None)
        if metric_type is None or faiss is None:
            return False
        try:
            return int(metric_type) == int(faiss.METRIC_L2)
        except Exception:
            return False

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
        use_l2 = self._uses_l2_distance(index, faiss_cfg)
        ranked: List[Dict[str, Any]] = []
        for rank, (score, vec_id) in enumerate(zip(distances[0], indices[0]), start=1):
            if vec_id < 0:
                continue
            meta = self._meta_by_id.get(int(vec_id))
            if not meta:
                continue
            raw_score = float(score)
            sim = 1.0 / (1.0 + max(raw_score, 0.0)) if use_l2 else raw_score
            payload = {
                "note_id": meta.get("note_id"),
                "score": sim,
                "rank": rank,
                "source": "emb",
            }
            if use_l2:
                payload["raw_distance"] = raw_score
            ranked.append(
                payload
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
