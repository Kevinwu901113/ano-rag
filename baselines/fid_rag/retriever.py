
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from config import config as config_loader
from utils.embedding_utils import EmbeddingEncoder

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    logger.warning("FAISS unavailable: {}", exc)


class NaiveIndex:
    """In-memory wrapper around naive chunks + FAISS index."""

    def __init__(
        self,
        index_path: str,
        chunks_path: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if faiss is None:
            raise RuntimeError("FAISS is required for naive retrieval")
        self.cfg = config or config_loader.load_config()
        retr_cfg = self.cfg.get("retriever", {}) or {}
        self.embed_cfg = retr_cfg.get("embedding", {}) or {}
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self._index = self._load_index()
        self._chunks = self._load_chunks()
        self._by_vec: Dict[int, Dict[str, Any]] = {
            int(c["vector_id"]): c for c in self._chunks if c.get("vector_id") is not None
        }
        if not self._by_vec:
            logger.warning("No vector_id found in chunks; assigning sequential ids.")
            for idx, chunk in enumerate(self._chunks):
                chunk["vector_id"] = idx
                self._by_vec[idx] = chunk
        self._encoder = self._init_encoder()

    def search(self, question: str, topk: int) -> List[Dict[str, Any]]:
        q = (question or "").strip()
        if not q:
            return []
        encoded = self._encoder.encode([q])
        if encoded.size == 0:
            return []
        if bool(self.embed_cfg.get("normalize", True)):
            faiss.normalize_L2(encoded)
        limit = min(topk, self._index.ntotal)
        if limit <= 0:
            return []
        scores, ids = self._index.search(encoded.astype("float32"), limit)
        ranked: List[Dict[str, Any]] = []
        for rank, (score, vec_id) in enumerate(zip(scores[0], ids[0]), start=1):
            if vec_id < 0:
                continue
            meta = self._by_vec.get(int(vec_id))
            if not meta:
                continue
            ranked.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "doc_id": meta.get("doc_id"),
                    "text": meta.get("text"),
                    "doc_title": meta.get("meta", {}).get("doc_title") or meta.get("doc_id"),
                    "score": float(score),
                    "rank": rank,
                    "vector_id": int(vec_id),
                }
            )
        return ranked

    def _load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        return faiss.read_index(str(self.index_path))

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")
        records: List[Dict[str, Any]] = []
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _init_encoder(self) -> EmbeddingEncoder:
        provider = self.embed_cfg.get("provider", "qwen3")
        model = self._resolve_model_name()
        cache_dir = self._clean_path(self.embed_cfg.get("cache_dir"))
        device = self._resolve_device()
        dtype = self.embed_cfg.get("dtype")
        max_len = int(self.embed_cfg.get("max_len_note", 384))
        return EmbeddingEncoder(provider, model, max_len, cache_dir=cache_dir, device=device, dtype=dtype)

    def _resolve_model_name(self) -> str:
        override = self.embed_cfg.get("model_path_override")
        base = self.embed_cfg.get("model", "Qwen/Qwen3-Embedding-8B")
        candidate = str(override or base).strip()
        if not candidate:
            raise ValueError("Embedding model name is not configured")
        if override:
            logger.info("Embedding model override detected: {}", candidate)
        return candidate

    def _resolve_device(self) -> Optional[str]:
        device = self.embed_cfg.get("device")
        if device:
            return str(device)
        system_cfg = self.cfg.get("system") or {}
        return system_cfg.get("device")

    def _clean_path(self, value: Any) -> Optional[str]:
        if not value:
            return None
        return str(Path(str(value)).expanduser())
