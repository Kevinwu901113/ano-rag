from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from adapters.mirage import _load_doc_pool, _paragraphs_from_record
from config import config as config_loader
from utils import TextUtils
from utils.device import run_with_fallback
from utils.embedding_utils import EmbeddingEncoder

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    logger.warning("FAISS unavailable: {}", exc)


@dataclass
class NaiveChunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: Dict[str, Any]
    vector_id: Optional[int] = None


class NaiveChunker:
    """Lightweight chunker for naive RAG: fixed token budget + simple overlap."""

    def __init__(
        self,
        target_tokens: int = 320,
        max_tokens: int = 384,
        overlap_tokens: int = 64,
        append_title: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if target_tokens <= 0 or target_tokens > max_tokens:
            raise ValueError("target_tokens must be in (0, max_tokens]")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = max(0, overlap_tokens)
        self.append_title = append_title

    def build_chunks(self, doc_pool_path: str) -> List[NaiveChunk]:
        docs = list(_iter_docs(doc_pool_path))
        chunks: List[NaiveChunk] = []
        for doc in docs:
            chunks.extend(self._chunk_document(doc))
        return chunks

    def _chunk_document(self, doc: Dict[str, Any]) -> List[NaiveChunk]:
        doc_id = doc["doc_id"]
        raw_id = doc["raw_id"]
        title = doc.get("title") or doc_id
        paragraphs = doc["paragraphs"]
        sentences: List[str] = []
        for para in paragraphs:
            para_text = str(para or "").strip()
            if not para_text:
                continue
            splits = TextUtils.split_by_sentence(para_text)
            sentences.extend(splits if splits else [para_text])
        if not sentences:
            return []

        sent_tokens = [max(1, TextUtils.rough_token_len(s)) for s in sentences]
        chunks: List[NaiveChunk] = []
        start = 0
        chunk_idx = 0

        while start < len(sentences):
            tokens = 0
            end = start
            while end < len(sentences):
                candidate = sent_tokens[end]
                if tokens and tokens + candidate > self.max_tokens:
                    break
                if tokens >= self.target_tokens and tokens + candidate > self.target_tokens:
                    break
                tokens += candidate
                end += 1
                if tokens >= self.target_tokens and (end == len(sentences) or tokens + sent_tokens[end] > self.max_tokens):
                    break
            if end == start:
                end = start + 1

            body = " ".join(sentences[start:end]).strip()
            if not body:
                start = end
                continue
            text = body if not self.append_title else f"{title}\n\n{body}"
            chunk = NaiveChunk(
                chunk_id=f"{raw_id}_{chunk_idx:04d}",
                doc_id=doc_id,
                text=text,
                meta={"source": "mirage", "doc_title": title},
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end >= len(sentences):
                break
            # Slide window backwards to keep simple token overlap
            back = end
            retained = 0
            while back > start and retained < self.overlap_tokens:
                back -= 1
                retained += sent_tokens[back]
            start = max(back, start + 1)
        return chunks


class MirageNaiveIndexer:
    """Offline builder for naive RAG index over MIRAGE doc_pool."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        chunker: Optional[NaiveChunker] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        retriever_cfg = self.cfg.get("retriever", {}) or {}
        self.embed_cfg = retriever_cfg.get("embedding", {}) or {}
        self.chunker = chunker or NaiveChunker()

    def build(
        self,
        doc_pool_path: str,
        out_dir: str,
        *,
        embed_model: Optional[str] = None,
        embed_device: str = "auto",
        embed_batch_size: Optional[int] = None,
        embed_max_length: Optional[int] = None,
        embed_normalize: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if faiss is None:
            raise RuntimeError("FAISS is required to build the naive index")
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        meta_path = out_root / "meta.json"

        chunk_records = self.chunker.build_chunks(doc_pool_path)
        if not chunk_records:
            raise RuntimeError("No chunks produced from doc_pool; aborting index build.")

        logger.info("Prepared {} chunks from doc_pool {}", len(chunk_records), doc_pool_path)
        model_name = embed_model or self._resolve_model_name()
        normalize = bool(self.embed_cfg.get("normalize", True) if embed_normalize is None else embed_normalize)
        max_len = int(embed_max_length or self.embed_cfg.get("max_len_note", self.chunker.max_tokens))
        batch_size = int(embed_batch_size or self.embed_cfg.get("batch_size", 4))

        signature_src = f"{model_name}|norm={int(normalize)}|max_len={max_len}"
        signature_hash = hashlib.sha1(signature_src.encode("utf-8")).hexdigest()[:10]
        model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(model_name).name)[:40].strip("_") or "model"
        signature = f"{model_tag}-{signature_hash}_n{int(normalize)}_l{max_len}"

        chunks_path = out_root / f"chunks_{signature}.jsonl"
        index_path = out_root / f"index_{signature}.faiss"

        self._write_chunks(chunks_path, chunk_records)

        texts = [c.text for c in chunk_records]

        provider = self.embed_cfg.get("provider", "qwen3")
        cache_dir = self._clean_path(self.embed_cfg.get("cache_dir"))
        dtype = self.embed_cfg.get("dtype")
        encoder = EmbeddingEncoder(
            provider,
            model_name,
            max_len,
            cache_dir=cache_dir,
            device=embed_device,
            dtype=dtype,
            fallback_to_cpu_on_oom=False,
            batch_size=batch_size,
            normalize=normalize,
        )

        vectors, used_device, fallback_reason = run_with_fallback(
            lambda device: encoder.encode(texts, device=device),
            prefer=embed_device,
        )
        if vectors.size == 0:
            raise RuntimeError("Embedding encoder produced empty vectors; cannot build index.")
        if normalize:
            faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype("float32"))
        faiss.write_index(index, str(index_path))

        # Attach vector_id to chunks for downstream retrieval
        for vec_id, chunk in enumerate(chunk_records):
            chunk.vector_id = vec_id
        self._write_chunks(chunks_path, chunk_records)

        stats = {
            "doc_pool": str(Path(doc_pool_path).resolve()),
            "chunk_count": len(chunk_records),
            "dim": dim,
            "index": str(index_path),
            "chunks": str(chunks_path),
            "artifact_id": signature,
            "normalize": normalize,
            "embed_model": model_name,
            "embed_device_used": used_device,
            "fallback_reason": fallback_reason,
            "embed_batch_size": batch_size,
            "embed_max_length": max_len,
            "target_tokens": self.chunker.target_tokens,
            "max_tokens": self.chunker.max_tokens,
            "overlap_tokens": self.chunker.overlap_tokens,
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
        logger.info("Naive index written to {} ({} vectors)", index_path, len(chunk_records))
        return stats

    def _init_encoder(self) -> EmbeddingEncoder:
        provider = self.embed_cfg.get("provider", "qwen3")
        model = self._resolve_model_name()
        cache_dir = self._clean_path(self.embed_cfg.get("cache_dir"))
        device = self._resolve_device()
        dtype = self.embed_cfg.get("dtype")
        max_len = int(self.embed_cfg.get("max_len_note", self.chunker.max_tokens))
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

    def _write_chunks(self, path: Path, chunks: Iterable[NaiveChunk]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "meta": chunk.meta,
                    "vector_id": chunk.vector_id,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _iter_docs(doc_pool_path: str) -> Iterable[Dict[str, Any]]:
    if not doc_pool_path:
        raise ValueError("doc_pool path must be provided")
    pool_path = Path(doc_pool_path)
    if not pool_path.exists():
        raise FileNotFoundError(f"doc_pool not found at {pool_path}")

    records = list(_load_doc_pool(str(pool_path)))
    for i, record in enumerate(records):
        raw_id = str(
            record.get("id")
            or record.get("doc_id")
            or record.get("_id")
            or record.get("mapped_id")
            or record.get("doc_name")
            or ""
        ).strip()
        
        # Ensure uniqueness by appending index if needed, similar to other baselines
        # Since doc_pool can have duplicate mapped_id/doc_name for different chunks
        raw_id = f"{raw_id}::{i}"
        
        if not raw_id:
            continue
        paragraphs = _paragraphs_from_record(record)
        if not paragraphs:
            continue
        yield {
            "raw_id": raw_id,
            "doc_id": f"mirage/{raw_id}",
            "title": record.get("title") or record.get("doc_name") or "",
            "paragraphs": paragraphs,
        }
