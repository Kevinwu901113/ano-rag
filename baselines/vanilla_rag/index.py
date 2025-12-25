from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger

from config import config as config_loader
from utils import TextUtils
from utils.embedding_utils import EmbeddingEncoder
from config.config_loader import DEFAULT_EMBED_MODEL


@dataclass
class VanillaChunk:
    chunk_id: str
    doc_id: str
    text: str
    vector_id: Optional[int] = None


class VanillaChunker:
    """Chunker for Vanilla RAG: fixed token budget + overlap."""

    def __init__(
        self,
        target_tokens: int = 512,
        max_tokens: int = 600,
        overlap_tokens: int = 50,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if target_tokens <= 0 or target_tokens > max_tokens:
            raise ValueError("target_tokens must be in (0, max_tokens]")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = max(0, overlap_tokens)

    def chunk_text(self, text: str, doc_id: str) -> List[VanillaChunk]:
        sentences = TextUtils.split_by_sentence(text)
        if not sentences:
            sentences = [text]

        sent_tokens = [max(1, TextUtils.rough_token_len(s)) for s in sentences]
        chunks: List[VanillaChunk] = []
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
            
            chunk_id = f"{doc_id}::chunk_{chunk_idx}"
            chunks.append(VanillaChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=body
            ))
            chunk_idx += 1

            if end >= len(sentences):
                break
            # Slide window backwards to keep overlap
            back = end
            retained = 0
            while back > start and retained < self.overlap_tokens:
                back -= 1
                retained += sent_tokens[back]
            start = max(back, start + 1)
        return chunks


class VanillaRAGIndexer:
    """Indexer for Vanilla RAG."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        retriever_cfg = self.cfg.get("retriever", {}) or {}
        self.embed_cfg = retriever_cfg.get("embedding", {}) or {}
        
        # Ensure embedding configuration exists or use defaults
        if not self.embed_cfg:
             # Try to find embedding config elsewhere if not under retriever.embedding
             # Some configs might put it under 'embedding' at root or similar, but sticking to standard
             pass

    def build(
        self,
        docs: Dict[str, str],
        output_index_path: str,
        output_chunks_path: str,
        *,
        output_embeddings_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        chunker = VanillaChunker(
            target_tokens=512,
            max_tokens=600,
            overlap_tokens=50
        )
        
        all_chunks: List[VanillaChunk] = []
        for doc_id, text in docs.items():
            chunks = chunker.chunk_text(text, doc_id)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            logger.warning("No chunks generated from documents.")
            return

        logger.info(f"Generated {len(all_chunks)} chunks from {len(docs)} documents.")

        # Embedding
        encoder = self._init_encoder()
        texts = [c.text for c in all_chunks]
        
        logger.info("Encoding chunks...")
        vectors = encoder.encode(texts)
        
        if vectors.size == 0:
             raise RuntimeError("Embedding encoder produced empty vectors.")

        if bool(self.embed_cfg.get("normalize", True)):
            faiss.normalize_L2(vectors)
        vectors_f32 = vectors.astype("float32")

        # Build Index
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors_f32)
        
        # Assign vector IDs
        for i, chunk in enumerate(all_chunks):
            chunk.vector_id = i
            
        # Save Index
        index_dir = Path(output_index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, output_index_path)
        logger.info(f"Saved vector index to {output_index_path}")

        if output_embeddings_path:
            emb_path = Path(output_embeddings_path)
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(emb_path, vectors_f32)
            logger.info(f"Saved embeddings to {output_embeddings_path}")

        # Save Chunk Store (Map chunk_id -> text)
        chunk_store = {c.chunk_id: c.text for c in all_chunks}
        chunk_store_dir = Path(output_chunks_path).parent
        chunk_store_dir.mkdir(parents=True, exist_ok=True)
        with open(output_chunks_path, "wb") as f:
            pickle.dump(chunk_store, f)
        logger.info(f"Saved chunk store to {output_chunks_path}")
        
        # Save Metadata (Chunk ID list to map vector ID back to chunk ID)
        # We need an ordered list of chunk_ids corresponding to vector_ids 0..N
        meta_path = output_index_path + ".meta.pkl"
        chunk_ids = [c.chunk_id for c in all_chunks]
        with open(meta_path, "wb") as f:
            pickle.dump(chunk_ids, f)
        logger.info(f"Saved index metadata to {meta_path}")

        return {
            "chunk_count": len(all_chunks),
            "vector_count": int(vectors_f32.shape[0]),
            "vector_dim": int(vectors_f32.shape[1]) if vectors_f32.size else 0,
            "embeddings_path": output_embeddings_path,
        }

    def _init_encoder(self) -> EmbeddingEncoder:
        provider = self.embed_cfg.get("provider", "qwen3")
        model = self._resolve_model_name()
        cache_dir = self._clean_path(self.embed_cfg.get("cache_dir"))
        device = self._resolve_device()
        dtype = self.embed_cfg.get("dtype")
        # Use a reasonable max length for embedding, matching chunk size
        max_len = int(self.embed_cfg.get("max_len_note", 512)) 
        return EmbeddingEncoder(provider, model, max_len, cache_dir=cache_dir, device=device, dtype=dtype)

    def _resolve_model_name(self) -> str:
        override = self.embed_cfg.get("model_path_override")
        base = self.embed_cfg.get("model", DEFAULT_EMBED_MODEL)
        candidate = str(override or base).strip()
        if not candidate:
            # Fallback if config is missing
            return DEFAULT_EMBED_MODEL
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
