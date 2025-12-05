from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from loguru import logger

from config import config as config_loader
from utils.embedding_utils import EmbeddingEncoder
from utils.answer_cleaner import _strip_reasoning

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    logger.warning("FAISS unavailable: {}", exc)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for multi-hop question answering.\n"
    "You are given several pieces of context that may come from different Wikipedia articles.\n"
    "You may need to combine information from multiple pieces to answer the question.\n"
    "Answer the question with a short phrase. If the answer is not contained in the context, say \"unknown\"."
)

# We relax the prompt slightly to allow chain of thought
# but we still want concise final answers if possible.
PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question with a short phrase. If the answer is not contained in the context, say "unknown"."""

def _paragraphs_from_record(record: Dict[str, Any]) -> List[str]:
    """Extract text paragraphs from a doc_pool record."""
    text = record.get("text") or record.get("doc_chunk") or ""
    if isinstance(text, list):
        return [str(t) for t in text if str(t).strip()]
    
    # Split by double newline as naive paragraph separator
    return [p.strip() for p in str(text).split("\n\n") if p.strip()]

def _load_doc_pool(path: str) -> Iterable[Dict[str, Any]]:
    """Stream records from doc_pool.json or .jsonl."""
    with open(path, "r", encoding="utf-8") as f:
        # Try to read as list of dicts
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # Load all at once (careful with memory)
            data = json.load(f)
            for item in data:
                yield item
        else:
            # Assume JSONL
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

class NaiveChunker:
    """Simple chunker that splits text by tokens (naive implementation)."""
    
    def __init__(
        self, 
        target_tokens: int = 320,
        max_tokens: int = 384,
        overlap_tokens: int = 64,
        append_title: bool = True
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.append_title = append_title
        
    def chunk(self, text: str, title: str = "") -> List[str]:
        # Naive whitespace splitting approximation
        words = text.split()
        chunks = []
        current_chunk: List[str] = []
        current_len = 0
        
        # Prepend title
        prefix = f"{title}\n" if self.append_title and title else ""
        prefix_len = len(prefix.split()) # Rough
        
        # If text is short, just return it
        if len(words) + prefix_len <= self.max_tokens:
            return [prefix + text]
            
        step = self.target_tokens - self.overlap_tokens
        if step <= 0:
            step = self.target_tokens // 2
            
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.target_tokens]
            chunk_text = " ".join(chunk_words)
            if prefix:
                chunk_text = prefix + chunk_text
            chunks.append(chunk_text)
            
        return chunks


class NaiveIndex:
    """In-memory FAISS index wrapper for retrieval."""

    def __init__(
        self, 
        index_path: str, 
        chunks_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.cfg = config or config_loader.load_config()
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")
            
        logger.info("Loading FAISS index from {}", index_path)
        self.index = faiss.read_index(str(index_path))
        
        logger.info("Loading chunks from {}", chunks_path)
        self.chunks: List[Dict[str, Any]] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))
        
        # Initialize encoder for query embedding
        self.encoder = self._init_encoder()
        
    def _init_encoder(self) -> EmbeddingEncoder:
        provider = self.embed_cfg.get("provider", "qwen3")
        model = self._resolve_model_name()
        cache_dir = self._clean_path(self.embed_cfg.get("cache_dir"))
        device = self._resolve_device()
        dtype = self.embed_cfg.get("dtype")
        max_len = int(self.embed_cfg.get("max_len_note", 384))
        return EmbeddingEncoder(provider, model, max_len, cache_dir=cache_dir, device=device, dtype=dtype)

    @property
    def embed_cfg(self) -> Dict[str, Any]:
        return (self.cfg.get("retriever", {}) or {}).get("embedding", {}) or {}
        
    def _resolve_model_name(self) -> str:
        override = self.embed_cfg.get("model_path_override")
        base = self.embed_cfg.get("model", "Qwen/Qwen3-Embedding-8B")
        return str(override or base).strip()

    def _resolve_device(self) -> Optional[str]:
        device = self.embed_cfg.get("device")
        if device:
            return str(device)
        return (self.cfg.get("system") or {}).get("device")

    def _clean_path(self, value: Any) -> Optional[str]:
        if not value: return None
        return str(Path(str(value)).expanduser())

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        emb = self.encoder.encode([query])
        if emb is None or len(emb) == 0:
            return []
            
        # Ensure dimension match
        if emb.shape[1] != self.index.d:
             logger.warning(f"Dimension mismatch: query {emb.shape[1]} vs index {self.index.d}")
             # Simple fix: padding or truncation
             if emb.shape[1] < self.index.d:
                 import numpy as np
                 padding = np.zeros((emb.shape[0], self.index.d - emb.shape[1]), dtype=emb.dtype)
                 emb = np.hstack([emb, padding])
             else:
                 emb = emb[:, :self.index.d]

        scores, indices = self.index.search(emb, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.get("text", ""),
                "score": float(score),
                "metadata": chunk
            })
        return results


class LLMClient:
    """Minimal LM Studio/OpenAI-compatible chat client."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        stop: Optional[List[str]] = None,
        retries: int = 2,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("Both endpoint and model are required for LLM calls")
        self.endpoint = endpoint.rstrip("/")
        if self.endpoint.endswith("/v1"):
            self.endpoint = self.endpoint[:-3]
        
        # Fix: Ensure endpoint has scheme
        if not self.endpoint.startswith("http://") and not self.endpoint.startswith("https://"):
            self.endpoint = "http://" + self.endpoint
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Default: do not set stop tokens. For some models a leading newline is common;
        # forcing "\n" as a stop can truncate the answer to empty content.
        self.stop = [] if stop is None else stop
        self.retries = max(0, retries)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.endpoint}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.stop:
            payload["stop"] = self.stop

        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return str(content)
            except Exception as e:
                if attempt == self.retries:
                    logger.error(f"LLM call failed after {self.retries} retries: {e}")
                    raise
                time.sleep(1)
        return ""


class NaiveRAGRunner:
    """End-to-end naive RAG over MIRAGE dataset.json."""

    def __init__(
        self,
        index_path: str,
        chunks_path: str,
        *,
        topk: int = 5,
        lm_endpoint: Optional[str] = None,
        lm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        self.topk = max(1, int(topk))
        lm_cfg = self.cfg.get("lmstudio", {}) or {}
        endpoint = lm_endpoint or lm_cfg.get("endpoint")
        model = lm_model or lm_cfg.get("model")
        
        # Ensure strings
        endpoint = str(endpoint) if endpoint else None
        model = str(model) if model else None
        
        temp = temperature if temperature is not None else lm_cfg.get("temperature", 0.0)
        max_new_tokens = max_tokens if max_tokens is not None else lm_cfg.get("max_tokens", 8192)
        self.retriever = NaiveIndex(index_path, chunks_path, config=self.cfg)
        
        if llm_client:
            self.lm = llm_client
        elif endpoint and model:
            self.lm = LLMClient(endpoint, model, temperature=float(temp or 0.0), max_tokens=int(max_new_tokens or 8192), stop=stop)
        else:
            raise ValueError("LLM Client configuration missing")

    def answer(self, question: str) -> str:
        # 1. Retrieve
        docs = self.retriever.retrieve(question, k=self.topk)
        
        # 2. Construct context
        context_blocks = []
        for i, doc in enumerate(docs):
            context_blocks.append(f"[{i+1}] {doc['text']}")
        context_str = "\n\n".join(context_blocks)
        
        # 3. Prompt
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)
        
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # 4. Generate
        ans = self.lm.chat(messages)
        return _strip_reasoning(ans)

def answer(
    question: str, 
    index_path: str, 
    chunk_store_path: str, 
    top_k: int = 5,
    llm_client: Optional[LLMClient] = None
) -> str:
    """Convenience function for external scripts."""
    # This instantiates a new runner every time, which is inefficient for loops.
    # But fine for simple scripts.
    runner = NaiveRAGRunner(
        index_path, 
        chunk_store_path, 
        topk=top_k,
        llm_client=llm_client
    )
    return runner.answer(question)
