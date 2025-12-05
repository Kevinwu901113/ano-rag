"""
Vanilla RAG baseline (Index/Retriever/Runner).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger

from config import config as config_loader
from structrag.llm_client import LLMChatClient
from utils.embedding_utils import EmbeddingEncoder

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for multi-hop question answering.\n"
    "You are given several pieces of context that may come from different Wikipedia articles.\n"
    "You may need to combine information from multiple pieces to answer the question.\n"
    "Answer the question with a short phrase. If the answer is not contained in the context, say \"unknown\"."
)

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question with a short phrase. If the answer is not contained in the context, say "unknown"."""

class VanillaRAGRetriever:
    def __init__(
        self,
        index_path: str,
        chunk_store_path: str,
        embedding_client: Optional[EmbeddingEncoder] = None,
        llm_client: Optional[LLMChatClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        self.index_path = Path(index_path)
        self.chunk_store_path = Path(chunk_store_path)
        
        # Load Index
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        
        # Load Metadata (Vector ID -> Chunk ID)
        meta_path = self.index_path.with_name(self.index_path.name + ".meta.pkl")
        if not meta_path.exists():
             # Fallback: try legacy or assume implicit ordering? 
             # For now, raise error as we expect meta file.
             raise FileNotFoundError(f"Index metadata file not found: {meta_path}")
             
        with open(meta_path, "rb") as f:
            self.chunk_ids: List[str] = pickle.load(f)
            
        # Load Chunk Store
        if not self.chunk_store_path.exists():
            raise FileNotFoundError(f"Chunk store file not found: {self.chunk_store_path}")
        with open(self.chunk_store_path, "rb") as f:
            self.chunk_store: Dict[str, str] = pickle.load(f)

        # Initialize Clients
        self.embedding_client = embedding_client or self._init_embedding_client()
        self.llm_client = llm_client or self._init_llm_client()

    def _init_embedding_client(self) -> EmbeddingEncoder:
        retriever_cfg = self.cfg.get("retriever", {}) or {}
        embed_cfg = retriever_cfg.get("embedding", {}) or {}
        
        provider = embed_cfg.get("provider", "qwen3")
        
        # Resolve model name
        override = embed_cfg.get("model_path_override")
        base = embed_cfg.get("model", "Qwen/Qwen3-Embedding-8B")
        model = str(override or base).strip()
        
        cache_dir = self._clean_path(embed_cfg.get("cache_dir"))
        device = self._resolve_device()
        dtype = embed_cfg.get("dtype")
        max_len = int(embed_cfg.get("max_len_note", 512))
        
        return EmbeddingEncoder(provider, model, max_len, cache_dir=cache_dir, device=device, dtype=dtype)

    def _init_llm_client(self) -> LLMChatClient:
        # Use global config for LLM
        lm_cfg = self.cfg.get("lmstudio", {})
        endpoint = lm_cfg.get("endpoint")
        model = lm_cfg.get("model")
        
        if not endpoint or not model:
            logger.warning("LLM endpoint/model not configured properly in config.yaml")
            
        return LLMChatClient(endpoint=endpoint, model=model, temperature=0.0)

    def _resolve_device(self) -> Optional[str]:
        device = (self.cfg.get("retriever", {}) or {}).get("embedding", {}).get("device")
        if device:
            return str(device)
        system_cfg = self.cfg.get("system") or {}
        return system_cfg.get("device")

    def _clean_path(self, value: Any) -> Optional[str]:
        if not value:
            return None
        return str(Path(str(value)).expanduser())

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve chunks for a query. Returns list of (chunk_text, score)."""
        emb = self.embedding_client.encode([query])
        if emb is None or len(emb) == 0:
            return []
            
        # Fix dimension mismatch
        if emb.shape[1] != self.index.d:
             if emb.shape[1] < self.index.d:
                 padding = np.zeros((emb.shape[0], self.index.d - emb.shape[1]), dtype=emb.dtype)
                 emb = np.hstack([emb, padding])
             else:
                 emb = emb[:, :self.index.d]

        scores, indices = self.index.search(emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            if chunk_id in self.chunk_store:
                results.append((self.chunk_store[chunk_id], float(score)))
                
        return results

    def answer(self, question: str, top_k: int = 5) -> str:
        """End-to-end retrieve and answer."""
        docs = self.retrieve(question, top_k=top_k)
        
        context_blocks = []
        for i, (text, score) in enumerate(docs):
            context_blocks.append(f"[{i+1}] {text}")
        context_str = "\n\n".join(context_blocks)
        
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)
        
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        return self.llm_client.chat(messages)


def answer(
    question: str, 
    index_path: str, 
    chunk_store_path: str,
    llm_client: Optional[LLMChatClient] = None
) -> str:
    retriever = VanillaRAGRetriever(index_path, chunk_store_path, llm_client=llm_client)
    return retriever.answer(question)
