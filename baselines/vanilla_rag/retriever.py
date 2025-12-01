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
             # Actually, let's just warn and assume we can't map back if missing, but that makes it useless.
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

    def _clean_path(self, value: Any) -> Optional[str]:
        if not value:
            return None
        return str(Path(str(value)).expanduser())

    def _resolve_device(self) -> Optional[str]:
        retriever_cfg = self.cfg.get("retriever", {}) or {}
        embed_cfg = retriever_cfg.get("embedding", {}) or {}
        device = embed_cfg.get("device")
        if device:
            return str(device)
        system_cfg = self.cfg.get("system") or {}
        return system_cfg.get("device")

    def retrieve(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k chunks for the question."""
        # Encode query
        q_vec = self.embedding_client.encode([question])
        if bool(self.cfg.get("retriever", {}).get("embedding", {}).get("normalize", True)):
            faiss.normalize_L2(q_vec)
            
        # Search
        scores, indices = self.index.search(q_vec.astype("float32"), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                text = self.chunk_store.get(chunk_id, "")
                if text:
                    results.append((text, float(score)))
        
        return results

    def answer(self, question: str, top_k: int = 5) -> str:
        chunks = self.retrieve(question, top_k=top_k)
        if not chunks:
            return "Insufficient evidence"
        
        context_texts = [text for text, _ in chunks]
        context_block = "\n\n".join(context_texts)
        
        prompt = f"""Answer the question based solely on the provided context.
If the answer is not in the context, say "Insufficient evidence".

Context:
{context_block}

Question: {question}

Answer:"""

        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_client.chat(messages, max_tokens=2048, temperature=0.0)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer"
