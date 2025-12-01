import os
import pickle
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from loguru import logger

from config.config_loader import config as global_config

# Reuse common components
from rag_core.embedding_client import EmbeddingEncoder
from rag_core.llm_client import LLMChatClient

class SimpleSelfRAGRetriever:
    def __init__(
        self, 
        index_path: str, 
        chunk_store_path: str, 
        config: Optional[Dict] = None,
        embedding_client: Optional[EmbeddingEncoder] = None,
        llm_client: Optional[LLMChatClient] = None
    ):
        self.config = config or global_config
        self.selfrag_config = self.config.get("retriever", {}).get("simple_selfrag", {})
        
        # Load Index
        logger.info(f"Loading index from {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Load Meta (chunk_ids)
        meta_path = str(index_path) + ".meta.pkl"
        logger.info(f"Loading meta from {meta_path}")
        with open(meta_path, "rb") as f:
            self.chunk_ids = pickle.load(f)
            
        # Load Chunk Store
        logger.info(f"Loading chunk store from {chunk_store_path}")
        with open(chunk_store_path, "rb") as f:
            self.chunk_store = pickle.load(f)
            
        # Initialize Embedding Encoder
        if embedding_client:
            self.encoder = embedding_client
        else:
            emb_cfg = self.selfrag_config.get("embedding") or self.config.get("retriever", {}).get("embedding")
            self.encoder = EmbeddingEncoder(
                provider=emb_cfg.get("provider", "huggingface"),
                model=emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                device=emb_cfg.get("device", "cpu")
            )
            
        # Initialize LLM Client
        if llm_client:
            self.llm = llm_client
        else:
            lm_cfg = self.config.get("lmstudio", {})
            self.llm = LLMChatClient(
                endpoint=lm_cfg.get("endpoint"),
                model=lm_cfg.get("model"),
                temperature=0.0
            )
            
        self.top_k_first = self.selfrag_config.get("top_k_first", 5)
        self.top_k_second = self.selfrag_config.get("top_k_second", 10)

    def retrieve(self, question: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Retrieve chunks for a question.
        Returns list of (chunk_text, score).
        """
        # Encode question
        q_vec = self.encoder.encode([question], normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(q_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            text = self.chunk_store.get(chunk_id, "")
            if text:
                results.append((text, float(score)))
                
        return results

    def answer(self, question: str) -> str:
        """
        Self-RAG answer generation process:
        1. Retrieve & Answer
        2. Critique
        3. (Optional) Retry & Answer
        """
        # Step 1: First Retrieval
        logger.info(f"First retrieval for: {question}")
        contexts_1 = self.retrieve(question, self.top_k_first)
        context_block_1 = "\n\n".join([f"[{i+1}] {c[0]}" for i, c in enumerate(contexts_1)])
        
        prompt_1 = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer is not contained in the context, say you are not sure.

Question:
{question}

Context:
{context_block_1}

Answer:"""
        
        answer_0 = self.llm.chat([{"role": "user", "content": prompt_1}])
        logger.info(f"Initial answer: {answer_0[:100]}...")
        
        # Step 2: Critique
        critique_prompt = f"""You are an impartial judge for a retrieval-augmented QA system.
Given the question, the retrieved context, and the model's answer, determine whether the answer is well-supported by the context.

If the answer is fully supported by the context, output exactly: sufficient.
If the answer is missing important evidence or contains unsupported claims, output exactly: insufficient.
Do not output anything else.

Question:
{question}

Context:
{context_block_1}

Answer:
{answer_0}

Verdict (sufficient / insufficient):"""

        verdict_raw = self.llm.chat([{"role": "user", "content": critique_prompt}])
        verdict = verdict_raw.strip().lower()
        logger.info(f"Critique verdict: {verdict}")
        
        if "sufficient" in verdict and "insufficient" not in verdict:
            return answer_0
            
        # Step 3: Second Retrieval (if needed)
        logger.info("Verdict insufficient, performing second retrieval...")
        
        # Retrieve more documents
        contexts_2 = self.retrieve(question, self.top_k_second)
        
        # Merge contexts (simple concatenation here, could be deduplicated)
        # We use the new contexts primarily
        context_block_2 = "\n\n".join([f"[{i+1}] {c[0]}" for i, c in enumerate(contexts_2)])
        
        prompt_2 = f"""The previous answer was judged insufficient.
Use the following context to give a more complete and well-supported answer.

Question:
{question}

New context:
{context_block_2}

Improved answer:"""

        answer_1 = self.llm.chat([{"role": "user", "content": prompt_2}])
        logger.info(f"Improved answer: {answer_1[:100]}...")
        
        return answer_1
