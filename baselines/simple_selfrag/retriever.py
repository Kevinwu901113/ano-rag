import os
import pickle
import faiss
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
from loguru import logger

from config.config_loader import config as global_config

# Reuse common components
from rag_core.embedding_client import EmbeddingEncoder
from rag_core.llm_client import LLMChatClient
from baselines.common.model_clients import get_default_llm_client, get_default_embedding_client
from utils.context_budget import pack_contexts
from utils.output_protocol import build_final_instruction

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for multi-hop question answering.\n"
    "You are given several pieces of context that may come from different Wikipedia articles.\n"
    "You may need to combine information from multiple pieces to answer the question.\n"
    "Answer the question with a short phrase. If the answer is not contained in the context, say \"unknown\"."
)

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}
{final_instruction}

Answer the question with a short phrase. If the answer is not contained in the context, say "unknown"."""

class SimpleSelfRAGRetriever:
    def __init__(
        self, 
        index_path: str, 
        chunk_store_path: str, 
        config: Optional[Dict] = None,
        embedding_client: Optional[EmbeddingEncoder] = None,
        llm_client: Optional[LLMChatClient] = None,
        context_budget: Optional[int] = None,
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
            self.encoder = get_default_embedding_client(self.config)
            
        # Initialize LLM Client
        if llm_client:
            self.llm = llm_client
        else:
            self.llm = get_default_llm_client(self.config)
            
        self.top_k_first = self.selfrag_config.get("top_k_first", 5)
        self.top_k_second = self.selfrag_config.get("top_k_second", 10)
        self.last_hits: List[Dict[str, Any]] = []
        self.context_budget = int(context_budget or 0)

    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve chunks for a question.
        Returns list of dicts with text and scores.
        """
        # Encode question
        q_vec = self.encoder.encode([question], normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(q_vec, top_k)
        
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            text = self.chunk_store.get(chunk_id, "")
            if text:
                doc_id = None
                if "::" in str(chunk_id):
                    doc_id = str(chunk_id).split("::", 1)[0]
                results.append(
                    {
                        "text": text,
                        "score": float(score),
                        "doc_id": doc_id,
                        "sent_ids": None,
                        "passage_id": str(chunk_id),
                    }
                )
        self.last_hits = results
                
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
        annotated_1 = [{**c, "text": f"[{i+1}] {c['text']}"} for i, c in enumerate(contexts_1)]
        context_block_1, _, _ = pack_contexts(annotated_1, self.context_budget)
        
        prompt_1 = PROMPT_TEMPLATE.format(
            context=context_block_1,
            question=question,
            final_instruction=build_final_instruction(),
        )
        
        messages_1 = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_1}
        ]
        
        answer_0 = self.llm.chat(messages_1)
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
            self.last_hits = contexts_1
            return answer_0
            
        # Step 3: Second Retrieval (if needed)
        logger.info("Verdict insufficient, performing second retrieval...")
        
        # Retrieve more documents
        contexts_2 = self.retrieve(question, self.top_k_second)
        
        # Merge contexts (simple concatenation here, could be deduplicated)
        # We use the new contexts primarily
        annotated_2 = [{**c, "text": f"[{i+1}] {c['text']}"} for i, c in enumerate(contexts_2)]
        context_block_2, _, _ = pack_contexts(annotated_2, self.context_budget)
        
        prompt_2 = f"""The previous answer was judged insufficient.
Use the following context to give a more complete and well-supported answer.
{build_final_instruction()}

Question:
{question}

New context:
{context_block_2}

Improved answer (short phrase):"""

        messages_2 = [
             {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
             {"role": "user", "content": prompt_2}
        ]

        answer_1 = self.llm.chat(messages_2)
        logger.info(f"Improved answer: {answer_1[:100]}...")
        
        # Keep the contexts used in the final round for logging
        self.last_hits = contexts_2
        return answer_1
