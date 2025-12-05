import re
import pickle
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict
from loguru import logger

from config.config_loader import config as global_config
from rag_core.embedding_client import EmbeddingEncoder
from rag_core.llm_client import LLMChatClient
from baselines.common.model_clients import get_default_embedding_client, get_default_llm_client
from baselines.simple_raptor.tree import TreeNode
from utils.answer_cleaner import clean_model_answer

class SimpleRaptorRetriever:
    def __init__(
        self, 
        index_path: str, 
        nodes_path: str,
        chunk_store_path: str,
        config: Optional[Dict] = None,
        embedding_client: Optional[EmbeddingEncoder] = None,
        llm_client: Optional[LLMChatClient] = None,
        top_k: Optional[int] = None
    ):
        self.config = config or global_config.load_config()
        self.raptor_config = self.config.get("retriever", {}).get("simple_raptor", {})
        
        # Load Index
        logger.info(f"Loading index from {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Load Nodes
        logger.info(f"Loading nodes from {nodes_path}")
        with open(nodes_path, "rb") as f:
            nodes_data = pickle.load(f)
            self.nodes = {n["node_id"]: TreeNode.from_dict(n) for n in nodes_data}
            
        # Load Node IDs Map
        node_ids_path = str(index_path) + ".node_ids.pkl"
        logger.info(f"Loading node_ids map from {node_ids_path}")
        with open(node_ids_path, "rb") as f:
            self.node_ids_map = pickle.load(f)
            
        # Load Chunk Store
        logger.info(f"Loading chunk store from {chunk_store_path}")
        with open(chunk_store_path, "rb") as f:
            self.chunk_store = pickle.load(f)
            
        # Initialize Clients
        if embedding_client:
            self.embedding = embedding_client
        else:
            self.embedding = get_default_embedding_client(self.config)
            
        if llm_client:
            self.llm = llm_client
        else:
            self.llm = get_default_llm_client(self.config)
            
        self.top_k_nodes = top_k or int(self.raptor_config.get("top_k_nodes", 10))
        self.max_answer_chunks = int(self.raptor_config.get("max_answer_chunks", 5))
        self.enable_name_rerank = bool(self.raptor_config.get("enable_name_rerank", True))

    def _extract_name_from_question(self, question: str) -> Optional[str]:
        """
        针对 MIRAGE 这种 "What is X's occupation?" 问法，
        简单抽出 X 当成实体名；不匹配就返回 None。
        """
        q = question.strip()
        lower = q.lower()
        # 只处理典型 "What is X's occupation" 格式
        if not lower.startswith("what is "):
            return None
        # 找到 "'s occupation"
        m = re.search(r"'s occupation", lower)
        if not m:
            return None
        # 截取 "What is " 和 "'s occupation" 之间的部分
        name = q[len("What is "): m.start()].strip()
        # 极端情况过滤一下空串
        return name or None

    def retrieve_nodes(self, question: str, top_k: Optional[int] = None) -> List[TreeNode]:
        """
        Retrieve relevant nodes from the tree.
        """
        search_k = top_k or self.top_k_nodes
        
        q_vec = self.embedding.encode([question])
        if len(q_vec) > 0:
             # Fix dimension mismatch for mock embeddings or different model
             if q_vec.shape[1] != self.index.d:
                 logger.warning(f"Dimension mismatch: query {q_vec.shape[1]} vs index {self.index.d}. Resizing query vector.")
                 if q_vec.shape[1] < self.index.d:
                     # Pad with zeros
                     padding = np.zeros((q_vec.shape[0], self.index.d - q_vec.shape[1]), dtype=q_vec.dtype)
                     q_vec = np.hstack([q_vec, padding])
                 else:
                     # Truncate
                     q_vec = q_vec[:, :self.index.d]
             
             faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx not in self.node_ids_map:
                continue
            node_id = self.node_ids_map[idx]
            if node_id in self.nodes:
                results.append(self.nodes[node_id])
        return results

    def answer(self, question: str) -> str:
        """
        Raptor answer generation process:
        1. Retrieve top nodes (mixed levels)
        2. Collapse to leaf chunks
        3. Generate Answer
        """
        # 1. Retrieve Nodes
        logger.info(f"Retrieving nodes for: {question}")
        retrieved_nodes = self.retrieve_nodes(question)
        
        # 2. Collect Leaf Chunks
        # We prioritize chunks from higher-ranked nodes
        candidate_chunk_ids = []
        seen_chunks = set()
        
        for node in retrieved_nodes:
            for cid in node.descendant_chunk_ids:
                if cid not in seen_chunks and cid in self.chunk_store:
                    candidate_chunk_ids.append(cid)
                    seen_chunks.add(cid)
        
        if not candidate_chunk_ids:
            logger.warning("No candidate chunks found for question: {}", question)
            return "Insufficient evidence"
        
        # 2) 如果开启 name_rerank，优先把“包含名字字符串”的 chunk 排到前面
        name = self._extract_name_from_question(question) if self.enable_name_rerank else None
        
        if name:
            name_lower = name.lower()
            
            def has_name(cid: int) -> bool:
                text = self.chunk_store.get(cid, "")
                return name_lower in text.lower()
                
            # True 排前面，False 排后面；保持原有顺序的稳定性
            candidate_chunk_ids.sort(key=lambda cid: (not has_name(cid)))
            logger.debug(
                "Name-aware rerank enabled for name='{}'. First candidate snippet: {}",
                name,
                self.chunk_store[candidate_chunk_ids[0]][:80].replace("\n", " "),
            )
            
        # 4. Truncate to max_answer_chunks
        chunk_ids = candidate_chunk_ids[:self.max_answer_chunks]
                
        # Retrieve chunk text
        context_texts = []
        for cid in chunk_ids:
            text = self.chunk_store.get(cid)
            if text:
                context_texts.append(text)
                
        context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_texts)])
        
        # Truncate to avoid 400 error (similar to index.py fix)
        # The model has 4096 limit, so we should be conservative.
        # 12000 chars approx 3000 tokens
        if len(context_block) > 12000:
             context_block = context_block[:12000] + "..."
        
        # Construct messages for LLMChatClient
        # Similar to simple_selfrag or naive_rag answer generation
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer is not contained in the context, say you are not sure.

Question:
{question}

Context:
{context_block}

Answer:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Using chat interface
            # Max tokens can be passed if needed, but client handles defaults
            # We use self.llm.chat directly
            logger.info(f"[RAPTOR] answering qid={question[:50]}..., ctx_len={len(context_block)}")
            answer = self.llm.chat(messages)
            
            # Apply strict filtering rules as requested
            final_answer = self._filter_answer(answer)
            return final_answer
            
        except Exception as e:
            logger.error(f"Raptor answer generation failed: {e}")
            return "Insufficient evidence"

    def _filter_answer(self, raw_answer: str) -> str:
        """
        Strictly filter the answer to match baseline standards.
        1. Remove reasoning/thinking process
        2. Apply keyword blacklist
        3. Enforce length limits
        """
        # 1. Clean reasoning and basic formatting
        cleaned = clean_model_answer(raw_answer)
        
        # 2. Keyword Blacklist (Case-insensitive)
        # Standard blacklist for "I don't know" responses
        blacklist = [
            "i am not sure",
            "i'm not sure", 
            "i do not know",
            "i don't know",
            "insufficient evidence",
            "not mentioned",
            "no information",
            "cannot answer",
            "cannot be answered",
            "context does not contain",
            "context does not provide",
            "you are not sure",  # Handle LLM echo of instructions
            "not sure"
        ]
        
        cleaned_lower = cleaned.lower()
        
        # Remove bold markers if present
        cleaned = cleaned.replace("**", "").strip()
        
        for phrase in blacklist:
            if phrase in cleaned_lower:
                return "Insufficient evidence"

                
        # 3. Length Limit
        # Standard short answer limit (e.g. < 100 chars or < 20 words)
        # If it's too long, it might be hallucinations or non-compliant
        if len(cleaned) > 200:
             logger.warning(f"Answer too long ({len(cleaned)} chars), truncating or rejecting. Answer: {cleaned[:50]}...")
             # Option A: Reject
             # return "Insufficient evidence"
             # Option B: Truncate (risky for correctness)
             # Let's reject for now to be safe and high-precision
             return "Insufficient evidence"
             
        return cleaned

