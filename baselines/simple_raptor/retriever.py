import re
import pickle
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from loguru import logger

from config.config_loader import config as global_config
from rag_core.embedding_client import EmbeddingEncoder
from rag_core.llm_client import LLMChatClient
from baselines.common.model_clients import get_default_embedding_client, get_default_llm_client
from baselines.simple_raptor.tree import TreeNode
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

class SimpleRaptorRetriever:
    def __init__(
        self, 
        index_path: str, 
        nodes_path: str, 
        chunk_store_path: str, 
        config: Optional[Dict] = None,
        embedding_client: Optional[EmbeddingEncoder] = None,
        llm_client: Optional[LLMChatClient] = None,
        top_k: Optional[int] = None,
        context_budget: Optional[int] = None,
    ):
        self.config = config or global_config.load_config()
        
        # Try new nested key first, then legacy key, then empty dict
        self.raptor_config = (
            self.config.get("retriever", {}).get("simple_raptor") or 
            self.config.get("simple_raptor") or 
            {}
        )
        
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
            self.llm = get_default_llm_client(self.config, llm_profile="generate")
            
        self.top_k_nodes = top_k or int(self.raptor_config.get("top_k_nodes", 10))
        self.max_answer_chunks = int(self.raptor_config.get("max_answer_chunks", 5))
        self.enable_name_rerank = bool(self.raptor_config.get("enable_name_rerank", True))
        self.last_hits: List[Dict[str, Any]] = []
        self.context_budget = int(context_budget or 0)

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

    def retrieve(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Public retrieval interface for evaluation.
        Mimics the node retrieval + chunk expansion logic but returns standard hits.
        """
        # 1. Retrieve Nodes
        retrieved_nodes = self.retrieve_nodes(question)
        
        # 2. Collect Leaf Chunks
        candidate_chunk_ids = []
        seen_chunks = set()
        
        for node in retrieved_nodes:
            for cid in node.descendant_chunk_ids:
                if cid not in seen_chunks and cid in self.chunk_store:
                    candidate_chunk_ids.append(cid)
                    seen_chunks.add(cid)
        
        # 3. Name-aware Rerank (optional)
        if self.enable_name_rerank:
            name = self._extract_name_from_question(question)
            if name:
                name_lower = name.lower()
                def has_name(cid: int) -> bool:
                    text = self.chunk_store.get(cid, "")
                    return name_lower in text.lower()
                candidate_chunk_ids.sort(key=lambda cid: (not has_name(cid)))

        # 4. Format hits
        hits = []
        limit = k if k > 0 else len(candidate_chunk_ids)
        for rank, cid in enumerate(candidate_chunk_ids[:limit]):
            doc_id = None
            if isinstance(cid, str) and "::" in cid:
                doc_id = cid.split("::", 1)[0]
            hits.append({
                "rank": rank + 1,
                "score": 1.0 / (rank + 1), # Dummy score
                "doc_id": doc_id,
                "sent_ids": None,
                "passage_id": str(cid),
                "text": self.chunk_store.get(cid),
            })
        
        self.last_hits = hits
        return hits

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
        # Record retrieval for logging
        self.last_hits = []
        for rank, cid in enumerate(chunk_ids):
            doc_id = None
            if isinstance(cid, str) and "::" in cid:
                doc_id = cid.split("::", 1)[0]
            self.last_hits.append(
                {
                    "rank": rank + 1,
                    "score": None,
                    "doc_id": doc_id,
                    "sent_ids": None,
                    "passage_id": str(cid),
                    "text": self.chunk_store.get(cid),
                }
            )
        annotated_hits = []
        for hit in self.last_hits:
            annotated_hits.append({**hit, "text": f"[{hit['rank']}] {hit.get('text', '')}"})
        context_block, _, _ = pack_contexts(annotated_hits, self.context_budget)
        
        # Construct messages for LLMChatClient
        prompt = PROMPT_TEMPLATE.format(
            context=context_block,
            question=question,
            final_instruction=build_final_instruction(),
        )
        
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Using chat interface
            # Max tokens can be passed if needed, but client handles defaults
            # We use self.llm.chat directly
            logger.info(f"[RAPTOR] answering qid={question[:50]}..., ctx_len={len(context_block)}")
            response = self.llm.chat(messages, llm_profile="generate")
            return response.content
            
        except Exception as e:
            logger.error(f"Raptor answer generation failed: {e}")
            return "Insufficient evidence"
