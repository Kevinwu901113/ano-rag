import pickle
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict
from loguru import logger

from config.config_loader import config as global_config
from rag_core.embedding_client import EmbeddingEncoder
from rag_core.llm_client import LLMChatClient
from baselines.simple_raptor.tree import TreeNode

def _strip_reasoning(answer: str) -> str:
    """
    Standard answer cleaning logic reused from other baselines.
    """
    text = answer
    while True:
        start = text.find("<think>")
        if start == -1:
            break
        end = text.find("</think>", start + 7)
        if end == -1:
            text = text[:start] + text[start + 7 :]
            break
        text = text[:start] + text[end + len("</think>") :]
    
    # Additional common cleaning
    text = text.strip()
    # Remove common prefixes
    if text.lower().startswith("answer:"):
        text = text[7:].strip()
    return text

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
            
        # Initialize Embedding Encoder
        if embedding_client:
            self.encoder = embedding_client
        else:
            # Use rag_core.embedding_client.EmbeddingEncoder
            
            # Base config from retriever section
            base_emb_cfg = self.config.get("retriever", {}).get("embedding") or {}
            # Raptor specific override (if any)
            raptor_emb_cfg = self.raptor_config.get("embedding") or {}
            
            # Merge: base -> raptor override
            final_cfg = base_emb_cfg.copy()
            final_cfg.update(raptor_emb_cfg)
            
            provider = final_cfg.get("provider", "huggingface")
            model = final_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            device = final_cfg.get("device", "cpu")
            
            # Additional args
            extra_kwargs = {k: v for k, v in final_cfg.items() if k not in ["provider", "model", "device"]}

            self.encoder = EmbeddingEncoder(
                provider=provider,
                model=model,
                device=device,
                **extra_kwargs
            )
            
        # Initialize LLM Client
        if llm_client:
            self.llm = llm_client
        else:
            lm_cfg = self.config.get("lmstudio", {})
            endpoint = lm_cfg.get("endpoint")
            model = lm_cfg.get("model")
            temp = lm_cfg.get("temperature", 0.0)
            # Raptor uses summarization and answering, which might need longer context
            # But we should respect global config if set
            self.max_tokens = int(lm_cfg.get("max_tokens", 8192))
            stop = lm_cfg.get("stop", [])
            
            if not endpoint or not model:
                 # Fallback if config is missing keys (e.g. loaded from minimal config)
                 # Try to reload global default
                 from config.config_loader import config as global_config_loader
                 full_cfg = global_config_loader.load_config()
                 full_lm = full_cfg.get("lmstudio", {})
                 endpoint = endpoint or full_lm.get("endpoint", "http://127.0.0.1:1234/v1")
                 model = model or full_lm.get("model", "qwen2.5-7b-instruct")
            
            self.llm = LLMChatClient(
                endpoint=endpoint,
                model=model,
                temperature=float(temp),
                stop=stop
            )
            
        self.top_k_nodes = top_k or self.raptor_config.get("top_k_nodes", 10)
        self.max_answer_chunks = self.raptor_config.get("max_answer_chunks", 5)

    def retrieve_nodes(self, question: str, top_k: Optional[int] = None) -> List[TreeNode]:
        """
        Retrieve relevant nodes from the tree.
        """
        search_k = top_k or self.top_k_nodes
        
        q_vec = self.encoder.encode([question])
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
        chunk_ids = []
        seen_chunks = set()
        
        for node in retrieved_nodes:
            for cid in node.descendant_chunk_ids:
                if cid not in seen_chunks:
                    chunk_ids.append(cid)
                    seen_chunks.add(cid)
                    if len(chunk_ids) >= self.max_answer_chunks:
                        break
            if len(chunk_ids) >= self.max_answer_chunks:
                break
                
        # Retrieve chunk text
        context_texts = []
        for cid in chunk_ids:
            text = self.chunk_store.get(cid)
            if text:
                context_texts.append(text)
                
        context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_texts)])
        
        # Truncate to avoid 400 error (similar to index.py fix)
        if len(context_block) > 20000:
             context_block = context_block[:20000] + "..."
        
        # Construct messages for LLMChatClient
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"}
        ]
        
        raw_answer = self.llm.chat(messages, max_tokens=getattr(self, 'max_tokens', 1024))
        
        # 4. Normalize/Clean Answer
        final_answer = _strip_reasoning(raw_answer)
        
        # Enforce "Insufficient evidence" normalization if close
        if "insufficient evidence" in final_answer.lower():
            final_answer = "Insufficient evidence"
            
        return final_answer
