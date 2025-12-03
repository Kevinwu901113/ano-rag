import pickle
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict
from loguru import logger

from config.config_loader import config as global_config
from utils.embedding_utils import EmbeddingEncoder
from baselines.naive_rag.runner import LLMClient
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
        llm_client: Optional[LLMClient] = None
    ):
        self.config = config or global_config
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
            # Reuse EmbeddingClient to support overrides and consistent loading
            from retriever.embedding_client import EmbeddingClient
            
            # Base config from retriever section
            base_emb_cfg = self.config.get("retriever", {}).get("embedding") or {}
            # Raptor specific override (if any)
            raptor_emb_cfg = self.raptor_config.get("embedding") or {}
            
            # Merge: base -> raptor override
            final_cfg = base_emb_cfg.copy()
            final_cfg.update(raptor_emb_cfg)
            
            # Instantiate EmbeddingClient with merged config
            # We don't set 'enabled'=True because we don't want it to load the main FAISS index
            # We only want the encoder.
            client = EmbeddingClient(final_cfg)
            client.load_encoder()
            self.encoder = client._encoder
            
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
            max_tokens = lm_cfg.get("max_tokens", 8192)
            stop = lm_cfg.get("stop", [])
            
            if not endpoint or not model:
                 # Fallback if config is missing keys (e.g. loaded from minimal config)
                 # Try to reload global default
                 from config.config_loader import config as global_config_loader
                 full_cfg = global_config_loader.load_config()
                 full_lm = full_cfg.get("lmstudio", {})
                 endpoint = endpoint or full_lm.get("endpoint", "http://127.0.0.1:1234/v1")
                 model = model or full_lm.get("model", "qwen2.5-7b-instruct")
            
            self.llm = LLMClient(
                endpoint=endpoint,
                model=model,
                temperature=float(temp),
                max_tokens=int(max_tokens),
                stop=stop
            )
            
        self.top_k_nodes = self.raptor_config.get("top_k_nodes", 10)
        self.max_answer_chunks = self.raptor_config.get("max_answer_chunks", 5)

    def retrieve_nodes(self, question: str, top_k: int) -> List[TreeNode]:
        """
        Retrieve relevant nodes from the tree.
        """
        q_vec = self.encoder.encode([question])
        if len(q_vec) > 0:
             faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, top_k)
        
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
        retrieved_nodes = self.retrieve_nodes(question, self.top_k_nodes)
        
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
        
        raw_answer = self.llm.answer(question, context_block)
        
        # 4. Normalize/Clean Answer
        final_answer = _strip_reasoning(raw_answer)
        
        # Enforce "Insufficient evidence" normalization if close
        if "insufficient evidence" in final_answer.lower():
            final_answer = "Insufficient evidence"
            
        return final_answer
