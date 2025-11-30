import os
from typing import Optional
from structrag.llm_client import LLMChatClient
from baselines.simple_graphrag.retriever import GraphRetriever
from baselines.simple_graphrag.build_graph import GraphBuilder
from config.config_loader import config as global_config

# Global retriever instance
_retriever: Optional[GraphRetriever] = None

def get_retriever() -> GraphRetriever:
    global _retriever
    if _retriever is None:
        # Load config
        lm_cfg = global_config.get("lmstudio", {})
        endpoint = lm_cfg.get("endpoint")
        model = lm_cfg.get("model")
        
        if not endpoint or not model:
             raise ValueError("LM Studio endpoint/model must be configured")
             
        llm_client = LLMChatClient(endpoint=endpoint, model=model, temperature=0.0)
        
        # Paths
        graph_path = "simple_graphrag_graph.pkl"
        chunk_store_path = "simple_graphrag_chunk_store.pkl"
        
        if not os.path.exists(graph_path) or not os.path.exists(chunk_store_path):
            raise FileNotFoundError("Graph files not found. Run build_graph first.")
            
        _retriever = GraphRetriever(graph_path, chunk_store_path, llm_client)
        
    return _retriever

def answer(question: str) -> str:
    """
    Main entry point for the baseline.
    """
    retriever = get_retriever()
    return retriever.answer(question)
