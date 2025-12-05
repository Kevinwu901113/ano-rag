import os
from typing import Optional
from structrag.llm_client import LLMChatClient
from baselines.simple_graphrag.retriever import GraphRetriever
from baselines.simple_graphrag.build_graph import GraphBuilder
from baselines.simple_graphrag.runner import SimpleGraphRAGRunner
from config.config_loader import config as global_config

# Global retriever instance
_retriever: Optional[GraphRetriever] = None

def get_retriever(index_dir: Optional[str] = None) -> GraphRetriever:
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
        base_dir = index_dir if index_dir else "."
        graph_path = os.path.join(base_dir, "simple_graphrag_graph.pkl")
        chunk_store_path = os.path.join(base_dir, "simple_graphrag_chunk_store.pkl")
        
        if not os.path.exists(graph_path) or not os.path.exists(chunk_store_path):
            raise FileNotFoundError(f"Graph files not found in {base_dir}. Run build_graph first.")
            
        _retriever = GraphRetriever(graph_path, chunk_store_path, llm_client)
        
    return _retriever

def answer(
    question: str, 
    index_dir: Optional[str] = None,
    graph_path: Optional[str] = None,
    chunk_store_path: Optional[str] = None
) -> str:
    """
    Main entry point for the baseline.
    Allows explicit path injection for flexibility.
    """
    global _retriever
    
    # If explicit paths are provided, bypass global singleton logic or re-init
    if graph_path and chunk_store_path:
        # Load config
        lm_cfg = global_config.load_config().get("lmstudio", {})
        endpoint = lm_cfg.get("endpoint")
        model = lm_cfg.get("model")
        
        if not endpoint or not model:
             raise ValueError("LM Studio endpoint/model must be configured")
             
        llm_client = LLMChatClient(endpoint=endpoint, model=model, temperature=0.0)
        
        # Create ephemeral retriever
        temp_retriever = GraphRetriever(graph_path, chunk_store_path, llm_client)
        return temp_retriever.answer(question)

    retriever = get_retriever(index_dir)
    return retriever.answer(question)
