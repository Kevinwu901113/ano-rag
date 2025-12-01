import os
from typing import Optional
from baselines.simple_selfrag.retriever import SimpleSelfRAGRetriever
from config.config_loader import config as global_config

_retriever: Optional[SimpleSelfRAGRetriever] = None

def get_retriever() -> SimpleSelfRAGRetriever:
    global _retriever
    if _retriever is None:
        # Default paths
        index_path = "indexes/simple_selfrag_index.faiss"
        chunk_store_path = "indexes/simple_selfrag_chunk_store.pkl"
        
        if not os.path.exists(index_path) or not os.path.exists(chunk_store_path):
            raise FileNotFoundError(f"Index files not found at {index_path} or {chunk_store_path}. Please run build_simple_selfrag_index.py first.")
            
        _retriever = SimpleSelfRAGRetriever(index_path, chunk_store_path)
        
    return _retriever

def answer(question: str) -> str:
    """
    Main entry point for Simple Self-RAG.
    """
    retriever = get_retriever()
    return retriever.answer(question)
