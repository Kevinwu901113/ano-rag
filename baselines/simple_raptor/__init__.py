import os
from typing import Optional
from baselines.simple_raptor.retriever import SimpleRaptorRetriever
from config.config_loader import config as global_config

_retriever: Optional[SimpleRaptorRetriever] = None

def get_retriever() -> SimpleRaptorRetriever:
    global _retriever
    if _retriever is None:
        # Default paths
        index_path = "indexes/simple_raptor_index.faiss"
        nodes_path = "indexes/simple_raptor_nodes.pkl"
        chunk_store_path = "indexes/simple_raptor_chunk_store.pkl"
        
        if not os.path.exists(index_path) or not os.path.exists(nodes_path) or not os.path.exists(chunk_store_path):
            raise FileNotFoundError(
                f"Simple RAPTOR index files not found. "
                f"Expected: {index_path}, {nodes_path}, {chunk_store_path}. "
                f"Please run build_simple_raptor_index.py first."
            )
            
        _retriever = SimpleRaptorRetriever(index_path, nodes_path, chunk_store_path, config=global_config.load_config())
        
    return _retriever

def answer(question: str) -> str:
    """
    Main entry point for Simple RAPTOR.
    """
    retriever = get_retriever()
    return retriever.answer(question)
