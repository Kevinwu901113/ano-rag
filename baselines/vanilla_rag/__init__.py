from __future__ import annotations

from typing import Optional
from loguru import logger

from baselines.vanilla_rag.retriever import VanillaRAGRetriever

_RETRIEVER: Optional[VanillaRAGRetriever] = None

def get_retriever(
    index_path: str = "indexes/vanilla_rag_index.faiss",
    chunk_store_path: str = "indexes/vanilla_rag_chunk_store.pkl",
) -> VanillaRAGRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        logger.info(f"Initializing VanillaRAGRetriever with index={index_path}, chunks={chunk_store_path}")
        _RETRIEVER = VanillaRAGRetriever(index_path, chunk_store_path)
    return _RETRIEVER

def answer(question: str, index_path: str = "indexes/vanilla_rag_index.faiss", chunk_store_path: str = "indexes/vanilla_rag_chunk_store.pkl") -> str:
    """
    Answer a question using Vanilla RAG.
    Lazily initializes the retriever on first call.
    """
    retriever = get_retriever(index_path, chunk_store_path)
    return retriever.answer(question)
