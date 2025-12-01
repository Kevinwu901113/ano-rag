import os
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
import numpy as np
import faiss
from loguru import logger
from tqdm import tqdm

# Reuse common component if available, or implement simple chunker
from structrag.embedding_client import EmbeddingEncoder

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str

class SimpleSelfRAGChunker:
    def __init__(self, target_tokens: int = 512, overlap_tokens: int = 50):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        # Simple word-based chunking as approximation for token-based
        # In a real scenario, we might use tiktoken or similar
        words = text.split()
        if not words:
            return []
        
        chunks = []
        step = self.target_tokens - self.overlap_tokens
        if step < 1:
            step = 1
            
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.target_tokens]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{doc_id}::chunk_{len(chunks)}"
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text))
            
            if i + self.target_tokens >= len(words):
                break
                
        return chunks

class SimpleSelfRAGIndexer:
    def __init__(self, embedding_config: Optional[Dict] = None):
        self.chunker = SimpleSelfRAGChunker()
        
        # Initialize embedding encoder
        if embedding_config:
            self.encoder = EmbeddingEncoder(
                provider=embedding_config.get("provider", "huggingface"),
                model=embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                device=embedding_config.get("device", "cpu")
            )
        else:
            # Default fallback
            self.encoder = EmbeddingEncoder(
                provider="huggingface",
                model="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
            
        self.chunks: List[Chunk] = []
        self.index = None
        self.chunk_ids: List[str] = []

    def build(self, docs: Dict[str, str]) -> Dict[str, Any]:
        """
        Build index from documents.
        docs: {doc_id: text}
        """
        logger.info(f"Chunking {len(docs)} documents...")
        all_chunks = []
        for doc_id, text in tqdm(docs.items(), desc="Chunking"):
            doc_chunks = self.chunker.chunk(doc_id, text)
            all_chunks.extend(doc_chunks)
        
        self.chunks = all_chunks
        self.chunk_ids = [c.chunk_id for c in all_chunks]
        chunk_texts = [c.text for c in all_chunks]
        
        logger.info(f"Encoding {len(all_chunks)} chunks...")
        # Batch encoding could be added here for efficiency
        embeddings = self.encoder.encode(chunk_texts, normalize_embeddings=True)
        
        # Build FAISS index
        d = embeddings.shape[1]
        logger.info(f"Building FAISS index (dim={d})...")
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        
        return {
            "num_docs": len(docs),
            "num_chunks": len(all_chunks),
            "dimension": d
        }

    def save(self, index_path: str, chunks_path: str) -> None:
        if self.index is None:
            raise ValueError("Index not built yet")
            
        # Save FAISS index
        logger.info(f"Saving index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        # Save chunk store (chunk_id -> text)
        chunk_store = {c.chunk_id: c.text for c in self.chunks}
        logger.info(f"Saving chunk store to {chunks_path}")
        with open(chunks_path, "wb") as f:
            pickle.dump(chunk_store, f)
            
        # Save meta (chunk_ids order)
        meta_path = index_path + ".meta.pkl"
        logger.info(f"Saving meta to {meta_path}")
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunk_ids, f)
