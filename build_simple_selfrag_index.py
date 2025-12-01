#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines.simple_selfrag.index import SimpleSelfRAGIndexer
from config.config_loader import config as global_config

def load_docs(doc_pool_path: str, limit: int = 0) -> Dict[str, str]:
    logger.info(f"Loading documents from {doc_pool_path}...")
    docs = {}
    with open(doc_pool_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Handle different formats
    if isinstance(data, dict):
        # {doc_id: text} or {doc_id: {"text": ...}}
        for doc_id, content in data.items():
            if isinstance(content, str):
                docs[doc_id] = content
            elif isinstance(content, dict):
                docs[doc_id] = content.get("text") or content.get("content", "")
    elif isinstance(data, list):
        # [{"doc_id": ..., "text": ...}, ...]
        for item in data:
            doc_id = item.get("doc_id") or item.get("id")
            text = item.get("text") or item.get("content", "")
            if doc_id and text:
                docs[str(doc_id)] = text
                
    if limit > 0:
        logger.info(f"Limiting to {limit} documents")
        docs = dict(list(docs.items())[:limit])
        
    logger.info(f"Loaded {len(docs)} documents")
    return docs

def main():
    parser = argparse.ArgumentParser(description="Build Simple Self-RAG Index")
    parser.add_argument("--doc-pool", type=str, required=True, help="Path to document pool JSON")
    parser.add_argument("--output-index", type=str, default="indexes/simple_selfrag_index.faiss", help="Output path for FAISS index")
    parser.add_argument("--output-chunks", type=str, default="indexes/simple_selfrag_chunk_store.pkl", help="Output path for chunk store")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents to process")
    
    args = parser.parse_args()
    
    # Load config for embedding
    embedding_config = global_config.get("retriever", {}).get("simple_selfrag", {}).get("embedding")
    if not embedding_config:
        # Fallback to global retriever embedding config or default
        embedding_config = global_config.get("retriever", {}).get("embedding")
        
    # Initialize indexer
    indexer = SimpleSelfRAGIndexer(embedding_config)
    
    # Load docs
    docs = load_docs(args.doc_pool, args.limit)
    
    # Build index
    stats = indexer.build(docs)
    logger.info(f"Build complete. Stats: {stats}")
    
    # Save index
    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_chunks), exist_ok=True)
    indexer.save(args.output_index, args.output_chunks)

if __name__ == "__main__":
    main()
