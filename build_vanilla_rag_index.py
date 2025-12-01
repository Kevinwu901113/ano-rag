import argparse
import json
import os
from pathlib import Path
from loguru import logger
from baselines.vanilla_rag.index import VanillaRAGIndexer

def main():
    parser = argparse.ArgumentParser(description="Build Index for Vanilla RAG Baseline")
    parser.add_argument("--doc-pool", type=str, required=True, help="Path to document pool (json/jsonl)")
    parser.add_argument("--output-index-path", type=str, default="indexes/vanilla_rag_index.faiss", help="Path to save FAISS index")
    parser.add_argument("--output-chunk-store-path", type=str, default="indexes/vanilla_rag_chunk_store.pkl", help="Path to save chunk store")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of docs to process (0 for all)")
    
    args = parser.parse_args()
    
    # 1. Load Documents
    docs = {}
    logger.info(f"Loading documents from {args.doc_pool}...")
    try:
        with open(args.doc_pool, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
            # Handle different doc pool formats (reusing logic from build_simple_graphrag_index.py)
            if isinstance(raw_data, dict):
                # Assume {doc_id: doc_obj} or {doc_id: text}
                for k, v in raw_data.items():
                    if isinstance(v, str):
                        docs[k] = v
                    elif isinstance(v, dict) and "text" in v:
                        docs[k] = v["text"]
                    elif isinstance(v, dict) and "content" in v:
                        docs[k] = v["content"]
            elif isinstance(raw_data, list):
                # Assume list of doc objects
                for item in raw_data:
                    if "mapped_id" in item and "doc_chunk" in item:
                        # MIRAGE format
                        key = f"{item['mapped_id']}_{len(docs)}"
                        text = item.get("doc_name", "") + "\n" + item["doc_chunk"]
                        docs[key] = text
                    elif "doc_id" in item and "text" in item:
                        docs[item["doc_id"]] = item["text"]
                    elif "id" in item and "content" in item:
                        docs[item["id"]] = item["content"]
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        return

    if args.limit > 0:
        docs = dict(list(docs.items())[:args.limit])
    
    logger.info(f"Loaded {len(docs)} documents.")
    
    # 2. Build Index
    indexer = VanillaRAGIndexer()
    indexer.build(docs, args.output_index_path, args.output_chunk_store_path)
    
if __name__ == "__main__":
    main()
