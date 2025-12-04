#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.simple_raptor.index import SimpleRaptorIndexer

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Simple Raptor index for MIRAGE dataset")
    parser.add_argument("--doc-pool", default="data/mirage_sample/doc_pool.json", help="Path to doc_pool.json")
    parser.add_argument("--out-dir", default="result/mirage_raptor", help="Directory to save index artifacts")
    parser.add_argument("--cluster-size", type=int, default=16, help="K-means cluster size")
    parser.add_argument("--lmstudio-endpoint", default=None)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    # 1. Load Docs
    doc_pool_path = Path(args.doc_pool)
    if not doc_pool_path.exists():
        raise FileNotFoundError(f"Doc pool not found: {doc_pool_path}")
        
    logger.info(f"Loading documents from {doc_pool_path}...")
    docs: Dict[str, str] = {}
    with open(doc_pool_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        # Handle list format (MIRAGE standard)
        if isinstance(raw_data, list):
            for i, item in enumerate(raw_data):
                # Unique ID construction
                base_id = item.get("doc_id") or item.get("mapped_id") or str(i)
                # Construct text from title + paragraphs
                title = item.get("title") or item.get("doc_name") or ""
                paragraphs = item.get("paragraphs")
                
                if paragraphs:
                    text = title + "\n" + "\n".join(str(p) for p in paragraphs if p)
                else:
                    # Fallback for doc_chunk (flat format)
                    text = title + "\n" + (item.get("doc_chunk") or "")
                
                # We use a compound key if needed, but doc_id usually suffices if unique
                docs[base_id] = text.strip()
        elif isinstance(raw_data, dict):
            # Handle simple dict format {id: text}
            docs = {k: str(v) for k, v in raw_data.items()}
            
    if not docs:
        logger.error("No documents found to index.")
        return

    # 2. Configure config overrides
    from config.config_loader import config as global_config
    
    if args.lmstudio_endpoint:
        global_config.set("lmstudio.endpoint", args.lmstudio_endpoint)
    if args.lmstudio_model:
        global_config.set("lmstudio.model", args.lmstudio_model)

    # Initialize Indexer
    # We rely on global config for embedding client (via EmbeddingClient inside Indexer)
    # And we pass explicit LLM config if needed, or let it load from global
    
    llm_config = None
    if args.lmstudio_endpoint or args.lmstudio_model:
        lm_cfg = global_config.load_config().get("lmstudio", {})
        llm_config = {
            "endpoint": args.lmstudio_endpoint or lm_cfg.get("endpoint"),
            "model": args.lmstudio_model or lm_cfg.get("model"),
            "temperature": args.temperature if args.temperature is not None else lm_cfg.get("temperature", 0.0)
        }
        
    indexer = SimpleRaptorIndexer(llm_config=llm_config)
    
    # 3. Build Index
    logger.info(f"Building Raptor index for {len(docs)} documents...")
    stats = indexer.build(docs, cluster_size=args.cluster_size)
    logger.info(f"Index built: {stats}")
    
    # 4. Save Artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = out_dir / "simple_raptor_index.faiss"
    nodes_path = out_dir / "simple_raptor_nodes.pkl"
    chunk_store_path = out_dir / "simple_raptor_chunk_store.pkl"
    
    indexer.save(str(index_path), str(nodes_path), str(chunk_store_path))
    logger.info(f"Artifacts saved to {out_dir}")

if __name__ == "__main__":
    main()
