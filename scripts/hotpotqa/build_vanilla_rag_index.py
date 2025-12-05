#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.vanilla_rag.index import VanillaRAGIndexer

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Vanilla RAG index for HotpotQA")
    parser.add_argument("--doc-pool", default="data/hotpotqa/doc_pool.json", help="Path to HotpotQA doc_pool.json")
    parser.add_argument("--out-dir", default="result/hotpot_vanilla", help="Output directory for FAISS + chunks.pkl")
    # Use same params as Naive RAG or adjusted for Vanilla
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=256, help="Chunk overlap in tokens")
    args = parser.parse_args()

    # Ensure output dir exists
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Indexer
    # Assuming VanillaRAGIndexer takes chunking params or similar
    indexer = VanillaRAGIndexer(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Build Index
    logger.info(f"Building Vanilla RAG index from {args.doc_pool}...")
    stats = indexer.build(args.doc_pool, str(out_dir))
    
    logger.info("Vanilla RAG index built. Stats:\n{}", json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
