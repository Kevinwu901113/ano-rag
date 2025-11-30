#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.naive_rag.index import MirageNaiveIndexer
from baselines.naive_rag.runner import NaiveRAGRunner
from main_mirage import _load_dataset, _select_workspace
from loguru import logger
from utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Naive RAG Baseline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Indexing command
    build = sub.add_parser("index", help="Build FAISS index from doc_pool")
    build.add_argument("--doc-pool", required=True, help="Path to doc_pool.json")
    build.add_argument("--out", required=True, help="Output directory for index")

    # Running command
    run = sub.add_parser("run", help="Run RAG on a dataset")
    run.add_argument("--index-dir", required=True, help="Directory containing index.faiss and chunks.jsonl")
    run.add_argument("--dataset", required=True, help="Path to dataset.json")
    run.add_argument("--work-dir", required=True, help="Output workspace directory")
    run.add_argument("--lm-endpoint", required=True, help="LLM API endpoint")
    run.add_argument("--lm-model", required=True, help="LLM model name")
    run.add_argument("--topk", type=int, default=5, help="Number of chunks to retrieve")
    run.add_argument("--limit", type=int, default=None, help="Limit number of examples")

    args = parser.parse_args()
    setup_logging()

    if args.cmd == "index":
        indexer = MirageNaiveIndexer()
        indexer.build(args.doc_pool, args.out)
    
    elif args.cmd == "run":
        index_path = Path(args.index_dir) / "index.faiss"
        chunks_path = Path(args.index_dir) / "chunks.jsonl"
        
        if not index_path.exists() or not chunks_path.exists():
            logger.error(f"Index or chunks not found in {args.index_dir}")
            sys.exit(1)

        runner = NaiveRAGRunner(
            index_path=str(index_path),
            chunks_path=str(chunks_path),
            topk=args.topk,
            lm_endpoint=args.lm_endpoint,
            lm_model=args.lm_model,
        )
        
        data = _load_dataset(Path(args.dataset))
        runner.run_dataset(
            data,
            work_dir=args.work_dir,
            limit=args.limit
        )

if __name__ == "__main__":
    main()
