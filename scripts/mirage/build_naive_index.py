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

from baselines.naive_rag import MirageNaiveIndexer, NaiveChunker


def main() -> None:
    parser = argparse.ArgumentParser(description="Build naive MIRAGE RAG index from doc_pool.json")
    parser.add_argument("--doc-pool", default="data/mirage_sample/doc_pool.json", help="Path to MIRAGE doc_pool.json")
    parser.add_argument("--out-dir", default="result/mirage_naive", help="Output directory for FAISS + chunks.jsonl")
    parser.add_argument("--target-tokens", type=int, default=320, help="Target tokens per chunk")
    parser.add_argument("--max-tokens", type=int, default=384, help="Hard cap tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=64, help="Token overlap between chunks")
    parser.add_argument("--no-title", action="store_true", help="Do not prepend doc title to chunk text")
    args = parser.parse_args()

    chunker = NaiveChunker(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        append_title=not args.no_title,
    )
    builder = MirageNaiveIndexer(chunker=chunker)
    stats = builder.build(args.doc_pool, args.out_dir)

    logger.info("Naive MIRAGE index built. Stats:\n{}", json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
