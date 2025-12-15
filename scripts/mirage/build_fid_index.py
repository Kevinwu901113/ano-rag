
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

from baselines.fid_rag.indexer import FiDNaiveIndexer, NaiveChunker


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FiD RAG index from doc_pool.json")
    parser.add_argument("--doc-pool", default="data/mirage_sample/doc_pool.json", help="Path to MIRAGE doc_pool.json")
    parser.add_argument("--out-dir", default="result/mirage_fid_index", help="Output directory for FAISS + chunks.jsonl")
    parser.add_argument(
        "--embed-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Embedding device preference (auto prefers CUDA, falls back to CPU)",
    )
    parser.add_argument("--embed-batch-size", type=int, default=4, help="Embedding batch size")
    parser.add_argument("--embed-max-length", type=int, default=512, help="Embedding max sequence length")
    parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-8B", help="Embedding model name or path")
    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--embed-normalize", dest="embed_normalize", action="store_true", help="L2-normalize embeddings")
    norm.add_argument(
        "--no-embed-normalize",
        dest="embed_normalize",
        action="store_false",
        help="Disable L2-normalization",
    )
    parser.set_defaults(embed_normalize=True)
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
    builder = FiDNaiveIndexer(chunker=chunker)
    stats = builder.build(
        args.doc_pool,
        args.out_dir,
        embed_model=args.embed_model,
        embed_device=args.embed_device,
        embed_batch_size=args.embed_batch_size,
        embed_max_length=args.embed_max_length,
        embed_normalize=args.embed_normalize,
    )

    logger.info("FiD MIRAGE index built. Stats:\n{}", json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
