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

from baselines.simple_raptor.index import SimpleRaptorIndexer

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Simple Raptor index for HotpotQA")
    parser.add_argument("--doc-pool", default="data/hotpotqa/doc_pool.json", help="Path to HotpotQA doc_pool.json")
    parser.add_argument("--out-dir", default="result/hotpot_raptor", help="Output directory")
    parser.add_argument("--lmstudio-endpoint", default=None, help="LLM Endpoint for summarization")
    parser.add_argument("--lmstudio-model", default=None, help="LLM Model for summarization")
    
    # Raptor specific params (optional overrides)
    parser.add_argument("--max-descendants", type=int, default=256)
    parser.add_argument("--top-k-nodes", type=int, default=32)
    parser.add_argument("--max-answer-chunks", type=int, default=32)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure global config overrides if provided
    from config.config_loader import config as global_config
    if args.lmstudio_endpoint:
        global_config.set("lmstudio.endpoint", args.lmstudio_endpoint)
    if args.lmstudio_model:
        global_config.set("lmstudio.model", args.lmstudio_model)
        
    # Update raptor specific config
    raptor_cfg = global_config.get("baselines.simple_raptor", {}) or {}
    raptor_cfg["max_descendants"] = args.max_descendants
    raptor_cfg["top_k_nodes"] = args.top_k_nodes
    raptor_cfg["max_answer_chunks"] = args.max_answer_chunks
    global_config.set("baselines.simple_raptor", raptor_cfg)

    # Initialize Indexer
    # We can just manually construct the client here since we have the values
    from baselines.common.model_clients import get_default_llm_client
    llm_client = get_default_llm_client()

    indexer = SimpleRaptorIndexer(
        llm_client=llm_client,
        config=global_config.load_config()
    )
    
    logger.info(f"Building Raptor index from {args.doc_pool}...")
    stats = indexer.build(args.doc_pool, str(out_dir))
    
    logger.info("Raptor index built. Stats:\n{}", json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
