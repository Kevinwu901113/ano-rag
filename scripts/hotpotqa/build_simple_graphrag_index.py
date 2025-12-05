#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.simple_graphrag.build_graph import GraphBuilder
from structrag.llm_client import LLMChatClient
from config.config_loader import config as global_config

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Simple GraphRAG index for HotpotQA")
    parser.add_argument("--doc-pool", default="data/hotpotqa/doc_pool.json", help="Path to HotpotQA doc_pool.json")
    parser.add_argument("--out-dir", default="result/hotpot_graphrag", help="Output directory")
    parser.add_argument("--lmstudio-endpoint", default=None, help="LLM Endpoint for graph extraction")
    parser.add_argument("--lmstudio-model", default=None, help="LLM Model for graph extraction")
    
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure global config overrides
    if args.lmstudio_endpoint:
        global_config.set("lmstudio.endpoint", args.lmstudio_endpoint)
    if args.lmstudio_model:
        global_config.set("lmstudio.model", args.lmstudio_model)
        
    lm_cfg = global_config.load_config().get("lmstudio", {})
    endpoint = args.lmstudio_endpoint or lm_cfg.get("endpoint")
    model = args.lmstudio_model or lm_cfg.get("model")

    if not endpoint or not model:
        logger.error("LLM endpoint and model must be configured for GraphRAG build.")
        return

    # Load documents
    docs = {}
    logger.info(f"Loading documents from {args.doc_pool}...")
    with open(args.doc_pool, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                doc_id = item.get("doc_id")
                text = item.get("text")
                if doc_id and text:
                    docs[doc_id] = text
            except json.JSONDecodeError:
                pass
    
    if not docs:
        logger.error("No documents found.")
        return
        
    logger.info(f"Building graph for {len(docs)} documents...")
    
    # Initialize Builder
    llm_client = LLMChatClient(
        endpoint=endpoint,
        model=model,
        temperature=0.0
    )
    
    builder = GraphBuilder(llm_client)
    
    # Run async build
    try:
        asyncio.run(builder.build(docs))
    except Exception as e:
        logger.exception("Error during graph build")
        return

    # Save
    graph_pkl = out_dir / "graph.pkl"
    chunk_store_pkl = out_dir / "chunk_store.pkl"
    
    builder.save(str(graph_pkl), str(chunk_store_pkl))
    logger.info(f"GraphRAG index built and saved to {out_dir}")


if __name__ == "__main__":
    main()
