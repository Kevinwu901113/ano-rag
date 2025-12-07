#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, List

from loguru import logger

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.simple_graphrag.build_graph import GraphBuilder
from structrag.llm_client import LLMChatClient
from config.config_loader import config as global_config

def main() -> None:
    parser = argparse.ArgumentParser(description="Build RelRAG (Simple GraphRAG) index for HotpotQA")
    parser.add_argument("--doc-pool", default="data/hotpotqa/doc_pool.json", help="Path to doc_pool.json")
    parser.add_argument("--out-dir", default="result/hotpot_relrag", help="Directory to save index artifacts")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of docs to process")
    parser.add_argument("--lmstudio-endpoint", default=None)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    
    args = parser.parse_args()

    # 1. Config
    if args.lmstudio_endpoint:
        global_config.set("lmstudio.endpoint", args.lmstudio_endpoint)
    if args.lmstudio_model:
        global_config.set("lmstudio.model", args.lmstudio_model)
        
    lm_cfg = global_config.load_config().get("lmstudio", {})
    endpoint = args.lmstudio_endpoint or lm_cfg.get("endpoint")
    model = args.lmstudio_model or lm_cfg.get("model")
    
    if not endpoint or not model:
        logger.error("LLM endpoint/model must be configured.")
        return

    llm_client = LLMChatClient(
        endpoint=endpoint,
        model=model,
        temperature=0.0
    )

    # 2. Load Documents
    doc_pool_path = Path(args.doc_pool)
    if not doc_pool_path.exists():
        logger.error(f"Doc pool not found: {doc_pool_path}")
        return
        
    logger.info(f"Loading documents from {doc_pool_path}...")
    docs: Dict[str, str] = {}
    with open(doc_pool_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
        # HotpotQA doc_pool format adaptation
        # Usually HotpotQA is list of [title, sentences] or similar
        # If it's our standardized doc_pool, it might be list of dicts
        
        if isinstance(raw_data, list):
            for i, item in enumerate(raw_data):
                # Try to extract ID and Text
                # HotpotQA original: [title, sentences]
                # Standardized: {"doc_id": ..., "title": ..., "text": ...}
                
                doc_id = str(item.get("doc_id") or item.get("id") or item.get("_id") or item.get("title") or str(i))
                title = item.get("title") or item.get("doc_name") or ""
                text = item.get("text") or item.get("content") or ""
                
                if not text and "sentences" in item:
                     # Reconstruct from sentences
                     sentences = item["sentences"]
                     if isinstance(sentences, list):
                         text = " ".join(sentences)
                
                if not text:
                     continue
                     
                # Prepend title for context
                if title and not text.startswith(title):
                    text = f"{title}\n{text}"
                    
                docs[doc_id] = text
                
        elif isinstance(raw_data, dict):
             # {doc_id: text} or {doc_id: {text: ...}}
             for k, v in raw_data.items():
                 if isinstance(v, str):
                     docs[k] = v
                 elif isinstance(v, dict):
                     t = v.get("text") or v.get("content")
                     if t:
                         docs[k] = t

    if not docs:
        logger.error("No documents loaded.")
        return

    if args.limit > 0:
        docs = dict(list(docs.items())[:args.limit])
    
    logger.info(f"Loaded {len(docs)} documents.")

    # 3. Build Graph
    builder = GraphBuilder(llm_client)
    # Since builder.build is async
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(builder.build(docs, concurrency=args.concurrency))

    # 4. Save
    os.makedirs(args.out_dir, exist_ok=True)
    output_graph = os.path.join(args.out_dir, "simple_graphrag_graph.pkl")
    output_chunks = os.path.join(args.out_dir, "simple_graphrag_chunk_store.pkl")
    
    builder.save(output_graph, output_chunks)
    logger.info(f"Saved graph to {output_graph} and chunks to {output_chunks}")

if __name__ == "__main__":
    main()
