#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.simple_raptor.retriever import SimpleRaptorRetriever
from rag_core.llm_client import LLMChatClient

def _select_workspace(root: Path, dataset: str, new: bool) -> Path:
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        
    import re
    pattern = re.compile(r"^(?P<idx>\d{3})-(?P<name>.+)$")
    candidates = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match and match.group("name") == dataset:
            candidates.append((int(match.group("idx")), entry))
    candidates.sort(key=lambda x: x[0])
    
    if new or not candidates:
        next_idx = candidates[-1][0] + 1 if candidates else 0
        name = f"{next_idx:03d}-{dataset}"
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        return target
    return candidates[-1][1]

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Simple Raptor baseline on MIRAGE dataset.json")
    parser.add_argument("--dataset-path", default="data/mirage_sample/dataset.json")
    parser.add_argument("--index-dir", default="result/mirage_raptor", help="Directory containing Raptor index files")
    parser.add_argument("--index-path", default=None, help="Optional explicit FAISS index path")
    parser.add_argument("--nodes-path", default=None, help="Optional explicit nodes.pkl path")
    parser.add_argument("--chunk-store-path", default=None, help="Optional explicit chunk_store.pkl path")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--work-dir", default=None, help="Where to write qa.tsv etc. Default: auto under result_root")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace (do not reuse latest)")
    parser.add_argument("--lmstudio-endpoint", default=None)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-debug", action="store_true", help="Skip writing retrieval debug JSONL")
    args = parser.parse_args()
    
    # 1. Setup config for the baseline (it uses global config)
    from config.config_loader import config as global_config
    if args.lmstudio_endpoint:
        global_config.set("lmstudio.endpoint", args.lmstudio_endpoint)
    if args.lmstudio_model:
        global_config.set("lmstudio.model", args.lmstudio_model)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    index_dir = Path(args.index_dir)
    index_path = Path(args.index_path) if args.index_path else index_dir / "simple_raptor_index.faiss"
    nodes_path = Path(args.nodes_path) if args.nodes_path else index_dir / "simple_raptor_nodes.pkl"
    chunk_store_path = Path(args.chunk_store_path) if args.chunk_store_path else index_dir / "simple_raptor_chunk_store.pkl"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Index missing: {index_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes missing: {nodes_path}")
    if not chunk_store_path.exists():
        raise FileNotFoundError(f"Chunk store missing: {chunk_store_path}")

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = _select_workspace(Path(args.result_root), "mirage_raptor", args.new)
    logger.info("Writing outputs to {}", work_dir)
    
    # Configure LLM Client explicit overrides if provided
    llm_client = None
    if args.lmstudio_endpoint or args.lmstudio_model:
        # We need to fetch defaults first
        lm_cfg = global_config.load_config().get("lmstudio", {})
        endpoint = args.lmstudio_endpoint or lm_cfg.get("endpoint")
        model = args.lmstudio_model or lm_cfg.get("model")
        temp = args.temperature if args.temperature is not None else lm_cfg.get("temperature", 0.0)
        max_tokens = args.max_new_tokens if args.max_new_tokens is not None else lm_cfg.get("max_tokens", 8192)
        
        llm_client = LLMChatClient(
            endpoint=endpoint,
            model=model,
            temperature=float(temp)
        )

    # Initialize Retriever
    retriever = SimpleRaptorRetriever(
        str(index_path),
        str(nodes_path),
        str(chunk_store_path),
        llm_client=llm_client,
        top_k=args.topk
    )
    
    # Run Dataset
    results = []
    qa_lines = []
    
    limit = args.limit if args.limit and args.limit > 0 else len(dataset)
    logger.info(f"Running Raptor baseline on {limit} questions...")
    
    for i, item in enumerate(dataset[:limit]):
        question = item.get("query") or item.get("question")
        qid = item.get("query_id") or str(i)
        
        try:
            logger.info(f"Processing Q{i}: {question}")
            ans = retriever.answer(question)
            
            results.append({
                "query_id": qid,
                "question": question,
                "answer": ans,
            })
            qa_lines.append(f"{question}\t{ans.replace(chr(10), ' ')}")
        except Exception as e:
            logger.exception(f"Error Q{i}: {e}")
            
    # Save results
    out_json = work_dir / "answers.json"
    out_qa = work_dir / "qa.tsv"
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    with open(out_qa, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_lines))
        
    logger.info("Raptor baseline complete. qa.tsv: {}", out_qa)

if __name__ == "__main__":
    main()
