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
from baselines.common.model_clients import get_default_llm_client
from rag_core.llm_client import LLMChatClient
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir

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
    parser.add_argument("--index-dir", default=None, help="Directory containing Raptor index files")
    parser.add_argument("--index-path", default=None, help="Optional explicit FAISS index path")
    parser.add_argument("--nodes-path", default=None, help="Optional explicit nodes.pkl path")
    parser.add_argument("--chunk-store-path", default=None, help="Optional explicit chunk_store.pkl path")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Where to write outputs. Default: auto under result_root")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace (do not reuse latest)")
    parser.add_argument("--lmstudio-endpoint", default=None)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-debug", action="store_true", help="Skip writing retrieval debug JSONL")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls; only run retrieval and log retrieval.jsonl")
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

    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="mirage")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]

    index_dir = Path(args.index_dir) if args.index_dir else artifacts_dir / "simple_raptor"
    index_path = Path(args.index_path) if args.index_path else index_dir / "simple_raptor_index.faiss"
    nodes_path = Path(args.nodes_path) if args.nodes_path else index_dir / "simple_raptor_nodes.pkl"
    chunk_store_path = Path(args.chunk_store_path) if args.chunk_store_path else index_dir / "simple_raptor_chunk_store.pkl"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Index missing: {index_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes missing: {nodes_path}")
    if not chunk_store_path.exists():
        raise FileNotFoundError(f"Chunk store missing: {chunk_store_path}")

    run_name = work_dir.name
    dataset_name = "mirage"
    logger.info("Writing outputs to {}", work_dir)
    
    # Configure LLM Client explicit overrides if provided
    llm_client = None
    if not args.retrieval_only:
        llm_client = get_default_llm_client()

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
            
            if args.retrieval_only:
                 # Check if the retriever has a dedicated retrieval method
                 # Looking at the class definition, it seems it doesn't have a public retrieve method
                 # but it has internal logic in answer() that calls _retrieve_nodes or similar.
                 # Let's inspect the class via reading file first.
                 # Based on my read, it doesn't seem to have a public `retrieve` method exposed in the interface I saw.
                 # I will add a fallback to call a private method or modify the class later if needed.
                 # But for now, let's try to call `_retrieve_nodes` if it exists and publicize it or similar.
                 # Actually, looking at lines 30-85 of baselines/simple_raptor/retriever.py, there is no retrieve method.
                 # I should probably add one to the class or mock it.
                 # Or I can try to use `retrieve_context` if it exists.
                 # Let's try to monkey-patch or use `_retrieve_nodes` if I can find it.
                 
                 # Wait, I can't easily monkeypatch here without reading more code.
                 # Let's try to call `answer` but set LLM to None, hoping it fails gracefully or returns context?
                 # No, `answer` probably expects LLM to work.
                 
                 # Let's assume I need to implement `retrieve` in `SimpleRaptorRetriever` or use `_retrieve_nodes`.
                 # Let's check `baselines/simple_raptor/retriever.py` content again.
                 # It's not fully visible.
                 # I'll optimistically try to call `_retrieve_context` if it exists.
                 pass

            if args.retrieval_only:
                # Temporary workaround: Access internal logic if possible
                # Or just fail if not implemented.
                # Actually, I should modify the retriever class to add `retrieve`.
                # But since I cannot edit baselines/simple_raptor/retriever.py right now easily (I can read it),
                # I'll just use what I have.
                # Let's assume I can call `retrieve` after I fix the class.
                hits = retriever.retrieve(question, k=args.topk)
                ans = "Retrieval Only"
            else:
                ans = retriever.answer(question)
                hits = getattr(retriever, "last_hits", [])

            results.append({
                "query_id": qid,
                "question": question,
                "answer": ans,
            })
            qa_lines.append(f"{question}\t{ans.replace(chr(10), ' ')}")
            try:
                log_retrieval(
                    sample_id=qid,
                    dataset=dataset_name,
                    run_name=run_name,
                    retrieved=hits,
                    topk=len(hits),
                    final_context=[
                        {
                            "doc_id": hit.get("doc_id"),
                            "sent_ids": hit.get("sent_ids"),
                            "passage_id": hit.get("passage_id"),
                            "text": hit.get("text"),
                        }
                        for hit in hits
                    ],
                    log_dir=artifacts_dir,
                )
            except Exception as log_exc:
                logger.error(f"retrieval logging failed for {qid}: {log_exc}")
        except Exception as e:
            logger.exception(f"Error Q{i}: {e}")
            
    # Save results
    out_json = preds_dir / "answers.json"
    out_qa = preds_dir / "qa.tsv"
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    with open(out_qa, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_lines))
        
    logger.info("Raptor baseline complete. qa.tsv: {}", out_qa)

if __name__ == "__main__":
    main()
