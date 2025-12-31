#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config_loader import config as global_config
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved

def _select_workspace(root: Path, prefix: str, force_new: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing: List[Path] = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if force_new or not existing:
        next_idx = len(existing)
        target = root / f"{prefix}_{next_idx:03d}"
        target.mkdir(parents=True, exist_ok=True)
        return target
    return existing[-1]

def main():
    parser = argparse.ArgumentParser(description="Run Simple Self-RAG Baseline on MIRAGE")
    parser.add_argument("--dataset-path", type=str, default="data/mirage/mirage_dataset.json", help="Path to MIRAGE dataset")
    parser.add_argument("--result-root", type=str, default="result_relrag", help="Root directory for results")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", type=str, help="Specific working directory (optional)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries")
    parser.add_argument("--lm-endpoint", type=str, help="LLM endpoint override")
    parser.add_argument("--lm-model", type=str, help="LLM model name override")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls; only run retrieval and log retrieval.jsonl")
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    
    args = parser.parse_args()

    # Override config if provided
    if args.lm_endpoint:
        global_config.set("vllm.endpoint", args.lm_endpoint)
    if args.lm_model:
        global_config.set("vllm.model", args.lm_model)

    # Setup workspace
    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="mirage")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    run_name = work_dir.name
    dataset_name = "mirage"
        
    setup_logging(str(work_dir / "run.log"))
    logger.info(f"Writing Simple Self-RAG outputs to {work_dir}")
    cfg_snapshot = global_config.load_config()
    lm_endpoint = args.lm_endpoint or cfg_snapshot.get("vllm", {}).get("endpoint")
    lm_model = args.lm_model or cfg_snapshot.get("vllm", {}).get("model")
    embedding_config = global_config.get("retriever", {}).get("simple_selfrag", {}).get("embedding") or global_config.get("retriever", {}).get("embedding")
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="mirage",
            model=lm_model or "unknown",
            endpoint=lm_endpoint or "unknown",
            temperature=None,
            max_tokens=None,
            context_budget=args.context_budget or None,
            embedding={
                "model": embedding_config.get("model") if embedding_config else "sentence-transformers/all-MiniLM-L6-v2",
                "device": embedding_config.get("device") if embedding_config else "cpu",
            },
            extra={
                "embedding_model_name": embedding_config.get("model") if embedding_config else "sentence-transformers/all-MiniLM-L6-v2",
                "max_rounds": 2, # Hardcoded in logic as 1st retrieval + optional 2nd
            }
        ),
    )

    # 2. Determine Index Paths
    index_path = artifacts_dir / "simple_selfrag_index.faiss"
    chunk_store_path = artifacts_dir / "simple_selfrag_chunk_store.pkl"
    
    # 3. Check/Build Index
    if not index_path.exists() or not chunk_store_path.exists():
        logger.info(f"Index not found at {index_path}. Attempting to build...")
        
        # Try to find doc_pool
        dataset_path = Path(args.dataset_path)
        doc_pool_path = dataset_path.parent / "doc_pool.json"
        
        if not doc_pool_path.exists():
             logger.error(f"Cannot build index: doc_pool.json not found at {doc_pool_path}")
             return

        logger.info(f"Building index from {doc_pool_path}...")
        
        # Build index directly using SimpleSelfRAGIndexer
        from baselines.simple_selfrag.index import SimpleSelfRAGIndexer
        # No need to import Retriever here for building index
        
        # Load docs
        with open(doc_pool_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        docs = {}
        if isinstance(raw_data, list):
            for i, item in enumerate(raw_data):
                 base_id = item.get("doc_id") or item.get("mapped_id") or str(i)
                 doc_id = f"{base_id}::{i}"
                 text = item.get("doc_chunk") or item.get("text") or item.get("content") or ""
                 
                 # Prepend title if available (consistent with Vanilla RAG)
                 title = item.get("doc_name") or item.get("title")
                 if title:
                     text = f"{title}\n{text}"
                     
                 docs[doc_id] = text
        elif isinstance(raw_data, dict):
            for k, v in raw_data.items():
                if isinstance(v, str):
                    docs[k] = v
                elif isinstance(v, dict):
                     text = v.get("text") or v.get("content") or ""
                     # Prepend title if available
                     title = v.get("title") or v.get("doc_name")
                     if title:
                         text = f"{title}\n{text}"
                     docs[k] = text
                     
        # Initialize indexer and build
        embedding_config = global_config.get("retriever", {}).get("simple_selfrag", {}).get("embedding") or global_config.get("retriever", {}).get("embedding")
        indexer = SimpleSelfRAGIndexer(embedding_config)
        stats = indexer.build(docs)
        logger.info(f"Build complete. Stats: {stats}")
        
        # Save index
        indexer.save(str(index_path), str(chunk_store_path))
        
    # Initialize retriever with specific paths
    from baselines.simple_selfrag.retriever import SimpleSelfRAGRetriever
    retriever = SimpleSelfRAGRetriever(
        str(index_path),
        str(chunk_store_path),
        context_budget=args.context_budget,
    )
    
    # Disable LLM if retrieval-only
    if args.retrieval_only:
         retriever.llm_client = None

    # Load dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if isinstance(dataset, dict):
        # Maybe wrapped in {"questions": [...]}
        dataset = dataset.get("questions", dataset.get("data", []))
        
    if args.limit > 0:
        dataset = dataset[:args.limit]
        
    logger.info(f"Loaded {len(dataset)} questions")

    # Run processing
    answers = []
    pred_raw_records: List[Dict[str, Any]] = []
    
    for item in tqdm(dataset, desc="Processing"):
        qid = item.get("id") or item.get("question_id") or item.get("query_id")
        question = item.get("question") or item.get("text") or item.get("query")
        
        if not question:
            continue
            
        try:
            # Callback for logging intermediate retrieval steps
            def retrieval_callback(step, hits):
                round_dir = artifacts_dir / f"round_{step}"
                round_dir.mkdir(parents=True, exist_ok=True)
                try:
                    log_retrieval(
                        sample_id=qid,
                        dataset=dataset_name,
                        run_name=run_name,
                        retrieved=[{**hit, "rank": i + 1} for i, hit in enumerate(hits)],
                        topk=len(hits),
                        log_dir=round_dir,
                    )
                except Exception as log_exc:
                    logger.error(f"retrieval logging failed for {qid} round {step}: {log_exc}")

            # Use the local retriever instance instead of the global singleton
            if args.retrieval_only:
                hits = retriever.retrieve(question, top_k=5)
                retrieval_callback(1, hits)
                ans_text = ""
            else:
                ans_text = retriever.answer(question, log_callback=retrieval_callback)
                hits = getattr(retriever, "last_hits", [])

            final_ans = ans_text.strip()
            
            answers.append({
                "question_id": qid,
                "question": question,
                "answer": final_ans,
                "raw_answer": ans_text,
                "gold_answer": item.get("answer")
            })
            annotated_hits = []
            for i, hit in enumerate(hits):
                annotated_hits.append({**hit, "text": f"[{i+1}] {hit.get('text', '')}"})
            context_str, contexts_used, context_tokens = pack_contexts(
                annotated_hits, int(args.context_budget or 0)
            )
            try:
                log_retrieval(
                    sample_id=qid,
                    dataset=dataset_name,
                    run_name=run_name,
                    retrieved=[{**hit, "rank": i + 1} for i, hit in enumerate(hits)],
                    topk=len(hits),
                    final_context=contexts_used,
                    final_context_tokens=context_tokens,
                    context_budget_tokens=int(args.context_budget or 0) or None,
                    log_dir=artifacts_dir,
                )
            except Exception as log_exc:
                logger.error(f"retrieval logging failed for {qid}: {log_exc}")
            pred_raw_records.append(
                {
                    "id": str(qid),
                    "question": question,
                    "pred_raw": ans_text,
                    "contexts_used": contexts_used,
                    "context_tokens_used": context_tokens,
                    "context_budget_tokens": int(args.context_budget or 0) or None,
                }
            )
            
        except Exception as e:
            logger.exception(f"Error processing question {qid}: {e}")
            answers.append({
                "question_id": qid,
                "question": question,
                "answer": "Error",
                "error": str(e)
            })

    # Save results
    output_path = preds_dir / "answers.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
        
    # Save QA TSV
    qa_path = preds_dir / "qa.tsv"
    with qa_path.open("w", encoding="utf-8") as f:
        for item in answers:
            q = item["question"].replace("\t", " ").strip()
            a = item["answer"].replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{q}\t{a}\n")
    write_jsonl(preds_dir / "pred_raw.jsonl", pred_raw_records)
            
    logger.info(f"Done. Saved {len(answers)} answers to {work_dir}")

if __name__ == "__main__":
    main()
