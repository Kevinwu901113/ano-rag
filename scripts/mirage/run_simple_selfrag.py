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
from utils.retrieval_logger import log_retrieval

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
    parser.add_argument("--result-root", type=str, default="result", help="Root directory for results")
    parser.add_argument("--work-dir", type=str, help="Specific working directory (optional)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries")
    parser.add_argument("--lmstudio-endpoint", type=str, help="LM Studio endpoint override")
    parser.add_argument("--lmstudio-model", type=str, help="LM Studio model name override")
    
    args = parser.parse_args()

    # Override config if provided
    if args.lmstudio_endpoint:
        global_config.set("lmstudio.endpoint", args.lmstudio_endpoint)
    if args.lmstudio_model:
        global_config.set("lmstudio.model", args.lmstudio_model)

    # Setup workspace
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = _select_workspace(Path(args.result_root), "mirage_simple_selfrag", args.new)
    run_name = work_dir.name
    dataset_name = "mirage"
        
    logger.add(work_dir / "simple_selfrag.log")
    logger.info(f"Writing Simple Self-RAG outputs to {work_dir}")

    # 2. Determine Index Paths
    index_path = work_dir / "simple_selfrag_index.faiss"
    chunk_store_path = work_dir / "simple_selfrag_chunk_store.pkl"
    
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
    retriever = SimpleSelfRAGRetriever(str(index_path), str(chunk_store_path))

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
    
    # Import answer cleaner
    try:
        from utils.answer_cleaner import clean_model_answer
    except ImportError:
        def clean_model_answer(x): return x.strip()
    
    for item in tqdm(dataset, desc="Processing"):
        qid = item.get("id") or item.get("question_id") or item.get("query_id")
        question = item.get("question") or item.get("text") or item.get("query")
        
        if not question:
            continue
            
        try:
            # Use the local retriever instance instead of the global singleton
            ans_text = retriever.answer(question)
            final_ans = clean_model_answer(ans_text)
            
            # Ensure we have some answer, even if cleaning stripped it
            if not final_ans and ans_text:
                final_ans = ans_text.strip()
            
            answers.append({
                "question_id": qid,
                "question": question,
                "answer": final_ans,
                "raw_answer": ans_text,
                "gold_answer": item.get("answer") # Preserve gold if available
            })
            try:
                hits = getattr(retriever, "last_hits", [])
                log_retrieval(
                    sample_id=qid,
                    dataset=dataset_name,
                    run_name=run_name,
                    retrieved=[{**hit, "rank": i + 1} for i, hit in enumerate(hits)],
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
                    log_dir=work_dir,
                )
            except Exception as log_exc:
                logger.error(f"retrieval logging failed for {qid}: {log_exc}")
            
        except Exception as e:
            logger.exception(f"Error processing question {qid}: {e}")
            answers.append({
                "question_id": qid,
                "question": question,
                "answer": "Error",
                "error": str(e)
            })

    # Save results
    output_path = work_dir / "answers.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
        
    # Save QA TSV
    qa_path = work_dir / "qa.tsv"
    with qa_path.open("w", encoding="utf-8") as f:
        for item in answers:
            q = item["question"].replace("\t", " ").strip()
            a = item["answer"].replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{q}\t{a}\n")
            
    logger.info(f"Done. Saved {len(answers)} answers to {work_dir}")

if __name__ == "__main__":
    main()
