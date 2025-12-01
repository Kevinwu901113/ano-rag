import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now we can import from project modules
try:
    from config.config_loader import config as global_config
    from utils.answer_cleaner import clean_model_answer
except ImportError:
    # Fallback if running as script without package context setup (though sys.path fix should handle it)
    # Try to mock or load manually if needed, but sys.path should work.
    logger.warning("Could not import config.config_loader or utils.answer_cleaner, ensuring PYTHONPATH is set correctly.")
    pass

from baselines.vanilla_rag import answer as vanilla_rag_answer
from baselines.vanilla_rag.index import VanillaRAGIndexer

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
    parser = argparse.ArgumentParser(description="Run Vanilla RAG Baseline on MIRAGE")
    parser.add_argument("--dataset-path", type=str, default="data/mirage/mirage_dataset.json", help="Path to MIRAGE dataset")
    parser.add_argument("--result-root", type=str, default="result", help="Root directory for results")
    parser.add_argument("--work-dir", type=str, help="Specific working directory (optional)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries")
    parser.add_argument("--lmstudio-endpoint", type=str, help="LM Studio endpoint override")
    parser.add_argument("--lmstudio-model", type=str, help="LM Studio model name override")
    parser.add_argument("--index-path", type=str, help="Path to FAISS index (optional, default to work_dir/vanilla_rag_index.faiss)")
    parser.add_argument("--chunk-store-path", type=str, help="Path to chunk store (optional, default to work_dir/vanilla_rag_chunk_store.pkl)")
    
    args = parser.parse_args()

    # 0. Setup Config Overrides
    if args.lmstudio_endpoint:
        global_config.setdefault("lmstudio", {})["endpoint"] = args.lmstudio_endpoint
    if args.lmstudio_model:
        global_config.setdefault("lmstudio", {})["model"] = args.lmstudio_model

    # 1. Setup Workspace
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = _select_workspace(Path(args.result_root), "mirage_vanilla_rag", args.new)
    
    logger.add(work_dir / "vanilla_rag.log")
    logger.info(f"Starting Vanilla RAG run in {work_dir}")

    # 2. Determine Index Paths
    index_path = Path(args.index_path) if args.index_path else work_dir / "vanilla_rag_index.faiss"
    chunk_store_path = Path(args.chunk_store_path) if args.chunk_store_path else work_dir / "vanilla_rag_chunk_store.pkl"

    # 3. Check/Build Index
    if not index_path.exists() or not chunk_store_path.exists():
        logger.info(f"Index not found at {index_path}. Attempting to build...")
        
        # Try to find doc_pool
        dataset_path = Path(args.dataset_path)
        doc_pool_path = dataset_path.parent / "doc_pool.json"
        
        if not doc_pool_path.exists():
             # Fallback to checking if dataset itself has documents or another location
             logger.error(f"Cannot build index: doc_pool.json not found at {doc_pool_path}")
             return

        logger.info(f"Building index from {doc_pool_path}...")
        
        # Load docs
        docs = {}
        try:
            with open(doc_pool_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                if isinstance(raw_data, list):
                    for i, item in enumerate(raw_data):
                         # MIRAGE doc pool format
                         doc_id = item.get("doc_id") or item.get("mapped_id") or str(i)
                         text = item.get("doc_chunk") or item.get("text") or item.get("content") or ""
                         title = item.get("doc_name", "")
                         if title:
                             text = f"{title}\n{text}"
                         docs[doc_id] = text
                elif isinstance(raw_data, dict):
                    for k, v in raw_data.items():
                        if isinstance(v, str):
                            docs[k] = v
                        elif isinstance(v, dict):
                             docs[k] = v.get("text") or v.get("content") or ""
        except Exception as e:
            logger.error(f"Failed to load doc pool: {e}")
            return
            
        if not docs:
             logger.error("No documents found to index.")
             return

        # Build
        indexer = VanillaRAGIndexer()
        indexer.build(docs, str(index_path), str(chunk_store_path))
        logger.info("Index built successfully.")

    # 4. Load Dataset
    try:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}: {e}")
        return

    if args.limit > 0:
        dataset = dataset[:args.limit]
        
    results = []
    
    # 5. Run Inference
    for i, item in enumerate(dataset):
        question = item.get("query") or item.get("question")
        qid = item.get("query_id") or str(i)
        
        try:
            logger.info(f"Processing Q{i}: {question}")
            ans = vanilla_rag_answer(
                question, 
                index_path=str(index_path), 
                chunk_store_path=str(chunk_store_path)
            )
            final_ans = clean_model_answer(ans)
            
            results.append({
                "query_id": qid,
                "question": question,
                "answer": final_ans,
                "raw_answer": ans
            })
            
        except Exception as e:
            logger.error(f"Error processing Q{i}: {e}")
            results.append({
                "query_id": qid,
                "question": question,
                "answer": "Error",
                "error": str(e)
            })

    # 6. Save Results
    # Save detailed JSONL
    out_jsonl = work_dir / "results.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    # Save QA TSV (compatible with eval scripts)
    out_tsv = work_dir / "qa.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        for res in results:
            # Format: query_text\tmodel_answer
            q_text = res["question"].replace("\t", " ").strip()
            a_text = res["answer"].replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{q_text}\t{a_text}\n")

    logger.info(f"Finished. Results saved to {work_dir}")

if __name__ == "__main__":
    main()
