import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from config.config_loader import config as global_config
from baselines.vanilla_rag import answer as vanilla_rag_answer

def _select_workspace(args: argparse.Namespace, base_dir: str = "result") -> Path:
    if args.work_dir:
        p = Path(args.work_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    # Generate a new timestamped directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"mirage_vanilla_rag_{ts}"
    p = Path(base_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    parser = argparse.ArgumentParser(description="Run Vanilla RAG Baseline on MIRAGE")
    parser.add_argument("--dataset-path", type=str, default="data/mirage/mirage_dataset.json", help="Path to MIRAGE dataset")
    parser.add_argument("--result-root", type=str, default="result", help="Root directory for results")
    parser.add_argument("--work-dir", type=str, help="Specific working directory (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries")
    parser.add_argument("--lmstudio-endpoint", type=str, help="LM Studio endpoint override")
    parser.add_argument("--lmstudio-model", type=str, help="LM Studio model name override")
    parser.add_argument("--embedding-endpoint", type=str, help="Embedding endpoint override")
    parser.add_argument("--index-path", type=str, default="indexes/vanilla_rag_index.faiss", help="Path to FAISS index")
    parser.add_argument("--chunk-store-path", type=str, default="indexes/vanilla_rag_chunk_store.pkl", help="Path to chunk store")
    
    args = parser.parse_args()

    # 0. Setup Config Overrides
    if args.lmstudio_endpoint:
        global_config.setdefault("lmstudio", {})["endpoint"] = args.lmstudio_endpoint
    if args.lmstudio_model:
        global_config.setdefault("lmstudio", {})["model"] = args.lmstudio_model
    if args.embedding_endpoint:
        # Assuming embedding config structure, adjust as needed
        # Typically under retriever.embedding or just embedding
        pass

    # 1. Setup Workspace
    work_dir = _select_workspace(args, args.result_root)
    logger.add(work_dir / "run.log")
    logger.info(f"Starting Vanilla RAG run in {work_dir}")

    # 2. Load Dataset
    try:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}: {e}")
        return

    if args.limit > 0:
        dataset = dataset[:args.limit]
        
    results = []
    qa_lines = []
    
    # Helper for cleaning answers (reusing simple_graphrag style logic if available or implementing simple one)
    # For Vanilla RAG, we might want to stick to what the model outputs, but let's do basic stripping
    def clean_answer(ans: str) -> str:
        ans = ans.strip()
        # Remove potential "Answer:" prefix if model chatted
        if ans.lower().startswith("answer:"):
            ans = ans[7:].strip()
        return ans

    # 3. Run Inference
    for i, item in enumerate(dataset):
        question = item.get("query") or item.get("question")
        qid = item.get("query_id") or str(i)
        
        try:
            logger.info(f"Processing Q{i}: {question}")
            ans = vanilla_rag_answer(
                question, 
                index_path=args.index_path, 
                chunk_store_path=args.chunk_store_path
            )
            final_ans = clean_answer(ans)
            
            results.append({
                "query_id": qid,
                "question": question,
                "answer": final_ans,
                "raw_answer": ans
            })
            
            clean_ans_line = " ".join(final_ans.split())
            qa_lines.append(f"{question}\t{clean_ans_line}")
            
        except Exception as e:
            logger.error(f"Error processing Q{i}: {e}")
            results.append({
                "query_id": qid,
                "question": question,
                "answer": "Error",
                "error": str(e)
            })

    # 4. Save Results
    output_path = work_dir / "answers.json"
    qa_log_path = work_dir / "qa.tsv"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    with open(qa_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_lines))
        
    logger.info(f"Finished. Results saved to {work_dir}")

if __name__ == "__main__":
    main()
