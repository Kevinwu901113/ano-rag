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

from baselines.simple_graphrag import answer as simple_graphrag_answer


def _select_workspace(root: Path, prefix: str, force_new: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing: List[Path] = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if force_new or not existing:
        next_idx = len(existing)
        target = root / f"{prefix}_{next_idx:03d}"
        target.mkdir(parents=True, exist_ok=True)
        return target
    return existing[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Simple GraphRAG baseline on MIRAGE dataset.json")
    parser.add_argument("--dataset-path", default="data/mirage_sample/dataset.json")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--work-dir", default=None, help="Where to write outputs. Default: auto under result_root")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--lmstudio-endpoint", required=True)
    parser.add_argument("--lmstudio-model", required=True)
    args = parser.parse_args()

    # 1. Setup config for the baseline (it uses global config)
    from config.config_loader import config as global_config
    global_config.setdefault("lmstudio", {})
    global_config["lmstudio"]["endpoint"] = args.lmstudio_endpoint
    global_config["lmstudio"]["model"] = args.lmstudio_model
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = _select_workspace(Path(args.result_root), "mirage_simple_graphrag", args.new)
    logger.info("Writing Simple GraphRAG outputs to {}", work_dir)

    # 2. Run inference
    if args.limit > 0:
        dataset = dataset[:args.limit]
        
    results = []
    qa_lines = []
    
    for i, item in enumerate(dataset):
        question = item.get("query") or item.get("question")
        qid = item.get("query_id") or str(i)
        
        try:
            logger.info(f"Processing Q{i}: {question}")
            ans = simple_graphrag_answer(question)
            
            results.append({
                "query_id": qid,
                "question": question,
                "answer": ans
            })
            
            clean_ans = " ".join(ans.split())
            qa_lines.append(f"{question}\t{clean_ans}")
            
        except Exception as e:
            logger.error(f"Error processing Q{i}: {e}")
            results.append({
                "query_id": qid,
                "question": question,
                "answer": "Error",
                "error": str(e)
            })

    # 3. Save results
    output_path = work_dir / "answers.json"
    qa_log_path = work_dir / "qa.tsv"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    with open(qa_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_lines))
        
    logger.info(f"Finished. Results saved to {work_dir}")

if __name__ == "__main__":
    main()
