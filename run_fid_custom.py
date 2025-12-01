#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.fid_rag import FiDRAGRunner
from config import config as global_config
from utils import setup_logging

def _select_workspace(root: Path, prefix: str, force_new: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing: List[Path] = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if force_new or not existing:
        next_idx = len(existing)
        target = root / f"{prefix}_{next_idx:03d}"
        target.mkdir(parents=True, exist_ok=True)
        return target
    return existing[-1]

def _load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("data") or data.get("examples") or data.get("questions") or []
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list or a dict containing list under data/examples/questions")
    return data

def main() -> None:
    parser = argparse.ArgumentParser(description="Custom FID Runner with configurable max tokens")
    parser.add_argument("--dataset-path", default="data/mirage_sample/dataset.json")
    parser.add_argument("--index-dir", default="result/mirage_naive")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--chunks-path", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--lmstudio-endpoint", default=None)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--gpu-device", type=str, default="1", help="CUDA_VISIBLE_DEVICES value")
    
    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    
    cfg = global_config.load_config()
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    dataset = _load_dataset(dataset_path)
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    result_root = Path(args.result_root)
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = _select_workspace(result_root, "mirage_fid_custom", args.new)
    
    setup_logging(str(work_dir / "fid_custom.log"))
    logger.info("Running FID custom workspace={} with max_tokens={}", work_dir, args.max_new_tokens)

    index_dir = Path(args.index_dir)
    index_path = Path(args.index_path) if args.index_path else index_dir / "index.faiss"
    chunks_path = Path(args.chunks_path) if args.chunks_path else index_dir / "chunks.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")

    lm_endpoint = args.lmstudio_endpoint or cfg.get("lmstudio.endpoint")
    lm_model = args.lmstudio_model or cfg.get("lmstudio.model")

    runner = FiDRAGRunner(
        str(index_path),
        str(chunks_path),
        topk=args.topk or 5,
        lm_endpoint=lm_endpoint,
        lm_model=lm_model,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    
    artifacts = runner.run_dataset(dataset, work_dir=str(work_dir))
    logger.info("FID RAG finished. qa.tsv at {}", artifacts.get("qa"))

if __name__ == "__main__":
    main()
