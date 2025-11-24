#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from baselines.direct_llm import DirectLLMRunner
from baselines.naive_rag import NaiveRAGRunner
from config import config as global_config
from structrag import StructRAGBaselineRunner
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
    parser = argparse.ArgumentParser(description="MIRAGE/MuSiQue runner with StructRAG baseline")
    parser.add_argument("--mode", choices=["structrag_baseline", "naive_rag", "direct_llm"], default="structrag_baseline")
    parser.add_argument("--dataset-path", default="data/mirage_sample/dataset.json")
    parser.add_argument("--doc-pool", default="data/mirage_sample/doc_pool.json")
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
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature override for naive/direct baselines")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional max tokens override for naive/direct baselines")
    args = parser.parse_args()

    cfg = global_config.load_config()
    dataset_path = Path(args.dataset_path)
    doc_pool_path = Path(args.doc_pool)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not doc_pool_path.exists():
        raise FileNotFoundError(f"Doc pool not found: {doc_pool_path}")
    dataset = _load_dataset(dataset_path)
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    result_root = Path(args.result_root)
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = _select_workspace(result_root, f"mirage_{args.mode}", args.new)
    setup_logging(str(work_dir / f"{args.mode}.log"))
    logger.info("Running mode={} workspace={}", args.mode, work_dir)

    index_dir = Path(args.index_dir)
    index_path = Path(args.index_path) if args.index_path else index_dir / "index.faiss"
    chunks_path = Path(args.chunks_path) if args.chunks_path else index_dir / "chunks.jsonl"

    lm_endpoint = args.lmstudio_endpoint or cfg.get("lmstudio.endpoint")
    lm_model = args.lmstudio_model or cfg.get("lmstudio.model")

    if args.mode == "structrag_baseline":
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")
        runner = StructRAGBaselineRunner(
            index_path=str(index_path),
            chunks_path=str(chunks_path),
            doc_pool_path=str(doc_pool_path),
            config=cfg,
            lm_endpoint=lm_endpoint,
            lm_model=lm_model,
            top_k=args.topk,
        )
        artifacts = runner.run_dataset(dataset, work_dir=str(work_dir))
        logger.info("StructRAG baseline outputs: {}", artifacts.get("answers"))
        return

    if args.mode == "naive_rag":
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")
        runner = NaiveRAGRunner(
            str(index_path),
            str(chunks_path),
            topk=args.topk or 5,
            lm_endpoint=lm_endpoint,
            lm_model=lm_model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        artifacts = runner.run_dataset(dataset, work_dir=str(work_dir))
        logger.info("Naive RAG finished. qa.tsv at {}", artifacts.get("qa"))
        return

    if args.mode == "direct_llm":
        runner = DirectLLMRunner(
            lm_endpoint=lm_endpoint,
            lm_model=lm_model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        artifacts = runner.run_dataset(dataset, work_dir=str(work_dir))
        logger.info("Direct LLM finished. answers at {}", artifacts.get("answers_jsonl"))


if __name__ == "__main__":
    main()
