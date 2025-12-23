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

from baselines.naive_rag import NaiveRAGRunner
from utils.logging_utils import setup_logging
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


def _resolve_index_artifacts(index_dir: Path) -> tuple[Path, Path]:
    meta_path = index_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            index_path = Path(str(meta.get("index") or "")).expanduser()
            chunks_path = Path(str(meta.get("chunks") or "")).expanduser()
            return index_path, chunks_path
        except Exception:
            pass
    return index_dir / "index.faiss", index_dir / "chunks.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run naive RAG baseline on MIRAGE dataset.json")
    parser.add_argument("--dataset-path", default="data/mirage_sample/dataset.json")
    parser.add_argument("--index-dir", default=None, help="Directory containing index.faiss + chunks.jsonl")
    parser.add_argument("--index-path", default=None, help="Optional explicit FAISS index path")
    parser.add_argument("--chunks-path", default=None, help="Optional explicit chunks.jsonl path")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Where to write outputs. Default: auto under result_root")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace (do not reuse latest)")
    parser.add_argument("--lm-endpoint", default=None)
    parser.add_argument("--lm-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--embed-model", default=None, help="Embedding model override for query encoding")
    parser.add_argument("--embed-provider", default=None, help="Embedding provider override (e.g., st, qwen3, mock)")
    parser.add_argument("--no-debug", action="store_true", help="Skip writing retrieval debug JSONL")
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs in workdir")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint every N samples (0 disables)")
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    args = parser.parse_args()
    
    # 1. Setup config for the baseline (it uses global config)
    from config.config_loader import config as global_config
    if args.lm_endpoint:
        global_config.set("vllm.endpoint", args.lm_endpoint)
    if args.lm_model:
        global_config.set("vllm.model", args.lm_model)
    if args.embed_model:
        global_config.set("retriever.embedding.model", args.embed_model)
    if args.embed_provider:
        global_config.set("retriever.embedding.provider", args.embed_provider)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="mirage")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    index_dir = Path(args.index_dir) if args.index_dir else artifacts_dir / "naive_index"
    if args.index_path or args.chunks_path:
        index_path = Path(args.index_path) if args.index_path else index_dir / "index.faiss"
        chunks_path = Path(args.chunks_path) if args.chunks_path else index_dir / "chunks.jsonl"
    else:
        index_path, chunks_path = _resolve_index_artifacts(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"index.faiss missing: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl missing: {chunks_path}")

    run_name = work_dir.name
    logger.info("Writing outputs to {}", work_dir)
    setup_logging(str(work_dir / "run.log"))
    cfg_snapshot = global_config.load_config()
    lm_endpoint = args.lm_endpoint or cfg_snapshot.get("vllm", {}).get("endpoint")
    lm_model = args.lm_model or cfg_snapshot.get("vllm", {}).get("model")
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="mirage",
            model=lm_model or "unknown",
            endpoint=lm_endpoint or "unknown",
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            context_budget=args.context_budget or None,
            topk=args.topk,
        ),
    )

    runner = NaiveRAGRunner(
        str(index_path),
        str(chunks_path),
        topk=args.topk,
        lm_endpoint=args.lm_endpoint,
        lm_model=args.lm_model,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        context_budget=args.context_budget,
    )
    limit = args.limit if args.limit and args.limit > 0 else None
    artifacts = runner.run_dataset(
        dataset,
        work_dir=str(work_dir),
        limit=limit,
        debug=not args.no_debug,
        dataset_name="mirage",
        run_name=run_name,
        resume=args.resume,
        save_every=args.save_every,
    )
    logger.info("Naive RAG finished. qa.tsv: {}", artifacts.get("qa"))


if __name__ == "__main__":
    main()
