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

from baselines.direct_llm import DirectLLMRunner


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
    parser = argparse.ArgumentParser(description="Run direct LLM baseline on MIRAGE dataset.json")
    parser.add_argument("--dataset-path", default="data/mirage_sample/dataset.json")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--work-dir", default=None, help="Where to write outputs. Default: auto under result_root")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace (do not reuse latest)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--lmstudio-endpoint", default=None)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-system-prompt", action="store_true", help="Skip the default direct-LLM system prompt")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = _select_workspace(Path(args.result_root), "mirage_direct_llm", args.new)
    logger.info("Writing direct LLM outputs to {}", work_dir)

    runner_kwargs = {
        "lm_endpoint": args.lmstudio_endpoint,
        "lm_model": args.lmstudio_model,
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
    }
    if args.no_system_prompt:
        runner_kwargs["system_prompt"] = ""

    runner = DirectLLMRunner(**runner_kwargs)
    limit = args.limit if args.limit and args.limit > 0 else None
    artifacts = runner.run_dataset(dataset, work_dir=str(work_dir), limit=limit)
    logger.info("Direct LLM baseline finished. qa.tsv: {}", artifacts.get("qa"))


if __name__ == "__main__":
    main()
