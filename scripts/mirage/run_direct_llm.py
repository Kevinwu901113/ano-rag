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
from utils.run_layout import ensure_workdir_layout, resolve_workdir


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
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Where to write outputs. Default: auto under result_root")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace (do not reuse latest)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--lmstudio-endpoint", default=None)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-system-prompt", action="store_true", help="Skip the default direct-LLM system prompt")
    args = parser.parse_args()

    # 1. Setup config for the baseline (it uses global config)
    from config.config_loader import config as global_config
    # global_config is a ConfigLoader instance, not a dict.
    # It has a .set(key, value) method.
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
    ensure_workdir_layout(work_dir)
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
