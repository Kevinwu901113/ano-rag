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
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (ignored for direct LLM)")
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
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    run_name = work_dir.name
    logger.info("Writing direct LLM outputs to {}", work_dir)
    setup_logging(str(work_dir / "run.log"))
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="mirage",
            model=args.lmstudio_model or "unknown",
            endpoint=args.lmstudio_endpoint or "unknown",
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            context_budget=args.context_budget or None,
            decode={
                "temperature": args.temperature,
                "top_p": None,
                "repetition_penalty": None,
                "max_tokens": args.max_new_tokens,
            },
            embedding={"model": None},
            budgets={"context_budget_tokens": args.context_budget or None},
        ),
    )

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
    artifacts = runner.run_dataset(
        dataset,
        work_dir=str(work_dir),
        limit=limit,
        context_budget_tokens=args.context_budget or 0,
        log_dir=str(artifacts_dir),
        run_name=run_name,
        dataset_name="mirage",
    )
    logger.info("Direct LLM baseline finished. qa.tsv: {}", artifacts.get("qa"))


if __name__ == "__main__":
    main()
