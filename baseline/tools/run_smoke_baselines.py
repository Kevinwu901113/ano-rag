#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence

METHOD_TO_RUNNER = {
    "lightrag": "baseline/runners/run_lightrag_qa.py",
    "graphrag": "baseline/runners/run_graphrag_qa.py",
    "raptor": "baseline/runners/run_raptor_qa.py",
    "bm25": "baseline/runners/run_bm25_qa.py",
    "dense": "baseline/runners/run_dense_qa.py",
}

DEFAULT_METHOD_ENVS = {
    "lightrag": "baseline-lightrag",
    "graphrag": "baseline-graphrag",
    "raptor": "baseline-raptor",
    "bm25": "baseline-lightrag",
    "dense": "baseline-lightrag",
}


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _method_extra_args(method: str) -> List[str]:
    # Smoke-only speed knobs; does not change runner defaults.
    if method == "lightrag":
        return [
            "--chunk_token_size",
            "2000",
            "--chunk_overlap_token_size",
            "0",
            "--extract_max_tokens",
            "1024",
            "--query_mode",
            "naive",
        ]
    if method == "graphrag":
        return ["--index_method", "fast"]
    if method == "raptor":
        return [
            "--tb_num_layers",
            "2",
            "--tb_max_tokens",
            "80",
            "--tb_summarization_length",
            "80",
            "--summarizer_max_input_chars",
            "12000",
            "--qa_max_input_chars",
            "12000",
        ]
    return []


def _run_one(
    *,
    method: str,
    dataset: str,
    backend: str,
    env_name: str,
    args: argparse.Namespace,
    deepseek_key: str,
    logs_dir: Path,
) -> Dict[str, object]:
    runner = METHOD_TO_RUNNER[method]
    pred_path = args.output_root / method / dataset / backend / "pred.jsonl"
    if pred_path.exists():
        pred_path.unlink()

    cmd: List[str] = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        runner,
        "--dataset",
        dataset,
        "--llm_backend",
        backend,
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.output_root),
        "--workspace_root",
        str(args.workspace_root),
        "--limit",
        str(args.limit),
        "--top_k",
        str(args.top_k),
        "--answer_max_tokens",
        str(args.answer_max_tokens),
        "--qa_prompt_mode",
        args.qa_prompt_mode,
        "--request_timeout",
        str(args.request_timeout),
    ]
    if args.rebuild_index:
        cmd.append("--rebuild_index")
    cmd.extend(_method_extra_args(method))

    run_env = os.environ.copy()
    if backend == "deepseek":
        run_env["OPENAI_API_KEY"] = deepseek_key

    log_path = logs_dir / f"{method}__{dataset}__{backend}.log"
    start = time.perf_counter()
    timed_out = False
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=run_env,
            check=False,
            timeout=max(1, int(args.case_timeout_sec)),
        )
        rc = int(cp.returncode)
        out_text = cp.stdout or ""
        err_text = cp.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        out_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        err_text = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    duration_s = round(time.perf_counter() - start, 3)

    log_path.write_text(
        "\n".join(
            [
                f"# command\n{' '.join(cmd)}",
                f"\n# returncode\n{rc}",
                f"\n# timed_out\n{timed_out}",
                "\n# stdout\n" + out_text,
                "\n# stderr\n" + err_text,
            ]
        ),
        encoding="utf-8",
    )

    ok = (rc == 0) and pred_path.exists()
    return {
        "method": method,
        "dataset": dataset,
        "backend": backend,
        "env": env_name,
        "rc": rc,
        "timed_out": int(timed_out),
        "ok": bool(ok),
        "duration_s": duration_s,
        "pred_path": str(pred_path),
        "pred_exists": int(pred_path.exists()),
        "log_path": str(log_path),
        "command": " ".join(cmd),
    }


def _write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "dataset",
        "backend",
        "env",
        "rc",
        "timed_out",
        "ok",
        "duration_s",
        "pred_exists",
        "pred_path",
        "log_path",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke matrix for all baseline runners")
    parser.add_argument("--methods", default="lightrag,graphrag,raptor,bm25,dense")
    parser.add_argument("--datasets", default="hotpotqa,musique,2wiki")
    parser.add_argument("--backends", default="qwen,deepseek")

    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--answer_max_tokens", type=int, default=96)
    parser.add_argument("--qa_prompt_mode", default="answer_only", choices=["answer_only", "default"])
    parser.add_argument("--request_timeout", type=float, default=60.0)

    parser.add_argument("--rebuild_index", action="store_true", default=True)
    parser.add_argument("--no_rebuild_index", action="store_false", dest="rebuild_index")

    parser.add_argument("--data_root", type=Path, default=Path("baseline/data"))
    parser.add_argument("--output_root", type=Path, default=Path("baseline/results"))
    parser.add_argument("--workspace_root", type=Path, default=Path("baseline/workspaces"))
    parser.add_argument("--summary_tsv", type=Path, default=Path("baseline/results/smoke_summary.tsv"))
    parser.add_argument("--logs_dir", type=Path, default=Path("baseline/results/smoke_logs"))
    parser.add_argument("--case_timeout_sec", type=int, default=900)

    parser.add_argument("--env_lightrag", default=DEFAULT_METHOD_ENVS["lightrag"])
    parser.add_argument("--env_graphrag", default=DEFAULT_METHOD_ENVS["graphrag"])
    parser.add_argument("--env_raptor", default=DEFAULT_METHOD_ENVS["raptor"])
    parser.add_argument("--env_bm25", default=DEFAULT_METHOD_ENVS["bm25"])
    parser.add_argument("--env_dense", default=DEFAULT_METHOD_ENVS["dense"])

    parser.add_argument(
        "--deepseek_api_key",
        default=(os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip(),
        help="Optional override. If empty, fallback to env OPENAI_API_KEY/DEEPSEEK_API_KEY.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    methods = _split_csv(args.methods)
    datasets = _split_csv(args.datasets)
    backends = _split_csv(args.backends)

    unknown = [m for m in methods if m not in METHOD_TO_RUNNER]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    for ds in datasets:
        if ds not in {"hotpotqa", "musique", "2wiki"}:
            raise ValueError(f"Unknown dataset: {ds}")

    for b in backends:
        if b not in {"qwen", "deepseek"}:
            raise ValueError(f"Unknown backend: {b}")

    deepseek_key = str(args.deepseek_api_key or "").strip()
    if "deepseek" in backends and not deepseek_key:
        raise RuntimeError("deepseek backend requested but DeepSeek API key is empty")

    env_map = {
        "lightrag": args.env_lightrag,
        "graphrag": args.env_graphrag,
        "raptor": args.env_raptor,
        "bm25": args.env_bm25,
        "dense": args.env_dense,
    }

    args.logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for method in methods:
        for dataset in datasets:
            for backend in backends:
                print(f"[start] {method}/{dataset}/{backend}")
                row = _run_one(
                    method=method,
                    dataset=dataset,
                    backend=backend,
                    env_name=env_map[method],
                    args=args,
                    deepseek_key=deepseek_key,
                    logs_dir=args.logs_dir,
                )
                rows.append(row)
                print(
                    f"[{method}/{dataset}/{backend}] rc={row['rc']} "
                    f"ok={row['ok']} dur={row['duration_s']}s"
                )

    _write_tsv(args.summary_tsv, rows)

    total = len(rows)
    ok_count = sum(1 for r in rows if r["ok"])
    failed = [r for r in rows if not r["ok"]]
    summary = {
        "total": total,
        "ok": ok_count,
        "failed": total - ok_count,
        "summary_tsv": str(args.summary_tsv),
        "logs_dir": str(args.logs_dir),
        "failed_cases": [
            {
                "method": r["method"],
                "dataset": r["dataset"],
                "backend": r["backend"],
                "rc": r["rc"],
                "log_path": r["log_path"],
            }
            for r in failed
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
