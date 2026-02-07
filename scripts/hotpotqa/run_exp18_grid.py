#!/usr/bin/env python3
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _load_grid(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    common = payload.get("common") or {}
    runs = payload.get("runs") or []
    if not isinstance(common, dict):
        raise ValueError("grid common must be a dict")
    if not isinstance(runs, list) or not runs:
        raise ValueError("grid runs must be a non-empty list")
    normalized_runs: List[Dict[str, Any]] = []
    for row in runs:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        normalized_runs.append(dict(row))
    if not normalized_runs:
        raise ValueError("no valid runs found in grid")
    return common, normalized_runs


def _as_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_hotpot_cmd(
    *,
    python_bin: str,
    repo_root: Path,
    config_path: Path,
    data_path: Path,
    output_dir: Path,
    limit: int,
    workers: int,
    force_build: bool,
    params: Dict[str, Any],
) -> List[str]:
    cmd = [
        python_bin,
        str(repo_root / "hotpot_entry.py"),
        "--config",
        str(config_path),
        "--data",
        str(data_path),
        "--output_dir",
        str(output_dir),
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if workers > 0:
        cmd.extend(["--workers", str(workers)])
    if force_build:
        cmd.append("--force_build")

    for key in sorted(params.keys()):
        if key == "name":
            continue
        value = params[key]
        cmd.extend([f"--{key}", _as_cli_value(value)])
    return cmd


def _run_cmd(cmd: List[str], *, cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise RuntimeError(message)
    return proc.stdout


def _resolve_pred_path(run_dir: Path, split: str, reader: str, retriever: str) -> Path:
    expected = run_dir / f"pred_{split}_{reader}_{retriever}.jsonl"
    if expected.exists():
        return expected
    candidates = sorted(run_dir.glob("pred_*.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no pred_*.jsonl found in {run_dir}")
    raise RuntimeError(f"multiple prediction files in {run_dir}, cannot resolve automatically")


def _eval_official(repo_root: Path, pred_json: Path, gold_json: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo_root / "eval" / "hotpot_evaluate_v1.py"),
        str(pred_json),
        str(gold_json),
    ]
    output = _run_cmd(cmd, cwd=repo_root).strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty evaluator output for {pred_json}")
    last = lines[-1]
    try:
        parsed = ast.literal_eval(last)
    except Exception as exc:
        raise RuntimeError(f"failed to parse evaluator output: {last}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected evaluator payload: {type(parsed)}")
    return parsed


def _read_manifest(align_dir: Path) -> Dict[str, Any]:
    manifest_path = align_dir / "alignment_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_results(out_root: Path, rows: List[Dict[str, Any]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "grid_results.json"
    md_path = out_root / "grid_results.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    ordered = sorted(rows, key=lambda row: float(row.get("f1", 0.0)), reverse=True)
    lines = [
        "# Experiment 18 Grid Results",
        "",
        "| run | f1 | sp_f1 | joint_f1 | sp_prec | sp_recall | topk_sp_f1 | refill_triggered | refill_added_mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ordered:
        lines.append(
            "| {run} | {f1:.4f} | {sp_f1:.4f} | {joint_f1:.4f} | {sp_prec:.4f} | {sp_recall:.4f} | {topk_sp_f1:.4f} | {refill_triggered:.4f} | {refill_added_mean:.4f} |".format(
                run=row.get("name", ""),
                f1=float(row.get("f1", 0.0)),
                sp_f1=float(row.get("sp_f1", 0.0)),
                joint_f1=float(row.get("joint_f1", 0.0)),
                sp_prec=float(row.get("sp_prec", 0.0)),
                sp_recall=float(row.get("sp_recall", 0.0)),
                topk_sp_f1=float(row.get("topk_sp_f1", 0.0)),
                refill_triggered=float(row.get("refill_triggered_ratio", 0.0)),
                refill_added_mean=float(row.get("refill_added_mean", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- `topk_sp_f1` is evaluated from `official_pred_topk.json` (legacy full top-k supporting facts).",
            "- `sp_f1` is evaluated from `official_pred.json` (current policy-selected `pred_sp`).",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hotpot Exp18 parameter grid and summarize results")
    parser.add_argument("--grid", default="scripts/hotpotqa/exp18_grid.yaml", help="Grid YAML path")
    parser.add_argument("--config", default="relrag/config/config.yaml", help="hotpot_entry config")
    parser.add_argument("--data", default="data/hotpot_dev_distractor_500_jsonl.jsonl", help="dataset jsonl")
    parser.add_argument("--output_root", default="result/experiment_18", help="experiment root")
    parser.add_argument("--python", default=sys.executable, help="Python binary for hotpot_entry")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for fast debug")
    parser.add_argument("--workers", type=int, default=1, help="Override workers")
    parser.add_argument("--force_build", action="store_true", help="Force index rebuild")
    parser.add_argument("--only", help="Comma-separated run names to execute")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    grid_path = (repo_root / args.grid).resolve() if not Path(args.grid).is_absolute() else Path(args.grid)
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    data_path = (repo_root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)

    common, runs = _load_grid(grid_path)
    if args.only:
        allow = {name.strip() for name in str(args.only).split(",") if name.strip()}
        runs = [run for run in runs if str(run.get("name")) in allow]
        if not runs:
            raise ValueError(f"--only filtered all runs: {sorted(allow)}")

    results: List[Dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for run in runs:
        run_name = str(run["name"])
        params = dict(common)
        params.update(run)
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_hotpot_cmd(
            python_bin=args.python,
            repo_root=repo_root,
            config_path=config_path,
            data_path=data_path,
            output_dir=run_dir,
            limit=int(args.limit),
            workers=int(args.workers),
            force_build=bool(args.force_build),
            params=params,
        )
        print(f"[grid] run={run_name}")
        print("[grid] cmd:", " ".join(cmd))
        if args.dry_run:
            continue

        _run_cmd(cmd, cwd=repo_root)

        split = str(params.get("split") or "dev")
        reader = str(params.get("reader") or "openai")
        retriever = str(params.get("retriever") or "bm25")
        pred_path = _resolve_pred_path(run_dir, split, reader, retriever)
        align_dir = run_dir / f"align_{pred_path.stem}"
        official_pred = align_dir / "official_pred.json"
        official_gold = align_dir / "official_gold.json"
        official_pred_topk = align_dir / "official_pred_topk.json"

        metrics = _eval_official(repo_root, official_pred, official_gold)
        topk_metrics: Dict[str, Any] = {}
        if official_pred_topk.exists():
            topk_metrics = _eval_official(repo_root, official_pred_topk, official_gold)

        manifest = _read_manifest(align_dir)
        shortage = ((manifest.get("retrieval") or {}).get("shortage_refill") or {})
        refill_added = shortage.get("added") or {}
        row = {
            "name": run_name,
            "params": {k: v for k, v in params.items() if k != "name"},
            "command": cmd,
            "f1": float(metrics.get("f1", 0.0)),
            "sp_f1": float(metrics.get("sp_f1", 0.0)),
            "joint_f1": float(metrics.get("joint_f1", 0.0)),
            "sp_prec": float(metrics.get("sp_prec", 0.0)),
            "sp_recall": float(metrics.get("sp_recall", 0.0)),
            "topk_sp_f1": float(topk_metrics.get("sp_f1", 0.0)) if topk_metrics else 0.0,
            "refill_triggered_ratio": float(shortage.get("triggered_ratio", 0.0)),
            "refill_added_mean": float(refill_added.get("mean", 0.0)),
            "pred_path": str(pred_path),
            "align_dir": str(align_dir),
        }
        results.append(row)

    if not args.dry_run:
        _write_results(output_root, results)
        print(f"[grid] wrote {output_root / 'grid_results.json'}")
        print(f"[grid] wrote {output_root / 'grid_results.md'}")


if __name__ == "__main__":
    main()
