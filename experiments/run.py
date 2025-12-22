#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

RESERVED_KEYS = {
    "meta",
    "runtime",
    "llm",
    "embedding",
    "budgets",
    "topk",
    "answer_format",
    "context",
    "exp",
    "datasets",
    "methods",
    "report",
    "retrieval_eval",
    "ablations",
    "llm_profiles",
}

ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slugify(text: str) -> str:
    if not text:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _load_with_includes(path: Path, seen: Optional[set] = None) -> Dict[str, Any]:
    if seen is None:
        seen = set()
    resolved = path.resolve()
    if resolved in seen:
        raise RuntimeError(f"Include cycle detected at {resolved}")
    seen.add(resolved)

    data = _load_yaml(path)
    includes = data.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        merged = _deep_merge(merged, _load_with_includes(inc_path, seen))
    merged = _deep_merge(merged, data)
    return merged


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            name = match.group(1)
            default = match.group(3) if match.group(2) else ""
            return os.environ.get(name, default or "")

        return ENV_PATTERN.sub(_replace, value)
    return value


def _collect_manifests(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    datasets: Dict[str, Any] = {}
    methods: Dict[str, Any] = {}
    for key, value in cfg.items():
        if key in RESERVED_KEYS:
            continue
        if not isinstance(value, dict):
            continue
        if _looks_like_method(value):
            methods[key] = value
        elif _looks_like_dataset(value):
            datasets[key] = value
    return datasets, methods


def _looks_like_dataset(value: Dict[str, Any]) -> bool:
    if "type" in value and ("path" in value or "dataset" in value or "doc_pool" in value):
        return True
    if "dataset" in value and "doc_pool" in value:
        return True
    return False


def _looks_like_method(value: Dict[str, Any]) -> bool:
    if "kind" in value or "entry" in value or "build_index" in value or "run" in value:
        return True
    for key in value.keys():
        if str(key).startswith("entry_"):
            return True
    return False


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _supported_flags(entry: Path, cache: Dict[str, set]) -> set:
    entry_key = str(entry)
    if entry_key in cache:
        return cache[entry_key]
    try:
        result = subprocess.run(
            [sys.executable, str(entry), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        flags = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", text))
    except Exception:
        flags = set()
    cache[entry_key] = flags
    return flags


def _pick_entry(method_def: Dict[str, Any], dataset_type: str) -> Tuple[Optional[Any], Optional[Any]]:
    entry = method_def.get(f"entry_{dataset_type}", method_def.get("entry"))
    build_entry = method_def.get(f"build_index_{dataset_type}", method_def.get("build_index"))
    return entry, build_entry


def _pick_cli_args(method_def: Dict[str, Any], dataset_type: str) -> List[str]:
    args: List[str] = []
    for key in ("cli_args", f"cli_args_{dataset_type}"):
        val = method_def.get(key)
        if val:
            args.extend(list(val))
    return args


def _resolve_repo_path(root: Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path)


def _add_first_supported(cmd: List[str], supported: set, flags: Iterable[str], value: Any) -> bool:
    if value is None:
        return False
    for flag in flags:
        if flag in supported:
            cmd.extend([flag, str(value)])
            return True
    return False


def _add_flag(cmd: List[str], supported: set, flags: Iterable[str]) -> bool:
    for flag in flags:
        if flag in supported:
            cmd.append(flag)
            return True
    return False


def _collect_metrics(workdir: Path) -> Dict[str, Any]:
    metrics_dir = workdir / "metrics"
    if not metrics_dir.exists():
        return {}
    metrics: Dict[str, Any] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = path.stem
        metrics[key] = data
    return metrics


def _normalize_metrics(metrics: Dict[str, Any], workdir_name: str) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for name, data in metrics.items():
        if isinstance(data, dict) and len(data) == 1 and workdir_name in data:
            normalized[name] = data[workdir_name]
        elif isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
            normalized[name] = data["metrics"]
        else:
            normalized[name] = data
    return normalized


def _run_cmd(cmd: List[str], log_path: Path, env: Dict[str, str], dry_run: bool) -> None:
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def _eval_command(
    dataset_def: Dict[str, Any], workdir: Path, cfg: Dict[str, Any], repo_root: Path
) -> Optional[List[str]]:
    dataset_type = dataset_def.get("type")
    dataset_path = dataset_def.get("path") or dataset_def.get("dataset")
    dataset_path = _resolve_repo_path(repo_root, dataset_path)
    if not dataset_path:
        return None

    if dataset_type == "hotpotqa":
        ks = cfg.get("retrieval_eval", {}).get("ks") or cfg.get("topk", {}).get("retrieve_k") or [1, 3, 5, 10]
        ks_str = ",".join(str(k) for k in ks)
        return [
            sys.executable,
            "scripts/evaluate_hotpotqa_metrics.py",
            "--dataset",
            str(dataset_path),
            "--workdir",
            str(workdir),
            "--ks",
            ks_str,
            "--output",
            str(workdir / "metrics" / "hotpot_metrics.json"),
        ]

    if dataset_type == "musique":
        return [
            sys.executable,
            "scripts/evaluate_musique_metrics.py",
            "--dataset",
            str(dataset_path),
            "--workdir",
            str(workdir),
            "--output",
            str(workdir / "metrics" / "musique_metrics.json"),
        ]

    if dataset_type == "mirage":
        return [
            sys.executable,
            "evaluate_mirage.py",
            "--dataset",
            str(dataset_path),
            "--workdir",
            str(workdir),
            "--output",
            str(workdir / "metrics" / "mirage_metrics.json"),
        ]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--only", default=None, help="Run only jobs whose id contains this substring")
    parser.add_argument("--max-jobs", type=int, default=0, help="Stop after N jobs (0 = no limit)")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after job failures")
    parser.add_argument("--force", action="store_true", help="Ignore resume and re-run jobs")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _expand_env(_load_with_includes(cfg_path))
    repo_root = Path(__file__).resolve().parents[1]

    datasets_manifest, methods_manifest = _collect_manifests(cfg)

    exp = cfg.get("exp", {})
    exp_name = exp.get("name", cfg_path.stem)

    result_root = cfg.get("meta", {}).get("result_root", "result_relrag")
    run_root = Path(result_root) / exp_name / f"run_{_timestamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    llm_default = cfg.get("llm", {})
    llm_profiles = cfg.get("llm_profiles")
    if not llm_profiles:
        llm_profiles = [{"name": "default", **llm_default}]
    exp_llm = exp.get("llm_profile")
    if exp_llm:
        filtered = [p for p in llm_profiles if p.get("name") == exp_llm]
        if filtered:
            llm_profiles = filtered

    runtime = cfg.get("runtime", {})
    embedding = cfg.get("embedding", {})
    topk = cfg.get("topk", {})
    budgets = cfg.get("report", {}).get("budgets", {}).get("token_budgets")
    if budgets is None:
        budgets = cfg.get("budgets", {}).get("token_budgets", [])

    datasets = cfg.get("datasets") or list(datasets_manifest.keys())
    methods = cfg.get("methods") or list(methods_manifest.keys())
    ablations = cfg.get("ablations") or []
    retrieval_only = bool(cfg.get("retrieval_eval"))

    config_snapshot = {
        "config": cfg,
        "exp_name": exp_name,
        "run_root": str(run_root),
        "git_commit": _git_commit(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
    }
    (run_root / "config.resolved.json").write_text(
        json.dumps(config_snapshot, indent=2), encoding="utf-8"
    )

    env = os.environ.copy()
    if llm_default.get("no_proxy"):
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"

    supported_cache: Dict[str, set] = {}
    built_indexes: set = set()
    job_results: List[Dict[str, Any]] = []

    jobs_run = 0
    for dataset_id in datasets:
        dataset_def = datasets_manifest.get(dataset_id)
        if not dataset_def:
            raise KeyError(f"Dataset not found in manifests: {dataset_id}")
        dataset_type = dataset_def.get("type", dataset_id)

        for method_id in methods:
            method_def = methods_manifest.get(method_id)
            if not method_def:
                raise KeyError(f"Method not found in manifests: {method_id}")

            entry, build_entry = _pick_entry(method_def, dataset_type)
            if entry is None:
                job_results.append(
                    {
                        "dataset": dataset_id,
                        "method": method_id,
                        "status": "skipped_missing_entry",
                    }
                )
                continue

            entry_path = Path(_resolve_repo_path(repo_root, entry))
            if not entry_path.exists():
                job_results.append(
                    {
                        "dataset": dataset_id,
                        "method": method_id,
                        "status": "skipped_missing_entry",
                        "message": f"Entry not found: {entry_path}",
                    }
                )
                continue

            build_entry_path = None
            if build_entry:
                build_entry_path = Path(_resolve_repo_path(repo_root, build_entry))
                if not build_entry_path.exists():
                    build_entry_path = None

            variants = ablations or [{"name": "full"}]
            for variant in variants:
                variant_name = variant.get("name", "full") if isinstance(variant, dict) else str(variant)
                variant_cli_args = []
                if isinstance(variant, dict) and variant.get("cli_args"):
                    variant_cli_args = list(variant.get("cli_args"))

                for llm in llm_profiles:
                    llm_name = llm.get("name", llm.get("model", "default"))

                    budget_list = budgets or [None]
                    for budget in budget_list:
                        job_id = f"{dataset_id}:{method_id}:{llm_name}:{budget}:{variant_name}"
                        if args.only and args.only not in job_id:
                            continue

                        workdir = run_root / _slugify(dataset_id) / _slugify(method_id)
                        workdir = workdir / f"llm_{_slugify(llm_name)}"
                        budget_tag = f"budget_{budget}" if budget is not None else "budget_default"
                        workdir = workdir / budget_tag / _slugify(variant_name)
                        workdir.mkdir(parents=True, exist_ok=True)

                        started = time.time()
                        started_at = datetime.now().isoformat(timespec="seconds")

                        status = "ok"
                        message = ""
                        cmd: List[str] = []

                        try:
                            supported = _supported_flags(entry_path, supported_cache)

                            if build_entry_path:
                                build_key = (dataset_id, method_id, str(workdir))
                                if args.force or build_key not in built_indexes:
                                    build_supported = _supported_flags(build_entry_path, supported_cache)
                                    build_cmd = [sys.executable, str(build_entry_path)]
                                    _add_first_supported(build_cmd, build_supported, ["--workdir", "--work-dir"], workdir)
                                    _add_first_supported(build_cmd, build_supported, ["--result-root"], result_root)
                                dataset_path = dataset_def.get("path") or dataset_def.get("dataset")
                                dataset_path = _resolve_repo_path(repo_root, dataset_path)
                                    if dataset_path:
                                        _add_first_supported(build_cmd, build_supported, ["--dataset"], dataset_path)
                                    doc_pool = dataset_def.get("doc_pool")
                                    doc_pool = _resolve_repo_path(repo_root, doc_pool)
                                    if doc_pool:
                                        _add_first_supported(build_cmd, build_supported, ["--doc-pool"], doc_pool)
                                    _run_cmd(build_cmd, workdir / "build.log", env, args.dry_run)
                                    built_indexes.add(build_key)

                            cmd = [sys.executable, str(entry_path)]
                            dataset_path = dataset_def.get("path") or dataset_def.get("dataset")
                            dataset_path = _resolve_repo_path(repo_root, dataset_path)
                            if dataset_path:
                                _add_first_supported(cmd, supported, ["--dataset"], dataset_path)

                            doc_pool = dataset_def.get("doc_pool")
                            doc_pool = _resolve_repo_path(repo_root, doc_pool)
                            if doc_pool:
                                _add_first_supported(cmd, supported, ["--doc-pool"], doc_pool)

                            _add_first_supported(cmd, supported, ["--workdir", "--work-dir"], workdir)
                            _add_first_supported(cmd, supported, ["--result-root"], result_root)

                            _add_first_supported(cmd, supported, ["--lm-endpoint"], llm.get("endpoint"))
                            _add_first_supported(cmd, supported, ["--lm-model"], llm.get("model"))

                            _add_first_supported(cmd, supported, ["--embed-model", "--emb-model"], embedding.get("model"))
                            _add_first_supported(cmd, supported, ["--embed-device", "--emb-device"], embedding.get("device"))
                            _add_first_supported(cmd, supported, ["--embed-batch-size"], embedding.get("batch_size"))
                            _add_first_supported(cmd, supported, ["--embed-max-length"], embedding.get("max_length"))
                            _add_first_supported(cmd, supported, ["--emb-dtype"], embedding.get("dtype"))

                            if embedding.get("normalize") is True:
                                _add_flag(cmd, supported, ["--embed-normalize"])
                            elif embedding.get("normalize") is False:
                                _add_flag(cmd, supported, ["--no-embed-normalize"])

                            _add_first_supported(cmd, supported, ["--num-workers"], runtime.get("workers"))
                            _add_first_supported(cmd, supported, ["--save-every"], runtime.get("save_every"))

                            if runtime.get("resume") and not args.force:
                                _add_flag(cmd, supported, ["--resume"])

                            if runtime.get("limit"):
                                _add_first_supported(cmd, supported, ["--limit"], runtime.get("limit"))

                            if topk.get("gen_k") is not None:
                                _add_first_supported(cmd, supported, ["--topk"], topk.get("gen_k"))

                            if budget is not None:
                                _add_first_supported(
                                    cmd,
                                    supported,
                                    ["--max-tokens", "--max-new-tokens", "--lm-max-tokens", "--context-max-tokens"],
                                    budget,
                                )

                            if retrieval_only:
                                _add_flag(cmd, supported, ["--retrieval-only"])

                            method_cli_args = _pick_cli_args(method_def, dataset_type)
                            cmd.extend(method_cli_args)
                            cmd.extend(variant_cli_args)

                            _run_cmd(cmd, workdir / "run.log", env, args.dry_run)
                        except Exception as exc:
                            status = "failed"
                            message = str(exc)
                            if not args.continue_on_error:
                                ended = time.time()
                                ended_at = datetime.now().isoformat(timespec="seconds")
                                job_results.append(
                                    {
                                        "dataset": dataset_id,
                                        "method": method_id,
                                        "llm": llm_name,
                                        "budget": budget,
                                        "variant": variant_name,
                                        "status": status,
                                        "message": message,
                                        "started_at": started_at,
                                        "ended_at": ended_at,
                                        "duration_s": round(ended - started, 2),
                                        "workdir": str(workdir),
                                        "cmd": cmd,
                                        "metrics": {},
                                    }
                                )
                                raise

                        eval_status = None
                        eval_message = None
                        if runtime.get("run_eval") and not args.dry_run:
                            eval_cmd = _eval_command(dataset_def, workdir, cfg, repo_root)
                            if eval_cmd:
                                try:
                                    _run_cmd(eval_cmd, workdir / "eval.log", env, args.dry_run)
                                    eval_status = "ok"
                                except Exception as exc:
                                    eval_status = "failed"
                                    eval_message = str(exc)
                            else:
                                eval_status = "skipped"
                                eval_message = "No evaluator configured"

                        ended = time.time()
                        ended_at = datetime.now().isoformat(timespec="seconds")

                        metrics = _normalize_metrics(_collect_metrics(workdir), workdir.name)
                        job_results.append(
                            {
                                "dataset": dataset_id,
                                "method": method_id,
                                "llm": llm_name,
                                "budget": budget,
                                "variant": variant_name,
                                "status": status,
                                "message": message,
                                "started_at": started_at,
                                "ended_at": ended_at,
                                "duration_s": round(ended - started, 2),
                                "workdir": str(workdir),
                                "cmd": cmd,
                                "metrics": metrics,
                                "eval_status": eval_status,
                                "eval_message": eval_message,
                                "overrides": variant.get("overrides") if isinstance(variant, dict) else None,
                            }
                        )

                        jobs_run += 1
                        if args.max_jobs and jobs_run >= args.max_jobs:
                            break
                    if args.max_jobs and jobs_run >= args.max_jobs:
                        break
                if args.max_jobs and jobs_run >= args.max_jobs:
                    break
            if args.max_jobs and jobs_run >= args.max_jobs:
                break
        if args.max_jobs and jobs_run >= args.max_jobs:
            break

    summary = {
        "exp_name": exp_name,
        "run_root": str(run_root),
        "git_commit": _git_commit(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": job_results,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
