#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from utils.llm_client import get_all_profile_snapshots

RESERVED_KEYS = {
    "meta",
    "runtime",
    "llm",
    "embedding",
    "budgets",
    "fairness",
    "decode",
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


def _pick_arg_list(method_def: Dict[str, Any], dataset_type: str, key: str, default: Iterable[str]) -> List[str]:
    val = method_def.get(f"{key}_{dataset_type}", method_def.get(key))
    if val is None:
        return list(default)
    if isinstance(val, list):
        return list(val)
    return [str(val)]


def _pick_prompt_paths(method_def: Dict[str, Any], dataset_type: str) -> List[str]:
    paths: List[str] = []
    for key in ("prompt_paths", f"prompt_paths_{dataset_type}"):
        val = method_def.get(key)
        if not val:
            continue
        if isinstance(val, list):
            paths.extend(str(v) for v in val)
        else:
            paths.append(str(val))
    return paths


def _resolve_repo_path(root: Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path)


def _hash_file(path: Path, cache: Dict[str, str]) -> str:
    key = str(path)
    if key in cache:
        return cache[key]
    if not path.exists():
        cache[key] = "missing"
        return cache[key]
    try:
        result = subprocess.run(
            ["git", "hash-object", key],
            check=False,
            capture_output=True,
            text=True,
        )
        digest = result.stdout.strip()
        if result.returncode == 0 and digest:
            cache[key] = f"git:{digest}"
            return cache[key]
    except Exception:
        pass
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    cache[key] = f"sha256:{digest}"
    return cache[key]


def _record_applied(applied: Dict[str, Any], flag: Optional[str], value: Any) -> None:
    if flag:
        applied[flag] = value


def _count_pred_entries(payload: Any) -> Optional[int]:
    if isinstance(payload, dict):
        answers = payload.get("answer")
        if isinstance(answers, dict):
            return len(answers)
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    return None


def _count_non_empty_lines(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return None


def _resume_state(workdir: Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    pred_path = workdir / "preds" / "pred.json"
    if pred_path.exists():
        try:
            payload = json.loads(pred_path.read_text(encoding="utf-8"))
            state["pred_json"] = {
                "path": str(pred_path),
                "count": _count_pred_entries(payload),
            }
        except Exception:
            state["pred_json"] = {"path": str(pred_path), "count": None}

    qa_path = workdir / "preds" / "qa.tsv"
    if qa_path.exists():
        state["qa_tsv"] = {
            "path": str(qa_path),
            "count": _count_non_empty_lines(qa_path),
        }

    retrieval_path = workdir / "artifacts" / "retrieval.jsonl"
    if retrieval_path.exists():
        state["retrieval_jsonl"] = {
            "path": str(retrieval_path),
            "count": _count_non_empty_lines(retrieval_path),
        }

    return state


def _resume_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for key in ("pred_json", "qa_tsv", "retrieval_jsonl"):
        before_count = before.get(key, {}).get("count") if before else None
        after_count = after.get(key, {}).get("count") if after else None
        if before_count is not None and after_count is not None:
            delta[key] = after_count - before_count
    return delta


def _add_first_supported(cmd: List[str], supported: set, flags: Iterable[str], value: Any) -> Optional[str]:
    if value is None:
        return None
    for flag in flags:
        if flag in supported:
            cmd.extend([flag, str(value)])
            return flag
    return None


def _add_flag(cmd: List[str], supported: set, flags: Iterable[str]) -> Optional[str]:
    for flag in flags:
        if flag in supported:
            cmd.append(flag)
            return flag
    return None


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

    if dataset_type in ("hotpotqa", "musique", "mirage"):
        ks = cfg.get("retrieval_eval", {}).get("ks") or cfg.get("topk", {}).get("retrieve_k") or [1, 3, 5, 10]
        if isinstance(ks, int):
            ks = [ks]
        ks_str = ",".join(str(k) for k in ks)
        return [
            sys.executable,
            "scripts/evaluate_relrag.py",
            "--dataset",
            str(dataset_path),
            "--dataset-name",
            dataset_type,
            "--workdir",
            str(workdir),
            "--ks",
            ks_str,
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
    decode = cfg.get("decode", {})
    fairness = cfg.get("fairness", {})
    budgets = cfg.get("report", {}).get("budgets", {}).get("token_budgets")
    if budgets is None:
        budgets = cfg.get("budgets", {}).get("token_budgets", [])

    datasets = cfg.get("datasets") or list(datasets_manifest.keys())
    methods = cfg.get("methods") or list(methods_manifest.keys())
    ablations = cfg.get("ablations") or []
    retrieval_only = bool(cfg.get("runtime", {}).get("retrieval_only", False))

    config_snapshot = {
        "config": cfg,
        "exp_name": exp_name,
        "run_root": str(run_root),
        "git_commit": _git_commit(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "llm_profiles_resolved": get_all_profile_snapshots(
            generate_max_tokens=decode.get("max_tokens"),
            generate_temperature=decode.get("temperature"),
        ),
    }
    (run_root / "config.resolved.json").write_text(
        json.dumps(config_snapshot, indent=2), encoding="utf-8"
    )

    env = os.environ.copy()
    if llm_default.get("no_proxy"):
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"
    if runtime.get("seed") is not None:
        env["PYTHONHASHSEED"] = str(runtime.get("seed"))

    supported_cache: Dict[str, set] = {}
    hash_cache: Dict[str, str] = {}
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
                    job_env = dict(env)
                    job_env["LLM_PROFILE"] = str(llm_name)

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
                        cmd_str = ""
                        applied_flags: Dict[str, Any] = {}
                        resume_state_before = _resume_state(workdir)
                        resume_state_after: Optional[Dict[str, Any]] = None
                        seed_wrapper_used = False
                        budget_flag = None
                        topk_flag = None
                        seed_flag = None
                        method_cli_args: List[str] = []

                        entry_hash = _hash_file(entry_path, hash_cache)
                        prompt_hashes: Dict[str, str] = {}
                        for prompt_path in _pick_prompt_paths(method_def, dataset_type):
                            resolved_prompt = Path(_resolve_repo_path(repo_root, prompt_path))
                            prompt_hashes[str(resolved_prompt)] = _hash_file(resolved_prompt, hash_cache)

                        requires_topk = bool(method_def.get("requires_topk", False))
                        enforce_budget = bool(fairness.get("enforce_budget", runtime.get("enforce_budget", False)))
                        enforce_topk = bool(fairness.get("enforce_topk", runtime.get("enforce_topk", False)))
                        require_prompt_hash = bool(
                            fairness.get("require_prompt_hash", runtime.get("require_prompt_hash", False))
                        )

                        try:
                            supported = _supported_flags(entry_path, supported_cache)

                            dataset_path = dataset_def.get("path") or dataset_def.get("dataset")
                            dataset_path = _resolve_repo_path(repo_root, dataset_path)
                            doc_pool = dataset_def.get("doc_pool")
                            doc_pool = _resolve_repo_path(repo_root, doc_pool)

                            dataset_flags = _pick_arg_list(
                                method_def,
                                dataset_type,
                                "dataset_args",
                                ["--dataset", "--dataset-path"],
                            )
                            doc_pool_flags = _pick_arg_list(method_def, dataset_type, "doc_pool_args", ["--doc-pool"])
                            lm_endpoint_flags = _pick_arg_list(
                                method_def,
                                dataset_type,
                                "lm_endpoint_args",
                                ["--lm-endpoint"],
                            )
                            lm_model_flags = _pick_arg_list(
                                method_def,
                                dataset_type,
                                "lm_model_args",
                                ["--lm-model"],
                            )

                            cmd = [sys.executable, str(entry_path)]
                            if dataset_path:
                                flag = _add_first_supported(cmd, supported, dataset_flags, dataset_path)
                                _record_applied(applied_flags, flag, dataset_path)

                            if doc_pool:
                                flag = _add_first_supported(cmd, supported, doc_pool_flags, doc_pool)
                                _record_applied(applied_flags, flag, doc_pool)

                            flag = _add_first_supported(cmd, supported, ["--workdir", "--work-dir"], workdir)
                            _record_applied(applied_flags, flag, str(workdir))
                            flag = _add_first_supported(cmd, supported, ["--result-root"], result_root)
                            _record_applied(applied_flags, flag, result_root)

                            flag = _add_first_supported(cmd, supported, lm_endpoint_flags, llm.get("endpoint"))
                            _record_applied(applied_flags, flag, llm.get("endpoint"))
                            flag = _add_first_supported(cmd, supported, lm_model_flags, llm.get("model"))
                            _record_applied(applied_flags, flag, llm.get("model"))

                            flag = _add_first_supported(cmd, supported, ["--embed-model", "--emb-model"], embedding.get("model"))
                            _record_applied(applied_flags, flag, embedding.get("model"))
                            flag = _add_first_supported(cmd, supported, ["--embed-device", "--emb-device"], embedding.get("device"))
                            _record_applied(applied_flags, flag, embedding.get("device"))
                            flag = _add_first_supported(cmd, supported, ["--embed-batch-size"], embedding.get("batch_size"))
                            _record_applied(applied_flags, flag, embedding.get("batch_size"))
                            flag = _add_first_supported(cmd, supported, ["--embed-max-length"], embedding.get("max_length"))
                            _record_applied(applied_flags, flag, embedding.get("max_length"))
                            flag = _add_first_supported(cmd, supported, ["--emb-dtype"], embedding.get("dtype"))
                            _record_applied(applied_flags, flag, embedding.get("dtype"))

                            if embedding.get("normalize") is True:
                                flag = _add_flag(cmd, supported, ["--embed-normalize"])
                                _record_applied(applied_flags, flag, True)
                            elif embedding.get("normalize") is False:
                                flag = _add_flag(cmd, supported, ["--no-embed-normalize"])
                                _record_applied(applied_flags, flag, True)

                            flag = _add_first_supported(cmd, supported, ["--num-workers"], runtime.get("workers"))
                            _record_applied(applied_flags, flag, runtime.get("workers"))
                            flag = _add_first_supported(cmd, supported, ["--save-every"], runtime.get("save_every"))
                            _record_applied(applied_flags, flag, runtime.get("save_every"))

                            resume_flag = None
                            if runtime.get("resume") and not args.force:
                                resume_flag = _add_flag(cmd, supported, ["--resume"])
                                _record_applied(applied_flags, resume_flag, True)

                            if runtime.get("limit"):
                                flag = _add_first_supported(cmd, supported, ["--limit"], runtime.get("limit"))
                                _record_applied(applied_flags, flag, runtime.get("limit"))

                            if topk.get("gen_k") is not None:
                                topk_flag = _add_first_supported(cmd, supported, ["--topk"], topk.get("gen_k"))
                                _record_applied(applied_flags, topk_flag, topk.get("gen_k"))

                            if budget is not None:
                                budget_flag = _add_first_supported(
                                    cmd,
                                    supported,
                                    [
                                        "--context-budget",
                                        "--context-budget-tokens",
                                        "--max-tokens",
                                        "--max-new-tokens",
                                        "--lm-max-tokens",
                                        "--context-max-tokens",
                                    ],
                                    budget,
                                )
                                _record_applied(applied_flags, budget_flag, budget)

                            if decode.get("temperature") is not None:
                                flag = _add_first_supported(cmd, supported, ["--temperature"], decode.get("temperature"))
                                _record_applied(applied_flags, flag, decode.get("temperature"))

                            if decode.get("top_p") is not None:
                                flag = _add_first_supported(cmd, supported, ["--top-p", "--top_p"], decode.get("top_p"))
                                _record_applied(applied_flags, flag, decode.get("top_p"))

                            if decode.get("repetition_penalty") is not None:
                                flag = _add_first_supported(
                                    cmd,
                                    supported,
                                    ["--repetition-penalty", "--repetition_penalty"],
                                    decode.get("repetition_penalty"),
                                )
                                _record_applied(applied_flags, flag, decode.get("repetition_penalty"))

                            if decode.get("max_tokens") is not None:
                                flag = _add_first_supported(
                                    cmd,
                                    supported,
                                    ["--max-new-tokens", "--max-tokens", "--lm-max-tokens"],
                                    decode.get("max_tokens"),
                                )
                                _record_applied(applied_flags, flag, decode.get("max_tokens"))

                            if retrieval_only:
                                flag = _add_flag(cmd, supported, ["--retrieval-only"])
                                _record_applied(applied_flags, flag, True)

                            seed = runtime.get("seed")
                            if seed is not None:
                                seed_flag = _add_first_supported(cmd, supported, ["--seed"], seed)
                                _record_applied(applied_flags, seed_flag, seed)

                            method_cli_args = _pick_cli_args(method_def, dataset_type)
                            cmd.extend(method_cli_args)
                            cmd.extend(variant_cli_args)

                            if seed is not None and seed_flag is None:
                                seed_wrapper_used = True
                                wrapper_path = repo_root / "experiments" / "seeded_run.py"
                                cmd = [sys.executable, str(wrapper_path), "--seed", str(seed), str(entry_path)] + cmd[2:]

                            cmd_str = shlex.join(cmd)

                            if require_prompt_hash and not prompt_hashes:
                                status = "skipped_prompt_hash_missing"
                                message = "No prompt_paths configured for this method"
                            elif budget is not None and enforce_budget and budget_flag is None:
                                status = "skipped_budget_not_applied"
                                message = "No supported budget flag in entry script"
                            elif requires_topk and enforce_topk and topk_flag is None:
                                status = "skipped_topk_not_applied"
                                message = "No supported --topk flag in entry script"

                            if status.startswith("skipped"):
                                budget_applied = budget_flag is not None if budget is not None else None
                                topk_applied = topk_flag is not None if topk.get("gen_k") is not None else None
                                resume_delta = _resume_delta(resume_state_before, resume_state_before)
                                job_results.append(
                                    {
                                        "dataset": dataset_id,
                                        "method": method_id,
                                        "dataset_def": dataset_def,
                                        "method_def": method_def,
                                        "llm": llm_name,
                                        "budget": budget,
                                        "variant": variant_name,
                                        "status": status,
                                        "message": message,
                                        "started_at": started_at,
                                        "ended_at": datetime.now().isoformat(timespec="seconds"),
                                        "duration_s": 0.0,
                                        "workdir": str(workdir),
                                        "cmd": cmd,
                                        "cmd_str": cmd_str,
                                        "entry": str(entry_path),
                                        "entry_hash": entry_hash,
                                        "prompt_hashes": prompt_hashes,
                                        "applied_flags": applied_flags,
                                        "method_cli_args": method_cli_args,
                                        "variant_cli_args": variant_cli_args,
                                        "budget_applied": budget_applied,
                                        "topk_applied": topk_applied,
                                        "resume": bool(runtime.get("resume")),
                                        "resume_state_before": resume_state_before,
                                        "resume_state_after": resume_state_before,
                                        "resume_delta": resume_delta,
                                        "seed_wrapper_used": seed_wrapper_used,
                                        "requires_topk": requires_topk,
                                        "effective_config": {
                                            "llm": llm,
                                            "embedding": embedding,
                                            "runtime": {
                                                "workers": runtime.get("workers"),
                                                "resume": runtime.get("resume"),
                                                "seed": runtime.get("seed"),
                                                "save_every": runtime.get("save_every"),
                                                "limit": runtime.get("limit"),
                                            },
                                            "decode": decode,
                                            "topk": topk,
                                            "budget": budget,
                                            "retrieval_only": retrieval_only,
                                        },
                                        "fairness": {
                                            "enforce_budget": enforce_budget,
                                            "enforce_topk": enforce_topk,
                                            "require_prompt_hash": require_prompt_hash,
                                        },
                                    }
                                )
                                continue

                            if build_entry_path:
                                build_key = (dataset_id, method_id, str(workdir))
                                if args.force or build_key not in built_indexes:
                                    build_supported = _supported_flags(build_entry_path, supported_cache)
                                    build_cmd = [sys.executable, str(build_entry_path)]
                                    _add_first_supported(build_cmd, build_supported, ["--workdir", "--work-dir"], workdir)
                                    _add_first_supported(build_cmd, build_supported, ["--result-root"], result_root)
                                    if dataset_path:
                                        _add_first_supported(build_cmd, build_supported, dataset_flags, dataset_path)
                                    if doc_pool:
                                        _add_first_supported(build_cmd, build_supported, doc_pool_flags, doc_pool)
                                    _run_cmd(build_cmd, workdir / "build.log", job_env, args.dry_run)
                                    built_indexes.add(build_key)

                            _run_cmd(cmd, workdir / "run.log", job_env, args.dry_run)
                            if not args.dry_run:
                                resume_state_after = _resume_state(workdir)
                        except Exception as exc:
                            status = "failed"
                            message = str(exc)
                            if not args.continue_on_error:
                                ended = time.time()
                                ended_at = datetime.now().isoformat(timespec="seconds")
                                budget_applied = budget_flag is not None if budget is not None else None
                                topk_applied = topk_flag is not None if topk.get("gen_k") is not None else None
                                job_results.append(
                                    {
                                        "dataset": dataset_id,
                                        "method": method_id,
                                        "dataset_def": dataset_def,
                                        "method_def": method_def,
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
                                        "cmd_str": cmd_str,
                                        "entry": str(entry_path),
                                        "entry_hash": entry_hash,
                                        "prompt_hashes": prompt_hashes,
                                        "applied_flags": applied_flags,
                                        "method_cli_args": method_cli_args,
                                        "variant_cli_args": variant_cli_args,
                                        "budget_applied": budget_applied,
                                        "topk_applied": topk_applied,
                                        "resume": bool(runtime.get("resume")),
                                        "resume_state_before": resume_state_before,
                                        "resume_state_after": resume_state_after or resume_state_before,
                                        "resume_delta": _resume_delta(
                                            resume_state_before, resume_state_after or resume_state_before
                                        ),
                                        "seed_wrapper_used": seed_wrapper_used,
                                        "requires_topk": requires_topk,
                                        "effective_config": {
                                            "llm": llm,
                                            "embedding": embedding,
                                            "runtime": {
                                                "workers": runtime.get("workers"),
                                                "resume": runtime.get("resume"),
                                                "seed": runtime.get("seed"),
                                                "save_every": runtime.get("save_every"),
                                                "limit": runtime.get("limit"),
                                            },
                                            "decode": decode,
                                            "topk": topk,
                                            "budget": budget,
                                            "retrieval_only": retrieval_only,
                                        },
                                        "fairness": {
                                            "enforce_budget": enforce_budget,
                                            "enforce_topk": enforce_topk,
                                            "require_prompt_hash": require_prompt_hash,
                                        },
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
                                    _run_cmd(eval_cmd, workdir / "eval.log", job_env, args.dry_run)
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
                        budget_applied = budget_flag is not None if budget is not None else None
                        topk_applied = topk_flag is not None if topk.get("gen_k") is not None else None
                        resume_state_final = resume_state_after or resume_state_before
                        
                        eval_entry_script = None
                        if runtime.get("run_eval") and not args.dry_run:
                            # Re-derive eval command to get the script path (safe even if skipped above).
                            temp_eval_cmd = _eval_command(dataset_def, workdir, cfg, repo_root)
                            if temp_eval_cmd and len(temp_eval_cmd) > 1:
                                eval_entry_script = temp_eval_cmd[1]

                        job_results.append(
                            {
                                "dataset": dataset_id,
                                "method": method_id,
                                "dataset_def": dataset_def,
                                "method_def": method_def,
                                "llm": llm_name,
                                "budget": budget,
                                "variant": variant_name,
                                "status": status,
                                "message": message,
                                "started_at": started_at,
                                "ended_at": ended_at,
                                "duration_s": round(ended - started, 2),
                                "workdir": str(workdir),
                                "pred_source": str(workdir / "preds" / "pred_raw.jsonl"),
                                "eval_entry": eval_entry_script or "none",
                                "cmd": cmd,
                                "cmd_str": cmd_str,
                                "entry": str(entry_path),
                                "entry_hash": entry_hash,
                                "prompt_hashes": prompt_hashes,
                                "applied_flags": applied_flags,
                                "method_cli_args": method_cli_args,
                                "variant_cli_args": variant_cli_args,
                                "budget_applied": budget_applied,
                                "topk_applied": topk_applied,
                                "resume": bool(runtime.get("resume")),
                                "resume_state_before": resume_state_before,
                                "resume_state_after": resume_state_final,
                                "resume_delta": _resume_delta(resume_state_before, resume_state_final),
                                "seed_wrapper_used": seed_wrapper_used,
                                "requires_topk": requires_topk,
                                "effective_config": {
                                    "llm": llm,
                                    "embedding": embedding,
                                    "runtime": {
                                        "workers": runtime.get("workers"),
                                        "resume": runtime.get("resume"),
                                        "seed": runtime.get("seed"),
                                        "save_every": runtime.get("save_every"),
                                        "limit": runtime.get("limit"),
                                    },
                                    "decode": decode,
                                    "topk": topk,
                                    "budget": budget,
                                    "retrieval_only": retrieval_only,
                                },
                                "fairness": {
                                    "enforce_budget": enforce_budget,
                                    "enforce_topk": enforce_topk,
                                    "require_prompt_hash": require_prompt_hash,
                                },
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
