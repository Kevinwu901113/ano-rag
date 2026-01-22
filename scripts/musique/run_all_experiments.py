import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

from relrag.config.config_loader import ConfigLoader, config as global_config
from relrag.config.dataset_config import get_dataset_config


RUN_MATRIX = [
    ("relrag", "bm25", "vllm"),
    ("relrag", "bm25", "openai"),
    ("relrag", "dense", "vllm"),
    ("relrag", "dense", "openai"),
    ("relrag", "hybrid", "vllm"),
    ("relrag", "hybrid", "openai"),
    ("purebaseline", "bm25", "vllm"),
    ("purebaseline", "bm25", "openai"),
    ("purebaseline", "dense", "vllm"),
    ("purebaseline", "dense", "openai"),
]


def _run_name(pipeline: str, retriever: str, reader: str) -> str:
    return f"{reader}_{pipeline}_{retriever}"


def _normalize_split(value: Optional[str]) -> str:
    raw = (value or "").strip().lower()
    if raw in {"dev", "val"}:
        return "dev"
    if raw in {"train", "dev", "test"}:
        return raw
    return "dev"


def _parse_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_only_skip(items: List[str]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        raw = item.strip()
        if raw.startswith("run_"):
            raw = raw[len("run_") :]
        normalized.append(raw)
    return normalized


def _parse_endpoint(endpoint: str) -> Dict[str, Any]:
    parsed = urlparse(endpoint)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    return {"scheme": scheme, "host": host, "port": port}


def _health_check(endpoint: str, timeout_sec: float = 2.0) -> bool:
    url = endpoint.rstrip("/") + "/models"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout_sec) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _wait_for_health(endpoint: str, retries: int = 30, sleep_sec: float = 2.0) -> bool:
    for _ in range(max(1, int(retries))):
        if _health_check(endpoint):
            return True
        time.sleep(sleep_sec)
    return False


def _start_vllm_dual_gpu(
    repo_root: Path,
    *,
    host: str,
    llm_port: int,
    embed_port: int,
    llm_model_id: Optional[str],
    embed_model_id: Optional[str],
    log_path: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(repo_root / "run_vllm_dual_gpu.py"),
        "--host",
        host,
        "--llm-port",
        str(llm_port),
        "--embed-port",
        str(embed_port),
    ]
    if llm_model_id:
        cmd.extend(["--llm-model", llm_model_id])
    if embed_model_id:
        cmd.extend(["--embed-model", embed_model_id])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle, cwd=str(repo_root))


def _tail_log(path: Path, limit: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    snippet = lines[-limit:] if lines else []
    return "\n".join(snippet)


def _read_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_run_command(
    repo_root: Path,
    *,
    run_dir: Path,
    pipeline: str,
    retriever: str,
    reader: str,
    config_path: Optional[str],
    data_path: str,
    split: str,
    limit: Optional[int],
    workers: Optional[int],
    vllm_endpoint: Optional[str],
) -> List[str]:
    script = "musique_entry.py" if pipeline == "relrag" else "musique_baseline_entry.py"
    cmd = [
        sys.executable,
        str(repo_root / script),
        "--run_dir",
        str(run_dir),
        "--data",
        data_path,
        "--retriever",
        retriever,
        "--reader",
        reader,
        "--split",
        split,
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if limit is not None and limit > 0:
        cmd.extend(["--limit", str(limit)])
    if workers is not None and workers > 0:
        cmd.extend(["--workers", str(workers)])
    if vllm_endpoint:
        cmd.extend(["--endpoint", vllm_endpoint])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MuSiQue experiment grid (relrag + pure baselines)")
    parser.add_argument("--config", help="Path to YAML config file (defaults to relrag/config/config.yaml)")
    parser.add_argument(
        "--data",
        default="/home/wjk/workplace/nq/ano-rag/data/musique_full_v1.0_dev_500.jsonl",
        help="Path to MuSiQue JSONL dataset",
    )
    parser.add_argument("--split", default="dev", help="Dataset split label")
    parser.add_argument(
        "--run_root",
        default="result/musique/dev_500",
        help="Output root for runs",
    )
    parser.add_argument("--n", type=int, default=0, help="Process only first N examples")
    parser.add_argument("--limit", type=int, help="Alias for --n")
    parser.add_argument("--workers", type=int, default=1, help="Per-run worker count")
    parser.add_argument("--openai_workers", type=int, help="Override workers for OpenAI runs")
    parser.add_argument("--vllm_workers", type=int, help="Override workers for vLLM runs")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs")
    parser.add_argument("--only", help="Comma-separated run names to execute")
    parser.add_argument("--skip", help="Comma-separated run names to skip")
    parser.add_argument("--openai", action="store_true", help="Run OpenAI queue")
    parser.add_argument("--vllm", action="store_true", help="Run vLLM queue")
    parser.add_argument("--parallel", action="store_true", help="Run OpenAI and vLLM queues in parallel")
    parser.add_argument("--no_parallel", action="store_false", dest="parallel")
    parser.set_defaults(parallel=True)
    parser.add_argument("--vllm_endpoint", help="Override vLLM endpoint (default from config)")
    parser.add_argument("--embed_endpoint", help="Override embedding endpoint (default from config)")
    parser.add_argument("--host", default="", help="Host override for auto-started servers")
    parser.add_argument("--vllm_port", type=int, default=0, help="Port override for vLLM server")
    parser.add_argument("--embed_port", type=int, default=0, help="Port override for embedding server")
    parser.add_argument("--llm_model_id", help="HF model id for vLLM server (optional)")
    parser.add_argument("--embed_model_id", help="HF model id for embedding server (optional)")
    parser.add_argument("--auto_start_servers", action="store_true", help="Auto-start vLLM/embedding if missing")
    parser.add_argument("--no_auto_start_servers", action="store_false", dest="auto_start_servers")
    parser.set_defaults(auto_start_servers=True)
    parser.add_argument("--shutdown_servers", action="store_true", help="Shutdown servers started by this script")
    parser.add_argument("--health_retries", type=int, default=30, help="Health check retries when starting servers")
    parser.add_argument("--health_sleep_sec", type=float, default=2.0, help="Sleep between health retries")

    args = parser.parse_args()
    repo_root = REPO_ROOT

    cfg = ConfigLoader(args.config).load_config() if args.config else global_config.load_config()
    dataset_cfg = get_dataset_config(cfg, "musique")
    split = _normalize_split(args.split)

    limit = args.limit if args.limit is not None else args.n
    if limit is None:
        limit = 0

    output_root = Path(args.run_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.openai and not args.vllm:
        args.openai = True
        args.vllm = True

    vllm_endpoint = args.vllm_endpoint or (cfg.get("vllm") or {}).get("endpoint")
    embed_endpoint = args.embed_endpoint or ((cfg.get("retriever") or {}).get("embedding") or {}).get("endpoint")
    if args.host or args.vllm_port:
        host = args.host or "127.0.0.1"
        port = args.vllm_port or _parse_endpoint(vllm_endpoint)["port"]
        vllm_endpoint = f"http://{host}:{port}/v1"
    if args.host or args.embed_port:
        host = args.host or "127.0.0.1"
        port = args.embed_port or _parse_endpoint(embed_endpoint)["port"]
        embed_endpoint = f"http://{host}:{port}/v1"

    only = set(_normalize_only_skip(_parse_list(args.only)))
    skip = set(_normalize_only_skip(_parse_list(args.skip)))
    run_defs: List[Dict[str, str]] = []
    for pipeline, retriever, reader in RUN_MATRIX:
        if reader == "openai" and not args.openai:
            continue
        if reader == "vllm" and not args.vllm:
            continue
        name = _run_name(pipeline, retriever, reader)
        if only and name not in only:
            continue
        if skip and name in skip:
            continue
        run_defs.append(
            {
                "name": name,
                "pipeline": pipeline,
                "retriever": retriever,
                "reader": reader,
            }
        )

    if not run_defs:
        raise ValueError("No runs selected (check --only/--skip filters).")

    needs_vllm = any(r["pipeline"] == "relrag" or r["reader"] == "vllm" for r in run_defs)
    needs_embed = any(r["retriever"] in {"dense", "hybrid"} for r in run_defs)

    server_proc: Optional[subprocess.Popen] = None
    server_info: Dict[str, Any] = {
        "vllm_endpoint": vllm_endpoint,
        "embed_endpoint": embed_endpoint,
        "vllm_status": "unknown",
        "embed_status": "unknown",
        "started_by_script": False,
        "log_path": None,
    }

    if args.auto_start_servers and (needs_vllm or needs_embed):
        vllm_ok = _health_check(vllm_endpoint) if needs_vllm else True
        embed_ok = _health_check(embed_endpoint) if needs_embed else True
        if vllm_ok:
            server_info["vllm_status"] = "reused"
        if embed_ok:
            server_info["embed_status"] = "reused"
        if (needs_vllm and not vllm_ok) or (needs_embed and not embed_ok):
            parsed = _parse_endpoint(vllm_endpoint)
            host = parsed["host"]
            llm_port = parsed["port"]
            embed_port = _parse_endpoint(embed_endpoint)["port"]
            log_path = output_root / "logs" / "vllm_dual_gpu.log"
            server_proc = _start_vllm_dual_gpu(
                repo_root,
                host=host,
                llm_port=llm_port,
                embed_port=embed_port,
                llm_model_id=args.llm_model_id,
                embed_model_id=args.embed_model_id,
                log_path=log_path,
            )
            server_info["started_by_script"] = True
            server_info["log_path"] = str(log_path)
            if needs_vllm:
                if not _wait_for_health(vllm_endpoint, retries=args.health_retries, sleep_sec=args.health_sleep_sec):
                    raise RuntimeError(f"vLLM server failed health check at {vllm_endpoint}")
                server_info["vllm_status"] = "started"
            if needs_embed:
                if not _wait_for_health(embed_endpoint, retries=args.health_retries, sleep_sec=args.health_sleep_sec):
                    raise RuntimeError(f"Embedding server failed health check at {embed_endpoint}")
                server_info["embed_status"] = "started"
    else:
        if needs_vllm and not _health_check(vllm_endpoint):
            raise RuntimeError(f"vLLM endpoint not healthy: {vllm_endpoint}")
        if needs_embed and not _health_check(embed_endpoint):
            raise RuntimeError(f"Embedding endpoint not healthy: {embed_endpoint}")

    results: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()
    queue_times: Dict[str, Dict[str, Any]] = {}

    def _run_queue(queue: List[Dict[str, str]], label: str) -> None:
        queue_times[label] = {"started_at": int(time.time())}
        for run_def in queue:
            name = run_def["name"]
            run_dir = output_root / f"run_{name}"
            log_dir = run_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "driver.log"
            completed_path = run_dir / "completed.json"
            if args.resume and completed_path.exists():
                with lock:
                    results[name] = {
                        "status": "skipped",
                        "run_dir": str(run_dir),
                        "log_path": str(log_path),
                    }
                continue

            workers = args.workers
            if run_def["reader"] == "openai" and args.openai_workers is not None:
                workers = args.openai_workers
            if run_def["reader"] == "vllm" and args.vllm_workers is not None:
                workers = args.vllm_workers

            cmd = _build_run_command(
                repo_root,
                run_dir=run_dir,
                pipeline=run_def["pipeline"],
                retriever=run_def["retriever"],
                reader=run_def["reader"],
                config_path=args.config,
                data_path=args.data,
                split=split,
                limit=limit if limit else None,
                workers=workers,
                vllm_endpoint=vllm_endpoint,
            )
            env = os.environ.copy()
            if embed_endpoint:
                env["EMB_ENDPOINT"] = embed_endpoint
            if vllm_endpoint and vllm_endpoint.rstrip("/") != "http://127.0.0.1:8000/v1":
                env["RELRAG_ALLOW_CUSTOM_LLM"] = "1"

            start_time = time.time()
            with log_path.open("w", encoding="utf-8") as log_handle:
                result = subprocess.run(cmd, cwd=str(repo_root), env=env, stdout=log_handle, stderr=log_handle)
            duration_sec = max(0.0, time.time() - start_time)
            metrics = _read_metrics(run_dir / "metrics.json")
            status = "ok" if result.returncode == 0 else "failed"
            error_snippet = None
            if status != "ok":
                error_snippet = _tail_log(log_path)
            with lock:
                results[name] = {
                    "status": status,
                    "pipeline": run_def["pipeline"],
                    "retriever": run_def["retriever"],
                    "reader": run_def["reader"],
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                    "duration_sec": round(duration_sec, 2),
                    "metrics": metrics,
                    "error": error_snippet,
                    "return_code": result.returncode,
                }
        queue_times[label]["ended_at"] = int(time.time())

    vllm_queue = [r for r in run_defs if r["reader"] == "vllm"]
    openai_queue = [r for r in run_defs if r["reader"] == "openai"]

    start_ts = int(time.time())
    parallel_used = bool(args.parallel and vllm_queue and openai_queue)
    if parallel_used:
        threads = [
            threading.Thread(target=_run_queue, args=(vllm_queue, "vllm")),
            threading.Thread(target=_run_queue, args=(openai_queue, "openai")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        _run_queue(vllm_queue + openai_queue, "serial")
    end_ts = int(time.time())

    if server_proc is not None and args.shutdown_servers:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except Exception:
            server_proc.kill()

    summary = {
        "run_root": str(output_root),
        "data": args.data,
        "split": split,
        "limit": limit,
        "started_at": start_ts,
        "ended_at": end_ts,
        "duration_sec": max(0, end_ts - start_ts),
        "parallel": parallel_used,
        "queue_times": queue_times,
        "config": args.config,
        "server": server_info,
        "runs": results,
    }
    _write_json(output_root / "summary.json", summary)

    lines = [
        "# MuSiQue Experiment Summary",
        "",
        f"- run_root: `{output_root}`",
        f"- split: `{split}`",
        f"- data: `{args.data}`",
        f"- parallel: `{parallel_used}`",
        f"- vllm_endpoint: `{vllm_endpoint}`",
        f"- embed_endpoint: `{embed_endpoint}`",
        "",
    ]
    if queue_times:
        lines.extend(
            [
                "| queue | started_at | ended_at |",
                "| --- | --- | --- |",
            ]
        )
        for label in sorted(queue_times.keys()):
            info = queue_times[label]
            lines.append(f"| {label} | {info.get('started_at', '-')} | {info.get('ended_at', '-')} |")
        lines.append("")

    lines.extend(
        [
            "| run | status | pipeline | retriever | reader | em | f1 | gold_sp_subset | top_k_hit | duplicate_rate | fallback_rate | count | duration_sec | run_dir |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name in sorted(results.keys()):
        info = results[name]
        metrics = (info.get("metrics") or {}).get("metrics") or {}
        lines.append(
            "| {name} | {status} | {pipeline} | {retriever} | {reader} | {em} | {f1} | {gold} | {topk} | {dup} | {fallback} | {count} | {dur} | {run_dir} |".format(
                name=name,
                status=info.get("status"),
                pipeline=info.get("pipeline"),
                retriever=info.get("retriever"),
                reader=info.get("reader"),
                em=metrics.get("em", "-"),
                f1=metrics.get("f1", "-"),
                gold=metrics.get("gold_sp_subset", "-"),
                topk=metrics.get("top_k_hit", "-"),
                dup=metrics.get("duplicate_rate", "-"),
                fallback=metrics.get("fallback_rate", "-"),
                count=(info.get("metrics") or {}).get("count", "-"),
                dur=info.get("duration_sec", "-"),
                run_dir=info.get("run_dir", "-"),
            )
        )
        if info.get("status") != "ok" and info.get("error"):
            lines.append("")
            lines.append(f"**{name} error**")
            lines.append("```")
            lines.append(info.get("error"))
            lines.append("```")

    lines.extend(
        [
            "",
            "## Notes",
            "- gold_sp uses paragraphs.is_supporting plus question_decomposition paragraph_support_idx when available.",
            "- answerable==false defaults to gold \"Insufficient evidence\"; metrics_alt reports always-answer scoring.",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = [name for name, info in results.items() if info.get("status") == "failed"]
    if failures:
        logger.error("Completed with failures: {}", ", ".join(sorted(failures)))
        sys.exit(1)
    logger.info("All experiments completed; summary written to {}", output_root / "summary.json")


if __name__ == "__main__":
    main()
