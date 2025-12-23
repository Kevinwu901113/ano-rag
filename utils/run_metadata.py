from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.llm_client import get_all_profile_snapshots


def write_config_resolved(work_dir: Path, payload: Dict[str, Any]) -> Path:
    path = Path(work_dir) / "config.resolved.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_summary(work_dir: Path, payload: Dict[str, Any]) -> Path:
    path = Path(work_dir) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def append_run_log(work_dir: Path, message: str) -> Path:
    path = Path(work_dir) / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
    return path


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


def build_basic_config(
    *,
    dataset: str,
    model: str,
    endpoint: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    context_budget: Optional[int] = None,
    topk: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    decode: Optional[Dict[str, Any]] = None,
    embedding: Optional[Dict[str, Any]] = None,
    budgets: Optional[Dict[str, Any]] = None,
    llm_profile: Optional[str] = None,
    git_commit: Optional[str] = None,
    llm_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "endpoint": endpoint,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if context_budget is not None:
        payload["context_budget_tokens"] = context_budget
    if topk is not None:
        payload["topk"] = topk
    decode_payload = dict(decode or {})
    if "temperature" not in decode_payload and temperature is not None:
        decode_payload["temperature"] = temperature
    if "max_tokens" not in decode_payload and max_tokens is not None:
        decode_payload["max_tokens"] = max_tokens
    if "top_p" not in decode_payload:
        decode_payload["top_p"] = None
    if "repetition_penalty" not in decode_payload:
        decode_payload["repetition_penalty"] = None

    budgets_payload = dict(budgets or {})
    if "context_budget_tokens" not in budgets_payload and context_budget is not None:
        budgets_payload["context_budget_tokens"] = context_budget
    if "topk" not in budgets_payload and topk is not None:
        budgets_payload["topk"] = topk

    payload["decode"] = decode_payload
    payload["embedding"] = dict(embedding or {})
    payload["budgets"] = budgets_payload
    payload["llm_profile"] = llm_profile or os.environ.get("LLM_PROFILE") or "default"
    payload["llm_profiles"] = llm_profiles or get_all_profile_snapshots(
        generate_max_tokens=max_tokens,
        generate_temperature=temperature,
    )
    payload["git_commit"] = git_commit or _git_commit()
    if extra:
        payload.update(extra)
    return payload
