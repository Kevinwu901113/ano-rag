from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


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
    if extra:
        payload.update(extra)
    return payload
