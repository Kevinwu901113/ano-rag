from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_LOCKS: Dict[Path, threading.Lock] = {}


def _get_lock(path: Path) -> threading.Lock:
    if path not in _LOCKS:
        _LOCKS[path] = threading.Lock()
    return _LOCKS[path]


def _normalize_retrieved(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        normalized.append(
            {
                "rank": item.get("rank", idx),
                "score": item.get("score"),
                # Compatibility: some evaluators use `title` (HotpotQA), while others use `doc_id`.
                "title": item.get("title") or item.get("doc_id"),
                "doc_id": item.get("doc_id"),
                "sent_ids": item.get("sent_ids"),
                "passage_id": item.get("passage_id"),
            }
        )
    return normalized


def _normalize_context(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    context: List[Dict[str, Any]] = []
    for item in items:
        context.append(
            {
                "title": item.get("title") or item.get("doc_id"),
                "doc_id": item.get("doc_id"),
                "sent_ids": item.get("sent_ids"),
                "passage_id": item.get("passage_id"),
                "text": item.get("text"),
            }
        )
    return context


def log_retrieval(
    *,
    sample_id: str,
    dataset: str,
    run_name: str,
    retrieved: Iterable[Dict[str, Any]],
    topk: Optional[int] = None,
    final_context: Optional[Iterable[Dict[str, Any]]] = None,
    final_context_tokens: Optional[int] = None,
    context_budget_tokens: Optional[int] = None,
    log_dir: Optional[Path] = None,
) -> Path:
    """
    Append one retrieval record to retrieval.jsonl under log_dir.
    log_dir should point to the workspace directory for this run.
    """
    log_root = Path(log_dir) if log_dir else Path(".")
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "retrieval.jsonl"

    retrieved_list = _normalize_retrieved(retrieved)
    record = {
        "id": sample_id,
        "dataset": dataset,
        "run_name": run_name,
        "topk": topk if topk is not None else len(retrieved_list),
        "retrieved": retrieved_list,
    }
    if final_context is not None:
        record["final_context"] = _normalize_context(final_context)
    if final_context_tokens is not None:
        record["final_context_tokens"] = int(final_context_tokens)
    if context_budget_tokens is not None:
        record["context_budget_tokens"] = int(context_budget_tokens)

    data = json.dumps(record, ensure_ascii=False)
    lock = _get_lock(log_path)
    with lock:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(data + "\n")
    return log_path
