from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable

_LOCKS: Dict[Path, threading.Lock] = {}


def _get_lock(path: Path) -> threading.Lock:
    if path not in _LOCKS:
        _LOCKS[path] = threading.Lock()
    return _LOCKS[path]


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(record, ensure_ascii=False)
    lock = _get_lock(path)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(data + "\n")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
