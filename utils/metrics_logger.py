from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from loguru import logger


class MetricsLogger:
    """Lightweight per-query diagnostics recorder."""

    def __init__(self, out_path: Optional[str] = None, auto_flush: bool = False) -> None:
        self.records: List[Dict[str, Any]] = []
        self.out_path = Path(out_path) if out_path else None
        self.auto_flush = auto_flush

    def log_query(
        self,
        query: str,
        *,
        top1_source: Optional[str] = None,
        timings: Optional[Dict[str, float]] = None,
        scores: Optional[Dict[str, float]] = None,
        notes: Optional[List[str]] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "ts": time.time(),
            "query": query,
            "top1_source": top1_source,
            "timings": timings or {},
            "scores": scores or {},
            "notes": notes or [],
            "labels": labels or {},
        }
        self.records.append(payload)
        if self.auto_flush:
            self.flush()

    def flush(self) -> None:
        if not self.out_path or not self.records:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.clear()

    def summarize(self) -> Dict[str, Any]:
        if not self.records:
            return {}
        sources = {}
        for record in self.records:
            src = record.get("top1_source") or "unknown"
            sources[src] = sources.get(src, 0) + 1
        timings = {}
        for record in self.records:
            for key, value in (record.get("timings") or {}).items():
                timings.setdefault(key, []).append(value)
        avg_timings = {k: mean(v) for k, v in timings.items() if v}
        return {"queries": len(self.records), "top1_source_hist": sources, "avg_timings": avg_timings}

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception as exc:
            logger.warning("MetricsLogger flush failed: {}", exc)
