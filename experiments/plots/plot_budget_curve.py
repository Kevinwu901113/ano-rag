#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _pick_metric_col(columns: List[str], metric: Optional[str], metric_col: Optional[str]) -> str:
    if metric_col:
        return metric_col
    if not metric:
        raise ValueError("Provide --metric or --metric-col")
    candidates = [c for c in columns if c.endswith(f".{metric}") or c == metric]
    if not candidates:
        raise ValueError(f"No metric column matches: {metric}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare budget curve data from aggregate CSV")
    parser.add_argument("--aggregate", required=True, help="Path to aggregate.csv")
    parser.add_argument("--metric", default=None, help="Metric suffix to match (e.g., AnswerF1)")
    parser.add_argument("--metric-col", default=None, help="Exact metric column name")
    parser.add_argument("--dataset", default=None, help="Filter by dataset id")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    agg_path = Path(args.aggregate)
    rows = _load_rows(agg_path)
    if not rows:
        raise SystemExit("No rows found in aggregate.csv")

    metric_col = _pick_metric_col(list(rows[0].keys()), args.metric, args.metric_col)

    out_rows = []
    for row in rows:
        if args.dataset and row.get("dataset") != args.dataset:
            continue
        status = row.get("status") or ""
        if status not in ("ok", "skipped", "skipped_missing_entry") and not status.startswith("skipped"):
            continue
        if not row.get("budget"):
            continue
        out_rows.append(
            {
                "dataset": row.get("dataset"),
                "method": row.get("method"),
                "llm": row.get("llm"),
                "budget": row.get("budget"),
                "variant": row.get("variant"),
                "metric": row.get(metric_col),
            }
        )

    output = Path(args.output) if args.output else agg_path.parent / "budget_curve.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "method", "llm", "budget", "variant", "metric"])
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
