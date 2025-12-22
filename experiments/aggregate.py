#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, val in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(next_prefix, val, out)
    else:
        out[prefix] = value


def _load_summary(run_root: Path) -> Dict[str, Any]:
    path = run_root / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found under {run_root}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for job in summary.get("jobs", []):
        row: Dict[str, Any] = {
            "dataset": job.get("dataset"),
            "method": job.get("method"),
            "llm": job.get("llm"),
            "budget": job.get("budget"),
            "variant": job.get("variant"),
            "status": job.get("status"),
            "duration_s": job.get("duration_s"),
            "workdir": job.get("workdir"),
            "eval_status": job.get("eval_status"),
        }
        metrics = job.get("metrics", {})
        flat: Dict[str, Any] = {}
        _flatten("metrics", metrics, flat)
        row.update(flat)
        rows.append(row)
    return rows


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run summary to CSV/JSON")
    parser.add_argument("--run-root", required=True, help="Path to result_relrag/<exp>/run_<timestamp>")
    parser.add_argument("--csv-out", default=None, help="Output CSV path (default: run_root/aggregate.csv)")
    parser.add_argument("--json-out", default=None, help="Output JSON path (default: run_root/aggregate.json)")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summary = _load_summary(run_root)
    rows = _rows_from_summary(summary)

    csv_out = Path(args.csv_out) if args.csv_out else run_root / "aggregate.csv"
    json_out = Path(args.json_out) if args.json_out else run_root / "aggregate.json"

    _write_csv(rows, csv_out)
    json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
