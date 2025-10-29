#!/usr/bin/env python3
"""
Merge MIRAGE shard results (JSONL) back into a single file.

Example:
    python scripts/merge_mirage_results.py \
        --inputs result/shard_00/mirage_results_shard_00.jsonl \
                 result/shard_01/mirage_results_shard_01.jsonl \
        --output result/mirage_results_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge MIRAGE shard JSONL results")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of shard JSONL files to merge",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to merged JSONL output",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="Optional: expected number of unique query IDs (sanity check)",
    )
    return parser.parse_args()


def merge_files(inputs, output: Path, expect: int | None) -> None:
    combined: Dict[str, Dict[str, Any]] = {}

    for path_str in inputs:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Input JSONL not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc

                query_id = record.get("id")
                if not query_id:
                    raise ValueError(f"Missing 'id' field in {path}:{line_no}")
                if query_id in combined:
                    raise ValueError(f"Duplicate query id '{query_id}' from {path}")
                combined[query_id] = record

    if expect is not None and len(combined) != expect:
        raise ValueError(f"Expected {expect} records, merged {len(combined)} instead")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out_f:
        for record in combined.values():
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Merged {len(inputs)} files -> {output} "
        f"({len(combined)} unique query ids)"
    )


def main() -> None:
    args = parse_args()
    merge_files(args.inputs, Path(args.output), args.expect)


if __name__ == "__main__":
    main()
