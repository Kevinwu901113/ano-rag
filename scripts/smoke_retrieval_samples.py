#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


DATASETS = ("hotpotqa", "musique", "2wiki")
METHODS = ("bm25", "dense", "lightrag", "raptor", "graphrag")
BACKENDS = ("qwen",)


def _resolve_pred_file(method: str, dataset: str, backend: str) -> Path | None:
    candidates = [
        Path(f"baseline/results/{method}/{dataset}/{backend}/pred_retrieval.jsonl"),
        Path(f"baseline/results/{method}/{dataset}/{backend}/pred.jsonl"),
        Path(f"baseline/results_retrieval_fix/{method}/{dataset}/{backend}/pred_retrieval.jsonl"),
        Path(f"baseline/results_retrieval_fix/{method}/{dataset}/{backend}/pred.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_first_jsonl_rows(path: Path, limit: int) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _extract_contexts(row: Dict) -> object:
    for key in ("ctxs", "retrieved_context_topk", "retrieved_context_raw", "retrieved_context"):
        if key in row:
            return row.get(key)
    return []


def _validate_structured_ctxs(rows: List[Dict]) -> Tuple[bool, str]:
    required = {"id", "title", "text", "rank"}
    for row in rows:
        ctxs = _extract_contexts(row)
        if not isinstance(ctxs, list) or not ctxs:
            return False, "missing/empty ctxs list"
        for ctx in ctxs:
            if not isinstance(ctx, dict):
                return False, "ctx item is not dict"
            if not required.issubset(ctx.keys()):
                return False, "ctx missing required keys"
    return True, "ok"


def _validate_text_ctxs(rows: List[Dict]) -> Tuple[bool, str]:
    for row in rows:
        ctxs = _extract_contexts(row)
        if not isinstance(ctxs, list):
            return False, "ctxs is not list"
        if ctxs and not all(isinstance(item, str) for item in ctxs):
            return False, "ctxs not pure text list"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check retrieval output schema on sampled rows.")
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args()

    sample_n = max(1, int(args.sample))
    failures: List[str] = []

    for method in METHODS:
        for dataset in DATASETS:
            for backend in BACKENDS:
                pred_file = _resolve_pred_file(method, dataset, backend)
                if pred_file is None:
                    continue
                metrics_file = pred_file.parent / "metrics_retrieval.json"
                if not metrics_file.exists():
                    failures.append(f"{method}/{dataset}/{backend}: missing metrics_retrieval.json")
                    continue

                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                recall_mode = str(metrics.get("recall_mode") or "")
                alignment_ok = bool(metrics.get("alignment_ok"))
                if not alignment_ok:
                    failures.append(f"{method}/{dataset}/{backend}: alignment_ok=false")
                    continue

                rows = _read_first_jsonl_rows(pred_file, sample_n)
                if not rows:
                    failures.append(f"{method}/{dataset}/{backend}: no sampled rows")
                    continue

                if recall_mode == "structured_ranked":
                    ok, message = _validate_structured_ctxs(rows)
                elif recall_mode == "text":
                    ok, message = _validate_text_ctxs(rows)
                else:
                    ok, message = False, f"unsupported recall_mode={recall_mode}"

                if not ok:
                    failures.append(f"{method}/{dataset}/{backend}: {message}")
                else:
                    print(f"[ok] {method}/{dataset}/{backend} mode={recall_mode} sample={len(rows)}")

    if failures:
        print("\n[fail] retrieval smoke checks failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\n[ok] retrieval smoke checks passed")


if __name__ == "__main__":
    main()
