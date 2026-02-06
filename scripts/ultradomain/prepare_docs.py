from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from scripts.ultradomain.common import (
    DOMAIN_LABELS,
    OUTPUT_ROOT,
    RUN_META_DIR,
    ensure_dirs,
    normalize_domain,
    now_iso,
    write_json,
    write_jsonl,
)


def _load_dataset(dataset_id: str, split: str, cache_dir: str | None, config_name: str | None):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets is required. Install with: pip install datasets") from exc
    kwargs: Dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if config_name:
        kwargs["name"] = config_name
    return load_dataset(dataset_id, split=split, **kwargs)


def _doc_record(row: Dict[str, Any], domain: str, fallback_id: str) -> Dict[str, Any] | None:
    text = row.get("context") or ""
    if not isinstance(text, str) or not text.strip():
        return None
    doc_id = row.get("context_id") or row.get("_id") or fallback_id
    doc_id = str(doc_id)
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    title = meta.get("title") if isinstance(meta.get("title"), str) else None
    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "dataset": domain,
        "meta": meta or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UltraDomain docs for Mix/Legal.")
    parser.add_argument(
        "--dataset_id",
        default="TBD_ULTRADOMAIN_DATASET_ID",
        help="HuggingFace dataset id for UltraDomain (must be set explicitly).",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--config_name", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-domain limit")
    args = parser.parse_args()

    if str(args.dataset_id).startswith("TBD_") or "TBD" in str(args.dataset_id):
        raise SystemExit("Please provide a real --dataset_id (placeholder TBD_* is not allowed).")

    ensure_dirs()
    dataset = _load_dataset(args.dataset_id, args.split, args.cache_dir, args.config_name)

    docs_by_domain: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    dup_counts = defaultdict(int)
    counts = defaultdict(int)

    for idx, row in enumerate(dataset):
        label = row.get("label")
        domain = normalize_domain(label)
        if domain not in DOMAIN_LABELS:
            continue
        if args.limit and counts[domain] >= args.limit:
            continue
        record = _doc_record(row, domain, fallback_id=f"{domain}_{idx}")
        if record is None:
            continue
        doc_id = record["doc_id"]
        if doc_id in docs_by_domain[domain]:
            dup_counts[domain] += 1
            continue
        docs_by_domain[domain][doc_id] = record
        counts[domain] += 1

    stats: Dict[str, Any] = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "config_name": args.config_name,
        "generated_at": now_iso(),
        "domains": {},
    }

    for domain in DOMAIN_LABELS:
        docs = list(docs_by_domain[domain].values())
        docs.sort(key=lambda item: item["doc_id"])
        out_path = RUN_META_DIR / f"docs_{domain}.jsonl"
        write_jsonl(out_path, docs)
        lengths = [len(doc.get("text") or "") for doc in docs]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
        else:
            avg_len = min_len = max_len = 0
        stats["domains"][domain] = {
            "doc_count": len(docs),
            "avg_chars": round(avg_len, 2),
            "min_chars": min_len,
            "max_chars": max_len,
            "duplicate_docs": dup_counts.get(domain, 0),
            "output": str(out_path),
        }

    write_json(RUN_META_DIR / "doc_stats.json", stats)
    print(f"Wrote docs to {RUN_META_DIR}")


if __name__ == "__main__":
    main()
