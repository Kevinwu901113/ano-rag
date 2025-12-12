#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.musique.baselines.musique_utils import load_dataset


def _sha1(text: str) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MIRAGE-style doc_pool.json from MuSiQue jsonl/json")
    parser.add_argument("--dataset", required=True, help="MuSiQue dataset path (.jsonl/.json)")
    parser.add_argument(
        "--out",
        default="data/musique/doc_pool.json",
        help="Output doc pool path (JSON list of docs)",
    )
    parser.add_argument("--max-paragraphs", type=int, default=0, help="Optional cap per question (0=all)")
    args = parser.parse_args()

    rows = load_dataset(args.dataset)
    max_p = int(args.max_paragraphs or 0)

    docs: Dict[str, Dict[str, Any]] = {}
    for item in rows:
        paragraphs = item.get("paragraphs") or item.get("contexts") or item.get("passages") or []
        if not isinstance(paragraphs, list):
            continue
        if max_p > 0:
            paragraphs = paragraphs[:max_p]
        for idx, p in enumerate(paragraphs):
            if not isinstance(p, dict):
                continue
            title = str(p.get("title") or "").strip()
            text = str(
                p.get("paragraph_text")
                or p.get("text")
                or p.get("content")
                or p.get("paragraph")
                or p.get("para")
                or ""
            ).strip()
            if not text:
                continue
            key = _sha1(f"{title}\n{text}")[:16]
            raw_id = f"{title}__{key}" if title else key
            if raw_id in docs:
                continue
            docs[raw_id] = {
                "id": raw_id,
                "title": title,
                "paragraphs": [text],
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(list(docs.values()), handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(docs)} docs to {out_path}")


if __name__ == "__main__":
    main()

