#!/usr/bin/env python3
"""
Split MIRAGE dataset and doc_pool into multiple shards.

Usage example:
    python scripts/split_mirage_dataset.py \
        --dataset MIRAGE/mirage/dataset.json \
        --doc-pool MIRAGE/mirage/doc_pool.json \
        --num-shards 8 \
        --output-dir MIRAGE/mirage/splits

Each shard will be written to:
    MIRAGE/mirage/splits/shard_00/dataset.json
    MIRAGE/mirage/splits/shard_00/doc_pool.json
along with a shard manifest file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def shard_dataset(dataset: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    """Split dataset list into `num_shards` nearly equal parts while preserving order."""
    shard_size = math.ceil(len(dataset) / num_shards)
    shards: List[List[Dict[str, Any]]] = []
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(dataset))
        if start >= len(dataset):
            break
        shards.append(dataset[start:end])
    return shards


def filter_doc_pool(doc_pool: List[Dict[str, Any]], valid_ids: Iterable[str]) -> List[Dict[str, Any]]:
    valid_id_set = set(valid_ids)
    return [chunk for chunk in doc_pool if str(chunk.get("mapped_id")) in valid_id_set]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split MIRAGE dataset/doc_pool into shards")
    parser.add_argument("--dataset", required=True, help="Path to MIRAGE dataset JSON")
    parser.add_argument("--doc-pool", required=True, help="Path to MIRAGE doc_pool JSON")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of shards to create (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        default="MIRAGE/mirage/splits",
        help="Directory to place shard folders into",
    )
    parser.add_argument(
        "--prefix",
        default="shard",
        help="Prefix for shard folder names (default: shard)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    doc_pool_path = Path(args.doc_pool)
    output_dir = Path(args.output_dir)

    if not dataset_path.exists():
        parser.error(f"Dataset file not found: {dataset_path}")
    if not doc_pool_path.exists():
        parser.error(f"Doc pool file not found: {doc_pool_path}")
    if args.num_shards < 1:
        parser.error("num_shards must be >= 1")

    dataset = load_json(dataset_path)
    doc_pool = load_json(doc_pool_path)

    if not isinstance(dataset, list):
        parser.error("Dataset JSON must contain a list")
    if not isinstance(doc_pool, list):
        parser.error("Doc pool JSON must contain a list")

    shards = shard_dataset(dataset, args.num_shards)
    if len(shards) < args.num_shards:
        print(f"[warning] Only produced {len(shards)} shards (requested {args.num_shards}).")

    manifest: Dict[str, Any] = {
        "dataset_source": str(dataset_path),
        "doc_pool_source": str(doc_pool_path),
        "total_queries": len(dataset),
        "total_doc_chunks": len(doc_pool),
        "num_shards": len(shards),
        "shards": [],
    }

    for shard_idx, shard_data in enumerate(shards):
        shard_name = f"{args.prefix}_{shard_idx:02d}"
        shard_dir = output_dir / shard_name
        shard_dataset_path = shard_dir / "dataset.json"
        shard_doc_pool_path = shard_dir / "doc_pool.json"

        query_ids = [item.get("query_id") or item.get("id") for item in shard_data]
        shard_doc_pool = filter_doc_pool(doc_pool, query_ids)

        save_json(shard_data, shard_dataset_path)
        save_json(shard_doc_pool, shard_doc_pool_path)

        manifest["shards"].append(
            {
                "name": shard_name,
                "dataset_path": str(shard_dataset_path),
                "doc_pool_path": str(shard_doc_pool_path),
                "num_queries": len(shard_data),
                "num_doc_chunks": len(shard_doc_pool),
            }
        )

        print(
            f"[shard {shard_name}] queries={len(shard_data)} "
            f"doc_chunks={len(shard_doc_pool)} -> {shard_dir}"
        )

    manifest_path = output_dir / "manifest.json"
    save_json(manifest, manifest_path)
    print(f"\nShard manifest written to: {manifest_path}")
    print("Shards ready. Run each shard independently with e.g.:")
    print(
        "  python main_mirage.py "
        "MIRAGE/mirage/splits/shard_00/dataset.json "
        "mirage_results_shard_00.jsonl "
        "--doc-pool-file MIRAGE/mirage/splits/shard_00/doc_pool.json "
        "--work-dir result/shard_00 --new"
    )


if __name__ == "__main__":
    main()
