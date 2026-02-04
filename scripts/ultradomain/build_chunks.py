from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

from scripts.ultradomain.common import (
    CHUNKS_DIR,
    DOMAIN_LABELS,
    RUN_META_DIR,
    assemble_budgeted_items,
    count_tokens,
    decode_tokens,
    encode_with_offsets,
    ensure_dirs,
    normalize_domain,
    now_iso,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

DEFAULT_BINS = [0, 200, 400, 600, 800, 1000, 1200]


def _iter_docs(domain: str) -> List[Dict[str, Any]]:
    docs_path = RUN_META_DIR / f"docs_{domain}.jsonl"
    docs = list(read_jsonl(docs_path))
    docs.sort(key=lambda item: str(item.get("doc_id")))
    return docs


def _slice_text_by_offsets(text: str, offsets: List[tuple[int, int]], start: int, end: int) -> str:
    if not offsets or start >= len(offsets) or end - 1 >= len(offsets):
        return ""
    start_char = offsets[start][0]
    end_char = offsets[end - 1][1]
    return text[start_char:end_char]


def _build_chunks_for_doc(doc: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    text = (doc.get("text") or "").strip()
    if not text:
        return []
    token_ids, offsets = encode_with_offsets(text)
    if not token_ids:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[Dict[str, Any]] = []
    for start in range(0, len(token_ids) - chunk_size + 1, step):
        end = start + chunk_size
        if end > len(token_ids):
            break
        if offsets:
            chunk_text = _slice_text_by_offsets(text, offsets, start, end)
        else:
            chunk_text = decode_tokens(token_ids[start:end])
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        chunk_id = f"{doc.get('doc_id')}::t{start}_{end}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc.get("doc_id"),
                "dataset": doc.get("dataset"),
                "text": chunk_text,
                "token_count": end - start,
                "start_token": start,
                "end_token": end,
            }
        )
    return chunks


def _histogram(lengths: List[int]) -> Dict[str, int]:
    bins = DEFAULT_BINS
    counts = {f"{bins[i]}-{bins[i+1]-1}": 0 for i in range(len(bins) - 1)}
    counts[f">={bins[-1]}"] = 0
    for length in lengths:
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                counts[f"{bins[i]}-{bins[i+1]-1}"] += 1
                placed = True
                break
        if not placed:
            counts[f">={bins[-1]}"] += 1
    return counts


def _describe_lengths(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "histogram": _histogram([]),
            "short_chunks": 0,
            "short_ratio": 0.0,
        }
    sorted_lengths = sorted(lengths)
    count = len(sorted_lengths)
    mean = sum(sorted_lengths) / count
    median = sorted_lengths[count // 2]
    p90 = sorted_lengths[int(math.floor(0.9 * (count - 1)))]
    p95 = sorted_lengths[int(math.floor(0.95 * (count - 1)))]
    short_chunks = sum(1 for x in sorted_lengths if x < 200)
    return {
        "count": count,
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": round(mean, 2),
        "median": float(median),
        "p90": float(p90),
        "p95": float(p95),
        "histogram": _histogram(sorted_lengths),
        "short_chunks": short_chunks,
        "short_ratio": round(short_chunks / count, 6) if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DeepSeek-tokenized chunks for UltraDomain.")
    parser.add_argument("--domain", default="all", help="mix|legal|all")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    ensure_dirs()
    domain_arg = args.domain.lower()
    domains = list(DOMAIN_LABELS.keys()) if domain_arg == "all" else [normalize_domain(domain_arg)]
    domains = [d for d in domains if d in DOMAIN_LABELS]
    if not domains:
        raise SystemExit("No valid domain provided.")

    stats_path = CHUNKS_DIR / "chunk_stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {"generated_at": now_iso(), "domains": {}}

    for domain in domains:
        docs = _iter_docs(domain)
        all_chunks: List[Dict[str, Any]] = []
        lengths: List[int] = []
        for doc in docs:
            chunks = _build_chunks_for_doc(doc, args.chunk_size, args.overlap)
            for chunk in chunks:
                all_chunks.append(chunk)
                lengths.append(int(chunk.get("token_count") or 0))
        out_path = CHUNKS_DIR / f"chunks_{domain}.jsonl"
        write_jsonl(out_path, all_chunks)
        stats["domains"][domain] = {
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "output": str(out_path),
            "tokenizer": "deepseek-ai/DeepSeek-V3.2",
            "generated_at": now_iso(),
            "stats": _describe_lengths(lengths),
        }

    write_json(stats_path, stats)
    print(f"Wrote chunks to {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
