from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Tuple

from relrag.utils import TextUtils
from scripts.ultradomain.common import (
    CHUNKS_DIR,
    DOMAIN_LABELS,
    RUN_META_DIR,
    TOKENIZER_ID,
    ensure_dirs,
    get_tokenizer,
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


def _split_sentences(text: str) -> List[str]:
    spans = TextUtils.split_with_spans(text or "")
    if not spans:
        cleaned = (text or "").strip()
        return [cleaned] if cleaned else []
    sentences: List[str] = []
    for span in spans:
        sentence = (span.get("text") or "").strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def _sentence_token_lengths(sentences: List[str]) -> Tuple[List[int], List[int]]:
    tokenizer = get_tokenizer()
    first_lens: List[int] = []
    follow_lens: List[int] = []
    for sent in sentences:
        first_lens.append(len(tokenizer.encode(sent, add_special_tokens=False)))
        follow_lens.append(len(tokenizer.encode(" " + sent, add_special_tokens=False)))
    return first_lens, follow_lens


def _doc_token_offsets(first_lens: List[int], follow_lens: List[int]) -> Tuple[List[int], List[int]]:
    starts: List[int] = []
    ends: List[int] = []
    cursor = 0
    for idx, first_len in enumerate(first_lens):
        starts.append(cursor)
        cursor += first_len if idx == 0 else follow_lens[idx]
        ends.append(cursor)
    return starts, ends


def _pack_chunks(
    doc: Dict[str, Any],
    sentences: List[str],
    first_lens: List[int],
    follow_lens: List[int],
    starts: List[int],
    ends: List[int],
    chunk_size: int,
    overlap_tokens: int,
) -> List[Dict[str, Any]]:
    tokenizer = get_tokenizer()
    chunks: List[Dict[str, Any]] = []
    n = len(sentences)
    i = 0
    while i < n:
        cur_tokens = 0
        j = i
        chunk_sents: List[str] = []
        while j < n:
            add_len = first_lens[j] if j == i else follow_lens[j]
            if chunk_sents and (cur_tokens + add_len > chunk_size):
                break
            if not chunk_sents and add_len > chunk_size:
                chunk_sents.append(sentences[j])
                cur_tokens += add_len
                j += 1
                break
            chunk_sents.append(sentences[j])
            cur_tokens += add_len
            j += 1

        if not chunk_sents:
            break

        chunk_text = " ".join(chunk_sents).strip()
        token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False)) if chunk_text else 0
        start_token = starts[i] if i < len(starts) else None
        end_token = ends[j - 1] if (j - 1) < len(ends) else None
        chunk_id = f"{doc.get('doc_id')}::t{start_token}_{end_token}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc.get("doc_id"),
                "dataset": doc.get("dataset"),
                "text": chunk_text,
                "token_count": token_count,
                "sent_start_idx": i,
                "sent_end_idx": j,
                "start_token": start_token,
                "end_token": end_token,
            }
        )

        if j >= n:
            break

        overlap = 0
        k = j - 1
        while k >= i and overlap < overlap_tokens:
            add_len = first_lens[k] if k == i else follow_lens[k]
            overlap += add_len
            k -= 1
        next_i = k + 1
        if next_i <= i:
            next_i = i + 1
        i = next_i
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


def _describe_token_lengths(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    sorted_lengths = sorted(lengths)
    count = len(sorted_lengths)
    mean = sum(sorted_lengths) / count
    median = sorted_lengths[count // 2]
    p90 = sorted_lengths[int(math.floor(0.9 * (count - 1)))]
    p95 = sorted_lengths[int(math.floor(0.95 * (count - 1)))]
    return {
        "count": count,
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": round(mean, 2),
        "median": float(median),
        "p90": float(p90),
        "p95": float(p95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sentence-aware DeepSeek chunks for UltraDomain.")
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
        doc_token_lengths: List[int] = []
        dropped_tail_tokens = 0
        for doc in docs:
            sentences = _split_sentences(doc.get("text") or "")
            if not sentences:
                continue
            doc_token_len = len(get_tokenizer().encode(doc.get("text") or "", add_special_tokens=False))
            doc_token_lengths.append(doc_token_len)
            first_lens, follow_lens = _sentence_token_lengths(sentences)
            starts, ends = _doc_token_offsets(first_lens, follow_lens)
            chunks = _pack_chunks(
                doc,
                sentences,
                first_lens,
                follow_lens,
                starts,
                ends,
                chunk_size=args.chunk_size,
                overlap_tokens=args.overlap,
            )
            for chunk in chunks:
                all_chunks.append(chunk)
                lengths.append(int(chunk.get("token_count") or 0))
        out_path = CHUNKS_DIR / f"chunks_{domain}.jsonl"
        write_jsonl(out_path, all_chunks)
        stats["domains"][domain] = {
            "chunk_size": args.chunk_size,
            "overlap_target_tokens": args.overlap,
            "output": str(out_path),
            "tokenizer": TOKENIZER_ID,
            "sentence_splitter": "TextUtils.split_with_spans",
            "generated_at": now_iso(),
            "stats": _describe_lengths(lengths),
            "doc_token_stats": _describe_token_lengths(doc_token_lengths),
            "dropped_tail_tokens": dropped_tail_tokens,
            "dropped_tail_ratio": 0.0,
        }

    write_json(stats_path, stats)
    print(f"Wrote chunks to {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
