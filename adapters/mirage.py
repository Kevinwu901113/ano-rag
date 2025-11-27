import json
import os
import re
from typing import Dict, Iterable, Iterator, List, Tuple

from utils import TextUtils


def _load_doc_pool(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        pool = json.load(handle)

    if isinstance(pool, dict):
        yield from pool.values()
    elif isinstance(pool, list):
        yield from pool
    else:
        raise ValueError(f"Unsupported doc_pool structure: {type(pool)!r}")


def _paragraphs_from_record(record: Dict) -> List[str]:
    paragraphs = record.get("paragraphs")
    if isinstance(paragraphs, list) and paragraphs:
        return [str(p) for p in paragraphs]

    contents = record.get("contents")
    if isinstance(contents, list) and contents:
        return [str(c) for c in contents]

    text = record.get("text")
    if isinstance(text, str):
        return [text]

    doc_chunk = record.get("doc_chunk")
    if isinstance(doc_chunk, str) and doc_chunk.strip():
        return [doc_chunk.strip()]

    doc_chunks = record.get("doc_chunks")
    if isinstance(doc_chunks, list) and doc_chunks:
        return [str(chunk) for chunk in doc_chunks if str(chunk).strip()]

    return []


def _make_chunks(
    doc_id: str,
    paragraphs: List[str],
    n_sent: int = 2,
    overlap: int = 0,
    doc_title: str | None = None,
) -> Iterator[Dict]:
    step = max(1, n_sent - overlap)
    chunk_idx = 0

    for para in paragraphs:
        sentences = TextUtils.split_by_sentence(para)
        if not sentences:
            continue

        cursor = 0
        while cursor < len(sentences):
            group = sentences[cursor : cursor + n_sent]
            if not group:
                break
            text = " ".join(group).strip()
            if text:
                yield {
                    "doc_id": doc_id,
                    "chunk_id": f"p{chunk_idx:04d}",
                    "text": text,
                    "meta": {"lang": "en", "doc_title": doc_title or doc_id},
                }
                chunk_idx += 1
            cursor += step


def iter_docs_and_chunks(
    data_dir: str,
    shard_idx: int = 0,
    shard_cnt: int = 1,
) -> Iterable[Tuple[Dict, Dict]]:
    doc_pool_path = os.path.join(data_dir, "doc_pool.json")
    if not os.path.exists(doc_pool_path):
        raise FileNotFoundError(f"MIRAGE doc_pool not found at {doc_pool_path}")

    records = list(_load_doc_pool(doc_pool_path))
    total = len(records)
    if shard_cnt > 1:
        size = max(1, total // shard_cnt)
        start = shard_idx * size
        end = total if shard_idx == shard_cnt - 1 else (shard_idx + 1) * size
        records = records[start:end]

    for record in records:
        mapped_id = str(record.get("mapped_id") or "").strip()
        doc_name = str(record.get("doc_name") or record.get("title") or "").strip()
        other_id = str(
            record.get("id")
            or record.get("doc_id")
            or record.get("_id")
            or ""
        ).strip()

        def _slug(s: str) -> str:
            if not s:
                return ""
            s = re.sub(r"\s+", "_", s.strip())
            s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
            return s.lower()

        if mapped_id and doc_name:
            raw_id = f"{_slug(doc_name)}__{mapped_id}"
        elif mapped_id:
            raw_id = mapped_id
        elif doc_name:
            raw_id = _slug(doc_name)
        else:
            raw_id = other_id
        if not raw_id:
            continue

        doc = {
            "doc_id": f"mirage/{raw_id}",
            "title": doc_name or (record.get("title") or record.get("doc_name") or ""),
            "meta": {"dataset": "mirage"},
        }

        paragraphs = _paragraphs_from_record(record)
        if not paragraphs:
            continue

        for chunk in _make_chunks(doc["doc_id"], paragraphs, n_sent=2, overlap=0, doc_title=doc.get("title")):
            yield doc, chunk
