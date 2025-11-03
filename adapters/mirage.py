import json
import os
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

    return []


def _make_chunks(doc_id: str, paragraphs: List[str], n_sent: int = 2, overlap: int = 0) -> Iterator[Dict]:
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
                    "meta": {"lang": "en"},
                }
                chunk_idx += 1
            cursor += step


def iter_docs_and_chunks(data_dir: str) -> Iterable[Tuple[Dict, Dict]]:
    doc_pool_path = os.path.join(data_dir, "doc_pool.json")
    if not os.path.exists(doc_pool_path):
        raise FileNotFoundError(f"MIRAGE doc_pool not found at {doc_pool_path}")

    for record in _load_doc_pool(doc_pool_path):
        raw_id = str(record.get("id") or record.get("doc_id") or record.get("_id") or "")
        if not raw_id:
            continue

        doc = {
            "doc_id": f"mirage/{raw_id}",
            "title": record.get("title") or "",
            "meta": {"dataset": "mirage"},
        }

        paragraphs = _paragraphs_from_record(record)
        if not paragraphs:
            continue

        for chunk in _make_chunks(doc["doc_id"], paragraphs, n_sent=2, overlap=0):
            yield doc, chunk
