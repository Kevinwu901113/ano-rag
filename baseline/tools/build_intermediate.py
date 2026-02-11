#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_path: Path


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_title(title: Any) -> str:
    return re.sub(r"\s+", " ", str(title or "")).strip()


def normalize_text(text: Any) -> str:
    return str(text or "").strip()


def doc_key(title: str, text: str) -> str:
    payload = f"{title}\n{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def extract_from_context(context: Any) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        for title, sent_list in zip(titles, sentences):
            if not isinstance(sent_list, list):
                continue
            text = " ".join(str(s).strip() for s in sent_list if str(s).strip())
            docs.append((normalize_title(title), normalize_text(text)))
        return docs

    if isinstance(context, list):
        for item in context:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title = normalize_title(item[0])
            sent_list = item[1]
            if isinstance(sent_list, list):
                text = " ".join(str(s).strip() for s in sent_list if str(s).strip())
            else:
                text = str(sent_list)
            docs.append((title, normalize_text(text)))
    return docs


def extract_musique_docs(paragraphs: Any) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    if not isinstance(paragraphs, list):
        return docs
    for para in paragraphs:
        if not isinstance(para, dict):
            continue
        title = normalize_title(para.get("title"))
        text = normalize_text(para.get("paragraph_text"))
        docs.append((title, text))
    return docs


def build_dataset(spec: DatasetSpec, out_root: Path) -> None:
    records = list(iter_jsonl(spec.source_path))
    corpus_by_key: Dict[str, Dict[str, Any]] = {}
    qa_rows: List[Dict[str, Any]] = []

    raw_doc_mentions = 0
    docs_per_question: List[int] = []

    for rec in records:
        if spec.name in {"hotpotqa", "2wiki"}:
            qid = str(rec.get("_id") or "").strip()
            question = str(rec.get("question") or "").strip()
            answer = str(rec.get("answer") or "").strip()
            answerable = True
            answer_aliases: List[str] = []
            docs = extract_from_context(rec.get("context"))
        elif spec.name == "musique":
            qid = str(rec.get("id") or "").strip()
            question = str(rec.get("question") or "").strip()
            answer = str(rec.get("answer") or "").strip()
            answerable = bool(rec.get("answerable", True))
            answer_aliases = [
                str(item).strip()
                for item in (rec.get("answer_aliases") or [])
                if str(item).strip()
            ]
            docs = extract_musique_docs(rec.get("paragraphs"))
        else:
            raise ValueError(f"Unsupported dataset: {spec.name}")

        local_doc_ids: List[str] = []
        for title, text in docs:
            if not title or not text:
                continue
            raw_doc_mentions += 1
            key = doc_key(title, text)
            if key not in corpus_by_key:
                doc_id = f"doc_{len(corpus_by_key) + 1:06d}"
                corpus_by_key[key] = {
                    "id": doc_id,
                    "title": title,
                    "text": text,
                    "dataset": spec.name,
                }
            local_doc_ids.append(corpus_by_key[key]["id"])

        docs_per_question.append(len(set(local_doc_ids)))

        qa_rows.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "answerable": answerable,
                "answer_aliases": answer_aliases,
                "dataset": spec.name,
            }
        )

    corpus_rows = list(corpus_by_key.values())

    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_json_path = out_dir / "corpus.json"
    corpus_txt_path = out_dir / "corpus.txt"
    qa_jsonl_path = out_dir / "qa.jsonl"
    meta_path = out_dir / "meta.json"

    corpus_json_path.write_text(
        json.dumps(corpus_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with corpus_txt_path.open("w", encoding="utf-8") as handle:
        for row in corpus_rows:
            handle.write(f"### DOC {row['id']} | Title: {row['title']}\n")
            handle.write(f"{row['text']}\n\n")

    with qa_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in qa_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    unique_docs = len(corpus_rows)
    avg_docs = sum(docs_per_question) / len(docs_per_question) if docs_per_question else 0.0
    dedup_rate = 0.0
    if raw_doc_mentions > 0:
        dedup_rate = 1.0 - (unique_docs / raw_doc_mentions)

    meta = {
        "dataset": spec.name,
        "source_path": str(spec.source_path),
        "num_questions": len(qa_rows),
        "num_unique_docs": unique_docs,
        "avg_docs_per_question": round(avg_docs, 4),
        "raw_doc_mentions": raw_doc_mentions,
        "dedup_rate": round(dedup_rate, 6),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] {spec.name}: questions={len(qa_rows)}, docs={unique_docs}, dedup_rate={meta['dedup_rate']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline intermediate data layer")
    parser.add_argument("--out-root", default="baseline/data")
    parser.add_argument("--hotpot", default="data/hotpot_dev_distractor_500_jsonl.jsonl")
    parser.add_argument("--musique", default="data/musique_ans_v1.0_dev_500.jsonl")
    parser.add_argument("--twowiki", default="data/2wiki_dev_sample_500.jsonl")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec("hotpotqa", Path(args.hotpot)),
        DatasetSpec("musique", Path(args.musique)),
        DatasetSpec("2wiki", Path(args.twowiki)),
    ]
    for spec in specs:
        if not spec.source_path.exists():
            raise FileNotFoundError(f"Missing source file: {spec.source_path}")
        build_dataset(spec, out_root)


if __name__ == "__main__":
    main()
