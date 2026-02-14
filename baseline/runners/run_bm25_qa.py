#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import (  # noqa: E402
    ensure_dataset,
    load_qa_with_docs,
    output_pred_path,
    resolve_llm_backend,
    write_pred_jsonl,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(str(text or ""))]


def _bm25_rank(
    question: str,
    docs: List[Dict[str, Any]],
    *,
    top_k: int,
    k1: float,
    b: float,
) -> List[Tuple[Dict[str, Any], float]]:
    query = _tokenize(question)
    if not query or not docs:
        return []

    doc_terms: List[List[str]] = []
    valid_docs: List[Dict[str, Any]] = []
    for doc in docs:
        terms = _tokenize(doc.get("title", "") + "\n" + doc.get("text", ""))
        if not terms:
            continue
        doc_terms.append(terms)
        valid_docs.append(doc)

    n_docs = len(valid_docs)
    if n_docs == 0:
        return []

    avgdl = sum(len(terms) for terms in doc_terms) / max(1, n_docs)

    df = Counter()
    for terms in doc_terms:
        for term in set(terms):
            df[term] += 1

    qtf = Counter(query)
    scored: List[Tuple[int, float]] = []
    for idx, terms in enumerate(doc_terms):
        tf = Counter(terms)
        dl = len(terms)
        score = 0.0
        for term, q_count in qtf.items():
            term_df = df.get(term, 0)
            if term_df <= 0:
                continue
            idf = math.log(1.0 + (n_docs - term_df + 0.5) / (term_df + 0.5))
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            denom = freq + k1 * (1 - b + b * (dl / (avgdl + 1e-12)))
            score += idf * ((freq * (k1 + 1)) / (denom + 1e-12)) * q_count
        scored.append((idx, float(score)))

    scored.sort(key=lambda item: item[1], reverse=True)
    out: List[Tuple[Dict[str, Any], float]] = []
    for idx, score in scored[: max(0, int(top_k))]:
        out.append((valid_docs[idx], score))
    return out


def _build_context(items: List[Tuple[Dict[str, Any], float]]) -> str:
    parts: List[str] = []
    for rank, (doc, score) in enumerate(items, start=1):
        title = str(doc.get("title") or "").strip()
        text = str(doc.get("text") or "").strip()
        doc_id = str(doc.get("id") or f"doc_{rank}")
        parts.append(
            f"[Rank {rank} | score={score:.4f} | id={doc_id}]\n"
            f"Title: {title}\n"
            f"Content: {text}"
        )
    return "\n\n".join(parts)


def _answer_with_llm(
    client: OpenAI,
    *,
    model: str,
    question: str,
    context: str,
    answer_max_tokens: int,
    qa_prompt_mode: str,
) -> str:
    if qa_prompt_mode == "answer_only":
        system_prompt = (
            "You are a factual answerer. Use the provided context to answer the question.\n"
            "If the context is partial, answer based on the best available information or reasonable inference. "
            "Only say 'Insufficient evidence' if absolutely no relevant information is present.\n"
            "If the context supports a reasonable answer (even if partial), choose the best answer rather than 'Insufficient evidence'.\n"
            "For yes/no questions, answer exactly 'yes' or 'no' (lowercase).\n"
            "Return only the final short answer text. Do not output analysis or rationale."
        )
    else:
        system_prompt = "Answer using only the provided context."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:",
            },
        ],
        temperature=0.0,
        max_tokens=max(16, int(answer_max_tokens)),
    )
    return (response.choices[0].message.content or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BM25 QA baseline")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wiki"])
    parser.add_argument("--llm_backend", required=True, choices=["qwen", "deepseek"])
    parser.add_argument("--data_root", default="baseline/data")
    parser.add_argument("--output_root", default="baseline/results")
    parser.add_argument("--workspace_root", default="baseline/workspaces")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--rebuild_index", action="store_true")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--answer_max_tokens", type=int, default=96)
    parser.add_argument("--qa_prompt_mode", default="answer_only", choices=["answer_only", "default"])
    parser.add_argument("--request_timeout", type=float, default=60.0)

    parser.add_argument("--bm25_k1", type=float, default=1.2)
    parser.add_argument("--bm25_b", type=float, default=0.75)
    args = parser.parse_args()

    dataset = ensure_dataset(args.dataset)
    backend = resolve_llm_backend(args.llm_backend)

    qa_path = Path(args.data_root) / dataset / "qa.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(
            f"Missing intermediate data for {dataset}. Run baseline/tools/build_intermediate.py first."
        )

    rows = load_qa_with_docs(qa_path, limit=args.limit)
    pred_path = output_pred_path(Path(args.output_root), "bm25", dataset, backend.name)

    llm_client = OpenAI(
        base_url=backend.base_url,
        api_key=backend.api_key,
        timeout=float(args.request_timeout),
    )

    pred_rows: List[Dict[str, str]] = []
    for row in rows:
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        docs = list(row.get("docs") or [])
        if args.max_docs > 0:
            docs = docs[: args.max_docs]

        try:
            ranked = _bm25_rank(
                question,
                docs,
                top_k=args.top_k,
                k1=args.bm25_k1,
                b=args.bm25_b,
            )
            context = _build_context(ranked)
            pred = _answer_with_llm(
                llm_client,
                model=backend.model,
                question=question,
                context=context,
                answer_max_tokens=args.answer_max_tokens,
                qa_prompt_mode=args.qa_prompt_mode,
            )
        except Exception:
            pred = ""

        pred_rows.append({"id": qid, "pred": pred})

    write_pred_jsonl(pred_path, pred_rows)
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
