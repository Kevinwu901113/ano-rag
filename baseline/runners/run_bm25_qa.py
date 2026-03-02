#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import (  # noqa: E402
    build_cost_record,
    ensure_dataset,
    load_aligned_reader_system_prompt,
    load_qa_with_docs,
    normalize_answer_for_eval,
    output_pred_path,
    render_aligned_reader_prompt,
    resolve_effective_reader_params,
    resolve_llm_backend,
    summarize_cost_records,
    usage_prompt_completion,
    write_json,
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


def _build_ctxs(items: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
    ctxs: List[Dict[str, Any]] = []
    for rank, (doc, score) in enumerate(items, start=1):
        ctxs.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "text": doc.get("text"),
                "score": float(score),
                "rank": rank,
                "is_supporting": doc.get("is_supporting"),
                "parent_doc_id": doc.get("parent_doc_id"),
                "chunk_idx": doc.get("chunk_idx"),
            }
        )
    return ctxs


def _estimate_tokens(text: str) -> int:
    return max(1, len(_TOKEN_RE.findall(str(text or ""))))


def _answer_with_llm(
    client: OpenAI,
    *,
    model: str,
    question: str,
    evidence_rows: List[Dict[str, Any]],
    temperature: float,
    answer_max_tokens: int,
    content_risk_retries: int,
    content_risk_retry_wait_sec: float,
) -> Tuple[str, Dict[str, Any]]:
    system_prompt = load_aligned_reader_system_prompt()
    user_prompt = render_aligned_reader_prompt(question=question, evidence_rows=evidence_rows)

    risk_retries = max(0, int(content_risk_retries))
    retry_wait_s = max(0.0, float(content_risk_retry_wait_sec))
    llm_calls = 0
    llm_retries = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    token_source = "unavailable"
    token_reason = "usage_not_provided"

    for risk_try in range(risk_retries + 1):
        try:
            llm_calls += 1
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(temperature),
                max_tokens=max(16, int(answer_max_tokens)),
            )
            content = (response.choices[0].message.content or "").strip()
            prompt_used, completion_used = usage_prompt_completion(getattr(response, "usage", None))
            if prompt_used is not None or completion_used is not None:
                token_source = "api_usage"
                token_reason = None
                prompt_tokens_total += int(prompt_used or 0)
                completion_tokens_total += int(completion_used or 0)
            else:
                token_source = "estimated"
                token_reason = "usage_not_provided"
                prompt_tokens_total += _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
                completion_tokens_total += _estimate_tokens(content)
            return content, {
                "llm_calls": llm_calls,
                "llm_retries": llm_retries,
                "prompt_tokens_total": prompt_tokens_total,
                "completion_tokens_total": completion_tokens_total,
                "token_source": token_source,
                "token_unavailable_reason": token_reason,
            }
        except Exception as exc:
            if "content exists risk" in str(exc or "").lower() and risk_try < risk_retries:
                llm_retries += 1
                if retry_wait_s > 0:
                    time.sleep(retry_wait_s)
                continue
            return "", {
                "llm_calls": llm_calls,
                "llm_retries": llm_retries,
                "prompt_tokens_total": None,
                "completion_tokens_total": None,
                "token_source": "unavailable",
                "token_unavailable_reason": str(exc)[:200],
            }

    return "", {
        "llm_calls": llm_calls,
        "llm_retries": llm_retries,
        "prompt_tokens_total": None,
        "completion_tokens_total": None,
        "token_source": "unavailable",
        "token_unavailable_reason": "unknown_error",
    }


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

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--answer_max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--relrag_config", default=None, help="Optional RelRAG config path for default reader policy.")
    parser.add_argument("--qa_prompt_mode", default="answer_only", choices=["answer_only", "default"])
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument(
        "--content_risk_retries",
        type=int,
        default=3,
        help="Question-level retries for backend moderation error: Content Exists Risk.",
    )
    parser.add_argument(
        "--content_risk_retry_wait_sec",
        type=float,
        default=1.0,
        help="Sleep time between content-risk retries.",
    )

    parser.add_argument("--bm25_k1", type=float, default=1.2)
    parser.add_argument("--bm25_b", type=float, default=0.75)
    parser.add_argument("--retrieval_only", action="store_true", help="Skip LLM generation, only output retrieval results.")
    args = parser.parse_args()

    dataset = ensure_dataset(args.dataset)
    backend = resolve_llm_backend(args.llm_backend)
    reader_params = resolve_effective_reader_params(
        dataset=dataset,
        backend=backend.name,
        answer_max_tokens=args.answer_max_tokens,
        temperature=args.temperature,
        config_path=args.relrag_config,
    )

    qa_path = Path(args.data_root) / dataset / "qa.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(
            f"Missing intermediate data for {dataset}. Run baseline/tools/build_intermediate.py first."
        )

    rows = load_qa_with_docs(qa_path, limit=args.limit)
    pred_path = output_pred_path(Path(args.output_root), "bm25", dataset, backend.name)
    if args.retrieval_only:
        pred_path = pred_path.with_name(pred_path.stem + "_retrieval.jsonl")

    llm_client = OpenAI(
        base_url=backend.base_url,
        api_key=backend.api_key,
        timeout=float(args.request_timeout),
    )

    pred_rows: List[Dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()

        chunks = list(row.get("chunks") or [])
        if not chunks:
            chunks = list(row.get("docs") or [])
        if args.max_docs > 0:
            chunks = chunks[: args.max_docs]

        retrieval_start = time.perf_counter()
        ranked = _bm25_rank(
            question,
            chunks,
            top_k=args.top_k,
            k1=args.bm25_k1,
            b=args.bm25_b,
        )
        ctxs = _build_ctxs(ranked)
        query_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        query_reader_ms = 0.0
        llm_meta = {
            "llm_calls": 0,
            "llm_retries": 0,
            "prompt_tokens_total": None,
            "completion_tokens_total": None,
            "token_source": "unavailable",
            "token_unavailable_reason": "retrieval_only",
        }

        if args.retrieval_only:
            pred = ""
        else:
            reader_start = time.perf_counter()
            pred_raw, llm_meta = _answer_with_llm(
                llm_client,
                model=backend.model,
                question=question,
                evidence_rows=ctxs,
                temperature=reader_params["temperature"],
                answer_max_tokens=reader_params["answer_max_tokens"],
                content_risk_retries=args.content_risk_retries,
                content_risk_retry_wait_sec=args.content_risk_retry_wait_sec,
            )
            query_reader_ms = (time.perf_counter() - reader_start) * 1000.0
            pred = normalize_answer_for_eval(pred_raw)

        cost = build_cost_record(
            index_time_ms=0.0,
            query_retrieval_ms=query_retrieval_ms,
            query_reader_ms=query_reader_ms,
            llm_calls=llm_meta.get("llm_calls", 0),
            llm_retries=llm_meta.get("llm_retries", 0),
            prompt_tokens_total=llm_meta.get("prompt_tokens_total"),
            completion_tokens_total=llm_meta.get("completion_tokens_total"),
            token_source=str(llm_meta.get("token_source") or "unavailable"),
            token_unavailable_reason=llm_meta.get("token_unavailable_reason"),
        ).to_dict()

        pred_rows.append({"id": qid, "pred": pred, "ctxs": ctxs, "cost": cost})

    write_pred_jsonl(pred_path, pred_rows)
    write_json(
        pred_path.parent / "cost_summary.json",
        summarize_cost_records(method="bm25", dataset=dataset, backend=backend.name, rows=pred_rows),
    )
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
