#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import (  # noqa: E402
    EMBED_BASE_URL,
    EMBED_MODEL,
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


def _sha1_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _sanitize_qid(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:160]


def _vector_norm(vec: List[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return -1.0
    dot = 0.0
    for av, bv in zip(a, b):
        dot += float(av) * float(bv)
    na = _vector_norm(a)
    nb = _vector_norm(b)
    if na <= 0.0 or nb <= 0.0:
        return -1.0
    return dot / (na * nb + 1e-12)


def _embed_texts(
    client: OpenAI,
    *,
    model: str,
    texts: List[str],
    batch_size: int,
) -> List[List[float]]:
    vectors: List[List[float]] = []
    if not texts:
        return vectors

    step = max(1, int(batch_size))
    for start in range(0, len(texts), step):
        batch = texts[start : start + step]
        response = client.embeddings.create(
            model=model,
            input=batch,
        )
        vectors.extend([list(item.embedding) for item in response.data])
    return vectors


def _cache_key_payload(
    *,
    qid: str,
    docs_hash: str,
    embed_model: str,
    embed_base_url: str,
) -> Dict[str, Any]:
    return {
        "qid": qid,
        "docs_hash": docs_hash,
        "embed_model": embed_model,
        "embed_base_url": embed_base_url,
    }


def _load_doc_embedding_cache(cache_path: Path, expect_key: Dict[str, Any]) -> List[List[float]] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("cache_key") != expect_key:
            return None
        vectors = payload.get("doc_embeddings") or []
        if not isinstance(vectors, list):
            return None
        out: List[List[float]] = []
        for vec in vectors:
            if not isinstance(vec, list):
                return None
            out.append([float(x) for x in vec])
        return out
    except Exception:
        return None


def _write_doc_embedding_cache(
    cache_path: Path,
    *,
    cache_key: Dict[str, Any],
    doc_embeddings: List[List[float]],
) -> None:
    payload = {
        "cache_key": cache_key,
        "num_docs": len(doc_embeddings),
        "doc_embeddings": doc_embeddings,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


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
    parser = argparse.ArgumentParser(description="Run dense-retrieval QA baseline")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wiki"])
    parser.add_argument("--llm_backend", required=True, choices=["qwen", "deepseek"])
    parser.add_argument("--data_root", default="baseline/data")
    parser.add_argument("--output_root", default="baseline/results")
    parser.add_argument("--workspace_root", default="baseline/workspaces")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--rebuild_index", action="store_true")

    parser.add_argument("--embed_base_url", default=EMBED_BASE_URL)
    parser.add_argument("--embed_model", default=EMBED_MODEL)
    parser.add_argument("--embed_batch_size", type=int, default=16)
    parser.add_argument("--cache_embeddings", action="store_true")
    parser.add_argument("--request_timeout", type=float, default=60.0)

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--answer_max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--relrag_config", default=None, help="Optional RelRAG config path for default reader policy.")
    parser.add_argument("--qa_prompt_mode", default="answer_only", choices=["answer_only", "default"])
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
    pred_path = output_pred_path(Path(args.output_root), "dense", dataset, backend.name)
    if args.retrieval_only:
        pred_path = pred_path.with_name(pred_path.stem + "_retrieval.jsonl")

    llm_client = OpenAI(
        base_url=backend.base_url,
        api_key=backend.api_key,
        timeout=float(args.request_timeout),
    )
    embed_client = OpenAI(
        base_url=args.embed_base_url,
        api_key="EMPTY",
        timeout=float(args.request_timeout),
    )

    pred_rows: List[Dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()

        docs = list(row.get("chunks") or [])
        if not docs:
            docs = list(row.get("docs") or [])
        if args.max_docs > 0:
            docs = docs[: args.max_docs]

        index_time_ms = 0.0
        retrieval_start = time.perf_counter()

        doc_payload = [
            {
                "id": str(doc.get("id") or ""),
                "title": str(doc.get("title") or ""),
                "text": str(doc.get("text") or ""),
            }
            for doc in docs
        ]
        docs_hash = _sha1_json(doc_payload)
        cache_key = _cache_key_payload(
            qid=qid,
            docs_hash=docs_hash,
            embed_model=args.embed_model,
            embed_base_url=args.embed_base_url,
        )

        q_workspace = (
            Path(args.workspace_root)
            / "dense"
            / dataset
            / backend.name
            / _sanitize_qid(qid)
        )
        cache_path = q_workspace / "doc_embeddings.json"

        doc_embeddings: List[List[float]] | None = None
        if args.cache_embeddings and (not args.rebuild_index):
            doc_embeddings = _load_doc_embedding_cache(cache_path, cache_key)

        if doc_embeddings is None:
            index_start = time.perf_counter()
            doc_texts: List[str] = []
            for doc in docs:
                title = str(doc.get("title") or "").strip()
                text = str(doc.get("text") or "").strip()
                doc_texts.append(f"{title}\n{text}" if title else text)
            doc_embeddings = _embed_texts(
                embed_client,
                model=args.embed_model,
                texts=doc_texts,
                batch_size=args.embed_batch_size,
            )
            index_time_ms += (time.perf_counter() - index_start) * 1000.0
            if args.cache_embeddings:
                _write_doc_embedding_cache(
                    cache_path,
                    cache_key=cache_key,
                    doc_embeddings=doc_embeddings,
                )

        question_vec = _embed_texts(
            embed_client,
            model=args.embed_model,
            texts=[question],
            batch_size=1,
        )[0]

        scored: List[Tuple[Dict[str, Any], float]] = []
        for doc, vec in zip(docs, doc_embeddings):
            scored.append((doc, _cosine_similarity(question_vec, vec)))
        scored.sort(key=lambda item: item[1], reverse=True)

        top_items = scored[: max(0, int(args.top_k))]
        ctxs = _build_ctxs(top_items)
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
            index_time_ms=index_time_ms,
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
        summarize_cost_records(method="dense", dataset=dataset, backend=backend.name, rows=pred_rows),
    )
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
