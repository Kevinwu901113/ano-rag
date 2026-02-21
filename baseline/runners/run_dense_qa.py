#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    ensure_dataset,
    load_qa_with_docs,
    output_pred_path,
    resolve_llm_backend,
    write_pred_jsonl,
)


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


def _build_context(items: List[Tuple[Dict[str, Any], float]]) -> str:
    parts: List[str] = []
    for rank, (doc, score) in enumerate(items, start=1):
        title = str(doc.get("title") or "").strip()
        text = str(doc.get("text") or "").strip()
        doc_id = str(doc.get("id") or f"qdoc_{rank:04d}")
        parts.append(
            f"[Rank {rank} | cosine={score:.6f} | id={doc_id}]\n"
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
    content_risk_retries: int,
    content_risk_retry_wait_sec: float,
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

    risk_retries = max(0, int(content_risk_retries))
    retry_wait_s = max(0.0, float(content_risk_retry_wait_sec))
    for risk_try in range(risk_retries + 1):
        try:
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
        except Exception as exc:
            if "content exists risk" in str(exc or "").lower() and risk_try < risk_retries:
                if retry_wait_s > 0:
                    time.sleep(retry_wait_s)
                continue
            raise
    return ""


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

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--answer_max_tokens", type=int, default=96)
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

    pred_rows: List[Dict[str, str]] = []
    for row in rows:
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        docs = list(row.get("docs") or [])
        if args.max_docs > 0:
            docs = docs[: args.max_docs]

        try:
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
            
            # Save retrieval context
            ctxs = []
            for rank, (doc, score) in enumerate(top_items, start=1):
                ctxs.append({
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "text": doc.get("text"),
                    "score": score,
                    "rank": rank
                })

            context = _build_context(top_items)
            if args.retrieval_only:
                pred = ""
            else:
                pred = _answer_with_llm(
                    llm_client,
                    model=backend.model,
                    question=question,
                    context=context,
                    answer_max_tokens=args.answer_max_tokens,
                    qa_prompt_mode=args.qa_prompt_mode,
                    content_risk_retries=args.content_risk_retries,
                    content_risk_retry_wait_sec=args.content_risk_retry_wait_sec,
                )
            
        except Exception:
            import traceback
            traceback.print_exc()
            pred = ""
            if 'ctxs' not in locals():
                ctxs = []

        pred_rows.append({"id": qid, "pred": pred, "ctxs": ctxs})

    write_pred_jsonl(pred_path, pred_rows)
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
