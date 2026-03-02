#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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


def _sha1_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _detect_embedding_dim(embed_client: OpenAI, model: str) -> int:
    probe_vec = embed_client.embeddings.create(
        model=model,
        input="hello",
    ).data[0].embedding
    observed_dim = len(probe_vec)
    if observed_dim <= 0:
        raise RuntimeError(f"Invalid embedding dim ({observed_dim}) from model {model}.")
    return observed_dim


def _sanitize_qid(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:160]


def _title_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _is_content_exists_risk(exc: Exception) -> bool:
    return "content exists risk" in str(exc or "").lower()


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    if "429" in msg:
        return True
    keys = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "tpm",
        "rpm",
        "请求速率",
    )
    return any(k in msg for k in keys)


def _retry_backoff_seconds(attempt: int, base_sec: float, max_sec: float) -> float:
    base = max(0.05, float(base_sec))
    cap = max(base, float(max_sec))
    delay = min(cap, base * (2 ** max(0, int(attempt))))
    jitter = random.uniform(0.0, min(0.25, delay * 0.1))
    return delay + jitter


def _guess_title_and_body(content: str) -> Tuple[str, str]:
    lines = [ln.strip() for ln in str(content or "").splitlines() if ln.strip()]
    if not lines:
        return "", ""

    first = lines[0]
    title = first
    body_lines = lines[1:]

    if first.lower().startswith("title:"):
        title = first.split(":", 1)[1].strip()
    elif first.lower().startswith("### doc "):
        title_match = re.search(r"\|\s*title:\s*(.+)$", first, flags=re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = ""
        body_lines = lines[1:]

    body = "\n".join(body_lines).strip()
    return title, body


def _build_retrieved_context_rows(
    *,
    docs: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    doc_by_title: Dict[str, List[Dict[str, Any]]] = {}
    for doc in docs:
        key = _title_key(doc.get("title"))
        if key:
            doc_by_title.setdefault(key, []).append(doc)

    out: List[Dict[str, Any]] = []
    for rank, chunk in enumerate(chunks[: max(0, int(top_k))], start=1):
        content = str(chunk.get("content") or "").strip()
        title_guess, body = _guess_title_and_body(content)
        matched_doc = None
        candidates = doc_by_title.get(_title_key(title_guess), [])
        if candidates:
            matched_doc = candidates[0]

        if matched_doc is None and body:
            probe = body[:220].strip()
            if probe:
                for doc in docs:
                    doc_text = str(doc.get("text") or "").strip()
                    if probe in doc_text:
                        matched_doc = doc
                        break

        if matched_doc is not None:
            title = str(matched_doc.get("title") or title_guess).strip()
            text = body or str(matched_doc.get("text") or "").strip()
            doc_id = str(matched_doc.get("id") or f"qdoc_{rank:04d}")
            is_supporting = (
                None
                if matched_doc.get("is_supporting") is None
                else bool(matched_doc.get("is_supporting"))
            )
        else:
            title = title_guess
            text = content
            doc_id = f"chunk_{rank:04d}"
            is_supporting = None

        out.append(
            {
                "id": doc_id,
                "title": title,
                "text": text,
                "is_supporting": is_supporting,
                "rank": rank,
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "reference_id": str(chunk.get("reference_id") or ""),
                "file_path": str(chunk.get("file_path") or ""),
            }
        )
    return out


def _build_qa_context(rows: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in rows:
        rank = int(row.get("rank") or 0)
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        doc_id = str(row.get("id") or "").strip()
        if not text:
            continue
        parts.append(
            f"[Rank {rank} | id={doc_id}]\n"
            f"Title: {title}\n"
            f"Content: {text}"
        )
    return "\n\n".join(parts)


def _estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9]+", str(text or ""))))


def _answer_with_llm(
    llm_client: OpenAI,
    *,
    model: str,
    question: str,
    evidence_rows: List[Dict[str, Any]],
    answer_max_tokens: int,
    temperature: float,
    content_risk_retries: int,
    content_risk_retry_wait_sec: float,
    rate_limit_retries: int,
    rate_limit_backoff_base_sec: float,
    rate_limit_backoff_max_sec: float,
) -> Tuple[str, Dict[str, Any]]:
    system_prompt = load_aligned_reader_system_prompt()
    user_prompt = render_aligned_reader_prompt(question=question, evidence_rows=evidence_rows)

    risk_retries = max(0, int(content_risk_retries))
    retry_wait_s = max(0.0, float(content_risk_retry_wait_sec))
    rate_retries = max(0, int(rate_limit_retries))
    rate_backoff_base_s = max(0.05, float(rate_limit_backoff_base_sec))
    rate_backoff_max_s = max(rate_backoff_base_s, float(rate_limit_backoff_max_sec))
    llm_calls = 0
    llm_retries = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    token_source = "unavailable"
    token_reason = "usage_not_provided"
    content_left = risk_retries
    rate_left = rate_retries
    rate_attempt = 0
    while True:
        try:
            llm_calls += 1
            response = llm_client.chat.completions.create(
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
            return normalize_answer_for_eval(content), {
                "llm_calls": llm_calls,
                "llm_retries": llm_retries,
                "prompt_tokens_total": prompt_tokens_total,
                "completion_tokens_total": completion_tokens_total,
                "token_source": token_source,
                "token_unavailable_reason": token_reason,
            }
        except Exception as exc:
            if _is_content_exists_risk(exc) and content_left > 0:
                content_left -= 1
                llm_retries += 1
                if retry_wait_s > 0:
                    time.sleep(retry_wait_s)
                continue
            if _is_rate_limit_error(exc) and rate_left > 0:
                rate_left -= 1
                llm_retries += 1
                wait_s = _retry_backoff_seconds(
                    rate_attempt,
                    rate_backoff_base_s,
                    rate_backoff_max_s,
                )
                rate_attempt += 1
                time.sleep(wait_s)
                continue
            return "", {
                "llm_calls": llm_calls,
                "llm_retries": llm_retries,
                "prompt_tokens_total": None,
                "completion_tokens_total": None,
                "token_source": "unavailable",
                "token_unavailable_reason": str(exc)[:200],
            }


def _fallback_lightrag_prompt(query_mode: str) -> str:
    context_key = "content_data" if str(query_mode).strip().lower() == "naive" else "context_data"
    return (
        "You are a QA assistant.\n"
        "Answer the user question using only the given context.\n"
        "Return only the final short answer text.\n"
        "Do not output markdown, explanations, or references.\n"
        "If the context is insufficient, return: Insufficient evidence.\n\n"
        f"Context:\n{{{context_key}}}"
    )


async def _run_single_question(
    args: argparse.Namespace,
    *,
    dataset: str,
    backend_name: str,
    backend_model: str,
    embed_client: OpenAI,
    llm_client: OpenAI,
    embedding_dim: int,
    workspace_root: Path,
    row: Dict[str, Any],
    reader_params: Dict[str, Any],
) -> Dict[str, Any]:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import wrap_embedding_func_with_attrs

    qid = str(row.get("id") or "").strip()
    question = str(row.get("question") or "").strip()
    docs = list(row.get("chunks") or [])
    if not docs:
        docs = list(row.get("docs") or [])
    if args.max_docs > 0:
        docs = docs[: args.max_docs]
    risk_retries = max(0, int(args.content_risk_retries))
    retry_wait_s = max(0.0, float(args.content_risk_retry_wait_sec))

    if not qid or not question:
        return {"id": qid, "pred": "", "ctxs": [], "retrieved_context_topk": [], "cost": {}}

    q_workspace = workspace_root / "lightrag" / dataset / backend_name / _sanitize_qid(qid)
    state_path = q_workspace / "index_state.json"
    docs_hash = _sha1_json(docs)

    reuse_index = (
        (not args.rebuild_index)
        and state_path.exists()
        and json.loads(state_path.read_text(encoding="utf-8")).get("docs_hash") == docs_hash
    )

    if args.rebuild_index or not reuse_index:
        if q_workspace.exists():
            shutil.rmtree(q_workspace)
    q_workspace.mkdir(parents=True, exist_ok=True)
    index_time_ms = 0.0

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim,
        max_token_size=args.embed_max_tokens,
        model_name=args.embed_model,
    )
    async def embedding_func(texts: List[str]) -> np.ndarray:
        def _call() -> np.ndarray:
            response = embed_client.embeddings.create(
                model=args.embed_model,
                input=texts,
            )
            return np.asarray([item.embedding for item in response.data], dtype="float32")

        return await asyncio.to_thread(_call)

    async def llm_model_func(prompt: str, **kwargs: Any) -> str:
        system_prompt = kwargs.get("system_prompt")
        history_messages = kwargs.get("history_messages") or []
        if not isinstance(history_messages, list):
            history_messages = []

        requested_raw = kwargs.get("max_tokens")
        if requested_raw is None:
            requested_raw = (
                args.answer_max_tokens
                if args.answer_max_tokens is not None
                else reader_params.get("answer_max_tokens")
            )
        requested = int(requested_raw or 256)
        max_tokens = max(16, min(requested, args.extract_max_tokens))

        messages: List[Dict[str, str]] = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(role, str) and isinstance(content, str):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": str(prompt)})

        def _call() -> str:
            response = llm_client.chat.completions.create(
                model=backend_model,
                messages=messages,
                temperature=(
                    float(args.temperature)
                    if args.temperature is not None
                    else float(reader_params.get("temperature") or 0.0)
                ),
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()

        content_left = risk_retries
        rate_left = max(0, int(args.rate_limit_retries))
        rate_attempt = 0
        while True:
            try:
                return await asyncio.to_thread(_call)
            except Exception as exc:
                if _is_content_exists_risk(exc) and content_left > 0:
                    content_left -= 1
                    if retry_wait_s > 0:
                        await asyncio.sleep(retry_wait_s)
                    continue
                if _is_rate_limit_error(exc) and rate_left > 0:
                    rate_left -= 1
                    wait_s = _retry_backoff_seconds(
                        rate_attempt,
                        float(args.rate_limit_backoff_base_sec),
                        float(args.rate_limit_backoff_max_sec),
                    )
                    rate_attempt += 1
                    await asyncio.sleep(wait_s)
                    continue
                raise
        return ""

    rag = LightRAG(
        working_dir=str(q_workspace),
        llm_model_func=llm_model_func,
        llm_model_name=backend_model,
        embedding_func=embedding_func,
        llm_model_max_async=int(args.llm_model_max_async),
        embedding_func_max_async=int(args.embedding_func_max_async),
        embedding_batch_num=int(args.embedding_batch_num),
        max_parallel_insert=int(args.max_parallel_insert),
        chunk_token_size=int(args.chunk_token_size),
        chunk_overlap_token_size=int(args.chunk_overlap_token_size),
        entity_extract_max_gleaning=int(args.max_gleaning),
    )

    if hasattr(rag, "initialize_storages"):
        await rag.initialize_storages()

    try:
        if args.rebuild_index or not reuse_index:
            index_start = time.perf_counter()
            insert_payload: List[str] = []
            for doc in docs:
                title = str(doc.get("title") or "").strip()
                text = str(doc.get("text") or "").strip()
                if not text:
                    continue
                doc_text = f"{title}\n{text}" if title else text
                insert_payload.append(doc_text)
            if insert_payload:
                await rag.ainsert(insert_payload)

            state_payload = {
                "dataset": dataset,
                "llm_backend": backend_name,
                "qid": qid,
                "docs_hash": docs_hash,
                "num_docs": len(docs),
            }
            state_path.write_text(
                json.dumps(state_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            index_time_ms = (time.perf_counter() - index_start) * 1000.0

        response_type = "Short Answer" if args.qa_prompt_mode == "answer_only" else "Multiple Paragraphs"
        query_param = QueryParam(
            mode=args.query_mode,
            enable_rerank=False,
            top_k=int(args.top_k),
            response_type=response_type,
        )

        if hasattr(rag, "aquery_data"):
            retrieval_start = time.perf_counter()
            query_data = await rag.aquery_data(
                question,
                param=query_param,
            )
            data = query_data.get("data") if isinstance(query_data, dict) else {}
            chunks = list((data or {}).get("chunks") or [])
            retrieved_rows = _build_retrieved_context_rows(
                docs=docs,
                chunks=chunks,
                top_k=int(args.top_k),
            )
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
            elif retrieved_rows:
                reader_start = time.perf_counter()
                pred, llm_meta = _answer_with_llm(
                    llm_client,
                    model=backend_model,
                    question=question,
                    evidence_rows=retrieved_rows,
                    answer_max_tokens=int(reader_params["answer_max_tokens"]),
                    temperature=float(reader_params["temperature"]),
                    content_risk_retries=args.content_risk_retries,
                    content_risk_retry_wait_sec=args.content_risk_retry_wait_sec,
                    rate_limit_retries=args.rate_limit_retries,
                    rate_limit_backoff_base_sec=args.rate_limit_backoff_base_sec,
                    rate_limit_backoff_max_sec=args.rate_limit_backoff_max_sec,
                )
                query_reader_ms = (time.perf_counter() - reader_start) * 1000.0
            elif args.qa_prompt_mode == "answer_only":
                pred = "Insufficient evidence"
            else:
                pred = ""

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

            return {
                "id": qid,
                "pred": normalize_answer_for_eval(pred),
                "ctxs": retrieved_rows,
                "retrieved_context_topk": retrieved_rows,
                "retrieved_context_raw": retrieved_rows,
                "cost": cost,
            }

        retrieval_start = time.perf_counter()
        result = await rag.aquery(
            question,
            param=query_param,
            system_prompt=_fallback_lightrag_prompt(args.query_mode),
        )
        query_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
        pred = normalize_answer_for_eval(str(result or ""))
        cost = build_cost_record(
            index_time_ms=index_time_ms,
            query_retrieval_ms=query_retrieval_ms,
            query_reader_ms=0.0,
            llm_calls=0,
            llm_retries=0,
            prompt_tokens_total=None,
            completion_tokens_total=None,
            token_source="unavailable",
            token_unavailable_reason="lightrag_internal_usage_unavailable",
        ).to_dict()
        return {"id": qid, "pred": pred, "ctxs": [], "retrieved_context_topk": [], "cost": cost}
    except Exception:
        cost = build_cost_record(
            index_time_ms=index_time_ms,
            query_retrieval_ms=0.0,
            query_reader_ms=0.0,
            llm_calls=0,
            llm_retries=0,
            prompt_tokens_total=None,
            completion_tokens_total=None,
            token_source="unavailable",
            token_unavailable_reason="exception",
        ).to_dict()
        return {"id": qid, "pred": "", "ctxs": [], "retrieved_context_topk": [], "cost": cost}
    finally:
        if hasattr(rag, "finalize_storages"):
            await rag.finalize_storages()


async def run(args: argparse.Namespace) -> Path:
    from lightrag.utils import setup_logger

    dataset = ensure_dataset(args.dataset)
    backend = resolve_llm_backend(args.llm_backend)

    if args.llm_model_max_async <= 0:
        args.llm_model_max_async = 12 if backend.name == "deepseek" else 4
    if args.embedding_func_max_async <= 0:
        args.embedding_func_max_async = 16 if backend.name == "deepseek" else 8
    if args.embedding_batch_num <= 0:
        args.embedding_batch_num = 16 if backend.name == "deepseek" else 10
    if args.max_parallel_insert <= 0:
        args.max_parallel_insert = 6 if backend.name == "deepseek" else 2
    if args.question_workers <= 0:
        args.question_workers = 3 if backend.name == "deepseek" else 1

    print(
        "[info] lightrag concurrency "
        f"backend={backend.name} llm_model_max_async={args.llm_model_max_async} "
        f"embedding_func_max_async={args.embedding_func_max_async} "
        f"embedding_batch_num={args.embedding_batch_num} "
        f"max_parallel_insert={args.max_parallel_insert} "
        f"question_workers={args.question_workers}"
    )

    reader_params = resolve_effective_reader_params(
        dataset=dataset,
        backend=backend.name,
        answer_max_tokens=args.answer_max_tokens,
        temperature=args.temperature,
        config_path=args.relrag_config,
    )

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    workspace_root = Path(args.workspace_root)

    qa_path = data_root / dataset / "qa.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(
            f"Missing intermediate data for {dataset}. Run baseline/tools/build_intermediate.py first."
        )

    qa_rows = load_qa_with_docs(qa_path, limit=args.limit)
    pred_path = output_pred_path(output_root, "lightrag", dataset, backend.name)
    if args.retrieval_only:
        pred_path = pred_path.with_name(pred_path.stem + "_retrieval.jsonl")

    setup_logger("lightrag", level="INFO")

    embed_client = OpenAI(
        base_url=args.embed_base_url,
        api_key="EMPTY",
        timeout=float(args.request_timeout),
    )
    llm_client = OpenAI(
        base_url=backend.base_url,
        api_key=backend.api_key,
        timeout=float(args.request_timeout),
    )

    observed_dim = _detect_embedding_dim(embed_client, args.embed_model)
    if args.embedding_dim <= 0:
        embedding_dim = observed_dim
        print(
            f"[info] Auto-detected embedding dim={embedding_dim} "
            f"from model={args.embed_model} ({args.embed_base_url})"
        )
    else:
        embedding_dim = int(args.embedding_dim)
    if observed_dim != embedding_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: expected {embedding_dim}, observed {observed_dim}. "
            f"Use --embedding_dim {observed_dim} (or --embedding_dim 0 for auto-detect)."
        )

    total_questions = len(qa_rows)
    question_workers = max(1, int(args.question_workers))
    pred_rows: List[Dict[str, Any]] = []
    if question_workers == 1 or total_questions <= 1:
        for row in qa_rows:
            pred_row = await _run_single_question(
                args,
                dataset=dataset,
                backend_name=backend.name,
                backend_model=backend.model,
                embed_client=embed_client,
                llm_client=llm_client,
                embedding_dim=embedding_dim,
                workspace_root=workspace_root,
                row=row,
                reader_params=reader_params,
            )
            pred_rows.append(pred_row)
    else:
        sem = asyncio.Semaphore(question_workers)
        qid_lock_guard = asyncio.Lock()
        qid_locks: Dict[str, asyncio.Lock] = {}
        ordered_rows: List[Dict[str, Any] | None] = [None] * total_questions
        done = 0
        done_lock = asyncio.Lock()

        async def _run_one(index: int, row_data: Dict[str, Any]) -> None:
            nonlocal done
            qid = str(row_data.get("id") or "").strip()
            row_lock: asyncio.Lock | None = None
            if qid:
                async with qid_lock_guard:
                    row_lock = qid_locks.get(qid)
                    if row_lock is None:
                        row_lock = asyncio.Lock()
                        qid_locks[qid] = row_lock
            if row_lock is not None:
                async with row_lock:
                    async with sem:
                        result = await _run_single_question(
                            args,
                            dataset=dataset,
                            backend_name=backend.name,
                            backend_model=backend.model,
                            embed_client=embed_client,
                            llm_client=llm_client,
                            embedding_dim=embedding_dim,
                            workspace_root=workspace_root,
                            row=row_data,
                            reader_params=reader_params,
                        )
            else:
                async with sem:
                    result = await _run_single_question(
                        args,
                        dataset=dataset,
                        backend_name=backend.name,
                        backend_model=backend.model,
                        embed_client=embed_client,
                        llm_client=llm_client,
                        embedding_dim=embedding_dim,
                        workspace_root=workspace_root,
                        row=row_data,
                        reader_params=reader_params,
                    )
            ordered_rows[index] = result
            async with done_lock:
                done += 1
                if done == 1 or done % 10 == 0 or done == total_questions:
                    print(
                        f"[progress] dataset={dataset} backend={backend.name} "
                        f"questions_done={done}/{total_questions}"
                    )

        tasks = [asyncio.create_task(_run_one(i, row)) for i, row in enumerate(qa_rows)]
        await asyncio.gather(*tasks)
        pred_rows = [row for row in ordered_rows if row is not None]

    write_pred_jsonl(pred_path, pred_rows)
    write_json(
        pred_path.parent / "cost_summary.json",
        summarize_cost_records(method="lightrag", dataset=dataset, backend=backend.name, rows=pred_rows),
    )
    return pred_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightRAG QA baseline")
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
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=0,
        help="Embedding dimension. Use <=0 to auto-detect from embedding endpoint.",
    )
    parser.add_argument("--embed_max_tokens", type=int, default=8192)
    parser.add_argument("--request_timeout", type=float, default=60.0)

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--answer_max_tokens", type=int, default=None)
    parser.add_argument("--relrag_config", default=None, help="Optional RelRAG config path for default reader policy.")
    parser.add_argument("--extract_max_tokens", type=int, default=6144)
    parser.add_argument(
        "--llm_model_max_async",
        type=int,
        default=0,
        help="LightRAG llm_model_max_async (<=0 means backend-specific default).",
    )
    parser.add_argument(
        "--embedding_func_max_async",
        type=int,
        default=0,
        help="LightRAG embedding_func_max_async (<=0 means backend-specific default).",
    )
    parser.add_argument(
        "--embedding_batch_num",
        type=int,
        default=0,
        help="LightRAG embedding_batch_num (<=0 means backend-specific default).",
    )
    parser.add_argument(
        "--max_parallel_insert",
        type=int,
        default=0,
        help="LightRAG max_parallel_insert (<=0 means backend-specific default).",
    )
    parser.add_argument(
        "--question_workers",
        type=int,
        default=0,
        help="Number of questions to process concurrently (<=0 means backend-specific default).",
    )
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
    parser.add_argument(
        "--rate_limit_retries",
        type=int,
        default=8,
        help="Retry times for rate-limit errors (HTTP 429 / RPM / TPM).",
    )
    parser.add_argument(
        "--rate_limit_backoff_base_sec",
        type=float,
        default=1.5,
        help="Initial backoff seconds for rate-limit retries.",
    )
    parser.add_argument(
        "--rate_limit_backoff_max_sec",
        type=float,
        default=30.0,
        help="Max backoff seconds for rate-limit retries.",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--retrieval_only", action="store_true", help="Skip LLM answer generation.")
    parser.add_argument("--query_mode", default="hybrid")
    parser.add_argument("--chunk_token_size", type=int, default=600)
    parser.add_argument("--chunk_overlap_token_size", type=int, default=80)
    parser.add_argument("--max_gleaning", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = asyncio.run(run(args))
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
