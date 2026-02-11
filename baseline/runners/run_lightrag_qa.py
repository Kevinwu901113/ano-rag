#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import (  # noqa: E402
    EMBED_BASE_URL,
    EMBED_MODEL,
    ensure_dataset,
    load_json,
    load_qa,
    output_pred_path,
    resolve_llm_backend,
    write_pred_jsonl,
)


def _sha1_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


async def run(args: argparse.Namespace) -> Path:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import setup_logger, wrap_embedding_func_with_attrs

    dataset = ensure_dataset(args.dataset)
    backend = resolve_llm_backend(args.llm_backend)

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    corpus_path = data_root / dataset / "corpus.json"
    qa_path = data_root / dataset / "qa.jsonl"
    if not corpus_path.exists() or not qa_path.exists():
        raise FileNotFoundError(
            f"Missing intermediate data for {dataset}. Run baseline/tools/build_intermediate.py first."
        )

    corpus_rows = load_json(corpus_path)
    if args.max_docs > 0:
        corpus_rows = corpus_rows[: args.max_docs]
    qa_rows = load_qa(qa_path, limit=args.limit)

    pred_path = output_pred_path(output_root, "lightrag", dataset, backend.name)

    workspace = Path(args.workspace_root) / "lightrag" / dataset / backend.name
    state_path = workspace / "index_state.json"
    corpus_hash = _sha1_json(corpus_rows)

    reuse_index = (
        (not args.rebuild_index)
        and state_path.exists()
        and json.loads(state_path.read_text(encoding="utf-8")).get("corpus_hash") == corpus_hash
    )
    if args.rebuild_index or not reuse_index:
        if workspace.exists():
            shutil.rmtree(workspace)

    workspace.mkdir(parents=True, exist_ok=True)

    setup_logger("lightrag", level="INFO")

    embed_client = OpenAI(base_url=args.embed_base_url, api_key="EMPTY")
    llm_client = OpenAI(base_url=backend.base_url, api_key=backend.api_key)

    probe_vec = embed_client.embeddings.create(
        model=args.embed_model,
        input="hello",
    ).data[0].embedding
    observed_dim = len(probe_vec)
    if observed_dim != args.embedding_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: expected {args.embedding_dim}, observed {observed_dim}."
        )

    @wrap_embedding_func_with_attrs(
        embedding_dim=args.embedding_dim,
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

        requested = int(kwargs.get("max_tokens") or args.max_tokens)
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
                model=backend.model,
                messages=messages,
                temperature=float(args.temperature),
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_call)

    rag = LightRAG(
        working_dir=str(workspace),
        llm_model_func=llm_model_func,
        llm_model_name=backend.model,
        embedding_func=embedding_func,
        chunk_token_size=int(args.chunk_token_size),
        chunk_overlap_token_size=int(args.chunk_overlap_token_size),
        entity_extract_max_gleaning=int(args.max_gleaning),
    )

    if hasattr(rag, "initialize_storages"):
        await rag.initialize_storages()

    try:
        if args.rebuild_index or not reuse_index:
            for row in corpus_rows:
                title = str(row.get("title") or "").strip()
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                doc_text = f"{title}\n{text}" if title else text
                await rag.ainsert(doc_text)

            state_payload = {
                "dataset": dataset,
                "llm_backend": backend.name,
                "corpus_hash": corpus_hash,
                "num_docs": len(corpus_rows),
            }
            state_path.write_text(
                json.dumps(state_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        pred_rows: List[Dict[str, str]] = []
        for row in qa_rows:
            qid = str(row.get("id") or "").strip()
            question = str(row.get("question") or "").strip()
            try:
                result = await rag.aquery(
                    question,
                    param=QueryParam(mode=args.query_mode, enable_rerank=False),
                )
                pred = str(result or "").strip()
            except Exception:
                pred = ""
            pred_rows.append({"id": qid, "pred": pred})

        write_pred_jsonl(pred_path, pred_rows)
    finally:
        if hasattr(rag, "finalize_storages"):
            await rag.finalize_storages()

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
    parser.add_argument("--embedding_dim", type=int, default=4096)
    parser.add_argument("--embed_max_tokens", type=int, default=8192)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--extract_max_tokens", type=int, default=6144)
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
