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
    load_qa_with_docs,
    output_pred_path,
    resolve_llm_backend,
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
) -> str:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import wrap_embedding_func_with_attrs

    qid = str(row.get("id") or "").strip()
    question = str(row.get("question") or "").strip()
    docs = list(row.get("docs") or [])
    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    if not qid or not question:
        return ""

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

        requested = int(kwargs.get("max_tokens") or args.answer_max_tokens)
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
                temperature=float(args.temperature),
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_call)

    rag = LightRAG(
        working_dir=str(q_workspace),
        llm_model_func=llm_model_func,
        llm_model_name=backend_model,
        embedding_func=embedding_func,
        chunk_token_size=int(args.chunk_token_size),
        chunk_overlap_token_size=int(args.chunk_overlap_token_size),
        entity_extract_max_gleaning=int(args.max_gleaning),
    )

    if hasattr(rag, "initialize_storages"):
        await rag.initialize_storages()

    try:
        if args.rebuild_index or not reuse_index:
            for doc in docs:
                title = str(doc.get("title") or "").strip()
                text = str(doc.get("text") or "").strip()
                if not text:
                    continue
                doc_text = f"{title}\n{text}" if title else text
                await rag.ainsert(doc_text)

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

        response_type = "Short Answer" if args.qa_prompt_mode == "answer_only" else "Multiple Paragraphs"
        result = await rag.aquery(
            question,
            param=QueryParam(
                mode=args.query_mode,
                enable_rerank=False,
                top_k=int(args.top_k),
                response_type=response_type,
            ),
        )
        return str(result or "").strip()
    except Exception:
        return ""
    finally:
        if hasattr(rag, "finalize_storages"):
            await rag.finalize_storages()


async def run(args: argparse.Namespace) -> Path:
    from lightrag.utils import setup_logger

    dataset = ensure_dataset(args.dataset)
    backend = resolve_llm_backend(args.llm_backend)

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

    pred_rows: List[Dict[str, str]] = []
    for row in qa_rows:
        qid = str(row.get("id") or "").strip()
        pred = await _run_single_question(
            args,
            dataset=dataset,
            backend_name=backend.name,
            backend_model=backend.model,
            embed_client=embed_client,
            llm_client=llm_client,
            embedding_dim=embedding_dim,
            workspace_root=workspace_root,
            row=row,
        )
        pred_rows.append({"id": qid, "pred": pred})

    write_pred_jsonl(pred_path, pred_rows)
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

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--answer_max_tokens", type=int, default=96)
    parser.add_argument("--extract_max_tokens", type=int, default=6144)
    parser.add_argument("--qa_prompt_mode", default="answer_only", choices=["answer_only", "default"])
    parser.add_argument("--top_k", type=int, default=5)
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
