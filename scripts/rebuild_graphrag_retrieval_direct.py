#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNERS_DIR = _REPO_ROOT / "baseline" / "runners"
if str(_RUNNERS_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNERS_DIR))

import run_graphrag_qa as graphrag_runner  # noqa: E402
from common import ensure_dataset, load_qa_with_docs, resolve_llm_backend  # noqa: E402


def _build_runner_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        max_docs=int(args.max_docs),
        rebuild_index=bool(args.rebuild_index),
        embed_base_url=str(args.embed_base_url),
        embed_model=str(args.embed_model),
        temperature=float(args.temperature),
        answer_max_tokens=int(args.answer_max_tokens),
        top_k=int(args.top_k),
        qa_prompt_mode=str(args.qa_prompt_mode),
        request_timeout=float(args.request_timeout),
        retrieval_only=True,
        index_method=str(args.index_method),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild GraphRAG retrieval-only outputs by directly calling _build_and_query_one."
    )
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wiki"])
    parser.add_argument("--llm_backend", default="qwen", choices=["qwen", "deepseek"])
    parser.add_argument("--data_root", default="baseline/data")
    parser.add_argument("--workspace_root", default="baseline/workspaces")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--graphrag_cli", default=None)
    parser.add_argument("--index_method", default="standard", choices=["standard", "fast"])
    parser.add_argument("--embed_base_url", default=graphrag_runner.EMBED_BASE_URL)
    parser.add_argument("--embed_model", default=graphrag_runner.EMBED_MODEL)
    parser.add_argument("--embedding_dim", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--answer_max_tokens", type=int, default=96)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--qa_prompt_mode", default="answer_only", choices=["answer_only", "default"])
    parser.add_argument("--request_timeout", type=float, default=60.0)
    args = parser.parse_args()

    dataset = ensure_dataset(args.dataset)
    backend = resolve_llm_backend(args.llm_backend)
    qa_path = Path(args.data_root) / dataset / "qa.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(f"Missing QA file: {qa_path}")

    qa_rows = load_qa_with_docs(qa_path, limit=int(args.limit))
    graphrag_cli = graphrag_runner._resolve_graphrag_cli(args.graphrag_cli)
    runner_args = _build_runner_args(args)

    observed_dim = graphrag_runner._detect_embedding_dim(
        args.embed_base_url,
        args.embed_model,
        request_timeout=args.request_timeout,
    )
    embedding_dim = int(args.embedding_dim) if int(args.embedding_dim) > 0 else observed_dim
    if embedding_dim != observed_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: expected {embedding_dim}, observed {observed_dim}"
        )

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = backend.api_key

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with_ctx = 0
    with output_file.open("w", encoding="utf-8") as writer:
        for row in qa_rows:
            qid = str(row.get("id") or "").strip()
            question = str(row.get("question") or "").strip()
            docs = list(row.get("docs") or [])

            ctxs: List[Dict[str, Any]] = []
            try:
                _, ctxs_any = graphrag_runner._build_and_query_one(
                    runner_args,
                    dataset=dataset,
                    backend_name=backend.name,
                    backend_model=backend.model,
                    backend_base_url=backend.base_url,
                    qid=qid,
                    question=question,
                    docs=docs,
                    workspace_root=Path(args.workspace_root),
                    graphrag_cli=graphrag_cli,
                    env=env,
                    embedding_dim=embedding_dim,
                )
                if isinstance(ctxs_any, list):
                    ctxs = [item for item in ctxs_any if isinstance(item, dict)]
            except Exception:
                ctxs = []

            out_row: Dict[str, Any] = {"id": qid, "pred": ""}
            if ctxs:
                out_row["ctxs"] = ctxs
                with_ctx += 1
            writer.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"[ok] wrote {output_file} rows={written} rows_with_ctx={with_ctx} "
        f"dataset={dataset} backend={backend.name}"
    )


if __name__ == "__main__":
    main()
