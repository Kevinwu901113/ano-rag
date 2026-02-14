#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import (  # noqa: E402
    EMBED_BASE_URL,
    EMBED_MODEL,
    ensure_dataset,
    load_qa_with_docs,
    normalize_answer_for_eval,
    output_pred_path,
    resolve_llm_backend,
    write_pred_jsonl,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAPTOR_PKG = _REPO_ROOT / "RAPTOR" / "raptor"
_RAPTOR_INIT_SENTINEL = _RAPTOR_PKG / "raptor" / "__init__.py"
if not _RAPTOR_INIT_SENTINEL.exists():
    raise RuntimeError(
        f"RAPTOR submodule is missing or empty at {_RAPTOR_PKG}.\n"
        "Run in repo root:\n"
        "  git submodule sync --recursive\n"
        "  git submodule update --init --recursive RAPTOR/raptor"
    )
if str(_RAPTOR_PKG) not in sys.path:
    sys.path.insert(0, str(_RAPTOR_PKG))

try:
    from raptor import (  # noqa: E402
        BaseEmbeddingModel,
        BaseQAModel,
        BaseSummarizationModel,
        RetrievalAugmentation,
        RetrievalAugmentationConfig,
    )
except Exception as exc:
    raise RuntimeError(
        f"Failed to import RAPTOR package from {_RAPTOR_PKG}. "
        f"Root cause: {type(exc).__name__}: {exc}\n"
        "If submodule was not initialized, run:\n"
        "  git submodule sync --recursive\n"
        "  git submodule update --init --recursive RAPTOR/raptor\n"
        "If this is a dependency error, run inside baseline-raptor env and install requirements."
    ) from exc


class OpenAICompatEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, base_url: str, api_key: str, model: str, request_timeout: float):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(request_timeout),
        )
        self.model = model

    def create_embedding(self, text: str):
        text = str(text or "").replace("\n", " ").strip()
        return self.client.embeddings.create(
            input=[text],
            model=self.model,
        ).data[0].embedding


class OpenAICompatSummarizationModel(BaseSummarizationModel):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_input_chars: int,
        request_timeout: float,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(request_timeout),
        )
        self.model = model
        self.max_input_chars = int(max_input_chars)

    def summarize(self, context: str, max_tokens: int = 150):
        prompt = str(context or "")[: self.max_input_chars]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarize factual content concisely."},
                {
                    "role": "user",
                    "content": f"Summarize the following in <= {max_tokens} tokens:\n\n{prompt}",
                },
            ],
            temperature=0.0,
            max_tokens=max(32, int(max_tokens)),
        )
        return (response.choices[0].message.content or "").strip()


class OpenAICompatQAModel(BaseQAModel):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_input_chars: int,
        max_tokens: int,
        qa_prompt_mode: str,
        request_timeout: float,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(request_timeout),
        )
        self.model = model
        self.max_input_chars = int(max_input_chars)
        self.max_tokens = int(max_tokens)
        self.qa_prompt_mode = str(qa_prompt_mode)

    def answer_question(self, context: str, question: str):
        ctx = str(context or "")[: self.max_input_chars]
        if self.qa_prompt_mode == "answer_only":
            sys_prompt = (
                "You are a factual answerer. Use the provided context to answer the question.\n"
                "If the context is partial, answer based on the best available information or reasonable inference. "
                "Only say 'Insufficient evidence' if absolutely no relevant information is present.\n"
                "If the context supports a reasonable answer (even if partial), choose the best answer rather than 'Insufficient evidence'.\n"
                "For yes/no questions, answer exactly 'yes' or 'no' (lowercase).\n"
                "Return only the final short answer text. Do not output analysis or rationale."
            )
        else:
            sys_prompt = "Answer using only the provided context."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:",
                },
            ],
            temperature=0.0,
            max_tokens=max(16, self.max_tokens),
        )
        return normalize_answer_for_eval(response.choices[0].message.content or "")


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sha1_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _sanitize_qid(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:160]


def _docs_to_text(docs: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in docs:
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        if title:
            parts.append(f"### DOC {row.get('id') or ''} | Title: {title}\n{text}\n")
        else:
            parts.append(f"### DOC {row.get('id') or ''}\n{text}\n")
    return "\n".join(parts).strip()


def build_config(args: argparse.Namespace, llm_base_url: str, llm_api_key: str, llm_model: str):
    embedding_model = OpenAICompatEmbeddingModel(
        base_url=args.embed_base_url,
        api_key="EMPTY",
        model=args.embed_model,
        request_timeout=args.request_timeout,
    )
    summarizer = OpenAICompatSummarizationModel(
        base_url=llm_base_url,
        api_key=llm_api_key,
        model=llm_model,
        max_input_chars=args.summarizer_max_input_chars,
        request_timeout=args.request_timeout,
    )
    qa_model = OpenAICompatQAModel(
        base_url=llm_base_url,
        api_key=llm_api_key,
        model=llm_model,
        max_input_chars=args.qa_max_input_chars,
        max_tokens=args.answer_max_tokens,
        qa_prompt_mode=args.qa_prompt_mode,
        request_timeout=args.request_timeout,
    )
    return RetrievalAugmentationConfig(
        embedding_model=embedding_model,
        summarization_model=summarizer,
        qa_model=qa_model,
        tb_max_tokens=args.tb_max_tokens,
        tb_num_layers=args.tb_num_layers,
        tb_summarization_length=args.tb_summarization_length,
        tr_top_k=args.top_k,
    )


def _answer_one_question(
    args: argparse.Namespace,
    *,
    dataset: str,
    backend_name: str,
    config: RetrievalAugmentationConfig,
    qid: str,
    question: str,
    docs: List[Dict[str, Any]],
    workspace_root: Path,
) -> str:
    q_workspace = workspace_root / "raptor" / dataset / backend_name / _sanitize_qid(qid)
    q_workspace.mkdir(parents=True, exist_ok=True)
    tree_path = q_workspace / "tree.pkl"
    state_path = q_workspace / "index_state.json"

    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    docs_hash = _sha1_json(docs)
    corpus_text = _docs_to_text(docs)
    corpus_hash = _sha1_text(corpus_text)

    reuse_index = (
        (not args.rebuild_index)
        and tree_path.exists()
        and state_path.exists()
        and json.loads(state_path.read_text(encoding="utf-8")).get("docs_hash") == docs_hash
    )

    if reuse_index:
        ra = RetrievalAugmentation(config=config, tree=str(tree_path))
    else:
        ra = RetrievalAugmentation(config=config)
        ra.add_documents(corpus_text)
        with tree_path.open("wb") as handle:
            pickle.dump(ra.tree, handle)
        state = {
            "dataset": dataset,
            "llm_backend": backend_name,
            "qid": qid,
            "docs_hash": docs_hash,
            "corpus_hash": corpus_hash,
            "tree_path": str(tree_path),
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        return normalize_answer_for_eval(str(ra.answer_question(question) or "").strip())
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAPTOR QA baseline")
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

    parser.add_argument("--tb_max_tokens", type=int, default=120)
    parser.add_argument("--tb_num_layers", type=int, default=4)
    parser.add_argument("--tb_summarization_length", type=int, default=120)
    parser.add_argument("--summarizer_max_input_chars", type=int, default=24000)
    parser.add_argument("--qa_max_input_chars", type=int, default=32000)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--answer_max_tokens", type=int, default=96)
    parser.add_argument("--qa_prompt_mode", default="answer_only", choices=["answer_only", "default"])
    parser.add_argument("--request_timeout", type=float, default=60.0)

    args = parser.parse_args()

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
    pred_path = output_pred_path(output_root, "raptor", dataset, backend.name)

    config = build_config(
        args,
        llm_base_url=backend.base_url,
        llm_api_key=backend.api_key,
        llm_model=backend.model,
    )

    pred_rows: List[Dict[str, str]] = []
    for row in qa_rows:
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        docs = list(row.get("docs") or [])
        pred = _answer_one_question(
            args,
            dataset=dataset,
            backend_name=backend.name,
            config=config,
            qid=qid,
            question=question,
            docs=docs,
            workspace_root=workspace_root,
        )
        pred_rows.append({"id": qid, "pred": pred})

    write_pred_jsonl(pred_path, pred_rows)
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
