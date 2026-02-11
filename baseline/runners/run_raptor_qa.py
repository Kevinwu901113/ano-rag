#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAPTOR_PKG = _REPO_ROOT / "RAPTOR" / "raptor"
if str(_RAPTOR_PKG) not in sys.path:
    sys.path.insert(0, str(_RAPTOR_PKG))

from raptor import (  # noqa: E402
    BaseEmbeddingModel,
    BaseQAModel,
    BaseSummarizationModel,
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
)


class OpenAICompatEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def create_embedding(self, text: str):
        text = str(text or "").replace("\n", " ").strip()
        return self.client.embeddings.create(
            input=[text],
            model=self.model,
        ).data[0].embedding


class OpenAICompatSummarizationModel(BaseSummarizationModel):
    def __init__(self, base_url: str, api_key: str, model: str, max_input_chars: int):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
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
    def __init__(self, base_url: str, api_key: str, model: str, max_input_chars: int, max_tokens: int):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_input_chars = int(max_input_chars)
        self.max_tokens = int(max_tokens)

    def answer_question(self, context: str, question: str):
        ctx = str(context or "")[: self.max_input_chars]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer using only the provided context. Return only the final short answer.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:",
                },
            ],
            temperature=0.0,
            max_tokens=max(16, self.max_tokens),
        )
        return (response.choices[0].message.content or "").strip()


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_config(args: argparse.Namespace, llm_base_url: str, llm_api_key: str, llm_model: str):
    embedding_model = OpenAICompatEmbeddingModel(
        base_url=args.embed_base_url,
        api_key="EMPTY",
        model=args.embed_model,
    )
    summarizer = OpenAICompatSummarizationModel(
        base_url=llm_base_url,
        api_key=llm_api_key,
        model=llm_model,
        max_input_chars=args.summarizer_max_input_chars,
    )
    qa_model = OpenAICompatQAModel(
        base_url=llm_base_url,
        api_key=llm_api_key,
        model=llm_model,
        max_input_chars=args.qa_max_input_chars,
        max_tokens=args.qa_max_tokens,
    )
    return RetrievalAugmentationConfig(
        embedding_model=embedding_model,
        summarization_model=summarizer,
        qa_model=qa_model,
        tb_max_tokens=args.tb_max_tokens,
        tb_num_layers=args.tb_num_layers,
        tb_summarization_length=args.tb_summarization_length,
        tr_top_k=args.tr_top_k,
    )


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
    parser.add_argument("--tr_top_k", type=int, default=8)
    parser.add_argument("--summarizer_max_input_chars", type=int, default=24000)
    parser.add_argument("--qa_max_input_chars", type=int, default=32000)
    parser.add_argument("--qa_max_tokens", type=int, default=64)

    args = parser.parse_args()

    dataset = ensure_dataset(args.dataset)
    backend = resolve_llm_backend(args.llm_backend)

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    qa_path = data_root / dataset / "qa.jsonl"
    corpus_txt_path = data_root / dataset / "corpus.txt"
    corpus_json_path = data_root / dataset / "corpus.json"
    if not qa_path.exists() or not corpus_txt_path.exists() or not corpus_json_path.exists():
        raise FileNotFoundError(
            f"Missing intermediate data for {dataset}. Run baseline/tools/build_intermediate.py first."
        )

    qa_rows = load_qa(qa_path, limit=args.limit)
    if args.max_docs > 0:
        corpus_rows = load_json(corpus_json_path)[: args.max_docs]
        parts = []
        for row in corpus_rows:
            parts.append(f"### DOC {row['id']} | Title: {row['title']}\n{row['text']}\n")
        corpus_text = "\n".join(parts)
    else:
        corpus_text = corpus_txt_path.read_text(encoding="utf-8")

    pred_path = output_pred_path(output_root, "raptor", dataset, backend.name)
    workspace = Path(args.workspace_root) / "raptor" / dataset / backend.name
    workspace.mkdir(parents=True, exist_ok=True)
    tree_path = workspace / "tree.pkl"
    state_path = workspace / "index_state.json"

    corpus_hash = _sha1_text(corpus_text)
    reuse_index = (
        (not args.rebuild_index)
        and tree_path.exists()
        and state_path.exists()
        and json.loads(state_path.read_text(encoding="utf-8")).get("corpus_hash") == corpus_hash
    )

    config = build_config(
        args,
        llm_base_url=backend.base_url,
        llm_api_key=backend.api_key,
        llm_model=backend.model,
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
            "llm_backend": backend.name,
            "corpus_hash": corpus_hash,
            "tree_path": str(tree_path),
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_rows: List[Dict[str, str]] = []
    for row in qa_rows:
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        try:
            pred = str(ra.answer_question(question) or "").strip()
        except Exception:
            pred = ""
        pred_rows.append({"id": qid, "pred": pred})

    write_pred_jsonl(pred_path, pred_rows)
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
