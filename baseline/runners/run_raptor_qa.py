#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
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
from retrieval_schema import build_raptor_ctxs  # noqa: E402

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
        temperature: float,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(request_timeout),
        )
        self.model = model
        self.max_input_chars = int(max_input_chars)
        self.temperature = float(temperature)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.elapsed_ms = 0.0

    def snapshot(self) -> Dict[str, float]:
        return {
            "calls": float(self.calls),
            "prompt_tokens": float(self.prompt_tokens),
            "completion_tokens": float(self.completion_tokens),
            "elapsed_ms": float(self.elapsed_ms),
        }

    def summarize(self, context: str, max_tokens: int = 150):
        prompt = str(context or "")[: self.max_input_chars]
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarize factual content concisely."},
                {
                    "role": "user",
                    "content": f"Summarize the following in <= {max_tokens} tokens:\n\n{prompt}",
                },
            ],
            temperature=self.temperature,
            max_tokens=max(32, int(max_tokens)),
        )
        self.calls += 1
        self.elapsed_ms += (time.perf_counter() - start) * 1000.0
        p_tok, c_tok = usage_prompt_completion(getattr(response, "usage", None))
        self.prompt_tokens += int(p_tok or 0)
        self.completion_tokens += int(c_tok or 0)
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
        temperature: float,
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
        self.temperature = float(temperature)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.elapsed_ms = 0.0

    def snapshot(self) -> Dict[str, float]:
        return {
            "calls": float(self.calls),
            "prompt_tokens": float(self.prompt_tokens),
            "completion_tokens": float(self.completion_tokens),
            "elapsed_ms": float(self.elapsed_ms),
        }

    def answer_question(self, context: str, question: str):
        ctx = str(context or "")[: self.max_input_chars]
        sys_prompt = load_aligned_reader_system_prompt()
        user_prompt = render_aligned_reader_prompt(
            question=question,
            evidence_rows=[{"id": "raptor_ctx", "title": "", "text": ctx, "rank": 1}],
        )
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=max(16, self.max_tokens),
        )
        self.calls += 1
        self.elapsed_ms += (time.perf_counter() - start) * 1000.0
        p_tok, c_tok = usage_prompt_completion(getattr(response, "usage", None))
        self.prompt_tokens += int(p_tok or 0)
        self.completion_tokens += int(c_tok or 0)
        return normalize_answer_for_eval(response.choices[0].message.content or "")


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sha1_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _sanitize_qid(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:160]


def _is_content_exists_risk(exc: Exception) -> bool:
    return "content exists risk" in str(exc or "").lower()


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


def _snapshot_delta(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
    keys = ("calls", "prompt_tokens", "completion_tokens", "elapsed_ms")
    out: Dict[str, float] = {}
    for key in keys:
        out[key] = float(after.get(key, 0.0) - before.get(key, 0.0))
    return out


def build_config(
    args: argparse.Namespace,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
) -> Tuple[RetrievalAugmentationConfig, OpenAICompatSummarizationModel, OpenAICompatQAModel]:
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
        temperature=args.temperature,
    )
    qa_model = OpenAICompatQAModel(
        base_url=llm_base_url,
        api_key=llm_api_key,
        model=llm_model,
        max_input_chars=args.qa_max_input_chars,
        max_tokens=args.answer_max_tokens,
        qa_prompt_mode=args.qa_prompt_mode,
        request_timeout=args.request_timeout,
        temperature=args.temperature,
    )
    config = RetrievalAugmentationConfig(
        embedding_model=embedding_model,
        summarization_model=summarizer,
        qa_model=qa_model,
        tb_max_tokens=args.tb_max_tokens,
        tb_num_layers=args.tb_num_layers,
        tb_summarization_length=args.tb_summarization_length,
        tr_top_k=args.top_k,
    )
    return config, summarizer, qa_model


def _answer_one_question(
    args: argparse.Namespace,
    *,
    dataset: str,
    backend_name: str,
    config: RetrievalAugmentationConfig,
    summarizer: OpenAICompatSummarizationModel,
    qa_model: OpenAICompatQAModel,
    qid: str,
    question: str,
    docs: List[Dict[str, Any]],
    workspace_root: Path,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    q_workspace = workspace_root / "raptor" / dataset / backend_name / _sanitize_qid(qid)
    q_workspace.mkdir(parents=True, exist_ok=True)
    tree_path = q_workspace / "tree.pkl"
    state_path = q_workspace / "index_state.json"

    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    docs_hash = _sha1_json(docs)
    corpus_text = _docs_to_text(docs)
    corpus_hash = _sha1_text(corpus_text)

    risk_retries = max(0, int(args.content_risk_retries))
    retry_wait_s = max(0.0, float(args.content_risk_retry_wait_sec))
    token_source = "api_usage"
    token_reason = None

    for risk_try in range(risk_retries + 1):
        try:
            sum_before = summarizer.snapshot()
            qa_before = qa_model.snapshot()
            reuse_index = (
                (not args.rebuild_index)
                and tree_path.exists()
                and state_path.exists()
                and json.loads(state_path.read_text(encoding="utf-8")).get("docs_hash") == docs_hash
            )
            index_time_ms = 0.0

            if reuse_index:
                ra = RetrievalAugmentation(config=config, tree=str(tree_path))
            else:
                index_start = time.perf_counter()
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
                index_time_ms = (time.perf_counter() - index_start) * 1000.0

            if args.retrieval_only:
                query_start = time.perf_counter()
                retrieval = ra.retrieve(
                    question,
                    top_k=args.top_k,
                    max_tokens=args.answer_max_tokens * 10,  # rough estimate for context
                    return_layer_information=True,
                )
                query_time_ms = (time.perf_counter() - query_start) * 1000.0
                if isinstance(retrieval, tuple) and len(retrieval) == 2:
                    context, layer_information = retrieval
                else:
                    context, layer_information = retrieval, []

                node_text_by_index: Dict[int, str] = {}
                try:
                    tree = getattr(ra, "tree", None)
                    all_nodes = getattr(tree, "all_nodes", {}) if tree is not None else {}
                    if isinstance(all_nodes, dict):
                        for key, node in all_nodes.items():
                            try:
                                node_index = int(key)
                            except (TypeError, ValueError):
                                continue
                            node_text_by_index[node_index] = str(getattr(node, "text", "") or "")
                except Exception:
                    node_text_by_index = {}

                ctxs = build_raptor_ctxs(
                    context_text=str(context or ""),
                    docs=docs,
                    top_k=args.top_k,
                    layer_information=layer_information if isinstance(layer_information, list) else [],
                    node_text_by_index=node_text_by_index,
                )
                sum_after = summarizer.snapshot()
                qa_after = qa_model.snapshot()
                sum_delta = _snapshot_delta(sum_after, sum_before)
                qa_delta = _snapshot_delta(qa_after, qa_before)
                prompt_tokens = int(sum_delta["prompt_tokens"] + qa_delta["prompt_tokens"])
                completion_tokens = int(sum_delta["completion_tokens"] + qa_delta["completion_tokens"])
                llm_calls = int(sum_delta["calls"] + qa_delta["calls"])
                if llm_calls == 0:
                    token_source = "unavailable"
                    token_reason = "raptor_no_usage_observed"
                    prompt_tokens_val = None
                    completion_tokens_val = None
                else:
                    prompt_tokens_val = prompt_tokens
                    completion_tokens_val = completion_tokens
                cost = build_cost_record(
                    index_time_ms=index_time_ms,
                    query_retrieval_ms=max(0.0, query_time_ms - qa_delta["elapsed_ms"]),
                    query_reader_ms=max(0.0, qa_delta["elapsed_ms"]),
                    llm_calls=llm_calls,
                    llm_retries=0,
                    prompt_tokens_total=prompt_tokens_val,
                    completion_tokens_total=completion_tokens_val,
                    token_source=token_source,
                    token_unavailable_reason=token_reason,
                ).to_dict()
                return "", ctxs, cost

            query_start = time.perf_counter()
            pred = normalize_answer_for_eval(str(ra.answer_question(question) or "").strip())
            query_time_ms = (time.perf_counter() - query_start) * 1000.0
            sum_after = summarizer.snapshot()
            qa_after = qa_model.snapshot()
            sum_delta = _snapshot_delta(sum_after, sum_before)
            qa_delta = _snapshot_delta(qa_after, qa_before)
            prompt_tokens = int(sum_delta["prompt_tokens"] + qa_delta["prompt_tokens"])
            completion_tokens = int(sum_delta["completion_tokens"] + qa_delta["completion_tokens"])
            llm_calls = int(sum_delta["calls"] + qa_delta["calls"])
            if llm_calls == 0:
                token_source = "unavailable"
                token_reason = "raptor_no_usage_observed"
                prompt_tokens_val = None
                completion_tokens_val = None
            else:
                prompt_tokens_val = prompt_tokens
                completion_tokens_val = completion_tokens
            cost = build_cost_record(
                index_time_ms=index_time_ms,
                query_retrieval_ms=max(0.0, query_time_ms - qa_delta["elapsed_ms"]),
                query_reader_ms=max(0.0, qa_delta["elapsed_ms"]),
                llm_calls=llm_calls,
                llm_retries=0,
                prompt_tokens_total=prompt_tokens_val,
                completion_tokens_total=completion_tokens_val,
                token_source=token_source,
                token_unavailable_reason=token_reason,
            ).to_dict()
            return pred, [], cost

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if _is_content_exists_risk(exc):
                if risk_try < risk_retries:
                    # Clean partial artifacts before retrying this question.
                    try:
                        if tree_path.exists():
                            tree_path.unlink()
                    except Exception:
                        pass
                    try:
                        if state_path.exists():
                            state_path.unlink()
                    except Exception:
                        pass
                    if retry_wait_s > 0:
                        time.sleep(retry_wait_s)
                    continue
            cost = build_cost_record(
                index_time_ms=0.0,
                query_retrieval_ms=0.0,
                query_reader_ms=0.0,
                llm_calls=0,
                llm_retries=0,
                prompt_tokens_total=None,
                completion_tokens_total=None,
                token_source="unavailable",
                token_unavailable_reason=str(exc)[:200],
            ).to_dict()
            return "", [], cost

    cost = build_cost_record(
        index_time_ms=0.0,
        query_retrieval_ms=0.0,
        query_reader_ms=0.0,
        llm_calls=0,
        llm_retries=0,
        prompt_tokens_total=None,
        completion_tokens_total=None,
        token_source="unavailable",
        token_unavailable_reason="unknown_error",
    ).to_dict()
    return "", [], cost


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

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--answer_max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--relrag_config", default=None, help="Optional RelRAG config path for default reader policy.")
    parser.add_argument("--qa_prompt_mode", default="answer_only", choices=["answer_only", "default"])
    parser.add_argument("--retrieval_only", action="store_true", help="Skip LLM generation, only output retrieval results.")
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
    args.answer_max_tokens = int(reader_params["answer_max_tokens"])
    args.temperature = float(reader_params["temperature"])

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
    if args.retrieval_only:
        pred_path = pred_path.with_name(pred_path.stem + "_retrieval.jsonl")

    config, summarizer, qa_model = build_config(
        args,
        llm_base_url=backend.base_url,
        llm_api_key=backend.api_key,
        llm_model=backend.model,
    )

    completed_ids = set()
    if pred_path.exists():
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    
    # Open file in append mode for incremental writing
    pred_handle = pred_path.open("a", encoding="utf-8")

    pred_rows: List[Dict[str, Any]] = []
    for row in qa_rows:
        qid = str(row.get("id") or "").strip()
        if qid in completed_ids:
            continue
            
        question = str(row.get("question") or "").strip()
        docs = list(row.get("docs") or [])
        try:
            pred, ctxs, cost = _answer_one_question(
                args,
                dataset=dataset,
                backend_name=backend.name,
                config=config,
                summarizer=summarizer,
                qa_model=qa_model,
                qid=qid,
                question=question,
                docs=docs,
                workspace_root=workspace_root,
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            pred = ""
            ctxs = []
            cost = build_cost_record(
                index_time_ms=0.0,
                query_retrieval_ms=0.0,
                query_reader_ms=0.0,
                llm_calls=0,
                llm_retries=0,
                prompt_tokens_total=None,
                completion_tokens_total=None,
                token_source="unavailable",
                token_unavailable_reason="exception",
            ).to_dict()
        
        out_row = {"id": qid, "pred": pred, "cost": cost}
        if ctxs:
            out_row["ctxs"] = ctxs

        # Write immediately
        pred_handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        pred_handle.flush()
        pred_rows.append(out_row)

    pred_handle.close()
    all_rows: List[Dict[str, Any]] = []
    if pred_path.exists():
        with pred_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_rows.append(json.loads(line))
                except Exception:
                    continue
    write_json(
        pred_path.parent / "cost_summary.json",
        summarize_cost_records(method="raptor", dataset=dataset, backend=backend.name, rows=all_rows),
    )
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
