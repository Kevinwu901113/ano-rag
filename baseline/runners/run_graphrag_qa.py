#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
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
from retrieval_schema import build_graphrag_ctxs_from_sources  # noqa: E402


ANSWER_ONLY_PROMPT = """You are a factual answerer. Use the provided context to answer the question.
If the context is partial, answer based on the best available information or reasonable inference. Only say "Insufficient evidence" if absolutely no relevant information is present.
If the context supports a reasonable answer (even if partial), choose the best answer rather than "Insufficient evidence".
For yes/no questions, answer exactly "yes" or "no" (lowercase).
Return only the final short answer text. Do not output analysis or rationale.
"""

RETRIEVAL_ONLY_PROMPT = """You are a retrieval debugger.
Please output the provided Context verbatim.
Do not summarize. Do not explain. Just output the context text.
"""


def _sha1_json(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _sanitize_qid(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:160]


def _resolve_graphrag_cli(explicit: str | None) -> str:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"graphrag cli not found: {p}")
        return str(p)
    detected = shutil.which("graphrag")
    if detected:
        return detected
    raise RuntimeError("graphrag CLI not found in PATH")


def _detect_embedding_dim(embed_base_url: str, embed_model: str, request_timeout: float) -> int:
    client = OpenAI(
        base_url=embed_base_url,
        api_key="EMPTY",
        timeout=float(request_timeout),
    )
    probe_vec = client.embeddings.create(model=embed_model, input="hello").data[0].embedding
    observed_dim = len(probe_vec)
    if observed_dim <= 0:
        raise RuntimeError(f"Invalid embedding dim ({observed_dim}) from model {embed_model}.")
    return observed_dim


def _run(cmd: List[str], *, env: Dict[str, str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        tail_out = "\n".join((result.stdout or "").splitlines()[-60:])
        tail_err = "\n".join((result.stderr or "").splitlines()[-60:])
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout tail:\n{tail_out}\n\n"
            f"stderr tail:\n{tail_err}"
        )
    return result


def _is_prune_empty_failure(exc: Exception) -> bool:
    text = str(exc)
    return (
        "Graph Pruning failed." in text
        and (
            "No entities remain" in text
            or "No relationships remain" in text
        )
    )


def _is_content_exists_risk(exc: Exception) -> bool:
    return "content exists risk" in str(exc or "").lower()


def _ensure_community_reports(workspace: Path) -> None:
    output_dir = workspace / "output"
    reports_path = output_dir / "community_reports.parquet"
    if reports_path.exists():
        return

    communities_path = output_dir / "communities.parquet"
    if not communities_path.exists():
        return

    communities = pd.read_parquet(communities_path)
    if communities.empty:
        reports = pd.DataFrame(
            columns=["id", "community", "level", "title", "summary", "full_content", "rank"]
        )
        reports.to_parquet(reports_path, index=False)
        return

    reports = pd.DataFrame()
    reports["id"] = communities["id"].astype(str)
    if "community" in communities.columns:
        reports["community"] = communities["community"].fillna(0).astype(int)
    else:
        reports["community"] = communities["human_readable_id"].fillna(0).astype(int)
    reports["level"] = communities.get("level", 0)
    reports["title"] = communities.get("title", "Community").fillna("Community")
    reports["summary"] = reports["title"]
    reports["full_content"] = reports["title"]
    reports["rank"] = 1.0
    if "period" in communities.columns:
        reports["period"] = communities["period"]
    if "size" in communities.columns:
        reports["size"] = communities["size"]
    reports.to_parquet(reports_path, index=False)


def _patch_settings(
    settings_path: Path,
    *,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    embed_dim: int,
    temperature: float,
    answer_max_tokens: int,
    top_k: int,
    qa_prompt_mode: str,
    relax_pruning: bool,
    request_timeout: float,
    retrieval_only: bool = False,
) -> None:
    cfg = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}

    cfg.setdefault("input", {})
    cfg["input"]["type"] = "json"
    cfg["input"]["file_pattern"] = ".*\\.json"
    cfg["input"]["id_column"] = "id"
    cfg["input"]["title_column"] = "title"
    cfg["input"]["text_column"] = "text"

    completion_models = cfg.setdefault("completion_models", {})
    if not completion_models:
        completion_models["default_completion_model"] = {}
    completion_id = cfg.get("local_search", {}).get("completion_model_id") or "default_completion_model"
    if completion_id not in completion_models:
        completion_models[completion_id] = {}
    completion_models[completion_id].update(
        {
            "model_provider": "openai",
            "model": llm_model,
            "auth_method": "api_key",
            "api_key": "${OPENAI_API_KEY}",
            "api_base": llm_base_url,
            "retry": {"type": "exponential_backoff"},
            "model_supports_json": False,
            "concurrent_requests": 8,
            "tokens_per_minute": 0,
            "requests_per_minute": 0,
            "max_retries": 5,
            "sleep_on_rate_limit_recommendation": True,
            "request_timeout": float(request_timeout),
            "api_version": None,
            "audience": None,
            "organization": None,
            "proxy": None,
            "encoding_model": "o200k_base",
            "call_args": {
                "temperature": float(temperature),
                "max_tokens": int(answer_max_tokens),
            },
        }
    )

    embedding_models = cfg.setdefault("embedding_models", {})
    if not embedding_models:
        embedding_models["default_embedding_model"] = {}
    embedding_id = cfg.get("local_search", {}).get("embedding_model_id") or "default_embedding_model"
    if embedding_id not in embedding_models:
        embedding_models[embedding_id] = {}
    embedding_models[embedding_id].update(
        {
            "model_provider": "openai",
            "model": embed_model,
            "auth_method": "api_key",
            "api_key": "${OPENAI_API_KEY}",
            "api_base": embed_base_url,
            "retry": {"type": "exponential_backoff"},
            "concurrent_requests": 8,
            "tokens_per_minute": 0,
            "requests_per_minute": 0,
            "max_retries": 5,
            "sleep_on_rate_limit_recommendation": True,
            "request_timeout": float(request_timeout),
            "api_version": None,
            "audience": None,
            "organization": None,
            "proxy": None,
            "encoding_model": "cl100k_base",
            "call_args": {
                "encoding_format": "float",
            },
        }
    )

    cfg.setdefault("embed_text", {})
    cfg["embed_text"]["embedding_model_id"] = embedding_id
    cfg["embed_text"]["names"] = [
        "entity_description",
        "text_unit_text",
    ]

    cfg.setdefault("vector_store", {})
    cfg["vector_store"]["index_schema"] = {
        "entity_description": {
            "index_name": "entity_description",
            "id_field": "id",
            "vector_field": "vector",
            "vector_size": int(embed_dim),
        },
        "text_unit_text": {
            "index_name": "text_unit_text",
            "id_field": "id",
            "vector_field": "vector",
            "vector_size": int(embed_dim),
        },
        "community_full_content": {
            "index_name": "community_full_content",
            "id_field": "id",
            "vector_field": "vector",
            "vector_size": int(embed_dim),
        },
    }

    cfg["workflows"] = [
        "load_input_documents",
        "create_base_text_units",
        "create_final_documents",
        "extract_graph_nlp",
        "prune_graph",
        "finalize_graph",
        "create_communities",
        "create_final_text_units",
        "generate_text_embeddings",
    ]

    cfg.setdefault("extract_graph", {})
    cfg["extract_graph"]["completion_model_id"] = completion_id

    cfg.setdefault("summarize_descriptions", {})
    cfg["summarize_descriptions"]["completion_model_id"] = completion_id

    cfg.setdefault("community_reports", {})
    cfg["community_reports"]["completion_model_id"] = completion_id

    cfg.setdefault("local_search", {})
    cfg["local_search"]["completion_model_id"] = completion_id
    cfg["local_search"]["embedding_model_id"] = embedding_id
    # Keep compatibility across GraphRAG versions:
    # some use `top_k_entities`, some use `top_k_mapped_entities`.
    cfg["local_search"]["top_k_entities"] = int(top_k)
    cfg["local_search"]["top_k_mapped_entities"] = int(top_k)
    cfg["local_search"]["top_k_relationships"] = int(top_k)
    
    if retrieval_only:
        cfg["local_search"]["prompt"] = "prompts/retrieval_only.txt"
        # Retrieval-only path only needs context_data for schema reconstruction.
        # Keep generation tiny to reduce per-query latency.
        cfg["local_search"]["llm_max_gen_tokens"] = 64
    elif qa_prompt_mode == "answer_only":
        cfg["local_search"]["prompt"] = "prompts/answer_only.txt"
    else:
        cfg["local_search"]["prompt"] = cfg["local_search"].get("prompt", "prompts/local_search_system_prompt.txt")

    if not retrieval_only:
        cfg["local_search"]["llm_max_gen_tokens"] = int(answer_max_tokens)

    cfg.setdefault("prune_graph", {})
    if relax_pruning:
        cfg["prune_graph"]["min_node_freq"] = 1
        cfg["prune_graph"]["min_node_degree"] = 0
        cfg["prune_graph"]["min_edge_weight_pct"] = 0.0
        cfg["prune_graph"]["remove_ego_nodes"] = False

    settings_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _prepare_workspace(
    workspace: Path,
    *,
    docs: List[Dict[str, Any]],
    graphrag_cli: str,
    env: Dict[str, str],
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    embed_dim: int,
    temperature: float,
    answer_max_tokens: int,
    top_k: int,
    qa_prompt_mode: str,
    relax_pruning: bool,
    request_timeout: float,
    retrieval_only: bool = False,
) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    _run(
        [
            graphrag_cli,
            "init",
            "--root",
            str(workspace),
            "--model",
            llm_model,
            "--embedding",
            embed_model,
        ],
        env=env,
    )

    prompts_dir = workspace / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "answer_only.txt").write_text(ANSWER_ONLY_PROMPT, encoding="utf-8")
    (prompts_dir / "retrieval_only.txt").write_text(RETRIEVAL_ONLY_PROMPT, encoding="utf-8")

    input_dir = workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "corpus.json").write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    settings_path = workspace / "settings.yaml"
    _patch_settings(
        settings_path,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_dim=embed_dim,
        temperature=temperature,
        answer_max_tokens=answer_max_tokens,
        top_k=top_k,
        qa_prompt_mode=qa_prompt_mode,
        relax_pruning=relax_pruning,
        request_timeout=request_timeout,
        retrieval_only=retrieval_only,
    )


def _index_workspace(
    workspace: Path,
    *,
    graphrag_cli: str,
    env: Dict[str, str],
    index_method: str,
) -> None:
    _run(
        [
            graphrag_cli,
            "index",
            "--root",
            str(workspace),
            "--method",
            index_method,
        ],
        env=env,
    )
    _ensure_community_reports(workspace)


def _build_docs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for row in rows:
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        docs.append(
            {
                "id": str(row.get("id") or f"qdoc_{len(docs)+1:04d}"),
                "title": title,
                "text": text,
            }
        )
    return docs


def _records_from_maybe_table(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        try:
            return pd.DataFrame(value).to_dict(orient="records")
        except Exception:
            return []
    return []


def _build_graphrag_ctxs(
    *,
    workspace: Path,
    context_data: Dict[str, Any],
    top_k: int,
    fallback_text: str,
) -> List[Dict[str, Any]]:
    sources_records = _records_from_maybe_table((context_data or {}).get("sources"))
    if not sources_records:
        if fallback_text.strip():
            return [
                {
                    "id": "graphrag_context_0001",
                    "title": "",
                    "text": fallback_text.strip(),
                    "rank": 1,
                    "provenance": {"source": "response_fallback"},
                }
            ]
        return []

    text_units_path = workspace / "output" / "text_units.parquet"
    documents_path = workspace / "output" / "documents.parquet"
    if not text_units_path.exists() or not documents_path.exists():
        if fallback_text.strip():
            return [
                {
                    "id": "graphrag_context_0001",
                    "title": "",
                    "text": fallback_text.strip(),
                    "rank": 1,
                    "provenance": {"source": "response_fallback"},
                }
            ]
        return []

    text_units = pd.read_parquet(text_units_path).to_dict(orient="records")
    documents = pd.read_parquet(documents_path).to_dict(orient="records")
    return build_graphrag_ctxs_from_sources(
        sources_records=sources_records,
        text_unit_records=text_units,
        document_records=documents,
        top_k=top_k,
    )


def _run_local_query(
    *,
    root_dir: Path,
    response_type: str,
    question: str,
) -> Tuple[str, Dict[str, Any]]:
    try:
        from graphrag.cli.query import run_local_search
    except Exception as exc:
        raise RuntimeError(
            "Failed to import GraphRAG Python query API (graphrag.cli.query.run_local_search)."
        ) from exc

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        response, context_data = run_local_search(
            data_dir=None,
            root_dir=root_dir,
            community_level=2,
            response_type=response_type,
            streaming=False,
            query=question,
            verbose=False,
    )
    return str(response or "").strip(), context_data if isinstance(context_data, dict) else {}


def _run_local_context_only(
    *,
    root_dir: Path,
    question: str,
) -> Tuple[str, Dict[str, Any]]:
    try:
        from graphrag.config.load_config import load_config
        import graphrag.api.query as query_api
    except Exception as exc:
        raise RuntimeError(
            "Failed to import GraphRAG Python context APIs for retrieval-only mode."
        ) from exc

    output_dir = root_dir / "output"
    communities = pd.read_parquet(output_dir / "communities.parquet")
    community_reports = pd.read_parquet(output_dir / "community_reports.parquet")
    text_units = pd.read_parquet(output_dir / "text_units.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")
    entities = pd.read_parquet(output_dir / "entities.parquet")
    covariates_path = output_dir / "covariates.parquet"
    covariates = pd.read_parquet(covariates_path) if covariates_path.exists() else None

    config = load_config(root_dir=root_dir, cli_overrides={})
    community_level = 2
    description_embedding_store = query_api.get_embedding_store(
        config=config.vector_store,
        embedding_name=query_api.entity_description_embedding,
    )
    entities_ = query_api.read_indexer_entities(entities, communities, community_level)
    covariates_ = query_api.read_indexer_covariates(covariates) if covariates is not None else []

    search_engine = query_api.get_local_search_engine(
        config=config,
        reports=query_api.read_indexer_reports(community_reports, communities, community_level),
        text_units=query_api.read_indexer_text_units(text_units),
        entities=entities_,
        relationships=query_api.read_indexer_relationships(relationships),
        covariates={"claims": covariates_},
        description_embedding_store=description_embedding_store,
        response_type="Short Answer",
        system_prompt=RETRIEVAL_ONLY_PROMPT,
        callbacks=[],
    )

    context_result = search_engine.context_builder.build_context(
        query=question,
        **(getattr(search_engine, "context_builder_params", {}) or {}),
    )
    context_text = str(getattr(context_result, "context_chunks", "") or "")
    context_data = getattr(context_result, "context_records", {})
    if not isinstance(context_data, dict):
        context_data = {}
    return context_text, context_data


def _build_and_query_one(
    args: argparse.Namespace,
    *,
    dataset: str,
    backend_name: str,
    backend_model: str,
    backend_base_url: str,
    qid: str,
    question: str,
    docs: List[Dict[str, Any]],
    workspace_root: Path,
    graphrag_cli: str,
    env: Dict[str, str],
    embedding_dim: int,
) -> Tuple[str, List[Any]]:
    if args.max_docs > 0:
        docs = docs[: args.max_docs]
    docs = _build_docs(docs)
    docs_hash = _sha1_json(docs)

    q_workspace = workspace_root / "graphrag" / dataset / backend_name / _sanitize_qid(qid)
    state_path = q_workspace / "index_state.json"

    reuse_index = (
        (not args.rebuild_index)
        and state_path.exists()
        and (q_workspace / "output").exists()
        and (q_workspace / "settings.yaml").exists()
        and json.loads(state_path.read_text(encoding="utf-8")).get("docs_hash") == docs_hash
    )

    prune_relaxed = False
    if not reuse_index:
        try:
            _prepare_workspace(
                q_workspace,
                docs=docs,
                graphrag_cli=graphrag_cli,
                env=env,
                llm_base_url=backend_base_url,
                llm_model=backend_model,
                embed_base_url=args.embed_base_url,
                embed_model=args.embed_model,
                embed_dim=embedding_dim,
                temperature=args.temperature,
                answer_max_tokens=args.answer_max_tokens,
                top_k=args.top_k,
                qa_prompt_mode=args.qa_prompt_mode,
                relax_pruning=False,
                request_timeout=args.request_timeout,
                retrieval_only=args.retrieval_only,
            )
            _index_workspace(
                q_workspace,
                graphrag_cli=graphrag_cli,
                env=env,
                index_method=args.index_method,
            )
        except Exception as exc:
            if not _is_prune_empty_failure(exc):
                raise
            prune_relaxed = True
            _prepare_workspace(
                q_workspace,
                docs=docs,
                graphrag_cli=graphrag_cli,
                env=env,
                llm_base_url=backend_base_url,
                llm_model=backend_model,
                embed_base_url=args.embed_base_url,
                embed_model=args.embed_model,
                embed_dim=embedding_dim,
                temperature=args.temperature,
                answer_max_tokens=args.answer_max_tokens,
                top_k=args.top_k,
                qa_prompt_mode=args.qa_prompt_mode,
                relax_pruning=True,
                request_timeout=args.request_timeout,
                retrieval_only=args.retrieval_only,
            )
            _index_workspace(
                q_workspace,
                graphrag_cli=graphrag_cli,
                env=env,
                index_method=args.index_method,
            )

        state = {
            "dataset": dataset,
            "llm_backend": backend_name,
            "qid": qid,
            "docs_hash": docs_hash,
            "index_method": args.index_method,
            "prune_relaxed": prune_relaxed,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    if reuse_index and args.retrieval_only:
        # Force update settings for retrieval only mode
        prompts_dir = q_workspace / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "retrieval_only.txt").write_text(RETRIEVAL_ONLY_PROMPT, encoding="utf-8")
        
        settings_path = q_workspace / "settings.yaml"
        # Try to load prune_relaxed from state if possible, default to False
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            prune_relaxed_val = state.get("prune_relaxed", False)
        except Exception:
            prune_relaxed_val = False

        _patch_settings(
            settings_path,
            llm_base_url=backend_base_url,
            llm_model=backend_model,
            embed_base_url=args.embed_base_url,
            embed_model=args.embed_model,
            embed_dim=embedding_dim,
            temperature=args.temperature,
            answer_max_tokens=args.answer_max_tokens,
            top_k=args.top_k,
            qa_prompt_mode=args.qa_prompt_mode,
            relax_pruning=prune_relaxed_val,
            request_timeout=args.request_timeout,
            retrieval_only=True,
        )

    response_type = "Short Answer" if (args.retrieval_only or args.qa_prompt_mode == "answer_only") else "Multiple Paragraphs"
    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = str(env.get("OPENAI_API_KEY") or "EMPTY")
    try:
        if args.retrieval_only:
            result_text, context_data = _run_local_context_only(
                root_dir=q_workspace,
                question=question,
            )
        else:
            result_text, context_data = _run_local_query(
                root_dir=q_workspace,
                response_type=response_type,
                question=question,
            )
    finally:
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key

    if args.retrieval_only:
        ctxs = _build_graphrag_ctxs(
            workspace=q_workspace,
            context_data=context_data,
            top_k=args.top_k,
            fallback_text=result_text,
        )
        return "", ctxs
    return normalize_answer_for_eval(result_text), []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphRAG QA baseline")
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

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--answer_max_tokens", type=int, default=96)
    parser.add_argument("--top_k", type=int, default=5)
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
    parser.add_argument("--index_method", default="standard", choices=["standard", "fast"])
    parser.add_argument("--graphrag_cli", default=None)
    parser.add_argument("--retrieval_only", action="store_true", help="Skip QA generation, output context dump.")
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
    pred_path = output_pred_path(output_root, "graphrag", dataset, backend.name)
    if args.retrieval_only:
        pred_path = pred_path.with_name(pred_path.stem + "_retrieval.jsonl")

    graphrag_cli = _resolve_graphrag_cli(args.graphrag_cli)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENAI_API_KEY"] = backend.api_key

    completed_ids = set()
    if pred_path.exists():
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    
    # Open file in append mode
    pred_handle = pred_path.open("a", encoding="utf-8")

    observed_dim = _detect_embedding_dim(
        args.embed_base_url,
        args.embed_model,
        request_timeout=args.request_timeout,
    )
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

    pred_rows: List[Dict[str, Any]] = []
    risk_retries = max(0, int(args.content_risk_retries))
    retry_wait_s = max(0.0, float(args.content_risk_retry_wait_sec))
    for row in qa_rows:
        qid = str(row.get("id") or "").strip()
        if qid in completed_ids:
            continue
            
        question = str(row.get("question") or "").strip()
        docs = list(row.get("docs") or [])

        pred = ""
        ctxs = []
        for risk_try in range(risk_retries + 1):
            try:
                pred, ctxs = _build_and_query_one(
                    args,
                    dataset=dataset,
                    backend_name=backend.name,
                    backend_model=backend.model,
                    backend_base_url=backend.base_url,
                    qid=qid,
                    question=question,
                    docs=docs,
                    workspace_root=workspace_root,
                    graphrag_cli=graphrag_cli,
                    env=env,
                    embedding_dim=embedding_dim,
                )
                break
            except Exception as exc:
                if _is_content_exists_risk(exc) and risk_try < risk_retries:
                    if retry_wait_s > 0:
                        time.sleep(retry_wait_s)
                    continue
                pred = ""
                ctxs = []
                break

        out_row = {"id": qid, "pred": pred}
        if ctxs:
            out_row["ctxs"] = ctxs
        
        # Write immediately
        pred_handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        pred_handle.flush()
        # pred_rows.append(out_row)

    pred_handle.close()
    # write_pred_jsonl(pred_path, pred_rows)
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
