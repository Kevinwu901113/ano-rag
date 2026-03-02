#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from urllib.parse import urlparse
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
from retrieval_schema import build_graphrag_ctxs_from_sources  # noqa: E402


RETRIEVAL_ONLY_PROMPT = """You are a retrieval debugger.
Please output the provided Context verbatim.
Do not summarize. Do not explain. Just output the context text.
"""
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _estimate_tokens(text: str) -> int:
    return max(1, len(_TOKEN_RE.findall(str(text or ""))))


def _extract_json_object_candidate(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    # Prefer fenced JSON blocks when present.
    if "```" in raw:
        fence_parts = raw.split("```")
        for part in fence_parts:
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                return part

    if raw.startswith("{") and raw.endswith("}"):
        return raw

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return None


def _model_validate_any(model_cls: Any, payload: Any) -> Any:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _model_validate_json_any(model_cls: Any, payload: str) -> Any:
    if hasattr(model_cls, "model_validate_json"):
        return model_cls.model_validate_json(payload)
    return model_cls.parse_raw(payload)


def _coerce_community_report_response(raw_text: str, response_model: Any) -> Any:
    candidate = _extract_json_object_candidate(raw_text)
    if candidate:
        try:
            return _model_validate_json_any(response_model, candidate)
        except Exception:
            try:
                data = json.loads(candidate)
                return _model_validate_any(response_model, data)
            except Exception:
                pass

    # Last-resort fallback to keep official workflow from hard-failing:
    # create a minimal valid structured payload from raw text.
    text = str(raw_text or "").strip()
    title = "Community Report"
    summary = text[:1000] if text else "No summary generated."
    fallback_payload = {
        "title": title,
        "summary": summary,
        "findings": [
            {
                "summary": "Generated Finding",
                "explanation": text[:2000] if text else "No details generated.",
            }
        ],
        "rating": 5.0,
        "rating_explanation": "Fallback report due to non-JSON model output.",
    }
    return _model_validate_any(response_model, fallback_payload)


def _patch_graphrag_schema_fallback() -> bool:
    """Patch GraphRAG community report extractor to tolerate non-schema models."""
    try:
        from graphrag.index.operations.summarize_communities import (  # type: ignore
            community_reports_extractor as extractor_mod,
        )
    except Exception:
        return False

    extractor_cls = getattr(extractor_mod, "CommunityReportsExtractor", None)
    response_model = getattr(extractor_mod, "CommunityReportResponse", None)
    result_model = getattr(extractor_mod, "CommunityReportsResult", None)
    logger = getattr(extractor_mod, "logger", None)
    if extractor_cls is None or response_model is None or result_model is None:
        return False
    if getattr(extractor_cls, "_relrag_schema_fallback_patched", False):
        return True

    def _is_response_format_error(exc: Exception) -> bool:
        msg = str(exc or "").lower()
        keys = (
            "response schema",
            "response schemas",
            "response_format",
            "response format",
            "invalid_request_error",
        )
        return any(k in msg for k in keys)

    async def _patched_call(self: Any, input_text: str):  # type: ignore[no-redef]
        output = None
        try:
            prompt = self._extraction_prompt.format(
                **{
                    extractor_mod.INPUT_TEXT_KEY: input_text,
                    extractor_mod.MAX_LENGTH_KEY: str(self._max_report_length),
                }
            )
            supports_structured = False
            try:
                supports_structured = bool(self._model.supports_structured_response())
            except Exception:
                supports_structured = False
            model_id = str(getattr(self._model, "_model_id", "") or "").lower()
            if model_id.startswith("deepseek/") or "deepseek/" in model_id:
                # DeepSeek currently rejects response_format in this pipeline.
                supports_structured = False

            async def _fallback_text_mode() -> Any:
                try:
                    response = await self._model.completion_async(
                        messages=prompt,
                        response_format_json_object=True,
                    )
                    raw_content = getattr(response, "content", "")
                except Exception as json_exc:
                    if not _is_response_format_error(json_exc):
                        raise
                    # DeepSeek may reject response_format entirely; do plain text call.
                    response = await self._model.completion_async(messages=prompt)
                    raw_content = getattr(response, "content", "")
                return _coerce_community_report_response(raw_content, response_model)

            if supports_structured:
                try:
                    response = await self._model.completion_async(
                        messages=prompt,
                        response_format=response_model,
                    )
                    output = response.formatted_response  # type: ignore[attr-defined]
                except Exception as exc:
                    # Official workflow currently hard-fails on some providers when
                    # response_format is unsupported. Fall back to text generation.
                    if not _is_response_format_error(exc):
                        raise
                    output = await _fallback_text_mode()
            else:
                output = await _fallback_text_mode()
        except Exception as exc:
            if logger is not None:
                logger.exception("error generating community report")
            self._on_error(exc, traceback.format_exc(), None)
            output = None

        text_output = self._get_text_output(output) if output else ""
        return result_model(  # type: ignore[call-arg]
            structured_output=output,
            output=text_output,
        )

    setattr(extractor_cls, "__call__", _patched_call)
    setattr(extractor_cls, "_relrag_schema_fallback_patched", True)
    return True


def _ensure_local_no_proxy(*base_urls: str) -> None:
    hosts: List[str] = []
    for url in base_urls:
        parsed = urlparse(str(url or ""))
        host = str(parsed.hostname or "").strip().lower()
        if host in {"127.0.0.1", "localhost"}:
            hosts.extend(["127.0.0.1", "localhost"])
    if not hosts:
        return

    wanted = {item for item in hosts if item}
    for key in ("NO_PROXY", "no_proxy"):
        existing = str(os.environ.get(key) or "")
        parts = [p.strip() for p in existing.split(",") if p.strip()]
        merged = parts[:]
        for host in sorted(wanted):
            if host not in merged:
                merged.append(host)
        os.environ[key] = ",".join(merged)


def _answer_with_llm(
    client: OpenAI,
    *,
    model: str,
    question: str,
    evidence_rows: List[Dict[str, Any]],
    temperature: float,
    answer_max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    system_prompt = load_aligned_reader_system_prompt()
    user_prompt = render_aligned_reader_prompt(question=question, evidence_rows=evidence_rows)
    try:
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
            return content, {
                "llm_calls": 1,
                "llm_retries": 0,
                "prompt_tokens_total": int(prompt_used or 0),
                "completion_tokens_total": int(completion_used or 0),
                "token_source": "api_usage",
                "token_unavailable_reason": None,
            }
        return content, {
            "llm_calls": 1,
            "llm_retries": 0,
            "prompt_tokens_total": _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt),
            "completion_tokens_total": _estimate_tokens(content),
            "token_source": "estimated",
            "token_unavailable_reason": "usage_not_provided",
        }
    except Exception as exc:
        return "", {
            "llm_calls": 1,
            "llm_retries": 0,
            "prompt_tokens_total": None,
            "completion_tokens_total": None,
            "token_source": "unavailable",
            "token_unavailable_reason": str(exc)[:200],
        }


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


def _detect_embedding_dim(
    embed_base_url: str,
    embed_model: str,
    request_timeout: float,
    *,
    max_retries: int = 8,
    backoff_base_sec: float = 1.0,
    backoff_max_sec: float = 20.0,
) -> int:
    client = OpenAI(
        base_url=embed_base_url,
        api_key="EMPTY",
        timeout=float(request_timeout),
    )
    last_exc: Exception | None = None
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            probe_vec = client.embeddings.create(model=embed_model, input="hello").data[0].embedding
            observed_dim = len(probe_vec)
            if observed_dim <= 0:
                raise RuntimeError(f"Invalid embedding dim ({observed_dim}) from model {embed_model}.")
            return observed_dim
        except Exception as exc:  # pragma: no cover - network/backend dependent
            last_exc = exc
            if attempt >= int(max_retries):
                break
            sleep_s = min(float(backoff_max_sec), float(backoff_base_sec) * (2 ** attempt))
            time.sleep(max(0.0, sleep_s))
    raise RuntimeError(
        f"Failed to detect embedding dim from {embed_model} ({embed_base_url}) "
        f"after {int(max_retries) + 1} attempts: {last_exc}"
    )


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


def _is_content_exists_risk(exc: Exception) -> bool:
    return "content exists risk" in str(exc or "").lower()


def _supports_response_schema(model_provider: str, model_name: str) -> bool:
    model_id = f"{str(model_provider or '').strip()}/{str(model_name or '').strip()}".strip("/")
    if not model_id:
        return False
    try:
        from litellm import supports_response_schema  # type: ignore

        return bool(supports_response_schema(model_id))
    except Exception:
        return False


def _resolve_community_report_workflow(
    mode: str,
    *,
    model_provider: str,
    model_name: str,
) -> str:
    raw = str(mode or "auto").strip().lower()
    if raw in {"create_community_reports", "structured"}:
        return "create_community_reports"
    if raw in {"create_community_reports_text", "text"}:
        return "create_community_reports_text"
    if raw != "auto":
        raise ValueError(
            "community_report_workflow must be one of: auto, "
            "create_community_reports, create_community_reports_text"
        )
    if _supports_response_schema(model_provider, model_name):
        return "create_community_reports"
    return "create_community_reports_text"


def _patch_workflows_for_community_reports(cfg: Dict[str, Any], workflow_name: str) -> None:
    desired = str(workflow_name)
    alt = "create_community_reports_text" if desired == "create_community_reports" else "create_community_reports"
    workflows = cfg.get("workflows")
    if not isinstance(workflows, list) or not workflows:
        cfg["workflows"] = [
            "load_input_documents",
            "create_base_text_units",
            "create_final_documents",
            "extract_graph",
            "finalize_graph",
            "extract_covariates",
            "create_communities",
            "create_final_text_units",
            desired,
            "generate_text_embeddings",
        ]
        return

    out: List[str] = []
    seen = set()
    replaced = False
    for item in workflows:
        if not isinstance(item, str):
            continue
        current = desired if item == alt else item
        if current == desired:
            replaced = True
        if current in seen:
            continue
        seen.add(current)
        out.append(current)

    if not replaced:
        if "generate_text_embeddings" in out:
            idx = out.index("generate_text_embeddings")
            out.insert(idx, desired)
        else:
            out.append(desired)
    cfg["workflows"] = out


def _patch_settings(
    settings_path: Path,
    *,
    llm_provider: str,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    embed_dim: int,
    temperature: float,
    index_max_tokens: int,
    answer_max_tokens: int,
    top_k: int,
    qa_prompt_mode: str,
    relax_pruning: bool,
    request_timeout: float,
    community_report_workflow: str,
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
            "model_provider": str(llm_provider or "openai"),
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
                "max_tokens": int(index_max_tokens),
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

    cfg.setdefault("extract_graph", {})
    cfg["extract_graph"]["completion_model_id"] = completion_id

    cfg.setdefault("summarize_descriptions", {})
    cfg["summarize_descriptions"]["completion_model_id"] = completion_id

    cfg.setdefault("community_reports", {})
    cfg["community_reports"]["completion_model_id"] = completion_id
    _patch_workflows_for_community_reports(cfg, community_report_workflow)

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
    llm_provider: str,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    embed_dim: int,
    temperature: float,
    index_max_tokens: int,
    answer_max_tokens: int,
    top_k: int,
    qa_prompt_mode: str,
    relax_pruning: bool,
    request_timeout: float,
    community_report_workflow: str,
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
    (prompts_dir / "answer_only.txt").write_text(load_aligned_reader_system_prompt(), encoding="utf-8")
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
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_dim=embed_dim,
        temperature=temperature,
        index_max_tokens=index_max_tokens,
        answer_max_tokens=answer_max_tokens,
        top_k=top_k,
        qa_prompt_mode=qa_prompt_mode,
        relax_pruning=relax_pruning,
        request_timeout=request_timeout,
        community_report_workflow=community_report_workflow,
        retrieval_only=retrieval_only,
    )


def _index_workspace(
    workspace: Path,
    *,
    graphrag_cli: str,
    env: Dict[str, str],
    index_method: str,
) -> None:
    # Prefer in-process indexing so we can apply compatibility patches for
    # schema-constrained community report extraction.
    try:
        from graphrag.api.index import build_index
        from graphrag.config.load_config import load_config
    except Exception:
        build_index = None  # type: ignore[assignment]
        load_config = None  # type: ignore[assignment]

    if build_index is None or load_config is None:
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
        return

    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = str(env.get("OPENAI_API_KEY") or "EMPTY")
    try:
        _patch_graphrag_schema_fallback()
        config = load_config(root_dir=workspace, cli_overrides={})
        results = asyncio.run(
            build_index(
                config=config,
                method=index_method,
                is_update_run=False,
                callbacks=None,
                additional_context=None,
                verbose=False,
            )
        )
    finally:
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key

    failed = [item for item in results if getattr(item, "error", None) is not None]
    if failed:
        first = failed[0]
        raise RuntimeError(
            f"GraphRAG index failed at workflow={getattr(first, 'workflow', '')}: "
            f"{getattr(first, 'error', None)}"
        )


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


def _resolve_community_level(root_dir: Path, preferred_level: int = 2) -> int:
    communities_path = root_dir / "output" / "communities.parquet"
    if not communities_path.exists():
        return max(0, int(preferred_level))
    try:
        communities = pd.read_parquet(communities_path)
    except Exception:
        return max(0, int(preferred_level))
    if communities.empty or "level" not in communities.columns:
        return max(0, int(preferred_level))
    try:
        max_level = int(communities["level"].fillna(0).max())
    except Exception:
        max_level = int(preferred_level)
    return max(0, min(int(preferred_level), max_level))


def _read_index_time_ms(root_dir: Path) -> float:
    stats_path = root_dir / "output" / "stats.json"
    if not stats_path.exists():
        return 0.0
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        total_runtime = float(payload.get("total_runtime") or 0.0)
        return max(0.0, total_runtime * 1000.0)
    except Exception:
        return 0.0


def _resolve_completion_provider(backend_name: str, llm_base_url: str) -> str:
    backend = str(backend_name or "").strip().lower()
    base_url = str(llm_base_url or "").strip().lower()
    if backend == "deepseek" or "deepseek" in base_url:
        return "deepseek"
    return "openai"


def _run_local_query(
    *,
    root_dir: Path,
    community_level: int,
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
            community_level=int(community_level),
            response_type=response_type,
            streaming=False,
            query=question,
            verbose=False,
    )
    return str(response or "").strip(), context_data if isinstance(context_data, dict) else {}


def _run_local_context_only(
    *,
    root_dir: Path,
    community_level: int,
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
    llm_client: OpenAI,
    qid: str,
    question: str,
    docs: List[Dict[str, Any]],
    workspace_root: Path,
    graphrag_cli: str,
    env: Dict[str, str],
    embedding_dim: int,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    if args.max_docs > 0:
        docs = docs[: args.max_docs]
    docs = _build_docs(docs)
    docs_hash = _sha1_json(docs)

    q_workspace = workspace_root / "graphrag" / dataset / backend_name / _sanitize_qid(qid)
    state_path = q_workspace / "index_state.json"
    llm_provider = _resolve_completion_provider(backend_name, backend_base_url)
    community_report_workflow = _resolve_community_report_workflow(
        args.community_report_workflow,
        model_provider=llm_provider,
        model_name=backend_model,
    )
    state_snapshot: Dict[str, Any] = {}
    if state_path.exists():
        try:
            state_snapshot = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state_snapshot, dict):
                state_snapshot = {}
        except Exception:
            state_snapshot = {}

    reuse_index = (
        (not args.rebuild_index)
        and (q_workspace / "output").exists()
        and (q_workspace / "settings.yaml").exists()
        and state_snapshot.get("docs_hash") == docs_hash
        and state_snapshot.get("community_report_workflow") == community_report_workflow
        and state_snapshot.get("llm_provider") == llm_provider
        and state_snapshot.get("llm_model") == backend_model
    )

    prune_relaxed = True
    index_time_ms = 0.0
    if not reuse_index:
        _prepare_workspace(
            q_workspace,
            docs=docs,
            graphrag_cli=graphrag_cli,
            env=env,
            llm_provider=llm_provider,
            llm_base_url=backend_base_url,
            llm_model=backend_model,
            embed_base_url=args.embed_base_url,
            embed_model=args.embed_model,
            embed_dim=embedding_dim,
            temperature=args.temperature,
            index_max_tokens=args.index_max_tokens,
            answer_max_tokens=args.answer_max_tokens,
            top_k=args.top_k,
            qa_prompt_mode=args.qa_prompt_mode,
            relax_pruning=True,
            request_timeout=args.request_timeout,
            community_report_workflow=community_report_workflow,
            retrieval_only=args.retrieval_only,
        )
        _index_workspace(
            q_workspace,
            graphrag_cli=graphrag_cli,
            env=env,
            index_method=args.index_method,
        )
        index_time_ms = _read_index_time_ms(q_workspace)

        state = {
            "dataset": dataset,
            "llm_backend": backend_name,
            "llm_provider": llm_provider,
            "llm_model": backend_model,
            "qid": qid,
            "docs_hash": docs_hash,
            "index_method": args.index_method,
            "prune_relaxed": prune_relaxed,
            "community_report_workflow": community_report_workflow,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        index_time_ms = _read_index_time_ms(q_workspace)

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
            llm_provider=llm_provider,
            llm_base_url=backend_base_url,
            llm_model=backend_model,
            embed_base_url=args.embed_base_url,
            embed_model=args.embed_model,
            embed_dim=embedding_dim,
            temperature=args.temperature,
            index_max_tokens=args.index_max_tokens,
            answer_max_tokens=args.answer_max_tokens,
            top_k=args.top_k,
            qa_prompt_mode=args.qa_prompt_mode,
            relax_pruning=prune_relaxed_val,
            request_timeout=args.request_timeout,
            community_report_workflow=community_report_workflow,
            retrieval_only=True,
        )

    community_level = _resolve_community_level(q_workspace, preferred_level=2)
    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = str(env.get("OPENAI_API_KEY") or "EMPTY")
    try:
        query_start = time.perf_counter()
        if args.retrieval_only:
            try:
                result_text, context_data = _run_local_context_only(
                    root_dir=q_workspace,
                    community_level=community_level,
                    question=question,
                )
                # Some GraphRAG runs return empty context records in context-only mode.
                # Fallback to local_query so retrieval outputs still carry evaluable context.
                has_sources = bool(
                    _records_from_maybe_table((context_data or {}).get("sources"))
                )
                has_text = bool(str(result_text or "").strip())
                if not has_sources and not has_text:
                    result_text, context_data = _run_local_query(
                        root_dir=q_workspace,
                        community_level=community_level,
                        response_type="Short Answer",
                        question=question,
                    )
            except Exception:
                result_text, context_data = _run_local_query(
                    root_dir=q_workspace,
                    community_level=community_level,
                    response_type="Short Answer",
                    question=question,
                )
        elif args.graphrag_qa_mode == "official":
            result_text, context_data = _run_local_query(
                root_dir=q_workspace,
                community_level=community_level,
                response_type="Short Answer",
                question=question,
            )
        else:
            try:
                result_text, context_data = _run_local_context_only(
                    root_dir=q_workspace,
                    community_level=community_level,
                    question=question,
                )
            except Exception:
                result_text, context_data = _run_local_query(
                    root_dir=q_workspace,
                    community_level=community_level,
                    response_type="Short Answer",
                    question=question,
                )
        query_time_ms = (time.perf_counter() - query_start) * 1000.0
    finally:
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key

    ctxs = _build_graphrag_ctxs(
        workspace=q_workspace,
        context_data=context_data,
        top_k=args.top_k,
        fallback_text=result_text,
    )
    if args.retrieval_only:
        cost = build_cost_record(
            index_time_ms=index_time_ms,
            query_retrieval_ms=query_time_ms,
            query_reader_ms=0.0,
            llm_calls=0,
            llm_retries=0,
            prompt_tokens_total=None,
            completion_tokens_total=None,
            token_source="unavailable",
            token_unavailable_reason="graphrag_usage_not_exposed",
        ).to_dict()
        return "", ctxs, cost

    if args.graphrag_qa_mode == "official":
        cost = build_cost_record(
            index_time_ms=index_time_ms,
            query_retrieval_ms=0.0,
            query_reader_ms=query_time_ms,
            llm_calls=0,
            llm_retries=0,
            prompt_tokens_total=None,
            completion_tokens_total=None,
            token_source="unavailable",
            token_unavailable_reason="graphrag_usage_not_exposed",
        ).to_dict()
        return normalize_answer_for_eval(result_text), ctxs, cost

    reader_start = time.perf_counter()
    pred_raw, llm_meta = _answer_with_llm(
        llm_client,
        model=backend_model,
        question=question,
        evidence_rows=ctxs,
        temperature=args.temperature,
        answer_max_tokens=args.answer_max_tokens,
    )
    query_reader_ms = (time.perf_counter() - reader_start) * 1000.0
    cost = build_cost_record(
        index_time_ms=index_time_ms,
        query_retrieval_ms=query_time_ms,
        query_reader_ms=query_reader_ms,
        llm_calls=llm_meta.get("llm_calls", 0),
        llm_retries=llm_meta.get("llm_retries", 0),
        prompt_tokens_total=llm_meta.get("prompt_tokens_total"),
        completion_tokens_total=llm_meta.get("completion_tokens_total"),
        token_source=str(llm_meta.get("token_source") or "unavailable"),
        token_unavailable_reason=llm_meta.get("token_unavailable_reason"),
    ).to_dict()
    return normalize_answer_for_eval(pred_raw), ctxs, cost


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

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--index_max_tokens", type=int, default=1024)
    parser.add_argument("--answer_max_tokens", type=int, default=None)
    parser.add_argument("--relrag_config", default=None, help="Optional RelRAG config path for default reader policy.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--qa_prompt_mode", default="answer_only", choices=["answer_only", "default"])
    parser.add_argument(
        "--graphrag_qa_mode",
        default="official",
        choices=["official", "aligned_reader"],
        help=(
            "official: use GraphRAG local_search answer directly (official pipeline). "
            "aligned_reader: GraphRAG retrieval + unified external reader prompt."
        ),
    )
    parser.add_argument(
        "--community_report_workflow",
        default="auto",
        choices=["auto", "create_community_reports", "create_community_reports_text"],
        help=(
            "Official GraphRAG community report workflow. "
            "auto=choose structured if model supports response schema, else text workflow."
        ),
    )
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
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Do not fail run when some samples error; failed rows are still recorded with error reason.",
    )
    args = parser.parse_args()

    dataset = ensure_dataset(args.dataset)
    backend = resolve_llm_backend(args.llm_backend)
    _ensure_local_no_proxy(backend.base_url, args.embed_base_url)
    reader_params = resolve_effective_reader_params(
        dataset=dataset,
        backend=backend.name,
        answer_max_tokens=args.answer_max_tokens,
        temperature=args.temperature,
        config_path=args.relrag_config,
    )
    args.answer_max_tokens = int(reader_params["answer_max_tokens"])
    args.temperature = float(reader_params["temperature"])

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    workspace_root = Path(args.workspace_root).resolve()

    qa_path = data_root / dataset / "qa.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(
            f"Missing intermediate data for {dataset}. Run baseline/tools/build_intermediate.py first."
        )

    qa_rows = load_qa_with_docs(qa_path, limit=args.limit)
    valid_qids = {
        str(row.get("id") or "").strip()
        for row in qa_rows
        if str(row.get("id") or "").strip()
    }
    pred_path = output_pred_path(output_root, "graphrag", dataset, backend.name)
    if args.retrieval_only:
        pred_path = pred_path.with_name(pred_path.stem + "_retrieval.jsonl")

    if args.rebuild_index and pred_path.exists():
        pred_path.unlink()

    graphrag_cli = _resolve_graphrag_cli(args.graphrag_cli)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENAI_API_KEY"] = backend.api_key
    llm_client = OpenAI(
        base_url=backend.base_url,
        api_key=backend.api_key,
        timeout=float(args.request_timeout),
    )

    completed_ids = set()
    if pred_path.exists():
        existing_rows: List[Dict[str, Any]] = []
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        existing_rows.append(row)
                except Exception:
                    pass

        retained_by_qid: Dict[str, Dict[str, Any]] = {}
        for row in existing_rows:
            qid = str(row.get("id") or "").strip()
            if not qid:
                continue
            if qid not in valid_qids:
                continue
            cost = row.get("cost") if isinstance(row.get("cost"), dict) else {}
            token_reason = str(cost.get("token_unavailable_reason") or "").strip().lower()
            has_error = bool(str(row.get("error") or "").strip())
            if token_reason == "not_run" or has_error:
                continue
            if args.retrieval_only and not isinstance(row.get("ctxs"), list):
                continue
            retained_by_qid[qid] = row

        retained_rows: List[Dict[str, Any]] = list(retained_by_qid.values())
        completed_ids = set(retained_by_qid.keys())

        if len(retained_rows) != len(existing_rows):
            write_pred_jsonl(pred_path, retained_rows)
    
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
    failed_ids: List[str] = []
    risk_retries = max(0, int(args.content_risk_retries))
    retry_wait_s = max(0.0, float(args.content_risk_retry_wait_sec))
    for row in qa_rows:
        qid = str(row.get("id") or "").strip()
        if not qid:
            continue
        if qid in completed_ids:
            continue
            
        question = str(row.get("question") or "").strip()
        docs = list(row.get("docs") or [])

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
            token_unavailable_reason="not_run",
        ).to_dict()
        row_error: str | None = None
        for risk_try in range(risk_retries + 1):
            try:
                pred, ctxs, cost = _build_and_query_one(
                    args,
                    dataset=dataset,
                    backend_name=backend.name,
                    backend_model=backend.model,
                    backend_base_url=backend.base_url,
                    llm_client=llm_client,
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
                row_error = f"{type(exc).__name__}: {exc}"
                cost = build_cost_record(
                    index_time_ms=0.0,
                    query_retrieval_ms=0.0,
                    query_reader_ms=0.0,
                    llm_calls=0,
                    llm_retries=0,
                    prompt_tokens_total=None,
                    completion_tokens_total=None,
                    token_source="unavailable",
                    token_unavailable_reason=(row_error[:220] if row_error else "run_error"),
                ).to_dict()
                break

        out_row = {"id": qid, "pred": pred, "cost": cost}
        if args.retrieval_only:
            out_row["ctxs"] = ctxs if isinstance(ctxs, list) else []
        elif ctxs:
            out_row["ctxs"] = ctxs
        if row_error:
            out_row["error"] = row_error[:1000]
            failed_ids.append(qid)

        # Write immediately
        pred_handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        pred_handle.flush()
        pred_rows.append(out_row)
        completed_ids.add(qid)

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
        summarize_cost_records(method="graphrag", dataset=dataset, backend=backend.name, rows=all_rows),
    )
    if failed_ids:
        print(f"[warn] failed samples: {len(failed_ids)}")
        if not args.allow_partial:
            raise RuntimeError(
                f"GraphRAG run has failed samples ({len(failed_ids)}). "
                "Re-run with --allow_partial to keep partial outputs."
            )
    print(f"[ok] wrote {pred_path}")


if __name__ == "__main__":
    main()
