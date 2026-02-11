import argparse
import csv
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

from loguru import logger

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    logger.warning("numpy unavailable: {}. Dense retrieval disabled.", exc)

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    BM25Okapi = None  # type: ignore
    logger.warning("rank_bm25 unavailable: {}. BM25 retrieval disabled.", exc)

from relrag.api import answer
from relrag.config.dataset_config import (
    get_dataset_config,
    resolve_openai_api_key,
    resolve_openai_config,
    resolve_reader,
)
from relrag.config.config_loader import ConfigLoader, config as global_config
from relrag.generator import answerer as answerer_module
from relrag.prompt import load_prompt
from relrag.utils.answer_source import resolve_short_answer, sha1_text
from relrag.utils.embedding_utils import get_shared_encoder
from relrag.utils.eval_metrics import score_metrics
from relrag.utils.openai_answer import generate_openai_answer
from relrag.utils.output_eval import has_final_tag
from relrag.utils.text_utils import TextUtils
from relrag.utils.vllm_runtime import resolve_vllm_endpoint_model


DEFAULT_TOP_K = 10
DEFAULT_LIMIT = 0
DEFAULT_WORKERS = 1
DEFAULT_CACHE_DIR = "result/narrativeqa/cache"
DEFAULT_OUTPUT_DIR = "result/narrativeqa"
DEFAULT_CONTEXT_MODE = "summary"
DEFAULT_MODES = ("bm25", "dense")
DEFAULT_SPLIT = "valid"
DEFAULT_STALL_WARN_SEC = 300.0
DEFAULT_STALL_ABORT_SEC = 900.0
DEFAULT_OVERFETCH = 1.0
MIN_OVERFETCH = 1.0
DEFAULT_BACKFILL_MAX_OVERFETCH = 4.0
DEFAULT_BACKFILL_STEP = 1.5
DEFAULT_BACKFILL_ROUNDS = 3
DEFAULT_LLM_RETRY_ON_EMPTY = 1
DEFAULT_LLM_RETRY_EVIDENCE = 6


class ProgressBar:
    def __init__(
        self,
        total: int,
        stream: Optional[TextIO] = None,
        min_interval_sec: float = 0.5,
    ) -> None:
        self.total = max(0, int(total))
        self.processed = 0
        self._stream = stream or sys.stderr
        self._min_interval_sec = max(0.05, float(min_interval_sec))
        self._last_update = 0.0
        self._last_len = 0

    def update(self, step: int = 1) -> None:
        self.processed += max(0, int(step))
        now = time.time()
        if self.total and self.processed < self.total:
            if (now - self._last_update) < self._min_interval_sec:
                return
        self._last_update = now
        self._render()

    def _render(self) -> None:
        total = self.total
        processed = min(self.processed, total) if total else self.processed
        remaining = max(0, total - processed) if total else 0
        if total > 0:
            columns = shutil.get_terminal_size((80, 20)).columns
            bar_width = max(10, min(40, columns - 40))
            ratio = min(1.0, processed / total)
            filled = int(bar_width * ratio)
            bar = "=" * filled + "-" * (bar_width - filled)
            msg = f"[{bar}] {processed}/{total} (remaining {remaining})"
        else:
            msg = f"{processed} processed"
        pad = " " * max(0, self._last_len - len(msg))
        self._stream.write("\r" + msg + pad)
        self._stream.flush()
        self._last_len = len(msg)

    def close(self) -> None:
        self._render()
        self._stream.write("\n")
        self._stream.flush()


class DocumentCache:
    def __init__(self) -> None:
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._units: Dict[str, List[Dict[str, Any]]] = {}

    def lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def get_units(self, key: str) -> Optional[List[Dict[str, Any]]]:
        return self._units.get(key)

    def set_units(self, key: str, units: List[Dict[str, Any]]) -> None:
        self._units[key] = units


def _load_entry_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    entry_cfg = cfg.get("narrativeqa_entry") or cfg.get("entry") or {}
    if not isinstance(entry_cfg, dict):
        return {}
    return entry_cfg


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pick_arg(
    args: argparse.Namespace,
    entry_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    if name in dataset_cfg:
        return dataset_cfg.get(name)
    if name in entry_cfg:
        return entry_cfg.get(name)
    return default


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalize_split(value: Optional[str]) -> str:
    raw = (value or "").strip().lower()
    if raw in {"dev", "val"}:
        return "valid"
    if raw in {"train", "valid", "test"}:
        return raw
    return DEFAULT_SPLIT


def _normalize_context_mode(value: Optional[str]) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return DEFAULT_CONTEXT_MODE
    if raw in {"summary", "summary-as-context"}:
        return "summary"
    if raw in {"story", "story-as-context"}:
        return "story"
    raise ValueError("context_mode must be summary-as-context or story-as-context")


def _parse_modes(value: Any) -> List[str]:
    allowed = {"dense", "bm25"}
    if not value:
        return list(DEFAULT_MODES)
    if isinstance(value, str):
        parts = [p for p in re.split(r"[,\s]+", value.strip()) if p]
    elif isinstance(value, (list, tuple)):
        parts = [str(p) for p in value if str(p)]
    else:
        parts = [str(value)]
    normalized: List[str] = []
    seen = set()
    for part in parts:
        key = part.strip().lower()
        if not key:
            continue
        if key not in allowed:
            raise ValueError(f"Unsupported mode '{part}' (allowed: {sorted(allowed)})")
        if key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized or list(DEFAULT_MODES)


def _resolve_retriever_modes(
    *,
    mode_arg: Optional[str],
    modes_arg: Optional[str],
    entry_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> List[str]:
    raw = None
    if mode_arg:
        raw = mode_arg
    elif modes_arg:
        raw = modes_arg
    elif dataset_cfg.get("retrievers") is not None:
        raw = dataset_cfg.get("retrievers")
    elif dataset_cfg.get("retriever_modes") is not None:
        raw = dataset_cfg.get("retriever_modes")
    elif entry_cfg.get("retrievers") is not None:
        raw = entry_cfg.get("retrievers")
    elif entry_cfg.get("modes") is not None:
        raw = entry_cfg.get("modes")
    return _parse_modes(raw) if raw is not None else list(DEFAULT_MODES)


def _apply_dataset_retriever(cfg: Dict[str, Any], dataset_key: str) -> Dict[str, Any]:
    dataset_cfg = get_dataset_config(cfg, dataset_key)
    retriever_override = dataset_cfg.get("retriever") if isinstance(dataset_cfg, dict) else None
    if isinstance(retriever_override, dict):
        base_retriever = cfg.get("retriever") or {}
        cfg["retriever"] = _deep_merge(base_retriever, retriever_override)
    return cfg


def _resolve_readers(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> List[str]:
    if args.reader:
        readers = [resolve_reader(args.reader, dataset_cfg)]
    else:
        if dataset_cfg.get("readers") is not None:
            raw = dataset_cfg.get("readers")
        elif dataset_cfg.get("models") is not None:
            raw = dataset_cfg.get("models")
        else:
            raw = dataset_cfg.get("reader")
        if raw is None:
            readers = [resolve_reader(None, dataset_cfg)]
        elif isinstance(raw, (list, tuple)):
            readers = [resolve_reader(str(item), dataset_cfg) for item in raw if str(item).strip()]
        else:
            readers = [resolve_reader(str(raw), dataset_cfg)]

    seen = set()
    ordered: List[str] = []
    for reader in readers:
        if reader in seen:
            continue
        ordered.append(reader)
        seen.add(reader)

    vllm_enabled = bool((cfg.get("vllm") or {}).get("enabled", True))
    openai_enabled = bool((cfg.get("openai") or {}).get("enabled", True))
    for reader in ordered:
        if reader == "vllm" and not vllm_enabled:
            raise ValueError("vLLM mode disabled in config (vllm.enabled=false).")
        if reader == "openai" and not openai_enabled:
            raise ValueError("OpenAI mode disabled in config (openai.enabled=false).")
    return ordered


def _pred_filename(split: str, reader: str, mode: str, reader_count: int, mode_count: int) -> str:
    if reader_count == 1 and mode_count > 1:
        return f"pred_{split}_{mode}.jsonl"
    if mode_count == 1 and reader_count > 1:
        return f"pred_{split}_{reader}.jsonl"
    return f"pred_{split}_{reader}_{mode}.jsonl"


def _resolve_llm_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[str, str]:
    return resolve_vllm_endpoint_model(
        endpoint_override=args.endpoint,
        model_override=args.model,
        vllm_cfg=cfg.get("vllm"),
    )


def _prompt_template_hash(prompt_name: Optional[str]) -> Optional[str]:
    if not prompt_name:
        return None
    try:
        content = load_prompt(prompt_name)
    except Exception:
        return None
    return sha1_text(content)


def _resolve_llm_input_hash(prompt_capture: Dict[str, Any]) -> str:
    prompt = prompt_capture.get("prompt") or ""
    system_prompt = prompt_capture.get("system_prompt") or ""
    if system_prompt:
        combined = f"system:{system_prompt}\nuser:{prompt}"
    else:
        combined = str(prompt)
    return sha1_text(combined)


def _classify_llm_exception(exc: Exception) -> Tuple[str, str]:
    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout", message[:200]
    if "context" in lowered and ("length" in lowered or "token" in lowered or "max" in lowered):
        return "context_len", message[:200]
    return "other", message[:200]


def generate_answer(
    question: str,
    evidences: List[Dict[str, Any]],
    *,
    reader: str,
    llm_endpoint: str,
    llm_model: str,
    openai_cfg: Optional[Dict[str, Any]],
    base_cfg: Optional[Dict[str, Any]] = None,
    run_dir: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Optional[str], Optional[str]]:
    prompt_capture: Dict[str, Any] = {}
    if reader == "vllm":
        try:
            raw_answer = answer(
                question=question,
                evidences=evidences,
                llm_endpoint=llm_endpoint,
                llm_model=llm_model,
                prompt_capture=prompt_capture,
                cfg=base_cfg,
                run_dir=run_dir,
            )
        except Exception as exc:
            reason, message = _classify_llm_exception(exc)
            prompt_name = prompt_capture.get("prompt_name") or answerer_module.ANSWER_PROMPT_NAME
            prompt_template_hash = _prompt_template_hash(prompt_name)
            llm_input_hash = _resolve_llm_input_hash(prompt_capture) if prompt_capture else ""
            return "", {
                "prompt_name": prompt_name,
                "prompt_template_hash": prompt_template_hash,
                "llm_input_hash": llm_input_hash,
            }, message, reason
        prompt_name = prompt_capture.get("prompt_name") or answerer_module.ANSWER_PROMPT_NAME
        prompt_template_hash = _prompt_template_hash(prompt_name)
        llm_input_hash = _resolve_llm_input_hash(prompt_capture)
        return raw_answer, {
            "prompt_name": prompt_name,
            "prompt_template_hash": prompt_template_hash,
            "llm_input_hash": llm_input_hash,
        }, None, None
    if reader == "openai":
        if not openai_cfg:
            raise ValueError("OpenAI config missing for reader=openai")
        try:
            raw_answer = generate_openai_answer(
                question,
                evidences,
                openai_cfg,
                prompt_capture=prompt_capture,
                run_dir=run_dir,
                cfg=base_cfg,
            )
        except Exception as exc:
            reason, message = _classify_llm_exception(exc)
            prompt_name = prompt_capture.get("prompt_name") or openai_cfg.get("answer_prompt_name")
            prompt_template_hash = _prompt_template_hash(prompt_name)
            system_prompt_name = openai_cfg.get("system_prompt_name")
            system_prompt_text = prompt_capture.get("system_prompt") or ""
            system_prompt_hash = sha1_text(system_prompt_text) if system_prompt_text else None
            llm_input_hash = _resolve_llm_input_hash(prompt_capture) if prompt_capture else ""
            return "", {
                "prompt_name": prompt_name,
                "prompt_template_hash": prompt_template_hash,
                "system_prompt_name": system_prompt_name,
                "system_prompt_hash": system_prompt_hash,
                "llm_input_hash": llm_input_hash,
            }, message, reason
        prompt_name = prompt_capture.get("prompt_name") or openai_cfg.get("answer_prompt_name")
        prompt_template_hash = _prompt_template_hash(prompt_name)
        system_prompt_name = openai_cfg.get("system_prompt_name")
        system_prompt_text = prompt_capture.get("system_prompt") or ""
        system_prompt_hash = sha1_text(system_prompt_text) if system_prompt_text else None
        llm_input_hash = _resolve_llm_input_hash(prompt_capture)
        return raw_answer, {
            "prompt_name": prompt_name,
            "prompt_template_hash": prompt_template_hash,
            "system_prompt_name": system_prompt_name,
            "system_prompt_hash": system_prompt_hash,
            "llm_input_hash": llm_input_hash,
        }, None, None
    raise ValueError(f"Unknown reader type: {reader}")


def _load_qaps(path: Path, split: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    doc_counts: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = _normalize_split(row.get("set"))
            if row_split != split:
                continue
            doc_id = (row.get("document_id") or "").strip()
            question = (row.get("question") or "").strip()
            if not doc_id or not question:
                continue
            idx = doc_counts.get(doc_id, 0)
            doc_counts[doc_id] = idx + 1
            qid = f"{doc_id}::{idx}"
            references = [
                (row.get("answer1") or "").strip(),
                (row.get("answer2") or "").strip(),
            ]
            references = [ref for ref in references if ref]
            entries.append(
                {
                    "qid": qid,
                    "document_id": doc_id,
                    "question": question,
                    "references": references,
                }
            )
    return entries


def _load_summaries(path: Path, split: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    split_map: Dict[str, str] = {}
    all_map: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_id = (row.get("document_id") or "").strip()
            summary = (row.get("summary") or "").strip()
            if not doc_id or not summary:
                continue
            all_map.setdefault(doc_id, summary)
            row_split = _normalize_split(row.get("set"))
            if row_split == split:
                split_map[doc_id] = summary
    return split_map, all_map


def _find_story_path(doc_id: str, stories_dir: Path) -> Optional[Path]:
    candidates = [
        stories_dir / f"{doc_id}.content",
        stories_dir / f"{doc_id}.txt",
        stories_dir / f"{doc_id}.story",
        stories_dir / f"{doc_id}.text",
        stories_dir / doc_id,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    matches = sorted(stories_dir.glob(f"{doc_id}.*"))
    if matches:
        return matches[0]
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _resolve_doc_text(
    doc_id: str,
    *,
    context_mode: str,
    summaries_map: Dict[str, str],
    summaries_all: Dict[str, str],
    stories_dir: Optional[Path],
) -> str:
    if context_mode == "summary":
        text = summaries_map.get(doc_id) or summaries_all.get(doc_id)
        if not text:
            raise KeyError(f"Summary not found for document {doc_id}")
        return text
    if stories_dir is None:
        raise ValueError("stories_dir is required for story-as-context mode")
    story_path = _find_story_path(doc_id, stories_dir)
    if not story_path:
        raise FileNotFoundError(f"Story file not found for document {doc_id}")
    text = _read_text(story_path)
    if not text.strip():
        raise ValueError(f"Story file empty for document {doc_id}")
    return text


def _build_sentence_units(doc_id: str, text: str) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    sentences = TextUtils.split_by_sentence(text)
    for idx, sentence in enumerate(sentences):
        clean = str(sentence).strip()
        if not clean:
            continue
        units.append(
            {
                "doc_id": doc_id,
                "title": doc_id,
                "sentence_idx": idx,
                "text": clean,
            }
        )
    return units


def _ensure_units_for_doc(
    item: Dict[str, Any],
    *,
    context_mode: str,
    summaries_map: Dict[str, str],
    summaries_all: Dict[str, str],
    stories_dir: Optional[Path],
    force_build: bool,
    doc_cache: DocumentCache,
) -> List[Dict[str, Any]]:
    doc_id = item["document_id"]
    cache_key = f"{context_mode}:{doc_id}"
    lock = doc_cache.lock_for(cache_key)
    with lock:
        cached = doc_cache.get_units(cache_key)
        if cached is not None and not force_build:
            return cached
        text = _resolve_doc_text(
            doc_id,
            context_mode=context_mode,
            summaries_map=summaries_map,
            summaries_all=summaries_all,
            stories_dir=stories_dir,
        )
        units = _build_sentence_units(doc_id, text)
        doc_cache.set_units(cache_key, units)
        return units


def _tokenize(text: str, ngram: List[int]) -> List[str]:
    base = [tok for tok in str(text).lower().split() if tok]
    if not base:
        return []
    if not ngram:
        return base
    tokens: List[str] = []
    for n in ngram:
        n = int(n)
        if n <= 1:
            tokens.extend(base)
            continue
        if n > len(base):
            continue
        for i in range(len(base) - n + 1):
            tokens.append(" ".join(base[i : i + n]))
    return tokens or base


def _bm25_search(
    question: str,
    units: List[Dict[str, Any]],
    bm25_cfg: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    if BM25Okapi is None:
        raise RuntimeError("rank_bm25 is required for BM25 baseline retrieval.")
    ngram = bm25_cfg.get("ngram") or [1]
    docs: List[List[str]] = []
    valid_units: List[Dict[str, Any]] = []
    for unit in units:
        tokens = _tokenize(unit.get("text", ""), ngram)
        if not tokens:
            continue
        docs.append(tokens)
        valid_units.append(unit)
    query_tokens = _tokenize(question, ngram)
    if not docs or not query_tokens:
        return []
    k1 = float(bm25_cfg.get("k1", 0.9))
    b = float(bm25_cfg.get("b", 0.4))
    bm25 = BM25Okapi(docs, k1=k1, b=b)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    results: List[Dict[str, Any]] = []
    for idx, score in ranked[: max(0, int(top_k))]:
        results.append({"unit": valid_units[idx], "score": float(score)})
    return results


def _dense_search(
    question: str,
    units: List[Dict[str, Any]],
    encoder,
    embed_cfg: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    if np is None:
        raise RuntimeError("numpy is required for dense baseline retrieval.")
    if encoder is None:
        raise RuntimeError("Dense encoder is not initialized.")
    texts: List[str] = []
    valid_units: List[Dict[str, Any]] = []
    for unit in units:
        text = unit.get("text")
        if not text:
            continue
        texts.append(text)
        valid_units.append(unit)
    if not texts or not question.strip():
        return []
    normalize = bool(embed_cfg.get("normalize", True))
    max_len = int(embed_cfg.get("max_len_note", 256))
    batch_size = _coerce_int(embed_cfg.get("batch_size", 16), 16)
    q_vec = encoder.encode([question.strip()], max_length=max_len, batch_size=batch_size, normalize=normalize)
    doc_vecs = encoder.encode(texts, max_length=max_len, batch_size=batch_size, normalize=normalize)
    if q_vec.size == 0 or doc_vecs.size == 0:
        return []
    query = np.asarray(q_vec[0], dtype="float32")
    doc_vecs = np.asarray(doc_vecs, dtype="float32")
    if normalize:
        scores = doc_vecs @ query
    else:
        denom = (np.linalg.norm(doc_vecs, axis=1) * (np.linalg.norm(query) + 1e-12)) + 1e-12
        scores = (doc_vecs @ query) / denom
    ranked_idx = np.argsort(scores)[::-1][: max(0, int(top_k))]
    results: List[Dict[str, Any]] = []
    for idx in ranked_idx:
        results.append({"unit": valid_units[idx], "score": float(scores[idx])})
    return results


def _build_retrieved_context(
    ranked: List[Dict[str, Any]],
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for rank, item in enumerate(ranked, start=1):
        unit = item.get("unit") or {}
        doc_id = unit.get("doc_id")
        sentence_idx = unit.get("sentence_idx")
        text = unit.get("text") or ""
        if doc_id is None or sentence_idx is None:
            continue
        note_id = f"{doc_id}#s{int(sentence_idx):04d}"
        chunk_id = f"{doc_id}#s{int(sentence_idx):04d}"
        contexts.append(
            {
                "note_id": note_id,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": doc_id,
                "sentence_idx": int(sentence_idx),
                "text": text,
                "text_hash": sha1_text(text) if text else None,
                "evidence": text,
                "canonical": f"[{doc_id}] {text}",
                "score": item.get("score"),
                "rank": rank,
                "source": mode,
            }
        )
    return contexts


def _dedup_key(ctx: Dict[str, Any], idx: int) -> Tuple[Tuple[Any, ...], str]:
    chunk_id = ctx.get("chunk_id")
    if chunk_id:
        return ("chunk_id", str(chunk_id)), "chunk_id"
    note_id = ctx.get("note_id")
    if note_id:
        return ("note_id", str(note_id)), "note_id"
    doc_id = ctx.get("doc_id")
    sent_idx = ctx.get("sentence_idx")
    if doc_id and sent_idx is not None:
        return ("doc_id", str(doc_id), int(sent_idx)), "doc_id"
    text = ctx.get("canonical") or ctx.get("evidence") or ""
    if doc_id and text:
        return ("doc_id_hash", str(doc_id), sha1_text(str(text))), "doc_id_hash"
    return ("fallback", idx), "fallback"


def _dedup_retrieved_context(
    retrieved_context: List[Dict[str, Any]],
    *,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_count = len(retrieved_context)
    seen: set[Tuple[Any, ...]] = set()
    deduped: List[Dict[str, Any]] = []
    duplicates = 0
    for idx, ctx in enumerate(retrieved_context):
        key, _ = _dedup_key(ctx, idx)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(ctx)
    unique_count = len(deduped)
    if top_k > 0:
        deduped = deduped[:top_k]
    duplicate_rate = duplicates / raw_count if raw_count else 0.0
    return deduped, {
        "top_k_raw": raw_count,
        "top_k_final": len(deduped),
        "unique_count": unique_count,
        "duplicate_rate": duplicate_rate,
    }


def _resolve_top_k_raw(
    top_k: int,
    top_k_raw: Optional[Any],
    overfetch: Optional[Any],
    min_overfetch: Optional[Any],
) -> Tuple[int, str]:
    min_factor = float(min_overfetch) if min_overfetch is not None else MIN_OVERFETCH
    min_factor = max(1.0, min_factor)
    min_raw = max(1, int(math.ceil(top_k * min_factor)))
    if top_k_raw is not None:
        raw = max(1, int(top_k_raw))
        return max(min_raw, raw), "top_k_raw"
    if overfetch is not None:
        try:
            factor = float(overfetch)
        except (TypeError, ValueError):
            factor = DEFAULT_OVERFETCH
        raw = max(1, int(math.ceil(top_k * max(1.0, factor))))
        return max(min_raw, raw), "overfetch"
    raw = max(1, int(math.ceil(top_k * DEFAULT_OVERFETCH)))
    return max(min_raw, raw), "default"


def _classify_topk_shortage(
    retrieved_context_raw: List[Dict[str, Any]],
    requested: int,
) -> str:
    if len(retrieved_context_raw) < requested:
        return "retriever_short_return"
    missing = sum(1 for ctx in retrieved_context_raw if ctx.get("chunk_id") is None)
    if missing:
        return "docstore_miss"
    return "capacity_shortage"


def _retrieve_with_backfill(
    *,
    question: str,
    units: List[Dict[str, Any]],
    mode: str,
    bm25_cfg: Dict[str, Any],
    embed_cfg: Dict[str, Any],
    dense_encoder,
    top_k: int,
    top_k_raw: int,
    backfill_max_overfetch: float,
    backfill_step: float,
    backfill_rounds: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Optional[str], int]:
    requested = max(1, int(top_k_raw))
    max_raw = requested
    if top_k > 0:
        max_raw = max(requested, int(math.ceil(top_k * float(backfill_max_overfetch))))

    attempt = 0
    last_raw_count = -1
    backfill_reason = None
    retrieved_context_raw: List[Dict[str, Any]] = []
    retrieved_context_topk: List[Dict[str, Any]] = []
    dedup_stats: Dict[str, Any] = {}

    while True:
        if mode == "bm25":
            ranked = _bm25_search(question, units, bm25_cfg, requested)
        elif mode == "dense":
            ranked = _dense_search(question, units, dense_encoder, embed_cfg, requested)
        else:
            raise ValueError(f"Unknown mode {mode}")
        retrieved_context_raw = _build_retrieved_context(ranked, mode=mode)
        retrieved_context_topk, dedup_stats = _dedup_retrieved_context(retrieved_context_raw, top_k=top_k)
        dedup_stats["top_k_raw_requested"] = requested

        if top_k <= 0 or len(retrieved_context_topk) >= top_k:
            backfill_reason = None
            break

        if requested >= max_raw:
            backfill_reason = _classify_topk_shortage(retrieved_context_raw, requested)
            break

        raw_count = len(retrieved_context_raw)
        if raw_count <= last_raw_count and raw_count < requested:
            backfill_reason = _classify_topk_shortage(retrieved_context_raw, requested)
            break
        last_raw_count = raw_count

        attempt += 1
        if attempt > max(0, int(backfill_rounds)):
            backfill_reason = _classify_topk_shortage(retrieved_context_raw, requested)
            break

        requested = int(math.ceil(requested * max(1.1, float(backfill_step))))
        if requested <= raw_count:
            requested = raw_count + max(1, int(math.ceil(top_k * 0.5)))
        requested = min(requested, max_raw)

    return (
        retrieved_context_raw,
        retrieved_context_topk,
        dedup_stats,
        backfill_reason,
        attempt,
    )


def _build_dense_encoder(embed_cfg: Dict[str, Any]):
    provider = embed_cfg.get("provider", "qwen3")
    model = embed_cfg.get("model", "qwen3-embedding")
    max_len = int(embed_cfg.get("max_len_note", 256))
    cache_dir = embed_cfg.get("cache_dir")
    device = embed_cfg.get("device")
    dtype = embed_cfg.get("dtype")
    endpoint = embed_cfg.get("endpoint")
    api_key = embed_cfg.get("api_key")
    timeout_s = embed_cfg.get("timeout_s") or embed_cfg.get("request_timeout_s")
    return get_shared_encoder(
        provider,
        model,
        max_length=max_len,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype,
        endpoint=endpoint,
        api_key=api_key,
        request_timeout_s=timeout_s,
    )


def _sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = deepcopy(cfg)
    openai_cfg = sanitized.get("openai")
    if isinstance(openai_cfg, dict) and openai_cfg.get("api_key"):
        openai_cfg["api_key"] = "***"
    retriever_cfg = sanitized.get("retriever")
    if isinstance(retriever_cfg, dict):
        embedding_cfg = retriever_cfg.get("embedding")
        if isinstance(embedding_cfg, dict) and embedding_cfg.get("api_key"):
            embedding_cfg["api_key"] = "***"
    return sanitized


def _sanitize_argv(argv: List[str]) -> List[str]:
    sanitized: List[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            sanitized.append("***")
            skip_next = False
            continue
        if arg.startswith("--openai_api_key="):
            sanitized.append("--openai_api_key=***")
            continue
        if arg == "--openai_api_key":
            sanitized.append(arg)
            skip_next = True
            continue
        sanitized.append(arg)
    return sanitized


def _git_info(repo_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
    except Exception:
        info["commit"] = None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except Exception:
        info["dirty"] = None
    return info


def _env_snapshot() -> Dict[str, Optional[str]]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "OPENAI_API_KEY",
        "RELRAG_ALLOW_CUSTOM_LLM",
        "EMB_ENDPOINT",
        "VLLM_ENDPOINT",
        "HF_ENDPOINT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ]
    snapshot: Dict[str, Optional[str]] = {}
    for key in keys:
        value = os.environ.get(key)
        if value and key == "OPENAI_API_KEY":
            snapshot[key] = "***"
        else:
            snapshot[key] = value
    return snapshot


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _accumulate_metrics(totals: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def _drain_futures(
    future_map: Dict[Any, str],
    handle,
    totals: Dict[str, float],
    raw_handle: Optional[TextIO] = None,
    topk_handle: Optional[TextIO] = None,
    progress: Optional[ProgressBar] = None,
    stall_warn_sec: float = DEFAULT_STALL_WARN_SEC,
    stall_abort_sec: float = DEFAULT_STALL_ABORT_SEC,
) -> Tuple[int, int]:
    completed = 0
    succeeded = 0
    pending = set(future_map)
    last_progress = time.time()
    stall_warn_sec = max(1.0, float(stall_warn_sec))
    stall_abort_sec = float(stall_abort_sec)
    if stall_abort_sec <= 0:
        stall_abort_sec = 0.0
    while pending:
        done, pending = wait(pending, timeout=stall_warn_sec, return_when=FIRST_COMPLETED)
        if not done:
            idle_for = time.time() - last_progress
            sample = [future_map[f] for f in list(pending)[:5]]
            logger.warning(
                "No completed futures for {:.0f}s; still waiting on {} examples (sample: {})",
                idle_for,
                len(pending),
                sample,
            )
            if stall_abort_sec and idle_for >= stall_abort_sec:
                logger.error(
                    "Aborting {} stalled examples after {:.0f}s idle",
                    len(pending),
                    idle_for,
                )
                for future in list(pending):
                    qid = future_map.get(future, "unknown")
                    if not future.cancel():
                        logger.warning("Failed to cancel stalled future qid={}", qid)
                    else:
                        logger.error("Cancelled stalled future qid={}", qid)
                if progress is not None:
                    progress.update(len(pending))
                completed += len(pending)
                pending.clear()
                break
            continue
        for future in done:
            qid = future_map.get(future, "unknown")
            try:
                record = future.result()
            except Exception as exc:
                logger.error("Failed example {}: {}", qid, exc)
            else:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if raw_handle is not None:
                    raw_payload = {
                        "qid": record.get("qid"),
                        "document_id": record.get("document_id"),
                        "question": record.get("question"),
                        "retrieved_context_raw": record.get("retrieved_context_raw") or [],
                        "top_k_raw": record.get("top_k_raw"),
                        "top_k_raw_requested": (record.get("intermediate") or {}).get("top_k_raw_requested"),
                        "top_k_raw_source": (record.get("intermediate") or {}).get("top_k_raw_source"),
                    }
                    raw_handle.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")
                    raw_handle.flush()
                if topk_handle is not None:
                    topk_payload = {
                        "qid": record.get("qid"),
                        "document_id": record.get("document_id"),
                        "question": record.get("question"),
                        "retrieved_context_topk": record.get("retrieved_context_topk") or [],
                        "top_k": record.get("top_k"),
                        "top_k_final": record.get("top_k_final"),
                        "duplicate_rate": record.get("duplicate_rate"),
                    }
                    topk_handle.write(json.dumps(topk_payload, ensure_ascii=False) + "\n")
                    topk_handle.flush()
                _accumulate_metrics(totals, record.get("metrics") or {})
                succeeded += 1
            completed += 1
            last_progress = time.time()
            if progress is not None:
                progress.update(1)
    return completed, succeeded


def _process_question(
    item: Dict[str, Any],
    *,
    cache_root: Path,
    split: str,
    context_mode: str,
    summaries_map: Dict[str, str],
    summaries_all: Dict[str, str],
    stories_dir: Optional[Path],
    base_cfg: Dict[str, Any],
    mode: str,
    top_k: int,
    top_k_raw: int,
    top_k_raw_source: str,
    backfill_max_overfetch: float,
    backfill_step: float,
    backfill_rounds: int,
    llm_endpoint: str,
    llm_model: str,
    reader: str,
    openai_cfg: Optional[Dict[str, Any]],
    llm_retry_on_empty: int,
    llm_retry_max_evidence: int,
    force_build: bool,
    doc_cache: DocumentCache,
    dense_encoder,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    doc_id = item["document_id"]
    units = _ensure_units_for_doc(
        item,
        context_mode=context_mode,
        summaries_map=summaries_map,
        summaries_all=summaries_all,
        stories_dir=stories_dir,
        force_build=force_build,
        doc_cache=doc_cache,
    )
    retriever_cfg = base_cfg.get("retriever") or {}
    bm25_cfg = retriever_cfg.get("bm25") or {}
    embed_cfg = retriever_cfg.get("embedding") or {}
    (
        retrieved_context_raw,
        retrieved_context_topk,
        dedup_stats,
        top_k_fill_reason,
        backfill_attempts,
    ) = _retrieve_with_backfill(
        question=item["question"],
        units=units,
        mode=mode,
        bm25_cfg=bm25_cfg,
        embed_cfg=embed_cfg,
        dense_encoder=dense_encoder,
        top_k=top_k,
        top_k_raw=top_k_raw,
        backfill_max_overfetch=backfill_max_overfetch,
        backfill_step=backfill_step,
        backfill_rounds=backfill_rounds,
    )
    evidences = retrieved_context_topk

    raw_answer, prompt_meta, llm_error, llm_error_reason = generate_answer(
        question=item["question"],
        evidences=evidences,
        reader=reader,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
        openai_cfg=openai_cfg,
        base_cfg=base_cfg,
        run_dir=run_dir,
    )
    short_answer, answer_source, answer_source_detail = resolve_short_answer(
        None,
        raw_answer,
        question=question,
    )
    if answer_source == "empty":
        answer_source = "llm_fallback"
        answer_source_detail["fallback_override"] = "empty"
    fallback_reason = None
    if llm_error_reason:
        fallback_reason = llm_error_reason
    elif answer_source == "llm_fallback":
        if not str(raw_answer or "").strip():
            fallback_reason = "empty_output"
            if not llm_error:
                llm_error = "empty_output"
        elif not has_final_tag(str(raw_answer)):
            fallback_reason = "parse_error"
            if not llm_error:
                llm_error = "missing_final_tag"

    llm_retry_used = False
    llm_retry_source = None
    llm_retry_reason = None
    if reader == "vllm" and llm_retry_on_empty > 0 and answer_source == "llm_fallback":
        retry_evidences = evidences
        if llm_retry_max_evidence > 0:
            retry_evidences = evidences[: int(llm_retry_max_evidence)]
        for _ in range(max(1, int(llm_retry_on_empty))):
            retry_raw, retry_meta, retry_error, retry_error_reason = generate_answer(
                question=item["question"],
                evidences=retry_evidences,
                reader=reader,
                llm_endpoint=llm_endpoint,
                llm_model=llm_model,
                openai_cfg=openai_cfg,
                base_cfg=base_cfg,
                run_dir=run_dir,
            )
            retry_short, retry_source, retry_detail = resolve_short_answer(
                None,
                retry_raw,
                question=question,
            )
            if retry_source == "empty":
                retry_source = "llm_fallback"
                retry_detail["fallback_override"] = "empty"
            retry_fallback_reason = None
            if retry_error_reason:
                retry_fallback_reason = retry_error_reason
            elif retry_source == "llm_fallback":
                if not str(retry_raw or "").strip():
                    retry_fallback_reason = "empty_output"
                elif not has_final_tag(str(retry_raw)):
                    retry_fallback_reason = "parse_error"
            llm_retry_used = True
            llm_retry_source = retry_source
            llm_retry_reason = retry_fallback_reason or retry_error_reason
            if retry_source in {"llm_final", "structured_answer"}:
                raw_answer = retry_raw
                prompt_meta = retry_meta
                llm_error = retry_error
                llm_error_reason = retry_error_reason
                short_answer = retry_short
                answer_source = retry_source
                answer_source_detail = retry_detail
                fallback_reason = retry_fallback_reason
                break

    answer_model = openai_cfg.get("model") if reader == "openai" and openai_cfg else llm_model
    metrics = score_metrics(short_answer, item["references"])
    top_k_raw_value = dedup_stats.get("top_k_raw")
    overfetch_factor = None
    if isinstance(top_k_raw_value, (int, float)) and top_k:
        overfetch_factor = float(top_k_raw_value) / float(top_k)
    top_k_raw_source_final = top_k_raw_source
    if backfill_attempts > 0:
        top_k_raw_source_final = "backfill"
    llm_input_hash = prompt_meta.get("llm_input_hash") or ""

    return {
        "qid": item["qid"],
        "document_id": doc_id,
        "split": split,
        "mode": mode,
        "reader": reader,
        "model": answer_model,
        "question": item["question"],
        "answer": short_answer,
        "short_answer": short_answer,
        "answer_source": answer_source,
        "answer_source_detail": answer_source_detail,
        "prediction": short_answer,
        "references": item["references"],
        "metrics": metrics,
        "retrieved_context_raw": retrieved_context_raw,
        "retrieved_context_topk": retrieved_context_topk,
        "retrieved_context": retrieved_context_topk,
        "top_k": top_k,
        "top_k_raw": dedup_stats.get("top_k_raw"),
        "top_k_final": dedup_stats.get("top_k_final"),
        "duplicate_rate": dedup_stats.get("duplicate_rate"),
        "overfetch_factor": overfetch_factor,
        "fallback_reason": fallback_reason,
        "llm_error": llm_error,
        "top_k_fill_reason": top_k_fill_reason,
        "llm_input_hash": llm_input_hash,
        "meta": {
            "retrieval_mode": mode,
            "reader": reader,
            "model": answer_model,
            "context_mode": context_mode,
            "top_k": top_k,
            "top_k_raw": dedup_stats.get("top_k_raw"),
            "top_k_raw_requested": dedup_stats.get("top_k_raw_requested"),
            "top_k_raw_source": top_k_raw_source_final,
            "top_k_backfill_rounds": backfill_attempts,
            "top_k_fill_reason": top_k_fill_reason,
        },
        "intermediate": {
            "llm_raw": raw_answer,
            "llm_has_final": has_final_tag(raw_answer),
            "retrieval_mode": mode,
            "reader": reader,
            "model": answer_model,
            "context_mode": context_mode,
            "top_k": top_k,
            "top_k_raw": dedup_stats.get("top_k_raw"),
            "top_k_raw_requested": dedup_stats.get("top_k_raw_requested"),
            "top_k_raw_source": top_k_raw_source_final,
            "top_k_backfill_rounds": backfill_attempts,
            "top_k_fill_reason": top_k_fill_reason,
            "prompt_name": prompt_meta.get("prompt_name"),
            "prompt_template_hash": prompt_meta.get("prompt_template_hash"),
            "system_prompt_name": prompt_meta.get("system_prompt_name"),
            "system_prompt_hash": prompt_meta.get("system_prompt_hash"),
            "llm_input_hash": llm_input_hash,
            "context_count": len(units),
            "retrieved_count": len(retrieved_context_topk),
            "llm_retry_used": llm_retry_used,
            "llm_retry_source": llm_retry_source,
            "llm_retry_reason": llm_retry_reason,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NarrativeQA baseline entry (BM25/Dense)")
    parser.add_argument("--config", help="Path to YAML config file (defaults to relrag/config/config.yaml)")
    parser.add_argument("--qaps", help="Path to NarrativeQA qaps.csv (fallback to config)")
    parser.add_argument("--summaries", help="Path to NarrativeQA summaries.csv (fallback to config)")
    parser.add_argument("--stories_dir", help="Directory containing full stories (story-as-context)")
    parser.add_argument("--split", help="Dataset split: train, valid, or test (fallback to config)")
    parser.add_argument("--context_mode", help="summary-as-context or story-as-context (fallback to config)")
    parser.add_argument("--retriever", help="Retriever mode: bm25 or dense (fallback to config)")
    parser.add_argument("--modes", help="Retrieval modes: bm25,dense (fallback to config)")
    parser.add_argument("--endpoint", help="vLLM endpoint (defaults to config)")
    parser.add_argument("--model", help="LLM model name (defaults to config)")
    parser.add_argument("--reader", help="Reader backend: vllm or openai (fallback to config)")
    parser.add_argument("--openai_model", help="OpenAI model name (fallback to config)")
    parser.add_argument("--openai_api_key", help="OpenAI API key (reads env if omitted)")
    parser.add_argument("--openai_temperature", type=float, help="OpenAI temperature (fallback to config)")
    parser.add_argument("--openai_max_tokens", type=int, help="OpenAI max tokens (fallback to config)")
    parser.add_argument("--top_k", type=int, help="Top-k retrieval fanout (fallback to config)")
    parser.add_argument("--top_k_raw", type=int, help="Raw top-k before dedup (fallback to config/overfetch)")
    parser.add_argument("--overfetch", type=float, help="Overfetch multiplier before dedup (fallback to config)")
    parser.add_argument("--min_overfetch", type=float, help="Minimum overfetch multiplier (fallback to config)")
    parser.add_argument("--backfill_max_overfetch", type=float, help="Max overfetch multiplier for backfill (fallback to config)")
    parser.add_argument("--backfill_step", type=float, help="Backfill growth factor per retry (fallback to config)")
    parser.add_argument("--backfill_rounds", type=int, help="Max backfill attempts (fallback to config)")
    parser.add_argument("--llm_retry_on_empty", type=int, help="Retry LLM on empty/parse fallback (fallback to config)")
    parser.add_argument("--llm_retry_max_evidence", type=int, help="Max evidences on retry (fallback to config)")
    parser.add_argument("--limit", type=int, help="Process only first N examples (fallback to config)")
    parser.add_argument("--max_examples", type=int, help="Alias for --limit")
    parser.add_argument("--workers", type=int, help="Parallel workers (fallback to config)")
    parser.add_argument("--cache_dir", help="Cache root for per-document indexes (fallback to config)")
    parser.add_argument("--output_dir", help="Output directory (fallback to config)")
    parser.add_argument("--run_dir", help="Write run artifacts to this directory (single mode/reader only)")
    parser.add_argument("--resume", action="store_true", help="Skip run if run_dir already has completed.json")
    parser.add_argument("--force_build", action="store_true", help="Rebuild caches even if cached")
    parser.add_argument("--stall_warn_sec", type=float, help="Warn if no worker finishes within this many seconds (fallback to config)")
    parser.add_argument("--stall_abort_sec", type=float, help="Abort pending workers after this many idle seconds (0 to disable, fallback to config)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent

    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else repo_root / path

    cfg = ConfigLoader(args.config).load_config() if args.config else global_config.load_config()
    dataset_cfg = get_dataset_config(cfg, "narrativeqa")
    entry_cfg = _load_entry_config(cfg)
    args.qaps = _pick_arg(args, entry_cfg, dataset_cfg, "qaps", None)
    args.summaries = _pick_arg(args, entry_cfg, dataset_cfg, "summaries", None)
    args.stories_dir = _pick_arg(args, entry_cfg, dataset_cfg, "stories_dir", None)
    args.split = _normalize_split(_pick_arg(args, entry_cfg, dataset_cfg, "split", DEFAULT_SPLIT))
    args.context_mode = _normalize_context_mode(
        _pick_arg(args, entry_cfg, dataset_cfg, "context_mode", DEFAULT_CONTEXT_MODE)
    )
    args.cache_dir = _pick_arg(args, entry_cfg, dataset_cfg, "cache_dir", DEFAULT_CACHE_DIR)
    args.output_dir = _pick_arg(args, entry_cfg, dataset_cfg, "output_dir", DEFAULT_OUTPUT_DIR)
    args.top_k = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "top_k", DEFAULT_TOP_K),
        DEFAULT_TOP_K,
    )
    args.top_k_raw = _pick_arg(args, entry_cfg, dataset_cfg, "top_k_raw", None)
    args.overfetch = _pick_arg(args, entry_cfg, dataset_cfg, "overfetch", None)
    args.min_overfetch = _coerce_float(
        _pick_arg(args, entry_cfg, dataset_cfg, "min_overfetch", MIN_OVERFETCH),
        MIN_OVERFETCH,
    )
    args.backfill_max_overfetch = _coerce_float(
        _pick_arg(args, entry_cfg, dataset_cfg, "backfill_max_overfetch", DEFAULT_BACKFILL_MAX_OVERFETCH),
        DEFAULT_BACKFILL_MAX_OVERFETCH,
    )
    args.backfill_step = _coerce_float(
        _pick_arg(args, entry_cfg, dataset_cfg, "backfill_step", DEFAULT_BACKFILL_STEP),
        DEFAULT_BACKFILL_STEP,
    )
    args.backfill_rounds = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "backfill_rounds", DEFAULT_BACKFILL_ROUNDS),
        DEFAULT_BACKFILL_ROUNDS,
    )
    args.llm_retry_on_empty = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "llm_retry_on_empty", DEFAULT_LLM_RETRY_ON_EMPTY),
        DEFAULT_LLM_RETRY_ON_EMPTY,
    )
    args.llm_retry_max_evidence = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "llm_retry_max_evidence", DEFAULT_LLM_RETRY_EVIDENCE),
        DEFAULT_LLM_RETRY_EVIDENCE,
    )
    args.limit = _coerce_int(_pick_arg(args, entry_cfg, dataset_cfg, "limit", DEFAULT_LIMIT), DEFAULT_LIMIT)
    if args.max_examples is not None:
        args.limit = _coerce_int(args.max_examples, args.limit)
    args.workers = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "workers", DEFAULT_WORKERS),
        DEFAULT_WORKERS,
    )
    args.stall_warn_sec = _coerce_float(
        _pick_arg(args, entry_cfg, dataset_cfg, "stall_warn_sec", DEFAULT_STALL_WARN_SEC),
        DEFAULT_STALL_WARN_SEC,
    )
    args.stall_abort_sec = _coerce_float(
        _pick_arg(args, entry_cfg, dataset_cfg, "stall_abort_sec", DEFAULT_STALL_ABORT_SEC),
        DEFAULT_STALL_ABORT_SEC,
    )
    if entry_cfg.get("force_build"):
        args.force_build = True

    openai_overrides: Dict[str, Any] = {}
    if args.openai_model:
        openai_overrides["model"] = args.openai_model
    if args.openai_temperature is not None:
        openai_overrides["temperature"] = args.openai_temperature
    if args.openai_max_tokens is not None:
        openai_overrides["max_tokens"] = args.openai_max_tokens

    openai_cfg = resolve_openai_config(cfg, dataset_cfg, overrides=openai_overrides)
    if args.openai_api_key:
        env_name = openai_cfg.get("api_key_env", "OPENAI_API_KEY")
        os.environ[str(env_name)] = args.openai_api_key

    if args.seed is not None:
        random.seed(int(args.seed))
        if np is not None:
            try:
                np.random.seed(int(args.seed))
            except Exception:
                pass

    if not args.qaps or not args.summaries:
        raise ValueError("qaps and summaries paths are required (use args or config)")

    qaps_path = _resolve_path(args.qaps)
    summaries_path = _resolve_path(args.summaries)
    if not qaps_path.exists():
        raise FileNotFoundError(f"qaps.csv not found: {qaps_path}")
    if not summaries_path.exists():
        raise FileNotFoundError(f"summaries.csv not found: {summaries_path}")

    stories_dir = _resolve_path(args.stories_dir) if args.stories_dir else None
    if args.context_mode == "story" and stories_dir is None:
        raise ValueError("stories_dir is required for story-as-context mode")
    if stories_dir and not stories_dir.exists():
        raise FileNotFoundError(f"stories_dir not found: {stories_dir}")

    llm_endpoint, llm_model = _resolve_llm_config(args, cfg)
    logger.info("Using vLLM endpoint={} model={}", llm_endpoint, llm_model)
    cache_root = _resolve_path(args.cache_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = args.split
    logger.info("Loading NarrativeQA split={} from {}", split, qaps_path)
    questions = _load_qaps(qaps_path, split)
    summaries_map, summaries_all = _load_summaries(summaries_path, split)

    if args.limit:
        questions = questions[: args.limit]
    total_examples = len(questions)
    if total_examples == 0:
        logger.warning("No examples found for split {}", split)
        return

    modes = _resolve_retriever_modes(
        mode_arg=args.retriever,
        modes_arg=args.modes,
        entry_cfg=entry_cfg,
        dataset_cfg=dataset_cfg,
    )
    readers = _resolve_readers(args, cfg, dataset_cfg)
    base_cfg = _apply_dataset_retriever(deepcopy(cfg), "narrativeqa")
    run_dir = _resolve_path(args.run_dir) if args.run_dir else None
    if run_dir:
        if len(readers) != 1 or len(modes) != 1:
            raise ValueError("--run_dir requires a single reader/mode selection.")
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.resume and (run_dir / "completed.json").exists():
            logger.info("Run already completed at {}; skipping (--resume).", run_dir)
            return
        base_cfg.setdefault("runtime", {})["run_dir"] = str(run_dir)
    openai_runtime_cfg: Optional[Dict[str, Any]] = None
    if "openai" in readers:
        if not openai_cfg.get("enabled", True):
            raise ValueError("OpenAI mode disabled in config (openai.enabled=false).")
        openai_cfg["api_key"] = resolve_openai_api_key(openai_cfg)
        openai_runtime_cfg = openai_cfg
        logger.info(
            "OpenAI reader enabled base_url={} model={} temperature={} max_tokens={} api_key_env={}",
            openai_runtime_cfg.get("base_url"),
            openai_runtime_cfg.get("model"),
            openai_runtime_cfg.get("temperature"),
            openai_runtime_cfg.get("max_tokens"),
            openai_runtime_cfg.get("api_key_env"),
        )

    doc_cache = DocumentCache()
    retriever_cfg = base_cfg.get("retriever") or {}
    bm25_cfg = retriever_cfg.get("bm25") or {}
    embed_cfg = retriever_cfg.get("embedding") or {}
    dense_encoder = None
    if "dense" in modes:
        dense_encoder = _build_dense_encoder(embed_cfg)

    summary_report: Dict[str, Any] = {
        "split": split,
        "context_mode": args.context_mode,
        "readers": readers,
        "retrievers": modes,
        "runs": {},
    }

    for reader in readers:
        reader_openai_cfg = openai_runtime_cfg if reader == "openai" else None
        answer_model = reader_openai_cfg.get("model") if reader == "openai" and reader_openai_cfg else llm_model
        summary_report["runs"].setdefault(reader, {})
        for mode in modes:
            topk_cfg = bm25_cfg if mode == "bm25" else embed_cfg
            mode_top_k = _coerce_int(topk_cfg.get("top_k", args.top_k), args.top_k)
            mode_top_k_raw, top_k_raw_source = _resolve_top_k_raw(
                mode_top_k,
                args.top_k_raw,
                args.overfetch,
                args.min_overfetch,
            )
            output_base = run_dir if run_dir else output_dir
            output_name = "predictions.jsonl" if run_dir else _pred_filename(split, reader, mode, len(readers), len(modes))
            output_path = output_base / output_name
            retrieval_raw_path = run_dir / "retrieval_raw.jsonl" if run_dir else None
            retrieval_topk_path = run_dir / "retrieval_topk.jsonl" if run_dir else None
            run_started_at = time.time()
            if run_dir:
                resolved_cfg = deepcopy(base_cfg)
                resolved_cfg.setdefault("vllm", {})["endpoint"] = llm_endpoint
                resolved_cfg.setdefault("vllm", {})["model"] = llm_model
                if openai_runtime_cfg:
                    resolved_cfg["openai"] = deepcopy(openai_runtime_cfg)
                entry_snapshot = resolved_cfg.setdefault("narrativeqa_entry", {})
                entry_snapshot.update(
                    {
                        "qaps": str(qaps_path),
                        "summaries": str(summaries_path),
                        "stories_dir": str(stories_dir) if stories_dir else None,
                        "split": split,
                        "context_mode": args.context_mode,
                        "reader": reader,
                        "retriever": mode,
                        "top_k": mode_top_k,
                        "top_k_raw": mode_top_k_raw,
                        "overfetch": args.overfetch,
                        "min_overfetch": args.min_overfetch,
                        "backfill_max_overfetch": args.backfill_max_overfetch,
                        "backfill_step": args.backfill_step,
                        "backfill_rounds": args.backfill_rounds,
                        "llm_retry_on_empty": args.llm_retry_on_empty,
                        "llm_retry_max_evidence": args.llm_retry_max_evidence,
                        "limit": args.limit,
                        "seed": args.seed,
                    }
                )
                _write_json(run_dir / "config.resolved.json", _sanitize_config(resolved_cfg))
                run_meta = {
                    "run_dir": str(run_dir),
                    "split": split,
                    "context_mode": args.context_mode,
                    "reader": reader,
                    "retriever": mode,
                    "top_k": mode_top_k,
                    "top_k_raw": mode_top_k_raw,
                    "top_k_raw_source": top_k_raw_source,
                    "limit": args.limit,
                    "seed": args.seed,
                    "command": _sanitize_argv(sys.argv),
                    "timestamp": int(time.time()),
                    "host": socket.gethostname(),
                    "platform": platform.platform(),
                    "python": sys.version,
                    "git": _git_info(repo_root),
                    "env": _env_snapshot(),
                }
                _write_json(run_dir / "run_meta.json", run_meta)

            logger.info("Running reader={} mode={} -> {}", reader, mode, output_path)
            totals = {"bleu1": 0.0, "bleu4": 0.0, "rougeL": 0.0, "meteor": 0.0}
            progress = ProgressBar(total_examples)
            processed = 0
            completed = 0
            if run_dir:
                raw_handle = retrieval_raw_path.open("w", encoding="utf-8")
                topk_handle = retrieval_topk_path.open("w", encoding="utf-8")
            else:
                raw_handle = None
                topk_handle = None
            try:
                with output_path.open("w", encoding="utf-8") as handle:
                    if args.workers <= 1:
                        for item in questions:
                            try:
                                record = _process_question(
                                    item,
                                    cache_root=cache_root,
                                    split=split,
                                    context_mode=args.context_mode,
                                    summaries_map=summaries_map,
                                    summaries_all=summaries_all,
                                    stories_dir=stories_dir,
                                    base_cfg=base_cfg,
                                    mode=mode,
                                    top_k=mode_top_k,
                                    top_k_raw=mode_top_k_raw,
                                    top_k_raw_source=top_k_raw_source,
                                    backfill_max_overfetch=args.backfill_max_overfetch,
                                    backfill_step=args.backfill_step,
                                    backfill_rounds=args.backfill_rounds,
                                    llm_endpoint=llm_endpoint,
                                    llm_model=llm_model,
                                    reader=reader,
                                    openai_cfg=reader_openai_cfg,
                                    llm_retry_on_empty=args.llm_retry_on_empty,
                                    llm_retry_max_evidence=args.llm_retry_max_evidence,
                                    force_build=args.force_build,
                                    doc_cache=doc_cache,
                                    dense_encoder=dense_encoder,
                                    run_dir=str(run_dir) if run_dir else None,
                                )
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                handle.flush()
                                if raw_handle is not None:
                                    raw_payload = {
                                        "qid": record.get("qid"),
                                        "document_id": record.get("document_id"),
                                        "question": record.get("question"),
                                        "retrieved_context_raw": record.get("retrieved_context_raw") or [],
                                        "top_k_raw": record.get("top_k_raw"),
                                        "top_k_raw_requested": (record.get("intermediate") or {}).get("top_k_raw_requested"),
                                        "top_k_raw_source": (record.get("intermediate") or {}).get("top_k_raw_source"),
                                    }
                                    raw_handle.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")
                                    raw_handle.flush()
                                if topk_handle is not None:
                                    topk_payload = {
                                        "qid": record.get("qid"),
                                        "document_id": record.get("document_id"),
                                        "question": record.get("question"),
                                        "retrieved_context_topk": record.get("retrieved_context_topk") or [],
                                        "top_k": record.get("top_k"),
                                        "top_k_final": record.get("top_k_final"),
                                        "duplicate_rate": record.get("duplicate_rate"),
                                    }
                                    topk_handle.write(json.dumps(topk_payload, ensure_ascii=False) + "\n")
                                    topk_handle.flush()
                                _accumulate_metrics(totals, record.get("metrics") or {})
                                processed += 1
                            except Exception as exc:
                                logger.error("Failed example {}: {}", item.get("qid"), exc)
                            finally:
                                completed += 1
                                progress.update(1)
                    else:
                        max_workers = max(1, int(args.workers))
                        buffer_cap = max_workers * 2
                        future_map: Dict[Any, str] = {}
                        scheduled = 0
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            for item in questions:
                                future = executor.submit(
                                    _process_question,
                                    item,
                                    cache_root=cache_root,
                                    split=split,
                                    context_mode=args.context_mode,
                                    summaries_map=summaries_map,
                                    summaries_all=summaries_all,
                                    stories_dir=stories_dir,
                                    base_cfg=base_cfg,
                                    mode=mode,
                                    top_k=mode_top_k,
                                    top_k_raw=mode_top_k_raw,
                                    top_k_raw_source=top_k_raw_source,
                                    backfill_max_overfetch=args.backfill_max_overfetch,
                                    backfill_step=args.backfill_step,
                                    backfill_rounds=args.backfill_rounds,
                                    llm_endpoint=llm_endpoint,
                                    llm_model=llm_model,
                                    reader=reader,
                                    openai_cfg=reader_openai_cfg,
                                    llm_retry_on_empty=args.llm_retry_on_empty,
                                    llm_retry_max_evidence=args.llm_retry_max_evidence,
                                    force_build=args.force_build,
                                    doc_cache=doc_cache,
                                    dense_encoder=dense_encoder,
                                )
                                future_map[future] = item.get("qid", "unknown")
                                scheduled += 1
                                if len(future_map) >= buffer_cap:
                                    done_count, ok_count = _drain_futures(
                                        future_map,
                                        handle,
                                        totals,
                                        raw_handle=raw_handle,
                                        topk_handle=topk_handle,
                                        progress=progress,
                                        stall_warn_sec=args.stall_warn_sec,
                                        stall_abort_sec=args.stall_abort_sec,
                                    )
                                    completed += done_count
                                    processed += ok_count
                                    future_map = {}
                            if future_map:
                                done_count, ok_count = _drain_futures(
                                    future_map,
                                    handle,
                                    totals,
                                    raw_handle=raw_handle,
                                    topk_handle=topk_handle,
                                    progress=progress,
                                    stall_warn_sec=args.stall_warn_sec,
                                    stall_abort_sec=args.stall_abort_sec,
                                )
                                completed += done_count
                                processed += ok_count
            finally:
                if raw_handle is not None:
                    raw_handle.close()
                if topk_handle is not None:
                    topk_handle.close()

            progress.close()
            failed = completed - processed
            logger.info("Reader {} mode {} completed {} examples (failed {})", reader, mode, processed, failed)
            denom = processed if processed > 0 else 1
            summary_report["runs"][reader][mode] = {
                "bleu1": round(totals["bleu1"] / denom, 4),
                "bleu4": round(totals["bleu4"] / denom, 4),
                "rougeL": round(totals["rougeL"] / denom, 4),
                "meteor": round(totals["meteor"] / denom, 4),
                "count": processed,
                "model": answer_model,
                "top_k": mode_top_k,
                "top_k_raw": mode_top_k_raw,
            }
            if run_dir:
                duration_sec = max(0.0, time.time() - run_started_at)
                metrics_payload = {
                    "split": split,
                    "context_mode": args.context_mode,
                    "reader": reader,
                    "retriever": mode,
                    "model": answer_model,
                    "count": processed,
                    "failed": failed,
                    "duration_sec": round(duration_sec, 2),
                    "metrics": summary_report["runs"][reader][mode],
                    "top_k": mode_top_k,
                    "top_k_raw": mode_top_k_raw,
                    "top_k_raw_source": top_k_raw_source,
                }
                _write_json(run_dir / "metrics.json", metrics_payload)
                _write_json(
                    run_dir / "completed.json",
                    {
                        "status": "ok" if failed == 0 else "partial",
                        "processed": processed,
                        "failed": failed,
                        "duration_sec": round(duration_sec, 2),
                        "timestamp": int(time.time()),
                    },
                )

    if len(readers) == 1:
        summary_report["modes"] = summary_report["runs"][readers[0]]
    if len(modes) == 1:
        summary_report["models"] = {reader: summary_report["runs"][reader][modes[0]] for reader in readers}

    if not run_dir:
        summary_path = output_dir / f"summary_{split}.json"
        summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Summary written to {}", summary_path)


if __name__ == "__main__":
    main()
