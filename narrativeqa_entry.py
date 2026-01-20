import argparse
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

from loguru import logger

from relrag.api import build_index, retrieve, answer
from relrag.config.dataset_config import (
    get_dataset_config,
    resolve_openai_api_key,
    resolve_openai_config,
    resolve_reader,
)
from relrag.config.config_loader import ConfigLoader, config as global_config
from relrag.indexer.bm25_index import BM25IndexBuilder
from relrag.indexer.embedding_index import EmbeddingIndexBuilder
from relrag.utils.openai_answer import generate_openai_answer as _generate_openai_answer
from relrag.utils.output_eval import extract_final_answer
from relrag.utils.eval_metrics import score_metrics


DEFAULT_TOP_K = 10
DEFAULT_LIMIT = 0
DEFAULT_WORKERS = 1
DEFAULT_CACHE_DIR = "result/narrativeqa/cache"
DEFAULT_OUTPUT_DIR = "result/narrativeqa"
DEFAULT_CONTEXT_MODE = "summary"
DEFAULT_MODES = ("bm25", "dense", "hybrid")
DEFAULT_SPLIT = "valid"
DEFAULT_STALL_WARN_SEC = 300.0
DEFAULT_STALL_ABORT_SEC = 900.0
FIXED_LLM_MODEL = "qwen3-30b-a3b"


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

    def lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock


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
    allowed = {"structured", "dense", "bm25", "hybrid"}
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


def _mode_config(cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    retriever_cfg = cfg.get("retriever") or {}
    if mode == "dense":
        dense_cfg = retriever_cfg.get("dense")
        if isinstance(dense_cfg, dict):
            return dense_cfg
        return retriever_cfg.get("embedding") or {}
    if mode == "bm25":
        return retriever_cfg.get("bm25") or {}
    if mode == "hybrid":
        return retriever_cfg.get("hybrid") or {}
    if mode == "structured":
        return retriever_cfg.get("structured") or {}
    return {}


def _mode_enabled(cfg: Dict[str, Any], mode: str) -> bool:
    mode_cfg = _mode_config(cfg, mode)
    return bool(mode_cfg.get("enabled", True))


def _resolve_mode_top_k(mode: str, cfg: Dict[str, Any], fallback: int) -> int:
    mode_cfg = _mode_config(cfg, mode)
    if "top_k" not in mode_cfg:
        return fallback
    return _coerce_int(mode_cfg.get("top_k"), fallback)


def _resolve_retriever_modes(
    *,
    mode_arg: Optional[str],
    modes_arg: Optional[str],
    entry_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
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

    modes = _parse_modes(raw) if raw is not None else list(DEFAULT_MODES)
    enabled_modes = [mode for mode in modes if _mode_enabled(base_cfg, mode)]
    if not enabled_modes:
        raise ValueError("No retriever modes enabled in config.")
    return enabled_modes


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
    endpoint = args.endpoint or (cfg.get("vllm") or {}).get("endpoint")
    if not endpoint:
        raise ValueError("LLM endpoint is required (use --endpoint or config vllm.endpoint)")
    if args.model and args.model != FIXED_LLM_MODEL:
        logger.warning("Ignoring --model {}; NarrativeQA uses {}", args.model, FIXED_LLM_MODEL)
    model_cfg = (cfg.get("vllm") or {}).get("model")
    if model_cfg and model_cfg != FIXED_LLM_MODEL:
        logger.warning("Overriding config model {} -> {}", model_cfg, FIXED_LLM_MODEL)
    return endpoint, FIXED_LLM_MODEL


def generate_openai_answer(question: str, evidences: List[Dict[str, Any]], openai_cfg: Dict[str, Any]) -> str:
    return _generate_openai_answer(question, evidences, openai_cfg)


def generate_answer(
    question: str,
    evidences: List[Dict[str, Any]],
    *,
    reader: str,
    llm_endpoint: str,
    llm_model: str,
    openai_cfg: Optional[Dict[str, Any]],
) -> str:
    if reader == "vllm":
        return answer(
            question=question,
            evidences=evidences,
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
        )
    if reader == "openai":
        if not openai_cfg:
            raise ValueError("OpenAI config missing for reader=openai")
        return generate_openai_answer(question, evidences, openai_cfg)
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


def _ensure_doc_text(
    doc_id: str,
    *,
    context_mode: str,
    docs_dir: Path,
    summaries_map: Dict[str, str],
    summaries_all: Dict[str, str],
    stories_dir: Optional[Path],
    force_build: bool,
) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_path = docs_dir / f"{doc_id}.txt"
    if doc_path.exists() and not force_build:
        return doc_path
    if context_mode == "summary":
        text = summaries_map.get(doc_id) or summaries_all.get(doc_id)
        if not text:
            raise KeyError(f"Summary not found for document {doc_id}")
    else:
        if stories_dir is None:
            raise ValueError("stories_dir is required for story-as-context mode")
        story_path = _find_story_path(doc_id, stories_dir)
        if not story_path:
            raise FileNotFoundError(f"Story file not found for document {doc_id}")
        text = _read_text(story_path)
        if not text.strip():
            raise ValueError(f"Story file empty for document {doc_id}")
    with doc_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return doc_path


def _ensure_index(
    docs_dir: Path,
    index_root: Path,
    llm_endpoint: str,
    llm_model: str,
    force_build: bool,
) -> Dict[str, Any]:
    notes_path = index_root / "notes.jsonl"
    indexes_dir = index_root / "indexes"
    if not force_build and notes_path.exists() and indexes_dir.exists():
        return {"status": "reused"}
    index_root.mkdir(parents=True, exist_ok=True)
    return build_index(
        docs_input=str(docs_dir),
        output_dir=str(index_root),
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
    )


def _apply_dataset_retriever(cfg: Dict[str, Any], dataset_key: str) -> Dict[str, Any]:
    dataset_cfg = get_dataset_config(cfg, dataset_key)
    retriever_override = dataset_cfg.get("retriever") if isinstance(dataset_cfg, dict) else None
    if isinstance(retriever_override, dict):
        base_retriever = cfg.get("retriever") or {}
        cfg["retriever"] = _deep_merge(base_retriever, retriever_override)
    return cfg


def _prepare_aux_config(base_cfg: Dict[str, Any], doc_root: Path) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    notes_path = doc_root / "notes.jsonl"
    indexes_root = doc_root / "indexes"
    cfg.setdefault("notes", {})["out_path"] = str(notes_path)
    retriever_cfg = cfg.setdefault("retriever", {})
    embed_cfg = retriever_cfg.setdefault("embedding", {})
    bm25_cfg = retriever_cfg.setdefault("bm25", {})
    embed_cfg["offline_index_path"] = str(indexes_root / "faiss" / "notes.faiss")
    embed_cfg["meta_path"] = str(indexes_root / "faiss" / "notes.meta.parquet")
    bm25_cfg["store_path"] = str(indexes_root / "bm25" / "notes")
    return cfg


def _ensure_aux_indexes(
    cfg: Dict[str, Any],
    *,
    build_embedding: bool,
    build_bm25: bool,
    force_build: bool,
) -> Dict[str, str]:
    stats: Dict[str, str] = {}
    notes_path = Path(cfg.get("notes", {}).get("out_path", ""))
    if not notes_path.exists():
        if build_embedding:
            stats["embedding"] = "skipped_missing_notes"
        if build_bm25:
            stats["bm25"] = "skipped_missing_notes"
        return stats

    retriever_cfg = cfg.get("retriever") or {}
    embed_cfg = retriever_cfg.get("embedding") or {}
    bm25_cfg = retriever_cfg.get("bm25") or {}

    if build_embedding:
        embed_cfg["enabled"] = True
        embed_index = Path(embed_cfg.get("offline_index_path", ""))
        embed_meta = Path(embed_cfg.get("meta_path", ""))
        embed_ready = embed_index.exists() and embed_meta.exists()
        if force_build or not embed_ready:
            try:
                EmbeddingIndexBuilder(cfg).build()
                stats["embedding"] = "ok"
            except Exception as exc:
                logger.warning("Embedding index build failed {}: {}", notes_path, exc)
                stats["embedding"] = f"error:{exc}"
        else:
            stats["embedding"] = "reused"
    else:
        stats["embedding"] = "disabled"

    if build_bm25:
        bm25_cfg["enabled"] = True
        bm25_corpus = Path(bm25_cfg.get("store_path", "")) / "notes.jsonl"
        if force_build or not bm25_corpus.exists():
            try:
                BM25IndexBuilder(cfg).build()
                stats["bm25"] = "ok"
            except Exception as exc:
                logger.warning("BM25 index build failed {}: {}", notes_path, exc)
                stats["bm25"] = f"error:{exc}"
        else:
            stats["bm25"] = "reused"
    else:
        stats["bm25"] = "disabled"

    return stats


def _prepare_retriever_config(
    base_cfg: Dict[str, Any],
    doc_root: Path,
    mode: str,
) -> Dict[str, Any]:
    cfg = _prepare_aux_config(base_cfg, doc_root)
    retriever_cfg = cfg.setdefault("retriever", {})
    structured_cfg = retriever_cfg.setdefault("structured", {})
    embed_cfg = retriever_cfg.setdefault("embedding", {})
    bm25_cfg = retriever_cfg.setdefault("bm25", {})
    hybrid_cfg = retriever_cfg.setdefault("hybrid", {})

    mode = mode.lower()
    if mode == "structured":
        structured_cfg["enabled"] = True
        structured_cfg["vector_fallback_enabled"] = False
        embed_cfg["enabled"] = False
        bm25_cfg["enabled"] = False
        hybrid_cfg["enabled"] = False
    elif mode == "dense":
        structured_cfg["enabled"] = False
        embed_cfg["enabled"] = True
        bm25_cfg["enabled"] = False
        hybrid_cfg["enabled"] = True
    elif mode == "bm25":
        structured_cfg["enabled"] = False
        embed_cfg["enabled"] = False
        bm25_cfg["enabled"] = True
        hybrid_cfg["enabled"] = True
    elif mode == "hybrid":
        structured_cfg["enabled"] = True
        embed_cfg["enabled"] = True
        bm25_cfg["enabled"] = True
        hybrid_cfg["enabled"] = True
    else:
        raise ValueError(f"Unknown mode {mode}")

    embed_index = Path(embed_cfg.get("offline_index_path", ""))
    embed_meta = Path(embed_cfg.get("meta_path", ""))
    bm25_corpus = Path(bm25_cfg.get("store_path", "")) / "notes.jsonl"

    if embed_cfg.get("enabled") and not (embed_index.exists() and embed_meta.exists()):
        embed_cfg["enabled"] = False
    if bm25_cfg.get("enabled") and not bm25_corpus.exists():
        bm25_cfg["enabled"] = False

    return cfg


def _mode_requirements(mode: str) -> Tuple[bool, bool]:
    if mode == "structured":
        return False, False
    if mode == "dense":
        return True, False
    if mode == "bm25":
        return False, True
    if mode == "hybrid":
        return True, True
    raise ValueError(f"Unsupported mode {mode}")


def _ensure_document_index(
    item: Dict[str, Any],
    *,
    cache_root: Path,
    context_mode: str,
    summaries_map: Dict[str, str],
    summaries_all: Dict[str, str],
    stories_dir: Optional[Path],
    llm_endpoint: str,
    llm_model: str,
    base_cfg: Dict[str, Any],
    force_build: bool,
    build_embedding: bool,
    build_bm25: bool,
    doc_cache: DocumentCache,
) -> Tuple[Path, Dict[str, Any], Dict[str, str]]:
    doc_id = item["document_id"]
    cache_key = f"{context_mode}:{doc_id}"
    lock = doc_cache.lock_for(cache_key)
    with lock:
        doc_root = cache_root / context_mode / doc_id
        docs_dir = doc_root / "docs"
        _ensure_doc_text(
            doc_id,
            context_mode=context_mode,
            docs_dir=docs_dir,
            summaries_map=summaries_map,
            summaries_all=summaries_all,
            stories_dir=stories_dir,
            force_build=force_build,
        )
        build_stats = _ensure_index(
            docs_dir,
            doc_root,
            llm_endpoint,
            llm_model,
            force_build=force_build,
        )
        cfg = _prepare_aux_config(base_cfg, doc_root)
        aux_stats = _ensure_aux_indexes(
            cfg,
            build_embedding=build_embedding,
            build_bm25=build_bm25,
            force_build=force_build,
        )
    return doc_root, build_stats, aux_stats


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
    llm_endpoint: str,
    llm_model: str,
    reader: str,
    openai_cfg: Optional[Dict[str, Any]],
    force_build: bool,
    doc_cache: DocumentCache,
) -> Dict[str, Any]:
    doc_id = item["document_id"]
    build_embedding, build_bm25 = _mode_requirements(mode)
    doc_root, build_stats, aux_stats = _ensure_document_index(
        item,
        cache_root=cache_root,
        context_mode=context_mode,
        summaries_map=summaries_map,
        summaries_all=summaries_all,
        stories_dir=stories_dir,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
        base_cfg=base_cfg,
        force_build=force_build,
        build_embedding=build_embedding,
        build_bm25=build_bm25,
        doc_cache=doc_cache,
    )

    notes_path = doc_root / "notes.jsonl"
    index_dir = doc_root / "indexes"
    retriever_cfg = _prepare_retriever_config(base_cfg, doc_root, mode)
    retrieve_result = retrieve(
        question=item["question"],
        index_dir=str(index_dir),
        notes_path=str(notes_path),
        top_k=top_k,
        cfg=retriever_cfg,
    )
    evidences = retrieve_result.get("evidence") or []
    raw_answer = generate_answer(
        question=item["question"],
        evidences=evidences,
        reader=reader,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
        openai_cfg=openai_cfg,
    )
    final_answer = extract_final_answer(raw_answer) or raw_answer
    metrics = score_metrics(final_answer, item["references"])
    answer_model = openai_cfg.get("model") if reader == "openai" and openai_cfg else llm_model

    return {
        "qid": item["qid"],
        "document_id": doc_id,
        "split": split,
        "mode": mode,
        "reader": reader,
        "model": answer_model,
        "question": item["question"],
        "prediction": final_answer,
        "references": item["references"],
        "metrics": metrics,
        "meta": {
            "retrieval_mode": mode,
            "reader": reader,
            "model": answer_model,
            "context_mode": context_mode,
            "top_k": top_k,
            "build_stats": build_stats,
            "aux_indexes": aux_stats,
        },
    }


def _accumulate_metrics(totals: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def _drain_futures(
    future_map: Dict[Any, str],
    handle,
    totals: Dict[str, float],
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
                _accumulate_metrics(totals, record.get("metrics") or {})
                succeeded += 1
            completed += 1
            last_progress = time.time()
            if progress is not None:
                progress.update(1)
    return completed, succeeded


def main() -> None:
    parser = argparse.ArgumentParser(description="NarrativeQA CSV entry for RelRAG")
    parser.add_argument("--config", help="Path to YAML config file (defaults to relrag/config/config.yaml)")
    parser.add_argument("--qaps", help="Path to NarrativeQA qaps.csv (fallback to config)")
    parser.add_argument("--summaries", help="Path to NarrativeQA summaries.csv (fallback to config)")
    parser.add_argument("--stories_dir", help="Directory containing full stories (story-as-context)")
    parser.add_argument("--split", help="Dataset split: train, valid, or test (fallback to config)")
    parser.add_argument("--context_mode", help="summary-as-context or story-as-context (fallback to config)")
    parser.add_argument("--retriever", help="Retriever mode: bm25, dense, or hybrid (fallback to config)")
    parser.add_argument("--modes", help="Retrieval modes: structured,dense,bm25,hybrid (fallback to config)")
    parser.add_argument("--endpoint", help="vLLM endpoint (defaults to config)")
    parser.add_argument("--model", help="Ignored; NarrativeQA uses qwen3-30b-a3b")
    parser.add_argument("--reader", help="Reader backend: vllm or openai (fallback to config)")
    parser.add_argument("--openai_model", help="OpenAI model name (fallback to config)")
    parser.add_argument("--openai_api_key", help="OpenAI API key (reads env if omitted)")
    parser.add_argument("--openai_temperature", type=float, help="OpenAI temperature (fallback to config)")
    parser.add_argument("--openai_max_tokens", type=int, help="OpenAI max tokens (fallback to config)")
    parser.add_argument("--top_k", type=int, help="Top-k retrieval fanout (fallback to config)")
    parser.add_argument("--limit", type=int, help="Process only first N examples (fallback to config)")
    parser.add_argument("--workers", type=int, help="Parallel workers (fallback to config)")
    parser.add_argument("--cache_dir", help="Cache root for per-document indexes (fallback to config)")
    parser.add_argument("--output_dir", help="Output directory (fallback to config)")
    parser.add_argument("--force_build", action="store_true", help="Rebuild indexes even if cached")
    parser.add_argument("--stall_warn_sec", type=float, help="Warn if no worker finishes within this many seconds (fallback to config)")
    parser.add_argument("--stall_abort_sec", type=float, help="Abort pending workers after this many idle seconds (0 to disable, fallback to config)")

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
    if args.modes is None and args.retriever is None:
        has_dataset_modes = dataset_cfg.get("retrievers") is not None or dataset_cfg.get("retriever_modes") is not None
        if not has_dataset_modes:
            args.modes = _pick_arg(args, entry_cfg, dataset_cfg, "modes", DEFAULT_MODES)
    args.cache_dir = _pick_arg(args, entry_cfg, dataset_cfg, "cache_dir", DEFAULT_CACHE_DIR)
    args.output_dir = _pick_arg(args, entry_cfg, dataset_cfg, "output_dir", DEFAULT_OUTPUT_DIR)
    args.top_k = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "top_k", DEFAULT_TOP_K),
        DEFAULT_TOP_K,
    )
    args.limit = _coerce_int(_pick_arg(args, entry_cfg, dataset_cfg, "limit", DEFAULT_LIMIT), DEFAULT_LIMIT)
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

    base_cfg = _apply_dataset_retriever(deepcopy(cfg), "narrativeqa")
    modes = _resolve_retriever_modes(
        mode_arg=args.retriever,
        modes_arg=args.modes,
        entry_cfg=entry_cfg,
        dataset_cfg=dataset_cfg,
        base_cfg=base_cfg,
    )
    readers = _resolve_readers(args, cfg, dataset_cfg)
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
            mode_top_k = _resolve_mode_top_k(mode, base_cfg, args.top_k)
            output_name = _pred_filename(split, reader, mode, len(readers), len(modes))
            output_path = output_dir / output_name
            logger.info("Running reader={} mode={} -> {}", reader, mode, output_path)
            totals = {"bleu1": 0.0, "bleu4": 0.0, "rougeL": 0.0, "meteor": 0.0}
            progress = ProgressBar(total_examples)
            processed = 0
            completed = 0
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
                                llm_endpoint=llm_endpoint,
                                llm_model=llm_model,
                                reader=reader,
                                openai_cfg=reader_openai_cfg,
                                force_build=args.force_build,
                                doc_cache=doc_cache,
                            )
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            handle.flush()
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
                                llm_endpoint=llm_endpoint,
                                llm_model=llm_model,
                                reader=reader,
                                openai_cfg=reader_openai_cfg,
                                force_build=args.force_build,
                                doc_cache=doc_cache,
                            )
                            future_map[future] = item.get("qid", "unknown")
                            scheduled += 1
                            if len(future_map) >= buffer_cap:
                                done_count, ok_count = _drain_futures(
                                    future_map,
                                    handle,
                                    totals,
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
                                progress=progress,
                                stall_warn_sec=args.stall_warn_sec,
                                stall_abort_sec=args.stall_abort_sec,
                            )
                            completed += done_count
                            processed += ok_count

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
            }

    if len(readers) == 1:
        summary_report["modes"] = summary_report["runs"][readers[0]]
    if len(modes) == 1:
        summary_report["models"] = {reader: summary_report["runs"][reader][modes[0]] for reader in readers}

    summary_path = output_dir / f"summary_{split}.json"
    summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary written to {}", summary_path)


if __name__ == "__main__":
    main()
