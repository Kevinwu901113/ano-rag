import argparse
import csv
import json
import math
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
from relrag.config.config_loader import config as global_config
from relrag.indexer.bm25_index import BM25IndexBuilder
from relrag.indexer.embedding_index import EmbeddingIndexBuilder
from relrag.utils.output_eval import extract_final_answer


DEFAULT_TOP_K = 10
DEFAULT_LIMIT = 0
DEFAULT_WORKERS = 1
DEFAULT_CACHE_DIR = "result/narrativeqa/cache"
DEFAULT_OUTPUT_DIR = "result/narrativeqa"
DEFAULT_CONTEXT_MODE = "summary"
DEFAULT_MODES = ("structured",)
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


def _load_entry_config() -> Dict[str, Any]:
    cfg = global_config.load_config()
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


def _pick_arg(args: argparse.Namespace, entry_cfg: Dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
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


def _resolve_llm_config(args: argparse.Namespace) -> Tuple[str, str]:
    cfg = global_config.load_config()
    endpoint = args.endpoint or (cfg.get("vllm") or {}).get("endpoint")
    if not endpoint:
        raise ValueError("LLM endpoint is required (use --endpoint or config vllm.endpoint)")
    if args.model and args.model != FIXED_LLM_MODEL:
        logger.warning("Ignoring --model {}; NarrativeQA uses {}", args.model, FIXED_LLM_MODEL)
    model_cfg = (cfg.get("vllm") or {}).get("model")
    if model_cfg and model_cfg != FIXED_LLM_MODEL:
        logger.warning("Overriding config model {} -> {}", model_cfg, FIXED_LLM_MODEL)
    return endpoint, FIXED_LLM_MODEL


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
    dataset_cfg = cfg.get(dataset_key) or {}
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


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if n <= 0:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _bleu_score(references: List[str], hypothesis: str, max_n: int) -> float:
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    ref_tokens_list = [_tokenize(ref) for ref in references if ref]
    if not ref_tokens_list:
        return 0.0

    hyp_len = len(hyp_tokens)
    ref_lens = [len(ref) for ref in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))
    if hyp_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (closest_ref_len / max(1, hyp_len)))

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        hyp_counts = _ngram_counts(hyp_tokens, n)
        max_ref_counts: Dict[Tuple[str, ...], int] = {}
        for ref_tokens in ref_tokens_list:
            ref_counts = _ngram_counts(ref_tokens, n)
            for gram, count in ref_counts.items():
                max_ref_counts[gram] = max(max_ref_counts.get(gram, 0), count)
        match = sum(min(count, max_ref_counts.get(gram, 0)) for gram, count in hyp_counts.items())
        total = sum(hyp_counts.values())
        if total == 0:
            precision = 0.0
        else:
            precision = match / total
        if precision == 0.0:
            precision = (match + 1.0) / (total + 1.0)
        precisions.append(precision)

    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return float(score)


def _lcs_alignment(ref_tokens: List[str], hyp_tokens: List[str]) -> List[Tuple[int, int]]:
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0 or m == 0:
        return []
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if ref_tokens[i] == hyp_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    i = n
    j = m
    alignment: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        if ref_tokens[i - 1] == hyp_tokens[j - 1]:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    alignment.reverse()
    return alignment


def _rouge_l_score(references: List[str], hypothesis: str) -> float:
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    best = 0.0
    beta = 1.2
    for ref in references:
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            continue
        alignment = _lcs_alignment(ref_tokens, hyp_tokens)
        lcs_len = len(alignment)
        if lcs_len == 0:
            continue
        prec = lcs_len / len(hyp_tokens)
        rec = lcs_len / len(ref_tokens)
        denom = rec + (beta * beta * prec)
        if denom == 0:
            f_score = 0.0
        else:
            f_score = (1 + beta * beta) * prec * rec / denom
        best = max(best, f_score)
    return float(best)


def _meteor_score(references: List[str], hypothesis: str) -> float:
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            continue
        alignment = _lcs_alignment(ref_tokens, hyp_tokens)
        matches = len(alignment)
        if matches == 0:
            continue
        prec = matches / len(hyp_tokens)
        rec = matches / len(ref_tokens)
        denom = rec + 9 * prec
        if denom == 0:
            f_mean = 0.0
        else:
            f_mean = (10 * prec * rec) / denom
        chunks = 1
        for idx in range(1, len(alignment)):
            prev = alignment[idx - 1]
            curr = alignment[idx]
            if curr[0] != prev[0] + 1 or curr[1] != prev[1] + 1:
                chunks += 1
        penalty = 0.5 * (chunks / matches) ** 3
        score = (1 - penalty) * f_mean
        best = max(best, score)
    return float(best)


def _score_metrics(prediction: str, references: List[str]) -> Dict[str, float]:
    if not references:
        return {"bleu1": 0.0, "bleu4": 0.0, "rougeL": 0.0, "meteor": 0.0}
    bleu1 = _bleu_score(references, prediction, 1)
    bleu4 = _bleu_score(references, prediction, 4)
    rouge_l = _rouge_l_score(references, prediction)
    meteor = _meteor_score(references, prediction)
    return {
        "bleu1": round(bleu1, 4),
        "bleu4": round(bleu4, 4),
        "rougeL": round(rouge_l, 4),
        "meteor": round(meteor, 4),
    }


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
    context_mode: str,
    summaries_map: Dict[str, str],
    summaries_all: Dict[str, str],
    stories_dir: Optional[Path],
    base_cfg: Dict[str, Any],
    mode: str,
    top_k: int,
    llm_endpoint: str,
    llm_model: str,
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
    raw_answer = answer(
        question=item["question"],
        evidences=evidences,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
    )
    final_answer = extract_final_answer(raw_answer) or raw_answer
    metrics = _score_metrics(final_answer, item["references"])

    return {
        "qid": item["qid"],
        "document_id": doc_id,
        "question": item["question"],
        "prediction": final_answer,
        "references": item["references"],
        "metrics": metrics,
        "meta": {
            "mode": mode,
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
    parser.add_argument("--qaps", help="Path to NarrativeQA qaps.csv (fallback to config)")
    parser.add_argument("--summaries", help="Path to NarrativeQA summaries.csv (fallback to config)")
    parser.add_argument("--stories_dir", help="Directory containing full stories (story-as-context)")
    parser.add_argument("--split", help="Dataset split: train, valid, or test (fallback to config)")
    parser.add_argument("--context_mode", help="summary-as-context or story-as-context (fallback to config)")
    parser.add_argument("--modes", help="Retrieval modes: structured,dense,bm25,hybrid (fallback to config)")
    parser.add_argument("--endpoint", help="vLLM endpoint (defaults to config)")
    parser.add_argument("--model", help="Ignored; NarrativeQA uses qwen3-30b-a3b")
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

    entry_cfg = _load_entry_config()
    args.qaps = _pick_arg(args, entry_cfg, "qaps", None)
    args.summaries = _pick_arg(args, entry_cfg, "summaries", None)
    args.stories_dir = _pick_arg(args, entry_cfg, "stories_dir", None)
    args.split = _normalize_split(_pick_arg(args, entry_cfg, "split", DEFAULT_SPLIT))
    args.context_mode = _normalize_context_mode(_pick_arg(args, entry_cfg, "context_mode", DEFAULT_CONTEXT_MODE))
    args.modes = _parse_modes(_pick_arg(args, entry_cfg, "modes", DEFAULT_MODES))
    args.cache_dir = _pick_arg(args, entry_cfg, "cache_dir", DEFAULT_CACHE_DIR)
    args.output_dir = _pick_arg(args, entry_cfg, "output_dir", DEFAULT_OUTPUT_DIR)
    args.top_k = _coerce_int(_pick_arg(args, entry_cfg, "top_k", DEFAULT_TOP_K), DEFAULT_TOP_K)
    args.limit = _coerce_int(_pick_arg(args, entry_cfg, "limit", DEFAULT_LIMIT), DEFAULT_LIMIT)
    args.workers = _coerce_int(_pick_arg(args, entry_cfg, "workers", DEFAULT_WORKERS), DEFAULT_WORKERS)
    args.stall_warn_sec = _coerce_float(
        _pick_arg(args, entry_cfg, "stall_warn_sec", DEFAULT_STALL_WARN_SEC),
        DEFAULT_STALL_WARN_SEC,
    )
    args.stall_abort_sec = _coerce_float(
        _pick_arg(args, entry_cfg, "stall_abort_sec", DEFAULT_STALL_ABORT_SEC),
        DEFAULT_STALL_ABORT_SEC,
    )
    if entry_cfg.get("force_build"):
        args.force_build = True

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

    llm_endpoint, llm_model = _resolve_llm_config(args)
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

    base_cfg = _apply_dataset_retriever(deepcopy(global_config.load_config()), "narrativeqa")
    doc_cache = DocumentCache()

    summary_report: Dict[str, Any] = {
        "split": split,
        "context_mode": args.context_mode,
        "model": llm_model,
        "modes": {},
    }

    for mode in args.modes:
        output_path = output_dir / f"pred_{split}_{mode}.jsonl"
        logger.info("Running mode={} -> {}", mode, output_path)
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
                            context_mode=args.context_mode,
                            summaries_map=summaries_map,
                            summaries_all=summaries_all,
                            stories_dir=stories_dir,
                            base_cfg=base_cfg,
                            mode=mode,
                            top_k=args.top_k,
                            llm_endpoint=llm_endpoint,
                            llm_model=llm_model,
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
                            context_mode=args.context_mode,
                            summaries_map=summaries_map,
                            summaries_all=summaries_all,
                            stories_dir=stories_dir,
                            base_cfg=base_cfg,
                            mode=mode,
                            top_k=args.top_k,
                            llm_endpoint=llm_endpoint,
                            llm_model=llm_model,
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
        logger.info("Mode {} completed {} examples (failed {})", mode, processed, failed)
        denom = processed if processed > 0 else 1
        summary_report["modes"][mode] = {
            "bleu1": round(totals["bleu1"] / denom, 4),
            "bleu4": round(totals["bleu4"] / denom, 4),
            "rougeL": round(totals["rougeL"] / denom, 4),
            "meteor": round(totals["meteor"] / denom, 4),
            "count": processed,
        }

    summary_path = output_dir / f"summary_{split}.json"
    summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary written to {}", summary_path)


if __name__ == "__main__":
    main()
