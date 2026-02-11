import argparse
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

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
from relrag.config.config_loader import ConfigLoader, config as global_config
from relrag.config.dataset_config import (
    get_dataset_config,
    resolve_openai_api_key,
    resolve_openai_config,
    resolve_reader,
)
from relrag.generator import answerer as answerer_module
from relrag.prompt import load_prompt
from relrag.utils.answer_source import resolve_short_answer, sha1_text
from relrag.utils.llm_stats import LLMCallStats, llm_stats_scope
from relrag.utils.embedding_utils import get_shared_encoder
from relrag.utils.openai_answer import generate_openai_answer
from relrag.utils.output_eval import has_final_tag
from relrag.utils.vllm_runtime import resolve_vllm_endpoint_model


DEFAULT_STALL_WARN_SEC = 300.0
DEFAULT_STALL_ABORT_SEC = 900.0
DEFAULT_TOP_K = 10
DEFAULT_LIMIT = 0
DEFAULT_WORKERS = 1
DEFAULT_SPLIT = "full"
DEFAULT_CACHE_DIR = "result/mirage/cache"
DEFAULT_OUTPUT_DIR = "result/mirage"
DEFAULT_OVERFETCH = 2.0
MIN_OVERFETCH = 2.0
DEFAULT_BACKFILL_MAX_OVERFETCH = 4.0
DEFAULT_BACKFILL_STEP = 1.5
DEFAULT_BACKFILL_ROUNDS = 3
DEFAULT_LLM_RETRY_ON_EMPTY = 1
DEFAULT_LLM_RETRY_EVIDENCE = 6
DEFAULT_DOC_CHUNK_MAX_CHARS = 0


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


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_answer(text: str) -> str:
    def _remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def _white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def _remove_punc(value: str) -> str:
        exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return "".join(ch for ch in value if ch not in exclude)

    def _lower(value: str) -> str:
        return value.lower()

    return _white_space_fix(_remove_articles(_remove_punc(_lower(text or ""))))


def _f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = _normalize_answer(prediction)
    normalized_ground_truth = _normalize_answer(ground_truth)
    zero = (0.0, 0.0, 0.0)
    special = {"yes", "no", "noanswer", "insufficient evidence"}
    if normalized_prediction in special and normalized_prediction != normalized_ground_truth:
        return zero
    if normalized_ground_truth in special and normalized_prediction != normalized_ground_truth:
        return zero

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    if not prediction_tokens or not ground_truth_tokens:
        return zero
    common: Dict[str, int] = {}
    for token in prediction_tokens:
        common[token] = common.get(token, 0) + 1
    num_same = 0
    for token in ground_truth_tokens:
        if token in common and common[token] > 0:
            num_same += 1
            common[token] -= 1
    if num_same == 0:
        return zero
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1), float(precision), float(recall)


def _exact_match_score(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _best_answer_metrics(prediction: str, golds: List[str]) -> Dict[str, float]:
    if not golds:
        return {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    best = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    for gold in golds:
        em = 1.0 if _exact_match_score(prediction, gold) else 0.0
        f1, prec, recall = _f1_score(prediction, gold)
        if f1 > best["f1"] or (f1 == best["f1"] and em > best["em"]):
            best = {"em": em, "f1": f1, "prec": prec, "recall": recall}
    return best


def _parse_slice(value: Optional[str]) -> Optional[Tuple[Optional[int], Optional[int]]]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if ":" not in raw:
        try:
            start = int(raw)
        except ValueError:
            return None
        return start, start + 1
    start_raw, end_raw = raw.split(":", 1)
    start = int(start_raw) if start_raw.strip() else None
    end = int(end_raw) if end_raw.strip() else None
    return start, end


def _slice_examples(
    examples: List[Dict[str, Any]],
    *,
    limit: int,
    slice_spec: Optional[Tuple[Optional[int], Optional[int]]],
) -> List[Dict[str, Any]]:
    if slice_spec:
        start, end = slice_spec
        examples = examples[slice(start, end)]
    if limit and limit > 0:
        examples = examples[: int(limit)]
    return examples


def _prepare_doc_pool_units(
    doc_pool: List[Dict[str, Any]],
    *,
    max_chars: Optional[int],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in doc_pool:
        mapped_id = str(item.get("mapped_id") or "")
        if not mapped_id:
            continue
        grouped.setdefault(mapped_id, []).append(item)

    units: List[Dict[str, Any]] = []
    for mapped_id, items in grouped.items():
        for idx, item in enumerate(items):
            doc_name = str(item.get("doc_name") or "").strip()
            text = str(item.get("doc_chunk") or "").strip()
            if max_chars and max_chars > 0 and len(text) > max_chars:
                text = text[: int(max_chars)].rstrip()
            if not text:
                continue
            support = int(item.get("support") or 0)
            chunk_id = f"{mapped_id}__c{idx:02d}"
            units.append(
                {
                    "note_id": chunk_id,
                    "chunk_id": chunk_id,
                    "mapped_id": mapped_id,
                    "doc_name": doc_name,
                    "text": text,
                    "support": support,
                    "chunk_idx": idx,
                }
            )
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


def _prepare_bm25(
    units: List[Dict[str, Any]],
    bm25_cfg: Dict[str, Any],
) -> Tuple[Any, List[Dict[str, Any]], List[int]]:
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
    k1 = float(bm25_cfg.get("k1", 0.9))
    b = float(bm25_cfg.get("b", 0.4))
    bm25 = BM25Okapi(docs, k1=k1, b=b)
    return bm25, valid_units, list(ngram)


def _bm25_search(
    question: str,
    bm25: Any,
    units: List[Dict[str, Any]],
    ngram: List[int],
    top_k: int,
) -> List[Dict[str, Any]]:
    query_tokens = _tokenize(question, ngram)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    results: List[Dict[str, Any]] = []
    for idx, score in ranked[: max(0, int(top_k))]:
        results.append({"unit": units[idx], "score": float(score)})
    return results


def _prepare_dense_embeddings(
    units: List[Dict[str, Any]],
    embed_cfg: Dict[str, Any],
    *,
    cache_root: Path,
    doc_pool_sha1: Optional[str],
) -> Tuple[Any, Any]:
    if np is None:
        raise RuntimeError("numpy is required for dense baseline retrieval.")
    provider = embed_cfg.get("provider", "qwen3")
    model = embed_cfg.get("model", "qwen3-embedding")
    cache_dir = embed_cfg.get("cache_dir")
    device = embed_cfg.get("device")
    dtype = embed_cfg.get("dtype")
    endpoint = embed_cfg.get("endpoint")
    api_key = embed_cfg.get("api_key")
    timeout_s = embed_cfg.get("timeout_s")
    normalize = bool(embed_cfg.get("normalize", True))
    max_len = int(embed_cfg.get("max_len_note", 256))
    batch_size = _coerce_int(embed_cfg.get("batch_size", 16), 16)

    encoder = get_shared_encoder(
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

    cache_dir_path = cache_root / "dense_cache"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir_path / "doc_pool_meta.json"
    data_path = cache_dir_path / "doc_pool_embeddings.npy"

    if meta_path.exists() and data_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if isinstance(meta, dict):
            if (
                meta.get("doc_pool_sha1") == doc_pool_sha1
                and meta.get("model") == model
                and meta.get("provider") == provider
                and bool(meta.get("normalize", True)) == normalize
                and int(meta.get("max_len", max_len)) == max_len
                and int(meta.get("count", -1)) == len(units)
            ):
                try:
                    vectors = np.load(str(data_path))
                    return encoder, vectors
                except Exception as exc:
                    logger.warning("Failed to load dense cache: {}", exc)

    texts = [unit.get("text") or "" for unit in units]
    vectors = encoder.encode(
        texts,
        max_length=max_len,
        batch_size=batch_size,
        normalize=normalize,
    )
    if vectors.size == 0:
        raise RuntimeError("Dense encoder returned empty vectors.")
    np.save(str(data_path), vectors)
    meta_payload = {
        "doc_pool_sha1": doc_pool_sha1,
        "model": model,
        "provider": provider,
        "normalize": normalize,
        "max_len": max_len,
        "count": len(units),
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return encoder, vectors


def _dense_search(
    question: str,
    units: List[Dict[str, Any]],
    encoder,
    doc_vectors,
    embed_cfg: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    if np is None:
        raise RuntimeError("numpy is required for dense baseline retrieval.")
    if encoder is None:
        raise RuntimeError("Dense encoder is not initialized.")
    if doc_vectors is None or len(units) == 0:
        return []
    normalize = bool(embed_cfg.get("normalize", True))
    max_len = int(embed_cfg.get("max_len_note", 256))
    batch_size = _coerce_int(embed_cfg.get("batch_size", 16), 16)
    q_vec = encoder.encode([question.strip()], max_length=max_len, batch_size=batch_size, normalize=normalize)
    if q_vec.size == 0:
        return []
    query = np.asarray(q_vec[0], dtype="float32")
    docs = np.asarray(doc_vectors, dtype="float32")
    if normalize:
        scores = docs @ query
    else:
        denom = (np.linalg.norm(docs, axis=1) * (np.linalg.norm(query) + 1e-12)) + 1e-12
        scores = (docs @ query) / denom
    ranked_idx = np.argsort(scores)[::-1][: max(0, int(top_k))]
    results: List[Dict[str, Any]] = []
    for idx in ranked_idx:
        results.append({"unit": units[int(idx)], "score": float(scores[int(idx)])})
    return results


def _build_retrieved_context(
    ranked: List[Dict[str, Any]],
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for rank, item in enumerate(ranked, start=1):
        unit = item.get("unit") or {}
        text = unit.get("text") or ""
        contexts.append(
            {
                "note_id": unit.get("note_id"),
                "mapped_id": unit.get("mapped_id"),
                "doc_name": unit.get("doc_name"),
                "chunk_id": unit.get("chunk_id"),
                "support": unit.get("support"),
                "text": text,
                "text_hash": sha1_text(text) if text else None,
                "evidence": text,
                "canonical": text,
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
    mapped_id = ctx.get("mapped_id")
    text = ctx.get("canonical") or ctx.get("evidence") or ""
    if mapped_id and text:
        return ("mapped_id_hash", str(mapped_id), sha1_text(str(text))), "mapped_id_hash"
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
    try:
        min_factor = float(min_overfetch) if min_overfetch is not None else MIN_OVERFETCH
    except (TypeError, ValueError):
        min_factor = MIN_OVERFETCH
    min_factor = max(1.0, min_factor)
    min_raw = int(math.ceil(top_k * min_factor))
    if top_k_raw is not None:
        raw = _coerce_int(top_k_raw, top_k)
        return max(min_raw, raw), "top_k_raw"
    factor = None
    if overfetch is not None:
        try:
            factor = float(overfetch)
        except (TypeError, ValueError):
            factor = DEFAULT_OVERFETCH
    if factor is None or factor <= 0:
        factor = DEFAULT_OVERFETCH
    factor = max(factor, min_factor)
    raw = int(math.ceil(top_k * factor))
    return max(min_raw, raw), "overfetch"


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
    bm25: Any,
    bm25_units: List[Dict[str, Any]],
    bm25_ngram: List[int],
    dense_encoder,
    dense_vectors,
    embed_cfg: Dict[str, Any],
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
    if units:
        max_raw = min(max_raw, len(units))

    attempt = 0
    last_raw_count = -1
    backfill_reason = None
    retrieved_context_raw: List[Dict[str, Any]] = []
    retrieved_context_topk: List[Dict[str, Any]] = []
    dedup_stats: Dict[str, Any] = {}

    while True:
        if mode == "bm25":
            ranked = _bm25_search(question, bm25, bm25_units, bm25_ngram, requested)
        elif mode == "dense":
            ranked = _dense_search(question, units, dense_encoder, dense_vectors, embed_cfg, requested)
        else:
            raise ValueError(f"Unsupported mode {mode}")

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

    return retrieved_context_raw, retrieved_context_topk, dedup_stats, backfill_reason, attempt


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


def _load_entry_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    entry_cfg = cfg.get("mirage_baseline_entry") or cfg.get("entry") or {}
    if not isinstance(entry_cfg, dict):
        return {}
    return entry_cfg


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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


def _apply_dataset_retriever(cfg: Dict[str, Any], dataset_key: str) -> Dict[str, Any]:
    dataset_cfg = get_dataset_config(cfg, dataset_key)
    for key in ("retriever", "answer", "answerer", "reranker", "llm", "llm_profiles", "openai", "vllm"):
        override = dataset_cfg.get(key) if isinstance(dataset_cfg, dict) else None
        if isinstance(override, dict):
            base_value = cfg.get(key) or {}
            cfg[key] = _deep_merge(base_value, override)
    return cfg


def _parse_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_retrievers(args: argparse.Namespace, dataset_cfg: Dict[str, Any]) -> List[str]:
    raw = args.retriever if args.retriever is not None else dataset_cfg.get("retriever")
    if raw is None:
        raw = dataset_cfg.get("retrievers") or dataset_cfg.get("retriever_modes")
    if raw is None:
        raw = "bm25"
    if isinstance(raw, (list, tuple)):
        modes = [str(item).strip().lower() for item in raw if str(item).strip()]
    else:
        modes = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    allowed = {"bm25", "dense"}
    normalized = [mode for mode in modes if mode in allowed]
    if not normalized:
        normalized = ["bm25"]
    seen = set()
    ordered: List[str] = []
    for mode in normalized:
        if mode in seen:
            continue
        ordered.append(mode)
        seen.add(mode)
    return ordered


def _resolve_readers(args: argparse.Namespace, cfg: Dict[str, Any], dataset_cfg: Dict[str, Any]) -> List[str]:
    raw = args.reader if args.reader is not None else dataset_cfg.get("reader")
    if raw is None:
        raw = dataset_cfg.get("readers") or dataset_cfg.get("models")
    if raw is None:
        raw = resolve_reader(None, dataset_cfg)
    if isinstance(raw, (list, tuple)):
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
    if "vllm" in ordered and "openai" in ordered:
        ordered = ["vllm", "openai"]
    vllm_enabled = bool((cfg.get("vllm") or {}).get("enabled", True))
    openai_enabled = bool((cfg.get("openai") or {}).get("enabled", True))
    for reader in ordered:
        if reader == "vllm" and not vllm_enabled:
            raise ValueError("vLLM mode disabled in config (vllm.enabled=false).")
        if reader == "openai" and not openai_enabled:
            raise ValueError("OpenAI mode disabled in config (openai.enabled=false).")
    return ordered


def _drain_futures(
    future_map: Dict[Any, str],
    handle: TextIO,
    totals: Dict[str, float],
    *,
    raw_handle: Optional[TextIO],
    topk_handle: Optional[TextIO],
    rollup: Optional[Dict[str, Any]],
    stat_rollup: Optional[Dict[str, List[float]]],
    progress: Optional[ProgressBar],
    stall_warn_sec: float,
    stall_abort_sec: float,
) -> Tuple[int, int]:
    completed = 0
    succeeded = 0
    start_time = time.time()
    last_progress = start_time
    pending = set(future_map.keys())
    while pending:
        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
        now = time.time()
        stalled = now - last_progress
        if pending and stalled > stall_warn_sec:
            logger.warning("Stalled for {:.1f}s; {} tasks pending", stalled, len(pending))
        if pending and stalled > stall_abort_sec:
            raise TimeoutError(f"No progress for {stalled:.1f}s with {len(pending)} pending tasks")
        if not done:
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
                _write_retrieval_payloads(record, raw_handle=raw_handle, topk_handle=topk_handle)
                _accumulate_metrics(totals, record.get("metrics") or {})
                _record_fallback_sample(rollup, record)
                _record_stat_rollup(stat_rollup, record)
                succeeded += 1
            completed += 1
            last_progress = time.time()
            if progress is not None:
                progress.update(1)
    return completed, succeeded


def _accumulate_metrics(totals: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def _write_retrieval_payloads(
    record: Dict[str, Any],
    *,
    raw_handle: Optional[TextIO],
    topk_handle: Optional[TextIO],
) -> None:
    if raw_handle is not None:
        raw_payload = {
            "id": record.get("id"),
            "question": record.get("question"),
            "retrieved_context_raw": record.get("retrieved_context_raw") or [],
            "top_k_raw": record.get("top_k_raw"),
            "top_k_raw_requested": (record.get("intermediate") or {}).get("top_k_raw_requested"),
            "top_k_raw_source": (record.get("intermediate") or {}).get("top_k_raw_source"),
            "overfetch_factor": record.get("overfetch_factor"),
            "top_k_backfill_rounds": (record.get("intermediate") or {}).get("top_k_backfill_rounds"),
            "top_k_fill_reason": record.get("top_k_fill_reason"),
        }
        raw_handle.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")
        raw_handle.flush()
    if topk_handle is not None:
        topk_payload = {
            "id": record.get("id"),
            "question": record.get("question"),
            "retrieved_context_topk": record.get("retrieved_context_topk") or [],
            "top_k": record.get("top_k"),
            "top_k_final": record.get("top_k_final"),
            "duplicate_rate": record.get("duplicate_rate"),
            "oracle_hit@k": (record.get("metrics") or {}).get("oracle_hit@k"),
        }
        topk_handle.write(json.dumps(topk_payload, ensure_ascii=False) + "\n")
        topk_handle.flush()


def _record_fallback_sample(rollup: Optional[Dict[str, Any]], record: Dict[str, Any]) -> None:
    if rollup is None:
        return
    if not record.get("fallback"):
        return
    samples = rollup.setdefault("fallback_samples", [])
    metrics = record.get("metrics") or {}
    samples.append(
        {
            "id": record.get("id"),
            "answer_source": record.get("answer_source"),
            "fallback_reason": record.get("fallback_reason"),
            "llm_error": record.get("llm_error"),
            "em": metrics.get("em"),
            "f1": metrics.get("f1"),
        }
    )


def _record_stat_rollup(stat_rollup: Optional[Dict[str, List[float]]], record: Dict[str, Any]) -> None:
    if stat_rollup is None:
        return
    meta = record.get("meta") or {}
    _append_stat(stat_rollup, "llm_calls", meta.get("llm_calls"))
    _append_stat(stat_rollup, "llm_retries", meta.get("llm_retries"))
    _append_stat(stat_rollup, "prompt_tokens", meta.get("prompt_tokens"))
    _append_stat(stat_rollup, "completion_tokens", meta.get("completion_tokens"))
    _append_stat(stat_rollup, "prompt_chars", meta.get("prompt_chars"))
    _append_stat(stat_rollup, "completion_chars", meta.get("completion_chars"))
    _append_stat(stat_rollup, "context_chars", meta.get("context_chars"))
    _append_stat(stat_rollup, "t_retrieve_ms", meta.get("t_retrieve_ms"))
    _append_stat(stat_rollup, "t_build_prompt_ms", meta.get("t_build_prompt_ms"))
    _append_stat(stat_rollup, "t_llm_ms", meta.get("t_llm_ms"))
    _append_stat(stat_rollup, "t_total_ms", meta.get("t_total_ms"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    ]
    return {key: os.environ.get(key) for key in keys}


def _hash_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        data = path.read_text(encoding="utf-8")
    except Exception:
        return None
    return sha1_text(data)


def _doc_pool_cache_signature(
    doc_pool_sha1: Optional[str],
    *,
    doc_chunk_max_chars: Optional[int],
    base_cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    retr_cfg = base_cfg.get("retriever") or {}
    embed_cfg = retr_cfg.get("embedding") or {}
    bm25_cfg = retr_cfg.get("bm25") or {}
    signature = {
        "doc_pool_sha1": doc_pool_sha1,
        "doc_chunk_max_chars": int(doc_chunk_max_chars or 0),
        "notes_version": 1,
        "embedding": {
            "provider": embed_cfg.get("provider"),
            "model": embed_cfg.get("model"),
            "endpoint": embed_cfg.get("endpoint"),
            "max_len_note": embed_cfg.get("max_len_note"),
            "normalize": embed_cfg.get("normalize"),
        },
        "bm25": {
            "backend": bm25_cfg.get("backend"),
            "k1": bm25_cfg.get("k1"),
            "b": bm25_cfg.get("b"),
            "ngram": bm25_cfg.get("ngram"),
            "field_weights": bm25_cfg.get("field_weights"),
        },
    }
    signature_text = json.dumps(signature, sort_keys=True, ensure_ascii=True)
    return sha1_text(signature_text), signature


def _load_prepared_meta(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _build_llm_meta(stats: LLMCallStats, *, t_retrieve_ms: float, t_total_ms: float) -> Dict[str, Any]:
    prompt_tokens = stats.prompt_tokens_max or None
    completion_tokens = stats.completion_tokens_max or None
    prompt_chars = stats.prompt_chars_max or None
    completion_chars = stats.completion_chars_max or None
    context_chars = stats.context_chars_max or None
    meta: Dict[str, Any] = {
        "llm_calls": int(stats.llm_calls),
        "llm_retries": int(stats.llm_retries),
        "t_retrieve_ms": round(float(t_retrieve_ms), 3),
        "t_build_prompt_ms": round(float(stats.t_build_prompt_ms), 3),
        "t_llm_ms": round(float(stats.t_llm_ms), 3),
        "t_total_ms": round(float(t_total_ms), 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_chars": prompt_chars,
        "completion_chars": completion_chars,
        "context_chars": context_chars,
        "finish_reason": stats.finish_reason,
        "error_type": stats.error_type,
    }
    return meta


def _append_stat(rollup: Dict[str, List[float]], key: str, value: Optional[float]) -> None:
    if value is None:
        return
    rollup.setdefault(key, []).append(float(value))


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def _summarize_stats(values: List[float]) -> Dict[str, Optional[float]]:
    return {
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "max": max(values) if values else None,
    }


def _process_example(
    example: Dict[str, Any],
    *,
    example_idx: int,
    units: List[Dict[str, Any]],
    bm25: Any,
    bm25_units: List[Dict[str, Any]],
    bm25_ngram: List[int],
    dense_encoder,
    dense_vectors,
    base_cfg: Dict[str, Any],
    llm_endpoint: str,
    llm_model: str,
    mode: str,
    reader: str,
    openai_cfg: Optional[Dict[str, Any]],
    top_k: int,
    top_k_raw: int,
    top_k_raw_source: str,
    backfill_max_overfetch: float,
    backfill_step: float,
    backfill_rounds: int,
    llm_retry_on_empty: int,
    llm_retry_max_evidence: int,
    oracle_map: Dict[str, Dict[str, Any]],
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    qid = str(example.get("query_id") or example.get("id") or "unknown")
    question = str(example.get("query") or example.get("question") or "").strip()
    embed_cfg = (base_cfg.get("retriever") or {}).get("embedding") or {}
    total_start = time.time()
    stats = LLMCallStats()
    with llm_stats_scope(stats):
        retrieve_start = time.time()
        (
            retrieved_context_raw,
            retrieved_context_topk,
            dedup_stats,
            top_k_fill_reason,
            backfill_attempts,
        ) = _retrieve_with_backfill(
            question=question,
            units=units,
            mode=mode,
            bm25=bm25,
            bm25_units=bm25_units,
            bm25_ngram=bm25_ngram,
            dense_encoder=dense_encoder,
            dense_vectors=dense_vectors,
            embed_cfg=embed_cfg,
            top_k=top_k,
            top_k_raw=top_k_raw,
            backfill_max_overfetch=backfill_max_overfetch,
            backfill_step=backfill_step,
            backfill_rounds=backfill_rounds,
        )
        t_retrieve_ms = (time.time() - retrieve_start) * 1000.0

        evidences = retrieved_context_topk
        raw_answer, prompt_meta, llm_error, llm_error_reason = generate_answer(
            question=question,
            evidences=evidences,
            reader=reader,
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            openai_cfg=openai_cfg,
            base_cfg=base_cfg,
            run_dir=run_dir,
        )
    t_total_ms = (time.time() - total_start) * 1000.0
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
                question=question,
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

    gold_answers = example.get("answer") or []
    if not isinstance(gold_answers, list):
        gold_answers = [str(gold_answers)]
    gold_answers = [str(item).strip() for item in gold_answers if str(item).strip()]
    metrics_primary = _best_answer_metrics(short_answer, gold_answers)

    oracle_hit = 0.0
    oracle_entry = oracle_map.get(qid) or {}
    oracle_mapped = oracle_entry.get("mapped_id")
    if oracle_mapped:
        oracle_hit = 1.0 if any(ctx.get("mapped_id") == oracle_mapped for ctx in retrieved_context_topk) else 0.0

    llm_input_hash = prompt_meta.get("llm_input_hash") or ""
    top_k_raw_value = dedup_stats.get("top_k_raw")
    overfetch_factor = None
    if isinstance(top_k_raw_value, (int, float)) and top_k:
        overfetch_factor = float(top_k_raw_value) / float(top_k)
    top_k_raw_source_final = top_k_raw_source
    if backfill_attempts > 0:
        top_k_raw_source_final = "backfill"
    duplicate_rate = float(dedup_stats.get("duplicate_rate") or 0.0)
    fallback_flag = 1.0 if (fallback_reason or answer_source != "llm_final") else 0.0
    llm_meta = _build_llm_meta(stats, t_retrieve_ms=t_retrieve_ms, t_total_ms=t_total_ms)

    metrics = {
        "em": metrics_primary.get("em", 0.0),
        "f1": metrics_primary.get("f1", 0.0),
        "prec": metrics_primary.get("prec", 0.0),
        "recall": metrics_primary.get("recall", 0.0),
        "oracle_hit@k": oracle_hit,
        "duplicate_rate": duplicate_rate,
        "top_k_raw": float(dedup_stats.get("top_k_raw") or 0.0),
        "top_k_final": float(dedup_stats.get("top_k_final") or 0.0),
        "overfetch_factor": float(overfetch_factor) if overfetch_factor is not None else 0.0,
        "fallback": fallback_flag,
    }

    output_record = {
        "_id": qid,
        "_example_idx": int(example_idx),
        "id": qid,
        "question": question,
        "gold_answers": gold_answers,
        "pred_answer": short_answer,
        "answer": short_answer,
        "answer_source": answer_source,
        "answer_source_detail": answer_source_detail,
        "metrics": metrics,
        "mode": mode,
        "reader": reader,
        "model": answer_model,
        "retrieved_context_raw": retrieved_context_raw,
        "retrieved_context_topk": retrieved_context_topk,
        "retrieved_context": retrieved_context_topk,
        "top_k": top_k,
        "top_k_raw": dedup_stats.get("top_k_raw"),
        "top_k_final": dedup_stats.get("top_k_final"),
        "duplicate_rate": duplicate_rate,
        "overfetch_factor": overfetch_factor,
        "fallback": fallback_flag,
        "fallback_reason": fallback_reason,
        "llm_error": llm_error,
        "top_k_fill_reason": top_k_fill_reason,
        "prompt_hash": prompt_meta.get("prompt_template_hash"),
        "prompt_meta": prompt_meta,
        "meta": llm_meta,
        "llm_input_hash": llm_input_hash,
        "intermediate": {
            "llm_raw": raw_answer,
            "llm_has_final": has_final_tag(raw_answer),
            "retrieval_mode": mode,
            "reader": reader,
            "model": answer_model,
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
            "llm_retry_used": llm_retry_used,
            "llm_retry_source": llm_retry_source,
            "llm_retry_reason": llm_retry_reason,
        },
    }
    return output_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIRAGE pure baseline pipeline")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument(
        "--dataset",
        default="mirage/dataset.json",
        help="Path to dataset.json",
    )
    parser.add_argument(
        "--doc_pool",
        default="mirage/doc_pool.json",
        help="Path to doc_pool.json",
    )
    parser.add_argument(
        "--oracle",
        default="mirage/oracle.json",
        help="Path to oracle.json",
    )
    parser.add_argument("--doc_chunk_max_chars", type=int, help="Max chars per doc_pool chunk (0 = no truncation)")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Split label for output")
    parser.add_argument("--retriever", help="Retriever mode: bm25/dense")
    parser.add_argument("--reader", help="Reader: vllm/openai")
    parser.add_argument("--cache_dir", help="Cache directory for embeddings")
    parser.add_argument("--output_dir", help="Output directory for predictions")
    parser.add_argument("--run_dir", help="Run directory for audit artifacts")
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--top_k_raw", type=int)
    parser.add_argument("--overfetch", type=float)
    parser.add_argument("--min_overfetch", type=float)
    parser.add_argument("--backfill_max_overfetch", type=float, default=DEFAULT_BACKFILL_MAX_OVERFETCH)
    parser.add_argument("--backfill_step", type=float, default=DEFAULT_BACKFILL_STEP)
    parser.add_argument("--backfill_rounds", type=int, default=DEFAULT_BACKFILL_ROUNDS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--slice", dest="slice_spec", help="Slice of dataset, e.g. 0:100")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--stall_warn_sec", type=float, default=DEFAULT_STALL_WARN_SEC)
    parser.add_argument("--stall_abort_sec", type=float, default=DEFAULT_STALL_ABORT_SEC)
    parser.add_argument("--endpoint", help="Override vLLM endpoint")
    parser.add_argument("--model", help="Override vLLM model id")
    parser.add_argument("--openai_api_key", help="Override OpenAI API key")
    parser.add_argument("--llm_retry_on_empty", type=int)
    parser.add_argument("--llm_retry_max_evidence", type=int)
    parser.add_argument("--force_build", action="store_true", help="Force rebuild cached artifacts")

    args = parser.parse_args()

    cfg = ConfigLoader(args.config).load_config() if args.config else global_config.load_config()
    entry_cfg = _load_entry_config(cfg)
    dataset_cfg = get_dataset_config(cfg, "mirage")

    args.cache_dir = _pick_arg(args, entry_cfg, dataset_cfg, "cache_dir", DEFAULT_CACHE_DIR)
    args.output_dir = _pick_arg(args, entry_cfg, dataset_cfg, "output_dir", DEFAULT_OUTPUT_DIR)
    args.top_k = _pick_arg(args, entry_cfg, dataset_cfg, "top_k", DEFAULT_TOP_K)
    args.doc_chunk_max_chars = _pick_arg(
        args,
        entry_cfg,
        dataset_cfg,
        "doc_chunk_max_chars",
        DEFAULT_DOC_CHUNK_MAX_CHARS,
    )
    args.llm_retry_on_empty = _pick_arg(
        args,
        entry_cfg,
        dataset_cfg,
        "llm_retry_on_empty",
        DEFAULT_LLM_RETRY_ON_EMPTY,
    )
    args.llm_retry_max_evidence = _pick_arg(
        args,
        entry_cfg,
        dataset_cfg,
        "llm_retry_max_evidence",
        DEFAULT_LLM_RETRY_EVIDENCE,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    base_cfg = _apply_dataset_retriever(deepcopy(cfg), "mirage")
    readers = _resolve_readers(args, cfg, dataset_cfg)
    modes = _resolve_retrievers(args, dataset_cfg)

    openai_cfg = resolve_openai_config(cfg, dataset_cfg)
    openai_cfg["api_key"] = args.openai_api_key or openai_cfg.get("api_key")

    llm_endpoint, llm_model = resolve_vllm_endpoint_model(
        endpoint_override=args.endpoint,
        model_override=args.model,
        vllm_cfg=cfg.get("vllm"),
    )

    if args.run_dir:
        if len(readers) != 1 or len(modes) != 1:
            raise ValueError("run_dir requires exactly one reader and one retriever.")
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

    dataset_path = Path(args.dataset)
    doc_pool_path = Path(args.doc_pool)
    oracle_path = Path(args.oracle)
    dataset = _load_json(dataset_path)
    doc_pool = _load_json(doc_pool_path)
    oracle_map = _load_json(oracle_path)
    if not isinstance(dataset, list):
        raise ValueError("dataset.json must be a list")
    if not isinstance(doc_pool, list):
        raise ValueError("doc_pool.json must be a list")
    if not isinstance(oracle_map, dict):
        raise ValueError("oracle.json must be a dict")

    slice_spec = _parse_slice(args.slice_spec)
    selected = _slice_examples(dataset, limit=args.limit, slice_spec=slice_spec)
    total_examples = len(selected)

    doc_pool_sha1 = _hash_file(doc_pool_path)
    cache_key, cache_signature = _doc_pool_cache_signature(
        doc_pool_sha1,
        doc_chunk_max_chars=args.doc_chunk_max_chars,
        base_cfg=base_cfg,
    )
    cache_root = Path(args.cache_dir) / cache_key
    cache_root.mkdir(parents=True, exist_ok=True)
    prepared_meta_path = cache_root / "prepared_meta.json"
    prepared_path = cache_root / "prepared.json"
    prepared_meta = _load_prepared_meta(prepared_meta_path)
    prepared_ok = bool(prepared_path.exists())
    prepared_match = bool(prepared_meta and prepared_meta.get("cache_key") == cache_key)

    if args.force_build or not (prepared_ok and prepared_match):
        units = _prepare_doc_pool_units(doc_pool, max_chars=args.doc_chunk_max_chars)
        prepared_meta = {
            "cache_key": cache_key,
            "doc_pool_sha1": doc_pool_sha1,
            "doc_pool_path": str(doc_pool_path),
            "doc_chunk_max_chars": int(args.doc_chunk_max_chars or 0),
            "notes_count": len(units),
            "signature": cache_signature,
        }
        _write_json(prepared_meta_path, prepared_meta)
        _write_json(
            prepared_path,
            {
                "status": "ok",
                "timestamp": int(time.time()),
                "cache_key": cache_key,
            },
        )
    else:
        units = _prepare_doc_pool_units(doc_pool, max_chars=args.doc_chunk_max_chars)

    bm25 = None
    bm25_units: List[Dict[str, Any]] = []
    bm25_ngram: List[int] = []
    dense_encoder = None
    dense_vectors = None

    summary_report: Dict[str, Any] = {
        "split": args.split,
        "readers": readers,
        "retrievers": modes,
        "runs": {},
    }

    for reader in readers:
        reader_openai_cfg = openai_runtime_cfg if reader == "openai" else None
        answer_model = reader_openai_cfg.get("model") if reader == "openai" and reader_openai_cfg else llm_model
        summary_report["runs"].setdefault(reader, {})
        for mode in modes:
            retriever_cfg = base_cfg.get("retriever") or {}
            bm25_cfg = retriever_cfg.get("bm25") or {}
            embed_cfg = retriever_cfg.get("embedding") or {}

            if mode == "bm25":
                bm25, bm25_units, bm25_ngram = _prepare_bm25(units, bm25_cfg)
            if mode == "dense":
                dense_encoder, dense_vectors = _prepare_dense_embeddings(
                    units,
                    embed_cfg,
                    cache_root=cache_root,
                    doc_pool_sha1=cache_key,
                )

            top_k_raw, top_k_raw_source = _resolve_top_k_raw(
                args.top_k,
                args.top_k_raw,
                args.overfetch,
                args.min_overfetch,
            )

            output_base = Path(args.run_dir) if args.run_dir else Path(args.output_dir)
            output_base.mkdir(parents=True, exist_ok=True)
            output_name = "predictions.jsonl" if args.run_dir else f"pred_{args.split}_{reader}_{mode}.jsonl"
            output_path = output_base / output_name
            retrieval_raw_path = output_base / "retrieved_context_raw.jsonl" if args.run_dir else None
            retrieval_topk_path = output_base / "retrieved_context_topk.jsonl" if args.run_dir else None

            run_started_at = time.time()
            run_meta: Optional[Dict[str, Any]] = None
            if args.run_dir:
                resolved_cfg = deepcopy(base_cfg)
                resolved_cfg.setdefault("vllm", {})["endpoint"] = llm_endpoint
                resolved_cfg.setdefault("vllm", {})["model"] = llm_model
                if reader_openai_cfg:
                    resolved_cfg["openai"] = deepcopy(reader_openai_cfg)
                entry_snapshot = resolved_cfg.setdefault("mirage_baseline_entry", {})
                entry_snapshot.update(
                    {
                        "dataset": str(dataset_path),
                        "doc_pool": str(doc_pool_path),
                        "oracle": str(oracle_path),
                        "cache_dir": str(cache_root),
                        "cache_key": cache_key,
                        "output_dir": str(output_base),
                        "split": args.split,
                        "reader": reader,
                        "retriever": mode,
                        "top_k": args.top_k,
                        "top_k_raw": top_k_raw,
                        "overfetch": args.overfetch,
                        "min_overfetch": args.min_overfetch,
                        "backfill_max_overfetch": args.backfill_max_overfetch,
                        "backfill_step": args.backfill_step,
                        "backfill_rounds": args.backfill_rounds,
                        "llm_retry_on_empty": args.llm_retry_on_empty,
                        "llm_retry_max_evidence": args.llm_retry_max_evidence,
                        "limit": args.limit,
                        "slice": args.slice_spec,
                        "workers": args.workers,
                        "doc_chunk_max_chars": args.doc_chunk_max_chars,
                    }
                )
                _write_json(output_base / "config.resolved.json", _sanitize_config(resolved_cfg))
                run_meta = {
                    "run_dir": str(output_base),
                    "dataset": "mirage",
                    "dataset_path": str(dataset_path),
                    "doc_pool_path": str(doc_pool_path),
                    "oracle_path": str(oracle_path),
                    "dataset_sha1": _hash_file(dataset_path),
                    "doc_pool_sha1": doc_pool_sha1,
                    "oracle_sha1": _hash_file(oracle_path),
                    "sample_count": total_examples,
                    "split": args.split,
                    "reader": reader,
                    "retriever": mode,
                    "cache_dir": str(cache_root),
                    "cache_key": cache_key,
                    "llm_endpoint": llm_endpoint,
                    "llm_model": llm_model,
                    "openai": {
                        "base_url": (reader_openai_cfg or {}).get("base_url"),
                        "model": (reader_openai_cfg or {}).get("model"),
                    }
                    if reader == "openai"
                    else None,
                    "embedding_endpoint": ((base_cfg.get("retriever") or {}).get("embedding") or {}).get("endpoint"),
                    "top_k": args.top_k,
                    "top_k_raw": top_k_raw,
                    "top_k_raw_source": top_k_raw_source,
                    "overfetch": args.overfetch,
                    "min_overfetch": args.min_overfetch,
                    "backfill_max_overfetch": args.backfill_max_overfetch,
                    "backfill_step": args.backfill_step,
                    "backfill_rounds": args.backfill_rounds,
                    "llm_retry_on_empty": args.llm_retry_on_empty,
                    "llm_retry_max_evidence": args.llm_retry_max_evidence,
                    "workers": args.workers,
                    "slice": args.slice_spec,
                    "command": _sanitize_argv(sys.argv),
                    "started_at": int(run_started_at),
                    "host": socket.gethostname(),
                    "platform": platform.platform(),
                    "python": sys.version,
                    "git": _git_info(Path(__file__).resolve().parent),
                    "env": _env_snapshot(),
                    "prepared": prepared_meta,
                }
                _write_json(output_base / "run_meta.json", run_meta)

            totals = {
                "em": 0.0,
                "f1": 0.0,
                "prec": 0.0,
                "recall": 0.0,
                "oracle_hit@k": 0.0,
                "duplicate_rate": 0.0,
                "top_k_raw": 0.0,
                "top_k_final": 0.0,
                "overfetch_factor": 0.0,
                "fallback": 0.0,
            }
            rollup: Dict[str, Any] = {"fallback_samples": []}
            stat_rollup: Dict[str, List[float]] = {}
            progress = ProgressBar(total_examples)
            processed = 0
            completed = 0

            if args.run_dir:
                raw_handle = retrieval_raw_path.open("w", encoding="utf-8")
                topk_handle = retrieval_topk_path.open("w", encoding="utf-8")
            else:
                raw_handle = None
                topk_handle = None

            try:
                with output_path.open("w", encoding="utf-8") as handle:
                    if args.workers <= 1:
                        for example_idx, example in enumerate(selected):
                            try:
                                record = _process_example(
                                    example,
                                    example_idx=example_idx,
                                    units=units,
                                    bm25=bm25,
                                    bm25_units=bm25_units,
                                    bm25_ngram=bm25_ngram,
                                    dense_encoder=dense_encoder,
                                    dense_vectors=dense_vectors,
                                    base_cfg=base_cfg,
                                    llm_endpoint=llm_endpoint,
                                    llm_model=llm_model,
                                    mode=mode,
                                    reader=reader,
                                    openai_cfg=reader_openai_cfg,
                                    top_k=args.top_k,
                                    top_k_raw=top_k_raw,
                                    top_k_raw_source=top_k_raw_source,
                                    backfill_max_overfetch=args.backfill_max_overfetch,
                                    backfill_step=args.backfill_step,
                                    backfill_rounds=args.backfill_rounds,
                                    llm_retry_on_empty=args.llm_retry_on_empty,
                                    llm_retry_max_evidence=args.llm_retry_max_evidence,
                                    oracle_map=oracle_map,
                                    run_dir=str(output_base) if args.run_dir else None,
                                )
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                handle.flush()
                                _write_retrieval_payloads(record, raw_handle=raw_handle, topk_handle=topk_handle)
                                _accumulate_metrics(totals, record.get("metrics") or {})
                                _record_fallback_sample(rollup, record)
                                _record_stat_rollup(stat_rollup, record)
                                processed += 1
                            except Exception as exc:
                                qid = example.get("query_id") or example.get("id")
                                logger.error("Failed example {}: {}", qid, exc)
                            finally:
                                completed += 1
                                progress.update(1)
                    else:
                        max_workers = max(1, int(args.workers))
                        future_map: Dict[Any, str] = {}
                        scheduled = 0
                        buffer_cap = max_workers * 2
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            for example_idx, example in enumerate(selected):
                                qid = str(example.get("query_id") or example.get("id") or "unknown")
                                future = executor.submit(
                                    _process_example,
                                    example,
                                    example_idx=example_idx,
                                    units=units,
                                    bm25=bm25,
                                    bm25_units=bm25_units,
                                    bm25_ngram=bm25_ngram,
                                    dense_encoder=dense_encoder,
                                    dense_vectors=dense_vectors,
                                    base_cfg=base_cfg,
                                    llm_endpoint=llm_endpoint,
                                    llm_model=llm_model,
                                    mode=mode,
                                    reader=reader,
                                    openai_cfg=reader_openai_cfg,
                                    top_k=args.top_k,
                                    top_k_raw=top_k_raw,
                                    top_k_raw_source=top_k_raw_source,
                                    backfill_max_overfetch=args.backfill_max_overfetch,
                                    backfill_step=args.backfill_step,
                                    backfill_rounds=args.backfill_rounds,
                                    llm_retry_on_empty=args.llm_retry_on_empty,
                                    llm_retry_max_evidence=args.llm_retry_max_evidence,
                                    oracle_map=oracle_map,
                                    run_dir=str(output_base) if args.run_dir else None,
                                )
                                future_map[future] = qid
                                scheduled += 1
                                if len(future_map) >= buffer_cap:
                                    done_count, ok_count = _drain_futures(
                                        future_map,
                                        handle,
                                        totals,
                                        raw_handle=raw_handle,
                                        topk_handle=topk_handle,
                                        rollup=rollup,
                                        stat_rollup=stat_rollup,
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
                                    rollup=rollup,
                                    stat_rollup=stat_rollup,
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
            duration_sec = max(0.0, time.time() - run_started_at)
            denom = processed if processed > 0 else 1
            metrics_summary = {
                "em": round(totals["em"] / denom, 4),
                "f1": round(totals["f1"] / denom, 4),
                "prec": round(totals["prec"] / denom, 4),
                "recall": round(totals["recall"] / denom, 4),
                "oracle_hit@k": round(totals["oracle_hit@k"] / denom, 4),
                "duplicate_rate": round(totals["duplicate_rate"] / denom, 6),
                "top_k_raw": round(totals["top_k_raw"] / denom, 3),
                "top_k_final": round(totals["top_k_final"] / denom, 3),
                "overfetch_factor": round(totals["overfetch_factor"] / denom, 4),
                "fallback_rate": round(totals["fallback"] / denom, 4),
            }
            summary_report["runs"][reader][mode] = {
                "em": metrics_summary["em"],
                "f1": metrics_summary["f1"],
                "oracle_hit@k": metrics_summary["oracle_hit@k"],
                "duplicate_rate": metrics_summary["duplicate_rate"],
                "count": processed,
                "model": answer_model,
                "top_k": args.top_k,
                "top_k_raw": top_k_raw,
            }

            if args.run_dir:
                instrumentation = {key: _summarize_stats(values) for key, values in stat_rollup.items() if values}
                metrics_payload = {
                    "split": args.split,
                    "reader": reader,
                    "retriever": mode,
                    "model": answer_model,
                    "count": processed,
                    "failed": failed,
                    "duration_sec": round(duration_sec, 2),
                    "metrics": metrics_summary,
                    "instrumentation": instrumentation,
                    "top_k": args.top_k,
                    "top_k_raw": top_k_raw,
                    "top_k_raw_source": top_k_raw_source,
                    "fallback_samples": rollup.get("fallback_samples", []),
                }
                _write_json(output_base / "metrics.json", metrics_payload)
                if run_meta is not None:
                    run_meta["ended_at"] = int(time.time())
                    run_meta["duration_sec"] = round(duration_sec, 2)
                    _write_json(output_base / "run_meta.json", run_meta)
                _write_json(
                    output_base / "completed.json",
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

    summary_path = Path(args.output_dir) / f"summary_{args.split}.json"
    summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary written to {}", summary_path)


if __name__ == "__main__":
    main()
