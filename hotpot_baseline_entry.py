import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
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


DEFAULT_STALL_WARN_SEC = 300.0
DEFAULT_STALL_ABORT_SEC = 900.0
DEFAULT_TOP_K = 10
DEFAULT_LIMIT = 0
DEFAULT_WORKERS = 1
DEFAULT_SPLIT = "dev"
DEFAULT_OUTPUT_DIR = "result"
DEFAULT_OVERFETCH = 1.0


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_examples(path: Path, limit: int) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            if limit and total >= limit:
                break
    return total


def _slugify_title(title: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", (title or "")).strip("_").lower()
    cleaned = cleaned or "doc"
    return cleaned[:max_len]


def _make_doc_id(qid: str, idx: int, title: str) -> str:
    slug = _slugify_title(title)
    return f"{qid}_{idx:02d}_{slug}"


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


def _build_sentence_units(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    context = example.get("context") or []
    for doc_idx, item in enumerate(context):
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, sentences = item
        if not isinstance(title, str) or not isinstance(sentences, list):
            continue
        for sent_idx, sentence in enumerate(sentences):
            text = str(sentence).strip()
            if not text:
                continue
            units.append(
                {
                    "doc_idx": doc_idx,
                    "title": title,
                    "sentence_idx": sent_idx,
                    "text": text,
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
    qid: str,
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for rank, item in enumerate(ranked, start=1):
        unit = item.get("unit") or {}
        title = unit.get("title")
        sentence_idx = unit.get("sentence_idx")
        text = unit.get("text") or ""
        doc_idx = unit.get("doc_idx")
        if title is None or sentence_idx is None:
            continue
        doc_id = _make_doc_id(qid, int(doc_idx or 0), str(title))
        note_id = f"{qid}_{int(doc_idx or 0):02d}_{int(sentence_idx):02d}"
        contexts.append(
            {
                "note_id": note_id,
                "doc_id": doc_id,
                "title": title,
                "sentence_idx": int(sentence_idx),
                "evidence": text,
                "canonical": f"[{title}] {text}",
                "score": item.get("score"),
                "rank": rank,
                "source": mode,
            }
        )
    return contexts


def _normalize_supporting_facts(raw: Any) -> List[List[Any]]:
    if not raw:
        return []
    normalized: List[List[Any]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = item[0]
        idx = item[1]
        if title is None:
            continue
        try:
            sent_idx = int(idx)
        except (TypeError, ValueError):
            continue
        normalized.append([str(title), sent_idx])
    return normalized


def _extract_gold_sp(example: Dict[str, Any]) -> List[List[Any]]:
    if "supporting_facts" in example:
        return _normalize_supporting_facts(example.get("supporting_facts"))
    if "sp" in example:
        return _normalize_supporting_facts(example.get("sp"))
    return []


def _build_pred_sp(retrieved_context: List[Dict[str, Any]]) -> List[List[Any]]:
    facts: List[List[Any]] = []
    seen = set()
    for ctx in retrieved_context:
        title = ctx.get("title")
        idx = ctx.get("sentence_idx")
        if title is None or idx is None:
            continue
        key = (title, int(idx))
        if key in seen:
            continue
        facts.append([title, int(idx)])
        seen.add(key)
    return facts


def _dedup_key(ctx: Dict[str, Any], idx: int) -> Tuple[Tuple[Any, ...], str]:
    doc_id = ctx.get("doc_id")
    sent_idx = ctx.get("sentence_idx")
    title = ctx.get("title")
    if doc_id and sent_idx is not None:
        return ("doc_id", str(doc_id), int(sent_idx)), "doc_id"
    if title and sent_idx is not None:
        return ("title", str(title), int(sent_idx)), "title"
    text = ctx.get("canonical") or ctx.get("evidence") or ""
    if title and text:
        return ("title_hash", str(title), sha1_text(str(text))), "title_hash"
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


def _resolve_llm_config(args: argparse.Namespace) -> Tuple[str, str]:
    cfg = global_config.load_config()
    endpoint = args.endpoint or (cfg.get("vllm") or {}).get("endpoint")
    model = args.model or (cfg.get("vllm") or {}).get("model")
    if not endpoint or not model:
        raise ValueError("LLM endpoint/model is required (use args or config)")
    return endpoint, model


def generate_answer(
    question: str,
    evidences: List[Dict[str, Any]],
    *,
    reader: str,
    llm_endpoint: str,
    llm_model: str,
    openai_cfg: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    prompt_capture: Dict[str, Any] = {}
    if reader == "vllm":
        raw_answer = answer(
            question=question,
            evidences=evidences,
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            prompt_capture=prompt_capture,
        )
        prompt_name = prompt_capture.get("prompt_name") or answerer_module.ANSWER_PROMPT_NAME
        prompt_template_hash = _prompt_template_hash(prompt_name)
        llm_input_hash = _resolve_llm_input_hash(prompt_capture)
        return raw_answer, {
            "prompt_name": prompt_name,
            "prompt_template_hash": prompt_template_hash,
            "llm_input_hash": llm_input_hash,
        }
    if reader == "openai":
        if not openai_cfg:
            raise ValueError("OpenAI config missing for reader=openai")
        raw_answer = generate_openai_answer(question, evidences, openai_cfg, prompt_capture=prompt_capture)
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
        }
    raise ValueError(f"Unknown reader type: {reader}")


def _load_entry_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    entry_cfg = cfg.get("hotpot_entry") or cfg.get("entry") or {}
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


def _resolve_top_k_raw(top_k: int, top_k_raw: Optional[Any], overfetch: Optional[Any]) -> Tuple[int, str]:
    if top_k_raw is not None:
        raw = _coerce_int(top_k_raw, top_k)
        return max(top_k, raw), "top_k_raw"
    if overfetch is not None:
        try:
            factor = float(overfetch)
        except (TypeError, ValueError):
            factor = DEFAULT_OVERFETCH
        if factor <= 0:
            factor = DEFAULT_OVERFETCH
        raw = int(math.ceil(top_k * factor))
        return max(top_k, raw), "overfetch"
    return top_k, "top_k"


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
    retriever_override = dataset_cfg.get("retriever") if isinstance(dataset_cfg, dict) else None
    if isinstance(retriever_override, dict):
        base_retriever = cfg.get("retriever") or {}
        cfg["retriever"] = _deep_merge(base_retriever, retriever_override)
    return cfg


def _parse_modes(value: Any) -> List[str]:
    allowed = {"bm25", "dense"}
    if not value:
        return ["bm25", "dense"]
    if isinstance(value, str):
        parts = [p for p in re.split(r"[,\s]+", value.strip()) if p]
    elif isinstance(value, (list, tuple)):
        parts = [str(p) for p in value if str(p)]
    else:
        parts = [str(value)]
    normalized: List[str] = []
    skipped: List[str] = []
    seen = set()
    for part in parts:
        key = part.strip().lower()
        if not key:
            continue
        if key not in allowed:
            skipped.append(part)
            continue
        if key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    if skipped:
        logger.warning("Skipping unsupported modes for baseline: {}", skipped)
    return normalized or ["bm25", "dense"]


def _mode_config(cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    retriever_cfg = cfg.get("retriever") or {}
    if mode == "dense":
        dense_cfg = retriever_cfg.get("dense")
        if isinstance(dense_cfg, dict):
            return dense_cfg
        return retriever_cfg.get("embedding") or {}
    if mode == "bm25":
        return retriever_cfg.get("bm25") or {}
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
    entry_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
) -> List[str]:
    raw = None
    if mode_arg:
        raw = mode_arg
    elif dataset_cfg.get("retrievers") is not None:
        raw = dataset_cfg.get("retrievers")
    elif dataset_cfg.get("retriever_modes") is not None:
        raw = dataset_cfg.get("retriever_modes")
    elif entry_cfg.get("retrievers") is not None:
        raw = entry_cfg.get("retrievers")
    elif entry_cfg.get("modes") is not None:
        raw = entry_cfg.get("modes")
    modes = _parse_modes(raw) if raw is not None else _parse_modes(None)
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


def _pred_filename(split: str, reader: str, mode: str, reader_count: int, mode_count: int) -> str:
    if reader_count == 1 and mode_count > 1:
        return f"pred_{split}_{mode}.jsonl"
    if mode_count == 1 and reader_count > 1:
        return f"pred_{split}_{reader}.jsonl"
    return f"pred_{split}_{reader}_{mode}.jsonl"


def _write_official_output(
    jsonl_path: Path,
    output_dir: Path,
    timestamp: int,
    suffix: Optional[str] = None,
) -> Path:
    answers: Dict[str, Any] = {}
    supports: Dict[str, Any] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = str(record.get("_id") or "")
            if not qid:
                continue
            if "short_answer" not in record:
                raise ValueError(f"missing short_answer for qid={qid}")
            if "pred_sp" not in record:
                raise ValueError(f"missing pred_sp for qid={qid}")
            answers[qid] = record.get("short_answer") or ""
            supports[qid] = record.get("pred_sp") or []
    suffix_part = f"_{suffix}" if suffix else ""
    official_path = output_dir / f"result_{timestamp}{suffix_part}_official.json"
    with official_path.open("w", encoding="utf-8") as handle:
        json.dump({"answer": answers, "sp": supports}, handle, ensure_ascii=False)
    return official_path


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


def _process_example(
    example: Dict[str, Any],
    *,
    base_cfg: Dict[str, Any],
    llm_endpoint: str,
    llm_model: str,
    mode: str,
    reader: str,
    openai_cfg: Optional[Dict[str, Any]],
    top_k: int,
    top_k_raw: int,
    top_k_raw_source: str,
    dense_encoder,
) -> Dict[str, Any]:
    qid = str(example.get("_id") or "unknown")
    question = str(example.get("question") or "")
    units = _build_sentence_units(example)

    retriever_cfg = base_cfg.get("retriever") or {}
    bm25_cfg = retriever_cfg.get("bm25") or {}
    embed_cfg = retriever_cfg.get("embedding") or {}

    if mode == "bm25":
        ranked = _bm25_search(question, units, bm25_cfg, top_k_raw)
    elif mode == "dense":
        ranked = _dense_search(question, units, dense_encoder, embed_cfg, top_k_raw)
    else:
        raise ValueError(f"Unsupported mode {mode}")

    retrieved_context_raw = _build_retrieved_context(ranked, mode=mode, qid=qid)
    retrieved_context_topk, dedup_stats = _dedup_retrieved_context(retrieved_context_raw, top_k=top_k)
    pred_sp = _build_pred_sp(retrieved_context_topk)
    gold_sp = _extract_gold_sp(example)
    evidences = retrieved_context_topk

    raw_answer, prompt_meta = generate_answer(
        question=question,
        evidences=evidences,
        reader=reader,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
        openai_cfg=openai_cfg,
    )
    short_answer, answer_source, answer_source_detail = resolve_short_answer(None, raw_answer)
    answer_model = openai_cfg.get("model") if reader == "openai" and openai_cfg else llm_model

    references: List[str] = []
    raw_reference = example.get("answer")
    if isinstance(raw_reference, list):
        references = [str(item).strip() for item in raw_reference if str(item).strip()]
    elif raw_reference:
        references = [str(raw_reference).strip()]
    metrics = score_metrics(short_answer, references)

    output_record = {
        "_id": qid,
        "question": question,
        "answer": short_answer,
        "short_answer": short_answer,
        "answer_source": answer_source,
        "answer_source_detail": answer_source_detail,
        "gold_sp": gold_sp,
        "pred_sp": pred_sp,
        "sp": pred_sp,
        "generated_answer": short_answer,
        "supporting_facts": gold_sp,
        "prediction": short_answer,
        "references": references,
        "metrics": metrics,
        "mode": mode,
        "reader": reader,
        "model": answer_model,
        "retrieved_context_raw": retrieved_context_raw,
        "retrieved_context_topk": retrieved_context_topk,
        "retrieved_context": retrieved_context_topk,
        "top_k_raw": dedup_stats.get("top_k_raw"),
        "top_k_final": dedup_stats.get("top_k_final"),
        "duplicate_rate": dedup_stats.get("duplicate_rate"),
        "llm_input_hash": prompt_meta.get("llm_input_hash") or "",
        "intermediate": {
            "llm_raw": raw_answer,
            "llm_has_final": has_final_tag(raw_answer),
            "retrieval_mode": mode,
            "reader": reader,
            "model": answer_model,
            "top_k": top_k,
            "top_k_raw": top_k_raw,
            "top_k_raw_source": top_k_raw_source,
            "prompt_name": prompt_meta.get("prompt_name"),
            "prompt_template_hash": prompt_meta.get("prompt_template_hash"),
            "system_prompt_name": prompt_meta.get("system_prompt_name"),
            "system_prompt_hash": prompt_meta.get("system_prompt_hash"),
            "llm_input_hash": prompt_meta.get("llm_input_hash"),
            "context_count": len(units),
            "retrieved_count": len(retrieved_context_topk),
        },
    }
    return output_record


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


def main() -> None:
    parser = argparse.ArgumentParser(description="HotpotQA JSONL entry for BM25/Dense baselines")
    parser.add_argument("--config", help="Path to YAML config file (defaults to relrag/config/config.yaml)")
    parser.add_argument("--data", help="Path to HotpotQA JSONL dataset (fallback to config)")
    parser.add_argument("--endpoint", help="vLLM endpoint (defaults to config)")
    parser.add_argument("--model", help="LLM model name (defaults to config)")
    parser.add_argument("--reader", help="Reader backend: vllm or openai (fallback to config)")
    parser.add_argument("--retriever", help="Retriever mode: bm25 or dense (fallback to config)")
    parser.add_argument("--split", help="Dataset split label for output naming (fallback to config)")
    parser.add_argument("--openai_model", help="OpenAI model name (fallback to config)")
    parser.add_argument("--openai_api_key", help="OpenAI API key (reads env if omitted)")
    parser.add_argument("--openai_temperature", type=float, help="OpenAI temperature (fallback to config)")
    parser.add_argument("--openai_max_tokens", type=int, help="OpenAI max tokens (fallback to config)")
    parser.add_argument("--top_k", type=int, help="Top-k retrieval fanout (fallback to config)")
    parser.add_argument("--top_k_raw", type=int, help="Raw top-k before dedup (fallback to config/overfetch)")
    parser.add_argument("--overfetch", type=float, help="Overfetch multiplier before dedup (fallback to config)")
    parser.add_argument("--limit", type=int, help="Process only first N examples (fallback to config)")
    parser.add_argument("--workers", type=int, help="Parallel workers (single process, fallback to config)")
    parser.add_argument("--output_dir", help="Output directory (fallback to config)")
    parser.add_argument("--stall_warn_sec", type=float, help="Warn if no worker finishes within this many seconds (fallback to config)")
    parser.add_argument("--stall_abort_sec", type=float, help="Abort pending workers after this many idle seconds (0 to disable, fallback to config)")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent

    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else repo_root / path

    cfg = ConfigLoader(args.config).load_config() if args.config else global_config.load_config()
    dataset_cfg = get_dataset_config(cfg, "hotpotqa")
    entry_cfg = _load_entry_config(cfg)
    args.data = _pick_arg(args, entry_cfg, dataset_cfg, "data", None)
    if not args.data:
        raise ValueError("Dataset path missing. Provide --data or set hotpot_entry.data in config.")
    args.split = _pick_arg(args, entry_cfg, dataset_cfg, "split", DEFAULT_SPLIT)
    args.output_dir = _pick_arg(args, entry_cfg, dataset_cfg, "output_dir", DEFAULT_OUTPUT_DIR)
    args.top_k = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "top_k", DEFAULT_TOP_K),
        DEFAULT_TOP_K,
    )
    args.top_k_raw = _pick_arg(args, entry_cfg, dataset_cfg, "top_k_raw", None)
    args.overfetch = _pick_arg(args, entry_cfg, dataset_cfg, "overfetch", None)
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

    data_path = _resolve_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    llm_endpoint, llm_model = _resolve_llm_config(args)
    logger.info("Using vLLM endpoint={} model={}", llm_endpoint, llm_model)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    total_examples = _count_examples(data_path, args.limit)

    base_cfg = _apply_dataset_retriever(deepcopy(cfg), "hotpotqa")
    modes = _resolve_retriever_modes(
        mode_arg=args.retriever,
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

    embed_cfg = (base_cfg.get("retriever") or {}).get("embedding") or {}
    dense_encoder = _build_dense_encoder(embed_cfg) if "dense" in modes else None

    split = str(args.split or DEFAULT_SPLIT).strip() or DEFAULT_SPLIT
    summary_report: Dict[str, Any] = {
        "split": split,
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
            mode_top_k_raw, top_k_raw_source = _resolve_top_k_raw(
                mode_top_k,
                args.top_k_raw,
                args.overfetch,
            )
            output_name = _pred_filename(split, reader, mode, len(readers), len(modes))
            output_path = output_dir / output_name
            logger.info("Writing results to {}", output_path)
            totals = {"bleu1": 0.0, "bleu4": 0.0, "rougeL": 0.0, "meteor": 0.0}
            progress = ProgressBar(total_examples)
            processed = 0
            completed = 0
            with output_path.open("w", encoding="utf-8") as handle:
                if args.workers <= 1:
                    for example in _load_jsonl(data_path):
                        if args.limit and completed >= args.limit:
                            break
                        try:
                            record = _process_example(
                                example,
                                base_cfg=base_cfg,
                                llm_endpoint=llm_endpoint,
                                llm_model=llm_model,
                                mode=mode,
                                reader=reader,
                                openai_cfg=reader_openai_cfg,
                                top_k=mode_top_k,
                                top_k_raw=mode_top_k_raw,
                                top_k_raw_source=top_k_raw_source,
                                dense_encoder=dense_encoder,
                            )
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            handle.flush()
                            _accumulate_metrics(totals, record.get("metrics") or {})
                            processed += 1
                        except Exception as exc:
                            qid = example.get("_id")
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
                        for example in _load_jsonl(data_path):
                            if args.limit and scheduled >= args.limit:
                                break
                            qid = str(example.get("_id") or "unknown")
                            future = executor.submit(
                                _process_example,
                                example,
                                base_cfg=base_cfg,
                                llm_endpoint=llm_endpoint,
                                llm_model=llm_model,
                                mode=mode,
                                reader=reader,
                                openai_cfg=reader_openai_cfg,
                                top_k=mode_top_k,
                                top_k_raw=mode_top_k_raw,
                                top_k_raw_source=top_k_raw_source,
                                dense_encoder=dense_encoder,
                            )
                            future_map[future] = qid
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

            if len(readers) == 1 and len(modes) == 1:
                official_path = _write_official_output(output_path, output_dir, timestamp)
                logger.info("Official-format output written to {}", official_path)

    if len(readers) == 1:
        summary_report["modes"] = summary_report["runs"][readers[0]]
    if len(modes) == 1:
        summary_report["models"] = {reader: summary_report["runs"][reader][modes[0]] for reader in readers}

    summary_path = output_dir / f"summary_{split}.json"
    summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary written to {}", summary_path)


if __name__ == "__main__":
    main()
