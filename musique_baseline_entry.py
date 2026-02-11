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
from relrag.utils.openai_answer import generate_openai_answer
from relrag.utils.output_eval import has_final_tag
from relrag.utils.vllm_runtime import resolve_vllm_endpoint_model


DEFAULT_STALL_WARN_SEC = 300.0
DEFAULT_STALL_ABORT_SEC = 900.0
DEFAULT_TOP_K = 10
DEFAULT_LIMIT = 0
DEFAULT_WORKERS = 1
DEFAULT_SPLIT = "dev"
DEFAULT_OUTPUT_DIR = "result/musique"
DEFAULT_OVERFETCH = 2.0
MIN_OVERFETCH = 2.0
DEFAULT_BACKFILL_MAX_OVERFETCH = 4.0
DEFAULT_BACKFILL_STEP = 1.5
DEFAULT_BACKFILL_ROUNDS = 3
DEFAULT_LLM_RETRY_ON_EMPTY = 1
DEFAULT_LLM_RETRY_EVIDENCE = 6
DEFAULT_OFFICIAL_SAMPLE = "sample/sample_dev_pred.json"
DEFAULT_UNANSWERABLE = "Insufficient evidence"


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
    common = {}
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


def _resolve_gold_answers(
    example: Dict[str, Any],
    *,
    use_answerable_policy: bool,
    unanswerable_token: str,
) -> Tuple[List[str], bool, List[str]]:
    answerable = bool(example.get("answerable", True))
    raw_answer = example.get("answer") or ""
    answer = str(raw_answer).strip()
    aliases = [str(item).strip() for item in (example.get("answer_aliases") or []) if str(item).strip()]
    if use_answerable_policy and not answerable:
        return [unanswerable_token], answerable, aliases
    golds = [answer] if answer else []
    golds.extend(aliases)
    return golds, answerable, aliases


def _slugify_title(title: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", (title or "")).strip("_").lower()
    cleaned = cleaned or "doc"
    return cleaned[:max_len]


def _make_doc_id(qid: str, idx: int, title: str) -> str:
    slug = _slugify_title(title)
    return f"{qid}_{idx:02d}_{slug}"


def _make_chunk_id(
    doc_id: Optional[str],
    sentence_idx: Optional[int],
    note_id: Optional[str],
) -> Optional[str]:
    if doc_id and sentence_idx is not None:
        return f"{doc_id}#s{int(sentence_idx):04d}"
    if note_id:
        return str(note_id)
    return None


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
    paragraphs = example.get("paragraphs") or []
    for para in paragraphs:
        if not isinstance(para, dict):
            continue
        idx = para.get("idx")
        try:
            para_idx = int(idx)
        except (TypeError, ValueError):
            continue
        title = str(para.get("title") or f"paragraph_{para_idx}")
        text = str(para.get("paragraph_text") or "").strip()
        if not text:
            continue
        units.append(
            {
                "doc_idx": para_idx,
                "title": title,
                "sentence_idx": para_idx,
                "paragraph_idx": para_idx,
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
        paragraph_idx = unit.get("paragraph_idx")
        if paragraph_idx is None:
            paragraph_idx = sentence_idx
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
                "chunk_id": _make_chunk_id(doc_id, int(sentence_idx), note_id),
                "title": title,
                "sentence_idx": int(sentence_idx),
                "paragraph_idx": int(paragraph_idx) if paragraph_idx is not None else int(sentence_idx),
                "idx": int(paragraph_idx) if paragraph_idx is not None else int(sentence_idx),
                "text": text,
                "text_hash": sha1_text(text) if text else None,
                "evidence": text,
                "canonical": f"[{title}] {text}",
                "score": item.get("score"),
                "rank": rank,
                "source": mode,
            }
        )
    return contexts


def _normalize_title(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _extract_gold_sp(
    example: Dict[str, Any],
    *,
    include_decomposition: bool = True,
) -> List[List[Any]]:
    paragraphs = example.get("paragraphs") or []
    idx_to_title: Dict[int, str] = {}
    for para in paragraphs:
        if not isinstance(para, dict):
            continue
        idx = para.get("idx")
        try:
            para_idx = int(idx)
        except (TypeError, ValueError):
            continue
        title = _normalize_title(para.get("title"), f"paragraph_{para_idx}")
        idx_to_title[para_idx] = title

    facts: List[List[Any]] = []
    seen = set()

    def _add(title: str, para_idx: int) -> None:
        key = (title, int(para_idx))
        if key in seen:
            return
        seen.add(key)
        facts.append([title, int(para_idx)])

    for para in paragraphs:
        if not isinstance(para, dict):
            continue
        if not bool(para.get("is_supporting")):
            continue
        try:
            para_idx = int(para.get("idx"))
        except (TypeError, ValueError):
            continue
        title = idx_to_title.get(para_idx, f"paragraph_{para_idx}")
        _add(title, para_idx)

    if include_decomposition:
        for step in example.get("question_decomposition") or []:
            if not isinstance(step, dict):
                continue
            idx = step.get("paragraph_support_idx")
            if idx is None:
                continue
            try:
                para_idx = int(idx)
            except (TypeError, ValueError):
                continue
            title = idx_to_title.get(para_idx)
            if not title:
                continue
            _add(title, para_idx)

    return facts


def _build_pred_sp(retrieved_context: List[Dict[str, Any]]) -> List[List[Any]]:
    facts: List[List[Any]] = []
    seen = set()
    for ctx in retrieved_context:
        title = ctx.get("title")
        idx = ctx.get("paragraph_idx")
        if idx is None:
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
    chunk_id = ctx.get("chunk_id")
    if chunk_id:
        return ("chunk_id", str(chunk_id)), "chunk_id"
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


def _classify_llm_exception(exc: Exception) -> Tuple[str, str]:
    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout", message[:200]
    if "context" in lowered and ("length" in lowered or "token" in lowered or "max" in lowered):
        return "context_len", message[:200]
    return "other", message[:200]


def _resolve_llm_config(args: argparse.Namespace) -> Tuple[str, str]:
    cfg = global_config.load_config()
    return resolve_vllm_endpoint_model(
        endpoint_override=args.endpoint,
        model_override=args.model,
        vllm_cfg=cfg.get("vllm"),
    )


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
    entry_cfg = cfg.get("musique_entry") or cfg.get("entry") or {}
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
    if top_k_raw is not None:
        raw = _coerce_int(top_k_raw, top_k)
        return max(int(math.ceil(top_k * min_factor)), raw), "top_k_raw"
    if overfetch is not None:
        try:
            factor = float(overfetch)
        except (TypeError, ValueError):
            factor = DEFAULT_OVERFETCH
        if factor <= 0:
            factor = DEFAULT_OVERFETCH
        factor = max(factor, min_factor)
        raw = int(math.ceil(top_k * factor))
        return max(int(math.ceil(top_k * min_factor)), raw), "overfetch"
    return int(math.ceil(top_k * min_factor)), "top_k"


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
    *,
    output_name: str = "official_pred.json",
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
    official_path = output_dir / output_name
    with official_path.open("w", encoding="utf-8") as handle:
        json.dump({"answer": answers, "sp": supports}, handle, ensure_ascii=False)
    return official_path


def _alignment_dir(output_dir: Path, pred_path: Path) -> Path:
    return output_dir / f"align_{pred_path.stem}"


def _ensure_alignment_artifacts(align_dir: Path) -> None:
    required = [
        "alignment_audit_snapshot.jsonl",
        "alignment_manifest.json",
        "alignment_audit_report.md",
        "official_pred.json",
    ]
    missing = [name for name in required if not (align_dir / name).exists()]
    if missing:
        raise RuntimeError(f"alignment artifacts missing in {align_dir}: {', '.join(missing)}")


def _run_alignment(
    *,
    repo_root: Path,
    pred_path: Path,
    gold_path: Path,
    output_dir: Path,
    split: str,
) -> None:
    align_dir = _alignment_dir(output_dir, pred_path)
    align_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "eval" / "hotpot_alignment.py"),
        "--pred_jsonl",
        str(pred_path),
        "--gold_jsonl",
        str(gold_path),
        "--output_dir",
        str(align_dir),
        "--pred_official_out",
        str(align_dir / "official_pred.json"),
        "--gold_official_out",
        str(align_dir / "official_gold.json"),
        "--split",
        str(split),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"alignment failed for {pred_path}: {message}")
    _ensure_alignment_artifacts(align_dir)


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
            "gold_sp_subset": (record.get("metrics") or {}).get("gold_sp_subset"),
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


def _update_answerable_counts(rollup: Optional[Dict[str, Any]], record: Dict[str, Any]) -> None:
    if rollup is None:
        return
    if record.get("answerable"):
        rollup["answerable_count"] = rollup.get("answerable_count", 0) + 1
    else:
        rollup["unanswerable_count"] = rollup.get("unanswerable_count", 0) + 1


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


def _drain_futures(
    future_map: Dict[Any, str],
    handle,
    totals: Dict[str, float],
    alt_totals: Optional[Dict[str, float]] = None,
    rollup: Optional[Dict[str, Any]] = None,
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
                _write_retrieval_payloads(record, raw_handle=raw_handle, topk_handle=topk_handle)
                _accumulate_metrics(totals, record.get("metrics") or {})
                if alt_totals is not None:
                    _accumulate_metrics(alt_totals, record.get("metrics_alt") or {})
                _record_fallback_sample(rollup, record)
                _update_answerable_counts(rollup, record)
                succeeded += 1
            completed += 1
            last_progress = time.time()
            if progress is not None:
                progress.update(1)
    return completed, succeeded


def _classify_topk_shortage(
    retrieved_context_raw: List[Dict[str, Any]],
    requested: int,
) -> str:
    if len(retrieved_context_raw) < requested:
        return "retriever_short_return"
    missing = sum(
        1
        for ctx in retrieved_context_raw
        if ctx.get("chunk_id") is None or ctx.get("sentence_idx") is None
    )
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
    qid: str,
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
            ranked = _bm25_search(question, units, bm25_cfg, requested)
        elif mode == "dense":
            ranked = _dense_search(question, units, dense_encoder, embed_cfg, requested)
        else:
            raise ValueError(f"Unsupported mode {mode}")

        retrieved_context_raw = _build_retrieved_context(ranked, mode=mode, qid=qid)
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
    backfill_max_overfetch: float,
    backfill_step: float,
    backfill_rounds: int,
    llm_retry_on_empty: int,
    llm_retry_max_evidence: int,
    dense_encoder,
    include_decomposition_sp: bool,
    unanswerable_token: str,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    qid = str(example.get("id") or "unknown")
    question = str(example.get("question") or "")
    units = _build_sentence_units(example)

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
        question=question,
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
        qid=qid,
    )
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

    gold_primary, answerable, aliases = _resolve_gold_answers(
        example,
        use_answerable_policy=True,
        unanswerable_token=unanswerable_token,
    )
    gold_alt, _, _ = _resolve_gold_answers(
        example,
        use_answerable_policy=False,
        unanswerable_token=unanswerable_token,
    )
    metrics_primary = _best_answer_metrics(short_answer, gold_primary)
    metrics_alt = _best_answer_metrics(short_answer, gold_alt)
    pred_sp = _build_pred_sp(retrieved_context_topk)
    gold_sp = _extract_gold_sp(example, include_decomposition=include_decomposition_sp)
    gold_set = set((item[0], int(item[1])) for item in gold_sp if len(item) >= 2)
    topk_set = set((item[0], int(item[1])) for item in pred_sp if len(item) >= 2)
    gold_subset = 1.0 if gold_set.issubset(topk_set) else 0.0

    top_k_raw_value = dedup_stats.get("top_k_raw")
    overfetch_factor = None
    if isinstance(top_k_raw_value, (int, float)) and top_k:
        overfetch_factor = float(top_k_raw_value) / float(top_k)
    top_k_raw_source_final = top_k_raw_source
    if backfill_attempts > 0:
        top_k_raw_source_final = "backfill"
    top_k_final_value = dedup_stats.get("top_k_final")
    top_k_hit = 1.0 if top_k_final_value == top_k else 0.0
    fallback_flag = 1.0 if (fallback_reason or answer_source != "llm_final") else 0.0
    duplicate_rate = float(dedup_stats.get("duplicate_rate") or 0.0)
    metrics = {
        "em": metrics_primary.get("em", 0.0),
        "f1": metrics_primary.get("f1", 0.0),
        "prec": metrics_primary.get("prec", 0.0),
        "recall": metrics_primary.get("recall", 0.0),
        "gold_sp_subset": gold_subset,
        "top_k_hit": top_k_hit,
        "fallback": fallback_flag,
        "duplicate_rate": duplicate_rate,
    }

    output_record = {
        "_id": qid,
        "id": qid,
        "question": question,
        "answer": short_answer,
        "short_answer": short_answer,
        "answer_source": answer_source,
        "answer_source_detail": answer_source_detail,
        "answerable": answerable,
        "gold_answer": example.get("answer"),
        "gold_aliases": aliases,
        "gold_answers_primary": gold_primary,
        "gold_answers_alt": gold_alt,
        "gold_sp": gold_sp,
        "pred_sp": pred_sp,
        "sp": pred_sp,
        "generated_answer": short_answer,
        "supporting_facts": gold_sp,
        "prediction": short_answer,
        "references": gold_primary,
        "metrics": metrics,
        "metrics_alt": metrics_alt,
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
        "llm_input_hash": prompt_meta.get("llm_input_hash") or "",
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
            "llm_input_hash": prompt_meta.get("llm_input_hash"),
            "context_count": len(units),
            "retrieved_count": len(retrieved_context_topk),
            "llm_retry_used": llm_retry_used,
            "llm_retry_source": llm_retry_source,
            "llm_retry_reason": llm_retry_reason,
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
    parser = argparse.ArgumentParser(description="MuSiQue JSONL entry for BM25/Dense baselines")
    parser.add_argument("--config", help="Path to YAML config file (defaults to relrag/config/config.yaml)")
    parser.add_argument("--data", help="Path to MuSiQue JSONL dataset (fallback to config)")
    parser.add_argument("--run_dir", help="Run directory (enables full artifact layout)")
    parser.add_argument(
        "--official_template",
        help="Official export template (default: sample/sample_dev_pred.json)",
    )
    parser.add_argument(
        "--include_decomposition_sp",
        action="store_true",
        help="Include question_decomposition paragraph_support_idx in gold_sp",
    )
    parser.add_argument(
        "--no_decomposition_sp",
        action="store_false",
        dest="include_decomposition_sp",
        help="Exclude question_decomposition paragraph_support_idx from gold_sp",
    )
    parser.set_defaults(include_decomposition_sp=None)
    parser.add_argument(
        "--unanswerable_token",
        help='Gold token when answerable==false (default: "Insufficient evidence")',
    )
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
    parser.add_argument("--min_overfetch", type=float, help="Minimum overfetch multiplier (fallback to config)")
    parser.add_argument("--backfill_max_overfetch", type=float, help="Max overfetch multiplier for backfill (fallback to config)")
    parser.add_argument("--backfill_step", type=float, help="Backfill growth factor per retry (fallback to config)")
    parser.add_argument("--backfill_rounds", type=int, help="Max backfill attempts (fallback to config)")
    parser.add_argument("--llm_retry_on_empty", type=int, help="Retry LLM on empty/parse fallback (fallback to config)")
    parser.add_argument("--llm_retry_max_evidence", type=int, help="Max evidences on retry (fallback to config)")
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
    dataset_cfg = get_dataset_config(cfg, "musique")
    entry_cfg = _load_entry_config(cfg)
    args.data = _pick_arg(args, entry_cfg, dataset_cfg, "data", None)
    if not args.data:
        raise ValueError("Dataset path missing. Provide --data or set musique_entry.data in config.")
    args.split = _pick_arg(args, entry_cfg, dataset_cfg, "split", DEFAULT_SPLIT)
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
    args.include_decomposition_sp = _pick_arg(
        args,
        entry_cfg,
        dataset_cfg,
        "include_decomposition_sp",
        True,
    )
    args.unanswerable_token = _pick_arg(
        args,
        entry_cfg,
        dataset_cfg,
        "unanswerable_token",
        DEFAULT_UNANSWERABLE,
    )
    args.official_template = _pick_arg(
        args,
        entry_cfg,
        dataset_cfg,
        "official_template",
        DEFAULT_OFFICIAL_SAMPLE,
    )
    args.run_dir = _pick_arg(args, entry_cfg, dataset_cfg, "run_dir", args.run_dir)

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
    official_template_path = _resolve_path(args.official_template)
    if not official_template_path.exists():
        logger.warning("Official template not found: {}", official_template_path)
    args.official_template = str(official_template_path)

    llm_endpoint, llm_model = _resolve_llm_config(args)
    logger.info("Using vLLM endpoint={} model={}", llm_endpoint, llm_model)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _resolve_path(args.run_dir) if args.run_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
    total_examples = _count_examples(data_path, args.limit)

    base_cfg = _apply_dataset_retriever(deepcopy(cfg), "musique")
    if run_dir:
        base_cfg.setdefault("runtime", {})["run_dir"] = str(run_dir)
    modes = _resolve_retriever_modes(
        mode_arg=args.retriever,
        entry_cfg=entry_cfg,
        dataset_cfg=dataset_cfg,
        base_cfg=base_cfg,
    )
    readers = _resolve_readers(args, cfg, dataset_cfg)
    if run_dir and (len(readers) != 1 or len(modes) != 1):
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
                args.min_overfetch,
            )
            output_base = run_dir if run_dir else output_dir
            output_name = "predictions.jsonl" if run_dir else _pred_filename(split, reader, mode, len(readers), len(modes))
            output_path = output_base / output_name
            retrieval_raw_path = run_dir / "retrieved_context_raw.jsonl" if run_dir else None
            retrieval_topk_path = run_dir / "retrieved_context_topk.jsonl" if run_dir else None
            run_started_at = time.time()

            run_meta: Optional[Dict[str, Any]] = None
            if run_dir:
                resolved_cfg = deepcopy(base_cfg)
                resolved_cfg.setdefault("vllm", {})["endpoint"] = llm_endpoint
                resolved_cfg.setdefault("vllm", {})["model"] = llm_model
                if reader_openai_cfg:
                    resolved_cfg["openai"] = deepcopy(reader_openai_cfg)
                entry_snapshot = resolved_cfg.setdefault("musique_entry", {})
                entry_snapshot.update(
                    {
                        "data": str(data_path),
                        "output_dir": str(run_dir),
                        "split": split,
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
                        "workers": args.workers,
                        "include_decomposition_sp": args.include_decomposition_sp,
                        "unanswerable_token": args.unanswerable_token,
                        "official_template": args.official_template,
                    }
                )
                _write_json(run_dir / "config.resolved.json", _sanitize_config(resolved_cfg))
                gold_sp_policy = "paragraphs.is_supporting"
                if args.include_decomposition_sp:
                    gold_sp_policy = gold_sp_policy + " + question_decomposition.paragraph_support_idx"
                run_meta = {
                    "run_dir": str(run_dir),
                    "dataset": "musique",
                    "data_path": str(data_path),
                    "sample_count": total_examples,
                    "split": split,
                    "reader": reader,
                    "retriever": mode,
                    "llm_endpoint": llm_endpoint,
                    "llm_model": llm_model,
                    "openai": {
                        "base_url": (reader_openai_cfg or {}).get("base_url"),
                        "model": (reader_openai_cfg or {}).get("model"),
                    }
                    if reader == "openai"
                    else None,
                    "embedding_endpoint": ((base_cfg.get("retriever") or {}).get("embedding") or {}).get("endpoint"),
                    "top_k": mode_top_k,
                    "top_k_raw": mode_top_k_raw,
                    "top_k_raw_source": top_k_raw_source,
                    "overfetch": args.overfetch,
                    "min_overfetch": args.min_overfetch,
                    "backfill_max_overfetch": args.backfill_max_overfetch,
                    "backfill_step": args.backfill_step,
                    "backfill_rounds": args.backfill_rounds,
                    "llm_retry_on_empty": args.llm_retry_on_empty,
                    "llm_retry_max_evidence": args.llm_retry_max_evidence,
                    "workers": args.workers,
                    "include_decomposition_sp": args.include_decomposition_sp,
                    "unanswerable_token": args.unanswerable_token,
                    "answer_policy_primary": f"answerable_false_as_{args.unanswerable_token}",
                    "answer_policy_secondary": "always_use_answer_field",
                    "gold_sp_policy": gold_sp_policy,
                    "pred_sp_policy": "retrieved_context_topk (title, paragraph_idx)",
                    "official_export": {
                        "template": str(args.official_template),
                        "mapping": "answer=short_answer, sp=pred_sp",
                    },
                    "command": _sanitize_argv(sys.argv),
                    "started_at": int(run_started_at),
                    "host": socket.gethostname(),
                    "platform": platform.platform(),
                    "python": sys.version,
                    "git": _git_info(repo_root),
                    "env": _env_snapshot(),
                }
                _write_json(run_dir / "run_meta.json", run_meta)

            logger.info("Running reader={} mode={} -> {}", reader, mode, output_path)
            totals = {
                "em": 0.0,
                "f1": 0.0,
                "prec": 0.0,
                "recall": 0.0,
                "gold_sp_subset": 0.0,
                "top_k_hit": 0.0,
                "fallback": 0.0,
                "duplicate_rate": 0.0,
            }
            alt_totals = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
            rollup: Dict[str, Any] = {"fallback_samples": [], "answerable_count": 0, "unanswerable_count": 0}
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
                                    backfill_max_overfetch=args.backfill_max_overfetch,
                                    backfill_step=args.backfill_step,
                                    backfill_rounds=args.backfill_rounds,
                                    llm_retry_on_empty=args.llm_retry_on_empty,
                                    llm_retry_max_evidence=args.llm_retry_max_evidence,
                                    dense_encoder=dense_encoder,
                                    include_decomposition_sp=args.include_decomposition_sp,
                                    unanswerable_token=args.unanswerable_token,
                                    run_dir=str(run_dir) if run_dir else None,
                                )
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                handle.flush()
                                _write_retrieval_payloads(record, raw_handle=raw_handle, topk_handle=topk_handle)
                                _accumulate_metrics(totals, record.get("metrics") or {})
                                _accumulate_metrics(alt_totals, record.get("metrics_alt") or {})
                                _record_fallback_sample(rollup, record)
                                _update_answerable_counts(rollup, record)
                                processed += 1
                            except Exception as exc:
                                qid = example.get("id")
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
                                qid = str(example.get("id") or "unknown")
                                future = executor.submit(
                                    _process_example,
                                    example,
                                    base_cfg,
                                    llm_endpoint,
                                    llm_model,
                                    mode,
                                    reader,
                                    reader_openai_cfg,
                                    mode_top_k,
                                    mode_top_k_raw,
                                    top_k_raw_source,
                                    args.backfill_max_overfetch,
                                    args.backfill_step,
                                    args.backfill_rounds,
                                    args.llm_retry_on_empty,
                                    args.llm_retry_max_evidence,
                                    dense_encoder,
                                    args.include_decomposition_sp,
                                    args.unanswerable_token,
                                )
                                future_map[future] = qid
                                scheduled += 1
                                if len(future_map) >= buffer_cap:
                                    done_count, ok_count = _drain_futures(
                                        future_map,
                                        handle,
                                        totals,
                                        alt_totals=alt_totals,
                                        rollup=rollup,
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
                                    alt_totals=alt_totals,
                                    rollup=rollup,
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
            duration_sec = max(0.0, time.time() - run_started_at)
            logger.info("Reader {} mode {} completed {} examples (failed {})", reader, mode, processed, failed)
            denom = processed if processed > 0 else 1
            metrics_summary = {
                "em": round(totals["em"] / denom, 4),
                "f1": round(totals["f1"] / denom, 4),
                "prec": round(totals["prec"] / denom, 4),
                "recall": round(totals["recall"] / denom, 4),
                "gold_sp_subset": round(totals["gold_sp_subset"] / denom, 4),
                "top_k_hit": round(totals["top_k_hit"] / denom, 4),
                "fallback_rate": round(totals["fallback"] / denom, 4),
                "duplicate_rate": round(totals["duplicate_rate"] / denom, 6),
            }
            metrics_alt_summary = {
                "em": round(alt_totals["em"] / denom, 4),
                "f1": round(alt_totals["f1"] / denom, 4),
                "prec": round(alt_totals["prec"] / denom, 4),
                "recall": round(alt_totals["recall"] / denom, 4),
            }
            summary_report["runs"][reader][mode] = {
                "em": metrics_summary["em"],
                "f1": metrics_summary["f1"],
                "gold_sp_subset": metrics_summary["gold_sp_subset"],
                "top_k_hit": metrics_summary["top_k_hit"],
                "fallback_rate": metrics_summary["fallback_rate"],
                "duplicate_rate": metrics_summary["duplicate_rate"],
                "count": processed,
                "model": answer_model,
                "top_k": mode_top_k,
                "top_k_raw": mode_top_k_raw,
            }

            if run_dir:
                metrics_payload = {
                    "split": split,
                    "reader": reader,
                    "retriever": mode,
                    "model": answer_model,
                    "count": processed,
                    "failed": failed,
                    "duration_sec": round(duration_sec, 2),
                    "metrics": metrics_summary,
                    "metrics_alt": metrics_alt_summary,
                    "top_k": mode_top_k,
                    "top_k_raw": mode_top_k_raw,
                    "top_k_raw_source": top_k_raw_source,
                    "answerable_count": rollup.get("answerable_count", 0),
                    "unanswerable_count": rollup.get("unanswerable_count", 0),
                    "fallback_samples": rollup.get("fallback_samples", []),
                }
                _write_json(run_dir / "metrics.json", metrics_payload)
                official_path = _write_official_output(output_path, run_dir, output_name="official_pred.json")
                logger.info("Official-format output written to {}", official_path)
                if run_meta is not None:
                    run_meta["ended_at"] = int(time.time())
                    run_meta["duration_sec"] = round(duration_sec, 2)
                    _write_json(run_dir / "run_meta.json", run_meta)
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

    summary_path = output_dir / f"summary_{split}.json"
    summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary written to {}", summary_path)


if __name__ == "__main__":
    main()
