import argparse
import json
import math
import os
import platform
import socket
import subprocess
import re
import shutil
import sys
import time
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

try:
    from filelock import FileLock
except Exception:  # pragma: no cover - optional dependency fallback
    class FileLock:  # type: ignore[no-redef]
        def __init__(self, _path: str) -> None:
            self.path = _path

        def __enter__(self) -> "FileLock":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False
from loguru import logger

from relrag.api import retrieve, answer
from relrag.config.dataset_config import (
    get_dataset_config,
    resolve_openai_api_key,
    resolve_openai_config,
    resolve_reader,
)
from relrag.config.config_loader import ConfigLoader, config as global_config
from relrag.generator import answerer as answerer_module
from relrag.indexer import IndexBuilder
from relrag.indexer.bm25_index import BM25IndexBuilder
from relrag.indexer.embedding_index import EmbeddingIndexBuilder
from relrag.postprocess.notes_postprocess import build_alias_map
from relrag.retriever.chunk_store import ChunkStore
from relrag.retriever.note_store import NoteStore
from relrag.prompt import load_prompt
from relrag.utils.answer_source import resolve_short_answer, sha1_text
from relrag.utils.openai_answer import generate_openai_answer
from relrag.utils.eval_metrics import score_metrics
from relrag.utils.output_eval import has_final_tag
from relrag.doc.chunking_strategies import SentenceAwareChunker, FixedWindowChunker, Chunker


DEFAULT_STALL_WARN_SEC = 300.0
DEFAULT_STALL_ABORT_SEC = 900.0
DEFAULT_TOP_K = 10
DEFAULT_LIMIT = 0
DEFAULT_WORKERS = 1
DEFAULT_SPLIT = "dev"
DEFAULT_CACHE_DIR = "result/cache"
DEFAULT_OUTPUT_DIR = "result"
DEFAULT_DEBUG_DIR = "result/debug"
DEFAULT_DEBUG_MAX_NOTES = 50
DEFAULT_OVERFETCH = 2.0
MIN_OVERFETCH = 2.0
DEFAULT_BACKFILL_MAX_OVERFETCH = 4.0
DEFAULT_BACKFILL_STEP = 1.5
DEFAULT_BACKFILL_ROUNDS = 3
DEFAULT_LLM_RETRY_ON_EMPTY = 1
DEFAULT_LLM_RETRY_EVIDENCE = 6
DEFAULT_SHORTAGE_REFILL_ENABLED = True
DEFAULT_SHORTAGE_REFILL_MAX_CANDIDATES = 48
DEFAULT_SHORTAGE_REFILL_MIN_SCORE = 0.35
DEFAULT_SHORTAGE_REFILL_PREFER_NEW_TITLES = True
DEFAULT_PRED_SP_POLICY = "high_confidence"
DEFAULT_PRED_SP_MAX_FACTS = 4
DEFAULT_PRED_SP_MIN_SCORE = 0.0
DEFAULT_PRED_SP_DROP_WEAK = True
DEFAULT_PRED_SP_PREFER_NEW_TITLES = True
DEFAULT_TITLE_DIVERSITY_ENABLED = True
DEFAULT_TITLE_DIVERSITY_TOP_N = 5
DEFAULT_TITLE_DIVERSITY_KEEP_FIRST = True
DEFAULT_QUERY_TITLE_PROMOTION_ENABLED = True
DEFAULT_QUERY_TITLE_PROMOTION_WINDOW = 5
DEFAULT_PREDICATE_MODE = "on"
DEFAULT_PREDICATE_RANDOM_SEED = 2026
DEFAULT_EXPORT_TOP_K_MAX = 50

_REFILL_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "in",
    "on",
    "for",
    "to",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "which",
    "who",
    "what",
    "when",
    "where",
    "how",
}


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def normalize_title(title: Any) -> str:
    """Normalize title conservatively for Hotpot title matching."""
    text = str(title or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


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


def _write_docs_for_example(
    example: Dict[str, Any],
    docs_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Dict[str, Any]]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_index: Dict[str, Dict[str, Any]] = {}
    context = example.get("context") or []
    qid = str(example.get("_id") or "unknown")

    for idx, item in enumerate(context):
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, sentences = item
        if not isinstance(title, str) or not isinstance(sentences, list):
            continue
        doc_id = _make_doc_id(qid, idx, title)
        doc_index[doc_id] = {"title": title, "sentences": sentences}

        text = " ".join(str(s).strip() for s in sentences if str(s).strip()).strip()
        if not text:
            continue
        doc_path = docs_dir / f"{doc_id}.txt"
        if doc_path.exists() and not overwrite:
            continue
        doc_path.write_text(text, encoding="utf-8")

    return doc_index


def _write_sentence_notes_for_example(
    doc_index: Dict[str, Dict[str, Any]],
    notes_path: Path,
    overwrite: bool = False,
) -> int:
    if notes_path.exists() and not overwrite:
        return 0
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with notes_path.open("w", encoding="utf-8") as handle:
        for doc_id, meta in doc_index.items():
            title = str(meta.get("title") or "").strip()
            sentences = [str(s).strip() for s in (meta.get("sentences") or []) if str(s).strip()]
            alias_map, alias_to_canonical = build_alias_map(sentences)
            canonical_title = title
            if title:
                canonical_title = alias_to_canonical.get(title.lower()) or title
            subject_aliases: List[str] = []
            if canonical_title and canonical_title in alias_map:
                subject_aliases.extend([a for a in (alias_map.get(canonical_title) or []) if a and a != title])
            if canonical_title and canonical_title != title:
                subject_aliases.insert(0, canonical_title)
            dedup_aliases: List[str] = []
            seen_aliases = set()
            for alias in subject_aliases:
                norm = str(alias).strip()
                if not norm:
                    continue
                key = norm.lower()
                if key in seen_aliases:
                    continue
                dedup_aliases.append(norm)
                seen_aliases.add(key)
            for sent_idx, sentence in enumerate(sentences):
                text = sentence
                note = {
                    "note_id": f"{doc_id}#s{sent_idx:04d}",
                    "subj": title or str(doc_id),
                    "pred": "sentence",
                    "obj": text,
                    "subj_type": "CONCEPT",
                    "obj_type": "CONCEPT",
                    "evidence": text,
                    "meta": {
                        "source": str(doc_id),
                        "sentence_idx": sent_idx,
                        "confidence": 1.0,
                        "final_conf": 1.0,
                        "quality_score": 1.0,
                        "evidence_canonical": text,
                        "alias_map": alias_map,
                        "subject_profile": {
                            "type": "CONCEPT",
                            "aliases": dedup_aliases,
                        },
                    },
                }
                handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                written += 1
    return written


def _write_chunks_for_example(
    doc_index: Dict[str, Dict[str, Any]],
    chunks_path: Path,
    chunker: Optional[Chunker] = None,
    overwrite: bool = False,
) -> int:
    if chunks_path.exists() and not overwrite:
        return 0
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    if chunker is None:
        chunker = SentenceAwareChunker()
    written = 0
    with chunks_path.open("w", encoding="utf-8") as handle:
        for idx, (doc_id, meta) in enumerate(doc_index.items()):
            title = str(meta.get("title") or doc_id)
            sentences = [str(s).strip() for s in (meta.get("sentences") or []) if str(s).strip()]
            if not sentences:
                continue
            text = " ".join(sentences)
            chunks = chunker.chunk(doc_id, text, meta)
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                written += 1
    return written


def _resolve_doc_id_from_source(source: Optional[str]) -> Optional[str]:
    text = str(source or "").strip()
    if not text:
        return None
    return text.split("#", 1)[0].strip() or None


def _resolve_preferred_doc_id(
    *,
    evidence: Dict[str, Any],
    note_id: Optional[str],
    source: Optional[str],
    doc_index: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    candidates: List[str] = []

    def _append(raw: Any) -> None:
        text = str(raw or "").strip()
        if text and text not in candidates:
            candidates.append(text)

    _append(evidence.get("doc_id"))
    _append(_resolve_doc_id_from_source(source))
    _append(_resolve_doc_id_from_source(evidence.get("chunk_id")))
    _append(_resolve_doc_id_from_source(note_id))
    _append(_resolve_doc_id_from_source(evidence.get("note_id")))

    for candidate in candidates:
        if candidate in doc_index:
            return candidate
    return candidates[0] if candidates else None


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\W+", "", (text or "").lower())


def _find_sentence_index(evidence: str, sentences: List[str]) -> Optional[int]:
    ev_norm = _normalize_for_match(evidence)
    if not ev_norm:
        return None
    for idx, sent in enumerate(sentences):
        sent_norm = _normalize_for_match(sent)
        if not sent_norm:
            continue
        if sent_norm in ev_norm or ev_norm in sent_norm:
            return idx
    return None


def _coerce_sentence_idx(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_retrieved_context(
    evidences: List[Dict[str, Any]],
    note_store: NoteStore,
    doc_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    def _first_non_empty(*values: Any) -> Optional[str]:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return None

    contexts: List[Dict[str, Any]] = []
    for ev in evidences:
        note_id = ev.get("note_id")
        note = note_store.get_weak(note_id) if note_id else None
        meta = (note or {}).get("meta") or {}
        source = meta.get("source") or ev.get("source")
        doc_id = _resolve_preferred_doc_id(
            evidence=ev,
            note_id=note_id,
            source=source,
            doc_index=doc_index,
        )
        doc_meta = doc_index.get(doc_id) if doc_id else None
        title = _first_non_empty(
            (doc_meta or {}).get("title"),
            meta.get("doc_title"),
            meta.get("title"),
            ev.get("doc_title"),
            ev.get("title"),
        )
        title = normalize_title(title) if title else None
        sentences = doc_meta.get("sentences") if doc_meta else []
        evidence_text = ev.get("canonical") or ev.get("evidence") or ""
        sentence_idx = _coerce_sentence_idx(meta.get("sentence_idx"))
        if sentence_idx is None:
            sentence_idx = _coerce_sentence_idx(ev.get("sentence_idx"))
        if sentence_idx is None and sentences:
            sentence_idx = _find_sentence_index(evidence_text, sentences)

        text = ev.get("evidence") or ev.get("canonical") or ""
        contexts.append(
            {
                "note_id": note_id,
                "doc_id": doc_id,
                "chunk_id": _make_chunk_id(doc_id, sentence_idx, note_id),
                "title": title,
                "doc_title": title,
                "sentence_idx": sentence_idx,
                "text": text,
                "text_hash": sha1_text(text) if text else None,
                "evidence": ev.get("evidence"),
                "canonical": ev.get("canonical"),
                "weak": bool(ev.get("weak", False)),
                "score": ev.get("score"),
                "source": source,
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


def _normalize_pred_sp_policy(policy: Any) -> str:
    key = str(policy or "").strip().lower().replace("-", "_")
    if key in {"", "topk", "legacy", "legacy_topk"}:
        return "topk"
    if key in {"high_confidence", "highconf", "confidence"}:
        return "high_confidence"
    raise ValueError(f"Unsupported pred_sp policy: {policy}")


def _safe_score(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _score_from_stage_candidate(candidate: Dict[str, Any], keys: List[str]) -> float:
    for key in keys:
        if key not in candidate:
            continue
        value = candidate.get(key)
        if value is None:
            continue
        return _safe_score(value)
    return 0.0


def _empty_retrieval_stage(stage_name: str, source: str) -> Dict[str, Any]:
    return {
        "name": stage_name,
        "source": source,
        "available": False,
        "candidate_count": 0,
        "contexts_raw": [],
        "contexts_topk": [],
        "pred_sp_topk": [],
        "top_k_raw": 0,
        "top_k_final": 0,
        "duplicate_rate": 0.0,
    }


def _build_stage_context_from_candidates(
    *,
    stage_name: str,
    source: str,
    candidates: List[Dict[str, Any]],
    score_keys: List[str],
    top_k: int,
    note_store: NoteStore,
    doc_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not candidates:
        return _empty_retrieval_stage(stage_name, source)

    evidences: List[Dict[str, Any]] = []
    for candidate in candidates:
        note_id = candidate.get("note_id")
        if not isinstance(note_id, str) or not note_id.strip():
            continue
        note = note_store.get_weak(note_id) or {}
        meta = (note.get("meta") or {}) if isinstance(note, dict) else {}
        evidence_text = (
            str(note.get("evidence") or "").strip()
            or str(meta.get("evidence_canonical") or "").strip()
            or str(note.get("obj") or "").strip()
        )
        source_hint = source
        candidate_sources = candidate.get("sources")
        if isinstance(candidate_sources, dict) and candidate_sources:
            source_hint = "+".join(sorted(str(key) for key in candidate_sources.keys()))
        evidences.append(
            {
                "note_id": note_id,
                "source": source_hint,
                "evidence": evidence_text,
                "canonical": evidence_text,
                "weak": bool(meta.get("weak", False)),
                "score": _score_from_stage_candidate(candidate, score_keys),
            }
        )

    contexts_raw = _build_retrieved_context(evidences, note_store, doc_index)
    contexts_topk, dedup_stats = _dedup_retrieved_context(
        contexts_raw,
        top_k=top_k,
        title_diversity_enabled=False,
        query_title_promotion_enabled=False,
    )
    pred_sp_topk = [row["fact"] for row in _collect_pred_sp_candidates(contexts_topk)]

    return {
        "name": stage_name,
        "source": source,
        "available": True,
        "candidate_count": len(candidates),
        "contexts_raw": contexts_raw,
        "contexts_topk": contexts_topk,
        "pred_sp_topk": pred_sp_topk,
        "top_k_raw": dedup_stats.get("top_k_raw", 0),
        "top_k_final": dedup_stats.get("top_k_final", 0),
        "duplicate_rate": dedup_stats.get("duplicate_rate", 0.0),
    }


def _build_retrieval_stage_contexts(
    *,
    retrieve_result: Dict[str, Any],
    note_store: NoteStore,
    doc_index: Dict[str, Dict[str, Any]],
    top_k: int,
) -> Dict[str, Dict[str, Any]]:
    stages = {
        "stage1": _empty_retrieval_stage("stage1", "hybrid.pre_candidates"),
        "stage2_no_fallback": _empty_retrieval_stage("stage2_no_fallback", "hybrid.final"),
    }
    hybrid = retrieve_result.get("hybrid")
    if not isinstance(hybrid, dict):
        return stages

    stage1_candidates = [
        row
        for row in (hybrid.get("pre_candidates") or [])
        if isinstance(row, dict)
    ]
    stage2_candidates = [
        row
        for row in (hybrid.get("final") or [])
        if isinstance(row, dict)
    ]

    stages["stage1"] = _build_stage_context_from_candidates(
        stage_name="stage1",
        source="hybrid.pre_candidates",
        candidates=stage1_candidates,
        score_keys=["score"],
        top_k=top_k,
        note_store=note_store,
        doc_index=doc_index,
    )
    stages["stage2_no_fallback"] = _build_stage_context_from_candidates(
        stage_name="stage2_no_fallback",
        source="hybrid.final",
        candidates=stage2_candidates,
        score_keys=["final_score", "pre_rrf", "llm_score", "struct_score"],
        top_k=top_k,
        note_store=note_store,
        doc_index=doc_index,
    )
    return stages


def _collect_pred_sp_candidates(retrieved_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for order, ctx in enumerate(retrieved_context):
        title = ctx.get("title")
        idx = _coerce_sentence_idx(ctx.get("sentence_idx"))
        if title is None or idx is None:
            continue
        title_str = str(title)
        key = (title_str, int(idx))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "fact": [title_str, int(idx)],
                "title": title_str,
                "score": _safe_score(ctx.get("score")),
                "weak": bool(ctx.get("weak", False)),
                "order": order,
            }
        )
    return rows


def _build_pred_sp(
    retrieved_context: List[Dict[str, Any]],
    *,
    policy: str,
    max_facts: int,
    min_score: float,
    drop_weak: bool,
    prefer_new_titles: bool,
) -> Tuple[List[List[Any]], List[List[Any]], Dict[str, Any]]:
    normalized_policy = _normalize_pred_sp_policy(policy)
    candidates = _collect_pred_sp_candidates(retrieved_context)
    pred_sp_topk = [row["fact"] for row in candidates]
    cap = max(0, int(max_facts))
    threshold = max(0.0, float(min_score))

    meta: Dict[str, Any] = {
        "policy": normalized_policy,
        "max_facts": cap,
        "min_score": threshold,
        "drop_weak": bool(drop_weak),
        "prefer_new_titles": bool(prefer_new_titles),
        "candidate_count": len(candidates),
        "dropped_weak": 0,
        "dropped_score": 0,
        "fallback_to_topk": False,
    }
    if normalized_policy == "topk":
        selected = pred_sp_topk[:cap] if cap > 0 else pred_sp_topk
        meta["selected_count"] = len(selected)
        return selected, pred_sp_topk, meta

    filtered: List[Dict[str, Any]] = []
    for row in candidates:
        if drop_weak and row["weak"]:
            meta["dropped_weak"] += 1
            continue
        if row["score"] < threshold:
            meta["dropped_score"] += 1
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: (-row["score"], row["order"]))

    selected_rows: List[Dict[str, Any]] = []
    selected_titles = set()
    if prefer_new_titles:
        for row in filtered:
            title_key = row["title"].strip().lower()
            if title_key in selected_titles:
                continue
            selected_rows.append(row)
            selected_titles.add(title_key)
            if cap > 0 and len(selected_rows) >= cap:
                break
    if cap <= 0 or len(selected_rows) < cap:
        selected_keys = {
            (row["fact"][0], row["fact"][1])
            for row in selected_rows
        }
        for row in filtered:
            fact_key = (row["fact"][0], row["fact"][1])
            if fact_key in selected_keys:
                continue
            selected_rows.append(row)
            selected_keys.add(fact_key)
            if cap > 0 and len(selected_rows) >= cap:
                break

    selected = [row["fact"] for row in selected_rows]
    if not selected and pred_sp_topk:
        selected = pred_sp_topk[:cap] if cap > 0 else pred_sp_topk
        meta["fallback_to_topk"] = True
    meta["selected_count"] = len(selected)
    return selected, pred_sp_topk, meta


def _tokenize_for_refill(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [token for token in tokens if token and token not in _REFILL_STOPWORDS]


def _seed_texts_for_refill(question: str, retrieved_context_topk: List[Dict[str, Any]]) -> List[str]:
    seeds: List[str] = []
    seen = set()
    for token in _tokenize_for_refill(question):
        if token in seen:
            continue
        seeds.append(token)
        seen.add(token)
    for ctx in retrieved_context_topk:
        title = str(ctx.get("title") or "").strip()
        if title and title.lower() not in seen:
            seeds.append(title)
            seen.add(title.lower())
        if len(seeds) >= 16:
            break
    return seeds


def _score_sentence_for_refill(
    sentence: str,
    *,
    q_tokens: set[str],
    seed_tokens: set[str],
    title_tokens: set[str],
) -> float:
    sent_tokens = set(_tokenize_for_refill(sentence))
    if not sent_tokens:
        return 0.0
    overlap = len(sent_tokens & q_tokens) / max(1, len(q_tokens))
    seed_overlap = len(sent_tokens & seed_tokens) / max(1, len(seed_tokens)) if seed_tokens else 0.0
    title_overlap = len(sent_tokens & title_tokens) / max(1, len(title_tokens)) if title_tokens else 0.0
    digit_bonus = 0.1 if any(tok.isdigit() for tok in sent_tokens) and any(tok.isdigit() for tok in q_tokens) else 0.0
    return overlap + (0.35 * seed_overlap) + (0.15 * title_overlap) + digit_bonus


def _collect_sentence_refill_candidates(
    *,
    question: str,
    doc_index: Dict[str, Dict[str, Any]],
    max_candidates: int,
    min_score: float,
    seed_texts: List[str],
) -> List[Dict[str, Any]]:
    q_tokens = set(_tokenize_for_refill(question))
    seed_tokens = set()
    for seed in seed_texts:
        seed_tokens.update(_tokenize_for_refill(seed))
    candidates: List[Dict[str, Any]] = []
    for doc_id, meta in doc_index.items():
        title = str(meta.get("title") or doc_id)
        title_tokens = set(_tokenize_for_refill(title))
        for sent_idx, sentence in enumerate(meta.get("sentences") or []):
            text = str(sentence).strip()
            if not text:
                continue
            score = _score_sentence_for_refill(
                text,
                q_tokens=q_tokens,
                seed_tokens=seed_tokens,
                title_tokens=title_tokens,
            )
            if score < min_score:
                continue
            note_id = f"{doc_id}#s{int(sent_idx):04d}"
            candidates.append(
                {
                    "note_id": note_id,
                    "doc_id": doc_id,
                    "chunk_id": _make_chunk_id(doc_id, sent_idx, note_id),
                    "title": title,
                    "sentence_idx": int(sent_idx),
                    "text": text,
                    "evidence": text,
                    "canonical": text,
                    "weak": False,
                    "score": round(score, 4),
                    "source": "shortage_refill_sentence_pool",
                }
            )
    candidates.sort(
        key=lambda row: (
            -_safe_score(row.get("score")),
            str(row.get("doc_id") or ""),
            int(row.get("sentence_idx") or 0),
        )
    )
    if max_candidates <= 0:
        return candidates
    return candidates[: int(max_candidates)]


def _collect_chunk_refill_candidates(
    *,
    question: str,
    notes_path: Path,
    doc_index: Dict[str, Dict[str, Any]],
    max_candidates: int,
    min_score: float,
    seed_texts: List[str],
) -> List[Dict[str, Any]]:
    chunks_path = notes_path.parent / "chunks.jsonl"
    if not chunks_path.exists() or max_candidates <= 0:
        return []
    chunk_store = ChunkStore(str(chunks_path))
    hits = chunk_store.search(question, seeds=seed_texts, top_k=max(1, int(max_candidates)))
    candidates: List[Dict[str, Any]] = []
    for hit in hits:
        score = _safe_score(hit.get("score"))
        if score < min_score:
            continue
        doc_id = str(hit.get("doc_id") or "").strip()
        doc_meta = doc_index.get(doc_id) if doc_id else None
        title = str((doc_meta or {}).get("title") or doc_id or "")
        sentences = (doc_meta or {}).get("sentences") or []
        text = str(hit.get("evidence") or hit.get("canonical") or "").strip()
        if not text:
            continue
        sentence_idx = _coerce_sentence_idx(hit.get("sentence_idx"))
        if sentence_idx is None and sentences:
            sentence_idx = _find_sentence_index(text, sentences)
        if sentence_idx is not None and doc_id:
            note_id = f"{doc_id}#s{int(sentence_idx):04d}"
        else:
            note_id = str(hit.get("note_id") or "")
        chunk_id = _make_chunk_id(doc_id or None, sentence_idx, note_id or None)
        if not chunk_id:
            chunk_id = str(hit.get("chunk_id") or note_id or "")
        candidates.append(
            {
                "note_id": note_id or None,
                "doc_id": doc_id or None,
                "chunk_id": chunk_id or None,
                "title": title or None,
                "sentence_idx": sentence_idx,
                "text": text,
                "evidence": text,
                "canonical": text,
                "weak": False,
                "score": round(score, 4),
                "source": "shortage_refill_chunk_store",
            }
        )
    candidates.sort(
        key=lambda row: (
            -_safe_score(row.get("score")),
            str(row.get("doc_id") or ""),
            int(row.get("sentence_idx") or 0),
        )
    )
    return candidates


def _select_refill_candidates(
    *,
    existing_context: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    need: int,
    prefer_new_titles: bool,
) -> List[Dict[str, Any]]:
    if need <= 0:
        return []
    existing_keys = set()
    existing_titles = set()
    for idx, ctx in enumerate(existing_context):
        key, _ = _dedup_key(ctx, idx)
        existing_keys.add(key)
        title = str(ctx.get("title") or "").strip().lower()
        if title:
            existing_titles.add(title)

    ordered = sorted(
        candidates,
        key=lambda row: (
            -_safe_score(row.get("score")),
            0 if row.get("source") == "shortage_refill_chunk_store" else 1,
            str(row.get("doc_id") or ""),
            int(row.get("sentence_idx") or 0),
        ),
    )
    new_title_rows: List[Dict[str, Any]] = []
    other_rows: List[Dict[str, Any]] = []
    selected_keys = set()
    for idx, row in enumerate(ordered):
        key, _ = _dedup_key(row, idx)
        if key in existing_keys or key in selected_keys:
            continue
        selected_keys.add(key)
        title = str(row.get("title") or "").strip().lower()
        if prefer_new_titles and title and title not in existing_titles:
            new_title_rows.append(row)
        else:
            other_rows.append(row)
    merged = new_title_rows + other_rows if prefer_new_titles else other_rows + new_title_rows
    return merged[: int(need)]


def _ctx_to_evidence(ctx: Dict[str, Any]) -> Dict[str, Any]:
    text = str(ctx.get("text") or ctx.get("evidence") or ctx.get("canonical") or "")
    note_id = ctx.get("note_id")
    doc_id = ctx.get("doc_id")
    return {
        "note_id": note_id,
        "doc_id": doc_id,
        "source": doc_id,
        "evidence": text,
        "canonical": text,
        "weak": bool(ctx.get("weak", False)),
        "score": _safe_score(ctx.get("score")),
        "subj": ctx.get("title") or doc_id,
        "pred": "sentence",
        "obj": text,
    }


def _normalize_title_key(value: Any) -> str:
    return normalize_title(value).strip().lower()


def _promote_title_diversity(
    rows: List[Dict[str, Any]],
    *,
    top_n: int,
    keep_first: bool,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Promote unique titles in early ranks to improve low-k document recall."""
    cap = max(0, int(top_n))
    if cap <= 1 or len(rows) <= 1:
        return rows, False
    cap = min(cap, len(rows))

    head_size = 1 if keep_first else 0
    head_size = min(head_size, len(rows))
    promoted: List[Dict[str, Any]] = list(rows[:head_size])
    delayed: List[Dict[str, Any]] = []
    seen_titles = set()
    for item in promoted:
        title_key = _normalize_title_key(item.get("title") or item.get("doc_title"))
        if title_key:
            seen_titles.add(title_key)

    for item in rows[head_size:]:
        title_key = _normalize_title_key(item.get("title") or item.get("doc_title"))
        if not title_key or title_key in seen_titles or len(promoted) >= cap:
            delayed.append(item)
            continue
        promoted.append(item)
        seen_titles.add(title_key)

    reordered = promoted + delayed
    return reordered, reordered != rows


def _promote_second_title_by_question(
    rows: List[Dict[str, Any]],
    *,
    question: str,
    window_n: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Promote one title that best matches question tokens into rank-2."""
    cap = max(0, int(window_n))
    if cap < 2 or len(rows) < 2:
        return rows, False
    cap = min(cap, len(rows))

    q_tokens = set(_tokenize_for_refill(question))
    question_text = " ".join(str(question or "").strip().lower().split())
    rank1_title = _normalize_title_key(rows[0].get("title") or rows[0].get("doc_title"))

    best_idx = 1
    best_score = (-1, -1, -1)
    for idx in range(1, len(rows)):
        row = rows[idx]
        title_raw = row.get("title") or row.get("doc_title")
        title_key = _normalize_title_key(title_raw)
        if not title_key or title_key == rank1_title:
            continue
        title_tokens = set(_tokenize_for_refill(title_key))
        overlap = len(title_tokens & q_tokens) if q_tokens else 0
        exact_phrase = 1 if title_key and title_key in question_text else 0
        in_window_bonus = 1 if idx < cap else 0
        cand = (exact_phrase, overlap, in_window_bonus)
        if cand > best_score:
            best_score = cand
            best_idx = idx

    if best_idx == 1:
        return rows, False
    if best_score[0] <= 0 and best_score[1] <= 0:
        return rows, False

    reordered = list(rows)
    promoted = reordered.pop(best_idx)
    reordered.insert(1, promoted)
    return reordered, True


def _apply_shortage_refill(
    *,
    question: str,
    top_k: int,
    requested: int,
    retrieve_result: Dict[str, Any],
    retrieved_context_raw: List[Dict[str, Any]],
    retrieved_context_topk: List[Dict[str, Any]],
    dedup_stats: Dict[str, Any],
    note_store: NoteStore,
    notes_path: Path,
    doc_index: Dict[str, Dict[str, Any]],
    max_candidates: int,
    min_score: float,
    prefer_new_titles: bool,
    title_diversity_enabled: bool,
    title_diversity_top_n: int,
    title_diversity_keep_first: bool,
    query_title_promotion_enabled: bool,
    query_title_promotion_window: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": True,
        "triggered": False,
        "needed": 0,
        "added": 0,
        "before_top_k_raw": len(retrieved_context_raw),
        "before_top_k_final": len(retrieved_context_topk),
        "after_top_k_raw": len(retrieved_context_raw),
        "after_top_k_final": len(retrieved_context_topk),
        "chunk_candidates": 0,
        "sentence_candidates": 0,
        "selected_from_chunk_store": 0,
        "selected_from_sentence_pool": 0,
        "max_candidates": int(max_candidates),
        "min_score": float(min_score),
        "prefer_new_titles": bool(prefer_new_titles),
    }
    if top_k <= 0:
        return retrieve_result, retrieved_context_raw, retrieved_context_topk, dedup_stats, info
    needed = max(0, int(top_k) - len(retrieved_context_topk))
    info["needed"] = needed
    if needed <= 0:
        return retrieve_result, retrieved_context_raw, retrieved_context_topk, dedup_stats, info

    info["triggered"] = True
    limit = max(needed, int(max_candidates))
    seed_texts = _seed_texts_for_refill(question, retrieved_context_topk)
    chunk_candidates = _collect_chunk_refill_candidates(
        question=question,
        notes_path=notes_path,
        doc_index=doc_index,
        max_candidates=limit,
        min_score=float(min_score),
        seed_texts=seed_texts,
    )
    sentence_candidates = _collect_sentence_refill_candidates(
        question=question,
        doc_index=doc_index,
        max_candidates=limit,
        min_score=float(min_score),
        seed_texts=seed_texts,
    )
    info["chunk_candidates"] = len(chunk_candidates)
    info["sentence_candidates"] = len(sentence_candidates)
    selected = _select_refill_candidates(
        existing_context=retrieved_context_raw,
        candidates=chunk_candidates + sentence_candidates,
        need=needed,
        prefer_new_titles=bool(prefer_new_titles),
    )
    if not selected:
        return retrieve_result, retrieved_context_raw, retrieved_context_topk, dedup_stats, info

    refill_evidences = [_ctx_to_evidence(row) for row in selected]
    existing_evidences = list(retrieve_result.get("evidence") or [])
    merged_evidences = existing_evidences + refill_evidences
    merged_result = dict(retrieve_result)
    merged_result["evidence"] = merged_evidences
    merged_result["shortage_refill"] = {
        "added": len(refill_evidences),
        "chunk_added": sum(1 for row in selected if row.get("source") == "shortage_refill_chunk_store"),
        "sentence_added": sum(1 for row in selected if row.get("source") == "shortage_refill_sentence_pool"),
    }

    new_raw = _build_retrieved_context(merged_evidences, note_store, doc_index)
    new_topk, new_dedup = _dedup_retrieved_context(
        new_raw,
        top_k=top_k,
        title_diversity_enabled=title_diversity_enabled,
        title_diversity_top_n=title_diversity_top_n,
        title_diversity_keep_first=title_diversity_keep_first,
        question=question,
        query_title_promotion_enabled=query_title_promotion_enabled,
        query_title_promotion_window=query_title_promotion_window,
    )
    new_dedup["top_k_raw_requested"] = requested

    info["added"] = len(refill_evidences)
    info["after_top_k_raw"] = len(new_raw)
    info["after_top_k_final"] = len(new_topk)
    info["selected_from_chunk_store"] = sum(
        1 for row in selected if row.get("source") == "shortage_refill_chunk_store"
    )
    info["selected_from_sentence_pool"] = sum(
        1 for row in selected if row.get("source") == "shortage_refill_sentence_pool"
    )
    return merged_result, new_raw, new_topk, new_dedup, info


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
    title_diversity_enabled: bool = DEFAULT_TITLE_DIVERSITY_ENABLED,
    title_diversity_top_n: int = DEFAULT_TITLE_DIVERSITY_TOP_N,
    title_diversity_keep_first: bool = DEFAULT_TITLE_DIVERSITY_KEEP_FIRST,
    question: str = "",
    query_title_promotion_enabled: bool = DEFAULT_QUERY_TITLE_PROMOTION_ENABLED,
    query_title_promotion_window: int = DEFAULT_QUERY_TITLE_PROMOTION_WINDOW,
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
    title_diversity_applied = False
    if bool(title_diversity_enabled):
        deduped, title_diversity_applied = _promote_title_diversity(
            deduped,
            top_n=max(0, int(title_diversity_top_n)),
            keep_first=bool(title_diversity_keep_first),
        )
    query_title_promotion_applied = False
    if bool(query_title_promotion_enabled):
        deduped, query_title_promotion_applied = _promote_second_title_by_question(
            deduped,
            question=str(question or ""),
            window_n=max(0, int(query_title_promotion_window)),
        )
    unique_count = len(deduped)
    if top_k > 0:
        deduped = deduped[:top_k]
    duplicate_rate = duplicates / raw_count if raw_count else 0.0
    return deduped, {
        "top_k_raw": raw_count,
        "top_k_final": len(deduped),
        "unique_count": unique_count,
        "duplicate_rate": duplicate_rate,
        "title_diversity_enabled": bool(title_diversity_enabled),
        "title_diversity_applied": bool(title_diversity_applied),
        "title_diversity_top_n": max(0, int(title_diversity_top_n)),
        "title_diversity_keep_first": bool(title_diversity_keep_first),
        "query_title_promotion_enabled": bool(query_title_promotion_enabled),
        "query_title_promotion_applied": bool(query_title_promotion_applied),
        "query_title_promotion_window": max(0, int(query_title_promotion_window)),
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


def _collect_debug_notes(
    note_store: NoteStore,
    note_ids: Iterable[str],
    max_notes: int,
) -> Dict[str, Any]:
    collected: Dict[str, Any] = {}
    limit = max(0, int(max_notes))
    for note_id in note_ids:
        if not note_id or note_id in collected:
            continue
        note = note_store.get_weak(note_id)
        if note:
            collected[note_id] = note
        if limit and len(collected) >= limit:
            break
    return collected


def _write_debug_artifacts(
    record: Dict[str, Any],
    *,
    debug_dir: Path,
    docs_dir: Path,
    notes_path: Path,
    index_dir: Path,
    doc_index: Dict[str, Dict[str, Any]],
    note_store: NoteStore,
    max_notes: int,
) -> Optional[Path]:
    qid = str(record.get("_id") or "unknown")
    debug_dir.mkdir(parents=True, exist_ok=True)
    retrieve_result = (record.get("intermediate") or {}).get("retrieve_result") or {}
    support_note_ids = retrieve_result.get("support_note_ids") or []
    evidence_note_ids = [
        ev.get("note_id")
        for ev in (retrieve_result.get("evidence") or [])
        if ev.get("note_id")
    ]
    merged_note_ids = list(dict.fromkeys(list(support_note_ids) + list(evidence_note_ids)))
    notes = _collect_debug_notes(note_store, merged_note_ids, max_notes=max_notes)

    payload = {
        "qid": qid,
        "question": record.get("question"),
        "paths": {
            "docs_dir": str(docs_dir),
            "notes_path": str(notes_path),
            "indexes_dir": str(index_dir),
        },
        "doc_index": doc_index,
        "note_ids": {
            "support": support_note_ids,
            "evidence": evidence_note_ids,
            "saved": list(notes.keys()),
        },
        "notes": notes,
        "record": record,
    }
    debug_path = debug_dir / f"{qid}.json"
    debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return debug_path


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


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = deepcopy(cfg)
    openai_cfg = sanitized.get("openai")
    if isinstance(openai_cfg, dict) and openai_cfg.get("api_key"):
        openai_cfg["api_key"] = "***"
    vllm_cfg = sanitized.get("vllm")
    if isinstance(vllm_cfg, dict) and vllm_cfg.get("api_key"):
        vllm_cfg["api_key"] = "***"
    retriever_cfg = sanitized.get("retriever")
    if isinstance(retriever_cfg, dict):
        embedding_cfg = retriever_cfg.get("embedding")
        if isinstance(embedding_cfg, dict) and embedding_cfg.get("api_key"):
            embedding_cfg["api_key"] = "***"
    reranker_cfg = sanitized.get("reranker")
    if isinstance(reranker_cfg, dict):
        rerank_openai = reranker_cfg.get("openai")
        if isinstance(rerank_openai, dict) and rerank_openai.get("api_key"):
            rerank_openai["api_key"] = "***"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _resolve_title_diversity_policy(base_cfg: Dict[str, Any]) -> Tuple[bool, int, bool]:
    entry_cfg = base_cfg.get("hotpot_entry") or {}
    if not isinstance(entry_cfg, dict):
        entry_cfg = {}
    retriever_cfg = base_cfg.get("retriever") or {}
    if not isinstance(retriever_cfg, dict):
        retriever_cfg = {}
    policy_cfg = retriever_cfg.get("title_diversity") or {}
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}

    enabled = _coerce_bool(
        policy_cfg.get("enabled", entry_cfg.get("title_diversity_enabled", DEFAULT_TITLE_DIVERSITY_ENABLED)),
        DEFAULT_TITLE_DIVERSITY_ENABLED,
    )
    top_n = _coerce_int(
        policy_cfg.get("top_n", entry_cfg.get("title_diversity_top_n", DEFAULT_TITLE_DIVERSITY_TOP_N)),
        DEFAULT_TITLE_DIVERSITY_TOP_N,
    )
    keep_first = _coerce_bool(
        policy_cfg.get("keep_first", entry_cfg.get("title_diversity_keep_first", DEFAULT_TITLE_DIVERSITY_KEEP_FIRST)),
        DEFAULT_TITLE_DIVERSITY_KEEP_FIRST,
    )
    return bool(enabled), max(0, int(top_n)), bool(keep_first)


def _resolve_query_title_promotion_policy(base_cfg: Dict[str, Any]) -> Tuple[bool, int]:
    entry_cfg = base_cfg.get("hotpot_entry") or {}
    if not isinstance(entry_cfg, dict):
        entry_cfg = {}
    retriever_cfg = base_cfg.get("retriever") or {}
    if not isinstance(retriever_cfg, dict):
        retriever_cfg = {}
    policy_cfg = retriever_cfg.get("query_title_promotion") or {}
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}

    enabled = _coerce_bool(
        policy_cfg.get("enabled", entry_cfg.get("query_title_promotion_enabled", DEFAULT_QUERY_TITLE_PROMOTION_ENABLED)),
        DEFAULT_QUERY_TITLE_PROMOTION_ENABLED,
    )
    window = _coerce_int(
        policy_cfg.get("window_n", entry_cfg.get("query_title_promotion_window", DEFAULT_QUERY_TITLE_PROMOTION_WINDOW)),
        DEFAULT_QUERY_TITLE_PROMOTION_WINDOW,
    )
    return bool(enabled), max(0, int(window))


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _parse_modes(value: Any) -> List[str]:
    allowed = {"structured", "dense", "bm25", "hybrid"}
    if not value:
        return ["bm25", "dense", "hybrid"]
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
    return normalized or ["bm25", "dense", "hybrid"]


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
        if isinstance(args.reader, list):
            readers = [resolve_reader(str(r), dataset_cfg) for r in args.reader]
        else:
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


def _apply_dataset_retriever(cfg: Dict[str, Any], dataset_key: str) -> Dict[str, Any]:
    dataset_cfg = get_dataset_config(cfg, dataset_key)
    retriever_override = dataset_cfg.get("retriever") if isinstance(dataset_cfg, dict) else None
    if isinstance(retriever_override, dict):
        base_retriever = cfg.get("retriever") or {}
        cfg["retriever"] = _deep_merge(base_retriever, retriever_override)
    return cfg


def _apply_openai_reranker(base_cfg: Dict[str, Any], openai_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not openai_cfg:
        return base_cfg
    cfg = deepcopy(base_cfg)
    reranker = cfg.setdefault("reranker", {})
    reranker["provider"] = "openai"
    reranker["openai"] = {
        "model": openai_cfg.get("model"),
        "api_key": openai_cfg.get("api_key"),
        "api_key_env": openai_cfg.get("api_key_env"),
        "base_url": openai_cfg.get("base_url"),
        "temperature": openai_cfg.get("temperature"),
        "timeout_sec": openai_cfg.get("timeout_sec"),
        "max_retries": openai_cfg.get("max_retries"),
        "retry_backoff_sec": openai_cfg.get("retry_backoff_sec"),
        "retry_backoff_max_sec": openai_cfg.get("retry_backoff_max_sec"),
    }
    return cfg


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


def _ensure_index(
    doc_index: Dict[str, Dict[str, Any]],
    index_root: Path,
    force_build: bool,
    chunker: Optional[Chunker] = None,
) -> Dict[str, Any]:
    notes_path = index_root / "notes.jsonl"
    chunks_path = index_root / "chunks.jsonl"
    indexes_dir = index_root / "indexes"
    lock_path = index_root / "build.lock"
    
    with FileLock(str(lock_path)):
        notes_ready = notes_path.exists()
        indexes_ready = indexes_dir.exists()
        chunks_ready = chunks_path.exists()
        if not force_build and notes_ready and indexes_ready and chunks_ready:
            return {"status": "reused"}
        if not force_build and notes_ready and indexes_ready and not chunks_ready:
            chunks_written = _write_chunks_for_example(doc_index, chunks_path, chunker=chunker, overwrite=True)
            return {"status": "reused_chunks", "chunks": chunks_written}
        
        index_root.mkdir(parents=True, exist_ok=True)
        notes_written = 0
        chunks_written = 0
        if force_build or not notes_path.exists():
            notes_written = _write_sentence_notes_for_example(doc_index, notes_path, overwrite=True)
        if force_build or not chunks_path.exists():
            chunks_written = _write_chunks_for_example(doc_index, chunks_path, chunker=chunker, overwrite=True)
        builder = IndexBuilder()
        builder.build_from_jsonl(str(notes_path))
        builder.dump(str(indexes_dir))
        return {"status": "ok", "notes": notes_written, "chunks": chunks_written}


def _prepare_aux_config(example_root: Path, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    notes_path = example_root / "notes.jsonl"
    indexes_root = example_root / "indexes"
    cfg.setdefault("notes", {})["out_path"] = str(notes_path)
    retriever_cfg = cfg.setdefault("retriever", {})
    embed_cfg = retriever_cfg.setdefault("embedding", {})
    bm25_cfg = retriever_cfg.setdefault("bm25", {})
    embed_cfg["offline_index_path"] = str(indexes_root / "faiss" / "notes.faiss")
    embed_cfg["meta_path"] = str(indexes_root / "faiss" / "notes.meta.parquet")
    bm25_cfg["store_path"] = str(indexes_root / "bm25" / "notes")
    return cfg


def _build_aux_indexes(
    example_root: Path,
    *,
    base_cfg: Dict[str, Any],
    build_embedding: bool,
    build_bm25: bool,
    force_build: bool,
) -> Dict[str, Any]:
    cfg = _prepare_aux_config(example_root, base_cfg)
    notes_path = Path(cfg.get("notes", {}).get("out_path", example_root / "notes.jsonl"))
    stats: Dict[str, Any] = {}
    if not notes_path.exists():
        if build_embedding:
            stats["embedding"] = "skipped_missing_notes"
        if build_bm25:
            stats["bm25"] = "skipped_missing_notes"
        return stats

    retriever_cfg = cfg.get("retriever") or {}
    embed_cfg = retriever_cfg.get("embedding") or {}
    bm25_cfg = retriever_cfg.get("bm25") or {}

    lock_path = example_root / "aux_build.lock"
    with FileLock(str(lock_path)):
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
                    logger.warning("BM25 corpus build failed {}: {}", notes_path, exc)
                    stats["bm25"] = f"error:{exc}"
            else:
                stats["bm25"] = "reused"
        else:
            stats["bm25"] = "disabled"

    return stats


def _prepare_retriever_config(
    example_root: Path,
    base_cfg: Dict[str, Any],
    mode: str,
    *,
    predicate_mode: str = DEFAULT_PREDICATE_MODE,
    predicate_random_seed: int = DEFAULT_PREDICATE_RANDOM_SEED,
    use_alias_binding: bool = True,
    use_alias_lookup: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = _prepare_aux_config(example_root, base_cfg)
    retriever_cfg = cfg.setdefault("retriever", {})
    structured_cfg = retriever_cfg.setdefault("structured", {})
    embed_cfg = retriever_cfg.setdefault("embedding", {})
    bm25_cfg = retriever_cfg.setdefault("bm25", {})
    hybrid_cfg = retriever_cfg.setdefault("hybrid", {})

    mode = mode.lower()
    if mode == "structured":
        structured_cfg["enabled"] = True
        # Keep vector fallback on for structured mode unless caller explicitly disables it.
        structured_cfg["vector_fallback_enabled"] = bool(
            structured_cfg.get("vector_fallback_enabled", True)
        )
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

    embed_ready = embed_index.exists() and embed_meta.exists()
    bm25_ready = bm25_corpus.exists()

    if embed_cfg.get("enabled") and not embed_ready:
        embed_cfg["enabled"] = False
    if bm25_cfg.get("enabled") and not bm25_ready:
        bm25_cfg["enabled"] = False
    hybrid_cfg["require_seed_match"] = False
    normalized_predicate_mode = str(predicate_mode or DEFAULT_PREDICATE_MODE).strip().lower()
    if normalized_predicate_mode not in {"on", "off", "random"}:
        normalized_predicate_mode = DEFAULT_PREDICATE_MODE
    structured_cfg["predicate_mode"] = normalized_predicate_mode
    structured_cfg["predicate_constraint_enabled"] = normalized_predicate_mode != "off"
    structured_cfg["random_predicate_enabled"] = normalized_predicate_mode == "random"
    structured_cfg["random_predicate_seed"] = int(predicate_random_seed)
    structured_cfg["use_alias_binding"] = use_alias_binding
    structured_cfg["use_alias_lookup"] = use_alias_lookup

    return cfg, {
        "mode": mode,
        "embedding_ready": embed_ready,
        "bm25_ready": bm25_ready,
        "embedding_index": str(embed_index),
        "embedding_meta": str(embed_meta),
        "bm25_corpus": str(bm25_corpus),
        "predicate_mode": normalized_predicate_mode,
        "predicate_constraint_enabled": normalized_predicate_mode != "off",
        "random_predicate_seed": int(predicate_random_seed),
        "use_alias_binding": use_alias_binding,
        "use_alias_lookup": use_alias_lookup,
    }


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
    index_dir: Path,
    notes_path: Path,
    doc_index: Dict[str, Dict[str, Any]],
    note_store: NoteStore,
    base_cfg: Dict[str, Any],
    mode: str,
    top_k: int,
    top_k_raw: int,
    backfill_max_overfetch: float,
    backfill_step: float,
    backfill_rounds: int,
    shortage_refill_enabled: bool,
    shortage_refill_max_candidates: int,
    shortage_refill_min_score: float,
    shortage_refill_prefer_new_titles: bool,
    predicate_mode: str,
    predicate_random_seed: int,
    use_alias_binding: bool = True,
    use_alias_lookup: bool = True,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Any],
    Optional[str],
    int,
    Dict[str, Any],
]:
    requested = max(1, int(top_k_raw))
    title_diversity_enabled, title_diversity_top_n, title_diversity_keep_first = _resolve_title_diversity_policy(
        base_cfg
    )
    query_title_promotion_enabled, query_title_promotion_window = _resolve_query_title_promotion_policy(base_cfg)
    max_raw = requested
    if top_k > 0:
        max_raw = max(requested, int(math.ceil(top_k * float(backfill_max_overfetch))))

    attempt = 0
    last_raw_count = -1
    backfill_reason = None
    retriever_paths: Dict[str, Any] = {}
    retrieve_result: Dict[str, Any] = {}
    retrieved_context_raw: List[Dict[str, Any]] = []
    retrieved_context_topk: List[Dict[str, Any]] = []
    dedup_stats: Dict[str, Any] = {}
    shortage_refill_info: Dict[str, Any] = {
        "enabled": bool(shortage_refill_enabled),
        "triggered": False,
        "added": 0,
    }

    while True:
        retriever_cfg, retriever_paths = _prepare_retriever_config(
            index_dir.parent,
            base_cfg,
            mode,
            predicate_mode=predicate_mode,
            predicate_random_seed=predicate_random_seed,
            use_alias_binding=use_alias_binding,
            use_alias_lookup=use_alias_lookup,
        )
        scheduler_cfg = retriever_cfg.setdefault("retriever", {}).setdefault("scheduler", {})
        scheduler_cfg["keep_at_least"] = top_k
        scheduler_cfg["min_confidence"] = 0.0
        scheduler_cfg["dedup_subject"] = False
        retriever_cfg.setdefault("retriever", {}).setdefault("chunk_fallback", {})["top_k"] = requested

        retrieve_result = retrieve(
            question=question,
            index_dir=str(index_dir),
            notes_path=str(notes_path),
            top_k=requested,
            cfg=retriever_cfg,
        )
        evidences = retrieve_result.get("evidence") or []
        retrieved_context_raw = _build_retrieved_context(evidences, note_store, doc_index)
        retrieved_context_topk, dedup_stats = _dedup_retrieved_context(
            retrieved_context_raw,
            top_k=top_k,
            title_diversity_enabled=title_diversity_enabled,
            title_diversity_top_n=title_diversity_top_n,
            title_diversity_keep_first=title_diversity_keep_first,
            question=question,
            query_title_promotion_enabled=query_title_promotion_enabled,
            query_title_promotion_window=query_title_promotion_window,
        )
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

    if (
        bool(shortage_refill_enabled)
        and top_k > 0
        and len(retrieved_context_topk) < top_k
    ):
        (
            retrieve_result,
            retrieved_context_raw,
            retrieved_context_topk,
            dedup_stats,
            shortage_refill_info,
        ) = _apply_shortage_refill(
            question=question,
            top_k=top_k,
            requested=requested,
            retrieve_result=retrieve_result,
            retrieved_context_raw=retrieved_context_raw,
            retrieved_context_topk=retrieved_context_topk,
            dedup_stats=dedup_stats,
            note_store=note_store,
            notes_path=notes_path,
            doc_index=doc_index,
            max_candidates=max(1, int(shortage_refill_max_candidates)),
            min_score=float(shortage_refill_min_score),
            prefer_new_titles=bool(shortage_refill_prefer_new_titles),
            title_diversity_enabled=title_diversity_enabled,
            title_diversity_top_n=title_diversity_top_n,
            title_diversity_keep_first=title_diversity_keep_first,
            query_title_promotion_enabled=query_title_promotion_enabled,
            query_title_promotion_window=query_title_promotion_window,
        )
        if len(retrieved_context_topk) >= top_k:
            backfill_reason = None
        elif not backfill_reason:
            backfill_reason = _classify_topk_shortage(retrieved_context_raw, requested)

    return (
        retrieve_result,
        retrieved_context_raw,
        retrieved_context_topk,
        dedup_stats,
        retriever_paths,
        backfill_reason,
        attempt,
        shortage_refill_info,
    )


def _process_example(
    example: Dict[str, Any],
    cache_root: Path,
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
    shortage_refill_enabled: bool,
    shortage_refill_max_candidates: int,
    shortage_refill_min_score: float,
    shortage_refill_prefer_new_titles: bool,
    predicate_mode: str,
    predicate_random_seed: int,
    use_alias_binding: bool,
    use_alias_lookup: bool,
    pred_sp_policy: str,
    pred_sp_max_facts: int,
    pred_sp_min_score: float,
    pred_sp_drop_weak: bool,
    pred_sp_prefer_new_titles: bool,
    retrieval_only: bool,
    llm_retry_on_empty: int,
    llm_retry_max_evidence: int,
    force_build: bool,
    debug_dir: Optional[Path],
    debug_max_notes: int,
    run_dir: Optional[str] = None,
    chunking_method: str = "sentence",
    chunking_size: int = 256,
    chunking_overlap: int = 32,
) -> Dict[str, Any]:
    qid = str(example.get("_id") or "unknown")
    question = str(example.get("question") or "")
    
    chunker: Optional[Chunker] = None
    if chunking_method == "fixed":
        chunker = FixedWindowChunker(chunk_size=chunking_size, overlap=chunking_overlap)
    else:
        chunker = SentenceAwareChunker()

    example_root = cache_root / qid
    docs_dir = example_root / "docs"
    doc_index = _write_docs_for_example(example, docs_dir, overwrite=force_build)
    build_stats = _ensure_index(doc_index, example_root, force_build, chunker=chunker)
    build_embedding, build_bm25 = _mode_requirements(mode)
    aux_stats = _build_aux_indexes(
        example_root,
        base_cfg=base_cfg,
        build_embedding=build_embedding,
        build_bm25=build_bm25,
        force_build=force_build,
    )
    notes_path = example_root / "notes.jsonl"
    index_dir = example_root / "indexes"
    note_store = NoteStore(str(notes_path))
    (
        retrieve_result,
        retrieved_context_raw,
        retrieved_context_topk,
        dedup_stats,
        retriever_paths,
        top_k_fill_reason,
        backfill_attempts,
        shortage_refill_info,
    ) = _retrieve_with_backfill(
        question=question,
        index_dir=index_dir,
        notes_path=notes_path,
        doc_index=doc_index,
        note_store=note_store,
        base_cfg=base_cfg,
        mode=mode,
        top_k=top_k,
        top_k_raw=top_k_raw,
        backfill_max_overfetch=backfill_max_overfetch,
        backfill_step=backfill_step,
        backfill_rounds=backfill_rounds,
        shortage_refill_enabled=shortage_refill_enabled,
        shortage_refill_max_candidates=shortage_refill_max_candidates,
        shortage_refill_min_score=shortage_refill_min_score,
        shortage_refill_prefer_new_titles=shortage_refill_prefer_new_titles,
        predicate_mode=predicate_mode,
        predicate_random_seed=predicate_random_seed,
        use_alias_binding=use_alias_binding,
        use_alias_lookup=use_alias_lookup,
    )

    evidences = retrieve_result.get("evidence") or []
    structured_answer = retrieve_result.get("answer")
    raw_answer = ""
    prompt_meta: Dict[str, Any] = {}
    llm_error: Optional[str] = None
    llm_error_reason: Optional[str] = None
    if retrieval_only:
        short_answer = str(structured_answer or "").strip()
        answer_source = "retrieval_only"
        answer_source_detail = {
            "source": "retrieval_only",
            "structured_answer_used": bool(short_answer),
        }
    else:
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
            structured_answer,
            raw_answer,
            question=question,
        )
        if answer_source == "empty":
            answer_source = "llm_fallback"
            answer_source_detail["fallback_override"] = "empty"
    fallback_reason = None
    if not retrieval_only:
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
    if (
        not retrieval_only
        and (reader == "vllm" or reader == "openai")
        and llm_retry_on_empty > 0
        and answer_source == "llm_fallback"
    ):
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
                structured_answer,
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

    references: List[str] = []
    raw_reference = example.get("answer")
    if isinstance(raw_reference, list):
        references = [str(item).strip() for item in raw_reference if str(item).strip()]
    elif raw_reference:
        references = [str(raw_reference).strip()]
    metrics = score_metrics(short_answer, references)

    pred_sp, pred_sp_topk, pred_sp_meta = _build_pred_sp(
        retrieved_context_topk,
        policy=pred_sp_policy,
        max_facts=pred_sp_max_facts,
        min_score=pred_sp_min_score,
        drop_weak=pred_sp_drop_weak,
        prefer_new_titles=pred_sp_prefer_new_titles,
    )
    stage_contexts = _build_retrieval_stage_contexts(
        retrieve_result=retrieve_result,
        note_store=note_store,
        doc_index=doc_index,
        top_k=top_k,
    )
    final_stage = {
        "name": "final_with_fallback",
        "source": "retrieved_context_topk",
        "available": True,
        "candidate_count": len(retrieved_context_raw),
        "contexts_raw": retrieved_context_raw,
        "contexts_topk": retrieved_context_topk,
        "pred_sp_topk": pred_sp_topk,
        "top_k_raw": dedup_stats.get("top_k_raw", 0),
        "top_k_final": dedup_stats.get("top_k_final", 0),
        "duplicate_rate": dedup_stats.get("duplicate_rate", 0.0),
        "fallback": retrieve_result.get("fallback"),
        "top_k_fill_reason": top_k_fill_reason,
        "top_k_shortage_refill": shortage_refill_info,
    }
    gold_sp = _extract_gold_sp(example)
    llm_input_hash = prompt_meta.get("llm_input_hash") or ""
    top_k_raw_value = dedup_stats.get("top_k_raw")
    overfetch_factor = None
    if isinstance(top_k_raw_value, (int, float)) and top_k:
        overfetch_factor = float(top_k_raw_value) / float(top_k)
    top_k_raw_source_final = top_k_raw_source
    if backfill_attempts > 0:
        top_k_raw_source_final = "backfill"
    if shortage_refill_info.get("added", 0) > 0:
        if top_k_raw_source_final == "backfill":
            top_k_raw_source_final = "backfill+shortage_refill"
        else:
            top_k_raw_source_final = "shortage_refill"

    output_record = {
        "_id": qid,
        "question": question,
        "answer": short_answer,
        "short_answer": short_answer,
        "answer_source": answer_source,
        "answer_source_detail": answer_source_detail,
        "gold_sp": gold_sp,
        "pred_sp": pred_sp,
        "pred_sp_topk": pred_sp_topk,
        "pred_sp_policy": pred_sp_meta.get("policy"),
        "pred_sp_policy_meta": pred_sp_meta,
        "sp": pred_sp,
        "generated_answer": short_answer,
        "supporting_facts": gold_sp,
        "prediction": short_answer,
        "references": references,
        "metrics": metrics,
        "mode": mode,
        "reader": reader,
        "model": answer_model,
        "retrieval_only": bool(retrieval_only),
        "predicate_mode": str(predicate_mode),
        "retrieved_context_raw": retrieved_context_raw,
        "retrieved_context_topk": retrieved_context_topk,
        "retrieved_context": retrieved_context_topk,
        "retrieval_stages": {
            "stage1": stage_contexts.get("stage1"),
            "stage2_no_fallback": stage_contexts.get("stage2_no_fallback"),
            "final_with_fallback": final_stage,
        },
        "top_k": top_k,
        "top_k_raw": dedup_stats.get("top_k_raw"),
        "top_k_final": dedup_stats.get("top_k_final"),
        "duplicate_rate": dedup_stats.get("duplicate_rate"),
        "title_diversity_applied": bool(dedup_stats.get("title_diversity_applied", False)),
        "title_diversity_top_n": dedup_stats.get("title_diversity_top_n"),
        "query_title_promotion_applied": bool(dedup_stats.get("query_title_promotion_applied", False)),
        "query_title_promotion_window": dedup_stats.get("query_title_promotion_window"),
        "overfetch_factor": overfetch_factor,
        "fallback_reason": fallback_reason,
        "llm_error": llm_error,
        "top_k_fill_reason": top_k_fill_reason,
        "top_k_shortage_refill": shortage_refill_info,
        "llm_input_hash": llm_input_hash,
        "intermediate": {
            "build_stats": build_stats,
            "aux_indexes": aux_stats,
            "retriever_paths": retriever_paths,
            "retrieve_result": retrieve_result,
            "structured_answer": structured_answer,
            "llm_raw": raw_answer,
            "llm_has_final": has_final_tag(raw_answer),
            "support_note_ids": retrieve_result.get("support_note_ids"),
            "paths": retrieve_result.get("paths"),
            "ir": retrieve_result.get("ir"),
            "fallback": retrieve_result.get("fallback"),
            "intent": retrieve_result.get("intent"),
            "retrieval_mode": mode,
            "reader": reader,
            "model": answer_model,
            "retrieval_only": bool(retrieval_only),
            "predicate_mode": str(predicate_mode),
            "predicate_constraint_enabled": str(predicate_mode).strip().lower() != "off",
            "predicate_random_seed": int(predicate_random_seed),
            "predicate_random_mode": "per_state_hash" if str(predicate_mode).strip().lower() == "random" else "none",
            "top_k": top_k,
            "top_k_raw": dedup_stats.get("top_k_raw"),
            "top_k_raw_requested": dedup_stats.get("top_k_raw_requested"),
            "top_k_raw_source": top_k_raw_source_final,
            "top_k_backfill_rounds": backfill_attempts,
            "top_k_fill_reason": top_k_fill_reason,
            "top_k_shortage_refill": shortage_refill_info,
            "title_diversity_enabled": bool(dedup_stats.get("title_diversity_enabled", False)),
            "title_diversity_applied": bool(dedup_stats.get("title_diversity_applied", False)),
            "title_diversity_top_n": dedup_stats.get("title_diversity_top_n"),
            "title_diversity_keep_first": bool(dedup_stats.get("title_diversity_keep_first", True)),
            "query_title_promotion_enabled": bool(dedup_stats.get("query_title_promotion_enabled", False)),
            "query_title_promotion_applied": bool(dedup_stats.get("query_title_promotion_applied", False)),
            "query_title_promotion_window": dedup_stats.get("query_title_promotion_window"),
            "chunk_fallback_top_k": dedup_stats.get("top_k_raw_requested"),
            "chunk_fallback_top_k_source": top_k_raw_source_final,
            "pred_sp_policy": pred_sp_meta.get("policy"),
            "pred_sp_topk_count": len(pred_sp_topk),
            "pred_sp_selected_count": len(pred_sp),
            "pred_sp_policy_meta": pred_sp_meta,
            "stage1_available": bool((stage_contexts.get("stage1") or {}).get("available", False)),
            "stage1_topk_count": int((stage_contexts.get("stage1") or {}).get("top_k_final", 0) or 0),
            "stage2_available": bool((stage_contexts.get("stage2_no_fallback") or {}).get("available", False)),
            "stage2_topk_count": int((stage_contexts.get("stage2_no_fallback") or {}).get("top_k_final", 0) or 0),
            "stage2_fallback_used": bool((retrieve_result.get("fallback") or {}).get("used", False)),
            "stage2_fallback_status": (retrieve_result.get("fallback") or {}).get("status"),
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
    if debug_dir:
        debug_path = _write_debug_artifacts(
            output_record,
            debug_dir=debug_dir,
            docs_dir=docs_dir,
            notes_path=notes_path,
            index_dir=index_dir,
            doc_index=doc_index,
            note_store=note_store,
            max_notes=debug_max_notes,
        )
        output_record["intermediate"]["debug_path"] = str(debug_path) if debug_path else None

    return output_record


def _accumulate_metrics(totals: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def _iter_final_ranked_rows(record: Dict[str, Any], top_k_export: int) -> Iterable[Dict[str, Any]]:
    qid = str(record.get("_id") or "")
    contexts = record.get("retrieved_context_topk") or []
    if not qid or not isinstance(contexts, list):
        return
    cap = max(0, int(top_k_export))
    if cap <= 0:
        return
    for idx, ctx in enumerate(contexts[:cap], start=1):
        if not isinstance(ctx, dict):
            continue
        dedup_key, dedup_source = _dedup_key(ctx, idx - 1)
        doc_title = normalize_title(ctx.get("doc_title") or ctx.get("title"))
        yield {
            "qid": qid,
            "rank": idx,
            "doc_title": doc_title,
            "chunk_id": str(ctx.get("chunk_id") or ""),
            "score": _safe_score(ctx.get("score")),
            "source": str(ctx.get("source") or ""),
            "dedup_key": f"{dedup_source}:{json.dumps(dedup_key, ensure_ascii=True)}",
            "text_hash": str(ctx.get("text_hash") or ""),
            "note_id": str(ctx.get("note_id") or ""),
            "doc_id": str(ctx.get("doc_id") or ""),
        }


def _write_final_ranked_rows(record: Dict[str, Any], handle: Optional[TextIO], top_k_export: int) -> None:
    if handle is None:
        return
    for row in _iter_final_ranked_rows(record, top_k_export):
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _collect_retrieval_stats(predictions_path: Path) -> Dict[str, Any]:
    count = 0
    top_k_raw_sum = 0.0
    top_k_final_sum = 0.0
    duplicate_rate_sum = 0.0
    shortage_count = 0
    title_diversity_applied_count = 0
    query_title_promotion_applied_count = 0
    pred_modes: Dict[str, int] = {}
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            count += 1
            top_k_raw_sum += _safe_score(row.get("top_k_raw"))
            top_k_final_sum += _safe_score(row.get("top_k_final"))
            duplicate_rate_sum += _safe_score(row.get("duplicate_rate"))
            if row.get("top_k_fill_reason"):
                shortage_count += 1
            if bool(row.get("title_diversity_applied")):
                title_diversity_applied_count += 1
            if bool(row.get("query_title_promotion_applied")):
                query_title_promotion_applied_count += 1
            mode = str(row.get("predicate_mode") or "").strip().lower()
            if mode:
                pred_modes[mode] = pred_modes.get(mode, 0) + 1
    if count <= 0:
        return {
            "count": 0,
            "top_k_raw_mean": 0.0,
            "top_k_final_mean": 0.0,
            "duplicate_rate_mean": 0.0,
            "shortage_ratio": 0.0,
            "title_diversity_applied_ratio": 0.0,
            "query_title_promotion_applied_ratio": 0.0,
            "predicate_mode_counts": pred_modes,
        }
    return {
        "count": count,
        "top_k_raw_mean": round(top_k_raw_sum / count, 4),
        "top_k_final_mean": round(top_k_final_sum / count, 4),
        "duplicate_rate_mean": round(duplicate_rate_sum / count, 6),
        "shortage_ratio": round(shortage_count / count, 6),
        "title_diversity_applied_ratio": round(title_diversity_applied_count / count, 6),
        "query_title_promotion_applied_ratio": round(query_title_promotion_applied_count / count, 6),
        "predicate_mode_counts": pred_modes,
    }


def _drain_futures(
    future_map: Dict[Any, str],
    handle,
    totals: Dict[str, float],
    final_topk_handle: Optional[TextIO] = None,
    final_topk_export: int = DEFAULT_EXPORT_TOP_K_MAX,
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
                _write_final_ranked_rows(
                    record,
                    handle=final_topk_handle,
                    top_k_export=final_topk_export,
                )
                _accumulate_metrics(totals, record.get("metrics") or {})
                succeeded += 1
            completed += 1
            last_progress = time.time()
            if progress is not None:
                progress.update(1)
    return completed, succeeded


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
        "--pred_official_topk_out",
        str(align_dir / "official_pred_topk.json"),
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


def run_experiment_task(
    reader: str,
    mode: str,
    split: str,
    data_path: Path,
    cache_root: Path,
    output_dir: Path,
    debug_dir: Optional[Path],
    run_dir: Optional[str],
    repo_root: Path,
    timestamp: int,
    base_cfg: Dict[str, Any],
    openai_runtime_cfg: Optional[Dict[str, Any]],
    llm_endpoint: str,
    llm_model: str,
    top_k: int,
    top_k_raw: Optional[int],
    overfetch: Optional[float],
    min_overfetch: float,
    export_top_k_max: int,
    predicate_mode: str,
    predicate_random_seed: int,
    use_alias_binding: bool,
    use_alias_lookup: bool,
    retrieval_only: bool,
    disable_retriever_llm: bool,
    backfill_max_overfetch: float,
    backfill_step: float,
    backfill_rounds: int,
    shortage_refill_enabled: bool,
    shortage_refill_max_candidates: int,
    shortage_refill_min_score: float,
    shortage_refill_prefer_new_titles: bool,
    title_diversity_enabled: bool,
    title_diversity_top_n: int,
    title_diversity_keep_first: bool,
    query_title_promotion_enabled: bool,
    query_title_promotion_window: int,
    pred_sp_policy: str,
    pred_sp_max_facts: int,
    pred_sp_min_score: float,
    pred_sp_drop_weak: bool,
    pred_sp_prefer_new_titles: bool,
    llm_retry_on_empty: int,
    llm_retry_max_evidence: int,
    limit: int,
    workers: int,
    force_build: bool,
    debug_max_notes: int,
    stall_warn_sec: float,
    stall_abort_sec: float,
    readers_count: int,
    modes_count: int,
    chunking_method: str = "sentence",
    chunking_size: int = 256,
    chunking_overlap: int = 32,
) -> Dict[str, Any]:
    reader_openai_cfg = openai_runtime_cfg if reader == "openai" else None
    answer_model = reader_openai_cfg.get("model") if reader == "openai" and reader_openai_cfg else llm_model
    effective_base_cfg = deepcopy(base_cfg)
    if retrieval_only and disable_retriever_llm:
        effective_base_cfg.setdefault("reranker", {})["enabled"] = False
    effective_entry_cfg = effective_base_cfg.setdefault("hotpot_entry", {})
    if isinstance(effective_entry_cfg, dict):
        effective_entry_cfg["title_diversity_enabled"] = bool(title_diversity_enabled)
        effective_entry_cfg["title_diversity_top_n"] = int(title_diversity_top_n)
        effective_entry_cfg["title_diversity_keep_first"] = bool(title_diversity_keep_first)
        effective_entry_cfg["query_title_promotion_enabled"] = bool(query_title_promotion_enabled)
        effective_entry_cfg["query_title_promotion_window"] = int(query_title_promotion_window)

    single_task = readers_count == 1 and modes_count == 1
    output_base = Path(run_dir) if run_dir else output_dir
    output_base.mkdir(parents=True, exist_ok=True)
    suffix = "" if single_task else f".{reader}.{mode}"

    mode_top_k = _resolve_mode_top_k(mode, effective_base_cfg, top_k)
    mode_top_k_raw, top_k_raw_source = _resolve_top_k_raw(
        mode_top_k,
        top_k_raw,
        overfetch,
        min_overfetch,
    )
    output_name = "predictions.jsonl" if run_dir else _pred_filename(split, reader, mode, readers_count, modes_count)
    output_path = output_base / output_name

    retrieval_dir = output_base / "retrieval"
    if single_task:
        final_topk_path = retrieval_dir / "final_top50.jsonl"
    else:
        final_topk_path = retrieval_dir / f"{reader}_{mode}" / "final_top50.jsonl"
    final_topk_path.parent.mkdir(parents=True, exist_ok=True)

    run_debug_dir = debug_dir
    if debug_dir and (readers_count > 1 or modes_count > 1):
        run_debug_dir = debug_dir / f"{reader}_{mode}"
        run_debug_dir.mkdir(parents=True, exist_ok=True)

    run_started_at = time.time()
    resolved_cfg = deepcopy(effective_base_cfg)
    if llm_endpoint:
        resolved_cfg.setdefault("vllm", {})["endpoint"] = llm_endpoint
    if llm_model:
        resolved_cfg.setdefault("vllm", {})["model"] = llm_model
    if reader_openai_cfg:
        resolved_cfg["openai"] = deepcopy(reader_openai_cfg)
    entry_snapshot = resolved_cfg.setdefault("hotpot_entry", {})
    entry_snapshot.update(
        {
            "data": str(data_path),
            "cache_dir": str(cache_root),
            "output_dir": str(output_base),
            "split": split,
            "reader": reader,
            "retriever": mode,
            "top_k": mode_top_k,
            "top_k_raw": mode_top_k_raw,
            "top_k_raw_source": top_k_raw_source,
            "export_top_k_max": int(export_top_k_max),
            "predicate_mode": str(predicate_mode),
            "predicate_random_seed": int(predicate_random_seed),
            "use_alias_binding": bool(use_alias_binding),
            "use_alias_lookup": bool(use_alias_lookup),
            "retrieval_only": bool(retrieval_only),
            "disable_retriever_llm": bool(disable_retriever_llm),
            "overfetch": overfetch,
            "min_overfetch": min_overfetch,
            "backfill_max_overfetch": backfill_max_overfetch,
            "backfill_step": backfill_step,
            "backfill_rounds": backfill_rounds,
            "shortage_refill_enabled": shortage_refill_enabled,
            "shortage_refill_max_candidates": shortage_refill_max_candidates,
            "shortage_refill_min_score": shortage_refill_min_score,
            "shortage_refill_prefer_new_titles": shortage_refill_prefer_new_titles,
            "title_diversity_enabled": bool(title_diversity_enabled),
            "title_diversity_top_n": int(title_diversity_top_n),
            "title_diversity_keep_first": bool(title_diversity_keep_first),
            "query_title_promotion_enabled": bool(query_title_promotion_enabled),
            "query_title_promotion_window": int(query_title_promotion_window),
            "pred_sp_policy": pred_sp_policy,
            "pred_sp_max_facts": pred_sp_max_facts,
            "pred_sp_min_score": pred_sp_min_score,
            "pred_sp_drop_weak": pred_sp_drop_weak,
            "pred_sp_prefer_new_titles": pred_sp_prefer_new_titles,
            "llm_retry_on_empty": llm_retry_on_empty,
            "llm_retry_max_evidence": llm_retry_max_evidence,
            "limit": limit,
            "workers": workers,
        }
    )
    config_path = output_base / f"config.resolved{suffix}.json"
    _write_json(config_path, _sanitize_config(resolved_cfg))

    answer_cfg = (effective_base_cfg.get("answer") or {})
    llm_cfg = (effective_base_cfg.get("llm") or {})
    retriever_cfg = (effective_base_cfg.get("retriever") or {})
    random_mode = "none"
    if str(predicate_mode).strip().lower() == "random":
        random_mode = "per_state_hash"
    postprocess_signature = {
        "version": "v1",
        "dedup_key": "chunk_id->doc_id+sentence_idx->title+sentence_idx->title_hash->fallback",
        "top_k_fill_policy": "missing_as_zero_in_eval",
        "overfetch": overfetch,
        "min_overfetch": min_overfetch,
        "backfill_max_overfetch": backfill_max_overfetch,
        "backfill_step": backfill_step,
        "backfill_rounds": backfill_rounds,
        "shortage_refill_enabled": bool(shortage_refill_enabled),
        "title_diversity_enabled": bool(title_diversity_enabled),
        "title_diversity_top_n": int(title_diversity_top_n),
        "title_diversity_keep_first": bool(title_diversity_keep_first),
        "query_title_promotion_enabled": bool(query_title_promotion_enabled),
        "query_title_promotion_window": int(query_title_promotion_window),
    }
    run_meta = {
        "run_dir": str(output_base),
        "dataset": "hotpotqa",
        "data_path": str(data_path),
        "sample_count": _count_examples(data_path, limit),
        "split": split,
        "reader": reader,
        "retriever": mode,
        "cache_dir": str(cache_root),
        "output_path": str(output_path),
        "final_top50_path": str(final_topk_path),
        "config_snapshot_path": str(config_path),
        "llm_endpoint": llm_endpoint,
        "llm_model": llm_model,
        "openai": {
            "base_url": (reader_openai_cfg or {}).get("base_url"),
            "model": (reader_openai_cfg or {}).get("model"),
            "temperature": (reader_openai_cfg or {}).get("temperature"),
            "max_tokens": (reader_openai_cfg or {}).get("max_tokens"),
        }
        if reader == "openai"
        else None,
        "retrieval_only": bool(retrieval_only),
        "disable_retriever_llm": bool(disable_retriever_llm),
        "llm_calls": 0 if (retrieval_only and disable_retriever_llm) else None,
        "top_k": mode_top_k,
        "top_k_raw": mode_top_k_raw,
        "top_k_raw_source": top_k_raw_source,
        "export_top_k_max": int(export_top_k_max),
        "overfetch": overfetch,
        "min_overfetch": min_overfetch,
        "backfill_max_overfetch": backfill_max_overfetch,
        "backfill_step": backfill_step,
        "backfill_rounds": backfill_rounds,
        "shortage_refill": {
            "enabled": bool(shortage_refill_enabled),
            "max_candidates": int(shortage_refill_max_candidates),
            "min_score": float(shortage_refill_min_score),
            "prefer_new_titles": bool(shortage_refill_prefer_new_titles),
        },
        "title_diversity": {
            "enabled": bool(title_diversity_enabled),
            "top_n": int(title_diversity_top_n),
            "keep_first": bool(title_diversity_keep_first),
        },
        "query_title_promotion": {
            "enabled": bool(query_title_promotion_enabled),
            "window_n": int(query_title_promotion_window),
        },
        "predicate_mode": str(predicate_mode),
        "predicate_constraint_enabled": str(predicate_mode).strip().lower() != "off",
        "predicate_random_seed": int(predicate_random_seed),
        "predicate_random_mode": random_mode,
        "pred_sp_policy": {
            "policy": pred_sp_policy,
            "max_facts": int(pred_sp_max_facts),
            "min_score": float(pred_sp_min_score),
            "drop_weak": bool(pred_sp_drop_weak),
            "prefer_new_titles": bool(pred_sp_prefer_new_titles),
        },
        "dedup_strategy": "chunk_id -> (doc_id,sentence_idx) -> (title,sentence_idx) -> title_hash -> fallback_index",
        "postprocess_signature": postprocess_signature,
        "budget_tokens": {
            "answer_max_evidence_tokens": answer_cfg.get("max_evidence_tokens"),
            "answer_max_evidence_items": answer_cfg.get("max_evidence_items"),
            "llm_max_context_len": llm_cfg.get("max_context_len"),
            "llm_safety_margin_tokens": llm_cfg.get("safety_margin_tokens"),
        },
        "rerank_enabled": bool((effective_base_cfg.get("reranker") or {}).get("enabled", False)),
        "hybrid_enabled": bool((retriever_cfg.get("hybrid") or {}).get("enabled", False)),
        "command": _sanitize_argv(sys.argv),
        "started_at": int(run_started_at),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_info(repo_root),
        "env": _env_snapshot(),
    }
    run_meta_path = output_base / f"run_meta{suffix}.json"
    _write_json(run_meta_path, run_meta)

    logger.info("Starting run: reader={} mode={} output={}", reader, mode, output_path)
    totals = {"bleu1": 0.0, "bleu4": 0.0, "rougeL": 0.0, "meteor": 0.0}
    total_examples = run_meta["sample_count"]
    show_progress = single_task
    progress = ProgressBar(total_examples) if show_progress else None
    processed = 0
    completed = 0

    final_topk_handle = final_topk_path.open("w", encoding="utf-8")
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            if workers <= 1:
                for example in _load_jsonl(data_path):
                    if limit and completed >= limit:
                        break
                    try:
                        record = _process_example(
                            example,
                            cache_root=cache_root,
                            base_cfg=effective_base_cfg,
                            llm_endpoint=llm_endpoint,
                            llm_model=llm_model,
                            mode=mode,
                            reader=reader,
                            openai_cfg=reader_openai_cfg,
                            top_k=mode_top_k,
                            top_k_raw=mode_top_k_raw,
                            top_k_raw_source=top_k_raw_source,
                            backfill_max_overfetch=backfill_max_overfetch,
                            backfill_step=backfill_step,
                            backfill_rounds=backfill_rounds,
                            shortage_refill_enabled=shortage_refill_enabled,
                            shortage_refill_max_candidates=shortage_refill_max_candidates,
                            shortage_refill_min_score=shortage_refill_min_score,
                            shortage_refill_prefer_new_titles=shortage_refill_prefer_new_titles,
                            predicate_mode=predicate_mode,
                            predicate_random_seed=predicate_random_seed,
                            use_alias_binding=use_alias_binding,
                            use_alias_lookup=use_alias_lookup,
                            pred_sp_policy=pred_sp_policy,
                            pred_sp_max_facts=pred_sp_max_facts,
                            pred_sp_min_score=pred_sp_min_score,
                            pred_sp_drop_weak=pred_sp_drop_weak,
                            pred_sp_prefer_new_titles=pred_sp_prefer_new_titles,
                            retrieval_only=retrieval_only,
                            llm_retry_on_empty=llm_retry_on_empty,
                            llm_retry_max_evidence=llm_retry_max_evidence,
                            force_build=force_build,
                            debug_dir=run_debug_dir,
                            debug_max_notes=debug_max_notes,
                            run_dir=str(output_base),
                            chunking_method=chunking_method,
                            chunking_size=chunking_size,
                            chunking_overlap=chunking_overlap,
                        )
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        _write_final_ranked_rows(
                            record,
                            handle=final_topk_handle,
                            top_k_export=export_top_k_max,
                        )
                        _accumulate_metrics(totals, record.get("metrics") or {})
                        processed += 1
                    except Exception as exc:
                        qid = example.get("_id")
                        logger.error("Failed example {}: {}", qid, exc)
                    finally:
                        completed += 1
                        if progress:
                            progress.update(1)
            else:
                max_workers = max(1, int(workers))
                future_map: Dict[Any, str] = {}
                scheduled = 0
                buffer_cap = max_workers * 2
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for example in _load_jsonl(data_path):
                        if limit and scheduled >= limit:
                            break
                        qid = str(example.get("_id") or "unknown")
                        future = executor.submit(
                            _process_example,
                            example,
                            cache_root,
                            effective_base_cfg,
                            llm_endpoint,
                            llm_model,
                            mode,
                            reader,
                            reader_openai_cfg,
                            mode_top_k,
                            mode_top_k_raw,
                            top_k_raw_source,
                            backfill_max_overfetch,
                            backfill_step,
                            backfill_rounds,
                            shortage_refill_enabled,
                            shortage_refill_max_candidates,
                            shortage_refill_min_score,
                            shortage_refill_prefer_new_titles,
                            predicate_mode,
                            predicate_random_seed,
                            use_alias_binding,
                            use_alias_lookup,
                            pred_sp_policy,
                            pred_sp_max_facts,
                            pred_sp_min_score,
                            pred_sp_drop_weak,
                            pred_sp_prefer_new_titles,
                            retrieval_only,
                            llm_retry_on_empty,
                            llm_retry_max_evidence,
                            force_build,
                            run_debug_dir,
                            debug_max_notes,
                            str(output_base),
                            chunking_method,
                            chunking_size,
                            chunking_overlap,
                        )
                        future_map[future] = qid
                        scheduled += 1
                        if len(future_map) >= buffer_cap:
                            done_count, ok_count = _drain_futures(
                                future_map,
                                handle,
                                totals,
                                final_topk_handle=final_topk_handle,
                                final_topk_export=export_top_k_max,
                                progress=progress,
                                stall_warn_sec=stall_warn_sec,
                                stall_abort_sec=stall_abort_sec,
                            )
                            completed += done_count
                            processed += ok_count
                            future_map = {}
                    if future_map:
                        done_count, ok_count = _drain_futures(
                            future_map,
                            handle,
                            totals,
                            final_topk_handle=final_topk_handle,
                            final_topk_export=export_top_k_max,
                            progress=progress,
                            stall_warn_sec=stall_warn_sec,
                            stall_abort_sec=stall_abort_sec,
                        )
                        completed += done_count
                        processed += ok_count
    finally:
        final_topk_handle.close()
        if progress:
            progress.close()

    failed = completed - processed
    duration_sec = max(0.0, time.time() - run_started_at)
    logger.info("Reader {} mode {} completed {} examples (failed {})", reader, mode, processed, failed)

    denom = processed if processed > 0 else 1
    stats = {
        "bleu1": round(totals["bleu1"] / denom, 4),
        "bleu4": round(totals["bleu4"] / denom, 4),
        "rougeL": round(totals["rougeL"] / denom, 4),
        "meteor": round(totals["meteor"] / denom, 4),
        "count": processed,
        "failed": failed,
        "duration_sec": round(duration_sec, 2),
        "model": answer_model,
        "top_k": mode_top_k,
        "top_k_raw": mode_top_k_raw,
        "top_k_raw_source": top_k_raw_source,
        "retrieval_only": bool(retrieval_only),
    }
    retrieval_stats = _collect_retrieval_stats(output_path) if output_path.exists() else {}

    metrics_path = output_base / f"metrics{suffix}.json"
    _write_json(
        metrics_path,
        {
            "reader": reader,
            "retriever": mode,
            "split": split,
            "stats": stats,
            "retrieval_stats": retrieval_stats,
            "output_path": str(output_path),
            "final_top50_path": str(final_topk_path),
            "timestamp": int(time.time()),
        },
    )

    if not retrieval_only and single_task:
        official_path = _write_official_output(output_path, output_base, timestamp)
        logger.info("Official-format output written to {}", official_path)
        try:
            _run_alignment(
                repo_root=repo_root,
                pred_path=output_path,
                gold_path=data_path,
                output_dir=output_base,
                split=split,
            )
            logger.info("Alignment artifacts written for {}", output_path)
        except Exception as exc:
            logger.error("Alignment generation failed for {}: {}", output_path, exc)

    run_meta["ended_at"] = int(time.time())
    run_meta["duration_sec"] = round(duration_sec, 2)
    run_meta["status"] = "ok" if failed == 0 else "partial"
    run_meta["stats_path"] = str(metrics_path)
    run_meta["retrieval_stats"] = retrieval_stats
    _write_json(run_meta_path, run_meta)

    return {
        "reader": reader,
        "mode": mode,
        "stats": stats,
        "output_path": str(output_path),
        "final_top50_path": str(final_topk_path),
        "run_meta_path": str(run_meta_path),
        "config_snapshot_path": str(config_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HotpotQA JSONL entry for RelRAG")
    parser.add_argument("--config", help="Path to YAML config file (defaults to relrag/config/config.yaml)")
    parser.add_argument("--data", help="Path to HotpotQA JSONL dataset (fallback to config)")
    parser.add_argument("--endpoint", help="vLLM endpoint (defaults to config)")
    parser.add_argument("--model", help="LLM model name (defaults to config)")
    parser.add_argument("--reader", nargs="+", help="Reader backend: vllm or openai (fallback to config)")
    parser.add_argument("--retriever", nargs="+", help="Retriever mode: bm25, dense, or hybrid (fallback to config)")
    parser.add_argument("--split", help="Dataset split label for output naming (fallback to config)")
    parser.add_argument("--openai_model", help="OpenAI model name (fallback to config)")
    parser.add_argument("--openai_api_key", help="OpenAI API key (reads env if omitted)")
    parser.add_argument("--openai_temperature", type=float, help="OpenAI temperature (fallback to config)")
    parser.add_argument("--openai_max_tokens", type=int, help="OpenAI max tokens (fallback to config)")
    parser.add_argument("--top_k", type=int, help="Top-k retrieval fanout (fallback to config)")
    parser.add_argument("--top_k_raw", type=int, help="Raw top-k before dedup (fallback to config/overfetch)")
    parser.add_argument("--export_top_k_max", type=int, help="Export top-K retrieval ranks to retrieval/final_top50.jsonl (fallback to config)")
    parser.add_argument("--retrieval_only", help="Run retrieval/export only (skip answer generation and reader calls)")
    parser.add_argument("--disable_retriever_llm", help="Disable LLM-based retriever stages (e.g., LLM reranker) for retrieval-only runs")
    parser.add_argument("--predicate_mode", help="Predicate constraint mode for structured walk: on, off, or random")
    parser.add_argument("--predicate_random_seed", type=int, help="Random seed for predicate_mode=random")
    parser.add_argument("--use_alias_binding", help="Enable alias binding for seeds (default: true)")
    parser.add_argument("--use_alias_lookup", help="Enable alias lookup for scoring (default: true)")
    parser.add_argument("--overfetch", type=float, help="Overfetch multiplier before dedup (fallback to config)")
    parser.add_argument("--min_overfetch", type=float, help="Minimum overfetch multiplier (fallback to config)")
    parser.add_argument("--backfill_max_overfetch", type=float, help="Max overfetch multiplier for backfill (fallback to config)")
    parser.add_argument("--backfill_step", type=float, help="Backfill growth factor per retry (fallback to config)")
    parser.add_argument("--backfill_rounds", type=int, help="Max backfill attempts (fallback to config)")
    parser.add_argument("--shortage_refill_enabled", help="Enable quality refill when top-k remains short (fallback to config)")
    parser.add_argument("--shortage_refill_max_candidates", type=int, help="Max refill candidates scanned per query (fallback to config)")
    parser.add_argument("--shortage_refill_min_score", type=float, help="Min lexical score for refill candidates (fallback to config)")
    parser.add_argument("--shortage_refill_prefer_new_titles", help="Prefer refill from unseen titles first (fallback to config)")
    parser.add_argument("--title_diversity_enabled", help="Promote unique document titles in early retrieval ranks (fallback to config)")
    parser.add_argument("--title_diversity_top_n", type=int, help="Apply title diversity within first N ranks (fallback to config)")
    parser.add_argument("--title_diversity_keep_first", help="Keep rank-1 fixed when applying title diversity (fallback to config)")
    parser.add_argument("--query_title_promotion_enabled", help="Promote a second title that overlaps question tokens (fallback to config)")
    parser.add_argument("--query_title_promotion_window", type=int, help="Scan this many early ranks when promoting second title (fallback to config)")
    parser.add_argument("--pred_sp_policy", help="Supporting fact policy: topk or high_confidence (fallback to config)")
    parser.add_argument("--pred_sp_max_facts", type=int, help="Max supporting facts for high-confidence policy (fallback to config)")
    parser.add_argument("--pred_sp_min_score", type=float, help="Min score for supporting facts under high-confidence policy (fallback to config)")
    parser.add_argument("--pred_sp_drop_weak", help="Drop weak evidences for high-confidence pred_sp (fallback to config)")
    parser.add_argument("--pred_sp_prefer_new_titles", help="Prefer diverse titles in high-confidence pred_sp (fallback to config)")
    parser.add_argument("--llm_retry_on_empty", type=int, help="Retry LLM on empty/parse fallback (fallback to config)")
    parser.add_argument("--llm_retry_max_evidence", type=int, help="Max evidences on retry (fallback to config)")
    parser.add_argument("--limit", type=int, help="Process only first N examples (fallback to config)")
    parser.add_argument("--workers", type=int, help="Parallel workers (single process, fallback to config)")
    parser.add_argument("--cache_dir", help="Cache root for per-question indexes (fallback to config)")
    parser.add_argument("--output_dir", help="Output directory (fallback to config)")
    parser.add_argument("--output", help="Alias for --output_dir")
    parser.add_argument("--force_build", action="store_true", help="Rebuild indexes even if cached")
    parser.add_argument("--debug_dir", help="Debug artifacts output directory (set empty to disable, fallback to config)")
    parser.add_argument("--debug_max_notes", type=int, help="Max notes to store per question in debug dump (fallback to config)")
    parser.add_argument("--stall_warn_sec", type=float, help="Warn if no worker finishes within this many seconds (fallback to config)")
    parser.add_argument("--stall_abort_sec", type=float, help="Abort pending workers after this many idle seconds (0 to disable, fallback to config)")
    parser.add_argument("--chunking_method", choices=["sentence", "fixed"], help="Chunking method: sentence or fixed")
    parser.add_argument("--chunking_size", type=int, help="Fixed chunk size in tokens")
    parser.add_argument("--chunking_overlap", type=int, help="Fixed chunk overlap in tokens")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent

    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else repo_root / path

    cfg = ConfigLoader(args.config).load_config() if args.config else global_config.load_config()
    dataset_cfg = get_dataset_config(cfg, "hotpotqa")
    entry_cfg = _load_entry_config(cfg)

    if args.output and not args.output_dir:
        args.output_dir = args.output

    args.data = _pick_arg(args, entry_cfg, dataset_cfg, "data", None)
    if not args.data:
        raise ValueError("Dataset path missing. Provide --data or set hotpot_entry.data in config.")
    args.split = _pick_arg(args, entry_cfg, dataset_cfg, "split", DEFAULT_SPLIT)
    args.cache_dir = _pick_arg(args, entry_cfg, dataset_cfg, "cache_dir", DEFAULT_CACHE_DIR)
    args.output_dir = _pick_arg(args, entry_cfg, dataset_cfg, "output_dir", DEFAULT_OUTPUT_DIR)
    args.chunking_method = _pick_arg(args, entry_cfg, dataset_cfg, "chunking_method", "sentence")
    args.chunking_size = _coerce_int(_pick_arg(args, entry_cfg, dataset_cfg, "chunking_size", 256), 256)
    args.chunking_overlap = _coerce_int(_pick_arg(args, entry_cfg, dataset_cfg, "chunking_overlap", 32), 32)
    args.top_k = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "top_k", DEFAULT_TOP_K),
        DEFAULT_TOP_K,
    )
    args.top_k_raw = _pick_arg(args, entry_cfg, dataset_cfg, "top_k_raw", None)
    args.export_top_k_max = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "export_top_k_max", DEFAULT_EXPORT_TOP_K_MAX),
        DEFAULT_EXPORT_TOP_K_MAX,
    )
    predicate_mode_raw = _pick_arg(args, entry_cfg, dataset_cfg, "predicate_mode", DEFAULT_PREDICATE_MODE)
    args.predicate_mode = str(predicate_mode_raw or DEFAULT_PREDICATE_MODE).strip().lower()
    if args.predicate_mode not in {"on", "off", "random"}:
        raise ValueError("predicate_mode must be one of: on, off, random")
    args.predicate_random_seed = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "predicate_random_seed", DEFAULT_PREDICATE_RANDOM_SEED),
        DEFAULT_PREDICATE_RANDOM_SEED,
    )
    args.use_alias_binding = _coerce_bool(
        _pick_arg(args, entry_cfg, dataset_cfg, "use_alias_binding", True),
        True,
    )
    args.use_alias_lookup = _coerce_bool(
        _pick_arg(args, entry_cfg, dataset_cfg, "use_alias_lookup", True),
        True,
    )
    args.retrieval_only = _coerce_bool(
        _pick_arg(args, entry_cfg, dataset_cfg, "retrieval_only", False),
        False,
    )
    args.disable_retriever_llm = _coerce_bool(
        _pick_arg(args, entry_cfg, dataset_cfg, "disable_retriever_llm", False),
        False,
    )
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
    args.shortage_refill_enabled = _coerce_bool(
        _pick_arg(args, entry_cfg, dataset_cfg, "shortage_refill_enabled", DEFAULT_SHORTAGE_REFILL_ENABLED),
        DEFAULT_SHORTAGE_REFILL_ENABLED,
    )
    args.shortage_refill_max_candidates = _coerce_int(
        _pick_arg(
            args,
            entry_cfg,
            dataset_cfg,
            "shortage_refill_max_candidates",
            DEFAULT_SHORTAGE_REFILL_MAX_CANDIDATES,
        ),
        DEFAULT_SHORTAGE_REFILL_MAX_CANDIDATES,
    )
    args.shortage_refill_min_score = _coerce_float(
        _pick_arg(args, entry_cfg, dataset_cfg, "shortage_refill_min_score", DEFAULT_SHORTAGE_REFILL_MIN_SCORE),
        DEFAULT_SHORTAGE_REFILL_MIN_SCORE,
    )
    args.shortage_refill_prefer_new_titles = _coerce_bool(
        _pick_arg(
            args,
            entry_cfg,
            dataset_cfg,
            "shortage_refill_prefer_new_titles",
            DEFAULT_SHORTAGE_REFILL_PREFER_NEW_TITLES,
        ),
        DEFAULT_SHORTAGE_REFILL_PREFER_NEW_TITLES,
    )
    args.title_diversity_enabled = _coerce_bool(
        _pick_arg(args, entry_cfg, dataset_cfg, "title_diversity_enabled", DEFAULT_TITLE_DIVERSITY_ENABLED),
        DEFAULT_TITLE_DIVERSITY_ENABLED,
    )
    args.title_diversity_top_n = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "title_diversity_top_n", DEFAULT_TITLE_DIVERSITY_TOP_N),
        DEFAULT_TITLE_DIVERSITY_TOP_N,
    )
    args.title_diversity_top_n = max(0, int(args.title_diversity_top_n))
    args.title_diversity_keep_first = _coerce_bool(
        _pick_arg(args, entry_cfg, dataset_cfg, "title_diversity_keep_first", DEFAULT_TITLE_DIVERSITY_KEEP_FIRST),
        DEFAULT_TITLE_DIVERSITY_KEEP_FIRST,
    )
    args.query_title_promotion_enabled = _coerce_bool(
        _pick_arg(
            args,
            entry_cfg,
            dataset_cfg,
            "query_title_promotion_enabled",
            DEFAULT_QUERY_TITLE_PROMOTION_ENABLED,
        ),
        DEFAULT_QUERY_TITLE_PROMOTION_ENABLED,
    )
    args.query_title_promotion_window = _coerce_int(
        _pick_arg(
            args,
            entry_cfg,
            dataset_cfg,
            "query_title_promotion_window",
            DEFAULT_QUERY_TITLE_PROMOTION_WINDOW,
        ),
        DEFAULT_QUERY_TITLE_PROMOTION_WINDOW,
    )
    args.query_title_promotion_window = max(0, int(args.query_title_promotion_window))
    pred_sp_policy_raw = _pick_arg(args, entry_cfg, dataset_cfg, "pred_sp_policy", DEFAULT_PRED_SP_POLICY)
    args.pred_sp_policy = _normalize_pred_sp_policy(pred_sp_policy_raw)
    args.pred_sp_max_facts = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "pred_sp_max_facts", DEFAULT_PRED_SP_MAX_FACTS),
        DEFAULT_PRED_SP_MAX_FACTS,
    )
    args.pred_sp_min_score = _coerce_float(
        _pick_arg(args, entry_cfg, dataset_cfg, "pred_sp_min_score", DEFAULT_PRED_SP_MIN_SCORE),
        DEFAULT_PRED_SP_MIN_SCORE,
    )
    args.pred_sp_drop_weak = _coerce_bool(
        _pick_arg(args, entry_cfg, dataset_cfg, "pred_sp_drop_weak", DEFAULT_PRED_SP_DROP_WEAK),
        DEFAULT_PRED_SP_DROP_WEAK,
    )
    args.pred_sp_prefer_new_titles = _coerce_bool(
        _pick_arg(
            args,
            entry_cfg,
            dataset_cfg,
            "pred_sp_prefer_new_titles",
            DEFAULT_PRED_SP_PREFER_NEW_TITLES,
        ),
        DEFAULT_PRED_SP_PREFER_NEW_TITLES,
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
    args.debug_dir = _pick_arg(args, entry_cfg, dataset_cfg, "debug_dir", DEFAULT_DEBUG_DIR)
    args.debug_max_notes = _coerce_int(
        _pick_arg(args, entry_cfg, dataset_cfg, "debug_max_notes", DEFAULT_DEBUG_MAX_NOTES),
        DEFAULT_DEBUG_MAX_NOTES,
    )
    args.stall_warn_sec = _coerce_float(
        _pick_arg(args, entry_cfg, dataset_cfg, "stall_warn_sec", DEFAULT_STALL_WARN_SEC),
        DEFAULT_STALL_WARN_SEC,
    )
    args.stall_abort_sec = _coerce_float(
        _pick_arg(args, entry_cfg, dataset_cfg, "stall_abort_sec", DEFAULT_STALL_ABORT_SEC),
        DEFAULT_STALL_ABORT_SEC,
    )
    if args.top_k > 0 and args.top_k < args.export_top_k_max:
        logger.warning(
            "top_k ({}) is smaller than export_top_k_max ({}); exported ranked list will be shorter than requested cap",
            args.top_k,
            args.export_top_k_max,
        )
    if args.retrieval_only and not args.disable_retriever_llm:
        logger.warning(
            "retrieval_only is enabled but disable_retriever_llm=false; retrieval pipeline may still invoke LLM reranker",
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

    data_path = _resolve_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    llm_endpoint, llm_model = _resolve_llm_config(args)
    logger.info("Using vLLM endpoint={} model={}", llm_endpoint, llm_model)
    cache_root = _resolve_path(args.cache_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir: Optional[Path] = None
    timestamp = int(time.time())
    total_examples = _count_examples(data_path, args.limit)
    debug_dir = _resolve_path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _apply_dataset_retriever(deepcopy(cfg), "hotpotqa")
    modes = _resolve_retriever_modes(
        mode_arg=None,
        modes_arg=args.retriever,
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

    split = str(args.split or DEFAULT_SPLIT).strip() or DEFAULT_SPLIT
    summary_report: Dict[str, Any] = {
        "split": split,
        "readers": readers,
        "retrievers": modes,
        "runs": {},
    }

    tasks = []
    for reader in readers:
        task_base_cfg = base_cfg
        if reader == "openai":
            task_base_cfg = _apply_openai_reranker(base_cfg, openai_runtime_cfg)
        summary_report["runs"].setdefault(reader, {})
        for mode in modes:
            tasks.append({
                "reader": reader,
                "mode": mode,
                "split": split,
                "data_path": data_path,
                "cache_root": cache_root,
                "output_dir": output_dir,
                "debug_dir": debug_dir,
                "run_dir": str(run_dir) if run_dir else None,
                "repo_root": repo_root,
                "timestamp": timestamp,
                "base_cfg": task_base_cfg,
                "openai_runtime_cfg": openai_runtime_cfg,
                "llm_endpoint": llm_endpoint,
                "llm_model": llm_model,
                "top_k": args.top_k,
                "top_k_raw": args.top_k_raw,
                "export_top_k_max": args.export_top_k_max,
                "predicate_mode": args.predicate_mode,
                "predicate_random_seed": args.predicate_random_seed,
                "use_alias_binding": args.use_alias_binding,
                "use_alias_lookup": args.use_alias_lookup,
                "retrieval_only": args.retrieval_only,
                "disable_retriever_llm": args.disable_retriever_llm,
                "overfetch": args.overfetch,
                "min_overfetch": args.min_overfetch,
                "backfill_max_overfetch": args.backfill_max_overfetch,
                "backfill_step": args.backfill_step,
                "backfill_rounds": args.backfill_rounds,
                "shortage_refill_enabled": args.shortage_refill_enabled,
                "shortage_refill_max_candidates": args.shortage_refill_max_candidates,
                "shortage_refill_min_score": args.shortage_refill_min_score,
                "shortage_refill_prefer_new_titles": args.shortage_refill_prefer_new_titles,
                "title_diversity_enabled": args.title_diversity_enabled,
                "title_diversity_top_n": args.title_diversity_top_n,
                "title_diversity_keep_first": args.title_diversity_keep_first,
                "query_title_promotion_enabled": args.query_title_promotion_enabled,
                "query_title_promotion_window": args.query_title_promotion_window,
                "pred_sp_policy": args.pred_sp_policy,
                "pred_sp_max_facts": args.pred_sp_max_facts,
                "pred_sp_min_score": args.pred_sp_min_score,
                "pred_sp_drop_weak": args.pred_sp_drop_weak,
                "pred_sp_prefer_new_titles": args.pred_sp_prefer_new_titles,
                "llm_retry_on_empty": args.llm_retry_on_empty,
                "llm_retry_max_evidence": args.llm_retry_max_evidence,
                "limit": args.limit,
                "workers": args.workers,
                "force_build": args.force_build,
                "debug_max_notes": args.debug_max_notes,
                "stall_warn_sec": args.stall_warn_sec,
                "stall_abort_sec": args.stall_abort_sec,
                "readers_count": len(readers),
                "modes_count": len(modes),
                "chunking_method": args.chunking_method,
                "chunking_size": args.chunking_size,
                "chunking_overlap": args.chunking_overlap,
            })

    results = []
    if len(tasks) == 1:
        results.append(run_experiment_task(**tasks[0]))
    else:
        max_proc = min(len(tasks), 8)
        logger.info("Running {} tasks in parallel with {} processes...", len(tasks), max_proc)
        with ProcessPoolExecutor(max_workers=max_proc) as executor:
            futures = [executor.submit(run_experiment_task, **task) for task in tasks]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("Experiment task failed: {}", exc)

    for res in results:
        reader = res["reader"]
        mode = res["mode"]
        stats = res["stats"]
        summary_report["runs"][reader][mode] = stats

    if len(readers) == 1:
        summary_report["modes"] = summary_report["runs"][readers[0]]
    if len(modes) == 1:
        summary_report["models"] = {reader: summary_report["runs"][reader][modes[0]] for reader in readers}

    summary_path = output_dir / f"summary_{split}.json"
    summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary written to {}", summary_path)


if __name__ == "__main__":
    main()
