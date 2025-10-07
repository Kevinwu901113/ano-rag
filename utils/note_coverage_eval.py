"""Coverage evaluation utilities for generated atomic notes."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from loguru import logger

from config import config
from utils.text_utils import TextUtils


def _tokenize(text: str) -> List[str]:
    if not text:
        return []

    normalized = TextUtils.normalize_text(text)
    tokens = [t for t in normalized.split() if t]
    if tokens:
        return tokens

    # Fallback for languages without whitespace tokenization (e.g., Chinese)
    stripped = text.strip()
    return list(stripped)


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0

    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def evaluate_note_coverage(
    text_chunks: List[Dict[str, Any]],
    notes: List[Dict[str, Any]],
) -> None:
    """Evaluate coverage of notes against source sentences and emit debug reports."""

    coverage_cfg = config.get("evaluation", {}).get("coverage", {}) or {}
    threshold = float(coverage_cfg.get("threshold", 0.6))
    min_tokens = int(coverage_cfg.get("min_sentence_tokens", 6))
    critical_threshold = float(coverage_cfg.get("critical_threshold", 0.5))
    report_path = coverage_cfg.get("report_path", "debug/coverage_report.json")
    missing_path = coverage_cfg.get("missing_sentences_path", "debug/missing_sentences.jsonl")

    if not text_chunks or notes is None:
        logger.debug("Coverage evaluation skipped: insufficient input data")
        return

    notes_by_chunk = defaultdict(list)
    for note in notes:
        idx = note.get("chunk_index")
        notes_by_chunk[idx].append(note)

    report: List[Dict[str, Any]] = []
    missing_records: List[Dict[str, Any]] = []

    for chunk in text_chunks:
        chunk_index = chunk.get("chunk_index")
        chunk_id = chunk.get("chunk_id")
        chunk_text = chunk.get("text", "")
        paragraph_info = chunk.get("paragraph_info") or []

        if not paragraph_info:
            paragraph_info = [{"idx": chunk_index, "text": chunk_text}]

        chunk_notes = notes_by_chunk.get(chunk_index, [])

        for para in paragraph_info:
            paragraph_idx = para.get("idx")
            para_text = para.get("text") or chunk_text
            sentences = TextUtils.split_by_sentence(para_text)

            filtered_sentences = [
                sentence for sentence in sentences if len(_tokenize(sentence)) >= min_tokens
            ]

            if not filtered_sentences:
                continue

            candidate_notes = []
            for note in chunk_notes:
                idxs = note.get("paragraph_idxs") or []
                if isinstance(idxs, list) and paragraph_idx in idxs:
                    candidate_notes.append(note)

            if not candidate_notes:
                candidate_notes = chunk_notes

            covered = 0
            note_tokens_cache = {
                id(note): _tokenize(note.get("content") or note.get("text") or "")
                for note in candidate_notes
            }

            for sentence in filtered_sentences:
                sent_tokens = _tokenize(sentence)
                matched = False
                for note in candidate_notes:
                    jaccard = _jaccard(sent_tokens, note_tokens_cache.get(id(note), []))
                    if jaccard >= threshold:
                        matched = True
                        break

                if matched:
                    covered += 1
                else:
                    missing_records.append(
                        {
                            "chunk_index": chunk_index,
                            "chunk_id": chunk_id,
                            "paragraph_idx": paragraph_idx,
                            "sentence": sentence,
                            "jaccard_max": max(
                                (_jaccard(sent_tokens, tokens) for tokens in note_tokens_cache.values()),
                                default=0.0,
                            ),
                        }
                    )

            coverage_rate = covered / len(filtered_sentences)
            report.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "paragraph_idx": paragraph_idx,
                    "sentences": len(filtered_sentences),
                    "covered": covered,
                    "coverage_rate": coverage_rate,
                }
            )

            if coverage_rate < critical_threshold:
                logger.error(
                    f"Coverage alert: chunk {chunk_index}, paragraph {paragraph_idx} at {coverage_rate:.2f}"
                )
            elif coverage_rate < threshold:
                logger.warning(
                    f"Coverage warning: chunk {chunk_index}, paragraph {paragraph_idx} at {coverage_rate:.2f}"
                )

    if not report:
        logger.debug("Coverage evaluation finished with no sentences to score")
        return

    report_dir = os.path.dirname(report_path) or '.'
    missing_dir = os.path.dirname(missing_path) or '.'
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(missing_dir, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(missing_path, "w", encoding="utf-8") as f:
        for record in missing_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        f"Coverage evaluation written to {report_path} with {len(missing_records)} uncovered sentences"
    )
