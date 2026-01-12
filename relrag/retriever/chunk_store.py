from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from relrag.utils import TextUtils


_STOPWORDS = {
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
    "between",
    "how",
    "many",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [tok for tok in tokens if tok and tok not in _STOPWORDS]


def _has_count_intent(question: str) -> bool:
    lowered = (question or "").lower()
    return "how many" in lowered or "more" in lowered or "number of" in lowered


class ChunkStore:
    """Lazy chunk loader + lightweight lexical search."""

    def __init__(self, chunks_path: str) -> None:
        self.chunks_path = Path(chunks_path)
        self._cache: List[Dict[str, str]] | None = None
        self._token_cache: List[Tuple[Dict[str, str], set[str]]] | None = None

    def _ensure_loaded(self) -> None:
        if self._cache is not None:
            return
        if not self.chunks_path.exists():
            self._cache = []
            self._token_cache = []
            return
        chunks: List[Dict[str, str]] = []
        try:
            with self.chunks_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue
                    if not record.get("text"):
                        continue
                    chunks.append(record)
        except Exception as exc:
            logger.warning("Failed to load chunks from {}: {}", self.chunks_path, exc)
            chunks = []
        self._cache = chunks
        token_cache = []
        for chunk in chunks:
            tokens = set(_tokenize(chunk.get("text") or ""))
            token_cache.append((chunk, tokens))
        self._token_cache = token_cache

    def search(
        self,
        question: str,
        *,
        seeds: Optional[Sequence[str]] = None,
        top_k: int = 6,
        doc_hint: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        self._ensure_loaded()
        if not self._token_cache:
            return []
        q_tokens = set(_tokenize(question))
        if seeds:
            for seed in seeds:
                q_tokens.update(_tokenize(seed))
        seed_texts = [s.strip().lower() for s in (seeds or []) if s and s.strip()]
        count_intent = _has_count_intent(question)
        doc_hint_norm = (doc_hint or "").strip().lower() or None

        scored: List[Tuple[float, Dict[str, str]]] = []
        for chunk, tokens in self._token_cache:
            doc_id = (chunk.get("doc_id") or "").strip()
            if doc_hint_norm and doc_id and doc_hint_norm not in doc_id.lower():
                continue
            overlap = len(q_tokens & tokens) if q_tokens else 0
            seed_bonus = 0.0
            chunk_text = (chunk.get("text") or "").lower()
            for seed in seed_texts:
                if seed and seed in chunk_text:
                    seed_bonus += 0.8
            count_bonus = 0.2 if count_intent and any(tok.isdigit() for tok in tokens) else 0.0
            score = (overlap / max(1, len(q_tokens))) + seed_bonus + count_bonus
            if score <= 0.0:
                continue
            scored.append((score, chunk))

        if not scored and doc_hint_norm:
            # Retry without doc hint if strict filter removes everything.
            return self.search(question, seeds=seeds, top_k=top_k, doc_hint=None)

        scored.sort(key=lambda item: item[0], reverse=True)
        results: List[Dict[str, str]] = []
        for score, chunk in scored[: max(1, top_k)]:
            doc_id = (chunk.get("doc_id") or "").strip()
            chunk_id = (chunk.get("chunk_id") or "").strip()
            best_sentence = _best_sentence(chunk, q_tokens)
            evidence = best_sentence or (chunk.get("text") or "")
            results.append(
                {
                    "note_id": f"{doc_id}#{chunk_id}#chunk",
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "evidence": evidence,
                    "canonical": evidence,
                    "score": round(score, 4),
                    "chunk": True,
                }
            )
        return results


def _best_sentence(chunk: Dict[str, str], q_tokens: set[str]) -> str:
    text = (chunk.get("text") or "").strip()
    if not text:
        return ""
    spans = (chunk.get("meta") or {}).get("sent_spans") if isinstance(chunk.get("meta"), dict) else None
    sentences = [span.get("text") for span in spans] if isinstance(spans, list) else []
    if not sentences:
        sentences = TextUtils.split_by_sentence(text)
    best = ""
    best_score = -1
    for sentence in sentences:
        tokens = set(_tokenize(sentence))
        score = len(tokens & q_tokens) if q_tokens else 0
        if score > best_score:
            best_score = score
            best = sentence
    return best or text
