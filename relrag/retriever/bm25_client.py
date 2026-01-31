from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    BM25Okapi = None  # type: ignore
    logger.warning("rank_bm25 unavailable: {}", exc)


class BM25Client:
    """File-based BM25 retrieval (rank_bm25 backend)."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled"))
        self.backend = str(self.cfg.get("backend", "rank_bm25")).strip().lower()
        self.store_path = Path(self.cfg.get("store_path", "indexes/bm25/notes"))
        self.field_weights = self.cfg.get("field_weights") or {"subj": 2.0, "pred": 1.5, "obj": 1.2, "ctx": 1.0}
        self.ngram = self.cfg.get("ngram") or [1]
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        if self.enabled:
            self._load_corpus()

    def _load_corpus(self) -> None:
        if self.backend not in {"rank_bm25", "pyserini"}:
            logger.warning("Unknown BM25 backend '{}'; falling back to rank_bm25.", self.backend)
            self.backend = "rank_bm25"
        if self.backend == "pyserini":
            logger.warning("BM25 backend 'pyserini' is not implemented; falling back to rank_bm25.")
            self.backend = "rank_bm25"
        if BM25Okapi is None:
            logger.warning("BM25 backend requested but rank_bm25 not installed.")
            self.enabled = False
            return
        corpus_file = self.store_path / "notes.jsonl"
        if not corpus_file.exists():
            logger.warning("BM25 corpus missing at {}. Disable BM25 channel.", corpus_file)
            self.enabled = False
            return
        documents: List[List[str]] = []
        doc_ids: List[str] = []
        with corpus_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                note_id = record.get("note_id")
                if not note_id:
                    continue
                tokens = self._build_tokens(record)
                if not tokens:
                    continue
                documents.append(tokens)
                doc_ids.append(note_id)
        if not documents:
            logger.warning("BM25 corpus at {} is empty.", corpus_file)
            self.enabled = False
            return
        k1 = float(self.cfg.get("k1", 0.9))
        b = float(self.cfg.get("b", 0.4))
        self._bm25 = BM25Okapi(documents, k1=k1, b=b)
        self._doc_ids = doc_ids

    def _build_tokens(self, record: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []
        fields = record.get("fields") or {}
        for field, weight in self.field_weights.items():
            text = (fields.get(field) or record.get(field) or record.get("text") or "").strip()
            if not text:
                continue
            weighted = self._tokenize(text, repeats=max(1, int(round(weight * 2))))
            tokens.extend(weighted)
        if not tokens and record.get("text"):
            tokens = self._tokenize(record["text"])
        return tokens

    def _tokenize(self, text: str, repeats: int = 1) -> List[str]:
        base = [tok for tok in text.lower().split() if tok]
        expanded: List[str] = []
        for _ in range(repeats):
            expanded.extend(base)
        ngrams: List[str] = []
        for n in self.ngram:
            n = int(n)
            if n <= 1:
                ngrams.extend(expanded)
                continue
            for i in range(len(base) - n + 1):
                ngrams.append(" ".join(base[i : i + n]))
        return ngrams or expanded

    def search(self, question: str, topn: int) -> List[Dict[str, Any]]:
        if not self.enabled or not question.strip():
            return []
        if self._bm25 is None:
            self._load_corpus()
        if self._bm25 is None:
            return []
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        paired = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(paired[:topn], start=1):
            note_id = self._doc_ids[idx]
            results.append({"note_id": note_id, "score": float(score), "rank": rank, "source": "bm25"})
        return results
