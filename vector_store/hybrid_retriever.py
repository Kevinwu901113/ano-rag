"""Hybrid retrieval that merges BM25 and ANN results."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from vector.encoder import encode_notes, encode_query
from vector.indexer import VectorIndexer

try:  # Optional import for type hints / structured logging
    from utils.logging import StructuredLogger
except Exception:  # pragma: no cover - optional dependency during type checking
    StructuredLogger = None  # type: ignore


@dataclass
class HybridRetrievalResult:
    """Container for the merged retrieval output."""

    merged: List[Tuple[str, float]]
    support_ids: List[str]
    mode: str
    breakdown: Dict[str, int]
    per_candidate: Dict[str, Dict[str, Any]]
    components: Dict[str, List[Tuple[str, float]]]


class HybridRetriever:
    """Run BM25 + ANN retrieval and fuse the results."""

    def __init__(
        self,
        *,
        notes: Sequence[Dict[str, Any]],
        config: Dict[str, Any],
        task_id: Any,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.notes = list(notes)
        self.config = config or {}
        self.task_id = task_id
        self.logger = logger

        retriever_cfg = (self.config.get("retriever") or {}).get("hybrid") or {}
        self.hybrid_enabled = self._as_bool(retriever_cfg.get("enabled", True), True)
        self.fusion_cfg = retriever_cfg.get("fusion") or {}
        self.fusion_strategy = str(self.fusion_cfg.get("strategy", "zscore")).lower()
        self.source_weights = {
            "bm25": float((self.fusion_cfg.get("weights") or {}).get("bm25", 1.0)),
            "vector": float((self.fusion_cfg.get("weights") or {}).get("vector", 1.0)),
        }
        self.rrf_k = float(self.fusion_cfg.get("rrf_k", 60.0))

        self.fusion_topk = int(retriever_cfg.get("fusion_topk", 50) or 50)
        self.bm25_topk = int(retriever_cfg.get("bm25_topk", self.fusion_topk) or self.fusion_topk)
        self.vector_topk = int(retriever_cfg.get("vector_topk", self.fusion_topk) or self.fusion_topk)
        self.bm25_k1 = float(retriever_cfg.get("bm25_k1", 1.5))
        self.bm25_b = float(retriever_cfg.get("bm25_b", 0.75))

        self.embedding_cfg = self.config.get("embedding") or {}
        self.ann_cfg = self.config.get("ann") or {}
        self.use_vector = bool(self.embedding_cfg) and self._as_bool(
            self.ann_cfg.get("enabled", False), False
        )
        self.model_name = self.embedding_cfg.get("model")
        self.dimension = self.embedding_cfg.get("dimension")
        if not self.model_name or not self.dimension:
            self.use_vector = False
            self._log_warning(
                "vector_config_incomplete",
                model=self.model_name,
                dimension=self.dimension,
            )

        self.normalize = self._as_bool(self.embedding_cfg.get("normalize", True), True)
        self.batch_size = int(self.embedding_cfg.get("batch_size", 32) or 32)
        instruction = self.embedding_cfg.get("instruction") or ""
        self.note_instruction = (
            self.embedding_cfg.get("note_instruction") or instruction or None
        )
        self.query_instruction = (
            self.embedding_cfg.get("query_instruction") or instruction or None
        )

        self.index_path = self._resolve_index_path(
            self.ann_cfg.get("index_path"), task_id
        )
        self.ann_params = self.ann_cfg.get("params") or {}
        self.vector_topk_raw = int(
            self.ann_cfg.get("topk_raw", max(self.vector_topk, 50))
            or max(self.vector_topk, 50)
        )

        self.indexer: Optional[VectorIndexer] = None
        self._vector_ready = False

        self._prepare_vector_index()
        self._prepare_bm25_corpus()

    # ------------------------------------------------------------------
    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        try:
            return bool(value)
        except Exception:
            return default

    def _log_debug(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.debug(event, **fields)

    def _log_info(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.info(event, **fields)

    def _log_warning(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.warning(event, **fields)

    def _log_error(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.error(event, **fields)

    # ------------------------------------------------------------------
    def _prepare_vector_index(self) -> None:
        if not self.use_vector:
            return
        try:
            dimension = int(self.dimension)
        except (TypeError, ValueError):
            self._log_warning(
                "vector_dimension_invalid", expected=self.dimension, task_id=self.task_id
            )
            self.use_vector = False
            return

        self.indexer = VectorIndexer(
            dimension,
            metric=self.ann_cfg.get("metric", "ip"),
            index_path=self.index_path,
            ann_params=self.ann_params,
        )

        fingerprint = self._notes_fingerprint(self.notes)
        loaded = False
        if self.index_path is not None:
            try:
                loaded = self.indexer.load(expected_version=fingerprint)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._log_warning(
                    "vector_index_load_failed",
                    error=str(exc),
                    index_path=str(self.index_path),
                )
        if loaded:
            self._log_debug(
                "vector_index_loaded",
                index_path=str(self.index_path),
                count=len(self.notes),
            )
        else:
            try:
                note_vectors = encode_notes(
                    self.notes,
                    model_name=self.model_name,
                    normalize=self.normalize,
                    batch_size=self.batch_size,
                    instruction=self.note_instruction,
                )
            except Exception as exc:  # pragma: no cover - encoding failure
                self._log_error("vector_note_encode_failed", error=str(exc))
                self.use_vector = False
                return

            if note_vectors.size == 0 or note_vectors.shape[1] != dimension:
                self._log_error(
                    "vector_dimension_mismatch",
                    expected=dimension,
                    actual=int(note_vectors.shape[1]) if note_vectors.size else 0,
                )
                self.use_vector = False
                return
            try:
                ids = []
                for idx, note in enumerate(self.notes):
                    note_id = note.get("id")
                    ids.append(str(note_id) if note_id is not None else f"note_{idx}")
                self.indexer.build(note_vectors, ids, note_version=fingerprint)
                if self.index_path is not None:
                    try:
                        self.indexer.persist()
                    except Exception as exc:  # pragma: no cover - persist failure
                        self._log_warning(
                            "vector_index_persist_failed",
                            error=str(exc),
                            index_path=str(self.index_path),
                        )
                self._log_debug(
                    "vector_index_built",
                    count=len(self.notes),
                    index_path=str(self.index_path) if self.index_path else None,
                )
            except Exception as exc:  # pragma: no cover - build failure
                self._log_error("vector_index_build_failed", error=str(exc))
                self.use_vector = False
                return

        self._vector_ready = bool(self.indexer and len(self.indexer) > 0)
        if not self._vector_ready:
            self._log_warning("vector_index_unavailable", index_path=str(self.index_path))

    # ------------------------------------------------------------------
    def _prepare_bm25_corpus(self) -> None:
        texts = [(note.get("text") or "") for note in self.notes]
        self._bm25_tokens = [self._tokenize(text) for text in texts]
        self._bm25_idf = self._idf(self._bm25_tokens)
        lengths = [len(tok) for tok in self._bm25_tokens]
        self._bm25_avgdl = sum(lengths) / max(1, len(lengths))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [
            token
            for token in "".join(
                ch.lower() if ch.isalnum() else " " for ch in (text or "")
            ).split()
            if token
        ]

    @staticmethod
    def _idf(corpus: Sequence[Sequence[str]]) -> Dict[str, float]:
        df: Dict[str, int] = {}
        N = len(corpus)
        for doc in corpus:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        if N == 0:
            return {}
        idf: Dict[str, float] = {}
        for token, freq in df.items():
            idf[token] = math.log((N + 1) / (freq + 0.5)) + 1.0
        return idf

    def _bm25_score(self, query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
        if not doc_tokens:
            return 0.0
        tf: Dict[str, int] = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        avgdl = self._bm25_avgdl or 1.0
        doc_len = len(doc_tokens)
        for token in set(query_tokens):
            idf = self._bm25_idf.get(token)
            if idf is None:
                continue
            tfw = tf.get(token, 0)
            if tfw == 0:
                continue
            denom = tfw + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_len / avgdl)
            score += idf * (tfw * (self.bm25_k1 + 1) / max(denom, 1e-9))
        return score

    # ------------------------------------------------------------------
    def retrieve(self, question: str, topk: int) -> HybridRetrievalResult:
        if not question:
            return HybridRetrievalResult(
                merged=[],
                support_ids=[],
                mode="empty",
                breakdown={"bm25_only": 0, "vector_only": 0, "both": 0},
                per_candidate={},
                components={"bm25": [], "vector": []},
            )

        fusion_cutoff = max(int(topk), self.fusion_topk)
        bm25_results = self._run_bm25(question, fusion_cutoff)
        vector_results = self._run_vector(question, fusion_cutoff)

        if not self.hybrid_enabled:
            if vector_results:
                bm25_results = []
            else:
                vector_results = []

        mode = self._determine_mode(bm25_results, vector_results)
        candidates = self._merge_candidates(bm25_results, vector_results)
        breakdown = self._source_breakdown(candidates.values())
        merged = self._rank_candidates(candidates, fusion_cutoff)
        support_ids = [note_id for note_id, _ in merged[:topk]]

        return HybridRetrievalResult(
            merged=merged,
            support_ids=support_ids,
            mode=mode,
            breakdown=breakdown,
            per_candidate=candidates,
            components={"bm25": bm25_results, "vector": vector_results},
        )

    def _determine_mode(
        self,
        bm25_results: Sequence[Tuple[str, float]],
        vector_results: Sequence[Tuple[str, float]],
    ) -> str:
        if bm25_results and vector_results:
            return "hybrid"
        if vector_results:
            return "vector"
        if bm25_results:
            return "bm25"
        return "empty"

    def _run_bm25(self, question: str, fusion_cutoff: int) -> List[Tuple[str, float]]:
        if not self.notes:
            return []
        q_tokens = self._tokenize(question)
        scores: List[Tuple[str, float]] = []
        target_k = min(len(self.notes), max(fusion_cutoff, self.bm25_topk))
        for note, doc_tokens in zip(self.notes, self._bm25_tokens):
            note_id = note.get("id")
            if note_id is None:
                continue
            score = self._bm25_score(q_tokens, doc_tokens)
            scores.append((str(note_id), score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:target_k]

    def _run_vector(self, question: str, fusion_cutoff: int) -> List[Tuple[str, float]]:
        if not self._vector_ready or not self.indexer or not self.use_vector:
            return []
        try:
            query_vec = encode_query(
                question,
                model_name=self.model_name,
                normalize=self.normalize,
                instruction=self.query_instruction,
            )
        except Exception as exc:  # pragma: no cover - encoding failure
            self._log_error("vector_query_encode_failed", error=str(exc))
            return []
        try:
            vector_topk = max(fusion_cutoff, self.vector_topk, self.vector_topk_raw)
            results = self.indexer.query(query_vec, vector_topk)
        except Exception as exc:  # pragma: no cover - query failure
            self._log_error("vector_query_failed", error=str(exc))
            return []
        return results[:vector_topk]

    def _merge_candidates(
        self,
        bm25_results: Sequence[Tuple[str, float]],
        vector_results: Sequence[Tuple[str, float]],
    ) -> Dict[str, Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        for rank, (note_id, score) in enumerate(bm25_results, start=1):
            entry = candidates.setdefault(
                note_id,
                {
                    "id": note_id,
                    "bm25_score": None,
                    "bm25_rank": None,
                    "vector_score": None,
                    "vector_rank": None,
                    "sources": set(),
                },
            )
            entry["bm25_score"] = float(score)
            entry["bm25_rank"] = rank
            entry["sources"].add("bm25")
        for rank, (note_id, score) in enumerate(vector_results, start=1):
            entry = candidates.setdefault(
                note_id,
                {
                    "id": note_id,
                    "bm25_score": None,
                    "bm25_rank": None,
                    "vector_score": None,
                    "vector_rank": None,
                    "sources": set(),
                },
            )
            entry["vector_score"] = float(score)
            entry["vector_rank"] = rank
            entry["sources"].add("vector")
        return candidates

    def _source_breakdown(self, entries: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        bm25_only = vector_only = both = 0
        for entry in entries:
            sources = entry.get("sources", set())
            if "bm25" in sources and "vector" in sources:
                both += 1
            elif "bm25" in sources:
                bm25_only += 1
            elif "vector" in sources:
                vector_only += 1
        return {
            "bm25_only": bm25_only,
            "vector_only": vector_only,
            "both": both,
        }

    def _rank_candidates(
        self,
        candidates: Dict[str, Dict[str, Any]],
        fusion_cutoff: int,
    ) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        bm25_scores = [
            entry["bm25_score"]
            for entry in candidates.values()
            if entry["bm25_score"] is not None
        ]
        vector_scores = [
            entry["vector_score"]
            for entry in candidates.values()
            if entry["vector_score"] is not None
        ]
        bm25_stats = self._score_stats(bm25_scores)
        vector_stats = self._score_stats(vector_scores)

        ranked: List[Tuple[str, float]] = []
        for note_id, entry in candidates.items():
            fused_score = self._fused_score(entry, bm25_stats, vector_stats)
            entry["fusion_score"] = fused_score
            entry["sources"] = sorted(entry.get("sources", []))
            ranked.append((note_id, fused_score))

        ranked.sort(key=lambda item: (-(item[1]), item[0]))
        return ranked[:fusion_cutoff]

    @staticmethod
    def _score_stats(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0}
        mean = float(sum(values) / len(values))
        variance = float(sum((v - mean) ** 2 for v in values) / max(len(values), 1))
        std = math.sqrt(variance)
        if std == 0.0:
            std = 1.0
        return {"mean": mean, "std": std}

    def _fused_score(
        self,
        entry: Dict[str, Any],
        bm25_stats: Dict[str, float],
        vector_stats: Dict[str, float],
    ) -> float:
        strategy = self.fusion_strategy
        if strategy in {"zscore", "z-score", "z_score", "zscore_linear", "z-score-linear"}:
            return self._zscore_fusion(entry, bm25_stats, vector_stats)
        if strategy in {"rrf", "reciprocal", "reciprocal_rank", "reciprocal-rank"}:
            return self._rrf_fusion(entry)
        # Default fallback
        return self._zscore_fusion(entry, bm25_stats, vector_stats)

    def _zscore_fusion(
        self,
        entry: Dict[str, Any],
        bm25_stats: Dict[str, float],
        vector_stats: Dict[str, float],
    ) -> float:
        score = 0.0
        if entry.get("bm25_score") is not None:
            z = (entry["bm25_score"] - bm25_stats["mean"]) / max(bm25_stats["std"], 1e-6)
            score += self.source_weights.get("bm25", 1.0) * z
        if entry.get("vector_score") is not None:
            z = (entry["vector_score"] - vector_stats["mean"]) / max(vector_stats["std"], 1e-6)
            score += self.source_weights.get("vector", 1.0) * z
        return score

    def _rrf_fusion(self, entry: Dict[str, Any]) -> float:
        score = 0.0
        rrf_k = max(self.rrf_k, 1.0)
        if entry.get("bm25_rank") is not None:
            score += self.source_weights.get("bm25", 1.0) / (rrf_k + entry["bm25_rank"] - 1)
        if entry.get("vector_rank") is not None:
            score += self.source_weights.get("vector", 1.0) / (rrf_k + entry["vector_rank"] - 1)
        return score

    # ------------------------------------------------------------------
    @staticmethod
    def _notes_fingerprint(notes: Sequence[Dict[str, Any]]) -> str:
        import hashlib

        sha1 = hashlib.sha1()
        for note in notes:
            sha1.update(str(note.get("id", "")).encode("utf-8"))
            sha1.update(b"\x1f")
            sha1.update((note.get("title", "") or "").encode("utf-8"))
            sha1.update(b"\x1f")
            sha1.update((note.get("text", "") or "").encode("utf-8"))
            sha1.update(b"\x1e")
        return sha1.hexdigest()

    @staticmethod
    def _safe_index_name(identifier: Any) -> str:
        text = str(identifier or "default")
        safe = re.sub(r"[^0-9a-zA-Z._-]", "_", text)
        return safe or "default"

    @classmethod
    def _resolve_index_path(cls, base_path: Optional[str], task_id: Any) -> Optional[Path]:
        if not base_path:
            return None
        base = Path(base_path)
        if base.suffix:
            return base
        return base / f"{cls._safe_index_name(task_id)}.faiss"

