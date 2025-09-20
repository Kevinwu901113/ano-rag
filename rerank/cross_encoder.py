"""Cross-encoder based reranker."""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency may be missing
    CrossEncoder = None  # type: ignore

try:  # Optional import used for structured logging
    from utils.logging import StructuredLogger
except Exception:  # pragma: no cover - logging is optional for unit tests
    StructuredLogger = None  # type: ignore

CandidateType = Union[Dict[str, Any], Sequence[Any], Any]


class CrossEncoderReranker:
    """Rerank retrieval candidates with a cross-encoder model."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        rerank_cfg = (config or {}).get("rerank") or {}
        self.logger = logger
        self.enabled = self._as_bool(rerank_cfg.get("enabled", False), False)
        self.model_name = rerank_cfg.get("model")
        self.top_n = int(rerank_cfg.get("topN", 50) or 50)
        self.top_k = int(rerank_cfg.get("topK", 5) or 5)
        self.max_len = int(rerank_cfg.get("max_len", 512) or 512)
        self.device = rerank_cfg.get("device")
        self.batch_size = int(rerank_cfg.get("batch_size", 16) or 16)

        if self.top_n < 1:
            self.top_n = 1
        if self.top_k < 1:
            self.top_k = 1
        if self.top_n < self.top_k:
            self.top_n = self.top_k

        if not self.model_name:
            self._log_warning("rerank_model_not_configured")
            self.enabled = False

        if CrossEncoder is None and self.enabled:
            self._log_warning("rerank_cross_encoder_unavailable", model=self.model_name)
            self.enabled = False

        self._model: Optional[CrossEncoder] = None  # type: ignore[assignment]
        self._lock = threading.Lock()

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

    # ------------------------------------------------------------------
    @property
    def is_enabled(self) -> bool:
        return self.enabled

    # ------------------------------------------------------------------
    def _log_warning(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.warning(event, **fields)

    def _log_error(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.error(event, **fields)

    # ------------------------------------------------------------------
    def _ensure_model(self) -> Optional[CrossEncoder]:  # type: ignore[override]
        if not self.enabled or CrossEncoder is None:
            return None
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._log_error("rerank_model_load_failed", error=str(exc), model=self.model_name)
                self.enabled = False
                self._model = None
            return self._model

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_candidates(candidates: Sequence[CandidateType]) -> List[Tuple[str, Optional[float]]]:
        normalised: List[Tuple[str, Optional[float]]] = []
        for candidate in candidates:
            cand_id: Optional[Any] = None
            score: Optional[float] = None
            raw_score: Any = None
            if isinstance(candidate, dict):
                cand_id = candidate.get("id") or candidate.get("candidate_id") or candidate.get("doc_id")
                raw_score = candidate.get("score") or candidate.get("weight")
            elif isinstance(candidate, (list, tuple)) and candidate:
                cand_id = candidate[0]
                if len(candidate) > 1:
                    raw_score = candidate[1]
            else:
                cand_id = candidate
            if cand_id is None:
                continue
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)
            else:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = None
            normalised.append((str(cand_id), score))
        return normalised

    @staticmethod
    def _note_lookup(notes: Union[Sequence[Dict[str, Any]], Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        if isinstance(notes, dict):
            return {str(k): v for k, v in notes.items()}
        lookup: Dict[str, Dict[str, Any]] = {}
        for note in notes or []:
            if not isinstance(note, dict):
                continue
            note_id = note.get("id")
            if note_id is None:
                continue
            lookup[str(note_id)] = note
        return lookup

    @staticmethod
    def _note_text(note: Dict[str, Any]) -> str:
        title = (note.get("title") or "").strip()
        body = (note.get("text") or "").strip()
        if title and body:
            return f"{title}\n\n{body}"
        if title:
            return title
        return body

    @staticmethod
    def _fallback_score(score: Optional[float], default: float = 0.0) -> float:
        if isinstance(score, (int, float)):
            return float(score)
        return float(default)

    def _resolve_top_k(self, override: Optional[int]) -> Optional[int]:
        value = override if override is not None else self.top_k
        if value is None:
            return None
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return self.top_k
        if value_int <= 0:
            return None
        return value_int

    # ------------------------------------------------------------------
    def rerank(
        self,
        question: str,
        candidates: Sequence[CandidateType],
        notes: Union[Sequence[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        *,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank candidates and return the top-K list."""

        normalised = self._normalise_candidates(candidates)
        if not normalised or not question:
            resolved_topk = self._resolve_top_k(top_k)
            if resolved_topk is None:
                return [(cand_id, self._fallback_score(score)) for cand_id, score in normalised]
            return [
                (cand_id, self._fallback_score(score))
                for cand_id, score in normalised[:resolved_topk]
            ]

        resolved_topn = min(len(normalised), max(self.top_n, 1))
        resolved_topk = self._resolve_top_k(top_k)
        top_candidates = normalised[:resolved_topn]

        if not self.enabled:
            fallback = [
                (cand_id, self._fallback_score(score))
                for cand_id, score in top_candidates
            ]
            if resolved_topk is None:
                return fallback
            return fallback[:resolved_topk]

        model = self._ensure_model()
        if model is None:
            fallback = [
                (cand_id, self._fallback_score(score))
                for cand_id, score in top_candidates
            ]
            if resolved_topk is None:
                return fallback
            return fallback[:resolved_topk]

        lookup = self._note_lookup(notes)
        sentence_pairs: List[Tuple[str, str]] = []
        kept: List[Tuple[str, Optional[float]]] = []
        for cand_id, score in top_candidates:
            note = lookup.get(cand_id)
            if note is None:
                continue
            text = self._note_text(note)
            if not text:
                continue
            sentence_pairs.append((question, text))
            kept.append((cand_id, score))

        if not sentence_pairs:
            fallback = [
                (cand_id, self._fallback_score(score))
                for cand_id, score in top_candidates
            ]
            if resolved_topk is None:
                return fallback
            return fallback[:resolved_topk]

        try:
            scores = model.predict(
                sentence_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                max_length=self.max_len,
            )
        except Exception as exc:  # pragma: no cover - inference failure
            self._log_error("rerank_inference_failed", error=str(exc))
            fallback = [
                (cand_id, self._fallback_score(score))
                for cand_id, score in top_candidates
            ]
            if resolved_topk is None:
                return fallback
            return fallback[:resolved_topk]

        reranked: List[Tuple[str, float]] = []
        for (cand_id, _), score in zip(kept, scores):
            reranked.append((cand_id, float(score)))
        reranked.sort(key=lambda item: item[1], reverse=True)

        if resolved_topk is not None:
            reranked = reranked[:resolved_topk]

        seen = {cand_id for cand_id, _ in reranked}
        if resolved_topk is None:
            target_len = len(top_candidates)
        else:
            target_len = resolved_topk
        if target_len is None:
            target_len = len(top_candidates)
        if len(reranked) < target_len:
            for cand_id, score in normalised:
                if cand_id in seen:
                    continue
                reranked.append((cand_id, self._fallback_score(score)))
                seen.add(cand_id)
                if len(reranked) >= target_len:
                    break

        return reranked
