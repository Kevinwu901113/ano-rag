import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


CandidateType = Union[Dict[str, Any], Sequence[Any]]


class StructuredLogger:
    """Lightweight structured logger with JSON-line support."""

    _LEVELS: Dict[str, int] = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    def __init__(
        self,
        name: str = "ano_rag",
        level: str = "INFO",
        json_lines: bool = True,
        stream = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.json_lines = json_lines
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        # Reset handlers so re-instantiation does not duplicate outputs
        if self.logger.handlers:
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)
        numeric_level = self._LEVELS.get(level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        if self.enabled:
            handler = logging.StreamHandler(stream or sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(handler)

    def _serialize(self, payload: Dict[str, Any]) -> str:
        if self.json_lines:
            safe_payload = {k: self._make_json_safe(v) for k, v in payload.items() if v is not None}
            return json.dumps(safe_payload, ensure_ascii=False)
        components: List[str] = []
        for key, value in payload.items():
            if value is None:
                continue
            components.append(f"{key}={self._stringify(value)}")
        return " | ".join(components)

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [StructuredLogger._make_json_safe(v) for v in value]
        if isinstance(value, dict):
            return {k: StructuredLogger._make_json_safe(v) for k, v in value.items()}
        return str(value)

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)
        return str(value)

    def log(self, level: str, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        numeric_level = self._LEVELS.get(level.upper(), self.logger.level)
        payload = {"event": event, **fields}
        message = self._serialize(payload)
        self.logger.log(numeric_level, message)

    def debug(self, event: str, **fields: Any) -> None:
        self.log("DEBUG", event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self.log("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self.log("WARNING", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.log("ERROR", event, **fields)

    @contextmanager
    def time_stage(self, stage: str, query_id: Optional[str] = None, **metadata: Any):
        """Context manager logging before/after timing for a stage."""
        info: Dict[str, Any] = {"elapsed_ms": None}
        if not self.enabled:
            yield info
            return

        context_payload = {"stage": stage, "query_id": query_id, **metadata}
        start = time.perf_counter()
        self.debug(f"{stage}_before", **context_payload)
        try:
            yield info
        except Exception as exc:  # pragma: no cover - defensive logging
            info["elapsed_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
            self.error(
                f"{stage}_error",
                error=str(exc),
                elapsed_ms=info["elapsed_ms"],
                **context_payload,
            )
            raise
        else:
            info["elapsed_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
            self.info(
                f"{stage}_after",
                elapsed_ms=info["elapsed_ms"],
                **context_payload,
            )

    def log_candidates(
        self,
        stage: str,
        query_id: Optional[str],
        candidates: Optional[Iterable[CandidateType]],
        *,
        latency_ms: Optional[float] = None,
        limit: int = 20,
        query: Optional[str] = None,
    ) -> None:
        if not self.enabled or candidates is None:
            return

        serialised: List[Dict[str, Any]] = []
        for rank, cand in enumerate(candidates, start=1):
            if rank > limit:
                break
            cand_id, score = self._candidate_fields(cand)
            serialised.append({"rank": rank, "id": cand_id, "score": score})
        payload: Dict[str, Any] = {
            "stage": stage,
            "query_id": query_id,
            "latency_ms": latency_ms,
            "candidates": serialised,
        }
        if query:
            preview = query if len(query) <= 200 else f"{query[:197]}..."
            payload["query_preview"] = preview
        self.info("candidate_trace", **payload)

    @staticmethod
    def _candidate_fields(candidate: CandidateType) -> Tuple[Any, Optional[float]]:
        cand_id: Any = None
        score: Optional[float] = None
        if isinstance(candidate, dict):
            cand_id = candidate.get("id") or candidate.get("candidate_id") or candidate.get("doc_id")
            raw_score = candidate.get("score") or candidate.get("weight")
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)
            else:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = None
        elif isinstance(candidate, (list, tuple)) and candidate:
            cand_id = candidate[0]
            if len(candidate) > 1:
                raw_score = candidate[1]
                if isinstance(raw_score, (int, float)):
                    score = float(raw_score)
                else:
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        score = None
        else:
            cand_id = candidate
        return cand_id, score
