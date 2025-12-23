from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from utils.text_builders import build_note_text_for_rank
from utils.llm_client import LLMChatClient


class LLMReranker:
    PROMPT_TEMPLATE = """You are re-ranking notes for answering a question.
Question: {question}
For each candidate note, provide a confidence score between 0 and 100 indicating how well it helps answer the question.
Return strict JSON list: [{{"idx": <int>, "score": <0-100>, "labels": ["pred_hit","alias_hit","pronoun_risk","year_hit"]}}, ...]
Candidates:
{candidates}
"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, lm_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.type = self.cfg.get("type", "llm")
        self.llm_cfg = self.cfg.get("llm") or lm_cfg or {}
        self.endpoint = self.llm_cfg.get("endpoint")
        self.model = self.llm_cfg.get("model")
        self.batch = int(self.llm_cfg.get("batch", 8))
        self.timeout = int(self.llm_cfg.get("timeout_s", 10))
        self.client: Optional[LLMChatClient] = None
        if self.type != "llm" or not self.endpoint or not self.model:
            self.enabled = False
        else:
            self.client = LLMChatClient(
                endpoint=self.endpoint,
                model=self.model,
                llm_profile="extract",
                timeout=self.timeout,
            )

    def score(self, question: str, candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not self.enabled or not candidates:
            return {}
        results: Dict[str, Dict[str, Any]] = {}
        for start in range(0, len(candidates), self.batch):
            chunk = candidates[start : start + self.batch]
            payload = self._build_payload(question, chunk)
            try:
                if not self.client:
                    raise RuntimeError("LLM reranker client not initialized")
                response = self.client.chat(
                    [{"role": "user", "content": payload}],
                    temperature=0.0,
                    max_tokens=64,
                    llm_profile="extract",
                )
                content = response.content
                scores = self._parse_scores(content, len(chunk))
            except Exception as exc:  # noqa: PERF203
                logger.warning("LLM rerank failed, fallback to lexical scores: {}", exc)
                scores = self._fallback_scores(question, chunk)
            for idx, item in enumerate(chunk, start=1):
                note_id = item.get("note_id")
                if not note_id:
                    continue
                entry = scores.get(idx, {"score": 0.0, "labels": []})
                results[note_id] = {
                    "score": float(entry.get("score", 0.0)),
                    "labels": entry.get("labels") or [],
                }
        return results

    def _build_payload(self, question: str, chunk: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for idx, candidate in enumerate(chunk, start=1):
            note = candidate.get("note") or {}
            text = build_note_text_for_rank(note, max_len=768)
            lines.append(f"{idx}. note_id={note.get('note_id')} :: {text}")
        return self.PROMPT_TEMPLATE.format(question=question, candidates="\n".join(lines))

    def _parse_scores(self, content: str, expected: int) -> Dict[int, Dict[str, Any]]:
        text = content.strip()
        if not text.startswith("["):
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        parsed: Dict[int, Dict[str, Any]] = {}
        if not isinstance(data, list):
            return parsed
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = int(item.get("idx", 0))
            if idx <= 0 or idx > expected:
                continue
            score = float(item.get("score", 0.0))
            labels = item.get("labels") or []
            if isinstance(labels, str):
                labels = [token.strip() for token in labels.split(",") if token.strip()]
            parsed[idx] = {"score": score, "labels": labels}
        return parsed

    def _fallback_scores(self, question: str, chunk: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        q_tokens = set(question.lower().split())
        scores: Dict[int, Dict[str, Any]] = {}
        for idx, candidate in enumerate(chunk, start=1):
            note = candidate.get("note") or {}
            text = build_note_text_for_rank(note, max_len=256).lower()
            overlap = len(q_tokens & set(text.split()))
            denom = max(len(q_tokens), 1)
            score = min(100.0, 100.0 * overlap / denom)
            scores[idx] = {"score": score, "labels": ["fallback"]}
        return scores
