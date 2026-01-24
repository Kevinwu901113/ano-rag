from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from relrag.prompt import render_prompt
from relrag.utils.text_builders import build_note_text_for_rank
from relrag.utils.llm_client import LLMChatClient
from relrag.utils.context_budget import (
    apply_load_shed,
    budget_rerank_prompt,
    log_budget_event,
)
from relrag.utils.llm_errors import ContextLengthError


class LLMReranker:
    PROMPT_NAME = "rerank.txt"

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        lm_cfg: Optional[Dict[str, Any]] = None,
        *,
        base_cfg: Optional[Dict[str, Any]] = None,
        run_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or {}
        self.base_cfg = base_cfg or {}
        runtime_cfg = self.base_cfg.get("runtime") if isinstance(self.base_cfg.get("runtime"), dict) else {}
        self.run_dir = run_dir or runtime_cfg.get("run_dir") or self.base_cfg.get("run_dir")
        self.enabled = bool(self.cfg.get("enabled", False))
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
        rerank_limits = self.base_cfg.get("rerank") if isinstance(self.base_cfg.get("rerank"), dict) else {}
        base_max_items = int(rerank_limits.get("max_candidates", 12))
        base_max_item_tokens = rerank_limits.get("max_candidate_tokens")
        try:
            base_max_item_tokens = int(base_max_item_tokens) if base_max_item_tokens is not None else None
        except (TypeError, ValueError):
            base_max_item_tokens = None
        requested_max_tokens = int(self.cfg.get("max_tokens", 64))
        for start in range(0, len(candidates), self.batch):
            chunk = candidates[start : start + self.batch]
            max_items_override = None
            max_item_tokens_override = None
            max_tokens_override = requested_max_tokens
            attempt = 0
            scores: Dict[int, Dict[str, Any]] = {}
            used_items: List[Dict[str, Any]] = []
            while True:
                budgeted = budget_rerank_prompt(
                    question,
                    chunk,
                    prompt_name=self.PROMPT_NAME,
                    cfg=self.base_cfg,
                    llm_cfg=self.llm_cfg,
                    requested_max_tokens=max_tokens_override,
                    max_items_override=max_items_override,
                    max_item_tokens_override=max_item_tokens_override,
                )
                used_items = budgeted.items
                if not used_items:
                    scores = self._fallback_scores(question, chunk)
                    used_items = chunk
                    break
                log_budget_event(
                    self.run_dir,
                    stage="rerank",
                    phase="pre",
                    report=budgeted.report,
                    attempt=attempt + 1,
                    extra={"chunk_size": len(chunk)},
                )
                try:
                    if not self.client:
                        raise RuntimeError("LLM reranker client not initialized")
                    response = self.client.chat(
                        budgeted.messages,
                        temperature=0.0,
                        max_tokens=budgeted.report.effective_max_tokens,
                        llm_profile="extract",
                    )
                    content = response.content
                    scores = self._parse_scores(content, len(used_items))
                    log_budget_event(
                        self.run_dir,
                        stage="rerank",
                        phase="post",
                        report=budgeted.report,
                        attempt=attempt + 1,
                        extra={"response_chars": len(content)},
                    )
                    break
                except ContextLengthError as exc:
                    if attempt >= 1:
                        raise
                    current_max_items = max_items_override or base_max_items
                    current_max_item_tokens = max_item_tokens_override or base_max_item_tokens
                    before = {
                        "max_items": current_max_items,
                        "max_item_tokens": current_max_item_tokens,
                        "requested_max_tokens": max_tokens_override,
                    }
                    max_items_override, max_item_tokens_override, max_tokens_override = apply_load_shed(
                        max_items=current_max_items,
                        max_item_tokens=current_max_item_tokens,
                        requested_max_tokens=max_tokens_override,
                    )
                    after = {
                        "max_items": max_items_override,
                        "max_item_tokens": max_item_tokens_override,
                        "requested_max_tokens": max_tokens_override,
                    }
                    log_budget_event(
                        self.run_dir,
                        stage="rerank",
                        phase="load_shed",
                        report=budgeted.report,
                        attempt=attempt + 1,
                        extra={
                            "reason": "context_len",
                            "error": str(exc)[:200],
                            "load_shed_before": before,
                            "load_shed_after": after,
                        },
                    )
                    attempt += 1
                except Exception as exc:  # noqa: PERF203
                    logger.warning("LLM rerank failed, fallback to lexical scores: {}", exc)
                    scores = self._fallback_scores(question, chunk)
                    used_items = chunk
                    break
            used_note_ids = {item.get("note_id") for item in used_items if item.get("note_id")}
            dropped_items = [item for item in chunk if item.get("note_id") not in used_note_ids]
            if dropped_items and not scores:
                scores = self._fallback_scores(question, dropped_items)
            fallback_scores = self._fallback_scores(question, dropped_items) if dropped_items else {}
            for idx, item in enumerate(used_items, start=1):
                note_id = item.get("note_id")
                if not note_id:
                    continue
                entry = scores.get(idx, {"score": 0.0, "labels": []})
                results[note_id] = {
                    "score": float(entry.get("score", 0.0)),
                    "labels": entry.get("labels") or [],
                }
            if dropped_items:
                for idx, item in enumerate(dropped_items, start=1):
                    note_id = item.get("note_id")
                    if not note_id or note_id in results:
                        continue
                    entry = fallback_scores.get(idx, {"score": 0.0, "labels": ["fallback"]})
                    results[note_id] = {
                        "score": float(entry.get("score", 0.0)),
                        "labels": entry.get("labels") or ["fallback"],
                    }
        return results

    def _build_payload(self, question: str, chunk: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for idx, candidate in enumerate(chunk, start=1):
            note = candidate.get("note") or {}
            text = build_note_text_for_rank(note, max_len=768)
            lines.append(f"{idx}. note_id={note.get('note_id')} :: {text}")
        return render_prompt(self.PROMPT_NAME, question=question, candidates="\n".join(lines))

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
