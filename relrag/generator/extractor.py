from __future__ import annotations

import json
from typing import Any, Dict, List

from loguru import logger

from relrag.prompt import render_prompt
from relrag.utils.text_builders import build_note_text_for_rank
from relrag.utils.llm_client import LLMChatClient

EXTRACT_PROMPT_NAME = "extractor.txt"


class EvidenceExtractor:
    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: int = 60,
        retries: int = 2,
        max_tokens: int = 128,
    ) -> None:
        self.max_tokens = max(16, int(max_tokens))
        self.client = LLMChatClient(
            endpoint=endpoint,
            model=model,
            llm_profile="extract",
            timeout=timeout,
            retries=max(0, int(retries)),
        )

    def judge_and_compress(self, question: str, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for note in notes:
            text = build_note_text_for_rank(note, max_len=768)
            payload = render_prompt(EXTRACT_PROMPT_NAME, question=question, note_text=text)
            try:
                response = self.client.chat(
                    [{"role": "user", "content": payload}],
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    llm_profile="extract",
                )
                content = response.content
                parsed = self._parse_response(content)
            except Exception as exc:  # noqa: PERF203
                logger.warning("Evidence extractor failed for note {}: {}", note.get("note_id"), exc)
                parsed = {"keep": True, "summary": note.get("evidence", ""), "labels": ["fallback"]}
            if not parsed.get("keep"):
                continue
            summary = parsed.get("summary") or note.get("evidence", "")
            results.append(
                {
                    "note_id": note.get("note_id"),
                    "summary": summary.strip(),
                    "labels": parsed.get("labels") or [],
                }
            )
        return results

    @staticmethod
    def _parse_response(content: str) -> Dict[str, Any]:
        text = content.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        keep = bool(data.get("keep"))
        summary = data.get("summary")
        labels = data.get("labels") or []
        if isinstance(labels, str):
            labels = [token.strip() for token in labels.split(",") if token.strip()]
        return {"keep": keep, "summary": summary, "labels": labels}


def judge_and_compress(question: str, notes: List[Dict[str, Any]], llm_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    endpoint = llm_cfg.get("endpoint")
    model = llm_cfg.get("model")
    if not endpoint or not model:
        logger.warning("LLM config missing for evidence extractor; returning original notes.")
        outputs = []
        for note in notes:
            outputs.append(
                {"note_id": note.get("note_id"), "summary": note.get("evidence", ""), "labels": ["no_llm"]}
            )
        return outputs
    timeout_s = llm_cfg.get("timeout_s", 60)
    retries = llm_cfg.get("retries", 2)
    max_tokens = llm_cfg.get("max_tokens", 128)
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = 60
    try:
        retries = int(retries)
    except (TypeError, ValueError):
        retries = 2
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 128
    extractor = EvidenceExtractor(
        endpoint,
        model,
        timeout=max(5, timeout_s),
        retries=max(0, retries),
        max_tokens=max(16, max_tokens),
    )
    return extractor.judge_and_compress(question, notes)
