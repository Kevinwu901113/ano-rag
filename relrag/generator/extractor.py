from __future__ import annotations

import json
from typing import Any, Dict, List

from loguru import logger

from relrag.utils.text_builders import build_note_text_for_rank
from relrag.utils.llm_client import LLMChatClient

EXTRACT_PROMPT = """You extract relevant evidence for answering a question.
Question: {question}
Evidence note:
{note_text}
Respond with strict JSON: {{"keep": true/false, "summary": "<compressed sentences>", "labels": ["reason_tag"]}}
If unsure, set keep to false.
"""


class EvidenceExtractor:
    def __init__(self, endpoint: str, model: str, timeout: int = 15) -> None:
        self.client = LLMChatClient(
            endpoint=endpoint,
            model=model,
            llm_profile="extract",
            timeout=timeout,
        )

    def judge_and_compress(self, question: str, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for note in notes:
            text = build_note_text_for_rank(note, max_len=768)
            payload = EXTRACT_PROMPT.format(question=question, note_text=text)
            try:
                response = self.client.chat(
                    [{"role": "user", "content": payload}],
                    temperature=0.0,
                    max_tokens=128,
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
    extractor = EvidenceExtractor(endpoint, model, timeout=int(llm_cfg.get("timeout_s", 15)))
    return extractor.judge_and_compress(question, notes)
