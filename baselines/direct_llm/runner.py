from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from loguru import logger

from config import config as config_loader

DEFAULT_SYSTEM_PROMPT = (
    "You are a factual question answering assistant. Use only your own knowledge to answer. "
    'Return ONLY one concise noun phrase in English (e.g., "American lawyer"). '
    "Do NOT include any reasoning, explanation, analysis, apologies, or restating of the question. "
    'If you do not know, reply with exactly "Insufficient evidence".'
)
DEFAULT_USER_TEMPLATE = (
    "Question: {question}\n"
    "Respond with a single short noun phrase in English. Do not add any other words or sentences."
)


@dataclass
class DirectLLMResult:
    query_id: str
    question: str
    answer: str
    meta: Dict[str, Any]
    evidence: List[Any]


class DirectLLMClient:
    """Minimal chat client for direct LLM baseline (no retrieval)."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 32,
        stop: Optional[List[str]] = None,
        retries: int = 2,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("Both endpoint and model are required for direct LLM baseline")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Default: cut at first newline to avoid long chatter.
        self.stop = ["\n"] if stop is None else stop
        self.retries = max(0, retries)

    def answer(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return "Insufficient evidence"
        user_msg = DEFAULT_USER_TEMPLATE.format(question=q)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if self.stop:
            payload["stop"] = self.stop

        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(f"{self.endpoint}/chat/completions", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _strip_reasoning(content)
            except requests.RequestException as exc:  # noqa: PERF203
                if attempt >= self.retries:
                    logger.error("Direct LLM call failed after {} attempts: {}", attempt + 1, exc)
                    return "Insufficient evidence"
                backoff = 2**attempt
                logger.warning("Direct LLM call failed (attempt {}): {}; retrying in {}s", attempt + 1, exc, backoff)
                time.sleep(backoff)
        return "Insufficient evidence"


class DirectLLMRunner:
    """Run direct LLM baseline on a dataset (no retrieval or indexes)."""

    def __init__(
        self,
        *,
        lm_endpoint: Optional[str] = None,
        lm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        lm_cfg = self.cfg.get("lmstudio", {}) or {}
        endpoint = lm_endpoint or lm_cfg.get("endpoint")
        model = lm_model or lm_cfg.get("model")
        temp = temperature if temperature is not None else 0.0
        max_new_tokens = max_tokens if max_tokens is not None else 32
        self.client = DirectLLMClient(
            endpoint=endpoint,
            model=model,
            system_prompt=system_prompt,
            temperature=float(temp or 0.0),
            max_tokens=int(max_new_tokens or 32),
            stop=stop,
        )

    def run_dataset(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        work_dir: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        items = list(dataset)
        if limit:
            items = items[:limit]

        out_root = Path(work_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        answers_json_path = out_root / "answers.json"
        answers_direct_path = out_root / "answers_direct_llm.json"
        answers_jsonl_path = out_root / "answers_direct_llm.jsonl"
        qa_path = out_root / "qa.tsv"
        qa_no_header_path = out_root / "qa.no_header.tsv"
        qa_with_question_path = out_root / "qa_with_question.tsv"

        results: List[DirectLLMResult] = []
        qa_lines: List[str] = []
        qa_no_header: List[str] = []
        qa_with_q: List[str] = []
        jsonl_lines: List[str] = []

        for idx, item in enumerate(items):
            question = str(item.get("query") or item.get("question") or "").strip()
            qid = str(item.get("query_id") or item.get("id") or idx)
            answer = self.client.answer(question)
            result = DirectLLMResult(
                query_id=qid,
                question=question,
                answer=answer,
                evidence=[],
                meta={
                    "mode": "direct_llm",
                    "model": self.client.model,
                    "temperature": self.client.temperature,
                },
            )
            results.append(result)
            qa_lines.append(f"{question}\t{answer}")
            qa_no_header.append(f"{question}\t{answer}")
            qa_with_q.append(f"{question}\t{answer}")
            jsonl_lines.append(
                json.dumps(
                    {
                        "query_id": result.query_id,
                        "question": result.question,
                        "answer": result.answer,
                        "evidence": result.evidence,
                        "meta": result.meta,
                    },
                    ensure_ascii=False,
                )
            )

        payload = [
            {
                "query_id": r.query_id,
                "question": r.question,
                "answer": r.answer,
                "evidence": r.evidence,
                "meta": r.meta,
            }
            for r in results
        ]

        answers_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        answers_direct_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        answers_jsonl_path.write_text("\n".join(jsonl_lines), encoding="utf-8")
        qa_path.write_text("\n".join(qa_lines), encoding="utf-8")
        qa_no_header_path.write_text("\n".join(qa_no_header), encoding="utf-8")
        qa_with_question_path.write_text("\n".join(qa_with_q), encoding="utf-8")

        logger.info("Direct LLM run complete: {} examples -> {}", len(results), qa_path)
        return {
            "answers_json": str(answers_json_path),
            "answers_jsonl": str(answers_jsonl_path),
            "answers_direct_json": str(answers_direct_path),
            "qa": str(qa_path),
            "qa_no_header": str(qa_no_header_path),
            "qa_with_question": str(qa_with_question_path),
        }


def _strip_reasoning(text: str) -> str:
    output = text or ""
    while True:
        start = output.find("<think>")
        if start == -1:
            break
        end = output.find("</think>", start + len("<think>"))
        if end == -1:
            output = output[:start] + output[start + len("<think>") :]
            break
        output = output[:start] + output[end + len("</think>") :]
    cleaned = output.strip()
    return cleaned or "Insufficient evidence"
