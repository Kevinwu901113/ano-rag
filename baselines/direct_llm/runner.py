from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from loguru import logger

from config import config as config_loader
from utils.jsonl_utils import write_jsonl
from utils.retrieval_logger import log_retrieval
from utils.output_protocol import build_final_instruction

DEFAULT_SYSTEM_PROMPT = (
    "You are a factual question answering assistant. Use only your own knowledge to answer. "
    'If you do not know, reply with exactly "Insufficient evidence".'
)
DEFAULT_USER_TEMPLATE = (
    "Question: {question}\n"
    "{final_instruction}\n"
    "Respond with a concise answer. You may include reasoning, but the FINAL line must follow the protocol."
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
        self._mock = str(endpoint).strip().lower() == "mock"
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.retries = max(0, retries)

    def answer(self, question: str) -> str:
        if self._mock:
            return "Mock Answer"
        q = (question or "").strip()
        if not q:
            return "Insufficient evidence"
        user_msg = DEFAULT_USER_TEMPLATE.format(question=q, final_instruction=build_final_instruction())
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
                return content
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
        
        # Ensure endpoint and model are strings
        endpoint = str(endpoint) if endpoint else None
        model = str(model) if model else None
        
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
        context_budget_tokens: Optional[int] = None,
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        items = list(dataset)
        if limit:
            items = items[:limit]

        out_root = Path(work_dir)
        preds_dir = out_root / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)
        answers_json_path = preds_dir / "answers.json"
        answers_direct_path = preds_dir / "answers_direct_llm.json"
        answers_jsonl_path = preds_dir / "answers_direct_llm.jsonl"
        pred_raw_path = preds_dir / "pred_raw.jsonl"
        qa_path = preds_dir / "qa.tsv"
        qa_no_header_path = preds_dir / "qa.no_header.tsv"
        qa_with_question_path = preds_dir / "qa_with_question.tsv"

        results: List[DirectLLMResult] = []
        qa_lines: List[str] = []
        qa_no_header: List[str] = []
        qa_with_q: List[str] = []
        jsonl_lines: List[str] = []
        pred_raw_records: List[Dict[str, Any]] = []

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
            pred_raw_records.append(
                {
                    "id": result.query_id,
                    "question": result.question,
                    "pred_raw": result.answer,
                    "contexts_used": [],
                    "context_tokens_used": 0,
                    "context_budget_tokens": (
                        int(context_budget_tokens) if context_budget_tokens is not None else None
                    ),
                }
            )
            if log_dir and dataset_name and run_name:
                log_retrieval(
                    sample_id=result.query_id,
                    dataset=dataset_name,
                    run_name=run_name,
                    retrieved=[],
                    topk=0,
                    final_context=[],
                    final_context_tokens=0,
                    context_budget_tokens=context_budget_tokens,
                    log_dir=Path(log_dir),
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
        write_jsonl(pred_raw_path, pred_raw_records)
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
