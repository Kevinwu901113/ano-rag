from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from loguru import logger

from baselines.naive_rag import NaiveIndex
from config import config as config_loader

def _enforce_short_answer(text: str) -> str:
    """Helper to truncate or normalize short answer."""
    # Implement simple logic or import from somewhere else if needed
    # For now, just basic cleanup
    return text.strip()

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
    return output.strip()


FID_PROMPT_TEMPLATE = """You are a question answering system.
Use ONLY the information from the passages below to answer the question.
If the passages are insufficient, reply EXACTLY with: Insufficient evidence.
Respond with EXACTLY TWO lines and nothing else.
  Line 1: ONLY the final answer as one short noun phrase (no quotes, no punctuation, no analysis).
  Line 2: Passages used: i1, i2, ... (list the passage indices you relied on).
Do NOT include any reasoning, explanation, bullet points, or extra lines.

Question:
{question}

Passages:
{passages}

First, output the two required lines.
"""


class LLMClient:
    """Minimal LM Studio/OpenAI-compatible chat client for FiD prompting."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        stop: Optional[List[str]] = None,
        retries: int = 2,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("Both endpoint and model are required for LLM calls")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = [] if stop is None else stop
        self.retries = max(0, retries)

    def answer_raw_prompt(self, prompt: str) -> str:
        if not prompt.strip():
            return "Insufficient evidence"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.stop:
            payload["stop"] = self.stop
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(f"{self.endpoint}/chat/completions", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                
                # Handle <think> blocks
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content.startswith("<think>"):
                    content = re.sub(r"^<think>.*", "", content, flags=re.DOTALL).strip()
                    
                cleaned = _strip_reasoning(content)
                return cleaned or "Insufficient evidence"
            except requests.RequestException as exc:  # noqa: PERF203
                if attempt >= self.retries:
                    logger.error("LLM call failed after {} attempts: {}", attempt + 1, exc)
                    return "Insufficient evidence"
                backoff = 2**attempt
                logger.warning("LLM call failed (attempt {}): {}; retrying in {}s", attempt + 1, exc, backoff)
                time.sleep(backoff)
        return "Insufficient evidence"


class FiDRAGRunner:
    """FiD-style RAG baseline with NaiveIndex retriever and LM Studio backend."""

    def __init__(
        self,
        index_path: str,
        chunks_path: str,
        *,
        topk: int = 5,
        lm_endpoint: Optional[str] = None,
        lm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        self.topk = max(1, int(topk))
        lm_cfg = self.cfg.get("lmstudio", {}) or {}
        endpoint = lm_endpoint or lm_cfg.get("endpoint")
        model = lm_model or lm_cfg.get("model")
        
        # Ensure endpoint and model are strings
        endpoint = str(endpoint) if endpoint else None
        model = str(model) if model else None
        
        temp = temperature if temperature is not None else lm_cfg.get("temperature", 0.0)
        max_new_tokens = max_tokens if max_tokens is not None else lm_cfg.get("max_tokens", 128)
        self.retriever = NaiveIndex(index_path, chunks_path, config=self.cfg)
        self.lm = LLMClient(
            endpoint,
            model,
            temperature=float(temp or 0.0),
            max_tokens=int(max_new_tokens or 128),
            stop=stop,
        )

    def run_dataset(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        work_dir: str,
        limit: Optional[int] = None,
        debug: bool = True,
    ) -> Dict[str, Any]:
        items = list(dataset)
        if limit:
            items = items[:limit]
        answers: List[Dict[str, Any]] = []
        qa_rows: List[str] = []
        qa_rows_no_header: List[str] = []
        qa_with_q: List[str] = []
        debug_records: List[str] = []
        for idx, item in enumerate(items):
            question = item.get("query") or item.get("question") or ""
            qid = item.get("query_id") or str(idx)
            hits = self.retriever.search(question, self.topk)
            context = _format_fid_context(hits)
            if not context.strip():
                raw_output = "Insufficient evidence"
            else:
                prompt = _build_fid_prompt(question, hits)
                raw_output = self.lm.answer_raw_prompt(prompt)
            answer_text, used_indices = _parse_fid_output(raw_output)
            used_indices = _filter_indices(used_indices, len(hits))
            answer_text = _enforce_short_answer(answer_text)
            answers.append(
                {
                    "query_id": qid,
                    "question": question,
                    "answer": answer_text,
                    "hits": hits,
                    "passages_used": used_indices,
                }
            )
            qa_rows.append(f"{question}\t{answer_text}")
            qa_rows_no_header.append(f"{question}\t{answer_text}")
            qa_with_q.append(f"{question}\t{answer_text}")
            if debug:
                debug_records.append(
                    json.dumps(
                        {
                            "query_id": qid,
                            "question": question,
                            "answer": answer_text,
                            "passages_used": used_indices,
                            "hits": hits,
                        },
                        ensure_ascii=False,
                    )
                )

        out_root = Path(work_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        answers_path = out_root / "answers.json"
        qa_path = out_root / "qa.tsv"
        qa_no_header_path = out_root / "qa.no_header.tsv"
        qa_q_path = out_root / "qa_with_question.tsv"
        debug_dir = out_root / "debug"
        answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
        qa_path.write_text("\n".join(qa_rows), encoding="utf-8")
        qa_no_header_path.write_text("\n".join(qa_rows_no_header), encoding="utf-8")
        qa_q_path.write_text("\n".join(qa_with_q), encoding="utf-8")
        if debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "retrieval.jsonl").write_text("\n".join(debug_records), encoding="utf-8")
        logger.info("FiD RAG run complete: {} examples -> {}", len(answers), qa_path)
        return {
            "answers": str(answers_path),
            "qa": str(qa_path),
            "qa_no_header": str(qa_no_header_path),
            "qa_with_question": str(qa_q_path),
            "debug": str(debug_dir) if debug else None,
        }


def _format_fid_context(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, hit in enumerate(hits, start=1):
        text = str(hit.get("text") or "")
        lines.append(f"[{idx}] {text}")
    return "\n\n".join(lines)


def _build_fid_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    context = _format_fid_context(hits)
    safe_context = context.replace("{", "{{").replace("}", "}}")
    safe_question = question.replace("{", "{{").replace("}", "}}")
    return FID_PROMPT_TEMPLATE.format(question=safe_question, passages=safe_context)


def _parse_fid_output(text: str) -> Tuple[str, List[int]]:
    """
    Parse LLM raw output and return (answer_text, used_indices).
    The first non-empty line is treated as the answer. A later line starting with
    "Passages used:" is parsed for referenced indices.
    """

    if not text:
        return "Insufficient evidence", []
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return "Insufficient evidence", []
    answer_text = lines[0]
    used_indices: List[int] = []
    answer_before_used: Optional[str] = None
    for idx, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith("passages used"):
            if idx > 0:
                answer_before_used = lines[idx - 1]
            _, _, suffix = line.partition(":")
            candidates = suffix.replace(",", " ").split()
            for token in candidates:
                if token.isdigit():
                    used_indices.append(int(token))
            break
    if answer_before_used:
        answer_text = answer_before_used
    return answer_text, used_indices


def _filter_indices(indices: List[int], max_index: int) -> List[int]:
    seen = set()
    filtered: List[int] = []
    for idx in indices:
        if idx <= 0 or idx > max_index:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        filtered.append(idx)
    return filtered
