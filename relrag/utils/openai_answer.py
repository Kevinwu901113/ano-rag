from __future__ import annotations

from typing import Any, Dict, List

from relrag.prompt import render_prompt
from relrag.utils.openai_client import chat_completion
from relrag.utils.output_protocol import build_final_instruction


ANSWER_PROMPT_NAME = "answerer.txt"


def _fmt_strong(item: Dict[str, Any], idx: int) -> str:
    canon = item.get("canonical") or item.get("evidence") or ""
    raw = item.get("evidence") or ""
    nid = item.get("note_id") or ""
    summary = item.get("summary")
    if summary:
        canon = summary
    return f"{idx + 1}) [{nid}] {canon} | {raw}"


def _fmt_weak(item: Dict[str, Any]) -> str:
    canon = item.get("canonical") or item.get("evidence") or ""
    raw = item.get("evidence") or ""
    nid = item.get("note_id") or ""
    score = item.get("score")
    subj_hint = item.get("subj") or item.get("anchor_entity") or "?"
    try:
        score_text = f"{float(score):.2f}"
    except (TypeError, ValueError):
        score_text = "~0.30"
    return f"- [{nid}] (score≈{score_text}, subj≈{subj_hint}) {canon} | {raw}"


def build_answer_prompt(question: str, evidences: List[Dict[str, Any]]) -> str:
    strong_items = [item for item in evidences if not item.get("weak")]
    weak_items = [item for item in evidences if item.get("weak")]
    strong_block = "\n".join(_fmt_strong(item, idx) for idx, item in enumerate(strong_items)) or "None"
    weak_block = "\n".join(_fmt_weak(item) for item in weak_items) or "None"
    label_instruction = "If you can answer, output the canonical label only."
    return render_prompt(
        ANSWER_PROMPT_NAME,
        q=question,
        strong_block=strong_block,
        weak_block=weak_block,
        label_instruction=label_instruction,
        final_instruction=build_final_instruction(),
    )


def generate_openai_answer(
    question: str,
    evidences: List[Dict[str, Any]],
    openai_cfg: Dict[str, Any],
) -> str:
    prompt = build_answer_prompt(question, evidences)
    system_prompt = openai_cfg.get("system_prompt")
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = chat_completion(
        messages,
        model=openai_cfg.get("model"),
        api_key=openai_cfg.get("api_key"),
        base_url=openai_cfg.get("base_url"),
        temperature=openai_cfg.get("temperature"),
        max_tokens=openai_cfg.get("max_tokens"),
        timeout_sec=openai_cfg.get("timeout_sec", 60.0),
        max_retries=openai_cfg.get("max_retries", 2),
        retry_backoff_sec=openai_cfg.get("retry_backoff_sec", 1.0),
        retry_backoff_max_sec=openai_cfg.get("retry_backoff_max_sec", 20.0),
    )
    return response.strip()
