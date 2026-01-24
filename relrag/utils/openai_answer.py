from __future__ import annotations

import time
from typing import Any, Dict, List

from relrag.prompt import load_prompt, render_prompt
from relrag.utils.openai_client import chat_completion
from relrag.utils.output_protocol import build_final_instruction
from relrag.utils.context_budget import apply_load_shed, budget_answer_prompt, log_budget_event, collapse_pipe_duplicates
from relrag.utils.llm_errors import ContextLengthError
from relrag.utils.llm_stats import get_active_llm_stats
from relrag.config.config_loader import config as global_config


ANSWER_PROMPT_NAME = "answerer.txt"
DEFAULT_SYSTEM_PROMPT_NAME = "system_prompt.txt"


def _resolve_system_prompt(openai_cfg: Dict[str, Any]) -> str:
    if "system_prompt" in openai_cfg:
        raw = openai_cfg.get("system_prompt")
        if raw is None:
            return ""
        return str(raw)
    name = openai_cfg.get("system_prompt_name")
    if name is None:
        return load_prompt(DEFAULT_SYSTEM_PROMPT_NAME)
    name = str(name).strip()
    if not name:
        return ""
    return load_prompt(name)


def _fmt_strong(item: Dict[str, Any], idx: int) -> str:
    canon = item.get("canonical") or item.get("evidence") or ""
    raw = item.get("evidence") or ""
    nid = item.get("note_id") or ""
    summary = item.get("summary")
    if summary:
        canon = summary
    canon = collapse_pipe_duplicates(str(canon).strip())
    raw = collapse_pipe_duplicates(str(raw).strip())
    if canon and raw and canon.strip().lower() == raw.strip().lower():
        raw = ""
    if raw:
        return f"{idx + 1}) [{nid}] {canon} | {raw}"
    return f"{idx + 1}) [{nid}] {canon}"


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
    canon = collapse_pipe_duplicates(str(canon).strip())
    raw = collapse_pipe_duplicates(str(raw).strip())
    if canon and raw and canon.strip().lower() == raw.strip().lower():
        raw = ""
    if raw:
        return f"- [{nid}] (score≈{score_text}, subj≈{subj_hint}) {canon} | {raw}"
    return f"- [{nid}] (score≈{score_text}, subj≈{subj_hint}) {canon}"


def build_answer_prompt(
    question: str,
    evidences: List[Dict[str, Any]],
    *,
    prompt_name: str = ANSWER_PROMPT_NAME,
) -> str:
    strong_items = [item for item in evidences if not item.get("weak")]
    weak_items = [item for item in evidences if item.get("weak")]
    strong_block = "\n".join(_fmt_strong(item, idx) for idx, item in enumerate(strong_items)) or "None"
    weak_block = "\n".join(_fmt_weak(item) for item in weak_items) or "None"
    label_instruction = "If you can answer, output the canonical label only."
    return render_prompt(
        prompt_name,
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
    *,
    prompt_capture: Dict[str, Any] | None = None,
    run_dir: str | None = None,
    cfg: Dict[str, Any] | None = None,
) -> str:
    prompt_name = openai_cfg.get("answer_prompt_name") or ANSWER_PROMPT_NAME
    system_prompt = _resolve_system_prompt(openai_cfg)
    resolved_cfg = cfg or global_config.load_config()
    answerer_cfg = resolved_cfg.get("answerer") if isinstance(resolved_cfg.get("answerer"), dict) else {}
    include_raw_evidence = bool(answerer_cfg.get("include_raw_evidence", True))
    evidence_max_chars = answerer_cfg.get("evidence_max_chars")
    stop_sequences = answerer_cfg.get("stop")
    normalized_stop: List[str] | None = None
    if stop_sequences:
        if isinstance(stop_sequences, str):
            normalized_stop = [stop_sequences]
        elif isinstance(stop_sequences, list):
            normalized_stop = [str(item) for item in stop_sequences if str(item).strip()]
    runtime_cfg = resolved_cfg.get("runtime") if isinstance(resolved_cfg.get("runtime"), dict) else {}
    run_dir = run_dir or runtime_cfg.get("run_dir")
    base_limits = resolved_cfg.get("answer") if isinstance(resolved_cfg.get("answer"), dict) else {}
    base_max_items = int(base_limits.get("max_evidence_items", 8))
    base_max_item_tokens = base_limits.get("max_evidence_tokens")
    try:
        base_max_item_tokens = int(base_max_item_tokens) if base_max_item_tokens is not None else None
    except (TypeError, ValueError):
        base_max_item_tokens = None
    requested_max_tokens = int(openai_cfg.get("max_tokens", 256) or 256)
    max_items_override = None
    max_item_tokens_override = None
    max_item_chars_override = None
    if evidence_max_chars is not None:
        try:
            max_item_chars_override = int(evidence_max_chars)
        except (TypeError, ValueError):
            max_item_chars_override = None
    attempt = 0
    while True:
        build_start = time.time()
        budgeted = budget_answer_prompt(
            question,
            evidences,
            prompt_name=prompt_name,
            label_instruction="If you can answer, output the canonical label only.",
            system_prompt=system_prompt,
            cfg=resolved_cfg,
            llm_cfg=openai_cfg,
            requested_max_tokens=requested_max_tokens,
            max_items_override=max_items_override,
            max_item_tokens_override=max_item_tokens_override,
            max_item_chars_override=max_item_chars_override,
            include_raw_evidence=include_raw_evidence,
        )
        stats = get_active_llm_stats()
        if stats is not None:
            context_chars = 0
            for item in budgeted.items or []:
                canon = item.get("canonical") or ""
                raw = item.get("evidence") or ""
                context_chars += len(str(canon))
                if include_raw_evidence and raw:
                    context_chars += len(str(raw))
            stats.record_prompt(
                prompt_tokens=budgeted.report.estimated_input_tokens,
                prompt_chars=len(budgeted.prompt or ""),
                context_chars=context_chars,
                build_ms=(time.time() - build_start) * 1000.0,
            )
        if prompt_capture is not None:
            prompt_capture["prompt"] = budgeted.prompt
            prompt_capture["system_prompt"] = system_prompt
            prompt_capture["prompt_name"] = prompt_name
        log_budget_event(
            run_dir,
            stage="answer",
            phase="pre",
            report=budgeted.report,
            attempt=attempt + 1,
        )
        try:
            response = chat_completion(
                budgeted.messages,
                model=openai_cfg.get("model"),
                api_key=openai_cfg.get("api_key"),
                base_url=openai_cfg.get("base_url"),
                temperature=openai_cfg.get("temperature"),
                max_tokens=budgeted.report.effective_max_tokens,
                timeout_sec=openai_cfg.get("timeout_sec", 60.0),
                max_retries=openai_cfg.get("max_retries", 2),
                retry_backoff_sec=openai_cfg.get("retry_backoff_sec", 1.0),
                retry_backoff_max_sec=openai_cfg.get("retry_backoff_max_sec", 20.0),
                stop=normalized_stop,
            )
            response_text = response.strip()
            log_budget_event(
                run_dir,
                stage="answer",
                phase="post",
                report=budgeted.report,
                attempt=attempt + 1,
                extra={"response_chars": len(response_text)},
            )
            return response_text
        except ContextLengthError as exc:
            if attempt >= 1:
                raise
            current_max_items = max_items_override or base_max_items
            current_max_item_tokens = max_item_tokens_override or base_max_item_tokens
            before = {
                "max_items": current_max_items,
                "max_item_tokens": current_max_item_tokens,
                "requested_max_tokens": requested_max_tokens,
            }
            max_items_override, max_item_tokens_override, requested_max_tokens = apply_load_shed(
                max_items=current_max_items,
                max_item_tokens=current_max_item_tokens,
                requested_max_tokens=requested_max_tokens,
            )
            after = {
                "max_items": max_items_override,
                "max_item_tokens": max_item_tokens_override,
                "requested_max_tokens": requested_max_tokens,
            }
            log_budget_event(
                run_dir,
                stage="answer",
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
