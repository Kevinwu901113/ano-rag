from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from relrag.config.config_loader import config as global_config
from relrag.prompt import render_prompt
from relrag.utils.output_protocol import build_final_instruction
from relrag.utils.token_counter import TokenCounter


CHAT_BASE_OVERHEAD_TOKENS = 12
CHAT_MESSAGE_OVERHEAD_TOKENS = 6
CHARS_PER_TOKEN = 3
MIN_OUTPUT_TOKENS = 16
MIN_ITEM_TOKENS = 32


@dataclass
class BudgetSettings:
    model_ctx_len: int
    safety_margin_tokens: int
    min_output_tokens: int
    max_items: int
    max_item_tokens: Optional[int]
    max_item_chars: Optional[int]


@dataclass
class BudgetReport:
    estimated_input_tokens: int
    requested_max_tokens: int
    effective_max_tokens: int
    model_ctx_len: int
    available_tokens: int
    dropped_items_count: int
    truncated_items_count: int
    deduped_items_count: int
    prompt_head: str
    items_count: int


@dataclass
class BudgetedPrompt:
    prompt: str
    messages: List[Dict[str, str]]
    items: List[Dict[str, Any]]
    report: BudgetReport


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _resolve_model_ctx_len(cfg: Dict[str, Any], llm_cfg: Optional[Dict[str, Any]]) -> int:
    for key in ("max_context_len", "context_len", "ctx_len", "model_ctx_len"):
        if llm_cfg and llm_cfg.get(key) is not None:
            return max(256, _coerce_int(llm_cfg.get(key), 8192))
    llm_root = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    if llm_root and llm_root.get("max_context_len") is not None:
        return max(256, _coerce_int(llm_root.get("max_context_len"), 8192))
    return 8192


def _resolve_safety_margin(cfg: Dict[str, Any]) -> int:
    llm_root = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    margin = llm_root.get("safety_margin_tokens", 256)
    return max(0, _coerce_int(margin, 256))


def _resolve_settings(
    cfg: Optional[Dict[str, Any]],
    llm_cfg: Optional[Dict[str, Any]],
    *,
    stage: str,
    max_items_override: Optional[int] = None,
    max_item_tokens_override: Optional[int] = None,
    max_item_chars_override: Optional[int] = None,
) -> BudgetSettings:
    resolved_cfg = cfg or global_config.load_config()
    model_ctx_len = _resolve_model_ctx_len(resolved_cfg, llm_cfg)
    safety_margin_tokens = _resolve_safety_margin(resolved_cfg)
    stage_cfg = resolved_cfg.get(stage) if isinstance(resolved_cfg.get(stage), dict) else {}
    max_items = stage_cfg.get("max_evidence_items") if stage == "answer" else stage_cfg.get("max_candidates")
    max_item_tokens = stage_cfg.get("max_evidence_tokens") if stage == "answer" else stage_cfg.get("max_candidate_tokens")
    max_item_chars = stage_cfg.get("max_evidence_chars") if stage == "answer" else stage_cfg.get("max_candidate_chars")
    if max_items_override is not None:
        max_items = max_items_override
    if max_item_tokens_override is not None:
        max_item_tokens = max_item_tokens_override
    if max_item_chars_override is not None:
        max_item_chars = max_item_chars_override
    max_items = max(1, _coerce_int(max_items, 8))
    max_item_tokens = _coerce_int(max_item_tokens, 0) if max_item_tokens is not None else None
    max_item_chars = _coerce_int(max_item_chars, 0) if max_item_chars is not None else None
    if max_item_tokens and not max_item_chars:
        max_item_chars = max(32, int(max_item_tokens * CHARS_PER_TOKEN))
    return BudgetSettings(
        model_ctx_len=model_ctx_len,
        safety_margin_tokens=safety_margin_tokens,
        min_output_tokens=MIN_OUTPUT_TOKENS,
        max_items=max_items,
        max_item_tokens=max_item_tokens if max_item_tokens and max_item_tokens > 0 else None,
        max_item_chars=max_item_chars if max_item_chars and max_item_chars > 0 else None,
    )


def _estimate_text_tokens(text: str) -> int:
    return TokenCounter.count_text(text or "")


def estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
    return TokenCounter.count_messages(messages or [])


def _truncate_text(text: str, max_chars: Optional[int]) -> str:
    if not text:
        return ""
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _normalize_text_key(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def collapse_pipe_duplicates(text: str) -> str:
    if not text:
        return ""
    parts = [part.strip() for part in text.split("|")]
    cleaned: List[str] = []
    for part in parts:
        if not part:
            continue
        if not cleaned or part != cleaned[-1]:
            cleaned.append(part)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    first_key = _normalize_text_key(cleaned[0])
    if first_key and all(_normalize_text_key(part) == first_key for part in cleaned[1:]):
        return cleaned[0]
    return " | ".join(cleaned)


def _format_strong(item: Dict[str, Any], idx: int) -> str:
    canon = item.get("canonical") or item.get("evidence") or ""
    raw = item.get("evidence") or ""
    nid = item.get("note_id") or ""
    if raw:
        return f"{idx + 1}) [{nid}] {canon} | {raw}"
    return f"{idx + 1}) [{nid}] {canon}"


def _format_weak(item: Dict[str, Any]) -> str:
    canon = item.get("canonical") or item.get("evidence") or ""
    raw = item.get("evidence") or ""
    nid = item.get("note_id") or ""
    score = item.get("score")
    subj_hint = item.get("subj") or item.get("anchor_entity") or "?"
    try:
        score_text = f"{float(score):.2f}"
    except (TypeError, ValueError):
        score_text = "~0.30"
    if raw:
        return f"- [{nid}] (score≈{score_text}, subj≈{subj_hint}) {canon} | {raw}"
    return f"- [{nid}] (score≈{score_text}, subj≈{subj_hint}) {canon}"


def _prepare_evidences(
    evidences: List[Dict[str, Any]],
    max_item_chars: Optional[int],
    *,
    include_raw_evidence: bool,
) -> Tuple[List[Dict[str, Any]], int, int]:
    prepared: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    deduped = 0
    truncated = 0
    for ev in evidences or []:
        note_id = str(ev.get("note_id") or "")
        summary = ev.get("summary")
        canon = summary or ev.get("canonical") or ev.get("evidence") or ""
        raw = ev.get("evidence") or ""
        canon = collapse_pipe_duplicates(str(canon).strip())
        raw = collapse_pipe_duplicates(str(raw).strip())
        if not canon and raw:
            canon = raw
            raw = ""
        if not include_raw_evidence:
            raw = ""
        if canon and raw and _normalize_text_key(canon) == _normalize_text_key(raw):
            raw = ""
        canon_trim = _truncate_text(canon, max_item_chars)
        raw_trim = _truncate_text(raw, max_item_chars)
        if canon_trim != canon or raw_trim != raw:
            truncated += 1
        canon = canon_trim
        raw = raw_trim
        key = (note_id, _normalize_text_key(canon or raw))
        if key in seen:
            deduped += 1
            continue
        seen.add(key)
        new_item = dict(ev)
        new_item["canonical"] = canon
        new_item["evidence"] = raw
        prepared.append(new_item)
    return prepared, deduped, truncated


def _estimate_prompt_tokens(prompt: str, system_prompt: Optional[str]) -> Tuple[int, List[Dict[str, str]]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return estimate_messages_tokens(messages), messages


def _prompt_head(prompt: str, max_chars: int = 200) -> str:
    if not prompt:
        return ""
    head = prompt[:max_chars].replace("\n", " ").strip()
    if len(prompt) > max_chars:
        return f"{head}..."
    return head


def _build_answer_prompt(
    question: str,
    strong_items: List[Dict[str, Any]],
    weak_items: List[Dict[str, Any]],
    *,
    prompt_name: str,
    label_instruction: str,
) -> str:
    strong_block = "\n".join(_format_strong(item, idx) for idx, item in enumerate(strong_items)) or "None"
    weak_block = "\n".join(_format_weak(item) for item in weak_items) or "None"
    return render_prompt(
        prompt_name,
        q=question,
        strong_block=strong_block,
        weak_block=weak_block,
        label_instruction=label_instruction,
        final_instruction=build_final_instruction(),
    )


def budget_answer_prompt(
    question: str,
    evidences: List[Dict[str, Any]],
    *,
    prompt_name: str,
    label_instruction: str,
    system_prompt: Optional[str],
    cfg: Optional[Dict[str, Any]] = None,
    llm_cfg: Optional[Dict[str, Any]] = None,
    requested_max_tokens: int,
    max_items_override: Optional[int] = None,
    max_item_tokens_override: Optional[int] = None,
    max_item_chars_override: Optional[int] = None,
    include_raw_evidence: bool = True,
) -> BudgetedPrompt:
    settings = _resolve_settings(
        cfg,
        llm_cfg,
        stage="answer",
        max_items_override=max_items_override,
        max_item_tokens_override=max_item_tokens_override,
        max_item_chars_override=max_item_chars_override,
    )
    prepared, deduped_count, truncated_count = _prepare_evidences(
        evidences,
        settings.max_item_chars,
        include_raw_evidence=include_raw_evidence,
    )
    strong_items = [item for item in prepared if not item.get("weak")]
    weak_items = [item for item in prepared if item.get("weak")]
    max_input_tokens = settings.model_ctx_len - settings.safety_margin_tokens - requested_max_tokens
    packed_strong: List[Dict[str, Any]] = []
    packed_weak: List[Dict[str, Any]] = []
    for item in strong_items:
        if len(packed_strong) + len(packed_weak) >= settings.max_items:
            break
        candidate_prompt = _build_answer_prompt(
            question,
            packed_strong + [item],
            packed_weak,
            prompt_name=prompt_name,
            label_instruction=label_instruction,
        )
        est_tokens, _ = _estimate_prompt_tokens(candidate_prompt, system_prompt)
        if est_tokens <= max_input_tokens:
            packed_strong.append(item)
        else:
            break
    for item in weak_items:
        if len(packed_strong) + len(packed_weak) >= settings.max_items:
            break
        candidate_prompt = _build_answer_prompt(
            question,
            packed_strong,
            packed_weak + [item],
            prompt_name=prompt_name,
            label_instruction=label_instruction,
        )
        est_tokens, _ = _estimate_prompt_tokens(candidate_prompt, system_prompt)
        if est_tokens <= max_input_tokens:
            packed_weak.append(item)
        else:
            break
    prompt = _build_answer_prompt(
        question,
        packed_strong,
        packed_weak,
        prompt_name=prompt_name,
        label_instruction=label_instruction,
    )
    est_tokens, messages = _estimate_prompt_tokens(prompt, system_prompt)
    available = settings.model_ctx_len - settings.safety_margin_tokens - est_tokens
    while available < settings.min_output_tokens and (packed_weak or packed_strong):
        if packed_weak:
            packed_weak.pop()
        else:
            packed_strong.pop()
        prompt = _build_answer_prompt(
            question,
            packed_strong,
            packed_weak,
            prompt_name=prompt_name,
            label_instruction=label_instruction,
        )
        est_tokens, messages = _estimate_prompt_tokens(prompt, system_prompt)
        available = settings.model_ctx_len - settings.safety_margin_tokens - est_tokens
    if available < settings.min_output_tokens and not (packed_weak or packed_strong):
        guard = 0
        while available < settings.min_output_tokens and question and guard < 3:
            shrink_by = (settings.min_output_tokens - available) * CHARS_PER_TOKEN
            if shrink_by <= 0:
                break
            truncated_count += 1
            updated = _truncate_text(question, max(32, len(question) - int(shrink_by)))
            if updated == question:
                break
            question = updated
            prompt = _build_answer_prompt(
                question,
                packed_strong,
                packed_weak,
                prompt_name=prompt_name,
                label_instruction=label_instruction,
            )
            est_tokens, messages = _estimate_prompt_tokens(prompt, system_prompt)
            available = settings.model_ctx_len - settings.safety_margin_tokens - est_tokens
            guard += 1
    effective_max_tokens = min(requested_max_tokens, max(settings.min_output_tokens, available))
    dropped_count = len(prepared) - len(packed_strong) - len(packed_weak)
    report = BudgetReport(
        estimated_input_tokens=est_tokens,
        requested_max_tokens=requested_max_tokens,
        effective_max_tokens=effective_max_tokens,
        model_ctx_len=settings.model_ctx_len,
        available_tokens=max(0, available),
        dropped_items_count=max(0, dropped_count),
        truncated_items_count=truncated_count,
        deduped_items_count=deduped_count,
        prompt_head=_prompt_head(prompt),
        items_count=len(packed_strong) + len(packed_weak),
    )
    return BudgetedPrompt(prompt=prompt, messages=messages, items=packed_strong + packed_weak, report=report)


def budget_rerank_prompt(
    question: str,
    candidates: List[Dict[str, Any]],
    *,
    prompt_name: str,
    cfg: Optional[Dict[str, Any]] = None,
    llm_cfg: Optional[Dict[str, Any]] = None,
    requested_max_tokens: int,
    max_items_override: Optional[int] = None,
    max_item_tokens_override: Optional[int] = None,
    max_item_chars_override: Optional[int] = None,
) -> BudgetedPrompt:
    settings = _resolve_settings(
        cfg,
        llm_cfg,
        stage="rerank",
        max_items_override=max_items_override,
        max_item_tokens_override=max_item_tokens_override,
        max_item_chars_override=max_item_chars_override,
    )
    from relrag.utils.text_builders import build_note_text_for_rank

    prepared: List[Dict[str, Any]] = []
    seen_note_ids: set[str] = set()
    truncated_count = 0
    deduped_count = 0
    for cand in candidates or []:
        note = cand.get("note") or {}
        note_id = str(note.get("note_id") or cand.get("note_id") or "")
        if note_id and note_id in seen_note_ids:
            deduped_count += 1
            continue
        text = build_note_text_for_rank(note, max_len=settings.max_item_chars)
        if settings.max_item_chars and text and len(text) >= settings.max_item_chars:
            truncated_count += 1
        prepared.append({**cand, "_note_id": note_id, "_note_text": text})
        if note_id:
            seen_note_ids.add(note_id)
    max_input_tokens = settings.model_ctx_len - settings.safety_margin_tokens - requested_max_tokens
    packed: List[Dict[str, Any]] = []
    lines: List[str] = []
    for cand in prepared:
        if len(packed) >= settings.max_items:
            break
        note_id = cand.get("_note_id") or ""
        note_text = cand.get("_note_text") or ""
        line = f"{len(lines) + 1}. note_id={note_id} :: {note_text}".strip()
        candidate_prompt = render_prompt(prompt_name, question=question, candidates="\n".join(lines + [line]))
        est_tokens, _ = _estimate_prompt_tokens(candidate_prompt, None)
        if est_tokens <= max_input_tokens:
            packed.append(cand)
            lines.append(line)
        else:
            break
    prompt = render_prompt(prompt_name, question=question, candidates="\n".join(lines))
    est_tokens, messages = _estimate_prompt_tokens(prompt, None)
    available = settings.model_ctx_len - settings.safety_margin_tokens - est_tokens
    while available < settings.min_output_tokens and packed:
        packed.pop()
        lines = [
            f"{idx + 1}. note_id={item.get('_note_id') or ''} :: {item.get('_note_text') or ''}".strip()
            for idx, item in enumerate(packed)
        ]
        prompt = render_prompt(prompt_name, question=question, candidates="\n".join(lines))
        est_tokens, messages = _estimate_prompt_tokens(prompt, None)
        available = settings.model_ctx_len - settings.safety_margin_tokens - est_tokens
    if available < settings.min_output_tokens and not packed:
        guard = 0
        while available < settings.min_output_tokens and question and guard < 3:
            shrink_by = (settings.min_output_tokens - available) * CHARS_PER_TOKEN
            if shrink_by <= 0:
                break
            truncated_count += 1
            updated = _truncate_text(question, max(32, len(question) - int(shrink_by)))
            if updated == question:
                break
            question = updated
            prompt = render_prompt(prompt_name, question=question, candidates="\n".join(lines))
            est_tokens, messages = _estimate_prompt_tokens(prompt, None)
            available = settings.model_ctx_len - settings.safety_margin_tokens - est_tokens
            guard += 1
    effective_max_tokens = min(requested_max_tokens, max(settings.min_output_tokens, available))
    dropped_count = len(prepared) - len(packed)
    report = BudgetReport(
        estimated_input_tokens=est_tokens,
        requested_max_tokens=requested_max_tokens,
        effective_max_tokens=effective_max_tokens,
        model_ctx_len=settings.model_ctx_len,
        available_tokens=max(0, available),
        dropped_items_count=max(0, dropped_count),
        truncated_items_count=truncated_count,
        deduped_items_count=deduped_count,
        prompt_head=_prompt_head(prompt),
        items_count=len(packed),
    )
    cleaned_items: List[Dict[str, Any]] = []
    for item in packed:
        entry = dict(item)
        entry.pop("_note_text", None)
        entry.pop("_note_id", None)
        cleaned_items.append(entry)
    return BudgetedPrompt(prompt=prompt, messages=messages, items=cleaned_items, report=report)


def log_budget_event(
    run_dir: Optional[str],
    *,
    stage: str,
    phase: str,
    report: BudgetReport,
    attempt: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not run_dir:
        return
    payload = {
        "timestamp": int(time.time()),
        "stage": stage,
        "phase": phase,
        "attempt": attempt,
        "estimated_input_tokens": report.estimated_input_tokens,
        "requested_max_tokens": report.requested_max_tokens,
        "effective_max_tokens": report.effective_max_tokens,
        "model_ctx_len": report.model_ctx_len,
        "available_tokens": report.available_tokens,
        "dropped_items_count": report.dropped_items_count,
        "truncated_items_count": report.truncated_items_count,
        "deduped_items_count": report.deduped_items_count,
        "items_count": report.items_count,
        "prompt_head": report.prompt_head,
    }
    if extra:
        payload.update(extra)
    try:
        path = Path(run_dir) / "llm_context_budget.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Failed to write budget log to {}: {}", run_dir, exc)


def apply_load_shed(
    *,
    max_items: int,
    max_item_tokens: Optional[int],
    requested_max_tokens: int,
    min_items: int = 1,
    min_item_tokens: int = MIN_ITEM_TOKENS,
    min_output_tokens: int = MIN_OUTPUT_TOKENS,
) -> Tuple[int, Optional[int], int]:
    new_max_items = max(min_items, int(max_items * 0.7))
    if max_item_tokens is not None:
        new_max_item_tokens = max(min_item_tokens, int(max_item_tokens * 0.7))
    else:
        new_max_item_tokens = None
    new_requested_max_tokens = max(min_output_tokens, int(requested_max_tokens * 0.7))
    return new_max_items, new_max_item_tokens, new_requested_max_tokens
