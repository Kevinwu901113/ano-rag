import time
from typing import Any, Dict, List, Optional

from loguru import logger

from relrag.generator.extractor import judge_and_compress
from relrag.utils.llm_stats import get_active_llm_stats
from relrag.utils.llm_client import LLMChatClient
from relrag.utils.context_budget import apply_load_shed, budget_answer_prompt, log_budget_event
from relrag.utils.llm_errors import ContextLengthError
from relrag.config.config_loader import config as global_config
from relrag.prompt import load_prompt


ANSWER_PROMPT_NAME = "answerer.txt"


def call_llm(
    endpoint: str,
    model: str,
    question: str,
    evidences: list,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    retries: int = 2,
    allowed_labels: Optional[List[str]] = None,
    attribute_name: Optional[str] = None,
    label_instruction_override: Optional[str] = None,
    prompt_name: str = ANSWER_PROMPT_NAME,
    system_prompt_name: Optional[str] = None,
    prompt_capture: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    run_dir: Optional[str] = None,
) -> str:
    resolved_cfg = cfg or global_config.load_config()
    answerer_cfg = resolved_cfg.get("answerer") if isinstance(resolved_cfg.get("answerer"), dict) else {}
    compress_cfg = answerer_cfg.get("compress_evidence") if isinstance(answerer_cfg.get("compress_evidence"), dict) else {}
    compress_enabled = bool(compress_cfg.get("enabled", True))
    include_raw_evidence = bool(answerer_cfg.get("include_raw_evidence", True))
    evidence_max_chars = answerer_cfg.get("evidence_max_chars")
    stop_sequences = answerer_cfg.get("stop")
    normalized_stop: Optional[List[str]] = None
    if stop_sequences:
        if isinstance(stop_sequences, str):
            normalized_stop = [stop_sequences]
        elif isinstance(stop_sequences, list):
            normalized_stop = [str(item) for item in stop_sequences if str(item).strip()]

    compressed: list = []
    if compress_enabled:
        compressed = _compress_evidence(question, evidences, endpoint, model, cfg=resolved_cfg)

    display_evs = compressed or evidences
    sanitized_labels = _prepare_allowed_labels(allowed_labels)
    label_instruction = _label_instruction(
        sanitized_labels,
        attribute_name,
        override=label_instruction_override,
    )
    runtime_cfg = resolved_cfg.get("runtime") if isinstance(resolved_cfg.get("runtime"), dict) else {}
    run_dir = run_dir or runtime_cfg.get("run_dir")
    base_limits = resolved_cfg.get("answer") if isinstance(resolved_cfg.get("answer"), dict) else {}
    base_max_items = int(base_limits.get("max_evidence_items", 8))
    base_max_item_tokens = base_limits.get("max_evidence_tokens")
    try:
        base_max_item_tokens = int(base_max_item_tokens) if base_max_item_tokens is not None else None
    except (TypeError, ValueError):
        base_max_item_tokens = None
    if max_tokens is None:
        profiles_cfg = resolved_cfg.get("llm_profiles") if isinstance(resolved_cfg.get("llm_profiles"), dict) else {}
        generate_cfg = profiles_cfg.get("generate") if isinstance(profiles_cfg.get("generate"), dict) else {}
        vllm_cfg = resolved_cfg.get("vllm") if isinstance(resolved_cfg.get("vllm"), dict) else {}
        max_tokens = (
            generate_cfg.get("max_tokens")
            if generate_cfg.get("max_tokens") is not None
            else vllm_cfg.get("max_tokens", 256)
        )
    requested_max_tokens = int(max_tokens)
    system_prompt_text: Optional[str] = None
    if system_prompt_name is not None:
        name = str(system_prompt_name).strip()
        if name:
            try:
                system_prompt_text = load_prompt(name)
            except Exception as exc:
                logger.warning("Failed to load system prompt {}: {}", name, exc)
                system_prompt_text = None
    max_items_override = None
    max_item_tokens_override = None
    max_item_chars_override = None
    if evidence_max_chars is not None:
        try:
            max_item_chars_override = int(evidence_max_chars)
        except (TypeError, ValueError):
            max_item_chars_override = None

    client = LLMChatClient(
        endpoint=endpoint,
        model=model,
        llm_profile="generate",
        retries=retries,
        timeout=60,
    )
    attempt = 0
    while True:
        build_start = time.time()
        budgeted = budget_answer_prompt(
            question,
            display_evs,
            prompt_name=prompt_name,
            label_instruction=label_instruction,
            system_prompt=system_prompt_text,
            cfg=resolved_cfg,
            llm_cfg=resolved_cfg.get("vllm"),
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
            prompt_capture["prompt_name"] = prompt_name
            if system_prompt_text is not None:
                prompt_capture["system_prompt"] = system_prompt_text
        log_budget_event(
            run_dir,
            stage="answer",
            phase="pre",
            report=budgeted.report,
            attempt=attempt + 1,
        )
        try:
            response = client.chat(
                budgeted.messages,
                temperature=temperature,
                max_tokens=budgeted.report.effective_max_tokens,
                stop=normalized_stop,
            )
            content = response.content.strip()
            log_budget_event(
                run_dir,
                stage="answer",
                phase="post",
                report=budgeted.report,
                attempt=attempt + 1,
                extra={"response_chars": len(content)},
            )
            return content
        except ContextLengthError as exc:
            if attempt >= 1:
                logger.warning("Answerer load-shed retries exhausted: {}", exc)
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
        except Exception as exc:  # noqa: PERF203
            logger.warning("Answerer call failed: {}", exc)
            raise


def _compress_evidence(
    question: str,
    evidences: list,
    endpoint: str,
    model: str,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> list:
    if not evidences:
        return []
    resolved_cfg = cfg or {}
    answerer_cfg = resolved_cfg.get("answerer") if isinstance(resolved_cfg.get("answerer"), dict) else {}
    compress_cfg = answerer_cfg.get("compress_evidence") if isinstance(answerer_cfg.get("compress_evidence"), dict) else {}
    vllm_cfg = resolved_cfg.get("vllm") if isinstance(resolved_cfg.get("vllm"), dict) else {}
    reranker_cfg = resolved_cfg.get("reranker") if isinstance(resolved_cfg.get("reranker"), dict) else {}
    rerank_llm_cfg = reranker_cfg.get("llm") if isinstance(reranker_cfg.get("llm"), dict) else {}
    concurrency_cfg = vllm_cfg.get("concurrency") if isinstance(vllm_cfg.get("concurrency"), dict) else {}

    timeout_s = compress_cfg.get("timeout_s")
    if timeout_s is None:
        timeout_s = rerank_llm_cfg.get("timeout_s")
    if timeout_s is None:
        timeout_s = concurrency_cfg.get("read_timeout_sec")
    if timeout_s is None:
        timeout_s = 60
    retries = compress_cfg.get("retries", 2)
    max_tokens = compress_cfg.get("max_tokens", 128)
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

    llm_cfg = {
        "endpoint": endpoint,
        "model": model,
        "timeout_s": max(5, timeout_s),
        "retries": max(0, retries),
        "max_tokens": max(16, max_tokens),
    }
    notes = []
    for ev in evidences:
        notes.append(
            {
                "note_id": ev.get("note_id"),
                "evidence": ev.get("canonical") or ev.get("evidence"),
                "meta": {"evidence_canonical": ev.get("canonical")},
            }
        )
    try:
        compressed = judge_and_compress(question, notes, llm_cfg)
        # merge with originals for formatting fallback
        merged = []
        for comp in compressed:
            match = next((ev for ev in evidences if ev.get("note_id") == comp.get("note_id")), {})
            merged.append({**match, **comp})
        return merged
    except Exception as exc:  # noqa: PERF203
        logger.warning("Evidence compression failed: {}", exc)
        return []


def _label_instruction(
    labels: List[str],
    attribute_name: Optional[str],
    *,
    override: Optional[str] = None,
) -> str:
    if override and str(override).strip():
        return str(override).strip()
    if labels:
        joined = ", ".join(labels)
        return f"Allowed labels ({attribute_name or 'answer'}): {joined}."
    if attribute_name and str(attribute_name).strip():
        return (
            f"Question focus attribute: {str(attribute_name).strip()}. "
            "Output only the attribute value as a short answer span copied from evidence."
        )
    return "If you can answer, output the canonical label only."


def _prepare_allowed_labels(labels: Optional[List[str]]) -> List[str]:
    prepared: List[str] = []
    if not labels:
        return prepared
    seen: set[str] = set()
    for label in labels:
        text = str(label).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        prepared.append(text)
    return prepared


def _enforce_single_label(text: str, allowed: List[str]) -> str:
    candidate = _strip_reasoning(text)
    candidate = candidate.splitlines()[0].strip()
    if not candidate:
        return ""
    if allowed:
        lowered = candidate.lower()
        for label in allowed:
            if lowered == label.lower():
                return label
        for label in allowed:
            if label.lower() in lowered:
                return label
        return allowed[0]
    return candidate


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
