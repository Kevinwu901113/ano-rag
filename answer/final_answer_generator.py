"""Final answer generation with evidence-first validation and EFSA hints."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import config
from llm.prompts.final_answer import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_USER_PROMPT,
)
from validators.final_answer_validator import validate_final_answer

log = logging.getLogger(__name__)


def _number_context_lines(raw_ctx: List[str]) -> str:
    """Attach `[L{n}]` numbering to each context line for evidence quoting."""
    lines: List[str] = []
    for i, line in enumerate(raw_ctx or [], start=1):
        ln = f"[L{i}] {line.strip()}"
        lines.append(ln)
    return "\n".join(lines)


def _select_candidate_from_efsa(
    efsa_output: Optional[Dict[str, Any]],
    extra_candidates: Optional[List[str]] = None,
) -> Tuple[Optional[str], bool]:
    """Select EFSA hint respecting threshold, deduplicate with extra candidates."""
    hint_cfg = (config.get("answering", {}) or {}).get("efsa_hint", {}) or {}
    enabled = bool(hint_cfg.get("enabled", True))
    threshold = float(hint_cfg.get("threshold", 0.70))
    top_n = int(hint_cfg.get("multi_candidate", 2))

    if not enabled:
        return None, False

    cands: List[str] = []
    passed_threshold = False

    efsa_output = efsa_output or {}
    efsa_answer = str(efsa_output.get("answer", ""))
    efsa_answer = efsa_answer.strip()
    try:
        efsa_score = float(efsa_output.get("score", 0.0))
    except (TypeError, ValueError):
        efsa_score = 0.0

    if efsa_answer and efsa_score >= threshold:
        cands.append(efsa_answer)
        passed_threshold = True

    for cand in extra_candidates or []:
        cand = (cand or "").strip()
        if cand:
            cands.append(cand)

    seen: set[str] = set()
    deduped: List[str] = []
    for cand in cands:
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
        if len(deduped) >= top_n:
            break

    if not deduped:
        return None, passed_threshold

    return " | ".join(deduped), passed_threshold


def _call_llm(llm: Any, system_prompt: str, user_prompt: str) -> str:
    """Invoke the provided LLM client in a provider-agnostic way."""
    if llm is None:
        raise ValueError("LLM client must not be None")

    chat_fn = getattr(llm, "chat", None)
    if callable(chat_fn):
        try:
            return chat_fn(system=system_prompt, user=user_prompt)
        except TypeError:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            return chat_fn(messages)

    generate_fn = getattr(llm, "generate", None)
    if callable(generate_fn):
        return generate_fn(user_prompt, system_prompt=system_prompt)

    raise AttributeError("LLM client does not expose chat or generate methods")


def generate_final_answer(
    question: str,
    context_lines: List[str],
    llm: Any,
    efsa_output: Optional[Dict[str, Any]] = None,
    extra_candidates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate and validate the final answer with optional EFSA hints."""

    candidate, passed_threshold = _select_candidate_from_efsa(efsa_output, extra_candidates)
    context_numbered = _number_context_lines(context_lines or [])
    candidate_field = (
        json.dumps(candidate, ensure_ascii=False) if candidate else "null"
    )

    user_prompt = FINAL_ANSWER_USER_PROMPT.format(
        question=question,
        context_numbered=context_numbered,
        candidate_answer=candidate_field,
    )

    raw = _call_llm(llm, FINAL_ANSWER_SYSTEM_PROMPT, user_prompt)

    ok, obj, report = validate_final_answer(raw, context_lines, question, candidate)

    used_candidate = bool(obj.get("used_candidate")) if obj else False
    efsa_score = (efsa_output or {}).get("score")

    log.info(
        "final_answer_validate ok=%s used_candidate=%s efsa_score=%s passed_threshold=%s report=%s",
        ok,
        used_candidate,
        efsa_score,
        passed_threshold,
        report,
    )

    force_insufficient = bool(
        (config.get("answering", {}) or {}).get(
            "force_insufficient_if_no_spans", True
        )
    )

    if not ok and force_insufficient:
        degraded_output = {
            "disambiguation": obj.get("disambiguation", "") if obj else "",
            "evidence_spans": [],
            "reason": "insufficient evidence",
            "answer": "insufficient",
            "used_candidate": False,
        }
        return {
            "raw": raw,
            "output": degraded_output,
            "validator": report,
            "candidate_answer": candidate,
            "efsa_score": efsa_score,
            "passed_threshold": passed_threshold,
            "validator_ok": False,
        }

    result_output = obj if ok else obj or {}
    return {
        "raw": raw,
        "output": result_output,
        "validator": report,
        "candidate_answer": candidate,
        "efsa_score": efsa_score,
        "passed_threshold": passed_threshold,
        "validator_ok": ok,
    }
