from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Tuple

from relrag.utils.output_eval import extract_final_answer, has_final_tag, normalize_text


def sha1_text(text: str) -> str:
    payload = (text or "").encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


_YES_NO_PREFIX = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|should|would|will|has|have|had|may|might|must)\b",
    flags=re.IGNORECASE,
)
_YES_NO_ANSWER = re.compile(r"^\s*(yes|no)\b", flags=re.IGNORECASE)


def _is_yes_no_question(question: str | None) -> bool:
    if not question:
        return False
    return bool(_YES_NO_PREFIX.match(str(question)))


def _normalize_yes_no_answer(answer: str | None) -> str:
    if not answer:
        return ""
    text = str(answer).strip().lower()
    if text in {"yes", "no"}:
        return text
    if text in {"true", "false"}:
        return "yes" if text == "true" else "no"
    match = _YES_NO_ANSWER.match(text)
    if match:
        return match.group(1).lower()
    return ""


def _extract_yes_no_from_text(text: str) -> str:
    if not text:
        return ""
    cleaned = normalize_text(text, lowercase=True)
    matches = list(re.finditer(r"\b(yes|no)\b", cleaned))
    if not matches:
        return ""
    return matches[-1].group(1).lower()


def _maybe_normalize_yes_no(
    answer: str,
    question: str | None,
    raw_text: str,
    detail: Dict[str, Any],
) -> str:
    if not answer:
        return answer
    if str(answer).strip().lower() == "insufficient evidence":
        return answer
    if not _is_yes_no_question(question):
        return answer
    normalized = _normalize_yes_no_answer(answer)
    if normalized:
        if normalized != str(answer).strip().lower():
            detail["yes_no_normalized"] = normalized
        return normalized
    inferred = _extract_yes_no_from_text(raw_text)
    if inferred:
        detail["yes_no_normalized"] = inferred
        detail["yes_no_from_raw"] = True
        return inferred
    return answer


def resolve_short_answer(
    structured_answer: Any,
    llm_raw: Any,
    *,
    max_tokens: int = 50,
    question: str | None = None,
) -> Tuple[str, str, Dict[str, Any]]:
    detail: Dict[str, Any] = {}
    raw_text = str(llm_raw or "")
    has_final = has_final_tag(raw_text)
    detail["llm_has_final"] = has_final
    if has_final:
        final = extract_final_answer(raw_text, max_tokens=max_tokens)
        if final:
            final = _maybe_normalize_yes_no(final, question, raw_text, detail)
            detail["source"] = "llm_raw:FINAL"
            return final, "llm_final", detail

    structured_text = str(structured_answer).strip() if structured_answer is not None else ""
    if structured_text:
        if structured_text.strip().lower() == "insufficient evidence":
            detail["source"] = "intermediate.structured_answer"
            return structured_text, "structured_answer", detail
        if _is_yes_no_question(question):
            normalized = _normalize_yes_no_answer(structured_text)
            if normalized:
                detail["source"] = "intermediate.structured_answer"
                if normalized != structured_text.strip().lower():
                    detail["yes_no_normalized"] = normalized
                return normalized, "structured_answer", detail
            detail["structured_ignored"] = "yes_no_question_non_yesno"
        else:
            detail["source"] = "intermediate.structured_answer"
            return structured_text, "structured_answer", detail

    fallback = extract_final_answer(raw_text, max_tokens=max_tokens)
    if fallback:
        fallback = _maybe_normalize_yes_no(fallback, question, raw_text, detail)
        detail["source"] = "llm_raw:extract_final_answer"
        return fallback, "llm_fallback", detail

    detail["source"] = "llm_raw:empty"
    return "", "empty", detail
