from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

from relrag.utils.output_eval import extract_final_answer, has_final_tag


def sha1_text(text: str) -> str:
    payload = (text or "").encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def resolve_short_answer(
    structured_answer: Any,
    llm_raw: Any,
    *,
    max_tokens: int = 50,
) -> Tuple[str, str, Dict[str, Any]]:
    detail: Dict[str, Any] = {}
    structured_text = str(structured_answer).strip() if structured_answer is not None else ""
    if structured_text:
        detail["source"] = "intermediate.structured_answer"
        return structured_text, "structured_answer", detail

    raw_text = str(llm_raw or "")
    has_final = has_final_tag(raw_text)
    detail["llm_has_final"] = has_final
    if has_final:
        final = extract_final_answer(raw_text, max_tokens=max_tokens)
        if final:
            detail["source"] = "llm_raw:FINAL"
            return final, "llm_final", detail

    fallback = extract_final_answer(raw_text, max_tokens=max_tokens)
    if fallback:
        detail["source"] = "llm_raw:extract_final_answer"
        return fallback, "llm_fallback", detail

    detail["source"] = "llm_raw:empty"
    return "", "empty", detail
