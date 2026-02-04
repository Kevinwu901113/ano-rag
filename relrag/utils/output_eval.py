from __future__ import annotations

import re
from typing import Iterable, Tuple


def has_final_tag(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"\bfinal\s*:", text, flags=re.IGNORECASE))


_ANSWER_PATTERNS = [
    re.compile(r"(?:^|[\n\r])\s*(?:final answer|answer|ans|response)\s*[:\-]\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:the\s+)?answer\s+is\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:therefore|thus|hence|so)[^\n\r]{0,120}?\banswer(?:\s+is)?\s+(.+)", re.IGNORECASE),
]


def _truncate_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    if max_tokens <= 0:
        return text.strip()
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text.strip()
    return " ".join(tokens[:max_tokens]).strip()


def _clean_candidate(text: str) -> str:
    if not text:
        return ""
    candidate = text.strip()
    if not candidate:
        return ""
    if re.search(r"\binsufficient evidence\b", candidate, flags=re.IGNORECASE):
        return "Insufficient evidence"
    candidate = candidate.strip(" \t\"'`")
    candidate = re.sub(r"[\s\-\u2013\u2014,;.!?]+$", "", candidate).strip()
    lowered = candidate.lower()
    for sep in (" because ", " since ", " as ", " therefore ", " thus ", " so "):
        idx = lowered.find(sep)
        if idx > 0:
            candidate = candidate[:idx].strip()
            break
    return candidate


def _find_answer_candidate(text: str) -> str:
    if not text:
        return ""
    last = ""
    for pattern in _ANSWER_PATTERNS:
        for match in pattern.finditer(text):
            value = (match.group(1) or "").strip()
            if value:
                last = value
    return last


def _clean_text_for_extract(text: str) -> str:
    if not text:
        return ""
    output = _strip_tag_markers(str(text))
    output = _strip_code_fence_markers(output)
    return output.strip()


def extract_final_answer(text: str, *, max_tokens: int = 50) -> str:
    if not text:
        return ""
    raw_text = str(text)
    lines = [ln.rstrip() for ln in raw_text.splitlines() if ln.strip()]
    final_line = ""
    for ln in reversed(lines):
        match = re.search(r"\bfinal\s*:\s*(.*)$", ln, flags=re.IGNORECASE)
        if match:
            final_line = match.group(1).strip()
            break
    if final_line:
        return _truncate_tokens(_clean_candidate(final_line), max_tokens)

    cleaned = _clean_text_for_extract(raw_text)
    candidate = _find_answer_candidate(cleaned)
    if not candidate:
        cleaned_lines = [ln.rstrip() for ln in cleaned.splitlines() if ln.strip()]
        if cleaned_lines:
            candidate = cleaned_lines[-1].strip()
    candidate = _clean_candidate(candidate)
    return _truncate_tokens(candidate, max_tokens)


def _strip_tag_markers(text: str) -> str:
    if not text:
        return ""
    # Remove <think>...</think> blocks including content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove any dangling <think> tag to end of text
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Cleanup any remaining tags just in case
    text = text.replace("<think>", "").replace("</think>", "")
    return text


def _strip_code_fence_markers(text: str) -> str:
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        lines.append(ln)
    return "\n".join(lines)


def normalize_text(
    text: str,
    *,
    lowercase: bool = False,
    normalize_punct: bool = False,
) -> str:
    if text is None:
        return ""
    output = str(text)
    output = _strip_tag_markers(output)
    output = _strip_code_fence_markers(output)
    output = "\n".join([ln.strip() for ln in output.splitlines()])
    output = " ".join(output.split())
    if lowercase:
        output = output.lower()
    if normalize_punct:
        output = _normalize_punct(output)
    return output.strip()


def _normalize_punct(text: str) -> str:
    if not text:
        return ""
    table = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\uff0c": ",",
        "\u3002": ".",
        "\uff1b": ";",
        "\uff1a": ":",
        "\uff01": "!",
        "\uff1f": "?",
        "\uff08": "(",
        "\uff09": ")",
        "\u3010": "[",
        "\u3011": "]",
    }
    return "".join(table.get(ch, ch) for ch in text)


def format_metrics(pred_finals: Iterable[str], *, require_final: bool = True) -> Tuple[float, float]:
    total = 0
    invalid = 0
    missing_final = 0
    for raw in pred_finals:
        total += 1
        text = raw if raw is not None else ""
        if require_final and not has_final_tag(text):
            missing_final += 1
        if not extract_final_answer(text):
            invalid += 1
    if total == 0:
        return 0.0, 0.0
    return invalid / total, missing_final / total
