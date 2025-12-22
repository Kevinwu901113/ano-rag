from __future__ import annotations

import re
from typing import Iterable, Tuple


FINAL_TAG = "FINAL:"


def build_final_instruction() -> str:
    return "Your output must end with a single line in the form: FINAL: <answer>."


def has_final_tag(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"^\s*final\s*:", text, flags=re.IGNORECASE | re.MULTILINE))


def extract_final_answer(text: str, *, max_tokens: int = 50) -> str:
    if not text:
        return ""
    lines = [ln.rstrip() for ln in str(text).splitlines() if ln.strip()]
    final_line = ""
    for ln in reversed(lines):
        match = re.match(r"^\s*final\s*:\s*(.*)$", ln, flags=re.IGNORECASE)
        if match:
            final_line = match.group(1).strip()
            break
    if not final_line and lines:
        final_line = lines[-1].strip()
    if not final_line:
        return ""
    tokens = final_line.split()
    if max_tokens > 0 and len(tokens) > max_tokens:
        final_line = " ".join(tokens[:max_tokens]).strip()
    return final_line


def _strip_tag_markers(text: str) -> str:
    if not text:
        return ""
    # Only remove the tag markers, keep the text inside.
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
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
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
