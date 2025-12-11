"""
Common utilities for cleaning model answers.
"""
import re


def _strip_reasoning(text: str) -> str:
    """
    Remove <think>...</think> tags and their content from the text.
    Handles cases where only the opening tag is present.
    """
    if not text:
        return ""
    output = text
    while True:
        start = output.find("<think>")
        if start == -1:
            break
        end = output.find("</think>", start + len("<think>"))
        if end == -1:
            # Only start tag found, remove everything after it
            output = output[:start]
            break
        output = output[:start] + output[end + len("</think>") :]
    return output.strip()


def _extract_after_marker(text: str) -> str:
    """
    If the model used an explicit marker like 'Answer:' or '答案：', keep the trailing part.
    """
    if not text:
        return ""
    # Find the last occurrence to avoid grabbing an early justification
    match = list(re.finditer(r"(?:answer|答案)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL))
    if match:
        return match[-1].group(1).strip()
    return text


def _short_line(text: str) -> str:
    """
    Keep the last non-empty line if it looks short enough to be a final answer.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()
    last = lines[-1]
    if len(last.split()) <= 30:
        return last
    return text.strip()


def _enforce_short_answer(text: str) -> str:
    """
    Favor a concise final span over long explanations.
    """
    if not text:
        return ""
    trimmed = _extract_after_marker(text)
    trimmed = _short_line(trimmed)
    return trimmed.strip()


def clean_model_answer(text: str) -> str:
    """
    Composite cleaner: strips reasoning and enforces short answer format if needed.
    """
    cleaned = _strip_reasoning(text)
    return _enforce_short_answer(cleaned)
