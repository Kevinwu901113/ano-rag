"""
Common utilities for cleaning model answers.
"""

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

def _enforce_short_answer(text: str) -> str:
    """
    Helper to truncate or normalize short answer.
    """
    if not text:
        return ""
    return text.strip()

def clean_model_answer(text: str) -> str:
    """
    Composite cleaner: strips reasoning and enforces short answer format if needed.
    """
    cleaned = _strip_reasoning(text)
    return _enforce_short_answer(cleaned)
