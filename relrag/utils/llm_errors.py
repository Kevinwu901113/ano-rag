import json
from typing import Optional


class ContextLengthError(RuntimeError):
    def __init__(self, message: str, *, response_text: Optional[str] = None) -> None:
        super().__init__(message)
        self.response_text = response_text


def _extract_error_message(raw_text: str) -> str:
    if not raw_text:
        return ""
    try:
        data = json.loads(raw_text)
    except ValueError:
        return raw_text.strip()
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            message = err.get("message")
            if message:
                return str(message)
    return str(data).strip()


def parse_error_message(raw_text: str) -> str:
    message = _extract_error_message(raw_text)
    if message:
        return message
    return raw_text.strip()


def is_context_length_error(message: str) -> bool:
    lowered = (message or "").lower()
    if not lowered:
        return False
    if "maximum context length" in lowered:
        return True
    if "context length" in lowered and ("token" in lowered or "max" in lowered):
        return True
    if "max_tokens" in lowered and "too large" in lowered:
        return True
    if "request has" in lowered and "input token" in lowered:
        return True
    return False
