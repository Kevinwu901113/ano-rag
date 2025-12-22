from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from utils.text_utils import TextUtils


def estimate_tokens(text: str) -> int:
    return TextUtils.rough_token_len(text or "")


def truncate_to_budget(text: str, max_tokens: int) -> Tuple[str, int]:
    if max_tokens <= 0:
        tokens = estimate_tokens(text)
        return text, tokens
    tokens = estimate_tokens(text)
    if tokens <= max_tokens:
        return text, tokens
    if not text:
        return "", 0
    ratio = max_tokens / max(tokens, 1)
    char_limit = max(1, int(len(text) * ratio))
    trimmed = text[:char_limit].rstrip()
    while trimmed and estimate_tokens(trimmed) > max_tokens:
        char_limit = max(1, int(len(trimmed) * 0.9))
        trimmed = trimmed[:char_limit].rstrip()
    return trimmed, estimate_tokens(trimmed)


def _coerce_items(items: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"text": item})
        elif isinstance(item, dict):
            if "text" not in item:
                continue
            normalized.append(dict(item))
    return normalized


def pack_contexts(
    items: Iterable[Any],
    budget_tokens: int,
    *,
    sep: str = "\n\n",
) -> Tuple[str, List[Dict[str, Any]], int]:
    normalized = _coerce_items(items)
    if budget_tokens <= 0:
        joined = sep.join([it.get("text", "") for it in normalized])
        return joined, normalized, estimate_tokens(joined)

    sep_tokens = estimate_tokens(sep)
    used = 0
    kept: List[Dict[str, Any]] = []
    parts: List[str] = []

    for idx, item in enumerate(normalized):
        text = item.get("text") or ""
        extra_sep = sep_tokens if parts else 0
        remaining = budget_tokens - used - extra_sep
        if remaining <= 0:
            break
        tokens = estimate_tokens(text)
        truncated = False
        if tokens > remaining:
            text, tokens = truncate_to_budget(text, remaining)
            truncated = True
        if not text:
            break
        used += extra_sep + tokens
        payload = dict(item)
        payload["text"] = text
        if truncated:
            payload["truncated"] = True
        payload["tokens"] = tokens
        kept.append(payload)
        parts.append(text)
        if truncated:
            break

    joined = sep.join(parts)
    return joined, kept, used
