from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def _normalize_base_url(base_url: Optional[str]) -> str:
    raw = (base_url or DEFAULT_OPENAI_BASE_URL).strip()
    if not raw:
        return DEFAULT_OPENAI_BASE_URL
    return raw.rstrip("/")


def _extract_error_message(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        message = err.get("message")
        if message:
            return str(message)
    return str(data)[:200]


def _extract_content(data: Dict[str, Any]) -> str:
    if "error" in data:
        raise RuntimeError(f"OpenAI error response: {data['error']}")
    choices = data.get("choices")
    if not choices:
        raise RuntimeError(f"OpenAI response missing choices: {json.dumps(data)[:200]}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not message:
        raise RuntimeError(f"OpenAI response missing message: {json.dumps(data)[:200]}")
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"OpenAI response missing content: {json.dumps(data)[:200]}")
    return str(content)


def _sleep_backoff(attempt: int, base_sec: float, max_sec: float) -> None:
    delay = min(max_sec, base_sec * (2 ** attempt))
    jitter = random.uniform(0.0, delay * 0.1)
    time.sleep(delay + jitter)


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_sec: float = 60.0,
    max_retries: int = 2,
    retry_backoff_sec: float = 1.0,
    retry_backoff_max_sec: float = 20.0,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    if not api_key:
        raise ValueError("OpenAI API key is required.")
    if not model:
        raise ValueError("OpenAI model is required.")

    url = f"{_normalize_base_url(base_url)}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if extra_body:
        payload.update(extra_body)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code in _RETRY_STATUS:
                message = _extract_error_message(resp)
                logger.warning("OpenAI {} (attempt {}): {}", resp.status_code, attempt + 1, message)
                if attempt < max_retries:
                    _sleep_backoff(attempt, retry_backoff_sec, retry_backoff_max_sec)
                    continue
            resp.raise_for_status()
            data = resp.json()
            return _extract_content(data)
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            message = _extract_error_message(exc.response) if exc.response is not None else str(exc)
            if status in _RETRY_STATUS and attempt < max_retries:
                logger.warning("OpenAI HTTP {} (attempt {}): {}", status, attempt + 1, message)
                _sleep_backoff(attempt, retry_backoff_sec, retry_backoff_max_sec)
                continue
            if status in {401, 403}:
                logger.error("OpenAI auth error {}: {}", status, message)
            else:
                logger.error("OpenAI HTTP error {}: {}", status, message)
        except requests.Timeout as exc:
            last_exc = exc
            logger.warning("OpenAI request timed out after {}s (attempt {})", timeout_sec, attempt + 1)
            if attempt < max_retries:
                _sleep_backoff(attempt, retry_backoff_sec, retry_backoff_max_sec)
                continue
        except requests.RequestException as exc:  # noqa: PERF203
            last_exc = exc
            logger.warning("OpenAI request failed (attempt {}): {}", attempt + 1, exc)
            if attempt < max_retries:
                _sleep_backoff(attempt, retry_backoff_sec, retry_backoff_max_sec)
                continue

    raise RuntimeError(f"OpenAI request failed after {max_retries + 1} attempts: {last_exc}")
