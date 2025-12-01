from __future__ import annotations

import json
import time
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


@dataclass
class LLMResponse:
    content: str
    raw: Dict[str, Any]


class LLMChatClient:
    """Lightweight OpenAI-compatible chat client with simple retries."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 60,
        retries: int = 2,
        stop: Optional[List[str]] = None,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("Both endpoint and model are required for LLM calls")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = max(0, retries)
        self.stop = stop or []

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.stop:
            payload["stop"] = self.stop
        if response_format:
            payload["response_format"] = response_format

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return LLMResponse(content=content, raw=data)
            except requests.RequestException as exc:  # noqa: PERF203
                last_exc = exc
                if attempt >= self.retries:
                    break
                backoff = 2**attempt
                logger.warning("LLM call failed (attempt {}): {}; retrying in {}s", attempt + 1, exc, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"LLM call failed after {self.retries + 1} attempts: {last_exc}")

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.stop:
            payload["stop"] = self.stop
        if response_format:
            payload["response_format"] = response_format

        last_exc: Optional[Exception] = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.retries + 1):
                try:
                    async with session.post(
                        f"{self.endpoint}/chat/completions",
                        json=payload,
                        timeout=self.timeout,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return LLMResponse(content=content, raw=data)
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    last_exc = exc
                    if attempt >= self.retries:
                        break
                    backoff = 2**attempt
                    logger.warning("LLM call failed (attempt {}): {}; retrying in {}s", attempt + 1, exc, backoff)
                    await asyncio.sleep(backoff)
        raise RuntimeError(f"LLM call failed after {self.retries + 1} attempts: {last_exc}")

    @staticmethod
    def safe_parse_json(text: str) -> Any:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                # Attempt to locate JSON substring
                start = cleaned.find("[")
                end = cleaned.rfind("]")
                if start != -1 and end != -1 and end > start:
                    return json.loads(cleaned[start : end + 1])
            except Exception:
                pass
        return None
