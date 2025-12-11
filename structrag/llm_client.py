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

    @staticmethod
    def _shorten_payload(data: Any, limit: int = 400) -> str:
        try:
            txt = json.dumps(data, ensure_ascii=False)
        except Exception:
            txt = str(data)
        return txt if len(txt) <= limit else txt[:limit] + "...(truncated)"

    @staticmethod
    def _shorten_text(text: str, limit: int = 400) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"

    def _extract_content(self, data: Dict[str, Any]) -> str:
        if "error" in data:
            raise RuntimeError(f"LLM error response: {self._shorten_payload(data['error'])}")
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"LLM response missing choices: {self._shorten_payload(data)}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not message:
            raise RuntimeError(f"LLM response missing message field: {self._shorten_payload(data)}")
        content = message.get("content")
        if content is None:
            raise RuntimeError(f"LLM response missing content: {self._shorten_payload(data)}")
        return content

    def _candidate_urls(self) -> List[str]:
        """
        Build a list of possible chat endpoints to be more lenient across servers:
        - Tries with and without /v1 suffix
        - Tries both /chat/completions (OpenAI style) and /chat (llama.cpp style)
        - Tries /completions for servers exposing only completion-style endpoints
        """
        base = self.endpoint.rstrip("/")
        bases = [base]
        if base.endswith("/v1"):
            bases.append(base[:-3].rstrip("/"))
        else:
            bases.append(f"{base}/v1")
        urls: List[str] = []
        for b in bases:
            urls.append(f"{b}/chat/completions")
            urls.append(f"{b}/chat")
            urls.append(f"{b}/completions")
        # preserve order but drop duplicates
        return list(dict.fromkeys(urls))

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Flatten chat messages to a simple prompt for completion-only endpoints.
        """
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nassistant:"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        url: str,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        temp = self.temperature if temperature is None else temperature
        max_tok = self.max_tokens if max_tokens is None else max_tokens
        if url.endswith("/completions") and not url.endswith("/chat/completions"):
            # completion-style
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": self._messages_to_prompt(messages),
                "temperature": temp,
                "max_tokens": max_tok,
                "stream": False,
            }
        else:
            # chat-style
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tok,
                "stream": False,
            }
            if self.stop:
                payload["stop"] = self.stop
            if response_format:
                payload["response_format"] = response_format
        return payload

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: int = 120,
        retries: int = 2,
        stop: Optional[List[str]] = None,
        api_key: str = "sk-no-key-required",
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
        self.api_key = api_key

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        urls = self._candidate_urls()

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            for url in urls:
                try:
                    payload = self._build_payload(
                        url,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                    )
                    resp = requests.post(
                        url,
                        json=payload,
                        headers=self._headers(),
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = self._extract_content(data)
                    return LLMResponse(content=content, raw=data)
                except requests.Timeout as exc:
                    last_exc = exc
                    logger.warning(
                        "LLM call timed out after {}s at {}; consider increasing the timeout",
                        self.timeout,
                        url,
                    )
                    break  # no point in trying alternate paths on a timeout
                except RuntimeError as exc:
                    last_exc = exc
                    # If server complains about endpoint, try next candidate before backing off
                    if "Unexpected endpoint" in str(exc):
                        logger.warning("LLM endpoint rejected path {}; trying alternate URL", url)
                        continue
                except requests.HTTPError as exc:
                    body = ""
                    if exc.response is not None:
                        try:
                            body = exc.response.text
                        except Exception:
                            body = ""
                    last_exc = requests.HTTPError(
                        f"{exc} body={self._shorten_text(body)}", response=exc.response
                    )
                    # Try the alternate URL if 404/405, otherwise respect retries/backoff
                    status = exc.response.status_code if exc.response is not None else None
                    if status in {404, 405} and url != urls[-1]:
                        logger.warning("LLM endpoint {} at {}; trying alternate URL", status, url)
                        continue
                    if attempt >= self.retries:
                        break
                except requests.RequestException as exc:  # noqa: PERF203
                    last_exc = exc
                    if attempt >= self.retries:
                        break
            backoff = 2**attempt
            logger.warning("LLM call failed (attempt {}): {}; retrying in {}s", attempt + 1, last_exc, backoff)
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
        urls = self._candidate_urls()

        last_exc: Optional[Exception] = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.retries + 1):
                for url in urls:
                    try:
                        payload = self._build_payload(
                            url,
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=response_format,
                        )
                        async with session.post(
                            url,
                            json=payload,
                            headers=self._headers(),
                            timeout=self.timeout,
                        ) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                            content = self._extract_content(data)
                            return LLMResponse(content=content, raw=data)
                    except asyncio.TimeoutError as exc:
                        last_exc = exc
                        logger.warning(
                            "LLM call timed out after {}s at {}; consider increasing the timeout",
                            self.timeout,
                            url,
                        )
                        break  # avoid pointless alternate endpoints on timeout
                    except RuntimeError as exc:
                        last_exc = exc
                        if "Unexpected endpoint" in str(exc):
                            logger.warning("LLM endpoint rejected path {}; trying alternate URL", url)
                            continue
                    except aiohttp.ClientResponseError as exc:
                        text = ""
                        try:
                            text = await resp.text()
                        except Exception:
                            text = ""
                        last_exc = aiohttp.ClientResponseError(
                            exc.request_info,
                            exc.history,
                            status=exc.status,
                            message=f"{exc.message} body={self._shorten_text(text)}",
                            headers=exc.headers,
                        )
                        if exc.status in {404, 405} and url != urls[-1]:
                            logger.warning("LLM endpoint {} at {}; trying alternate URL", exc.status, url)
                            continue
                        if attempt >= self.retries:
                            break
                    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                        last_exc = exc
                        if attempt >= self.retries:
                            break
                backoff = 2**attempt
                logger.warning("LLM call failed (attempt {}): {}; retrying in {}s", attempt + 1, last_exc, backoff)
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
