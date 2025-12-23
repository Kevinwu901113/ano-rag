from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import aiohttp
from loguru import logger

from config.config_loader import config as global_config


VLLM_ENDPOINT = "http://127.0.0.1:8000/v1"
SERVED_MODEL_NAME = "qwen3-30b-a3b"
HF_MODEL_ID = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"

_DEFAULT_PROFILES = {
    "extract": {"temperature": 0.0, "max_tokens": 256, "thinking": False},
    "generate": {"temperature": 0.2, "max_tokens": 128, "thinking": None},
}


@dataclass(frozen=True)
class LLMProfileConfig:
    name: str
    temperature: float
    max_tokens: int
    thinking: Optional[bool]


@dataclass
class LLMResponse:
    content: str
    raw: Dict[str, Any]


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _normalize_endpoint(endpoint: Optional[str]) -> str:
    raw = (endpoint or "").strip()
    if not raw:
        return VLLM_ENDPOINT
    normalized = raw.rstrip("/")
    if normalized != VLLM_ENDPOINT:
        logger.warning("Overriding LLM endpoint {} -> {}", normalized, VLLM_ENDPOINT)
        return VLLM_ENDPOINT
    return normalized


def _normalize_model(model: Optional[str]) -> str:
    raw = (model or "").strip()
    if not raw:
        return SERVED_MODEL_NAME
    lowered = raw.lower()
    if lowered in {SERVED_MODEL_NAME, HF_MODEL_ID.lower()}:
        return SERVED_MODEL_NAME
    logger.warning("Overriding LLM model {} -> {}", raw, SERVED_MODEL_NAME)
    return SERVED_MODEL_NAME


def get_profile_config(profile: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> LLMProfileConfig:
    resolved_profile = (profile or "generate").strip().lower()
    cfg = config or global_config.load_config()
    profile_cfg = (cfg.get("llm_profiles") or {}).get(resolved_profile, {})
    defaults = _DEFAULT_PROFILES.get(resolved_profile, _DEFAULT_PROFILES["generate"])
    temperature = _coerce_float(profile_cfg.get("temperature"), defaults["temperature"])
    max_tokens = _coerce_int(profile_cfg.get("max_tokens"), defaults["max_tokens"])
    thinking = profile_cfg.get("thinking")
    if thinking is None:
        thinking = defaults.get("thinking")
    if resolved_profile == "extract":
        thinking = False
    return LLMProfileConfig(
        name=resolved_profile,
        temperature=temperature,
        max_tokens=max(1, max_tokens),
        thinking=thinking,
    )


def get_profile_snapshot(
    profile: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    thinking: Optional[bool] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = get_profile_config(profile, config=config)
    thinking_value = cfg.thinking if thinking is None else bool(thinking)
    if cfg.name == "extract":
        thinking_value = False
    return {
        "endpoint": VLLM_ENDPOINT,
        "model": SERVED_MODEL_NAME,
        "temperature": cfg.temperature if temperature is None else float(temperature),
        "max_tokens": cfg.max_tokens if max_tokens is None else int(max_tokens),
        "thinking": thinking_value,
    }


def get_all_profile_snapshots(
    *,
    config: Optional[Dict[str, Any]] = None,
    generate_max_tokens: Optional[int] = None,
    generate_temperature: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    return {
        "extract": get_profile_snapshot("extract", config=config),
        "generate": get_profile_snapshot(
            "generate",
            config=config,
            max_tokens=generate_max_tokens,
            temperature=generate_temperature,
        ),
    }


class LLMChatClient:
    """OpenAI-compatible vLLM chat client with extract/generate profile controls."""

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        llm_profile: str = "generate",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float | tuple[float, float] = 120,
        retries: int = 2,
        stop: Optional[List[str]] = None,
        api_key: str = "sk-no-key-required",
    ) -> None:
        self.endpoint = _normalize_endpoint(endpoint)
        self.model = _normalize_model(model)
        self.llm_profile = llm_profile or "generate"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = max(0, retries)
        self.stop = stop or []
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _extract_content(self, data: Dict[str, Any]) -> str:
        if "error" in data:
            raise RuntimeError(f"LLM error response: {data['error']}")
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"LLM response missing choices: {json.dumps(data)[:200]}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not message:
            raise RuntimeError(f"LLM response missing message: {json.dumps(data)[:200]}")
        content = message.get("content")
        if content is None:
            raise RuntimeError(f"LLM response missing content: {json.dumps(data)[:200]}")
        return content

    def _apply_profile(
        self,
        payload: Dict[str, Any],
        profile: str,
    ) -> None:
        cfg = get_profile_config(profile)
        if cfg.thinking is False:
            payload.setdefault("chat_template_kwargs", {})
            payload["chat_template_kwargs"]["enable_thinking"] = False

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
        extra_body: Optional[Dict[str, Any]],
        stop: Optional[List[str]],
        profile: str,
    ) -> Dict[str, Any]:
        cfg = get_profile_config(profile)
        temp = cfg.temperature if temperature is None else float(temperature)
        max_tok = cfg.max_tokens if max_tokens is None else int(max_tokens)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "stream": False,
        }
        if self.stop or stop:
            payload["stop"] = stop if stop is not None else self.stop
        if response_format:
            payload["response_format"] = response_format
        if extra_body:
            payload.setdefault("extra_body", {})
            payload["extra_body"].update(extra_body)
        self._apply_profile(payload, profile)
        return payload

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        llm_profile: Optional[str] = None,
        endpoint_override: Optional[str] = None,
        timeout: Optional[float | tuple[float, float]] = None,
        session: Optional[requests.Session] = None,
    ) -> LLMResponse:
        profile = (llm_profile or self.llm_profile or "generate").strip().lower()
        endpoint = _normalize_endpoint(endpoint_override or self.endpoint)
        url = f"{endpoint}/chat/completions"
        payload = self._build_payload(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            response_format=response_format,
            extra_body=extra_body,
            stop=stop,
            profile=profile,
        )
        req_timeout = timeout if timeout is not None else self.timeout

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                if session is not None:
                    resp = session.post(url, json=payload, headers=self._headers(), timeout=req_timeout)
                else:
                    resp = requests.post(url, json=payload, headers=self._headers(), timeout=req_timeout)
                resp.raise_for_status()
                data = resp.json()
                content = self._extract_content(data)
                return LLMResponse(content=content, raw=data)
            except requests.Timeout as exc:
                last_exc = exc
                logger.warning("LLM call timed out after {}s (attempt {})", req_timeout, attempt + 1)
            except requests.RequestException as exc:  # noqa: PERF203
                last_exc = exc
                logger.warning("LLM call failed (attempt {}): {}", attempt + 1, exc)
            if attempt < self.retries:
                backoff = 2 ** attempt
                time.sleep(backoff)
        raise RuntimeError(f"LLM call failed after {self.retries + 1} attempts: {last_exc}")

    def post_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        llm_profile: Optional[str] = None,
        endpoint_override: Optional[str] = None,
        timeout: Optional[float | tuple[float, float]] = None,
        session: Optional[requests.Session] = None,
    ) -> requests.Response:
        profile = (llm_profile or self.llm_profile or "generate").strip().lower()
        endpoint = _normalize_endpoint(endpoint_override or self.endpoint)
        url = f"{endpoint}/chat/completions"
        payload = self._build_payload(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            response_format=response_format,
            extra_body=extra_body,
            stop=stop,
            profile=profile,
        )
        req_timeout = timeout if timeout is not None else self.timeout
        if session is not None:
            return session.post(url, json=payload, headers=self._headers(), timeout=req_timeout)
        return requests.post(url, json=payload, headers=self._headers(), timeout=req_timeout)

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        llm_profile: Optional[str] = None,
        endpoint_override: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        profile = (llm_profile or self.llm_profile or "generate").strip().lower()
        endpoint = _normalize_endpoint(endpoint_override or self.endpoint)
        url = f"{endpoint}/chat/completions"
        payload = self._build_payload(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            response_format=response_format,
            extra_body=extra_body,
            stop=stop,
            profile=profile,
        )
        req_timeout = timeout if timeout is not None else self.timeout
        if isinstance(req_timeout, tuple):
            timeout_obj = aiohttp.ClientTimeout(sock_connect=req_timeout[0], sock_read=req_timeout[1])
        else:
            timeout_obj = aiohttp.ClientTimeout(total=req_timeout)

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                    async with session.post(url, json=payload, headers=self._headers()) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        content = self._extract_content(data)
                        return LLMResponse(content=content, raw=data)
            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning("LLM async call timed out (attempt {})", attempt + 1)
            except aiohttp.ClientError as exc:
                last_exc = exc
                logger.warning("LLM async call failed (attempt {}): {}", attempt + 1, exc)
            if attempt < self.retries:
                backoff = 2 ** attempt
                await asyncio.sleep(backoff)
        raise RuntimeError(f"LLM async call failed after {self.retries + 1} attempts: {last_exc}")
