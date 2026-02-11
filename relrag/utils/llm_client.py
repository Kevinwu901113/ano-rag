from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import aiohttp
from loguru import logger

from relrag.config.config_loader import config as global_config
from relrag.utils.llm_errors import ContextLengthError, is_context_length_error, parse_error_message
from relrag.utils.llm_stats import get_active_llm_stats
from relrag.utils.vllm_runtime import detect_vllm_served_model
from relrag.utils.token_counter import TokenCounter
from relrag.utils.text_utils import TextUtils


VLLM_ENDPOINT = "http://127.0.0.1:8000/v1"
SERVED_MODEL_NAME = "qwen3-30b-a3b"
HF_MODEL_ID = "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
ALLOW_CUSTOM_ENV = "RELRAG_ALLOW_CUSTOM_LLM"

_DEFAULT_PROFILES = {
    "extract": {"temperature": 0.0, "max_tokens": 256, "thinking": False},
    "generate": {"temperature": 0.2, "max_tokens": 128, "thinking": False},
}
MIN_OUTPUT_TOKENS = 16


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


def _allow_custom_llm() -> bool:
    return os.environ.get(ALLOW_CUSTOM_ENV) == "1"

def _normalize_endpoint(endpoint: Optional[str]) -> str:
    raw = (endpoint or "").strip()
    if not raw:
        return VLLM_ENDPOINT
    normalized = raw.rstrip("/")
    return normalized

def _normalize_custom_endpoint(endpoint: Optional[str]) -> str:
    raw = (endpoint or "").strip()
    if not raw:
        return VLLM_ENDPOINT
    return raw.rstrip("/")


def _normalize_model(model: Optional[str], endpoint: Optional[str]) -> str:
    raw = (model or "").strip()
    if not raw:
        detected = detect_vllm_served_model(endpoint)
        return detected or SERVED_MODEL_NAME

    lowered = raw.lower()
    if lowered == HF_MODEL_ID.lower():
        return SERVED_MODEL_NAME

    # Keep explicit user model unchanged when custom mode is enabled.
    if _allow_custom_llm():
        return raw

    # Default-config model may be stale if server was started with another served name.
    if lowered == SERVED_MODEL_NAME:
        detected = detect_vllm_served_model(endpoint)
        if detected:
            return detected
        return SERVED_MODEL_NAME

    return raw


def _trim_messages(messages: List[Dict[str, str]], max_chars: int = 200) -> List[Dict[str, str]]:
    trimmed: List[Dict[str, str]] = []
    for msg in messages or []:
        content = str(msg.get("content") or "")
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        trimmed.append({**msg, "content": content})
    return trimmed


def _resolve_model_ctx_len(cfg: Dict[str, Any]) -> int:
    vllm_cfg = cfg.get("vllm") if isinstance(cfg.get("vllm"), dict) else {}
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    max_model_len = vllm_cfg.get("max_model_len")
    if max_model_len is None:
        max_model_len = llm_cfg.get("max_context_len")
    return max(256, _coerce_int(max_model_len, 8192))


def _resolve_safety_margin_tokens(cfg: Dict[str, Any]) -> int:
    vllm_cfg = cfg.get("vllm") if isinstance(cfg.get("vllm"), dict) else {}
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    safety_margin = vllm_cfg.get("context_safety_margin")
    if safety_margin is None:
        safety_margin = llm_cfg.get("safety_margin_tokens")
    return max(0, _coerce_int(safety_margin, 256))


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

    # Respect global forbid_think_tags
    if thinking is not False:
        answer_format = cfg.get("answer_format", {})
        if answer_format.get("forbid_think_tags"):
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
        provider: Optional[str] = None,
    ) -> None:
        self.provider = (provider or "vllm").strip().lower()
        if self.provider == "openai":
            self.endpoint = _normalize_custom_endpoint(endpoint)
            self.model = str(model or "")
        else:
            self.endpoint = _normalize_endpoint(endpoint)
            self.model = _normalize_model(model, endpoint=self.endpoint)
        self.llm_profile = llm_profile or "generate"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = max(0, retries)
        self.stop = stop or []
        self.api_key = api_key

    def _is_local_url(self, url: str) -> bool:
        return "127.0.0.1" in url or "localhost" in url

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

    def _resolve_endpoint(self, endpoint_override: Optional[str]) -> str:
        if self.provider == "openai":
            raw = (endpoint_override or self.endpoint or "").strip()
            return raw.rstrip("/")
        return _normalize_endpoint(endpoint_override or self.endpoint)

    def _clamp_payload_max_tokens(
        self,
        payload: Dict[str, Any],
        messages: List[Dict[str, str]],
    ) -> int:
        prompt_tokens_real = TokenCounter.count_messages(messages)
        requested = _coerce_int(payload.get("max_tokens"), MIN_OUTPUT_TOKENS)
        cfg = global_config.load_config()
        model_ctx_len = _resolve_model_ctx_len(cfg)
        safety_margin = _resolve_safety_margin_tokens(cfg)
        allowed = model_ctx_len - safety_margin - prompt_tokens_real
        clamped = min(requested, max(MIN_OUTPUT_TOKENS, allowed))
        if clamped != requested:
            logger.debug(
                "LLM clamp max_tokens: prompt_tokens={} model_ctx_len={} safety_margin={} allowed={} requested={} final={}",
                prompt_tokens_real,
                model_ctx_len,
                safety_margin,
                allowed,
                requested,
                clamped,
            )
            payload["max_tokens"] = clamped
        return prompt_tokens_real

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
        if response_format and self.provider != "openai":
            payload["response_format"] = response_format
        if extra_body and self.provider != "openai":
            payload.setdefault("extra_body", {})
            payload["extra_body"].update(extra_body)
        if self.provider != "openai":
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
        endpoint = self._resolve_endpoint(endpoint_override)
        url = f"{endpoint}/chat/completions"
        stats = get_active_llm_stats()
        prompt_chars = 0
        for msg in messages or []:
            content = str(msg.get("content") or "")
            prompt_chars += len(content)
        payload = self._build_payload(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            response_format=response_format,
            extra_body=extra_body,
            stop=stop,
            profile=profile,
        )
        prompt_tokens_real = self._clamp_payload_max_tokens(payload, messages)
        req_timeout = timeout if timeout is not None else self.timeout

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            start = time.time()
            proxies = {"http": None, "https": None} if self._is_local_url(url) else None
            try:
                if session is not None:
                    resp = session.post(url, json=payload, headers=self._headers(), timeout=req_timeout, proxies=proxies)
                else:
                    resp = requests.post(url, json=payload, headers=self._headers(), timeout=req_timeout, proxies=proxies)
                resp.raise_for_status()
                data = resp.json()
                content = self._extract_content(data)
                if stats is not None:
                    usage = data.get("usage") if isinstance(data, dict) else {}
                    prompt_used = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                    completion_used = usage.get("completion_tokens") if isinstance(usage, dict) else None
                    if completion_used is None:
                        completion_used = TextUtils.rough_token_len(content)
                    finish_reason = None
                    choices = data.get("choices") if isinstance(data, dict) else None
                    if isinstance(choices, list) and choices:
                        finish_reason = choices[0].get("finish_reason")
                    stats.record_llm_attempt(
                        prompt_tokens=prompt_used if prompt_used is not None else prompt_tokens_real,
                        completion_tokens=completion_used,
                        prompt_chars=prompt_chars,
                        completion_chars=len(content),
                        duration_ms=(time.time() - start) * 1000.0,
                        finish_reason=finish_reason,
                        retry=attempt > 0,
                    )
                return LLMResponse(content=content, raw=data)
            except requests.HTTPError as exc:
                last_exc = exc
                response = exc.response
                status = response.status_code if response is not None else None
                if status == 400 and response is not None:
                    message = parse_error_message(response.text or "")
                    if is_context_length_error(message):
                        if stats is not None:
                            stats.record_llm_attempt(
                                prompt_tokens=prompt_tokens_real,
                                completion_tokens=None,
                                prompt_chars=prompt_chars,
                                completion_chars=None,
                                duration_ms=(time.time() - start) * 1000.0,
                                error_type="context_len",
                                retry=attempt > 0,
                            )
                        raise ContextLengthError(message, response_text=response.text) from exc
                    logger.error("LLM 400 Bad Request:\nResponse: {}\nPayload: {}", response.text, json.dumps({
                        "model": payload.get("model"),
                        "messages_sample": _trim_messages(payload.get("messages", [])[:1]),
                        "max_tokens": payload.get("max_tokens"),
                        "temperature": payload.get("temperature"),
                        "has_chat_template_kwargs": "chat_template_kwargs" in payload
                    }, indent=2))
                if stats is not None:
                    stats.record_llm_attempt(
                        prompt_tokens=prompt_tokens_real,
                        completion_tokens=None,
                        prompt_chars=prompt_chars,
                        completion_chars=None,
                        duration_ms=(time.time() - start) * 1000.0,
                        error_type=f"http_{status}" if status is not None else "http_error",
                        retry=attempt > 0,
                    )
                logger.warning("LLM HTTP error (attempt {}): {}", attempt + 1, exc)
            except requests.Timeout as exc:
                last_exc = exc
                logger.warning("LLM call timed out after {}s (attempt {})", req_timeout, attempt + 1)
                if stats is not None:
                    stats.record_llm_attempt(
                        prompt_tokens=prompt_tokens_real,
                        completion_tokens=None,
                        prompt_chars=prompt_chars,
                        completion_chars=None,
                        duration_ms=(time.time() - start) * 1000.0,
                        error_type="timeout",
                        retry=attempt > 0,
                    )
            except requests.RequestException as exc:  # noqa: PERF203
                last_exc = exc
                logger.warning("LLM call failed (attempt {}): {}", attempt + 1, exc)
                if stats is not None:
                    stats.record_llm_attempt(
                        prompt_tokens=prompt_tokens_real,
                        completion_tokens=None,
                        prompt_chars=prompt_chars,
                        completion_chars=None,
                        duration_ms=(time.time() - start) * 1000.0,
                        error_type="request_error",
                        retry=attempt > 0,
                    )
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
        endpoint = self._resolve_endpoint(endpoint_override)
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
        self._clamp_payload_max_tokens(payload, messages)
        req_timeout = timeout if timeout is not None else self.timeout
        proxies = {"http": None, "https": None} if self._is_local_url(url) else None
        if session is not None:
            return session.post(url, json=payload, headers=self._headers(), timeout=req_timeout, proxies=proxies)
        return requests.post(url, json=payload, headers=self._headers(), timeout=req_timeout, proxies=proxies)

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
        endpoint = self._resolve_endpoint(endpoint_override)
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
        self._clamp_payload_max_tokens(payload, messages)
        req_timeout = timeout if timeout is not None else self.timeout
        if isinstance(req_timeout, tuple):
            timeout_obj = aiohttp.ClientTimeout(sock_connect=req_timeout[0], sock_read=req_timeout[1])
        else:
            timeout_obj = aiohttp.ClientTimeout(total=req_timeout)

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                trust_env = not self._is_local_url(url)
                async with aiohttp.ClientSession(timeout=timeout_obj, trust_env=trust_env) as session:
                    async with session.post(url, json=payload, headers=self._headers()) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        content = self._extract_content(data)
                        return LLMResponse(content=content, raw=data)
            except aiohttp.ClientResponseError as exc:
                last_exc = exc
                if exc.status == 400:
                    message = parse_error_message(exc.message or "")
                    if is_context_length_error(message):
                        raise ContextLengthError(message, response_text=exc.message) from exc
                    logger.error("LLM Async 400 Bad Request:\nPayload: {}", json.dumps({
                        "model": payload.get("model"),
                        "messages_sample": _trim_messages(payload.get("messages", [])[:1]),
                        "max_tokens": payload.get("max_tokens"),
                        "temperature": payload.get("temperature"),
                        "has_chat_template_kwargs": "chat_template_kwargs" in payload
                    }, indent=2))
                logger.warning("LLM async HTTP error (attempt {}): {}", attempt + 1, exc)
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
