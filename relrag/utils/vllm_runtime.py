from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from loguru import logger

_MODEL_CACHE: Dict[str, str] = {}
_CACHE_LOCK = threading.Lock()
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _normalize_endpoint(endpoint: Optional[str]) -> str:
    return str(endpoint or "").strip().rstrip("/")


def _model_urls(endpoint: str) -> List[str]:
    if not endpoint:
        return []
    urls: List[str] = []
    if endpoint.endswith("/v1"):
        urls.append(f"{endpoint}/models")
        root = endpoint[: -len("/v1")]
        if root:
            urls.append(f"{root}/models")
    else:
        urls.append(f"{endpoint}/v1/models")
        urls.append(f"{endpoint}/models")
    dedup: List[str] = []
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        dedup.append(url)
    return dedup


def _extract_model_id(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if model_id:
                return model_id
    model_id = str(payload.get("id") or "").strip()
    return model_id or None


def _is_local_endpoint(endpoint: str) -> bool:
    raw = endpoint if "://" in endpoint else f"http://{endpoint}"
    parsed = urlparse(raw)
    host = (parsed.hostname or "").strip().lower()
    return host in _LOCAL_HOSTS


def detect_vllm_served_model(endpoint: Optional[str], timeout_s: float = 1.5) -> Optional[str]:
    normalized_endpoint = _normalize_endpoint(endpoint)
    if not normalized_endpoint:
        return None

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(normalized_endpoint)
    if cached:
        return cached

    force_no_proxy = _is_local_endpoint(normalized_endpoint)
    urls = _model_urls(normalized_endpoint)
    for url in urls:
        try:
            if force_no_proxy:
                with requests.Session() as session:
                    session.trust_env = False
                    response = session.get(url, timeout=timeout_s)
            else:
                response = requests.get(url, timeout=timeout_s)
            if response.status_code >= 400:
                continue
            payload = response.json()
            model_id = _extract_model_id(payload)
            if not model_id:
                continue
            with _CACHE_LOCK:
                _MODEL_CACHE[normalized_endpoint] = model_id
            return model_id
        except Exception:
            continue
    return None


def resolve_vllm_endpoint_model(
    *,
    endpoint_override: Optional[str],
    model_override: Optional[str],
    vllm_cfg: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    cfg_vllm = vllm_cfg if isinstance(vllm_cfg, dict) else {}
    endpoint = _normalize_endpoint(endpoint_override or cfg_vllm.get("endpoint"))
    configured_model = str(cfg_vllm.get("model") or "").strip()
    explicit_model = str(model_override or "").strip()

    if not endpoint:
        raise ValueError("LLM endpoint/model is required (use args or config)")

    if explicit_model:
        return endpoint, explicit_model

    runtime_model = detect_vllm_served_model(endpoint)
    if runtime_model:
        if configured_model and configured_model != runtime_model:
            logger.info(
                "Detected running vLLM model {} at {}; overriding configured model {}",
                runtime_model,
                endpoint,
                configured_model,
            )
        return endpoint, runtime_model

    if configured_model:
        return endpoint, configured_model

    raise ValueError("LLM endpoint/model is required (use args or config)")
