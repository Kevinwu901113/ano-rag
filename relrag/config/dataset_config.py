from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional


DEFAULT_OPENAI_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "model": "gpt-4",
    "api_key_env": "OPENAI_API_KEY",
    "temperature": 0.2,
    "max_tokens": 128,
    "base_url": "https://api.openai.com/v1",
    "timeout_sec": 60.0,
    "max_retries": 2,
    "retry_backoff_sec": 1.0,
    "retry_backoff_max_sec": 20.0,
    "system_prompt_name": "",
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_dataset_config(cfg: Dict[str, Any], dataset_key: str) -> Dict[str, Any]:
    dataset_cfg: Dict[str, Any] = {}
    datasets_cfg = cfg.get("datasets")
    if isinstance(datasets_cfg, dict):
        nested = datasets_cfg.get(dataset_key)
        if isinstance(nested, dict):
            dataset_cfg = deepcopy(nested)
    legacy_cfg = cfg.get(dataset_key)
    if isinstance(legacy_cfg, dict):
        dataset_cfg = _deep_merge(legacy_cfg, dataset_cfg)
    return dataset_cfg


def resolve_reader(
    reader_arg: Optional[str],
    dataset_cfg: Dict[str, Any],
    *,
    default_reader: str = "vllm",
) -> str:
    raw = reader_arg if reader_arg is not None else dataset_cfg.get("reader")
    if raw is None:
        raw = default_reader
    normalized = str(raw).strip().lower()
    if not normalized:
        normalized = default_reader
    if normalized not in {"vllm", "openai"}:
        raise ValueError(f"Unknown reader type: {raw}")
    return normalized


def resolve_openai_config(
    cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    dataset_openai = dataset_cfg.get("openai") if isinstance(dataset_cfg.get("openai"), dict) else {}
    merged = _deep_merge(DEFAULT_OPENAI_CONFIG, base_cfg)
    merged = _deep_merge(merged, dataset_openai)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return _normalize_openai_config(merged)


def resolve_openai_api_key(
    openai_cfg: Dict[str, Any],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    cfg_key = openai_cfg.get("api_key")
    if cfg_key:
        return str(cfg_key)
    env_name = openai_cfg.get("api_key_env", DEFAULT_OPENAI_CONFIG["api_key_env"])
    env_map = env or os.environ
    key = env_map.get(str(env_name))
    if not key:
        raise ValueError(f"OpenAI API key not found. Set {env_name} or pass --openai_api_key.")
    return key


def _normalize_openai_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(cfg or {})
    normalized["enabled"] = bool(normalized.get("enabled", True))
    normalized["model"] = str(normalized.get("model") or DEFAULT_OPENAI_CONFIG["model"])
    normalized["api_key_env"] = str(normalized.get("api_key_env") or DEFAULT_OPENAI_CONFIG["api_key_env"])
    normalized["temperature"] = _coerce_float(
        normalized.get("temperature", DEFAULT_OPENAI_CONFIG["temperature"]),
        DEFAULT_OPENAI_CONFIG["temperature"],
    )
    normalized["max_tokens"] = _coerce_int(
        normalized.get("max_tokens", DEFAULT_OPENAI_CONFIG["max_tokens"]),
        DEFAULT_OPENAI_CONFIG["max_tokens"],
    )
    normalized["base_url"] = str(normalized.get("base_url") or DEFAULT_OPENAI_CONFIG["base_url"])
    normalized["timeout_sec"] = _coerce_float(
        normalized.get("timeout_sec", DEFAULT_OPENAI_CONFIG["timeout_sec"]),
        DEFAULT_OPENAI_CONFIG["timeout_sec"],
    )
    normalized["max_retries"] = _coerce_int(
        normalized.get("max_retries", DEFAULT_OPENAI_CONFIG["max_retries"]),
        DEFAULT_OPENAI_CONFIG["max_retries"],
    )
    normalized["retry_backoff_sec"] = _coerce_float(
        normalized.get("retry_backoff_sec", DEFAULT_OPENAI_CONFIG["retry_backoff_sec"]),
        DEFAULT_OPENAI_CONFIG["retry_backoff_sec"],
    )
    normalized["retry_backoff_max_sec"] = _coerce_float(
        normalized.get("retry_backoff_max_sec", DEFAULT_OPENAI_CONFIG["retry_backoff_max_sec"]),
        DEFAULT_OPENAI_CONFIG["retry_backoff_max_sec"],
    )
    name = normalized.get("system_prompt_name")
    if name is None:
        if "system_prompt" not in normalized:
            normalized["system_prompt_name"] = DEFAULT_OPENAI_CONFIG["system_prompt_name"]
    else:
        normalized["system_prompt_name"] = str(name)
    return normalized
