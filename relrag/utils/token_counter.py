from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List

from loguru import logger

from relrag.config.config_loader import config as global_config
from relrag.utils.text_utils import TextUtils


FALLBACK_MESSAGE_OVERHEAD = 4
FALLBACK_BASE_OVERHEAD = 8


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


class TokenCounter:
    """Tokenizer-backed counter with one-time warning fallback."""

    _lock = threading.Lock()
    _tokenizer: Any = None
    _load_attempted = False
    _fallback_warned = False
    _fallback_reason: str | None = None

    @classmethod
    def _warn_fallback_once(cls, reason: str) -> None:
        if cls._fallback_warned:
            return
        cls._fallback_warned = True
        logger.warning("TokenCounter fallback to rough estimator: {}", reason)

    @classmethod
    def _set_fallback_reason(cls, reason: str) -> None:
        if not cls._fallback_reason:
            cls._fallback_reason = reason

    @classmethod
    def _candidate_model_sources(cls) -> List[str]:
        candidates: List[str] = []
        local_path = global_config.get("ultradomain.tokenizer.local_path")
        model = global_config.get("ultradomain.tokenizer.model")
        fallback_model = global_config.get("ultradomain.tokenizer.fallback_model")
        vllm_model = global_config.get("vllm.model")
        for val in (local_path, model, vllm_model, fallback_model):
            text = str(val or "").strip()
            if text and text not in candidates:
                candidates.append(text)
        return candidates

    @classmethod
    def _load_tokenizer(cls) -> Any:
        if cls._tokenizer is not None:
            return cls._tokenizer
        with cls._lock:
            if cls._tokenizer is not None:
                return cls._tokenizer
            if cls._load_attempted:
                return cls._tokenizer
            try:
                from transformers import AutoTokenizer  # type: ignore
            except Exception as exc:  # noqa: BLE001
                cls._set_fallback_reason(f"transformers import failed: {exc}")
                cls._load_attempted = True
                return None

            local_only = bool(global_config.get("ultradomain.tokenizer.local_only", True))
            trust_remote_code = bool(global_config.get("ultradomain.tokenizer.trust_remote_code", True))
            kwargs = {
                "trust_remote_code": trust_remote_code,
                "local_files_only": local_only,
            }
            candidates = cls._candidate_model_sources()
            if not candidates:
                cls._set_fallback_reason("no tokenizer candidates configured")
                cls._load_attempted = True
                return None
            for source in candidates:
                try:
                    cls._tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
                    logger.info("TokenCounter loaded tokenizer from {}", source)
                    cls._load_attempted = True
                    return cls._tokenizer
                except Exception as exc:  # noqa: BLE001
                    cls._set_fallback_reason(f"load failed from {source}: {exc}")
                    continue
            if cls._fallback_reason is None:
                cls._set_fallback_reason("all tokenizer candidates failed")
            cls._load_attempted = True
            return None

    @classmethod
    def count_text(cls, text: str) -> int:
        tokenizer = cls._load_tokenizer()
        if tokenizer is None:
            cls._warn_fallback_once(cls._fallback_reason or "tokenizer unavailable")
            return max(0, TextUtils.rough_token_len(text or ""))
        try:
            token_ids = tokenizer.encode(text or "", add_special_tokens=False)
            return len(token_ids)
        except Exception as exc:  # noqa: BLE001
            cls._warn_fallback_once(f"encode() failed: {exc}")
            return max(0, TextUtils.rough_token_len(text or ""))

    @classmethod
    def _normalize_message(cls, msg: Dict[str, Any]) -> Dict[str, str]:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if isinstance(content, list):
            # OpenAI-style multimodal content; collapse text-like pieces conservatively.
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            content_text = "\n".join(parts)
        else:
            content_text = str(content or "")
        return {"role": role, "content": content_text}

    @classmethod
    def _fallback_count_messages(cls, messages: Iterable[Dict[str, Any]]) -> int:
        total = FALLBACK_BASE_OVERHEAD
        for msg in messages:
            normalized = cls._normalize_message(msg)
            total += TextUtils.rough_token_len(normalized["role"])
            total += TextUtils.rough_token_len(normalized["content"])
            total += FALLBACK_MESSAGE_OVERHEAD
        # Approximate assistant-generation prefix overhead.
        total += FALLBACK_MESSAGE_OVERHEAD
        return max(1, total)

    @classmethod
    def _len_from_chat_template_tokens(cls, tokenized: Any) -> int:
        if isinstance(tokenized, list):
            if tokenized and isinstance(tokenized[0], list):
                return len(tokenized[0])
            return len(tokenized)
        if hasattr(tokenized, "shape"):
            shape = getattr(tokenized, "shape")
            if isinstance(shape, tuple) and shape:
                return _safe_int(shape[-1], 0)
        if hasattr(tokenized, "numel"):
            return _safe_int(tokenized.numel(), 0)
        return 0

    @classmethod
    def count_messages(cls, messages: List[Dict[str, Any]]) -> int:
        normalized = [cls._normalize_message(msg if isinstance(msg, dict) else {}) for msg in (messages or [])]
        tokenizer = cls._load_tokenizer()
        if tokenizer is None:
            cls._warn_fallback_once(cls._fallback_reason or "tokenizer unavailable")
            return cls._fallback_count_messages(normalized)

        try:
            if hasattr(tokenizer, "apply_chat_template"):
                tokenized = tokenizer.apply_chat_template(
                    normalized,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                token_count = cls._len_from_chat_template_tokens(tokenized)
                if token_count > 0:
                    return token_count
        except Exception as exc:  # noqa: BLE001
            logger.debug("TokenCounter apply_chat_template unavailable, fallback to encode(): {}", exc)

        try:
            total = FALLBACK_BASE_OVERHEAD
            for msg in normalized:
                merged = f"{msg['role']}\n{msg['content']}"
                total += len(tokenizer.encode(merged, add_special_tokens=False))
                total += FALLBACK_MESSAGE_OVERHEAD
            total += FALLBACK_MESSAGE_OVERHEAD
            return max(1, total)
        except Exception as exc:  # noqa: BLE001
            cls._warn_fallback_once(f"message encode() failed: {exc}")
            return cls._fallback_count_messages(normalized)
