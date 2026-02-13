from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

from loguru import logger
import requests

from relrag.config.config_loader import config as global_config
from relrag.utils.text_utils import TextUtils
from relrag.utils.vllm_runtime import detect_vllm_served_model


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
    _tokenizer_source: str | None = None
    _candidate_signature: tuple[str, ...] = ()
    _load_attempted = False
    _fallback_warned = False
    _fallback_reason: str | None = None
    _vllm_tokenize_url: str | None = None
    _vllm_tokenize_checked = False

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

    @staticmethod
    def _is_local_url(url: str) -> bool:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = (parsed.hostname or "").strip().lower()
        return host in {"127.0.0.1", "localhost", "::1"}

    @classmethod
    def _resolve_vllm_tokenize_url(cls) -> str | None:
        if cls._vllm_tokenize_checked:
            return cls._vllm_tokenize_url
        with cls._lock:
            if cls._vllm_tokenize_checked:
                return cls._vllm_tokenize_url
            endpoint = str(
                os.environ.get("VLLM_ENDPOINT")
                or global_config.get("vllm.endpoint")
                or ""
            ).strip().rstrip("/")
            if not endpoint:
                cls._vllm_tokenize_url = None
                cls._vllm_tokenize_checked = True
                return None
            if endpoint.endswith("/v1"):
                cls._vllm_tokenize_url = f"{endpoint[:-3]}/tokenize".rstrip("/")
            else:
                cls._vllm_tokenize_url = f"{endpoint}/tokenize"
            cls._vllm_tokenize_checked = True
            return cls._vllm_tokenize_url

    @classmethod
    def _call_vllm_tokenize(cls, payload: Dict[str, Any]) -> int | None:
        url = cls._resolve_vllm_tokenize_url()
        if not url:
            return None
        try:
            proxies = {"http": None, "https": None} if cls._is_local_url(url) else None
            response = requests.post(
                url,
                json=payload,
                timeout=2.0,
                proxies=proxies,
            )
            if response.status_code >= 400:
                cls._set_fallback_reason(f"vLLM /tokenize HTTP {response.status_code}")
                return None
            data = response.json()
            if isinstance(data, dict):
                count = data.get("count")
                if isinstance(count, int) and count >= 0:
                    return int(count)
                tokens = data.get("tokens")
                if isinstance(tokens, list):
                    return len(tokens)
            cls._set_fallback_reason("vLLM /tokenize invalid response")
            return None
        except Exception as exc:  # noqa: BLE001
            cls._set_fallback_reason(f"vLLM /tokenize failed: {exc}")
            return None

    @staticmethod
    def _expand_model_aliases(model: Any) -> List[str]:
        raw = str(model or "").strip()
        if not raw:
            return []
        aliases: List[str] = []

        def _add(value: str) -> None:
            text = str(value or "").strip()
            if text and text not in aliases:
                aliases.append(text)

        _add(raw)
        lowered = raw.lower()
        if "/" not in raw:
            if lowered.startswith("gpt-oss"):
                _add(f"openai/{raw}")
            elif lowered == "qwen3-30b-a3b":
                _add("Qwen/Qwen3-30B-A3B")
                _add("cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
        else:
            provider, _, name = raw.partition("/")
            if provider.lower() == "openai" and name:
                _add(name)

        return aliases

    @classmethod
    def _resolve_active_llm_model(cls) -> str:
        env_model = str(
            os.environ.get("VLLM_MODEL")
            or os.environ.get("RELRAG_VLLM_MODEL")
            or ""
        ).strip()
        if env_model:
            return env_model

        endpoint = str(
            os.environ.get("VLLM_ENDPOINT")
            or global_config.get("vllm.endpoint")
            or ""
        ).strip()
        runtime_model = detect_vllm_served_model(endpoint) if endpoint else None
        if runtime_model:
            return str(runtime_model).strip()

        return str(global_config.get("vllm.model") or "").strip()

    @classmethod
    def _should_include_local_path(cls, local_path: str, active_model: str) -> bool:
        if not local_path:
            return False
        if not active_model:
            return True
        lowered_path = local_path.lower()
        lowered_model = active_model.lower()
        model_leaf = lowered_model.split("/")[-1]
        return lowered_model in lowered_path or model_leaf in lowered_path

    @staticmethod
    def _same_model_family(model_a: Any, model_b: Any) -> bool:
        a = str(model_a or "").strip().lower()
        b = str(model_b or "").strip().lower()
        if not a or not b:
            return False
        if a == b:
            return True
        return a.split("/")[-1] == b.split("/")[-1]

    @classmethod
    def _cache_roots(cls) -> List[Path]:
        roots: List[Path] = []

        def _add(raw: Any) -> None:
            text = str(raw or "").strip()
            if not text:
                return
            base = Path(text).expanduser()
            candidates = [base]
            if base.name != "hub":
                candidates.append(base / "hub")
            for item in candidates:
                if item.exists() and item not in roots:
                    roots.append(item)

        _add(os.environ.get("HF_HUB_CACHE"))
        _add(os.environ.get("HF_HOME"))
        _add(os.environ.get("TRANSFORMERS_CACHE"))
        _add("/home/wjk/.cache/hf")
        _add(str(Path.home() / ".cache" / "huggingface"))
        return roots

    @classmethod
    def _resolve_local_snapshots(cls, model: Any) -> List[str]:
        raw = str(model or "").strip()
        if not raw:
            return []
        snapshots: List[str] = []

        def _add(path_obj: Path) -> None:
            resolved = str(path_obj.expanduser())
            if resolved and resolved not in snapshots:
                snapshots.append(resolved)

        path = Path(raw).expanduser()
        if path.exists():
            if path.is_dir():
                if (path / "tokenizer.json").exists():
                    _add(path)
                snap_root = path / "snapshots"
                if snap_root.exists():
                    for snap in sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                        if snap.is_dir() and (snap / "tokenizer.json").exists():
                            _add(snap)
            return snapshots

        repo_key = raw.replace("/", "--")
        for root in cls._cache_roots():
            model_dir = root / f"models--{repo_key}"
            snap_root = model_dir / "snapshots"
            if not snap_root.exists():
                continue
            for snap in sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if snap.is_dir() and (snap / "tokenizer.json").exists():
                    _add(snap)
        return snapshots

    @classmethod
    def _candidate_model_sources(cls) -> List[str]:
        candidates: List[str] = []
        active_model = cls._resolve_active_llm_model()
        local_path = str(global_config.get("ultradomain.tokenizer.local_path") or "").strip()
        model = global_config.get("ultradomain.tokenizer.model")
        fallback_model = global_config.get("ultradomain.tokenizer.fallback_model")
        vllm_model = global_config.get("vllm.model")

        def _extend(values: List[str]) -> None:
            for text in values:
                item = str(text or "").strip()
                if item and item not in candidates:
                    candidates.append(item)

        def _extend_model_and_snapshots(raw_model: Any) -> None:
            aliases = cls._expand_model_aliases(raw_model)
            for alias in aliases:
                _extend(cls._resolve_local_snapshots(alias))
            _extend(aliases)

        # Keep tokenizer aligned with active runtime model first.
        _extend_model_and_snapshots(active_model)
        if active_model:
            if cls._same_model_family(active_model, vllm_model):
                _extend_model_and_snapshots(vllm_model)
            if cls._same_model_family(active_model, model):
                _extend_model_and_snapshots(model)
        else:
            _extend_model_and_snapshots(vllm_model)
            _extend_model_and_snapshots(model)
        if cls._should_include_local_path(local_path, active_model):
            _extend([local_path])
        if not active_model:
            _extend_model_and_snapshots(fallback_model)
        return candidates

    @staticmethod
    def _encode_length(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> int:
        try:
            encoded = tokenizer.encode(text or "", add_special_tokens=add_special_tokens)
        except TypeError:
            try:
                encoded = tokenizer.encode(text or "")
            except Exception:
                return 0
        except Exception:
            return 0
        if hasattr(encoded, "ids"):
            ids = getattr(encoded, "ids")
            if isinstance(ids, list):
                return len(ids)
        if isinstance(encoded, list):
            return len(encoded)
        try:
            return len(encoded)
        except Exception:
            return 0

    @staticmethod
    def _load_tokenizers_json(source: str) -> Any:
        path = Path(str(source or "").strip()).expanduser()
        if not path.exists():
            return None
        tok_json = path / "tokenizer.json" if path.is_dir() else path
        if not tok_json.exists() or tok_json.name != "tokenizer.json":
            return None
        try:
            from tokenizers import Tokenizer as RawTokenizer  # type: ignore
            return RawTokenizer.from_file(str(tok_json))
        except Exception:
            return None

    @classmethod
    def _load_tokenizer(cls) -> Any:
        candidates = cls._candidate_model_sources()
        signature = tuple(candidates)
        if cls._tokenizer is not None and cls._candidate_signature == signature:
            return cls._tokenizer
        with cls._lock:
            candidates = cls._candidate_model_sources()
            signature = tuple(candidates)
            if cls._tokenizer is not None and cls._candidate_signature == signature:
                return cls._tokenizer
            if cls._load_attempted and cls._candidate_signature == signature:
                return cls._tokenizer
            if cls._tokenizer is not None and cls._candidate_signature != signature:
                logger.info(
                    "TokenCounter tokenizer target changed ({} -> {}), reloading tokenizer",
                    cls._tokenizer_source,
                    candidates[0] if candidates else "<none>",
                )
                cls._tokenizer = None
                cls._tokenizer_source = None
            try:
                from transformers import AutoTokenizer  # type: ignore
            except Exception as exc:  # noqa: BLE001
                cls._set_fallback_reason(f"transformers import failed: {exc}")
                cls._candidate_signature = signature
                cls._load_attempted = True
                return None

            local_only = bool(global_config.get("ultradomain.tokenizer.local_only", True))
            trust_remote_code = bool(global_config.get("ultradomain.tokenizer.trust_remote_code", True))
            kwargs = {
                    "trust_remote_code": trust_remote_code,
                    "local_files_only": local_only,
                }
            if not candidates:
                cls._set_fallback_reason("no tokenizer candidates configured")
                cls._candidate_signature = signature
                cls._load_attempted = True
                return None
            for source in candidates:
                raw_tokenizer = cls._load_tokenizers_json(source)
                if raw_tokenizer is not None:
                    cls._tokenizer = raw_tokenizer
                    cls._tokenizer_source = source
                    cls._candidate_signature = signature
                    logger.info("TokenCounter loaded tokenizer.json from {}", source)
                    cls._load_attempted = True
                    return cls._tokenizer
                try:
                    cls._tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
                    logger.info("TokenCounter loaded tokenizer from {}", source)
                    cls._tokenizer_source = source
                    cls._candidate_signature = signature
                    cls._load_attempted = True
                    return cls._tokenizer
                except Exception as exc:  # noqa: BLE001
                    cls._set_fallback_reason(f"load failed from {source}: {exc}")
                    continue
            if cls._fallback_reason is None:
                cls._set_fallback_reason("all tokenizer candidates failed")
            cls._candidate_signature = signature
            cls._load_attempted = True
            return None

    @classmethod
    def count_text(cls, text: str) -> int:
        remote_count = cls._call_vllm_tokenize({"prompt": text or ""})
        if remote_count is not None:
            return max(0, int(remote_count))
        tokenizer = cls._load_tokenizer()
        if tokenizer is None:
            cls._warn_fallback_once(cls._fallback_reason or "tokenizer unavailable")
            return max(0, TextUtils.rough_token_len(text or ""))
        try:
            length = cls._encode_length(tokenizer, text or "", add_special_tokens=False)
            if length > 0:
                return length
            raise RuntimeError("encode returned empty")
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
        remote_count = cls._call_vllm_tokenize(
            {
                "messages": normalized,
                "add_generation_prompt": True,
            }
        )
        if remote_count is not None:
            return max(1, int(remote_count))
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
                encoded_len = cls._encode_length(tokenizer, merged, add_special_tokens=False)
                if encoded_len <= 0:
                    raise RuntimeError("message encode returned empty")
                total += encoded_len
                total += FALLBACK_MESSAGE_OVERHEAD
            total += FALLBACK_MESSAGE_OVERHEAD
            return max(1, total)
        except Exception as exc:  # noqa: BLE001
            cls._warn_fallback_once(f"message encode() failed: {exc}")
            return cls._fallback_count_messages(normalized)
