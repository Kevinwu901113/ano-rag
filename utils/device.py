from __future__ import annotations

import gc
from typing import Any, Callable, Optional, Tuple

from loguru import logger


def pick_embed_device(prefer: str) -> str:
    prefer_norm = str(prefer or "").strip().lower()
    if prefer_norm not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"prefer must be one of auto|cuda|cpu, got: {prefer!r}")

    if prefer_norm == "cpu":
        return "cpu"

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        if prefer_norm == "cuda":
            raise RuntimeError("prefer=cuda requires torch with CUDA support") from exc
        return "cpu"

    cuda_ok = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
    if prefer_norm == "cuda":
        if not cuda_ok:
            raise RuntimeError("CUDA is not available (torch.cuda.is_available() is False)")
        return "cuda"

    # auto
    return "cuda" if cuda_ok else "cpu"


def _should_fallback_to_cpu(exc: BaseException) -> bool:
    try:
        import torch  # type: ignore

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass

    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "cuda out of memory" in msg:
            return True
        if "cublas" in msg:
            return True
        if "cuda error" in msg:
            return True

    # Built-in ConnectionError
    if isinstance(exc, ConnectionError):
        return True

    # requests.exceptions.ConnectionError (does not subclass built-in ConnectionError)
    try:
        import requests  # type: ignore

        if isinstance(exc, requests.exceptions.ConnectionError):
            return True
    except Exception:
        pass

    return False


def _maybe_empty_cuda_cache() -> None:
    try:
        import torch  # type: ignore

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _call_on_fallback(on_fallback: Callable[..., Any], used_device: str, reason: str) -> None:
    try:
        on_fallback(used_device, reason)
        return
    except TypeError:
        pass
    try:
        on_fallback(reason)
        return
    except TypeError:
        pass
    try:
        on_fallback()
    except Exception:
        return


def run_with_fallback(
    fn: Callable[[str], Any],
    prefer: str,
    on_fallback: Optional[Callable[..., Any]] = None,
) -> Tuple[Any, str, Optional[str]]:
    device = pick_embed_device(prefer)
    try:
        result = fn(device)
        return result, device, None
    except BaseException as exc:
        if device == "cpu" or not _should_fallback_to_cpu(exc):
            raise

        reason = f"{exc.__class__.__name__}: {exc}"
        logger.warning("fallback to cpu because {}", reason)
        _maybe_empty_cuda_cache()
        gc.collect()
        if on_fallback is not None:
            _call_on_fallback(on_fallback, "cpu", reason)

        result = fn("cpu")
        return result, "cpu", reason

