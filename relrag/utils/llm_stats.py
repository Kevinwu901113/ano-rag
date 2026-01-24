from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMCallStats:
    llm_calls: int = 0
    llm_retries: int = 0
    t_llm_ms: float = 0.0
    t_build_prompt_ms: float = 0.0
    prompt_tokens_max: int = 0
    completion_tokens_max: int = 0
    prompt_chars_max: int = 0
    completion_chars_max: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    prompt_chars_total: int = 0
    completion_chars_total: int = 0
    context_chars_max: int = 0
    context_chars_total: int = 0
    finish_reason: Optional[str] = None
    error_type: Optional[str] = None

    def record_prompt(
        self,
        *,
        prompt_tokens: Optional[int],
        prompt_chars: Optional[int],
        context_chars: Optional[int],
        build_ms: Optional[float],
    ) -> None:
        if build_ms is not None:
            self.t_build_prompt_ms += float(build_ms)
        if prompt_tokens is not None:
            value = max(0, int(prompt_tokens))
            self.prompt_tokens_total += value
            if value > self.prompt_tokens_max:
                self.prompt_tokens_max = value
        if prompt_chars is not None:
            value = max(0, int(prompt_chars))
            self.prompt_chars_total += value
            if value > self.prompt_chars_max:
                self.prompt_chars_max = value
        if context_chars is not None:
            value = max(0, int(context_chars))
            self.context_chars_total += value
            if value > self.context_chars_max:
                self.context_chars_max = value

    def record_llm_attempt(
        self,
        *,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        prompt_chars: Optional[int],
        completion_chars: Optional[int],
        duration_ms: Optional[float],
        finish_reason: Optional[str] = None,
        error_type: Optional[str] = None,
        retry: bool = False,
    ) -> None:
        self.llm_calls += 1
        if retry:
            self.llm_retries += 1
        if duration_ms is not None:
            self.t_llm_ms += float(duration_ms)
        if prompt_tokens is not None:
            value = max(0, int(prompt_tokens))
            self.prompt_tokens_total += value
            if value > self.prompt_tokens_max:
                self.prompt_tokens_max = value
        if completion_tokens is not None:
            value = max(0, int(completion_tokens))
            self.completion_tokens_total += value
            if value > self.completion_tokens_max:
                self.completion_tokens_max = value
        if prompt_chars is not None:
            value = max(0, int(prompt_chars))
            self.prompt_chars_total += value
            if value > self.prompt_chars_max:
                self.prompt_chars_max = value
        if completion_chars is not None:
            value = max(0, int(completion_chars))
            self.completion_chars_total += value
            if value > self.completion_chars_max:
                self.completion_chars_max = value
        if finish_reason:
            self.finish_reason = str(finish_reason)
        if error_type:
            self.error_type = str(error_type)

    def mark_error(self, error_type: str) -> None:
        if error_type:
            self.error_type = str(error_type)


_LLM_STATS_CTX: contextvars.ContextVar[Optional[LLMCallStats]] = contextvars.ContextVar(
    "llm_stats",
    default=None,
)


def get_active_llm_stats() -> Optional[LLMCallStats]:
    return _LLM_STATS_CTX.get()


@contextlib.contextmanager
def llm_stats_scope(stats: LLMCallStats):
    token = _LLM_STATS_CTX.set(stats)
    try:
        yield stats
    finally:
        _LLM_STATS_CTX.reset(token)
