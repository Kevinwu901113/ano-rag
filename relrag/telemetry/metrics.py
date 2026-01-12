from __future__ import annotations

from collections import defaultdict
from typing import Dict


class _MetricRegistry:
    def __init__(self) -> None:
        self.counters: Dict[str, int] = defaultdict(int)

    def incr(self, key: str, value: int = 1) -> None:
        self.counters[key] += value

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counters)


_REGISTRY = _MetricRegistry()


def record_attribute_guard(attribute: str, reason: str) -> None:
    key = f"attr_guard.{attribute or 'unknown'}.{reason}"
    _REGISTRY.incr(key)


def record_answer_outcome(attribute: str | None, outcome: str) -> None:
    key = f"answer.{attribute or 'none'}.{outcome}"
    _REGISTRY.incr(key)


def record_binding_strength(strength: str) -> None:
    key = f"binding.{strength}"
    _REGISTRY.incr(key)


def record_anchor_usage(anchor: bool) -> None:
    key = "evidence.anchor" if anchor else "evidence.non_anchor"
    _REGISTRY.incr(key)


def record_pronoun_stat(stat: str, value: int = 1) -> None:
    key = f"coref.pronoun.{stat}"
    _REGISTRY.incr(key, value)


def record_weak_ratio(ratio: float) -> None:
    clamped = max(0.0, min(1.0, ratio))
    scaled = int(round(clamped * 1000))
    _REGISTRY.incr("weak_ratio.sum", scaled)
    _REGISTRY.incr("weak_ratio.count", 1)


def record_retrieval_total() -> None:
    _REGISTRY.incr("retrieval.total")


def record_retrieval_no_path() -> None:
    _REGISTRY.incr("retrieval.no_path")


def record_retrieval_empty_context() -> None:
    _REGISTRY.incr("retrieval.empty_context")


def export_metrics() -> Dict[str, int]:
    return _REGISTRY.snapshot()
