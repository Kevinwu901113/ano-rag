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


def export_metrics() -> Dict[str, int]:
    return _REGISTRY.snapshot()
