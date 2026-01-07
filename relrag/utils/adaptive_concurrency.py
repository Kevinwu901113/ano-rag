import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class AdaptiveConfig:
    enabled: bool = True
    min_workers: int = 2
    max_workers: int = 64
    target_p50_ms: int = 1200
    target_p95_ms: int = 3500
    step_up: int = 2
    step_down: int = 2
    window_size: int = 50  # number of samples for window
    cool_down_sec: float = 5.0
    jitter: int = 0  # optional random jitter to avoid sync oscillation


class AdaptiveConcurrencyController:
    """Thread-safe adaptive controller for adjusting worker count.

    - Feeds latency samples via `record_latency_ms()`.
    - Periodically re-evaluates and suggests `current_workers`.
    - Simple heuristic: if p50<=target_p50 and p95<=target_p95 -> scale up;
      if p95>target_p95 -> scale down. Respects min/max bounds.
    """

    def __init__(self, cfg: AdaptiveConfig):
        self.cfg = cfg
        self._latencies: list[int] = []
        self._lock = threading.Lock()
        self._last_adjust_ts = 0.0
        self._workers = max(cfg.min_workers, 1)

    def current_workers(self) -> int:
        with self._lock:
            return self._workers

    def record_latency_ms(self, ms: int) -> None:
        if ms <= 0:
            return
        with self._lock:
            self._latencies.append(ms)
            if len(self._latencies) > self.cfg.window_size:
                # keep most recent window
                self._latencies = self._latencies[-self.cfg.window_size :]

    def try_adjust(self) -> Optional[int]:
        """Return new worker count if changed, else None."""
        now = time.time()
        with self._lock:
            if (now - self._last_adjust_ts) < self.cfg.cool_down_sec:
                return None
            if not self._latencies:
                return None

            lat_sorted = sorted(self._latencies)
            p50 = lat_sorted[max(0, (len(lat_sorted) * 50) // 100 - 1)]
            p95 = lat_sorted[max(0, (len(lat_sorted) * 95) // 100 - 1)]

            new_workers = self._workers
            if p95 > self.cfg.target_p95_ms:
                new_workers = max(self.cfg.min_workers, self._workers - self.cfg.step_down)
            elif p50 <= self.cfg.target_p50_ms and p95 <= self.cfg.target_p95_ms:
                new_workers = min(self.cfg.max_workers, self._workers + self.cfg.step_up)

            if new_workers != self._workers:
                self._workers = new_workers
                self._last_adjust_ts = now
                # reset window to avoid reacting twice to the same burst
                self._latencies.clear()
                return new_workers
            return None