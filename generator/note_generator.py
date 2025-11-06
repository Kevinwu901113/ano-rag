from __future__ import annotations

import json
import time
from typing import Any, Dict, List
import random

import requests
from requests.adapters import HTTPAdapter
import threading
import itertools
from loguru import logger

from config.config_loader import config as global_config
from generator.note_parsing import NoteParsingPipeline
from validators.note_validator import validate_and_normalize
from utils.adaptive_concurrency import AdaptiveConcurrencyController, AdaptiveConfig


class NoteGenerator:
    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8000,
        parsing_config: Dict[str, Any] | None = None,
        schema_guard_config: Dict[str, Any] | None = None,
    ):
        # Concurrency config
        vllm_cfg = global_config.get("vllm", {}) or {}
        ccfg = vllm_cfg.get("concurrency", {}) or {}

        # Endpoint pool: use provided list if non-empty; else fallback to single endpoint
        endpoints = ccfg.get("endpoints") or []
        if endpoints and isinstance(endpoints, list):
            self._endpoints = [str(ep).rstrip("/") for ep in endpoints if ep]
        else:
            self._endpoints = [endpoint.rstrip("/")]

        # Round-robin iterator for endpoints (protected by lock for thread safety)
        self._endpoint_cycle = itertools.cycle(self._endpoints)
        self._endpoint_lock = threading.Lock()
        # Basic per-endpoint health with temporary blacklist and half-open recovery
        self._health: Dict[str, Dict[str, float]] = {
            ep: {"score": 1.0, "blacklist_until": 0.0} for ep in self._endpoints
        }

        # Thread-local HTTP session per worker for connection reuse
        self._local = threading.local()

        # Timeouts and backoff
        # Separate connect/read timeouts; defaults to (3.05s, 20s)
        self._connect_timeout_sec = float(ccfg.get("connect_timeout_sec", 3.05))
        self._read_timeout_sec = float(ccfg.get("read_timeout_sec", 20.0))
        # Jittered exponential backoff with overall cap and limited retries
        self._retry_max_attempts = int(ccfg.get("retry_max_attempts", 2))
        self._retry_total_cap_sec = float(ccfg.get("retry_total_cap_sec", 30.0))
        self._retry_backoff_base = float(ccfg.get("retry_backoff_base", 1.0))
        self._retry_backoff_max_sec = float(ccfg.get("retry_backoff_max_sec", 6.0))
        self._retry_jitter_frac = float(ccfg.get("retry_jitter_frac", 0.5))
        # Endpoint health/backoff config
        self._blacklist_duration_sec = float(ccfg.get("blacklist_duration_sec", 15.0))
        self._endpoint_log_every = int(ccfg.get("endpoint_log_every", 0))
        # HTTP connection pool size guided by concurrency
        self._configured_max_workers = int(ccfg.get("max_workers", 8))
        self._pool_maxsize = max(16, self._configured_max_workers * 2)

        self.endpoint = self._endpoints[0]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._stats: Dict[str, int] = {}

        # Adaptive concurrency hooks (latency sampling)
        acfg_dict = (vllm_cfg.get("adaptive", {}) or {})
        acfg = AdaptiveConfig(
            enabled=bool(acfg_dict.get("enabled", False)),
            min_workers=int(acfg_dict.get("min_workers", 2)),
            max_workers=int(acfg_dict.get("max_workers", max(8, self._configured_max_workers))),
            target_p50_ms=int(acfg_dict.get("target_p50_ms", 1200)),
            target_p95_ms=int(acfg_dict.get("target_p95_ms", 3500)),
            step_up=int(acfg_dict.get("step_up", 2)),
            step_down=int(acfg_dict.get("step_down", 2)),
            window_size=int(acfg_dict.get("window_size", 50)),
            cool_down_sec=float(acfg_dict.get("cool_down_sec", 5.0)),
        )
        self._adaptive = AdaptiveConcurrencyController(acfg) if acfg.enabled else None
        self._latency_samples = 0

        # Rolling windows for error/timeout rates and stage timings
        self._error_window_size = max(30, int(acfg.window_size if acfg.enabled else 50))
        self._recent_errors: List[str] = []  # values in {"timeout","http"}
        self._recent_calls: int = 0
        self._recent_timeouts: int = 0
        self._stage_times = {"call_ms": [], "parse_ms": [], "validate_ms": []}
        self._stage_window_size = max(50, int(acfg.window_size if acfg.enabled else 100))
        self._processed_chunks = 0

        if parsing_config is None:
            parsing_config = global_config.get("parsing", {}) or {}
        if schema_guard_config is None:
            schema_guard_config = global_config.get("schema_guard", {}) or {}

        self.parser = NoteParsingPipeline(parsing_config, schema_guard_config)
        self._stop_sequences = parsing_config.get("stop") or ['"]\n', "\n]", "\n\nEND", "END_JSON"]
        self._parsing_max_tokens = parsing_config.get("max_tokens")

    # -----------------------------
    # Prompt：严格 JSON 输出
    # -----------------------------
    @staticmethod
    def build_prompt(doc_text: str, doc_id: str) -> str:
        source_text = doc_text or ""
        return (
            "You are an ontology-aligned information extraction system. From the following text, extract factual notes.\n"
            "Return ONLY valid JSON (RFC 8259). Prefer a JSON array; JSONL is allowed if necessary (one object per line, no surrounding brackets).\n"
            "Each note object MUST contain the keys:\n"
            '  "subj", "pred", "obj", "subj_type", "obj_type", "evidence", "meta"\n'
            "Populate them as follows:\n"
            '  - "subj","pred","obj","evidence" are non-empty strings; evidence is a verbatim snippet (>=4 chars).\n'
            '  - "subj_type","obj_type" must be one of ["PERSON","WORK","ORG","PLACE","EVENT","CONCEPT","TIME"].\n'
            '  - "pred" should use canonical attributes like ["occupation","title","category","nationality","born_on","died_on","spouse","parent","authored_by","performed_by","member_of","located_in","headquartered_in","label","same_as","alias_of","type"].\n'
            '  - For occupations, acceptable surface forms include ["occupation","profession","job","works as","career","title (when occupational)"]; ALWAYS output meta.attribute.name="occupation".\n'
            f'  - "meta" MUST include: {{"source": "{doc_id}", "confidence": float 0-1, "subject_profile": {{}}, "attribute": {{...}}}}\n'
            '       * "subject_profile" = {"type": <subj_type>, "aliases": [], "nationality": [], "birth": null, "death": null, "occupations": [], "titles": [], "categories": [], "same_as": []}. Fill lists when evidence gives the data; use [] when unknown.\n'
            f'       * "attribute" = {{"name": <same as pred>, "values": [{{"value": <raw>, "normalized": <canonical or same>, "confidence": 0-1, "source": "{doc_id}", "evidence": <snippet>}}]}}\n'
            '       * Set "object_profile" when the object is an entity (type + aliases). Otherwise omit or use null.\n'
            "Use canonical vocabulary (e.g., map 'comic artist' -> 'cartoonist', 'American' -> 'United States') when obvious; otherwise repeat the raw value.\n"
            "Text:\n"
            f'"""{source_text}"""\n'
            "Output only the JSON."
        )

    def _get_session(self) -> requests.Session:
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=self._pool_maxsize,
                pool_maxsize=self._pool_maxsize,
                max_retries=0,
                pool_block=False,
            )
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            sess.headers.update({"Connection": "keep-alive", "Accept": "application/json"})
            self._local.session = sess
        return sess

    def _next_endpoint(self) -> str:
        try:
            with self._endpoint_lock:
                now = time.time()
                # Try cycle until a non-blacklisted endpoint is found
                for _ in range(len(self._endpoints)):
                    candidate = next(self._endpoint_cycle)
                    bl_until = (self._health.get(candidate) or {}).get("blacklist_until", 0.0)
                    if bl_until <= now:
                        return candidate
                # If all are blacklisted, pick the one with nearest expiry
                least = sorted(
                    self._endpoints,
                    key=lambda ep: (self._health.get(ep) or {}).get("blacklist_until", 0.0),
                )[0]
                return least
        except Exception:
            return self.endpoint

    def _mark_endpoint_success(self, endpoint: str) -> None:
        h = self._health.setdefault(endpoint, {"score": 1.0, "blacklist_until": 0.0})
        h["blacklist_until"] = 0.0

    def _reset_session(self) -> None:
        sess = getattr(self._local, "session", None)
        if sess is not None:
            try:
                sess.close()
            except Exception:
                pass
            self._local.session = None

    def _mark_endpoint_failure(self, endpoint: str, kind: str) -> None:
        h = self._health.setdefault(endpoint, {"score": 1.0, "blacklist_until": 0.0})
        h["blacklist_until"] = time.time() + max(0.0, self._blacklist_duration_sec)
        # On timeouts, reset the thread-local session to avoid stale keep-alive sockets
        if kind == "timeout":
            self._reset_session()

    def _call(self, prompt: str, *, stop: List[str] | None = None, max_tokens: int | None = None) -> str:
        attempts = max(1, self._retry_max_attempts)
        last_exc: Exception | None = None
        total_wait: float = 0.0
        for attempt in range(attempts):
            try:
                t0 = time.time()
                payload: Dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens or min(self.max_tokens, 2000),
                    "messages": [{"role": "user", "content": prompt}],
                }
                if stop:
                    payload["stop"] = stop
                endpoint = self._next_endpoint()
                if self._endpoint_log_every and attempt == 0:
                    try:
                        if (self._recent_calls % self._endpoint_log_every) == 0:
                            logger.info("vLLM call via endpoint {}", endpoint)
                    except Exception:
                        pass
                session = self._get_session()
                response = session.post(
                    f"{endpoint}/chat/completions",
                    json=payload,
                    timeout=(self._connect_timeout_sec, self._read_timeout_sec),
                )
                response.raise_for_status()
                data = response.json()
                # record latency
                if self._adaptive:
                    self._adaptive.record_latency_ms(int((time.time() - t0) * 1000))
                    self._latency_samples += 1
                # success accounting
                self._mark_endpoint_success(endpoint)
                self._recent_calls += 1
                return data["choices"][0]["message"]["content"]
            except requests.Timeout as exc:  # noqa: PERF203
                last_exc = exc
                self._recent_errors.append("timeout")
                self._recent_timeouts += 1
                self._recent_errors = self._recent_errors[-self._error_window_size :]
                try:
                    self._mark_endpoint_failure(endpoint, "timeout")
                except Exception:
                    pass
                if attempt == attempts - 1:
                    logger.error("Note generator timeout after {} attempts: {}", attempts, exc)
                    raise
                # jittered exponential backoff with cap and total time limit
                nominal = self._retry_backoff_base * (2 ** attempt)
                jitter = nominal * random.uniform(0.0, self._retry_jitter_frac)
                wait = min(self._retry_backoff_max_sec, nominal + jitter)
                remain = max(0.0, self._retry_total_cap_sec - total_wait)
                wait = min(wait, remain)
                logger.warning(
                    "Timeout (attempt={}/{}); switching endpoint and backing off {:.2f}s (cap left {:.2f}s): {}",
                    attempt + 1,
                    attempts,
                    wait,
                    remain,
                    exc,
                )
                if wait > 0:
                    time.sleep(wait)
                    total_wait += wait
            except requests.RequestException as exc:  # noqa: PERF203
                last_exc = exc
                self._recent_errors.append("http")
                self._recent_errors = self._recent_errors[-self._error_window_size :]
                try:
                    self._mark_endpoint_failure(endpoint, "http")
                except Exception:
                    pass
                if attempt == attempts - 1:
                    logger.error("Note generator call failed after {} attempts: {}", attempts, exc)
                    raise
                nominal = self._retry_backoff_base * (2 ** attempt)
                jitter = nominal * random.uniform(0.0, self._retry_jitter_frac)
                wait = min(self._retry_backoff_max_sec, nominal + jitter)
                remain = max(0.0, self._retry_total_cap_sec - total_wait)
                wait = min(wait, remain)
                logger.warning(
                    "HTTP error (attempt={}/{}); switching endpoint and backing off {:.2f}s (cap left {:.2f}s): {}",
                    attempt + 1,
                    attempts,
                    wait,
                    remain,
                    exc,
                )
                if wait > 0:
                    time.sleep(wait)
                    total_wait += wait

    def generate_for_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id, chunk_id = chunk["doc_id"], chunk["chunk_id"]
        chunk_text = chunk["text"]
        prompt = self.build_prompt(chunk_text, doc_id)
        stop_sequences = self._stop_sequences or None
        call_max_tokens = self._parsing_max_tokens or self.max_tokens
        t0 = time.time()
        raw = self._call(prompt, stop=stop_sequences, max_tokens=call_max_tokens)
        t1 = time.time()

        parsed_notes = self.parser.parse(raw, doc_id)
        t2 = time.time()
        parser_run_stats = self.parser.get_stats(cumulative=False)
        if parser_run_stats.get("json_parse_failures"):
            logger.warning(
                "Parsing failed doc={} chunk={} stats={}",
                doc_id,
                chunk_id,
                parser_run_stats,
            )
        if not parsed_notes:
            # Do NOT heavy-retry on parsing quality; at most one light retry could be added here
            # but we prefer to return empty to avoid storming upstream.
            self._record_stage_times(int((t1 - t0) * 1000), int((t2 - t1) * 1000), 0)
            self._processed_chunks += 1
            self._maybe_log_stage_p95()
            return []

        serialized = json.dumps(parsed_notes, ensure_ascii=False)
        t3 = time.time()
        ok, notes_out, metrics = validate_and_normalize(serialized, doc_id, chunk_id)
        t4 = time.time()
        if not ok:
            logger.warning("Validation failed doc={} chunk={} details={}", doc_id, chunk_id, metrics)
            self._stats["validation_failures"] = self._stats.get("validation_failures", 0) + 1
            self._record_stage_times(int((t1 - t0) * 1000), int((t2 - t1) * 1000), int((t4 - t3) * 1000))
            self._processed_chunks += 1
            self._maybe_log_stage_p95()
            return []
        self._record_stage_times(int((t1 - t0) * 1000), int((t2 - t1) * 1000), int((t4 - t3) * 1000))
        self._processed_chunks += 1
        self._maybe_log_stage_p95()
        return notes_out

    def export_stats(self) -> Dict[str, int]:
        combined = self.parser.get_stats()
        for key, value in self._stats.items():
            combined[key] = combined.get(key, 0) + value
        return combined

    # Concurrency suggestion for builders
    def suggest_concurrency(self) -> int:
        """Return suggested worker count based on adaptive controller.

        Builders can call this periodically to tune inflight size.
        """
        if self._adaptive is None:
            # fall back to configured max_workers
            # apply simple error-rate based downshift even without adaptive controller
            err_rate = self.recent_error_rate()
            timeout_rate = self.recent_timeout_rate()
            if err_rate > 0.25 or timeout_rate > 0.15:
                return max(1, self._configured_max_workers // 2)
            return max(1, self._configured_max_workers)
        # Warm-up: use configured max until we have enough samples
        warmup_needed = max(10, self._adaptive.cfg.window_size // 2)
        if self._latency_samples < warmup_needed:
            return max(1, self._configured_max_workers)
        # After warm-up, try adjust and return controller target
        try:
            _ = self._adaptive.try_adjust()
        except Exception:
            pass
        # Combine latency-based suggestion with error-rate guardrail
        suggested = max(1, int(self._adaptive.current_workers()))
        err_rate = self.recent_error_rate()
        timeout_rate = self.recent_timeout_rate()
        if err_rate > 0.25 or timeout_rate > 0.15:
            suggested = max(1, min(suggested, self._adaptive.cfg.max_workers) // 2)
        return suggested

    # -----------------------------
    # Observability helpers
    # -----------------------------
    def recent_error_rate(self) -> float:
        if not self._recent_errors:
            return 0.0
        return sum(1 for e in self._recent_errors if e in ("http", "timeout")) / float(len(self._recent_errors))

    def recent_timeout_rate(self) -> float:
        if not self._recent_errors:
            return 0.0
        return sum(1 for e in self._recent_errors if e == "timeout") / float(len(self._recent_errors))

    def _record_stage_times(self, call_ms: int, parse_ms: int, validate_ms: int) -> None:
        self._stage_times["call_ms"].append(call_ms)
        self._stage_times["parse_ms"].append(parse_ms)
        self._stage_times["validate_ms"].append(validate_ms)
        for k in self._stage_times:
            if len(self._stage_times[k]) > self._stage_window_size:
                self._stage_times[k] = self._stage_times[k][-self._stage_window_size :]

    def _maybe_log_stage_p95(self) -> None:
        # Every 100 chunks, log p95 stage timings and recent error/timeout rates
        if self._processed_chunks % 100 != 0:
            return
        def p95(vals: List[int]) -> int:
            if not vals:
                return 0
            s = sorted(vals)
            idx = max(0, int(len(s) * 0.95) - 1)
            return int(s[idx])
        call_p95 = p95(self._stage_times["call_ms"]) 
        parse_p95 = p95(self._stage_times["parse_ms"]) 
        validate_p95 = p95(self._stage_times["validate_ms"]) 
        logger.info(
            "Stage p95 (last {}): call={}ms parse={}ms validate={}ms | err_rate={:.1%} timeout_rate={:.1%}",
            len(self._stage_times["call_ms"]),
            call_p95,
            parse_p95,
            validate_p95,
            self.recent_error_rate(),
            self.recent_timeout_rate(),
        )
