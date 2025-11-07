from __future__ import annotations

import json
import time
from typing import Any, Dict, List
import random

import requests
from requests.adapters import HTTPAdapter
import threading
import itertools
import os
import re
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
        strict_endpoint: bool = False,
    ):
        # Concurrency config
        vllm_cfg = global_config.get("vllm", {}) or {}
        ccfg = vllm_cfg.get("concurrency", {}) or {}

        # Endpoint pool: prefer env-based endpoints in non-strict mode; fall back to config list
        endpoints_cfg = ccfg.get("endpoints") or []
        self._endpoint_lock = threading.Lock()
        self._strict_endpoint = bool(strict_endpoint)
        # Resolve endpoint pool
        if self._strict_endpoint:
            self._endpoints = [endpoint.rstrip("/")]
        else:
            env_eps = self._resolve_env_endpoints()
            if env_eps:
                self._endpoints = env_eps
            elif endpoints_cfg and isinstance(endpoints_cfg, list):
                self._endpoints = [str(ep).rstrip("/") for ep in endpoints_cfg if ep]
            else:
                # Fallback to single endpoint provided
                self._endpoints = [endpoint.rstrip("/")]

        # Round-robin iterator and health map (boolean healthy flags)
        self._endpoint_cycle = itertools.cycle(self._endpoints)
        self._healthy: Dict[str, bool] = {ep: True for ep in self._endpoints}
        # Recovery delay for half-open circuit (default 30s)
        self._recovery_delay_sec = float(ccfg.get("blacklist_duration_sec", 30.0))

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
    # Backend pool helpers
    # -----------------------------
    def _resolve_env_endpoints(self) -> List[str]:
        """Resolve endpoints strictly from environment variables VLLM_ENDPOINT{N}.

        Returns an ordered list if any are set; otherwise returns empty.
        """
        pattern = re.compile(r"^VLLM_ENDPOINT(\d+)$")
        pairs: List[tuple[int, str]] = []
        for key, val in os.environ.items():
            m = pattern.match(key)
            if m and val:
                try:
                    idx = int(m.group(1))
                except Exception:
                    continue
                pairs.append((idx, str(val).rstrip("/")))
        if pairs:
            pairs.sort(key=lambda x: x[0])
            seen = set()
            out: List[str] = []
            for _, ep in pairs:
                if ep and ep not in seen:
                    out.append(ep)
                    seen.add(ep)
            return out
        return []

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
            "Use canonical vocabulary (e.g., map 'comic artist' -> 'cartoonist', 'American' -> 'United States') when obvious; otherwise repeat the raw value.\n\n"
            "STYLE / 写作规范:\n"
            "1) 不得使用代词（如 他/她/它/他们/其/该/this/that/they 等）作为主语。\n"
            "2) 始终使用最具体、可辨识的实体全名或规范简称（如“Tim Berners-Lee”，“万科企业股份有限公司（万科）”）。\n"
            "3) 若上下文能确定实体，统一回填实体全称，不要写‘他/她/其/该公司’。\n"
            "4) Replace ALL pronouns with resolved entity names: he, she, it, they, this, that, these, those, his, her, their（以及中文代词：他、她、它、他们、其、该、这、那、这些、那些）均需替换为明确实体或名词。不得在“subj”“obj”“attribute.values”中保留代词。\n"
            "5) Evidence must be verbatim. 同时尽可能提供 meta.evidence_canonical（在不改变事实的前提下，将句首或指代代词替换为正确实体名）。\n"
            "6) 若无法确定代词指代的实体，请跳过该条笔记。\n\n"
            "Examples / 例子:\n"
            "[Bad] 他在1998年加入公司。\n"
            "[Good] Tim Berners-Lee 在 1998 年加入万维网联盟（W3C）。\n"
            "[Bad] She was born in 1988.\n"
            "[Good] Ada Lovelace was born in 1815.\n\n"
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
            # In strict endpoint (e.g., shard to dedicated port), disable keep-alive
            # to avoid accumulating CLOSE_WAIT sockets on the server side.
            if self._strict_endpoint:
                sess.headers.update({"Connection": "close", "Accept": "application/json"})
            else:
                sess.headers.update({"Connection": "keep-alive", "Accept": "application/json"})
            # Avoid inheriting system proxy settings that may misroute local endpoints
            try:
                sess.trust_env = False
            except Exception:
                pass
            self._local.session = sess
        return sess

    def _choose_endpoint(self) -> str:
        """Round-robin with health preference; if all unhealthy, fall back to first.
        Thread-safe via endpoint lock.
        """
        with self._endpoint_lock:
            for _ in range(len(self._endpoints)):
                ep = next(self._endpoint_cycle)
                if self._healthy.get(ep, True):
                    return ep
            # All marked unhealthy: fall back to first and rely on short retries
            return self._endpoints[0]

    def _mark_endpoint_success(self, endpoint: str) -> None:
        # On success, mark endpoint healthy
        self._healthy[endpoint] = True

    def _reset_session(self) -> None:
        sess = getattr(self._local, "session", None)
        if sess is not None:
            try:
                sess.close()
            except Exception:
                pass
            self._local.session = None

    def _mark_unhealthy(self, endpoint: str) -> None:
        # Circuit-breaker: mark unhealthy and schedule half-open recovery
        self._healthy[endpoint] = False
        try:
            threading.Timer(self._recovery_delay_sec, lambda: self._healthy.update({endpoint: True})).start()
        except Exception:
            # If timer fails, rely on next success to flip healthy
            pass

    def _mark_endpoint_failure(self, endpoint: str, kind: str) -> None:
        # Mark endpoint unhealthy and reset session on transport issues
        self._mark_unhealthy(endpoint)
        if kind in ("timeout", "http"):
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
                endpoint = self._choose_endpoint()
                if self._endpoint_log_every and attempt == 0:
                    try:
                        if (self._recent_calls % self._endpoint_log_every) == 0:
                            logger.info("vLLM call via endpoint {}", endpoint)
                    except Exception:
                        pass
                session = self._get_session()
                # Ensure the response is fully closed even on success to release the socket.
                with session.post(
                    f"{endpoint}/chat/completions",
                    json=payload,
                    timeout=(self._connect_timeout_sec, self._read_timeout_sec),
                ) as response:
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
