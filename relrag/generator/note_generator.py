from __future__ import annotations

import json
import time
from typing import Any, Dict, List
import random

import requests
from requests.adapters import HTTPAdapter
import threading
import os
import re
from loguru import logger

from relrag.config.config_loader import config as global_config
from relrag.config.attributes_loader import load_attributes_config
from relrag.generator.note_parsing import NoteParsingPipeline
from relrag.schema.note_schema_v1 import NOTE_GEN_JSON_SCHEMA, NOTE_JSON_SCHEMA
from relrag.validators.note_validator import validate_and_normalize
from relrag.utils import TextUtils
from relrag.doc import split_into_entity_aware_spans
from relrag.utils.adaptive_concurrency import AdaptiveConcurrencyController, AdaptiveConfig
from relrag.utils.llm_client import LLMChatClient
from relrag.prompt import load_prompt, render_prompt


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
        routing_cfg = global_config.get("routing", {}) or {}
        json_cfg = vllm_cfg.get("json_mode", {}) or {}
        self._use_guided_json = bool(json_cfg.get("use_guided_json", False))
        self._use_response_format = bool(json_cfg.get("use_response_format", False))
        self._schema_name = str(json_cfg.get("schema_name", "ano-note") or "ano-note")
        self._note_schema = NOTE_GEN_JSON_SCHEMA or NOTE_JSON_SCHEMA

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

        # Token-budget-aware load tracking
        raw_budget = routing_cfg.get("token_budget_hint")
        if raw_budget is None:
            raw_budget = ccfg.get("token_budget_hint", 320000)
        try:
            self._token_budget_hint = max(0, int(raw_budget))
        except (TypeError, ValueError):
            self._token_budget_hint = 0
        self._inflight_tokens: Dict[str, int] = {ep: 0 for ep in self._endpoints}
        self._endpoint_rt_ms: Dict[str, List[int]] = {ep: [] for ep in self._endpoints}

        # Health map (boolean healthy flags)
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
        self._llm = LLMChatClient(
            endpoint=endpoint,
            model=model,
            llm_profile="extract",
            timeout=(self._connect_timeout_sec, self._read_timeout_sec),
            retries=0,
        )

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
        parsing_config = dict(parsing_config)
        if self._use_guided_json or self._use_response_format:
            parsing_config["assume_valid_json"] = True
        if schema_guard_config is None:
            schema_guard_config = global_config.get("schema_guard", {}) or {}
        else:
            schema_guard_config = dict(schema_guard_config)

        self.parser = NoteParsingPipeline(parsing_config, schema_guard_config)
        self._stop_sequences = parsing_config.get("stop") or ['"]\n', "\n]", "\n\nEND", "END_JSON"]
        self._parsing_max_tokens = parsing_config.get("max_tokens")
        self._retry_reminder_enabled = bool(parsing_config.get("enable_retry_reminder", True))
        try:
            self._parse_retry = max(0, int(parsing_config.get("parse_retry", 0)))
        except Exception:
            self._parse_retry = 0
        try:
            self._validation_retry = max(0, int(parsing_config.get("validation_retry", 1)))
        except Exception:
            self._validation_retry = 1

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
    def build_prompt(doc_text: str, doc_id: str, doc_title: str | None = None) -> str:
        source_text = doc_text or ""
        attr_rules = NoteGenerator._attribute_instruction_block()
        main_entity = (doc_title or doc_id or "").strip() or doc_id
        return render_prompt(
            "note_extract.txt",
            doc_id=doc_id,
            main_entity=main_entity,
            attr_rules=attr_rules,
            source_text=source_text,
        )

    @staticmethod
    def _attach_format_reminder(prompt: str) -> str:
        return prompt + load_prompt("note_format_reminder.txt")

    @staticmethod
    def _summarize_validation_errors(
        errors: List[Dict[str, Any]],
        *,
        limit: int = 3,
        max_chars: int = 1500,
    ) -> str:
        if not errors:
            return ""
        parts: List[str] = []
        for err in errors[:limit]:
            msg = err.get("message") if isinstance(err, dict) else str(err)
            if msg:
                parts.append(str(msg).strip())
        summary = " || ".join(parts)
        if len(summary) > max_chars:
            summary = summary[:max_chars].rstrip() + "..."
        return summary

    @staticmethod
    def _attribute_instruction_block() -> str:
        cache = getattr(NoteGenerator, "_cached_attr_rules", None)
        if cache is not None:
            return cache
        cfg = load_attributes_config()
        if not cfg:
            NoteGenerator._cached_attr_rules = ""
            return ""
        lines = ["Attribute-specific rules (配置驱动):"]
        for name, payload in cfg.items():
            title = name.upper()
            defs = payload.get("definition_patterns") or []
            appos = payload.get("appositive_patterns") or []
            negatives = payload.get("negative_verbs") or []
            examples = payload.get("examples") or {}
            allowed = payload.get("value_lexicon") or []
            lines.append(f"- {title}:")
            if defs:
                lines.append(f"  • Accept only definition/appositive sentences such as: {', '.join(defs[:2])}.")
            if appos:
                lines.append(f"  • Appositive cues: {', '.join(appos[:2])}.")
            if negatives:
                lines.append(f"  • Reject sentences containing narrative verbs: {', '.join(negatives[:6])}.")
            if allowed:
                lines.append(f"  • Canonical values subset: {', '.join(str(v) for v in allowed[:10])} (normalize synonyms).")
            good_examples = (examples.get("good") if isinstance(examples, dict) else []) or []
            bad_examples = (examples.get("bad") if isinstance(examples, dict) else []) or []
            if good_examples:
                lines.append(f"  • Good evidence: {good_examples[0]}")
            if bad_examples:
                lines.append(f"  • Reject evidence like: {bad_examples[0]}")
        block = "\n".join(lines) + "\n"
        NoteGenerator._cached_attr_rules = block
        return block

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

    def _choose_endpoint(self, hint_tokens: int = 0) -> str:
        """Pick the least congested healthy endpoint (tokens + recent latency)."""

        def _p50_latency(ep: str) -> int:
            samples = self._endpoint_rt_ms.get(ep) or []
            if not samples:
                return 0
            ordered = sorted(samples)
            idx = min(len(ordered) - 1, len(ordered) // 2)
            return ordered[idx]

        with self._endpoint_lock:
            candidates = [ep for ep in self._endpoints if self._healthy.get(ep, True)]
            if not candidates:
                return self._endpoints[0]
            candidates.sort(
                key=lambda ep: (self._inflight_tokens.get(ep, 0), _p50_latency(ep))
            )
            if hint_tokens and self._token_budget_hint and len(candidates) > 1:
                best = candidates[0]
                projected = self._inflight_tokens.get(best, 0) + hint_tokens
                if projected > self._token_budget_hint:
                    return candidates[1]
            return candidates[0]

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

    @staticmethod
    def _error_snippet(response: requests.Response | None) -> str:
        if response is None:
            return ""
        try:
            text = response.text or ""
        except Exception:
            return ""
        text = text.strip().replace("\n", " ")
        return text[:200]

    def _maybe_disable_json_mode(self, response: requests.Response | None, body_snippet: str | None = None) -> bool:
        if not (self._use_guided_json or self._use_response_format):
            return False
        status = getattr(response, "status_code", None)
        if status is None or status >= 500:
            return False
        snippet = (body_snippet or "").lower()
        if not snippet and response is not None:
            try:
                snippet = (response.text or "").lower()
            except Exception:
                snippet = ""
        keywords = ("guided_json", "json_schema", "response_format", "schema")
        if any(token in snippet for token in keywords):
            if self._use_guided_json:
                self._use_guided_json = False
            elif self._use_response_format:
                self._use_response_format = False
            return True
        return False

    @staticmethod
    def _parse_context_limit(snippet: str) -> Optional[tuple[int, int]]:
        if not snippet:
            return None
        match_ctx = re.search(r"maximum context length is\s+(\d+)", snippet, re.I)
        match_in = re.search(r"request has\s+(\d+)\s+input tokens", snippet, re.I)
        if not match_ctx or not match_in:
            return None
        try:
            return int(match_ctx.group(1)), int(match_in.group(1))
        except (TypeError, ValueError):
            return None

    def _call(self, prompt: str, *, stop: List[str] | None = None, max_tokens: int | None = None) -> str:
        attempts = max(1, self._retry_max_attempts)
        last_exc: Exception | None = None
        total_wait: float = 0.0
        call_max_tokens = int(max_tokens or self.max_tokens)
        call_max_tokens = max(1, call_max_tokens)
        prompt_tokens = max(1, TextUtils.rough_token_len(prompt))
        vllm_cfg = global_config.get("vllm", {}) or {}
        max_context = (
            vllm_cfg.get("max_context_tokens")
            or vllm_cfg.get("max_model_len")
            or vllm_cfg.get("max_seq_len")
        )
        try:
            max_context_val = int(max_context) if max_context is not None else None
        except (TypeError, ValueError):
            max_context_val = None
        safety_margin = int(vllm_cfg.get("context_safety_margin", 256) or 256)
        if max_context_val:
            available = max_context_val - prompt_tokens - safety_margin
            if available < 1:
                available = 1
            if call_max_tokens > available:
                call_max_tokens = available
        for attempt in range(attempts):
            req_tokens = max(1, prompt_tokens + call_max_tokens)
            endpoint = self._choose_endpoint(req_tokens)
            self._inflight_tokens[endpoint] = self._inflight_tokens.get(endpoint, 0) + req_tokens
            t0 = time.time()
            try:
                extra_body = None
                response_format = None
                if self._use_guided_json:
                    extra_body = {"guided_json": self._note_schema}
                elif self._use_response_format:
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {"name": self._schema_name, "schema": self._note_schema},
                    }
                if self._endpoint_log_every and attempt == 0:
                    try:
                        if (self._recent_calls % self._endpoint_log_every) == 0:
                            logger.info("vLLM call via endpoint {}", endpoint)
                    except Exception:
                        pass
                session = self._get_session()
                # Ensure the response is fully closed even on success to release the socket.
                with self._llm.post_chat(
                    [{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=call_max_tokens,
                    stop=stop,
                    llm_profile="extract",
                    endpoint_override=endpoint,
                    response_format=response_format,
                    extra_body=extra_body,
                    timeout=(self._connect_timeout_sec, self._read_timeout_sec),
                    session=session,
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
                rt_ms = int((time.time() - t0) * 1000)
                samples = self._endpoint_rt_ms.setdefault(endpoint, [])
                samples.append(rt_ms)
                if len(samples) > 60:
                    del samples[: len(samples) - 60]
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
            except requests.HTTPError as exc:  # noqa: PERF203
                last_exc = exc
                resp = exc.response if hasattr(exc, "response") else None
                status = getattr(resp, "status_code", None)
                snippet = self._error_snippet(resp)
                if status == 400 and snippet and ("max_tokens" in snippet or "max_completion_tokens" in snippet):
                    parsed = self._parse_context_limit(snippet)
                    if parsed:
                        max_ctx, input_tokens = parsed
                        allowed = max(1, max_ctx - input_tokens - 8)
                        if allowed < call_max_tokens:
                            logger.warning(
                                "Reducing max_tokens from {} to {} due to context limit (max_ctx={} input={})",
                                call_max_tokens,
                                allowed,
                                max_ctx,
                                input_tokens,
                            )
                            call_max_tokens = allowed
                            continue
                if self._maybe_disable_json_mode(resp, snippet):
                    logger.warning(
                        "Disabling JSON mode after HTTP error status={} body_snippet={}",
                        status,
                        snippet,
                    )
                    continue
                self._recent_errors.append("http")
                self._recent_errors = self._recent_errors[-self._error_window_size :]
                if status and status >= 500:
                    try:
                        self._mark_endpoint_failure(endpoint, "http")
                    except Exception:
                        pass
                if attempt == attempts - 1:
                    logger.error(
                        "Note generator call failed after {} attempts (status={}): {} snippet={}",
                        attempts,
                        status,
                        exc,
                        snippet,
                    )
                    raise
                nominal = self._retry_backoff_base * (2 ** attempt)
                jitter = nominal * random.uniform(0.0, self._retry_jitter_frac)
                wait = min(self._retry_backoff_max_sec, nominal + jitter)
                remain = max(0.0, self._retry_total_cap_sec - total_wait)
                wait = min(wait, remain)
                logger.warning(
                    "HTTP error (attempt={}/{} status={}); backing off {:.2f}s (cap left {:.2f}s): {} snippet={}",
                    attempt + 1,
                    attempts,
                    status,
                    wait,
                    remain,
                    exc,
                    snippet,
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
            finally:
                self._inflight_tokens[endpoint] = max(
                    0, self._inflight_tokens.get(endpoint, 0) - req_tokens
                )

    def generate_for_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id, chunk_id = chunk["doc_id"], chunk["chunk_id"]
        chunk_text = chunk["text"]
        doc_title = (chunk.get("meta") or {}).get("doc_title") or chunk.get("doc_title") or doc_id
        prompt = self.build_prompt(chunk_text, doc_id, doc_title=doc_title)
        stop_sequences = self._stop_sequences or None
        call_max_tokens = self._parsing_max_tokens or self.max_tokens

        attempts = max(1, self._parse_retry + 1)
        parsed_notes: List[Dict[str, Any]] = []
        parser_run_stats: Dict[str, Any] = {}
        last_raw = ""
        t0 = t1 = t2 = time.time()
        max_attempts = attempts
        prompt_for_attempt = prompt
        reminder_used = False
        attempt = 0
        while attempt < max_attempts:
            t0 = time.time()
            raw = self._call(prompt_for_attempt, stop=stop_sequences, max_tokens=call_max_tokens)
            last_raw = raw
            t1 = time.time()

            parsed_notes = self.parser.parse(raw, doc_id)
            if parsed_notes:
                try:
                    self._resolve_pronoun_subjects(chunk, parsed_notes)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Pronoun resolution skipped doc={} chunk={} err={}",
                        doc_id,
                        chunk_id,
                        exc,
                    )
            t2 = time.time()
            parser_run_stats = self.parser.get_stats(cumulative=False)
            total_budget = max_attempts + (1 if self._retry_reminder_enabled and not reminder_used else 0)
            if parser_run_stats.get("json_parse_failures"):
                logger.warning(
                    "Parsing failed doc={} chunk={} attempt={}/{} stats={}",
                    doc_id,
                    chunk_id,
                    attempt + 1,
                    total_budget,
                    parser_run_stats,
                )
            if parsed_notes:
                break

            should_remind = (
                self._retry_reminder_enabled
                and not reminder_used
                and parser_run_stats.get("json_parse_failures")
            )
            attempt += 1
            if should_remind and attempt >= max_attempts:
                max_attempts += 1
            if attempt >= max_attempts:
                break

            if should_remind:
                prompt_for_attempt = self._attach_format_reminder(prompt)
                reminder_used = True
            else:
                prompt_for_attempt = prompt
            # Light backoff before reissuing the prompt to reduce upstream churn.
            time.sleep(min(1.0, 0.25 * attempt))

        # Final salvage: if仍然无法解析，则请求模型把上一次输出修正为严格 JSON
        if not parsed_notes and last_raw:
            try:
                repair_prompt = render_prompt("note_repair_parse.txt", last_raw=last_raw)
                t0 = time.time()
                repaired_raw = self._call(repair_prompt, stop=stop_sequences, max_tokens=call_max_tokens)
                t1 = time.time()
                parsed_notes = self.parser.parse(repaired_raw, doc_id)
                if parsed_notes:
                    try:
                        self._resolve_pronoun_subjects(chunk, parsed_notes)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "Pronoun resolution skipped after repair doc={} chunk={} err={}",
                            doc_id,
                            chunk_id,
                            exc,
                        )
                t2 = time.time()
                parser_run_stats = self.parser.get_stats(cumulative=False)
                if parser_run_stats.get("json_parse_failures"):
                    logger.warning(
                        "Parsing failed after repair doc={} chunk={} stats={}",
                        doc_id,
                        chunk_id,
                        parser_run_stats,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Repair attempt failed doc={} chunk={} err={}", doc_id, chunk_id, exc)

        if not parsed_notes:
            self._record_stage_times(int((t1 - t0) * 1000), int((t2 - t1) * 1000), 0)
            self._processed_chunks += 1
            self._maybe_log_stage_p95()
            return []

        serialized = json.dumps(parsed_notes, ensure_ascii=False)
        t3 = time.time()
        validation_result = validate_and_normalize(serialized, doc_id, chunk_id)
        t4 = time.time()
        errors = validation_result.get("errors") or []
        if errors and self._validation_retry > 0 and last_raw:
            best_result = validation_result
            best_errors = errors
            best_count = len(best_result.get("valid_notes") or [])
            for retry_idx in range(self._validation_retry):
                error_summary = self._summarize_validation_errors(best_errors)
                repair_prompt = render_prompt(
                    "note_repair_validation.txt",
                    error_summary=error_summary,
                    last_raw=last_raw,
                )
                repaired_raw = self._call(repair_prompt, stop=stop_sequences, max_tokens=call_max_tokens)
                parsed_notes = self.parser.parse(repaired_raw, doc_id)
                if parsed_notes:
                    try:
                        self._resolve_pronoun_subjects(chunk, parsed_notes)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "Pronoun resolution skipped after validation retry doc={} chunk={} err={}",
                            doc_id,
                            chunk_id,
                            exc,
                        )
                validation_result = validate_and_normalize(
                    json.dumps(parsed_notes, ensure_ascii=False), doc_id, chunk_id
                )
                errors = validation_result.get("errors") or []
                note_count = len(validation_result.get("valid_notes") or [])
                if note_count > best_count or len(errors) < len(best_errors):
                    best_result = validation_result
                    best_errors = errors
                    best_count = note_count
                last_raw = repaired_raw
                if not errors:
                    break
            validation_result = best_result
            errors = best_errors
            if best_errors:
                self._stats["validation_retries_exhausted"] = (
                    self._stats.get("validation_retries_exhausted", 0) + 1
                )
        if errors:
            logger.warning("Validation issues doc={} chunk={} details={}", doc_id, chunk_id, errors)
            self._stats["validation_failures"] = self._stats.get("validation_failures", 0) + 1
        self._record_stage_times(int((t1 - t0) * 1000), int((t2 - t1) * 1000), int((t4 - t3) * 1000))
        self._processed_chunks += 1
        self._maybe_log_stage_p95()
        return validation_result

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

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip())

    @staticmethod
    def _pick_entity_candidate(candidates: List[str], subj_type: str | None) -> str | None:
        if not candidates:
            return None
        prefer_person = (subj_type or "").upper() == "PERSON"
        if prefer_person:
            for cand in candidates:
                tokens = cand.split()
                if len(tokens) >= 2:
                    return cand
        return candidates[0]

    def _resolve_pronoun_subjects(self, chunk: Dict[str, Any], notes: List[Dict[str, Any]]) -> None:
        """Best-effort coref: replace pronoun-only subjects using nearby context."""
        if not chunk or not notes:
            return
        text = chunk.get("text") or ""
        if not text.strip():
            return
        cleaned_chunk = self._normalize_text(text)
        sent_spans = chunk.get("meta", {}).get("sent_spans")
        if not isinstance(sent_spans, list) or not sent_spans:
            sent_spans = split_into_entity_aware_spans(text)

        sentence_records: List[Dict[str, Any]] = []
        for span in sent_spans:
            if isinstance(span, dict):
                sentence_text = span.get("text") or ""
                start = int(span.get("start", 0))
                end = int(span.get("end", start))
            else:
                sentence_text = str(span)
                normalized_sentence = self._normalize_text(sentence_text)
                start = cleaned_chunk.find(normalized_sentence)
                end = start + len(normalized_sentence) if start >= 0 else start
            entities = TextUtils.extract_entity_candidates(sentence_text)
            sentence_records.append(
                {
                    "text": sentence_text,
                    "start": max(0, start),
                    "end": max(0, end),
                    "entities": [e for e in entities if not TextUtils.is_pronoun(e)],
                }
            )

        if not sentence_records:
            return

        def locate_sentence(evidence_text: str) -> int | None:
            if not evidence_text:
                return None
            needle = self._normalize_text(evidence_text).lower()
            if not needle:
                return None
            idx = cleaned_chunk.lower().find(needle)
            if idx == -1:
                return None
            for i, record in enumerate(sentence_records):
                if record["start"] <= idx < record["end"]:
                    return i
            # fallback: substring containment
            for i, record in enumerate(sentence_records):
                if needle in self._normalize_text(record["text"]).lower():
                    return i
            return None

        for note in notes:
            subj = (note.get("subj") or "").strip()
            if not subj or not TextUtils.is_pronoun(subj):
                continue
            meta = note.setdefault("meta", {})
            profile = meta.get("subject_profile")
            subj_type = note.get("subj_type") or "CONCEPT"
            resolved = None
            if isinstance(profile, dict):
                aliases = profile.get("aliases") or []
                for alias in aliases:
                    if alias and not TextUtils.is_pronoun(alias):
                        resolved = alias
                        break
            target_sentence = locate_sentence(note.get("evidence") or "")
            if not resolved and target_sentence is not None:
                for idx in range(target_sentence - 1, -1, -1):
                    candidate = self._pick_entity_candidate(sentence_records[idx]["entities"], subj_type)
                    if candidate:
                        resolved = candidate
                        break
            if not resolved:
                continue

            note["subj"] = resolved
            if not isinstance(profile, dict):
                profile = {
                    "type": subj_type,
                    "aliases": [],
                    "nationality": [],
                    "birth": None,
                    "death": None,
                    "occupations": [],
                    "titles": [],
                    "categories": [],
                    "same_as": [],
                    "description": None,
                }
            profile.setdefault("aliases", [])
            if resolved not in profile["aliases"]:
                profile["aliases"].insert(0, resolved)
            profile["type"] = subj_type
            meta["subject_profile"] = profile
            meta["subject_source"] = meta.get("subject_source") or "chunk_context"
            prior_conf = float(meta.get("subject_confidence") or 0.0)
            meta["subject_confidence"] = max(prior_conf, 0.75)

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
