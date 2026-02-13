from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from relrag.prompt import render_prompt
from relrag.utils.text_builders import build_note_text_for_rank
from relrag.utils.llm_client import LLMChatClient
from relrag.utils.openai_client import chat_completion as openai_chat_completion
from relrag.utils.context_budget import (
    apply_load_shed,
    budget_rerank_prompt,
    log_budget_event,
)
from relrag.utils.llm_errors import ContextLengthError
from relrag.utils.vllm_runtime import detect_vllm_served_model
import os


class LLMReranker:
    PROMPT_NAME = "rerank.txt"

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        lm_cfg: Optional[Dict[str, Any]] = None,
        *,
        base_cfg: Optional[Dict[str, Any]] = None,
        run_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or {}
        self.base_cfg = base_cfg or {}
        runtime_cfg = self.base_cfg.get("runtime") if isinstance(self.base_cfg.get("runtime"), dict) else {}
        self.run_dir = run_dir or runtime_cfg.get("run_dir") or self.base_cfg.get("run_dir")
        self.enabled = bool(self.cfg.get("enabled", False))
        self.type = self.cfg.get("type", "llm")
        self.provider = str(self.cfg.get("provider") or self.cfg.get("backend") or "vllm").strip().lower()
        self.openai_cfg = self.cfg.get("openai") if isinstance(self.cfg.get("openai"), dict) else {}
        if self.provider == "openai":
            self.llm_cfg = self.openai_cfg
        else:
            rerank_llm_cfg = self.cfg.get("llm") if isinstance(self.cfg.get("llm"), dict) else {}
            shared_llm_cfg = lm_cfg if isinstance(lm_cfg, dict) else {}
            merged_llm_cfg: Dict[str, Any] = dict(rerank_llm_cfg)
            # Enforce reranker endpoint/model to follow current reader runtime.
            if shared_llm_cfg.get("endpoint"):
                merged_llm_cfg["endpoint"] = shared_llm_cfg.get("endpoint")
            if shared_llm_cfg.get("model"):
                merged_llm_cfg["model"] = shared_llm_cfg.get("model")
            self.llm_cfg = merged_llm_cfg
        self.endpoint = self.llm_cfg.get("endpoint")
        self.model = self.llm_cfg.get("model")
        if self.provider != "openai" and self.endpoint:
            runtime_model = detect_vllm_served_model(self.endpoint)
            if runtime_model:
                configured = str(self.model or "").strip()
                if configured and configured != runtime_model:
                    logger.info(
                        "Reranker detected running vLLM model {} at {}; overriding configured model {}",
                        runtime_model,
                        self.endpoint,
                        configured,
                    )
                self.model = runtime_model
        self.batch = int(self.llm_cfg.get("batch", 8))
        if isinstance(self.model, str) and "gpt-oss" in self.model.lower():
            # Smaller batch lowers formatting errors for strict rank-only output.
            self.batch = min(self.batch, 6)
        self.timeout = int(self.llm_cfg.get("timeout_s", 10))
        self.client: Optional[LLMChatClient] = None
        if self.type != "llm":
            self.enabled = False
        elif self.provider == "openai":
            if not self.openai_cfg.get("model"):
                self.enabled = False
        else:
            if not self.endpoint or not self.model:
                self.enabled = False
            else:
                self.client = LLMChatClient(
                    endpoint=self.endpoint,
                    model=self.model,
                    llm_profile="extract",
                    timeout=self.timeout,
                )

    def _resolve_openai_key(self) -> str:
        key = self.openai_cfg.get("api_key")
        if key:
            return str(key)
        env_name = self.openai_cfg.get("api_key_env", "OPENAI_API_KEY")
        return os.environ.get(str(env_name), "")

    def score(self, question: str, candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not self.enabled or not candidates:
            return {}
        results: Dict[str, Dict[str, Any]] = {}
        rerank_limits = self.base_cfg.get("rerank") if isinstance(self.base_cfg.get("rerank"), dict) else {}
        base_max_items = int(rerank_limits.get("max_candidates", 12))
        base_max_item_tokens = rerank_limits.get("max_candidate_tokens")
        try:
            base_max_item_tokens = int(base_max_item_tokens) if base_max_item_tokens is not None else None
        except (TypeError, ValueError):
            base_max_item_tokens = None
        # Rank-only output requires much fewer tokens than score+label JSON.
        requested_max_tokens = int(self.cfg.get("max_tokens", 128))
        is_gpt_oss = isinstance(self.model, str) and "gpt-oss" in self.model.lower()
        if is_gpt_oss:
            # gpt-oss may prepend boilerplate text before rank output; keep a safer output budget.
            requested_max_tokens = max(requested_max_tokens, 512)
        for start in range(0, len(candidates), self.batch):
            chunk = candidates[start : start + self.batch]
            max_items_override = None
            max_item_tokens_override = None
            max_tokens_override = requested_max_tokens
            attempt = 0
            parse_retry_done = False
            scores: Dict[int, Dict[str, Any]] = {}
            used_items: List[Dict[str, Any]] = []
            while True:
                budgeted = budget_rerank_prompt(
                    question,
                    chunk,
                    prompt_name=self.PROMPT_NAME,
                    cfg=self.base_cfg,
                    llm_cfg=self.llm_cfg,
                    requested_max_tokens=max_tokens_override,
                    max_items_override=max_items_override,
                    max_item_tokens_override=max_item_tokens_override,
                )
                used_items = budgeted.items
                if not used_items:
                    scores = self._prefused_scores(chunk)
                    used_items = chunk
                    break
                log_budget_event(
                    self.run_dir,
                    stage="rerank",
                    phase="pre",
                    report=budgeted.report,
                    attempt=attempt + 1,
                    extra={"chunk_size": len(chunk)},
                )
                try:
                    if self.provider == "openai":
                        api_key = self._resolve_openai_key()
                        if not api_key:
                            raise RuntimeError("OpenAI API key is required for reranker.")
                        content = openai_chat_completion(
                            budgeted.messages,
                            model=self.openai_cfg.get("model"),
                            api_key=api_key,
                            base_url=self.openai_cfg.get("base_url"),
                            temperature=0.0,
                            max_tokens=budgeted.report.effective_max_tokens,
                            timeout_sec=self.openai_cfg.get("timeout_sec", 60.0),
                            max_retries=self.openai_cfg.get("max_retries", 2),
                            retry_backoff_sec=self.openai_cfg.get("retry_backoff_sec", 1.0),
                            retry_backoff_max_sec=self.openai_cfg.get("retry_backoff_max_sec", 20.0),
                        )
                    else:
                        if not self.client:
                            raise RuntimeError("LLM reranker client not initialized")
                        response = self.client.chat(
                            budgeted.messages,
                            temperature=0.0,
                            max_tokens=budgeted.report.effective_max_tokens,
                            llm_profile="extract",
                        )
                        content = response.content
                    note_id_to_idx = self._note_id_to_idx(used_items)
                    expected_count = len(used_items)
                    scores = self._parse_scores(content, expected_count, note_id_to_idx=note_id_to_idx)
                    if not scores:
                        # Local auto-repair for almost-complete rank lists (common in gpt-oss):
                        # accept and append only a few missing indices in original order.
                        auto_min = max(2, expected_count - 2)
                        locally_repaired = self._parse_scores(
                            content,
                            expected_count,
                            note_id_to_idx=note_id_to_idx,
                            min_items_for_autocomplete=auto_min,
                        )
                        if locally_repaired:
                            scores = locally_repaired
                    if not scores:
                        repaired = self._repair_scores(
                            question,
                            content,
                            expected_count,
                            note_id_to_idx=note_id_to_idx,
                        )
                        if repaired:
                            scores = repaired
                        elif not parse_retry_done:
                            parse_retry_done = True
                            retry_cap = 1536 if is_gpt_oss else 1024
                            retry_floor = 512 if is_gpt_oss else 256
                            max_tokens_override = min(retry_cap, max(int(max_tokens_override * 2), retry_floor))
                            logger.debug(
                                "LLM rerank returned invalid payload; retrying once with larger max_tokens={} (chunk_size={})",
                                max_tokens_override,
                                len(used_items),
                            )
                            continue
                        else:
                            preview = (content or "").replace("\n", " ")[:320]
                            raise RuntimeError(
                                f"LLM rerank returned empty/invalid score payload (preview={preview})"
                            )
                    log_budget_event(
                        self.run_dir,
                        stage="rerank",
                        phase="post",
                        report=budgeted.report,
                        attempt=attempt + 1,
                        extra={"response_chars": len(content)},
                    )
                    break
                except ContextLengthError as exc:
                    if attempt >= 1:
                        raise
                    current_max_items = max_items_override or base_max_items
                    current_max_item_tokens = max_item_tokens_override or base_max_item_tokens
                    before = {
                        "max_items": current_max_items,
                        "max_item_tokens": current_max_item_tokens,
                        "requested_max_tokens": max_tokens_override,
                    }
                    max_items_override, max_item_tokens_override, max_tokens_override = apply_load_shed(
                        max_items=current_max_items,
                        max_item_tokens=current_max_item_tokens,
                        requested_max_tokens=max_tokens_override,
                    )
                    after = {
                        "max_items": max_items_override,
                        "max_item_tokens": max_item_tokens_override,
                        "requested_max_tokens": max_tokens_override,
                    }
                    log_budget_event(
                        self.run_dir,
                        stage="rerank",
                        phase="load_shed",
                        report=budgeted.report,
                        attempt=attempt + 1,
                        extra={
                            "reason": "context_len",
                            "error": str(exc)[:200],
                            "load_shed_before": before,
                            "load_shed_after": after,
                        },
                    )
                    attempt += 1
                except Exception as exc:  # noqa: PERF203
                    logger.warning("LLM rerank failed, fallback to pre-fused scores: {}", exc)
                    scores = self._prefused_scores(chunk)
                    used_items = chunk
                    break
            used_note_ids = {item.get("note_id") for item in used_items if item.get("note_id")}
            dropped_items = [item for item in chunk if item.get("note_id") not in used_note_ids]
            if dropped_items and not scores:
                scores = self._prefused_scores(dropped_items)
            fallback_scores = self._prefused_scores(dropped_items) if dropped_items else {}
            for idx, item in enumerate(used_items, start=1):
                note_id = item.get("note_id")
                if not note_id:
                    continue
                entry = scores.get(idx, {"score": 0.0, "labels": []})
                results[note_id] = {
                    "score": float(entry.get("score", 0.0)),
                    "labels": entry.get("labels") or [],
                }
            if dropped_items:
                for idx, item in enumerate(dropped_items, start=1):
                    note_id = item.get("note_id")
                    if not note_id or note_id in results:
                        continue
                    entry = fallback_scores.get(idx, {"score": 0.0, "labels": ["fallback"]})
                    results[note_id] = {
                        "score": float(entry.get("score", 0.0)),
                        "labels": entry.get("labels") or ["fallback"],
                    }
        return results

    def _build_payload(self, question: str, chunk: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for idx, candidate in enumerate(chunk, start=1):
            note = candidate.get("note") or {}
            text = build_note_text_for_rank(note, max_len=768)
            lines.append(f"{idx}. note_id={note.get('note_id')} :: {text}")
        return render_prompt(self.PROMPT_NAME, question=question, candidates="\n".join(lines))

    def _parse_scores(
        self,
        content: str,
        expected: int,
        *,
        note_id_to_idx: Optional[Dict[str, int]] = None,
        min_items_for_autocomplete: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        text = self._strip_reasoning(content.strip())
        parsed = self._parse_scores_json(
            text,
            expected,
            note_id_to_idx=note_id_to_idx,
            min_items_for_autocomplete=min_items_for_autocomplete,
        )
        if parsed:
            return parsed
        parsed = self._parse_scores_lines(text, expected, note_id_to_idx=note_id_to_idx)
        if parsed:
            return parsed
        parsed = self._parse_rank_only(
            text,
            expected,
            note_id_to_idx=note_id_to_idx,
            min_items_for_autocomplete=min_items_for_autocomplete,
        )
        return parsed

    @staticmethod
    def _note_id_to_idx(items: List[Dict[str, Any]]) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for idx, item in enumerate(items, start=1):
            note = item.get("note") if isinstance(item.get("note"), dict) else {}
            note_id = item.get("note_id") or note.get("note_id")
            if not note_id:
                continue
            mapping[str(note_id)] = idx
        return mapping

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return stripped

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        if not text:
            return ""
        output = re.sub(r"(?is)<think>.*?</think>", "", text)
        output = output.replace("<think>", "").replace("</think>", "")
        return output.strip()

    @staticmethod
    def _normalize_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if 0.0 <= score <= 1.0:
            score *= 100.0
        return max(0.0, min(100.0, score))

    @staticmethod
    def _normalize_labels(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return []

    @staticmethod
    def _scores_from_rank_order(
        rank_order: List[int],
        expected: int,
        *,
        require_complete: bool = True,
        min_items_for_autocomplete: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        order: List[int] = []
        seen = set()
        for idx in rank_order:
            if idx <= 0 or idx > expected:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
        if require_complete:
            if len(order) != expected:
                if min_items_for_autocomplete > 0 and len(order) >= min_items_for_autocomplete:
                    for idx in range(1, expected + 1):
                        if idx not in seen:
                            order.append(idx)
                else:
                    return {}
        else:
            for idx in range(1, expected + 1):
                if idx not in seen:
                    order.append(idx)
        if not order:
            return {}
        if len(order) == 1:
            return {order[0]: {"score": 100.0, "labels": ["rank_only"]}}
        denom = float(len(order) - 1)
        scores: Dict[int, Dict[str, Any]] = {}
        for pos, idx in enumerate(order):
            score = 100.0 * (1.0 - (pos / denom))
            scores[idx] = {"score": max(0.0, min(100.0, score)), "labels": ["rank_only"]}
        return scores

    @staticmethod
    def _coerce_index(
        raw: Any,
        expected: int,
        note_id_to_idx: Optional[Dict[str, int]] = None,
    ) -> Optional[int]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            idx = int(raw)
            if 1 <= idx <= expected:
                return idx
            return None
        token = str(raw).strip()
        if not token:
            return None

        variants: List[str] = [token]
        stripped_quote = token.strip("`'\"")
        if stripped_quote and stripped_quote not in variants:
            variants.append(stripped_quote)
        stripped_punct = stripped_quote.strip("[](){}<>:;,.")
        if stripped_punct and stripped_punct not in variants:
            variants.append(stripped_punct)

        if note_id_to_idx:
            for candidate in variants:
                idx = note_id_to_idx.get(candidate)
                if idx is not None and 1 <= idx <= expected:
                    return idx

        for candidate in variants:
            if candidate.isdigit():
                idx = int(candidate)
                if 1 <= idx <= expected:
                    return idx
                continue

            match = re.fullmatch(
                r"(?i)(?:idx|index|id|rank|candidate|cand|no|num)?\s*[:=#-]?\s*(\d+)",
                candidate,
            )
            if match:
                idx = int(match.group(1))
                if 1 <= idx <= expected:
                    return idx
                continue

            # Accept wrappers like "#3", "(4)", "3." while avoiding note_id parsing.
            if re.search(r"[A-Za-z]", candidate):
                continue
            nums = re.findall(r"\d+", candidate)
            if len(nums) == 1:
                idx = int(nums[0])
                if 1 <= idx <= expected:
                    return idx
        return None

    def _parse_scores_payload(
        self,
        payload: Any,
        expected: int,
        *,
        note_id_to_idx: Optional[Dict[str, int]] = None,
        min_items_for_autocomplete: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        items: Any = payload
        if isinstance(payload, dict):
            for key in (
                "ranking",
                "ranked_indices",
                "ranked_index",
                "ranked_indexes",
                "ranked_ids",
                "order",
                "indices",
                "scores",
                "items",
                "results",
                "data",
            ):
                value = payload.get(key)
                if isinstance(value, (list, str)):
                    items = value
                    break
        parsed: Dict[int, Dict[str, Any]] = {}
        rank_order: List[int] = []
        if isinstance(items, list) and items and all(isinstance(item, (int, float)) for item in items):
            numeric = [int(item) for item in items]
            if numeric and all(1 <= idx <= expected for idx in numeric):
                return self._scores_from_rank_order(
                    numeric,
                    expected,
                    require_complete=True,
                    min_items_for_autocomplete=min_items_for_autocomplete,
                )
            for idx, score in enumerate(items, start=1):
                if idx > expected:
                    break
                parsed[idx] = {"score": self._normalize_score(score), "labels": []}
            return parsed
        if isinstance(items, list) and items and all(isinstance(item, str) for item in items):
            for token in items:
                idx = self._coerce_index(token, expected, note_id_to_idx=note_id_to_idx)
                if idx is not None:
                    rank_order.append(idx)
            if rank_order:
                return self._scores_from_rank_order(
                    rank_order,
                    expected,
                    require_complete=True,
                    min_items_for_autocomplete=min_items_for_autocomplete,
                )
        if isinstance(items, str):
            parsed = self._parse_rank_only(
                items,
                expected,
                note_id_to_idx=note_id_to_idx,
                min_items_for_autocomplete=min_items_for_autocomplete,
            )
            if parsed:
                return parsed
        if not isinstance(items, list):
            return parsed
        for item in items:
            idx = 0
            score = 0.0
            labels: List[str] = []
            if isinstance(item, dict):
                raw_idx = (
                    item.get("idx")
                    or item.get("index")
                    or item.get("id")
                    or item.get("rank")
                    or item.get("note_id")
                )
                resolved_idx = self._coerce_index(raw_idx, expected, note_id_to_idx=note_id_to_idx)
                if any(key in item for key in ("score", "confidence", "relevance", "value")):
                    if resolved_idx is None:
                        continue
                    idx = resolved_idx
                    score = self._normalize_score(
                        item.get("score", item.get("confidence", item.get("relevance", item.get("value", 0.0))))
                    )
                    labels = self._normalize_labels(item.get("labels", item.get("tags", item.get("reasons", []))))
                elif resolved_idx is not None:
                    rank_order.append(resolved_idx)
                    continue
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    idx = int(item[0])
                except (TypeError, ValueError):
                    idx = 0
                score = self._normalize_score(item[1])
                if len(item) >= 3:
                    labels = self._normalize_labels(item[2])
            if idx <= 0 or idx > expected:
                continue
            parsed[idx] = {"score": score, "labels": labels}
        if parsed:
            return parsed
        if rank_order:
            return self._scores_from_rank_order(
                rank_order,
                expected,
                require_complete=True,
                min_items_for_autocomplete=min_items_for_autocomplete,
            )
        return parsed

    def _parse_scores_json(
        self,
        content: str,
        expected: int,
        *,
        note_id_to_idx: Optional[Dict[str, int]] = None,
        min_items_for_autocomplete: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        text = self._strip_code_fence(content)
        candidates: List[str] = [text]
        start_list = text.find("[")
        end_list = text.rfind("]")
        if start_list >= 0 and end_list > start_list:
            candidates.append(text[start_list : end_list + 1])
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj >= 0 and end_obj > start_obj:
            candidates.append(text[start_obj : end_obj + 1])
        seen = set()
        for segment in candidates:
            segment = segment.strip()
            if not segment or segment in seen:
                continue
            seen.add(segment)
            for loader in (json.loads, ast.literal_eval):
                try:
                    payload = loader(segment)
                except Exception:
                    continue
                parsed = self._parse_scores_payload(
                    payload,
                    expected,
                    note_id_to_idx=note_id_to_idx,
                    min_items_for_autocomplete=min_items_for_autocomplete,
                )
                if parsed:
                    return parsed
        return {}

    def _parse_scores_lines(
        self,
        content: str,
        expected: int,
        *,
        note_id_to_idx: Optional[Dict[str, int]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        text = self._strip_code_fence(content)
        parsed: Dict[int, Dict[str, Any]] = {}
        patterns = [
            re.compile(
                r"(?i)(?:idx|index|id|rank)\s*[:=#]?\s*(\d+).*?"
                r"(?:score|confidence|relevance)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"
            ),
            re.compile(
                r"(?i)(?:note_id)\s*[:=#]?\s*([A-Za-z0-9_.:/@-]+).*?"
                r"(?:score|confidence|relevance)\s*[:=]?\s*(-?\d+(?:\.\d+)?)"
            ),
            re.compile(r"^\s*(\d+)\s*[:\-]\s*(-?\d+(?:\.\d+)?)\b"),
            re.compile(r"^\s*\|\s*(\d+)\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|"),
        ]
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for pattern in patterns:
                match = pattern.search(stripped)
                if not match:
                    continue
                idx = self._coerce_index(match.group(1), expected, note_id_to_idx=note_id_to_idx)
                if idx is None:
                    continue
                if idx <= 0 or idx > expected:
                    continue
                score = self._normalize_score(match.group(2))
                labels_match = re.search(r"\[(.*?)\]", stripped)
                labels = []
                if labels_match:
                    labels = [token.strip() for token in labels_match.group(1).split(",") if token.strip()]
                parsed[idx] = {"score": score, "labels": labels}
                break
        return parsed

    def _parse_rank_only(
        self,
        content: str,
        expected: int,
        *,
        note_id_to_idx: Optional[Dict[str, int]] = None,
        min_items_for_autocomplete: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        text = self._strip_code_fence(content)
        if not text:
            return {}
        rank_order: List[int] = []

        # Case 1: a plain list literal, e.g. [3,1,2] or ["note_a","note_b"].
        if text.startswith("[") and text.endswith("]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    payload = loader(text)
                except Exception:
                    continue
                if isinstance(payload, list):
                    for raw in payload:
                        idx = self._coerce_index(raw, expected, note_id_to_idx=note_id_to_idx)
                        if idx is not None:
                            rank_order.append(idx)
                    break
        if rank_order:
            return self._scores_from_rank_order(
                rank_order,
                expected,
                require_complete=True,
                min_items_for_autocomplete=min_items_for_autocomplete,
            )

        # Case 2: numbered lines where left side is rank and right side is candidate id.
        line_pairs: List[tuple[int, int]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^\s*(\d+)\s*[\).:\-]\s*(.+?)\s*$", stripped)
            if not match:
                continue
            try:
                rank_pos = int(match.group(1))
            except (TypeError, ValueError):
                continue
            right = match.group(2).strip()
            idx = self._coerce_index(right, expected, note_id_to_idx=note_id_to_idx)
            if idx is None:
                for token in re.split(r"[\s,>\-\|]+", right):
                    idx = self._coerce_index(token, expected, note_id_to_idx=note_id_to_idx)
                    if idx is not None:
                        break
            if idx is not None:
                line_pairs.append((rank_pos, idx))
        if line_pairs:
            line_pairs.sort(key=lambda pair: pair[0])
            ordered = [idx for _, idx in line_pairs]
            parsed = self._scores_from_rank_order(
                ordered,
                expected,
                require_complete=True,
                min_items_for_autocomplete=min_items_for_autocomplete,
            )
            if parsed:
                return parsed

        # Case 3: sequence text like "3, 1, 2" / "3 > 1 > 2" / "idx=3 idx=1 idx=2".
        raw_tokens = re.split(r"[\s,>\-\|\u3001\uFF0C\u2192\u2190;]+", text)
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            idx = self._coerce_index(token, expected, note_id_to_idx=note_id_to_idx)
            if idx is not None:
                rank_order.append(idx)
        if len({idx for idx in rank_order if 1 <= idx <= expected}) >= 1:
            return self._scores_from_rank_order(
                rank_order,
                expected,
                require_complete=True,
                min_items_for_autocomplete=min_items_for_autocomplete,
            )
        return {}

    def _repair_scores(
        self,
        question: str,
        raw_content: str,
        expected: int,
        *,
        note_id_to_idx: Optional[Dict[str, int]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        if not raw_content or expected <= 0:
            return {}
        clipped = raw_content[:6000]
        repair_prompt = (
            "You are a ranking-output repair assistant.\n"
            f"Question: {question}\n"
            f"Convert the following output into a single line of ranked candidate indices with exactly {expected} integers.\n"
            "Output format example: 3 1 2 4\n"
            "Rules: use 1-based candidate numbers only, no duplicates, no markdown, no prose.\n\n"
            "Model output to repair:\n"
            f"{clipped}"
        )
        messages = [{"role": "user", "content": repair_prompt}]
        max_tokens = min(256, max(64, expected * 8))
        try:
            if self.provider == "openai":
                api_key = self._resolve_openai_key()
                if not api_key:
                    return {}
                repaired_text = openai_chat_completion(
                    messages,
                    model=self.openai_cfg.get("model"),
                    api_key=api_key,
                    base_url=self.openai_cfg.get("base_url"),
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout_sec=self.openai_cfg.get("timeout_sec", 60.0),
                    max_retries=0,
                    retry_backoff_sec=self.openai_cfg.get("retry_backoff_sec", 1.0),
                    retry_backoff_max_sec=self.openai_cfg.get("retry_backoff_max_sec", 20.0),
                )
            else:
                if not self.client:
                    return {}
                repaired_resp = self.client.chat(
                    messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    llm_profile="extract",
                )
                repaired_text = repaired_resp.content
            parsed = self._parse_scores(repaired_text, expected, note_id_to_idx=note_id_to_idx)
            if parsed:
                return parsed
        except Exception as exc:
            logger.debug("LLM rerank repair failed: {}", exc)
        return {}

    def _prefused_scores(self, chunk: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        scores: Dict[int, Dict[str, Any]] = {}
        for idx, candidate in enumerate(chunk, start=1):
            base = candidate.get("pre_score")
            if base is None:
                base = candidate.get("hybrid_score", 0.0)
            try:
                score = float(base) * 100.0
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(100.0, score))
            scores[idx] = {"score": score, "labels": ["prefused"]}
        return scores
