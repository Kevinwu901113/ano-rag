"""Runtime patch for GraphRAG structured-output policy.

Loaded automatically via PYTHONPATH. This patch modifies GraphRAG's
LiteLLMCompletion guards so callers can choose behavior when LiteLLM reports a
model as "schema unsupported".

Policy is controlled by GRAPHRAG_SCHEMA_STRATEGY:
- strict: keep default behavior (raise).
- force: bypass guard and still call native schema mode.
- fallback: try native schema mode first, then fallback to json_object.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, get_args, get_origin

import pandas as pd
from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

LOGGER = logging.getLogger("graphrag.schema_policy")

_ORIGINAL_COMPLETION = None
_ORIGINAL_COMPLETION_ASYNC = None
_ORIGINAL_FINALIZE_COMMUNITY_REPORTS = None


def _schema_strategy() -> str:
    strategy = str(os.getenv("GRAPHRAG_SCHEMA_STRATEGY", "force")).strip().lower()
    if strategy not in {"strict", "force", "fallback"}:
        return "force"
    return strategy


def _supports_response_schema(model_id: str) -> bool:
    try:
        from graphrag_llm.completion.lite_llm_completion import supports_response_schema

        return bool(supports_response_schema(model_id))
    except Exception:
        return False


def _extract_json_dict(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        from graphrag.query.llm.text_utils import try_parse_json_object

        _, parsed = try_parse_json_object(text, verbose=False)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _default_value_for_field(field: Any) -> Any:
    default = getattr(field, "default", PydanticUndefined)
    if default is not PydanticUndefined:
        return default

    default_factory = getattr(field, "default_factory", None)
    if callable(default_factory):
        try:
            return default_factory()
        except Exception:
            pass

    annotation = getattr(field, "annotation", Any)
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list:
        return []
    if origin is dict:
        return {}
    if origin is tuple:
        return ()
    if origin is set:
        return set()
    if origin is frozenset:
        return frozenset()
    if origin is not None and type(None) in args:
        return None

    if annotation is str:
        return ""
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.model_construct()
    return None


def _normalize_model_payload(model_cls: type[BaseModel], payload: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if isinstance(payload, dict):
        normalized.update(payload)

    fields = getattr(model_cls, "model_fields", {})
    sentinel = object()
    for name, field in fields.items():
        value = normalized.get(name, sentinel)
        if value is sentinel or value is None:
            normalized[name] = _default_value_for_field(field)
            continue

        annotation = getattr(field, "annotation", Any)
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is list and args:
            item_type = args[0]
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                if not isinstance(value, list):
                    normalized[name] = []
                    continue
                fixed_items: list[Any] = []
                for item in value:
                    if isinstance(item, item_type):
                        fixed_items.append(item.model_dump())
                        continue
                    if isinstance(item, dict):
                        fixed_items.append(_normalize_model_payload(item_type, item))
                        continue
                    if isinstance(item, str):
                        sub_fields = getattr(item_type, "model_fields", {})
                        if "summary" in sub_fields and "explanation" in sub_fields:
                            fixed_items.append({"summary": item, "explanation": ""})
                normalized[name] = fixed_items
    return normalized


def _structured_response_from_json_mode(content: str, response_model: Any) -> Any:
    from graphrag_llm.utils import structure_completion_response

    try:
        return structure_completion_response(content, response_model)
    except Exception:
        if isinstance(response_model, type) and issubclass(response_model, BaseModel):
            payload = _extract_json_dict(content)
            normalized_payload = _normalize_model_payload(response_model, payload)
            try:
                return response_model.model_validate(normalized_payload)
            except Exception:
                return response_model.model_validate(
                    _normalize_model_payload(response_model, {})
                )
        raise


def _handle_streaming_with_schema(kwargs: dict[str, Any]) -> None:
    if kwargs.get("stream") is True:
        msg = "response_format is not supported for streaming completions."
        raise ValueError(msg)


def _schema_parse_retries() -> int:
    raw = str(os.getenv("GRAPHRAG_SCHEMA_PARSE_RETRIES", "2")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 2
    return max(0, min(value, 10))


def _apply_community_report_patch() -> None:
    global _ORIGINAL_FINALIZE_COMMUNITY_REPORTS

    try:
        import graphrag.index.operations.finalize_community_reports as finalize_module
        from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS
    except Exception:
        return

    if getattr(
        finalize_module, "_graphrag_empty_community_reports_patched", False
    ):
        return

    _ORIGINAL_FINALIZE_COMMUNITY_REPORTS = finalize_module.finalize_community_reports

    def _finalize_community_reports_safe(
        reports: pd.DataFrame,
        communities: pd.DataFrame,
    ) -> pd.DataFrame:
        if reports is None or len(reports) == 0 or "community" not in reports.columns:
            LOGGER.warning(
                "GraphRAG generated no usable community reports; writing empty community_reports table."
            )
            return pd.DataFrame(columns=COMMUNITY_REPORTS_FINAL_COLUMNS)
        return _ORIGINAL_FINALIZE_COMMUNITY_REPORTS(reports, communities)

    finalize_module.finalize_community_reports = _finalize_community_reports_safe
    finalize_module._graphrag_empty_community_reports_patched = True
    LOGGER.info("Enabled GraphRAG empty community report safeguard patch.")


def _build_completion_args(self: Any, kwargs: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    local_kwargs = dict(kwargs)
    messages = local_kwargs.pop("messages")
    response_format = local_kwargs.pop("response_format", None)
    request_metrics = local_kwargs.pop("metrics", None) or {}
    if not getattr(self, "_track_metrics", False):
        request_metrics = None
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return messages, response_format, request_metrics, local_kwargs


def _post_metrics(self: Any, request_metrics: Any) -> None:
    if request_metrics is not None:
        self._metrics_store.update_metrics(metrics=request_metrics)


def _completion_with_schema_policy(self: Any, /, **kwargs: Any) -> Any:
    response_format = kwargs.get("response_format")
    if not response_format or _supports_response_schema(getattr(self, "_model_id", "")):
        return _ORIGINAL_COMPLETION(self, **kwargs)

    strategy = _schema_strategy()
    if strategy == "strict":
        return _ORIGINAL_COMPLETION(self, **kwargs)

    _handle_streaming_with_schema(kwargs)
    messages, response_format, request_metrics, local_kwargs = _build_completion_args(
        self, kwargs
    )
    try:
        from graphrag_llm.utils import structure_completion_response

        retries = _schema_parse_retries()
        last_native_error: Exception | None = None
        total_attempts = retries + 1
        for attempt in range(total_attempts):
            native_resp = self._completion(
                messages=messages,
                metrics=request_metrics,
                response_format=response_format,
                **local_kwargs,
            )
            try:
                native_resp.formatted_response = structure_completion_response(
                    native_resp.content, response_format
                )
                return native_resp
            except Exception as native_error:  # pragma: no cover - runtime dependent
                last_native_error = native_error
                if attempt + 1 < total_attempts:
                    LOGGER.warning(
                        "Native schema parse failed for model '%s' (attempt %d/%d): %s",
                        getattr(self, "_model_id", "unknown"),
                        attempt + 1,
                        total_attempts,
                        native_error,
                    )

        if strategy != "fallback":
            if last_native_error is not None:
                raise last_native_error
            raise RuntimeError("Native schema parse failed without exception context.")

        LOGGER.warning(
            "Native schema parse failed for model '%s' after %d attempt(s); switching to json fallback.",
            getattr(self, "_model_id", "unknown"),
            total_attempts,
        )
        try:
            fallback_resp = self._completion(
                messages=messages,
                metrics=request_metrics,
                response_format_json_object=True,
                **local_kwargs,
            )
            fallback_resp.formatted_response = _structured_response_from_json_mode(
                fallback_resp.content, response_format
            )
            return fallback_resp
        except Exception:
            if last_native_error is not None:
                raise last_native_error
            raise
    finally:
        _post_metrics(self, request_metrics)


async def _completion_async_with_schema_policy(self: Any, /, **kwargs: Any) -> Any:
    response_format = kwargs.get("response_format")
    if not response_format or _supports_response_schema(getattr(self, "_model_id", "")):
        return await _ORIGINAL_COMPLETION_ASYNC(self, **kwargs)

    strategy = _schema_strategy()
    if strategy == "strict":
        return await _ORIGINAL_COMPLETION_ASYNC(self, **kwargs)

    _handle_streaming_with_schema(kwargs)
    messages, response_format, request_metrics, local_kwargs = _build_completion_args(
        self, kwargs
    )
    try:
        from graphrag_llm.utils import structure_completion_response

        retries = _schema_parse_retries()
        last_native_error: Exception | None = None
        total_attempts = retries + 1
        for attempt in range(total_attempts):
            native_resp = await self._completion_async(
                messages=messages,
                metrics=request_metrics,
                response_format=response_format,
                **local_kwargs,
            )
            try:
                native_resp.formatted_response = structure_completion_response(
                    native_resp.content, response_format
                )
                return native_resp
            except Exception as native_error:  # pragma: no cover - runtime dependent
                last_native_error = native_error
                if attempt + 1 < total_attempts:
                    LOGGER.warning(
                        "Native schema parse failed for model '%s' (attempt %d/%d): %s",
                        getattr(self, "_model_id", "unknown"),
                        attempt + 1,
                        total_attempts,
                        native_error,
                    )

        if strategy != "fallback":
            if last_native_error is not None:
                raise last_native_error
            raise RuntimeError("Native schema parse failed without exception context.")

        LOGGER.warning(
            "Native schema parse failed for model '%s' after %d attempt(s); switching to json fallback.",
            getattr(self, "_model_id", "unknown"),
            total_attempts,
        )
        try:
            fallback_resp = await self._completion_async(
                messages=messages,
                metrics=request_metrics,
                response_format_json_object=True,
                **local_kwargs,
            )
            fallback_resp.formatted_response = _structured_response_from_json_mode(
                fallback_resp.content, response_format
            )
            return fallback_resp
        except Exception:
            if last_native_error is not None:
                raise last_native_error
            raise
    finally:
        _post_metrics(self, request_metrics)


def _apply_patch() -> None:
    global _ORIGINAL_COMPLETION
    global _ORIGINAL_COMPLETION_ASYNC

    try:
        from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion
    except Exception:
        _apply_community_report_patch()
        return

    if getattr(LiteLLMCompletion, "_graphrag_schema_policy_patched", False):
        _apply_community_report_patch()
        return

    _ORIGINAL_COMPLETION = LiteLLMCompletion.completion
    _ORIGINAL_COMPLETION_ASYNC = LiteLLMCompletion.completion_async
    LiteLLMCompletion.completion = _completion_with_schema_policy
    LiteLLMCompletion.completion_async = _completion_async_with_schema_policy
    LiteLLMCompletion._graphrag_schema_policy_patched = True
    LOGGER.info(
        "Enabled GraphRAG schema policy patch (strategy=%s).",
        _schema_strategy(),
    )
    _apply_community_report_patch()


_apply_patch()
