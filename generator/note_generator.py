from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import requests
from loguru import logger

from config.config_loader import config as global_config
from generator.note_parsing import NoteParsingPipeline
from validators.note_validator import validate_and_normalize


class NoteGenerator:
    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
        parsing_config: Dict[str, Any] | None = None,
        schema_guard_config: Dict[str, Any] | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._stats: Dict[str, int] = {}

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
            "You are an information extraction system. From the following text, extract factual triple-notes.\n"
            "Return ONLY a valid JSON array (RFC 8259). Every property name and string value MUST be enclosed in double quotes. "
            "Do NOT add comments or trailing commas. If the array form is unstable, you may instead output JSONL (one JSON object per line, no commas, no surrounding brackets).\n"
            "Each item must be an object with keys:\n"
            '  "subj", "pred", "obj", "subj_type", "obj_type", "evidence", "meta"\n'
            "Constraints:\n"
            '  - "subj","pred","obj","evidence" MUST be non-empty strings.\n'
            '  - "evidence" is a verbatim snippet from the text (>= 4 chars).\n'
            '  - "subj_type" and "obj_type" MUST be one of ["PERSON","WORK","ORG","PLACE","EVENT","CONCEPT","TIME"].\n'
            f'  - "meta" = {{"source": "{doc_id}", "confidence": a float in [0,1]}}\n'
            "Text:\n"
            f'"""{source_text}"""\n'
            "Output JSON:\n"
            "[\n"
            f'  {{"subj":"...","pred":"...","obj":"...","subj_type":"PERSON","obj_type":"ORG","evidence":"...","meta":{{"source":"{doc_id}","confidence":0.95}}}}\n'
            "]"
        )

    def _call(self, prompt: str, *, stop: List[str] | None = None, max_tokens: int | None = None) -> str:
        retries = 2
        for attempt in range(retries + 1):
            try:
                payload: Dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if stop:
                    payload["stop"] = stop
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except requests.RequestException as exc:  # noqa: PERF203
                if attempt == retries:
                    raise
                wait = 2**attempt
                logger.warning("Note generator call failed (attempt={}): {}", attempt + 1, exc)
                time.sleep(wait)

    def generate_for_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id, chunk_id = chunk["doc_id"], chunk["chunk_id"]
        chunk_text = chunk["text"]
        prompt = self.build_prompt(chunk_text, doc_id)
        stop_sequences = self._stop_sequences or None
        call_max_tokens = self._parsing_max_tokens or self.max_tokens
        raw = self._call(prompt, stop=stop_sequences, max_tokens=call_max_tokens)

        parsed_notes = self.parser.parse(raw, doc_id)
        parser_run_stats = self.parser.get_stats(cumulative=False)
        if parser_run_stats.get("json_parse_failures"):
            logger.warning(
                "Parsing failed doc={} chunk={} stats={}",
                doc_id,
                chunk_id,
                parser_run_stats,
            )
        if not parsed_notes:
            return []

        serialized = json.dumps(parsed_notes, ensure_ascii=False)
        ok, notes_out, metrics = validate_and_normalize(serialized, doc_id, chunk_id)
        if not ok:
            logger.warning("Validation failed doc={} chunk={} details={}", doc_id, chunk_id, metrics)
            self._stats["validation_failures"] = self._stats.get("validation_failures", 0) + 1
            return []
        return notes_out

    def export_stats(self) -> Dict[str, int]:
        combined = self.parser.get_stats()
        for key, value in self._stats.items():
            combined[key] = combined.get(key, 0) + value
        return combined
