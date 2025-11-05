from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple
import threading


class NoteParsingPipeline:
    _BARE_KEY = re.compile(r'([\{\s,])(\w+)(\s*:)')
    _TRAILING_COMMA = re.compile(r',(\s*[}\]])')
    _BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')
    _UNTERMINATED_FIX = re.compile(r'"\s*([}\]])')
    _ADJACENT_OBJECTS = re.compile(r'}\s*{')
    _FIELD_PATTERN = re.compile(
        r'"(?P<key>subj|pred|obj|subj_type|obj_type|evidence|meta)"\s*:\s*(?P<value>"(?:\\.|[^"])*"|\{[^{}]*\}|[^,\n]+)',
        re.IGNORECASE,
    )

    _CANONICAL_KEYS = {
        "subj": "subj",
        "pred": "pred",
        "obj": "obj",
        "subj_type": "subj_type",
        "obj_type": "obj_type",
        "evidence": "evidence",
        "meta": "meta",
    }

    _ALLOWED_TYPES = {"PERSON", "WORK", "ORG", "PLACE", "EVENT", "CONCEPT", "TIME"}

    DEFAULT_PARSING_CONFIG: Dict[str, Any] = {
        "allow_jsonl": True,
        "enable_array_packer": True,
        "enable_bare_key_fix": True,
        "enable_loose_extractor": True,
        "loose_split_key": "subj",
        "max_tokens": 1024,
        "stop": ['"]\n', "\n]", "\n\nEND", "END_JSON"],
    }

    DEFAULT_SCHEMA_CONFIG: Dict[str, Any] = {
        "min_evidence_len": 4,
        "max_evidence_len": 512,
        "type_map": {
            "GROUP": "ORG",
            "OBJECT": "CONCEPT",
            "NUMBER": "CONCEPT",
            "QUANTITY": "CONCEPT",
            "PERCENT": "CONCEPT",
            "DATE": "TIME",
            "YEAR": "TIME",
        },
    }

    def __init__(self, parsing_config: Dict[str, Any] | None = None, schema_config: Dict[str, Any] | None = None):
        self.parsing_config = self._merge_dict(self.DEFAULT_PARSING_CONFIG, parsing_config or {})
        self.schema_config = self._merge_dict(self.DEFAULT_SCHEMA_CONFIG, schema_config or {})
        self._type_map = {k.upper(): v.upper() for k, v in self.schema_config.get("type_map", {}).items()}
        self._stats: Dict[str, int] = {}
        self._last_stats: Dict[str, int] = {}
        self._lock = threading.Lock()

    def parse(self, text: str, doc_id: str | None = None) -> List[Dict[str, Any]]:
        normalized = self._normalize_json_text(text)
        run_stats: Dict[str, int] = {}
        packed_text: str | None = None

        if self._looks_like_array(normalized):
            parsed = self._try_load_array(normalized)
            if parsed is not None:
                return self._finalize(parsed, doc_id, run_stats, "strict_json_ok")

        if self.parsing_config.get("enable_array_packer", True):
            packed_text = self._pack_to_array(normalized)
            if packed_text != normalized:
                parsed = self._try_load_array(packed_text)
                if parsed is not None:
                    return self._finalize(parsed, doc_id, run_stats, "array_packer_used")

        if self.parsing_config.get("enable_bare_key_fix", True):
            target = packed_text or normalized
            quoted = self._quote_bare_keys(target)
            parsed = self._try_load_array(quoted)
            if parsed is not None:
                return self._finalize(parsed, doc_id, run_stats, "bare_key_fix_used")

        if self.parsing_config.get("enable_loose_extractor", True):
            loose = self._loose_extract(normalized)
            if loose:
                return self._finalize(loose, doc_id, run_stats, "loose_extractor_used")

        run_stats["json_parse_failures"] = 1
        return self._finalize([], doc_id, run_stats, None)

    def get_stats(self, cumulative: bool = True) -> Dict[str, int]:
        with self._lock:
            stats = self._stats if cumulative else self._last_stats
            return dict(stats)

    def _finalize(
        self,
        candidates: List[Dict[str, Any]] | None,
        doc_id: str | None,
        run_stats: Dict[str, int],
        stage_flag: str | None,
    ) -> List[Dict[str, Any]]:
        if stage_flag:
            run_stats[stage_flag] = run_stats.get(stage_flag, 0) + 1
        ready, drops, coerced = self._apply_schema_guard(candidates or [], doc_id)
        if drops:
            run_stats["schema_drops"] = run_stats.get("schema_drops", 0) + drops
        if coerced:
            run_stats["type_coercions"] = run_stats.get("type_coercions", 0) + coerced
        self._record_stats(run_stats)
        with self._lock:
            self._last_stats = dict(run_stats)
        return ready

    def _record_stats(self, run_stats: Dict[str, int]) -> None:
        with self._lock:
            for key, value in run_stats.items():
                self._stats[key] = self._stats.get(key, 0) + value

    @classmethod
    def _merge_dict(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def _normalize_json_text(cls, text: str) -> str:
        sanitized = (text or "").strip().strip("`").replace("\ufeff", "")
        sanitized = cls._BAD_ESCAPE.sub(r"\\\\", sanitized)
        sanitized = cls._TRAILING_COMMA.sub(r"\1", sanitized)
        sanitized = cls._UNTERMINATED_FIX.sub(r'"\1', sanitized)
        sanitized = cls._ADJACENT_OBJECTS.sub(r'}, {', sanitized)
        return sanitized

    @staticmethod
    def _looks_like_array(text: str) -> bool:
        stripped = text.lstrip()
        return stripped.startswith('[')

    def _pack_to_array(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith('['):
            return text

        if self.parsing_config.get("allow_jsonl", True):
            lines = [ln.strip().rstrip(',') for ln in text.splitlines() if ln.strip()]
            if lines and all(line.startswith('{') and line.endswith('}') for line in lines):
                return '[\n' + ',\n'.join(lines) + '\n]'

        sliced = self._slice_json_objects(text)
        if sliced:
            return '[\n' + ',\n'.join(sliced) + '\n]'

        return text

    @staticmethod
    def _slice_json_objects(text: str) -> List[str]:
        objs: List[str] = []
        depth = 0
        in_string = False
        escaped = False
        start = -1
        for idx, ch in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    if depth == 0:
                        start = idx
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start >= 0:
                            objs.append(text[start : idx + 1])
                            start = -1
        return objs

    @classmethod
    def _quote_bare_keys(cls, text: str) -> str:
        return cls._BARE_KEY.sub(r'\1"\2"\3', text)

    @staticmethod
    def _try_load_array(text: str) -> List[Dict[str, Any]] | None:
        try:
            data = json.loads(text)
        except Exception:
            return None
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return None
        if not data:
            return []
        objs = [obj for obj in data if isinstance(obj, dict)]
        return objs if objs else None

    def _loose_extract(self, text: str) -> List[Dict[str, Any]]:
        matches = list(self._FIELD_PATTERN.finditer(text))
        if not matches:
            return []

        split_key = (self.parsing_config.get("loose_split_key") or "subj").lower()
        split_key = self._CANONICAL_KEYS.get(split_key, split_key)

        objects: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        for match in matches:
            key = match.group("key").lower()
            canonical = self._CANONICAL_KEYS.get(key)
            if not canonical:
                continue
            value = self._clean_value(match.group("value"))
            if canonical == split_key and current:
                objects.append(current)
                current = {}
            if canonical == "meta":
                current[canonical] = self._parse_meta_value(value)
            else:
                current[canonical] = value
        if current:
            objects.append(current)
        return objects

    @staticmethod
    def _clean_value(raw: str) -> Any:
        value = raw.strip()
        if not value:
            return ""
        if value.startswith('"') and value.endswith('"'):
            inner = value[1:-1]
            try:
                return json.loads(f'"{inner}"')
            except Exception:
                return inner.replace('\\"', '"')
        if value.startswith('{') and value.endswith('}'):
            try:
                return json.loads(value)
            except Exception:
                return value
        if value.lower() == "null":
            return ""
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return value.strip()

    @staticmethod
    def _parse_meta_value(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _apply_schema_guard(
        self, candidates: List[Dict[str, Any]], doc_id: str | None
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        ready: List[Dict[str, Any]] = []
        drops = 0
        coerced = 0
        for candidate in candidates:
            normalized, coerced_flag = self._normalize_note(candidate, doc_id)
            if not normalized:
                drops += 1
                continue
            if coerced_flag:
                coerced += 1
            ready.append(normalized)
        return ready, drops, coerced

    def _normalize_note(self, note: Dict[str, Any], doc_id: str | None) -> Tuple[Dict[str, Any] | None, bool]:
        fields = {}
        for key in ("subj", "pred", "obj", "evidence"):
            value = note.get(key)
            if value is None:
                value = ""
            elif not isinstance(value, str):
                value = str(value)
            value = value.strip()
            fields[key] = value

        if not fields["subj"] or not fields["pred"] or not fields["obj"]:
            return None, False

        min_len = max(1, int(self.schema_config.get("min_evidence_len", 4)))
        max_len = max(min_len, int(self.schema_config.get("max_evidence_len", 512)))
        evidence = fields["evidence"]
        if len(evidence) < min_len:
            return None, False
        if len(evidence) > max_len:
            fields["evidence"] = evidence[:max_len]

        subj_type, subj_coerced = self._normalize_type(note.get("subj_type"), fields["subj"])
        obj_type, obj_coerced = self._normalize_type(note.get("obj_type"), fields["obj"])
        if not subj_type or not obj_type:
            return None, False

        meta = note.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        source = (meta.get("source") or "").strip() or (doc_id or "")
        confidence = self._coerce_confidence(meta.get("confidence"))
        meta = {"source": source, "confidence": confidence}

        normalized_note = {
            "subj": fields["subj"],
            "pred": fields["pred"],
            "obj": fields["obj"],
            "evidence": fields["evidence"],
            "subj_type": subj_type,
            "obj_type": obj_type,
            "meta": meta,
        }
        return normalized_note, bool(subj_coerced or obj_coerced)

    def _normalize_type(self, raw: Any, value: str) -> Tuple[str | None, bool]:
        if raw is None:
            return None, False
        candidate = str(raw).strip().upper()
        if not candidate:
            return None, False

        coerced = False
        value_clean = (value or "").strip()
        if re.fullmatch(r"\d{4}", value_clean) and candidate in {"NUMBER", "QUANTITY", "YEAR"}:
            return "TIME", True

        mapped = self._type_map.get(candidate, candidate)
        if mapped != candidate:
            coerced = True
        if mapped not in self._ALLOWED_TYPES:
            return None, False
        return mapped, coerced

    @staticmethod
    def _coerce_confidence(raw: Any) -> float:
        try:
            value = float(raw)
        except Exception:
            value = 0.0
        return min(1.0, max(0.0, value))
