from __future__ import annotations

import json
import ast
import re
from typing import Any, Dict, List, Optional, Tuple
import threading

from config.attributes_loader import load_attributes_config
from schema.vocabulary import normalize_slot_value
from telemetry.metrics import record_attribute_guard


class NoteParsingPipeline:
    _BARE_KEY = re.compile(r'([\{\s,])(\w+)(\s*:)')
    _TRAILING_COMMA = re.compile(r',(\s*[}\]])')
    _BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')
    _UNTERMINATED_FIX = re.compile(r'"\s*([}\]])')
    _ADJACENT_OBJECTS = re.compile(r'}\s*{')
    _FIELD_PATTERN = re.compile(
        r'(?P<quote>["\'])(?P<key>subj|pred|obj|subj_type|obj_type|evidence|meta|subject|predicate|object|evidence_text|evid)\1\s*:\s*(?P<value>"(?:\\.|[^"])*"|\'(?:\\.|[^\'])*\'|\{[^{}]*\}|[^,\n]+)',
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
        "subject": "subj",
        "predicate": "pred",
        "object": "obj",
        "evidence_text": "evidence",
        "evid": "evidence",
    }

    _ALLOWED_TYPES = {"PERSON", "WORK", "ORG", "PLACE", "EVENT", "CONCEPT", "TIME"}

    DEFAULT_PARSING_CONFIG: Dict[str, Any] = {
        "allow_jsonl": True,
        "enable_array_packer": True,
        "enable_bare_key_fix": True,
        "enable_error_repair": True,
        "enable_bracket_balance_fix": True,
        "enable_loose_extractor": True,
        "loose_split_key": "subj",
        "max_tokens": 1024,
        "stop": ['"]\n', "\n]", "\n\nEND", "END_JSON"],
        "assume_valid_json": False,
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
        self._attribute_guard = AttributeGuard()
        self.assume_valid_json = bool(self.parsing_config.get("assume_valid_json", False))

    def parse(self, text: str, doc_id: str | None = None) -> List[Dict[str, Any]]:
        run_stats: Dict[str, int] = {}
        if self.assume_valid_json:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    data = [data]
                if isinstance(data, list):
                    return self._finalize(data, doc_id, run_stats, "strict_json_ok")
            except Exception:
                pass

        normalized = self._normalize_json_text(text)
        if self.parsing_config.get("enable_bracket_balance_fix", True):
            balanced = self._balance_brackets(normalized)
            if balanced != normalized:
                normalized = balanced
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

        if self.parsing_config.get("enable_error_repair", True):
            target = quoted or packed_text or normalized
            repaired, repair_flag = self._error_guided_repair(target)
            if repaired is not None:
                return self._finalize(repaired, doc_id, run_stats, repair_flag or "error_repair_used")

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
        if ready and self._attribute_guard:
            ready, guard_drops = self._attribute_guard.filter_notes(ready)
            if guard_drops:
                run_stats["attribute_guard_drops"] = run_stats.get("attribute_guard_drops", 0) + guard_drops
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
        parsed, _ = NoteParsingPipeline._load_json(text)
        return parsed

    @staticmethod
    def _load_json(text: str) -> Tuple[List[Dict[str, Any]] | None, Exception | None]:
        try:
            data = json.loads(text)
        except Exception as exc:
            json_err = exc
            try:
                data = ast.literal_eval(text)
            except Exception:
                return None, json_err
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return None, TypeError("json_not_array")
        if not data:
            return [], None
        objs = [obj for obj in data if isinstance(obj, dict)]
        return (objs if objs else None), None

    @staticmethod
    def _balance_brackets(text: str) -> str:
        stack: List[str] = []
        in_string = False
        escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
        if not stack:
            return text
        closers = "".join("}" if ch == "{" else "]" for ch in reversed(stack))
        return text + closers

    def _error_guided_repair(self, text: str) -> Tuple[List[Dict[str, Any]] | None, str | None]:
        parsed, err = self._load_json(text)
        if parsed is not None:
            return parsed, "strict_json_ok"
        if not isinstance(err, json.JSONDecodeError):
            return None, None

        trimmed = self._trim_extra_data(text, err)
        if trimmed and trimmed != text:
            parsed, _ = self._load_json(trimmed)
            if parsed is not None:
                return parsed, "extra_data_trimmed"

        balanced = self._balance_brackets(text)
        if balanced != text:
            parsed, _ = self._load_json(balanced)
            if parsed is not None:
                return parsed, "balance_fix_used"

        stripped = text.strip().rstrip(",")
        if stripped.startswith("{") and stripped.endswith("}"):
            wrapped = "[\n" + stripped + "\n]"
            parsed, _ = self._load_json(wrapped)
            if parsed is not None:
                return parsed, "array_wrap_used"

        return None, None

    @staticmethod
    def _trim_extra_data(text: str, err: json.JSONDecodeError) -> Optional[str]:
        message = (err.msg or "").lower()
        if "extra data" in message:
            cut = max(0, err.pos)
            candidate = text[:cut].rstrip()
            return candidate if candidate else None
        return None

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
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            inner = value[1:-1]
            quote = value[0]
            try:
                # Use the matching quote to avoid escaping issues.
                escaped = inner.replace("\\" + quote, quote)
                if quote == '"':
                    return json.loads(f'"{escaped}"')
                return ast.literal_eval(f"{quote}{escaped}{quote}")
            except Exception:
                return inner.replace('\\"', '"').replace("\\'", "'")
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

        raw_meta = note.get("meta")
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        source = (meta.get("source") or "").strip() or (doc_id or "")
        confidence = self._coerce_confidence(meta.get("confidence"))
        meta["source"] = source
        meta["confidence"] = confidence

        attribute_meta = meta.get("attribute")
        if not isinstance(attribute_meta, dict):
            attribute_meta = {}
        attr_name = attribute_meta.get("name") or fields["pred"]
        attr_name = str(attr_name or "").strip()
        attribute_meta["name"] = attr_name
        values = attribute_meta.get("values")
        if not isinstance(values, list) or not values:
            values = [
                {
                    "value": fields["obj"],
                    "normalized": fields["obj"],
                    "confidence": confidence,
                    "source": source,
                    "evidence": fields["evidence"],
                }
            ]
        attribute_meta["values"] = values
        meta["attribute"] = attribute_meta

        canonical_evidence = self._canonicalize_evidence(fields["evidence"], fields["subj"], subj_type)
        if canonical_evidence:
            meta["evidence_canonical"] = canonical_evidence
        meta.setdefault("validation", "strict")

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

    @staticmethod
    def _canonicalize_evidence(evidence: str, subject: str, subj_type: str) -> Optional[str]:
        if not evidence or not subject or (subj_type or "").upper() != "PERSON":
            return None
        pronouns = (
            "he",
            "she",
            "they",
            "his",
            "her",
            "their",
            "him",
            "hers",
        )
        pattern = re.compile(rf"^({'|'.join(pronouns)})\b", re.IGNORECASE)
        match = pattern.search(evidence.strip())
        if not match:
            return None
        replaced = pattern.sub(subject.strip(), evidence.strip(), count=1)
        return replaced


class AttributeGuard:
    def __init__(self) -> None:
        cfg = load_attributes_config()
        self.rules: Dict[str, Dict[str, Any]] = {}
        for name, payload in cfg.items():
            if not isinstance(payload, dict):
                continue
            entry: Dict[str, Any] = {}
            entry["lexicon"] = {
                str(val).strip().lower() for val in payload.get("value_lexicon", []) if isinstance(val, str) and val.strip()
            }
            blacklist = payload.get("value_blacklist") or []
            entry["blacklist"] = [str(val).strip().lower() for val in blacklist if isinstance(val, str) and val.strip()]
            def_patterns = payload.get("definition_patterns") or []
            entry["definition_patterns"] = [re.compile(pat, re.IGNORECASE) for pat in def_patterns if isinstance(pat, str)]
            appos_patterns = payload.get("appositive_patterns") or []
            entry["appositive_patterns"] = [re.compile(pat, re.IGNORECASE) for pat in appos_patterns if isinstance(pat, str)]
            negatives = payload.get("negative_verbs") or []
            entry["negative_verbs"] = [re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE) for word in negatives if isinstance(word, str)]
            self.rules[name] = entry

    def filter_notes(self, notes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        kept: List[Dict[str, Any]] = []
        dropped = 0
        for note in notes:
            if self._accept(note):
                kept.append(note)
            else:
                dropped += 1
        return kept, dropped

    def _accept(self, note: Dict[str, Any]) -> bool:
        attr_meta = (note.get("meta") or {}).get("attribute") or {}
        attr_name = (attr_meta.get("name") or note.get("pred") or "").strip().lower()
        if not attr_name:
            return True
        rules = self.rules.get(attr_name)
        if not rules:
            return True

        evidence = (note.get("evidence") or "").strip()
        if evidence and not self._evidence_allowed(evidence, rules):
            record_attribute_guard(attr_name, "definition_miss")
            return False
        if evidence and self._has_negative_verb(evidence, rules):
            record_attribute_guard(attr_name, "negative_verb")
            return False

        # Normalize value & enforce lexicon
        obj = (note.get("obj") or "").strip()
        normalized_value, _ = normalize_slot_value(attr_name, obj)
        normalized_lower = (normalized_value or "").strip().lower()
        blacklist = rules.get("blacklist") or []
        if normalized_lower:
            for bad in blacklist:
                if bad and bad in normalized_lower:
                    record_attribute_guard(attr_name, "blacklist")
                    return False
        lexicon = rules.get("lexicon") or set()
        if lexicon and normalized_lower and normalized_lower not in lexicon:
            record_attribute_guard(attr_name, "out_of_lexicon")
            return False

        if normalized_value:
            note["obj"] = normalized_value
            if isinstance(attr_meta, dict):
                values = attr_meta.get("values")
                if isinstance(values, list):
                    for item in values:
                        if isinstance(item, dict):
                            item["normalized"] = normalized_value
        record_attribute_guard(attr_name, "kept")
        return True

    @staticmethod
    def _evidence_allowed(evidence: str, rules: Dict[str, Any]) -> bool:
        patterns: List[re.Pattern] = rules.get("definition_patterns") or []
        appos: List[re.Pattern] = rules.get("appositive_patterns") or []
        if patterns and any(pat.search(evidence) for pat in patterns):
            return True
        if appos and any(pat.search(evidence) for pat in appos):
            return True
        # fallback: check simple "is a/an" pattern
        return bool(re.search(r"\b(is|was|are|were)\b\s+(?:an?|the)\s+[A-Za-z]", evidence))

    @staticmethod
    def _has_negative_verb(evidence: str, rules: Dict[str, Any]) -> bool:
        verbs: List[re.Pattern] = rules.get("negative_verbs") or []
        return any(pat.search(evidence) for pat in verbs)
