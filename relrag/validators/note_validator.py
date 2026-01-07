from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
import re
from pathlib import Path

from jsonschema import Draft7Validator, ValidationError

from relrag.schema.note_schema_v1 import NOTE_JSON_SCHEMA, PRED_SYNONYM_SETS, PRED2ATTR
from relrag.schema.note_schema_v1 import ALLOWED_PREDICATES
from relrag.schema.vocabulary import normalize_entity_name, normalize_slot_value
from relrag.utils import TextUtils


PROFILE_SLOT_MAP = {
    "occupation": "occupations",
    "title": "titles",
    "category": "categories",
    "nationality": "nationality",
    "born_on": "birth",
    "died_on": "death",
    "alias_of": "aliases",
    "same_as": "same_as",
    "type": "type",
}

DEFINITION_TRIGGERS = (" is a ", " is an ", " was a ", " is the ", " is an honorary ")


def _clamp(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


_ALIAS_PATTERNS_PATH = Path(__file__).resolve().parents[1] / "schema" / "predicate_alias.json"
_COMPILED_ALIAS_PATTERNS: List[Tuple[re.Pattern, str]] = []
if _ALIAS_PATTERNS_PATH.exists():
    try:
        with open(_ALIAS_PATTERNS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                for pat, target in data.items():
                    try:
                        compiled = re.compile(pat, re.IGNORECASE)
                        _COMPILED_ALIAS_PATTERNS.append((compiled, str(target).strip().lower()))
                    except re.error:
                        pass
    except Exception:
        pass


def _looks_like_occupation(noun_phrase: str) -> bool:
    np = (noun_phrase or "").strip().lower()
    if not np:
        return False
    # Use vocabulary slot to decide occupation-likeness
    from relrag.schema.vocabulary import load_vocab, load_alias_overrides
    occ_map = set(load_vocab().get("occupation", {}).keys())
    occ_over = set(load_alias_overrides().get("occupation", {}).keys())
    # Simplify phrase tokens
    simplified = re.sub(r"[\W_]+", " ", np)
    simplified = simplified.replace("-", " ")
    simplified = simplified.strip()
    # Direct match
    if simplified in occ_map or simplified in occ_over:
        return True
    # Gender/number variants
    variants = {simplified}
    if simplified.endswith("s"):
        variants.add(simplified[:-1])
    if simplified.endswith("es"):
        variants.add(simplified[:-2])
    # actress → actor mapping and common families
    families = {
        "actress": "actor",
        "businessman": "businessperson",
        "businesswoman": "businessperson",
        "executive": "businessperson",
        "attorney": "lawyer",
        "barrister": "lawyer",
        "solicitor": "lawyer",
    }
    mapped = families.get(simplified)
    if mapped and (mapped in occ_map or mapped in occ_over):
        return True
    return any(v in occ_map or v in occ_over for v in variants)


def _normalize_pred(raw_pred: str) -> Tuple[str, float]:
    value = (raw_pred or "").strip().lower()
    if not value:
        return "", 0.5
    alias_value = value.replace("_", " ")
    variants = {value, alias_value, alias_value.replace(" ", "_")}
    # 先用正则别名匹配
    for pat, target in _COMPILED_ALIAS_PATTERNS:
        if pat.search(alias_value):
            return target, 0.95
    # 再用同义集合归一
    for canon, synonyms in PRED_SYNONYM_SETS.items():
        if canon in variants:
            return canon, 1.0
        if variants & synonyms:
            return canon, 0.95
    # 属性映射表强归一（例如 profession/job/works as → occupation）
    for variant in variants:
        mapped = PRED2ATTR.get(variant)
        if mapped:
            return mapped, 0.95
    # 兜底：X is/was/became a/an <NOUN> 且 <NOUN> 像职业
    m = re.search(r"\b(is|was|became|becomes)\b .*?\b([A-Za-z- ]+)\b", alias_value)
    if m and _looks_like_occupation(m.group(2)):
        return "occupation", 0.9
    return value, 0.9


def _type_pattern_ok(subj_type: str, pred: str, obj_type: str) -> float:
    allowed = {
        "performed_by": {("WORK", "PERSON")},
        "authored_by": {("WORK", "PERSON")},
        "directed_by": {("WORK", "PERSON")},
        "spouse": {("PERSON", "PERSON")},
        "parent": {("PERSON", "PERSON")},
        "born_in": {("PERSON", "PLACE")},
        "died_in": {("PERSON", "PLACE")},
        "acted_in": {("PERSON", "WORK")},
        "headquartered_in": {("ORG", "PLACE")},
        "located_in": {("ORG", "PLACE"), ("PLACE", "PLACE")},
        "occupation": {("PERSON", "CONCEPT"), ("PERSON", "ORG")},
        "title": {("PERSON", "CONCEPT"), ("PERSON", "WORK"), ("ORG", "CONCEPT")},
        "category": {
            ("PERSON", "CONCEPT"),
            ("WORK", "CONCEPT"),
            ("ORG", "CONCEPT"),
            ("PLACE", "CONCEPT"),
        },
        "nationality": {("PERSON", "PLACE")},
        "born_on": {("PERSON", "TIME")},
        "died_on": {("PERSON", "TIME")},
        "released_in": {("WORK", "TIME")},
        "produced_by": {("WORK", "PERSON"), ("WORK", "ORG")},
        "label": {("WORK", "ORG")},
        "member_of": {("PERSON", "ORG"), ("ORG", "ORG")},
        "award_received": {("PERSON", "CONCEPT"), ("WORK", "CONCEPT")},
        "works_for": {("PERSON", "ORG")},
        "educated_at": {("PERSON", "ORG")},
        "position_held": {("PERSON", "CONCEPT"), ("PERSON", "ORG")},
        "founded": {("PERSON", "ORG")},
        "parent_of": {("PERSON", "PERSON")},
        "child_of": {("PERSON", "PERSON")},
        "residence": {("PERSON", "PLACE")},
        "alias_of": {("PERSON", "PERSON"), ("ORG", "ORG"), ("WORK", "WORK")},
        "same_as": {("PERSON", "PERSON"), ("ORG", "ORG"), ("WORK", "WORK")},
        "type": {
            ("PERSON", "CONCEPT"),
            ("WORK", "CONCEPT"),
            ("ORG", "CONCEPT"),
            ("PLACE", "CONCEPT"),
            ("EVENT", "CONCEPT"),
        },
    }
    allowed_pairs = allowed.get(pred)
    if not allowed_pairs:
        return 1.0
    return 1.0 if (subj_type, obj_type) in allowed_pairs else 0.75


def _ensure_profile(
    raw_profile: Any, *, default_type: str, entity_name: str, allow_null: bool = False
) -> Dict[str, Any] | None:
    if raw_profile is None and allow_null:
        return None

    profile = dict(raw_profile) if isinstance(raw_profile, dict) else {}
    normalized_name, alias_hit = normalize_entity_name(entity_name)
    alias_candidates: List[str] = []
    existing_aliases = profile.get("aliases")
    if isinstance(existing_aliases, list):
        alias_candidates.extend(
            [alias.strip() for alias in existing_aliases if isinstance(alias, str) and alias.strip()]
        )

    if alias_hit and entity_name.strip():
        alias_candidates.append(entity_name.strip())
    if normalized_name and normalized_name.strip() and normalized_name != entity_name:
        alias_candidates.insert(0, normalized_name.strip())

    deduped_aliases: List[str] = []
    for alias in alias_candidates:
        if alias and alias not in deduped_aliases:
            deduped_aliases.append(alias)

    profile["type"] = profile.get("type") or default_type
    profile["aliases"] = deduped_aliases
    for key in ("nationality", "occupations", "titles", "categories", "same_as"):
        value = profile.get(key)
        if value is not None and not isinstance(value, list):
            profile[key] = [value] if value else []
        elif value is None:
            profile[key] = []
    for key in ("birth", "death", "description"):
        value = profile.get(key)
        profile[key] = value if isinstance(value, str) and value.strip() else None

    return profile


def _update_profile_with_attr(profile: Dict[str, Any], predicate: str, values: List[Dict[str, Any]]) -> None:
    slot = PROFILE_SLOT_MAP.get(predicate)
    if not slot or slot not in profile:
        return
    if slot in {"birth", "death", "type"}:
        profile[slot] = values[0]["normalized"] or values[0]["value"]
        return
    existing = profile.get(slot) or []
    for val in values:
        norm = val.get("normalized") or val.get("value")
        if norm and norm not in existing:
            existing.append(norm)
    profile[slot] = existing


def _compute_quality(evidence: str, profile: Dict[str, Any], values: List[Dict[str, Any]], alias_hits: int) -> Dict[str, Any]:
    score = 0.6
    issues: List[str] = []
    trimmed_evidence = evidence.strip()

    if len(trimmed_evidence) >= 60:
        score += 0.1
    else:
        issues.append("evidence_short")

    if alias_hits:
        score += 0.05

    if all(v.get("normalized") for v in values):
        score += 0.1
    else:
        issues.append("value_not_normalized")

    if profile.get("aliases"):
        score += 0.05

    has_definition = any(trigger in trimmed_evidence.lower() for trigger in DEFINITION_TRIGGERS)
    if has_definition:
        score += 0.05

    score = max(0.0, min(1.0, round(score, 4)))
    return {
        "score": score,
        "issues": issues or None,
        "has_definition": has_definition or None,
    }


def _chunk_rank(chunk_id: str) -> Optional[int]:
    if not chunk_id:
        return None
    match = re.search(r"(\d+)", chunk_id)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def validate_and_normalize(raw_text: str, doc_id: str, chunk_id: str):
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:  # noqa: BLE001
        return {
            "valid_notes": [],
            "pronoun_notes": [],
            "errors": [{"type": "json_parse", "message": str(exc)}],
            "stats": {"json_parse_failures": 1},
        }

    if not isinstance(parsed, list):
        return {
            "valid_notes": [],
            "pronoun_notes": [],
            "errors": [{"type": "json_type", "message": f"expect array, got {type(parsed).__name__}"}],
            "stats": {"json_type_failures": 1},
        }

    patched: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        patched_obj = dict(obj)
        raw_meta = patched_obj.get("meta")
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}

        source = meta.get("source")
        if not isinstance(source, str) or not source.strip():
            meta["source"] = f"{doc_id}#{chunk_id}"

        meta["confidence"] = _clamp(meta.get("confidence"), default=0.8)
        section_rank = _chunk_rank(chunk_id)
        if section_rank is not None:
            meta["section_rank"] = section_rank
            meta["anchor"] = bool(section_rank == 0)
        else:
            meta.setdefault("anchor", False)

        subj_type = patched_obj.get("subj_type") or "CONCEPT"
        subj_profile = meta.get("subject_profile")
        if not isinstance(subj_profile, dict):
            meta["subject_profile"] = {
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
        else:
            subj_profile.setdefault("type", subj_type)
            subj_profile.setdefault("aliases", [])

        attr = meta.get("attribute") if isinstance(meta.get("attribute"), dict) else {}
        attr_name = attr.get("name") or patched_obj.get("pred") or ""
        attr_values = attr.get("values") if isinstance(attr.get("values"), list) else []
        if not attr_values:
            attr_values = [
                {
                    "value": patched_obj.get("obj", ""),
                    "normalized": None,
                    "confidence": meta["confidence"],
                    "source": meta["source"],
                    "evidence": patched_obj.get("evidence", ""),
                }
            ]
        meta["attribute"] = {
            "name": attr_name,
            "values": attr_values,
            "role": attr.get("role"),
            "target_type": attr.get("target_type"),
        }
        raw_pred = str(patched_obj.get("pred") or "").strip().lower()
        if not raw_pred:
            fixed_pred = str(attr_name).strip().lower()
            evidence_text = str(patched_obj.get("evidence") or "")
            if not fixed_pred and evidence_text:
                for pat, target in _COMPILED_ALIAS_PATTERNS:
                    if pat.search(evidence_text):
                        fixed_pred = target
                        break
            if not fixed_pred and evidence_text:
                m = re.search(r"\b(is|was|became|becomes)\b .*?\b([A-Za-z- ]+)\b", evidence_text.lower())
                if m and _looks_like_occupation(m.group(2)):
                    fixed_pred = "occupation"
            if not fixed_pred:
                meta.setdefault("violations", {})
                meta["violations"]["pred_missing"] = True
                fixed_pred = "__missing__"
            patched_obj["pred"] = fixed_pred
            if not attr_name:
                meta["attribute"]["name"] = fixed_pred

        patched_obj["meta"] = meta
        patched.append(patched_obj)

    per_item_validator = Draft7Validator(NOTE_JSON_SCHEMA["items"])

    def _is_pronoun_violation(error: ValidationError) -> bool:
        schema = error.schema or {}
        not_schema = schema.get("not") if isinstance(schema, dict) else {}
        pattern = not_schema.get("pattern") if isinstance(not_schema, dict) else None
        if pattern and "he|she|they" in pattern.lower():
            return True
        msg = str(error.message or "")
        return "should not be valid under" in msg and "he|she|they" in msg.lower()

    valid_notes: List[Dict[str, Any]] = []
    pronoun_notes: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    unmatched_counter: Dict[str, int] = {}
    skipped_count = 0

    def _normalize_one(idx: int, item: Dict[str, Any]) -> dict | None:
        nonlocal skipped_count
        attr = item["meta"].get("attribute") or {}
        raw_attr_name = (attr.get("name") or item.get("pred") or "").strip()
        evidence_text = (item.get("evidence") or "").strip()
        pred, pred_weight = _normalize_pred(raw_attr_name)
        if not pred or pred.strip() == "":
            for pat, target in _COMPILED_ALIAS_PATTERNS:
                if pat.search(evidence_text):
                    pred = target
                    pred_weight = max(pred_weight, 0.9)
                    break
            if not pred:
                m = re.search(r"\b(is|was|became|becomes)\b .*?\b([A-Za-z- ]+)\b", evidence_text.lower())
                if m and _looks_like_occupation(m.group(2)):
                    pred = "occupation"
                    pred_weight = max(pred_weight, 0.9)
            # Definitional noun phrase without explicit verb, e.g., "American cartoonist and illustrator"
            if not pred and evidence_text:
                try:
                    from relrag.schema.vocabulary import load_vocab, load_alias_overrides
                    occ_map = set(load_vocab().get("occupation", {}).keys())
                    occ_over = set(load_alias_overrides().get("occupation", {}).keys())
                    tokens = re.sub(r"[\W_]+", " ", evidence_text.lower()).split()
                    if any(tok in occ_map or tok in occ_over for tok in tokens):
                        pred = "occupation"
                        pred_weight = max(pred_weight, 0.85)
                except Exception:
                    pass
        if raw_attr_name and raw_attr_name.strip().lower() in {"title", "titles"}:
            role = (attr.get("role") or "").strip().lower()
            if role not in {"honorific", "position_title"}:
                pred = "occupation"
                pred_weight = min(pred_weight, 0.95)

        if pred not in ALLOWED_PREDICATES:
            for pat, target in _COMPILED_ALIAS_PATTERNS:
                if pat.search(evidence_text):
                    pred = target
                    pred_weight = max(pred_weight, 0.9)
                    break
        if pred not in ALLOWED_PREDICATES:
            key = raw_attr_name or "<blank>"
            unmatched_counter[key] = unmatched_counter.get(key, 0) + 1
            skipped_count += 1
            return None

        raw_values = attr.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raw_values = [{"value": item["obj"], "evidence": item["evidence"]}]

        normalized_values: List[Dict[str, Any]] = []
        alias_hits = 0
        for value in raw_values:
            if isinstance(value, str):
                raw_val = value
                evidence = item["evidence"]
                val_conf = item["meta"]["confidence"]
                value_source = item["meta"]["source"]
            else:
                raw_val = value.get("value", "")
                evidence = value.get("evidence") or item["evidence"]
                val_conf = _clamp(value.get("confidence"), item["meta"]["confidence"])
                value_source = value.get("source") or item["meta"]["source"]

            normalized_val, alias_hit = normalize_slot_value(pred, raw_val)
            alias_hits += int(alias_hit)
            normalized_values.append(
                {
                    "value": raw_val.strip(),
                    "normalized": normalized_val.strip() if isinstance(normalized_val, str) else normalized_val,
                    "confidence": val_conf,
                    "source": value_source,
                    "evidence": evidence.strip(),
                    "qualifiers": value.get("qualifiers") if isinstance(value, dict) else None,
                }
            )

        subject_profile = _ensure_profile(
            item["meta"].get("subject_profile"),
            default_type=item["subj_type"],
            entity_name=item["subj"],
        )
        object_profile = _ensure_profile(
            item["meta"].get("object_profile"),
            default_type=item["obj_type"],
            entity_name=item["obj"],
            allow_null=True,
        )
        _update_profile_with_attr(subject_profile, pred, normalized_values)

        type_weight = _type_pattern_ok(item["subj_type"], pred, item["obj_type"])
        base_conf = float(item["meta"]["confidence"])
        final_conf = round(base_conf * pred_weight * type_weight, 4)

        meta_ev_canon = (item.get("meta", {}) or {}).get("evidence_canonical")
        canonical_evidence = meta_ev_canon.strip() if isinstance(meta_ev_canon, str) and meta_ev_canon.strip() else evidence_text
        quality = _compute_quality(evidence_text, subject_profile, normalized_values, alias_hits)

        note_id = f"{doc_id}#{chunk_id}#{idx}"
        return {
            "note_id": note_id,
            "subj": normalize_entity_name(item["subj"])[0],
            "pred": pred,
            "obj": normalized_values[0].get("normalized") or item["obj"].strip(),
            "subj_type": item["subj_type"],
            "obj_type": item["obj_type"],
            "evidence": evidence_text,
            "meta": {
                **item["meta"],
                "source": item["meta"]["source"],
                "confidence": base_conf,
                "final_conf": final_conf,
                "evidence_canonical": canonical_evidence,
                "attribute": {
                    "name": pred,
                    "values": normalized_values,
                    "role": attr.get("role"),
                    "target_type": attr.get("target_type"),
                },
                "quality": quality,
                "quality_score": quality.get("score"),
                "subject_profile": subject_profile,
                "object_profile": object_profile,
            },
        }

    for idx, item in enumerate(patched):
        subj = (item.get("subj") or "").strip()
        obj = (item.get("obj") or "").strip()
        meta = item.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        if TextUtils.is_pronoun(subj):
            meta["pronoun_subj"] = True
        if TextUtils.is_pronoun(obj):
            meta["pronoun_obj"] = True
        item["meta"] = meta

        normalized = _normalize_one(idx, item)
        if normalized is None:
            continue

        try:
            per_item_validator.validate(item)
            valid_notes.append(normalized)
        except ValidationError as exc:
            if _is_pronoun_violation(exc):
                pronoun_notes.append(
                    {
                        "note": normalized,
                        "field": str(exc.path[-1]) if exc.path else None,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                    }
                )
                stats["pronoun_schema_hits"] = stats.get("pronoun_schema_hits", 0) + 1
            else:
                errors.append({"index": idx, "message": str(exc), "doc_id": doc_id, "chunk_id": chunk_id})

    if skipped_count:
        stats["unmatched_predicates"] = unmatched_counter
        stats["skipped_count"] = skipped_count

    return {
        "valid_notes": valid_notes,
        "pronoun_notes": pronoun_notes,
        "errors": errors,
        "stats": stats,
    }
