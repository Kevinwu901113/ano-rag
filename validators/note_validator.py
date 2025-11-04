from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from jsonschema import Draft7Validator, ValidationError

from schema.note_schema_v1 import NOTE_JSON_SCHEMA, PRED_SYNONYM_SETS, PRED2ATTR
from schema.vocabulary import normalize_entity_name, normalize_slot_value


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


def _normalize_pred(raw_pred: str) -> Tuple[str, float]:
    value = (raw_pred or "").strip().lower()
    if not value:
        return "", 0.5
    # 首先用同义集合归一
    for canon, synonyms in PRED_SYNONYM_SETS.items():
        if value == canon:
            return canon, 1.0
        if value in synonyms:
            return canon, 0.95
    # 再用属性映射表做强归一（例如 profession/job/works as → occupation）
    mapped = PRED2ATTR.get(value)
    if mapped:
        return mapped, 0.95
    return value, 0.9


def _type_pattern_ok(subj_type: str, pred: str, obj_type: str) -> float:
    allowed = {
        "performed_by": {("WORK", "PERSON")},
        "authored_by": {("WORK", "PERSON")},
        "spouse": {("PERSON", "PERSON")},
        "parent": {("PERSON", "PERSON")},
        "born_in": {("PERSON", "PLACE")},
        "acted_in": {("PERSON", "WORK")},
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


def validate_and_normalize(raw_text: str, doc_id: str, chunk_id: str):
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:  # noqa: BLE001
        return False, [], {"violations": {"json_parse": str(exc)}}

    if not isinstance(parsed, list):
        return False, [], {
            "violations": {
                "json_type": f"expect array, got {type(parsed).__name__}",
            }
        }

    patched: List[Dict[str, Any]] = []
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
        patched_obj["meta"] = meta
        patched.append(patched_obj)

    try:
        Draft7Validator(NOTE_JSON_SCHEMA).validate(patched)
    except ValidationError as exc:
        return False, [], {"violations": {"json_schema": str(exc)}}

    notes: List[Dict[str, Any]] = []
    for idx, item in enumerate(patched):
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

        attr = item["meta"].get("attribute") or {}
        raw_attr_name = attr.get("name") or item["pred"]
        pred, pred_weight = _normalize_pred(raw_attr_name)
        # 如果是title但非honorific/position_title则并入occupation
        if raw_attr_name and raw_attr_name.strip().lower() in {"title", "titles"}:
            role = (attr.get("role") or "").strip().lower()
            if role not in {"honorific", "position_title"}:
                pred = "occupation"
                pred_weight = min(pred_weight, 0.95)

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

        _update_profile_with_attr(subject_profile, pred, normalized_values)

        type_weight = _type_pattern_ok(item["subj_type"], pred, item["obj_type"])
        base_conf = float(item["meta"]["confidence"])
        final_conf = round(base_conf * pred_weight * type_weight, 4)

        quality = _compute_quality(item["evidence"], subject_profile, normalized_values, alias_hits)

        note_id = f"{doc_id}#{chunk_id}#{idx}"
        notes.append(
            {
                "note_id": note_id,
                "subj": normalize_entity_name(item["subj"])[0],
                "pred": pred,
                "obj": normalized_values[0].get("normalized") or item["obj"].strip(),
                "subj_type": item["subj_type"],
                "obj_type": item["obj_type"],
                "evidence": item["evidence"].strip(),
                "meta": {
                    **item["meta"],
                    "source": item["meta"]["source"],
                    "confidence": base_conf,
                    "final_conf": final_conf,
                    "attribute": {
                        # 保证 occupation 的规范输出
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
        )

    return True, notes, {"count": len(notes)}
