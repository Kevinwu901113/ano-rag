from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from jsonschema import Draft7Validator, ValidationError

from schema.note_schema_v1 import NOTE_JSON_SCHEMA, PRED_SYNONYM_SETS


def _normalize_pred(pred: str) -> Tuple[str, float]:
    value = pred.strip().lower()
    for canon, synonyms in PRED_SYNONYM_SETS.items():
        if value in synonyms:
            return canon, (1.0 if value == canon else 0.95)
    return value, 0.9


def _type_pattern_ok(subj_type: str, pred: str, obj_type: str) -> float:
    allowed = {
        "performed_by": ("WORK", "PERSON"),
        "authored_by": ("WORK", "PERSON"),
        "spouse": ("PERSON", "PERSON"),
        "parent": ("PERSON", "PERSON"),
        "born_in": ("PERSON", "PLACE"),
        "acted_in": ("PERSON", "WORK"),
    }
    if pred in allowed and allowed[pred] != (subj_type, obj_type):
        return 0.7
    return 1.0


def validate_and_normalize(raw_text: str, doc_id: str, chunk_id: str):
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:  # noqa: BLE001
        return False, [], {"violations": {"json_parse": str(exc)}}

    try:
        Draft7Validator(NOTE_JSON_SCHEMA).validate(parsed)
    except ValidationError as exc:
        return False, [], {"violations": {"json_schema": str(exc)}}

    notes: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed):
        pred, pred_weight = _normalize_pred(item["pred"])
        type_weight = _type_pattern_ok(item["subj_type"], pred, item["obj_type"])
        base_conf = float(item.get("meta", {}).get("confidence", 0.8))
        final_conf = round(base_conf * pred_weight * type_weight, 4)

        notes.append(
            {
                "note_id": f"{doc_id}#{chunk_id}#{idx}",
                "subj": item["subj"].strip(),
                "pred": pred,
                "obj": item["obj"].strip(),
                "subj_type": item["subj_type"],
                "obj_type": item["obj_type"],
                "evidence": item["evidence"].strip(),
                "meta": {
                    **item.get("meta", {}),
                    "source": f"{doc_id}#{chunk_id}",
                    "final_conf": final_conf,
                },
            }
        )

    return True, notes, {"count": len(notes)}
