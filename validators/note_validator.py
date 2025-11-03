from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from jsonschema import Draft7Validator, ValidationError

from schema.note_schema_v1 import (
    ALLOWED_TYPES,
    NOTE_JSON_SCHEMA,
    PRED_SYNONYM_SETS,
)


def _normalize_pred(pred: str) -> Tuple[str, float]:
    p = pred.strip().lower()
    for canon, syns in PRED_SYNONYM_SETS.items():
        if p in syns:
            penalty = 0.05 if p != canon else 0.0
            return canon, 1.0 - penalty
    return p, 0.9


def _check_type_pattern(subj_t: str, pred: str, obj_t: str) -> float:
    patterns = {
        "performed_by": ("WORK", "PERSON"),
        "authored_by": ("WORK", "PERSON"),
        "spouse": ("PERSON", "PERSON"),
        "parent": ("PERSON", "PERSON"),
        "born_in": ("PERSON", "PLACE"),
        "located_in": ("ORG", "PLACE"),
        "acted_in": ("PERSON", "WORK"),
    }
    if pred in patterns and patterns[pred] != (subj_t, obj_t):
        return 0.7
    return 1.0


def validate_and_normalize(
    raw_text: str, doc_id: str, chunk_id: str
) -> Tuple[bool, List[Dict], Dict]:
    metrics: Dict[str, Any] = {}
    try:
        notes = json.loads(raw_text)
    except Exception as exc:
        return False, [], {"violations": {"json_parse": str(exc)}}

    try:
        Draft7Validator(NOTE_JSON_SCHEMA).validate(notes)
    except ValidationError as exc:
        return False, [], {"violations": {"json_schema": str(exc)}}

    out = []
    for i, n in enumerate(notes):
        subj_type = n["subj_type"]
        obj_type = n["obj_type"]
        pred_norm, pred_w = _normalize_pred(n["pred"])
        type_w = _check_type_pattern(subj_type, pred_norm, obj_type)
        conf = float(n.get("meta", {}).get("confidence", 0.8)) * pred_w * type_w

        if subj_type not in ALLOWED_TYPES or obj_type not in ALLOWED_TYPES:
            conf *= 0.8

        ev = n["evidence"]
        if not any(x for x in [n["subj"], n["obj"]] if x.split()[0] in ev):
            conf *= 0.9

        normalized = {
            "note_id": f"{doc_id}#{chunk_id}#{i}",
            "subj": n["subj"].strip(),
            "pred": pred_norm,
            "obj": n["obj"].strip(),
            "subj_type": n["subj_type"],
            "obj_type": n["obj_type"],
            "evidence": ev.strip(),
            "meta": {
                **n.get("meta", {}),
                "source": f"{doc_id}#{chunk_id}",
                "final_conf": round(conf, 4),
            },
        }
        out.append(normalized)

    metrics["count"] = len(out)
    return True, out, metrics
