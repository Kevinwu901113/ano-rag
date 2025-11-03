NOTE_JSON_SCHEMA = {
    "type": "array",
    "minItems": 0,
    "items": {
        "type": "object",
        "required": [
            "subj",
            "pred",
            "obj",
            "subj_type",
            "obj_type",
            "evidence",
            "meta",
        ],
        "properties": {
            "subj": {"type": "string", "minLength": 1},
            "pred": {"type": "string", "minLength": 1},
            "obj": {"type": "string", "minLength": 1},
            "subj_type": {
                "type": "string",
                "enum": [
                    "PERSON",
                    "WORK",
                    "ORG",
                    "PLACE",
                    "EVENT",
                    "CONCEPT",
                    "TIME",
                ],
            },
            "obj_type": {
                "type": "string",
                "enum": [
                    "PERSON",
                    "WORK",
                    "ORG",
                    "PLACE",
                    "EVENT",
                    "CONCEPT",
                    "TIME",
                ],
            },
            "evidence": {"type": "string", "minLength": 4},
            "meta": {
                "type": "object",
                "required": ["source", "confidence"],
                "properties": {
                    "source": {"type": "string"},
                    "domain": {"type": ["string", "null"]},
                    "year": {"type": ["string", "number", "null"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "aliases": {
                        "type": ["object", "null"],
                        "properties": {
                            "subj": {"type": "array", "items": {"type": "string"}},
                            "obj": {"type": "array", "items": {"type": "string"}},
                        },
                        "additionalProperties": False,
                    },
                    "entity_links": {
                        "type": ["object", "null"],
                        "properties": {
                            "subj_wikidata": {"type": ["string", "null"]},
                            "obj_wikidata": {"type": ["string", "null"]},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": False,
    },
}

ALLOWED_TYPES = ["PERSON", "WORK", "ORG", "PLACE", "EVENT", "CONCEPT", "TIME"]

ALLOWED_PREDICATES = [
    "performed_by",
    "authored_by",
    "spouse",
    "parent",
    "born_in",
    "located_in",
    "member_of",
    "acted_in",
    "produced_by",
    "released_in",
    "label",
    "founded_by",
    "headquartered_in",
    "winner_of",
    "part_of",
]

PRED_SYNONYM_SETS = {
    "performed_by": {"recorded_by", "artist", "performed_by"},
    "authored_by": {"written_by", "authored_by"},
    "spouse": {"married_to", "partner", "spouse"},
    "parent": {"father", "mother", "parent"},
    "born_in": {"place_of_birth", "born_in"},
    "acted_in": {"starring", "cast_in", "acted_in"},
    "located_in": {"located_in"},
    "produced_by": {"produced_by"},
    "released_in": {"released_in"},
    "label": {"label"},
    "member_of": {"member_of"},
    "founded_by": {"founded_by"},
    "headquartered_in": {"headquartered_in"},
    "winner_of": {"winner_of"},
    "part_of": {"part_of"},
}
