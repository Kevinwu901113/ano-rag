PROFILE_SCHEMA = {
    "type": "object",
    "required": ["type", "aliases"],
    "properties": {
        "type": {
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
        "aliases": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "default": [],
        },
        "nationality": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "birth": {"type": ["string", "null"], "minLength": 1},
        "death": {"type": ["string", "null"], "minLength": 1},
        "occupations": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "titles": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "categories": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "same_as": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "description": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

ATTRIBUTE_VALUE_SCHEMA = {
    "type": "object",
    "required": ["value"],
    "properties": {
        "value": {"type": "string", "minLength": 1},
        "normalized": {"type": ["string", "null"], "minLength": 1},
        "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "source": {"type": ["string", "null"], "minLength": 1},
        "evidence": {"type": ["string", "null"], "minLength": 4},
        "qualifiers": {"type": ["object", "null"]},
        "notes": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

ATTRIBUTE_SCHEMA = {
    "type": "object",
    "required": ["name", "values"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "values": {
            "type": "array",
            "minItems": 1,
            "items": ATTRIBUTE_VALUE_SCHEMA,
        },
        "role": {"type": ["string", "null"]},
        "target_type": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "issues": {"type": ["array", "null"], "items": {"type": "string"}},
        "has_definition": {"type": ["boolean", "null"]},
    },
    "additionalProperties": False,
}

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
            "note_id": {"type": ["string", "null"], "minLength": 1},
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
                "required": ["source", "confidence", "subject_profile"],
                "properties": {
                    "source": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "subject_profile": PROFILE_SCHEMA,
                    "object_profile": {"anyOf": [PROFILE_SCHEMA, {"type": "null"}]},
                    "attribute": ATTRIBUTE_SCHEMA,
                    "quality": QUALITY_SCHEMA,
                    "quality_score": {
                        "type": ["number", "null"],
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "domain": {"type": ["string", "null"]},
                    "year": {"type": ["string", "number", "null"]},
                    "entity_links": {
                        "type": ["object", "null"],
                        "properties": {
                            "subj_wikidata": {"type": ["string", "null"]},
                            "obj_wikidata": {"type": ["string", "null"]},
                        },
                        "additionalProperties": False,
                    },
                    "render_hint": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
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
    "occupation",
    "title",
    "category",
    "nationality",
    "born_on",
    "died_on",
    "alias_of",
    "same_as",
    "type",
]

# 属性映射表：谓词到规范化属性名，用于索引层归一
PRED2ATTR = {
    "occupation": "occupation",
    "profession": "occupation",
    "professions": "occupation",
    "job": "occupation",
    "jobs": "occupation",
    "works as": "occupation",
    "works_as": "occupation",
    "career": "occupation",
    "careers": "occupation",
}

PRED_SYNONYM_SETS = {
    "performed_by": {"recorded_by", "artist", "performed_by"},
    "authored_by": {"written_by", "authored_by"},
    "spouse": {"married_to", "partner", "spouse", "spouse_of"},
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
    # 强制归一：职业相关同义词全部归并到 "occupation"
    "occupation": {
        "occupation",
        "occupations",
        "profession",
        "professions",
        "job",
        "jobs",
        "works as",
        "works_as",
        "career",
        "careers",
        "title (when occupational)",
    },
    "title": {"title", "position", "role", "titles"},
    "category": {"category", "categories", "classification"},
    "nationality": {"nationality", "citizenship", "country_of_citizenship"},
    "born_on": {"born_on", "birth_date", "date_of_birth", "born"},
    "died_on": {"died_on", "death_date", "date_of_death", "died"},
    "alias_of": {"alias_of", "aka", "also_known_as"},
    "same_as": {"same_as", "identical_to"},
    "type": {"type", "entity_type", "category_type"},
}
