"""JSON schema definition for atomic note outputs."""

NOTE_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "required": ["content", "keywords", "entities"],
        "properties": {
            "content": {"type": "string", "minLength": 8},
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": False,
            },
            "entities": {
                "type": "object",
                "required": ["PERSON", "ORG", "LOC", "DATE", "MISC"],
                "properties": {
                    "PERSON": {"type": "array", "items": {"type": "string"}},
                    "ORG":    {"type": "array", "items": {"type": "string"}},
                    "LOC":    {"type": "array", "items": {"type": "string"}},
                    "DATE":   {"type": "array", "items": {"type": "string"}},
                    "MISC":   {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    },
}
