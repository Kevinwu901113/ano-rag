"""Schema definitions for structured notes."""

from .note_schema_v1 import (
    ALLOWED_PREDICATES,
    ALLOWED_TYPES,
    NOTE_JSON_SCHEMA,
    PRED_SYNONYM_SETS,
)

__all__ = [
    "NOTE_JSON_SCHEMA",
    "ALLOWED_TYPES",
    "ALLOWED_PREDICATES",
    "PRED_SYNONYM_SETS",
]
