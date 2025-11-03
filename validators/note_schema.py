"""Compatibility shim for legacy imports.

The canonical schema now lives under :mod:`schema.note_schema_v1`.
"""

from schema.note_schema_v1 import (  # noqa: F401,F403
    ALLOWED_PREDICATES,
    ALLOWED_TYPES,
    NOTE_JSON_SCHEMA,
    PRED_SYNONYM_SETS,
)

NOTE_SCHEMA = NOTE_JSON_SCHEMA

__all__ = [
    "NOTE_JSON_SCHEMA",
    "ALLOWED_TYPES",
    "ALLOWED_PREDICATES",
    "PRED_SYNONYM_SETS",
    "NOTE_SCHEMA",
]
