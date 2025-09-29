from __future__ import annotations

from typing import Any, Dict, Iterable


def format_diagnostics(diagnostics: Dict[str, Any]) -> str:
    lines = ["Configuration diagnostics:"]
    deprecated = diagnostics.get("deprecated_paths") or []
    unknown = diagnostics.get("unknown_keys") or []
    missing = diagnostics.get("missing_required") or []

    def _format_block(title: str, values: Iterable[str]) -> None:
        values = list(values)
        if not values:
            lines.append(f"  - {title}: none")
        else:
            lines.append(f"  - {title} ({len(values)}):")
            lines.extend([f"      * {value}" for value in values])

    _format_block("deprecated", deprecated)
    _format_block("unknown", unknown)
    _format_block("missing", missing)
    return "\n".join(lines)
