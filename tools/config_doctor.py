from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from core.config.loader import load_config
from core.config.printer import format_diagnostics
from core.config.schema import RootConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and validate configuration state.")
    parser.add_argument("command", choices=["validate", "print-schema", "doctor"], nargs="?", default="validate")
    parser.add_argument("--profile", default=None, help="Profile name used when loading configs.")
    parser.add_argument("--schema-path", default="configs/schema.json", help="Where to write the exported JSON schema.")
    parser.add_argument("--show-config", action="store_true", help="Print the normalized configuration dictionary.")
    parser.add_argument(
        "--strict-unknowns",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fail-fast on unknown keys (default: true in development, false when "
            "ANO_RAG_ENV/ENVIRONMENT indicates production)."
        ),
    )
    return parser.parse_args()


def run_validate(profile: str | None, show_config: bool, strict_unknowns: bool | None) -> None:
    try:
        frozen, model, normalized, layers = load_config(profile=profile, strict_unknowns=strict_unknowns)
    except ValueError as exc:
        print(f"Configuration validation failed: {exc}")
        raise SystemExit(1) from exc

    print("Loaded configuration layers:")
    for layer in layers:
        print(f"  - {layer}")
    print(format_diagnostics(normalized.get("_diagnostics", {})))
    if show_config:
        print(json.dumps(normalized, indent=2, ensure_ascii=False))


def run_print_schema(schema_path: str) -> None:
    schema = RootConfig.model_json_schema()
    o
