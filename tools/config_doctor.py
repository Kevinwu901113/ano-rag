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
    return parser.parse_args()


def run_validate(profile: str | None, show_config: bool) -> None:
    try:
        frozen, model, normalized, layers = load_config(profile=profile)
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
    output = Path(schema_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Schema written to {output}")


def run_doctor(profile: str | None, schema_path: str, show_config: bool) -> None:
    run_validate(profile=profile, show_config=show_config)
    run_print_schema(schema_path)


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        run_validate(profile=args.profile, show_config=args.show_config)
    elif args.command == "print-schema":
        run_print_schema(args.schema_path)
    else:
        run_doctor(profile=args.profile, schema_path=args.schema_path, show_config=args.show_config)


if __name__ == "__main__":
    main()
