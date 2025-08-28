"""Scan configuration file for deprecated keys."""
import sys
from pathlib import Path
import yaml
from typing import Any, Dict

# Ensure repository root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config_loader import DEFAULT_CONFIG


def flatten(data: Dict[str, Any], prefix: str = "") -> set[str]:
    paths = set()
    for k, v in data.items():
        new_prefix = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            paths.update(flatten(v, new_prefix))
        else:
            paths.add(new_prefix)
    return paths


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    allowed = flatten(DEFAULT_CONFIG)
    current = flatten(cfg)
    deprecated = sorted(p for p in current if p not in allowed)

    if deprecated:
        print("Deprecated keys detected:")
        for key in deprecated:
            print(key)
    else:
        print("No deprecated keys found.")


if __name__ == "__main__":
    main()
