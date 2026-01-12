from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


_PROMPT_DIR = Path(__file__).resolve().parent
_CACHE: Dict[str, str] = {}


def _escape_braces(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


def load_prompt(name: str) -> str:
    if name not in _CACHE:
        path = _PROMPT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        _CACHE[name] = path.read_text(encoding="utf-8")
    return _CACHE[name]


def render_prompt(name: str, **kwargs: Any) -> str:
    template = load_prompt(name)
    if not kwargs:
        return template
    safe_kwargs = {key: _escape_braces(str(value)) for key, value in kwargs.items()}
    return template.format(**safe_kwargs)
