"""Utilities for validating note completeness based on configuration rules."""

from functools import lru_cache
import re
from typing import Dict, List, Any

from config import config


@lru_cache(maxsize=1)
def _compiled_rules() -> Dict[str, Any]:
    """Load and compile completeness rules from configuration."""

    cfg = config.get("note_completeness", {}) or {}

    def get_list(key: str) -> List[str]:
        values = cfg.get(key, [])
        if isinstance(values, list):
            return [str(x) for x in values]
        if values is None:
            return []
        return [str(values)]

    def get_bool(key: str, default: bool = False) -> bool:
        value = cfg.get(key, default)
        return bool(value)

    def get_int(key: str, default: int = 0) -> int:
        value = cfg.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    rules = {
        "require_sentence_terminal": get_bool("require_sentence_terminal", True),
        "allowed_sentence_terminals": get_list("allowed_sentence_terminals") or ["。", ".", "!", "?"],
        "min_word_count_en": get_int("min_word_count_en", 5),
        "min_char_count_zh": get_int("min_char_count_zh", 10),
        "verb_patterns_en": [re.compile(pattern, re.I) for pattern in get_list("verb_patterns_en")],
        "verb_patterns_zh": [re.compile(pattern) for pattern in get_list("verb_patterns_zh")],
        "bad_starts_en": [re.compile(pattern, re.I) for pattern in get_list("bad_starts_en")],
        "bad_starts_zh": [re.compile(pattern) for pattern in get_list("bad_starts_zh")],
        "require_entities": get_bool("require_entities", True),
    }

    return rules


def _word_count_en(text: str) -> int:
    """Approximate English word count in the text."""

    return len(re.findall(r"\w+", text))


def _char_count_zh(text: str) -> int:
    """Approximate count of Chinese characters in the text."""

    return len(re.findall(r"[\u4e00-\u9fff]", text))


def is_complete_sentence(text: str, entities: List[str]) -> bool:
    """Determine whether a note text qualifies as a complete proposition."""

    if not text:
        return False

    trimmed = text.strip()
    if not trimmed:
        return False

    rules = _compiled_rules()

    if rules["require_sentence_terminal"] and not any(
        trimmed.endswith(char) for char in rules["allowed_sentence_terminals"]
    ):
        return False

    if _word_count_en(trimmed) < rules["min_word_count_en"] and _char_count_zh(trimmed) < rules["min_char_count_zh"]:
        return False

    has_verb = any(pattern.search(trimmed) for pattern in rules["verb_patterns_en"]) or any(
        pattern.search(trimmed) for pattern in rules["verb_patterns_zh"]
    )
    if not has_verb:
        return False

    if any(pattern.search(trimmed) for pattern in rules["bad_starts_en"]) or any(
        pattern.search(trimmed) for pattern in rules["bad_starts_zh"]
    ):
        return False

    if rules["require_entities"]:
        if not isinstance(entities, list) or len(entities) == 0:
            return False

    return True
