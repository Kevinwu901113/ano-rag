from __future__ import annotations

import re
from typing import Optional


NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}

NUM_WORD_PATTERN = "|".join(sorted(NUM_WORDS.keys(), key=len, reverse=True))


def parse_int(text: Optional[str]) -> Optional[int]:
    raw = str(text or "").strip().lower()
    if not raw:
        return None
    raw = re.sub(r"[,_]", "", raw)
    if raw.isdigit():
        try:
            return int(raw)
        except ValueError:
            return None
    tokens = raw.replace("-", " ").split()
    total = 0
    current = 0
    for tok in tokens:
        if tok not in NUM_WORDS:
            return None
        value = NUM_WORDS[tok]
        if value in (100, 1000):
            if current == 0:
                current = 1
            current *= value
            if value == 1000:
                total += current
                current = 0
        else:
            current += value
    return total + current
