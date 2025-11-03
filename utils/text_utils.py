import re
from typing import List


class TextUtils:
    @staticmethod
    def split_by_sentence(text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text.strip())
        parts = re.split(r"(?<=[。！？.!?])\s+", cleaned)
        return [p.strip() for p in parts if p.strip()]
