import re
from typing import List


SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    return [s.strip() for s in SENTENCE_END_RE.split(text) if s.strip()]


def chunk_text(text: str, max_length: int = 200) -> List[str]:
    """Chunk text into pieces, preserving sentence boundaries."""
    sentences = split_sentences(text)
    chunks = []
    current = ''
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_length and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        chunks.append(current.strip())
    return chunks
