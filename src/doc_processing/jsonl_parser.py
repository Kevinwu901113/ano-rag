import json
from typing import List

from .notes import AtomicNote
from .chunking import chunk_text


def parse_jsonl(path: str) -> List[AtomicNote]:
    """Parse a JSONL file and return a list of AtomicNote objects."""
    notes = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                text = data.get('text', '') if isinstance(data, dict) else str(data)
                metadata = data if isinstance(data, dict) else {'index': idx}
                metadata = {k: v for k, v in metadata.items() if k != 'text'}
            except json.JSONDecodeError:
                text = line.strip()
                metadata = {'index': idx}
            for chunk in chunk_text(text):
                notes.append(AtomicNote(doc_id=f"{path}#{idx}", content=chunk, metadata=metadata))
    return notes
