import json
from dataclasses import dataclass
from typing import List, Dict

from .notes import AtomicNote
from .chunking import chunk_text


def parse_json(path: str) -> List[AtomicNote]:
    """Parse a JSON file and return a list of AtomicNote objects."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    notes = []
    if isinstance(data, dict):
        # Single document stored in a dict
        text = data.get('text', '')
        metadata = {k: v for k, v in data.items() if k != 'text'}
        for chunk in chunk_text(text):
            notes.append(AtomicNote(doc_id=path, content=chunk, metadata=metadata))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            text = item.get('text', '') if isinstance(item, dict) else str(item)
            metadata = item if isinstance(item, dict) else {'index': idx}
            metadata = {k: v for k, v in metadata.items() if k != 'text'}
            for chunk in chunk_text(text):
                notes.append(AtomicNote(doc_id=f"{path}#{idx}", content=chunk, metadata=metadata))
    return notes
