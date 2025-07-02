from typing import List
from docx import Document

from .notes import AtomicNote
from .chunking import chunk_text


def parse_word(path: str) -> List[AtomicNote]:
    """Parse a Word (.docx) file and return AtomicNote objects."""
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    notes = [AtomicNote(doc_id=path, content=chunk, metadata={}) for chunk in chunk_text(text)]
    return notes
