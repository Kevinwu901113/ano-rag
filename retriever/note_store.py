from __future__ import annotations

import json
from typing import Dict, List, Optional


class NoteStore:
    """Lazy note_id -> note lookup to avoid repeated file scans."""

    def __init__(self, notes_path: str) -> None:
        self.notes_path = notes_path
        self._cache: Dict[str, Dict] | None = None

    def _ensure_loaded(self) -> None:
        if self._cache is not None:
            return
        cache: Dict[str, Dict] = {}
        with open(self.notes_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                note = json.loads(line)
                note_id = note.get("note_id")
                if note_id:
                    cache[note_id] = note
        self._cache = cache

    def get(self, note_id: str) -> Optional[Dict]:
        self._ensure_loaded()
        return self._cache.get(note_id) if self._cache else None

    def get_many(self, note_ids: List[str]) -> List[Dict]:
        self._ensure_loaded()
        if not self._cache:
            return []
        return [self._cache[nid] for nid in note_ids if nid in self._cache]
