from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class NoteStore:
    """Lazy note_id -> note lookup to avoid repeated file scans."""

    def __init__(self, notes_path: str, weak_notes_path: Optional[str] = None) -> None:
        self.notes_path = notes_path
        self._cache: Dict[str, Dict] | None = None
        self.weak_notes_path = weak_notes_path or self._infer_weak_path()
        self._weak_cache: Dict[str, Dict] | None = None

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

    def _infer_weak_path(self) -> Optional[str]:
        base = Path(self.notes_path)
        new_candidate = base.parent / "weak" / "weak_notes.jsonl"
        if new_candidate.exists():
            return str(new_candidate)
        legacy = base.with_name("weak_notes.jsonl")
        return str(legacy) if legacy.exists() else None

    def get(self, note_id: str) -> Optional[Dict]:
        self._ensure_loaded()
        return self._cache.get(note_id) if self._cache else None

    def get_many(self, note_ids: List[str]) -> List[Dict]:
        self._ensure_loaded()
        if not self._cache:
            return []
        return [self._cache[nid] for nid in note_ids if nid in self._cache]

    def _ensure_weak_loaded(self) -> None:
        if self._weak_cache is not None or not self.weak_notes_path:
            return
        cache: Dict[str, Dict] = {}
        try:
            with open(self.weak_notes_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    note = json.loads(line)
                    nid = note.get("note_id")
                    if nid:
                        cache[nid] = note
        except FileNotFoundError:
            cache = {}
        self._weak_cache = cache

    def get_weak(self, note_id: str) -> Optional[Dict]:
        if not note_id:
            return None
        self._ensure_weak_loaded()
        if self._weak_cache and note_id in self._weak_cache:
            return self._weak_cache[note_id]
        return self.get(note_id)

    def iter_weak(self) -> List[Dict]:
        self._ensure_weak_loaded()
        if not self._weak_cache:
            return []
        return list(self._weak_cache.values())
