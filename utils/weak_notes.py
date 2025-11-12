from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


class _WeakNoteWriter:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / "weak" / "weak_notes.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._initialized = False
        self._lock = threading.Lock()

    def _ensure_handle(self) -> None:
        if self._handle is None:
            mode = "w" if not self._initialized else "a"
            self._handle = open(self.path, mode, encoding="utf-8")
            self._initialized = True

    def write(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._ensure_handle()
            self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._handle.close()
                self._handle = None


_WRITER_CACHE: Dict[str, _WeakNoteWriter] = {}
_CACHE_LOCK = threading.Lock()


def _get_writer(base_dir: str) -> _WeakNoteWriter:
    normalized = str(Path(base_dir).resolve())
    with _CACHE_LOCK:
        writer = _WRITER_CACHE.get(normalized)
        if writer is None:
            writer = _WeakNoteWriter(normalized)
            _WRITER_CACHE[normalized] = writer
        return writer


def write_weak_note(
    out_dir: str,
    note: Dict[str, Any],
    coref_candidates: Optional[List[Dict[str, Any]]] = None,
    anchor: Optional[str] = None,
) -> str:
    """Persist a weak note entry alongside its candidate coreference hints."""
    base_id = str(note.get("note_id") or "weak")
    weak_id = base_id if base_id.endswith("#weak") else f"{base_id}#weak"
    payload = deepcopy(note)
    payload["note_id"] = weak_id
    subj = (payload.get("subj") or "").strip()
    if subj and subj.startswith("<") and subj.endswith(">"):
        resolved_subj = subj
    else:
        resolved_subj = "<UNRESOLVED>"
    payload["subj"] = resolved_subj
    meta = deepcopy(payload.get("meta") or {})
    meta["coref_candidates"] = deepcopy(coref_candidates or meta.get("coref_candidates") or [])
    if anchor:
        meta["anchor_entity"] = anchor
    meta["coref_unresolved"] = True
    meta["filter_out_strict"] = True
    meta["weak_note"] = True
    if "evidence_canonical" not in meta:
        meta["evidence_canonical"] = payload.get("evidence")
    payload["meta"] = meta
    writer = _get_writer(out_dir)
    writer.write(payload)
    return weak_id


def close_weak_note_writer(out_dir: str) -> None:
    normalized = str(Path(out_dir).resolve())
    with _CACHE_LOCK:
        writer = _WRITER_CACHE.pop(normalized, None)
    if writer:
        writer.close()


def close_all_weak_note_writers() -> None:
    with _CACHE_LOCK:
        writers = list(_WRITER_CACHE.items())
        _WRITER_CACHE.clear()
    for _, writer in writers:
        writer.close()
