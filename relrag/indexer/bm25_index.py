from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from relrag.config import config as config_loader
from relrag.utils.text_builders import build_note_text_for_embed


class BM25IndexBuilder:
    """Simple JSONL-based BM25 corpus builder."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or config_loader.load_config()
        self.notes_path = Path(self.cfg.get("notes", {}).get("out_path", "notes/notes.jsonl"))
        retriever_cfg = self.cfg.get("retriever", {})
        self.bm25_cfg = retriever_cfg.get("bm25", {})

    def build(self) -> None:
        if not self.bm25_cfg.get("enabled", False):
            logger.info("BM25 backend disabled; skip index build.")
            return
        if not self.notes_path.exists():
            raise FileNotFoundError(f"Notes file not found: {self.notes_path}")

        store_path = Path(self.bm25_cfg.get("store_path", "indexes/bm25/notes"))
        store_path.mkdir(parents=True, exist_ok=True)
        out_file = store_path / "notes.jsonl"

        count = 0
        with self.notes_path.open("r", encoding="utf-8") as reader, out_file.open("w", encoding="utf-8") as writer:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                note = json.loads(line)
                record = self._note_record(note)
                if not record:
                    continue
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        logger.info("BM25 corpus written: {} notes -> {}", count, out_file)

    def _note_record(self, note: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        note_id = note.get("note_id")
        if not note_id:
            return None
        meta = (note.get("meta") or {}) if isinstance(note, dict) else {}
        fields = {
            "subj": (note.get("subj") or "").strip(),
            "pred": (note.get("pred") or "").strip(),
            "obj": (note.get("obj") or "").strip(),
            "ctx": meta.get("evidence_canonical") or note.get("evidence") or "",
        }
        # Fall back to embedding builder text for richer context
        full_text = build_note_text_for_embed(note, max_len=self.bm25_cfg.get("max_len_note", 512))
        return {
            "note_id": note_id,
            "fields": fields,
            "text": full_text,
        }


def main() -> None:
    cfg = config_loader.load_config()
    builder = BM25IndexBuilder(cfg)
    builder.build()


if __name__ == "__main__":
    main()
