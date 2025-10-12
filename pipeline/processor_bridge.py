"""Bridges Musique processing pipeline for reuse in other entry points."""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional

from loguru import logger

from llm import LocalLLM
from main_musique import (
    MusiqueProcessor,
    create_item_workdir,
    create_new_workdir,
)


class AnoragProcessorBridge:
    """Thin wrapper that exposes a Musique-style processor for NQ inputs."""

    def __init__(
        self,
        max_workers: int = 4,
        debug: bool = False,
        work_dir: Optional[str] = None,
        enable_cor: bool = False,
        llm: Optional[LocalLLM] = None,
    ) -> None:
        self.debug = debug
        self.base_work_dir = work_dir or create_new_workdir()
        os.makedirs(self.base_work_dir, exist_ok=True)
        self.llm = llm or LocalLLM()
        self.proc = MusiqueProcessor(
            max_workers=max_workers,
            debug=debug,
            work_dir=self.base_work_dir,
            llm=self.llm,
            enable_cor=enable_cor,
        )

    def _cleanup_dir(self, path: str) -> None:
        if self.debug:
            return
        try:
            shutil.rmtree(path)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.debug(f"Failed to cleanup work dir {path}: {exc}")

    def process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qid = str(item.get("id"))
        item_work_dir = create_item_workdir(
            self.base_work_dir,
            qid,
            debug_mode=self.debug,
        )
        try:
            result, atomic_notes_info = self.proc.process_single_item(item, item_work_dir)
        finally:
            self._cleanup_dir(item_work_dir)

        result = result or {}
        if isinstance(atomic_notes_info, dict):
            result = dict(result)
            result.setdefault("atomic_notes_info", atomic_notes_info)

        atomic_notes_info = result.get("atomic_notes_info", {}) or {}
        recalled_notes: List[Dict[str, Any]] = atomic_notes_info.get("recalled_atomic_notes", [])
        if not isinstance(recalled_notes, list):
            recalled_notes = []

        try:
            recalled_notes = sorted(
                recalled_notes,
                key=lambda n: float(n.get("similarity_score", 0.0)),
                reverse=True,
            )
        except Exception:
            pass

        retrieved_doc_ids: List[str] = []
        seen: set[str] = set()
        for note in recalled_notes:
            paragraph_idxs = note.get("paragraph_idxs") or []
            for idx in paragraph_idxs:
                try:
                    key = f"{qid}_{int(idx)}"
                except (TypeError, ValueError):
                    continue
                if key not in seen:
                    retrieved_doc_ids.append(key)
                    seen.add(key)

        return {
            "id": qid,
            "predicted_answer": result.get("predicted_answer", ""),
            "predicted_support_idxs": result.get("predicted_support_idxs", []),
            "retrieved_doc_ids": retrieved_doc_ids,
            "extra": {
                "raw": result,
            },
        }
