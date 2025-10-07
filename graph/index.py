from collections import defaultdict
from typing import Any, Dict, List, Tuple

from config import config


class NoteGraph:
    """A lightweight graph where atomic notes become nodes connected via literals."""

    def __init__(self) -> None:
        self.notes: Dict[str, Dict[str, Any]] = {}
        self.adj: Dict[str, List[Tuple[str, str, str, float, int]]] = defaultdict(list)

        edge_cfg = config.get("graph.edge", {}) or {}
        self.base_weight = float(edge_cfg.get("base_weight", 0.0))
        self.w_key = float(edge_cfg.get("key_match_weight", 1.5))
        self.w_type = float(edge_cfg.get("type_compat_weight", 1.0))
        self.b_para = float(edge_cfg.get("same_paragraph_bonus", 0.3))
        self.b_title = float(edge_cfg.get("same_title_bonus", 0.0))
        self.default_rel = str(config.get("note_keys.default_rel", "related_to"))

    def add_note(self, note: Dict[str, Any]) -> None:
        text = str(note.get("text") or "").strip()
        if not text:
            return

        note_id = note.get("id") or note.get("note_id") or str(hash(text))
        note["id"] = note_id
        self.notes[note_id] = note

        head_key = note.get("head_key") or ""
        tail_key = note.get("tail_key") or ""
        if not head_key or not tail_key:
            return

        rel = note.get("rel") or self.default_rel
        paragraph_idxs = note.get("paragraph_idxs") or []
        paragraph_idx = paragraph_idxs[0] if paragraph_idxs else -1

        weight = self.base_weight
        if head_key and tail_key:
            weight += self.w_key

        type_head = note.get("type_head")
        type_tail = note.get("type_tail")
        if type_head and type_tail:
            weight += self.w_type
        if paragraph_idx >= 0:
            weight += self.b_para

        title = (note.get("title") or "").strip()
        if title:
            weight += self.b_title

        self.adj[head_key].append((rel, tail_key, note_id, weight, paragraph_idx))

    def neighbors(self, head_key: str) -> List[Tuple[str, str, str, float, int]]:
        return list(self.adj.get(head_key, []))
