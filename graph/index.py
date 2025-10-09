from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from config import config


class NoteGraph:
    """A lightweight graph where atomic notes become nodes connected via literals."""

    def __init__(self) -> None:
        self.notes: Dict[str, Dict[str, Any]] = {}
        self.adj: Dict[str, List[Tuple[str, str, str, float, int]]] = defaultdict(list)
        self._neighbor_cache: Dict[str, List[str]] = {}

        edge_cfg = config.get("graph.edge", {}) or {}
        self.w_key = float(edge_cfg.get("key_match_weight", 1.5))
        self.w_type = float(edge_cfg.get("type_compat_weight", 1.0))
        self.b_para = float(edge_cfg.get("same_paragraph_bonus", 0.3))
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

        weight = self.w_key

        type_head = note.get("type_head")
        type_tail = note.get("type_tail")
        if type_head or type_tail:
            weight += self.w_type
        if paragraph_idx >= 0:
            weight += self.b_para

        self.adj[head_key].append((rel, tail_key, note_id, weight, paragraph_idx))

    def neighbors(self, head_key: str) -> List[Tuple[str, str, str, float, int]]:
        return list(self.adj.get(head_key, []))

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "NoteGraph":
        """Factory helper kept for API compatibility."""

        _ = cfg  # 当前实现不依赖外部配置，保留接口以便未来扩展
        return cls()

    def seed_recall(self, question: str, top_k: int = 40, diversify: bool = True) -> List[str]:
        """Return a lightweight ranked list of note ids for the query.

        This minimal implementation performs lexical matching over the stored
        notes. Projects with dedicated dense/BM25 retrievers can replace the
        logic while keeping the interface expected by ``chain_of_retrieval``.
        """

        if not self.notes:
            return []

        query = (question or "").lower()
        if not query.strip():
            return list(self.notes.keys())[:top_k]

        q_tokens = {tok for tok in query.split() if tok}
        scored: List[Tuple[str, float]] = []
        for note_id, payload in self.notes.items():
            text = str(payload.get("text", "")).lower()
            head = str(payload.get("head_key", "")).lower()
            tail = str(payload.get("tail_key", "")).lower()
            candidate_tokens = [tok for tok in (text.split() + head.split() + tail.split()) if tok]
            if not candidate_tokens:
                continue

            overlap = sum(1 for tok in candidate_tokens if tok in q_tokens)
            if overlap == 0:
                continue
            length_norm = len(candidate_tokens)
            score = overlap / max(1, length_norm)
            scored.append((note_id, score))

        if not scored:
            # 回退到简单的 note_id 顺序，确保不会返回空列表
            return list(self.notes.keys())[:top_k]

        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = [nid for nid, _ in scored]

        if diversify:
            seen_heads = set()
            diversified: List[str] = []
            for nid in ranked:
                head_key = str(self.notes.get(nid, {}).get("head_key") or "")
                if head_key and head_key in seen_heads:
                    continue
                if head_key:
                    seen_heads.add(head_key)
                diversified.append(nid)
            ranked = diversified if diversified else ranked

        return ranked[:top_k]

    def get_neighbors(self, note_id: str, cap: int = 8) -> List[str]:
        """Return neighbour note ids for the provided note id."""

        if not note_id:
            return []

        cached = self._neighbor_cache.get(note_id)
        if cached is not None and len(cached) >= min(cap, len(cached)):
            return cached[:cap]

        note = self.notes.get(note_id)
        if not note:
            return []

        head_key = note.get("head_key") or ""
        tail_key = note.get("tail_key") or ""
        neighbour_edges: List[Tuple[str, str, str, float, int]] = []
        if head_key:
            neighbour_edges.extend(self.adj.get(head_key, []))
        if tail_key and tail_key != head_key:
            neighbour_edges.extend(self.adj.get(tail_key, []))

        neighbour_edges.sort(key=lambda item: item[3], reverse=True)
        neighbours = []
        for _, _, nid, _, _ in neighbour_edges:
            if nid == note_id:
                continue
            if nid not in neighbours:
                neighbours.append(nid)
            if len(neighbours) >= cap:
                break

        self._neighbor_cache[note_id] = neighbours
        return neighbours[:cap]
