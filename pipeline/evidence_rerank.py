"""Lightweight evidence re-ranking heuristics."""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from config import config


class EvidenceReranker:
    """Apply type- and prompt-aware heuristics to reorder evidence notes."""

    def __init__(self) -> None:
        cfg = config.get("evidence_rerank", {}) or {}
        enabled = cfg.get("enable")
        if enabled is None:
            enabled = cfg.get("enabled")
        self.enabled: bool = bool(True if enabled is None else enabled)

        self.w_album: float = float(cfg.get("w_album", 0.0))
        self.w_song: float = float(cfg.get("w_song", 0.0))
        self.w_supporting: float = float(cfg.get("w_supporting", 0.0))
        self.w_q_performer_album: float = float(cfg.get("w_q_performer_album", 0.0))

        self.album_tokens: List[str] = [str(t).lower() for t in cfg.get("album_tokens", ["(album)", " album"]) if t]
        self.song_tokens: List[str] = [str(t).lower() for t in cfg.get("song_tokens", ["(song)", " single"]) if t]
        self.support_flag_keys: List[str] = [
            str(t) for t in cfg.get("support_flag_keys", ["is_supporting", "supporting"]) if t
        ]
        self.query_performer_terms: List[str] = [
            str(t).lower() for t in cfg.get("query_performer_terms", ["performer", "singer", "vocalist"]) if t
        ]
        self.query_album_terms: List[str] = [
            str(t).lower() for t in cfg.get("query_album_terms", ["album", "record", "lp"]) if t
        ]

    def rerank(self, notes: List[Dict[str, Any]], query: str | None) -> List[Dict[str, Any]]:
        """Return a reordered list based on heuristic scores."""

        if not self.enabled or not notes:
            return notes

        query_lower = (query or "").lower()
        query_has_performer = any(term in query_lower for term in self.query_performer_terms)
        query_has_album = any(term in query_lower for term in self.query_album_terms)

        scored: List[tuple[float, int, Dict[str, Any]]] = []

        for idx, note in enumerate(notes):
            base_score = float(note.get("final_score", note.get("salience", 0.0)))
            score = base_score

            text_parts = [
                str(note.get("title", "")),
                str(note.get("raw_span", "")),
                str(note.get("content", "")),
            ]
            combined_text = " \n".join(text_parts).lower()

            has_album = any(token in combined_text for token in self.album_tokens)
            has_song = any(token in combined_text for token in self.song_tokens)

            if has_album:
                score += self.w_album
            if has_song:
                score += self.w_song
            if has_album and query_has_performer and query_has_album:
                score += self.w_q_performer_album
            if self._is_supporting(note):
                score += self.w_supporting

            note['rerank_score'] = score
            scored.append((score, -idx, note))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        reordered = [item[2] for item in scored]

        logger.debug(
            f"Evidence rerank applied: {len(notes)} -> {len(reordered)} candidates"
        )
        return reordered

    def _is_supporting(self, note: Dict[str, Any]) -> bool:
        tags = note.get("tags") or {}
        for key in self.support_flag_keys:
            value = note.get(key)
            if isinstance(value, bool) and value:
                return True
            if isinstance(value, str) and value.lower() in {"true", "supporting", "yes"}:
                return True
            if isinstance(tags, dict):
                tag_value = tags.get(key)
                if isinstance(tag_value, bool) and tag_value:
                    return True
                if isinstance(tag_value, str) and tag_value.lower() in {"true", "supporting", "yes"}:
                    return True
        return False
