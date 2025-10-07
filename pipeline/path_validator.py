"""Graph pattern validator for multi-hop evidence bundles."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from config import config


class PathValidator:
    """Validate and repair evidence bundles using coarse graph patterns."""

    def __init__(self) -> None:
        cfg = config.get("path_validator", {}) or {}
        enabled = cfg.get("enable")
        if enabled is None:
            enabled = cfg.get("enabled")
        self.enabled: bool = bool(enabled) if enabled is not None else False

        self.patterns: List[List[str]] = [
            [str(token) for token in pattern]
            for pattern in cfg.get("patterns", [])
            if isinstance(pattern, (list, tuple))
        ]
        self.max_bundle_size: int = int(cfg.get("max_bundle_size", 12))

        relation_synonyms = cfg.get("relation_synonyms", {}) or {}
        self.relation_map: Dict[str, List[str]] = {
            key.lower(): [str(v).lower() for v in value]
            for key, value in relation_synonyms.items()
        }

        # Built-in fallbacks for common relations
        self.relation_map.setdefault("performed_by", ["performed by", "performer", "performed"])
        self.relation_map.setdefault(
            "spouse_of", ["spouse", "married", "husband", "wife", "spouse of"]
        )
        self.relation_map.setdefault(
            "partner_of", ["partner", "partnered", "partner of", "partnered with"]
        )

    def ensure_valid_bundle(
        self,
        selected_notes: List[Dict[str, Any]],
        candidate_notes: Optional[List[Dict[str, Any]]],
        query: Optional[str] = None,
        target_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Ensure a bundle satisfies at least one configured pattern."""

        if not self.enabled or not self.patterns:
            return selected_notes

        if self.validate(selected_notes):
            return selected_notes

        candidate_pool = candidate_notes or selected_notes
        desired_size = target_size or len(selected_notes)
        if self.max_bundle_size:
            desired_size = min(desired_size, self.max_bundle_size)

        replacement = self._build_valid_bundle(candidate_pool, desired_size)
        if replacement:
            logger.info("Path validator replaced evidence bundle with compliant set")
            return replacement

        logger.debug("Path validator found no compliant bundle; keeping original set")
        return selected_notes

    def validate(self, notes: List[Dict[str, Any]]) -> bool:
        if not self.enabled or not notes or not self.patterns:
            return True

        for pattern in self.patterns:
            if self._find_bundle_for_pattern(notes, pattern, 0):
                return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_valid_bundle(
        self, candidates: List[Dict[str, Any]], target_size: int
    ) -> Optional[List[Dict[str, Any]]]:
        for pattern in self.patterns:
            bundle = self._find_bundle_for_pattern(candidates, pattern, target_size)
            if bundle:
                return bundle
        return None

    def _find_bundle_for_pattern(
        self,
        candidates: List[Dict[str, Any]],
        pattern: List[str],
        target_size: int,
    ) -> Optional[List[Dict[str, Any]]]:
        tokens = [self._split_token(token) for token in pattern]
        if len(tokens) < 5 or len(tokens) % 2 == 0:
            return None

        first_entity_terms, relation1_terms, bridge_terms, relation2_terms, final_entity_terms = (
            tokens[0],
            tokens[1],
            tokens[2],
            tokens[3],
            tokens[4],
        )

        album_candidates = [
            note
            for note in candidates
            if self._note_matches_terms(note, first_entity_terms)
            and self._note_contains_relation(note, relation1_terms)
        ]

        if not album_candidates:
            return None

        for first_note in album_candidates:
            bridge_entities = [
                entity
                for entity in self._extract_entities(first_note)
                if self._entity_matches_terms(entity, bridge_terms)
            ]
            if not bridge_entities:
                continue

            for bridge_entity in bridge_entities:
                spouse_notes = [
                    note
                    for note in self._find_notes_with_entity(candidates, bridge_entity, exclude=first_note)
                    if self._note_contains_relation(note, relation2_terms)
                    and self._note_matches_terms(note, final_entity_terms)
                ]

                if not spouse_notes:
                    continue

                core_notes = [first_note, spouse_notes[0]]
                if target_size <= 0:
                    return core_notes
                return self._assemble_bundle(core_notes, candidates, target_size)

        return None

    def _assemble_bundle(
        self, core_notes: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        target_size: int,
    ) -> List[Dict[str, Any]]:
        bundle: List[Dict[str, Any]] = []
        seen = set()

        for note in core_notes:
            note_id = id(note)
            if note_id in seen:
                continue
            bundle.append(note)
            seen.add(note_id)

        if len(bundle) >= target_size:
            return bundle[:target_size]

        sorted_candidates = sorted(
            candidates,
            key=lambda item: item.get("final_score", item.get("salience", 0.0)),
            reverse=True,
        )

        for note in sorted_candidates:
            if id(note) in seen:
                continue
            bundle.append(note)
            seen.add(id(note))
            if len(bundle) >= target_size:
                break

        return bundle

    def _split_token(self, token: str) -> List[str]:
        return [part.strip().lower() for part in str(token).split("|") if part]

    def _note_matches_terms(self, note: Dict[str, Any], terms: Iterable[str]) -> bool:
        terms = list(terms)
        if not terms:
            return True

        combined = self._note_text(note)
        entities = [str(e).lower() for e in note.get("entities", []) if isinstance(e, str)]

        for term in terms:
            if term in {"*", "person"}:
                if entities or any(keyword in combined for keyword in ["singer", "artist", "actor", "actress"]):
                    return True
                continue
            if term == "album":
                if "(album)" in combined or " album" in combined:
                    return True
                continue
            if term and term in combined:
                return True
        return False

    def _entity_matches_terms(self, entity: str, terms: Iterable[str]) -> bool:
        terms = list(terms)
        if not terms:
            return True
        entity_lower = entity.lower()
        for term in terms:
            if term in {"*", "person"}:
                return True
            if term and term in entity_lower:
                return True
        return False

    def _note_contains_relation(self, note: Dict[str, Any], terms: Iterable[str]) -> bool:
        terms = list(terms)
        if not terms:
            return True

        combined = self._note_text(note)
        for term in terms:
            normalized = term.replace("_", " ")
            if normalized and normalized in combined:
                return True
            synonyms = self.relation_map.get(term, [])
            if any(syn in combined for syn in synonyms):
                return True
        return False

    def _find_notes_with_entity(
        self, notes: List[Dict[str, Any]], entity: str, exclude: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        entity_lower = entity.lower()
        for note in notes:
            if exclude is not None and note is exclude:
                continue
            if entity_lower in [str(e).lower() for e in note.get("entities", []) if isinstance(e, str)]:
                matches.append(note)
                continue
            if entity_lower and entity_lower in self._note_text(note):
                matches.append(note)
        return matches

    def _extract_entities(self, note: Dict[str, Any]) -> List[str]:
        entities = note.get("entities", [])
        if isinstance(entities, list):
            return [str(e) for e in entities if e]
        return []

    def _note_text(self, note: Dict[str, Any]) -> str:
        parts = [
            str(note.get("title", "")),
            str(note.get("raw_span", "")),
            str(note.get("content", "")),
        ]
        return " \n".join(parts).lower()
