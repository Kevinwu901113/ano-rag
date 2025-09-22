"""Utilities for representing structured retrieval results."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence


def resolve_candidate_id(candidate: Mapping[str, Any]) -> Optional[str]:
    """Derive a stable identifier for a retrieved candidate."""

    for key in ("note_id", "id", "uuid", "document_id", "doc_id"):
        value = candidate.get(key)
        if value is not None:
            return str(value)

    content = candidate.get("content") or candidate.get("text")
    if isinstance(content, str) and content:
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
        return f"content:{digest}"

    retrieval = candidate.get("retrieval_info")
    if isinstance(retrieval, Mapping):
        rid = retrieval.get("candidate_id")
        if rid is not None:
            return str(rid)

    return None


def merge_per_candidate_maps(
    base: MutableMapping[str, Any], incoming: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Merge two per-candidate metadata mappings."""

    for key, value in incoming.items():
        if key in base and isinstance(base[key], Mapping) and isinstance(value, Mapping):
            merged = dict(base[key])
            merged.update(value)
            base[key] = merged
        elif isinstance(value, Mapping):
            base[key] = dict(value)
        else:
            base[key] = value
    return base


def normalize_timestamp(value: Any) -> Optional[float]:
    """Normalize various timestamp representations to epoch seconds."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            try:
                return float(text)
            except ValueError:
                return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(text, fmt).timestamp()
                except ValueError:
                    continue
    return None


def collect_topical_tags(note: Mapping[str, Any]) -> Sequence[str]:
    """Collect topical hints from candidate level metadata."""

    tags: list[str] = []

    def _extend(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            tags.extend(part for part in parts if part)
        elif isinstance(value, Iterable):
            for item in value:
                if isinstance(item, str) and item:
                    tags.append(item)

    for key in ("topical_tags", "keywords", "concepts", "topics", "tags"):
        _extend(note.get(key))

    metadata = note.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("keywords", "concepts", "topics", "tags"):
            _extend(metadata.get(key))

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = tag.strip()
        if not normalized:
            continue
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        deduped.append(normalized)

    return deduped


@dataclass
class HybridRetrievalResult(Sequence[Dict[str, Any]]):
    """Container that couples candidates with per-candidate metadata."""

    candidates: list[Dict[str, Any]] = field(default_factory=list)
    per_candidate: dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.candidates[item]

    def extend(self, other: "HybridRetrievalResult") -> None:
        self.candidates.extend(other.candidates)
        merge_per_candidate_maps(self.per_candidate, other.per_candidate)

    def to_list(self) -> list[Dict[str, Any]]:
        return list(self.candidates)

