from __future__ import annotations

from typing import Any, Dict, List, Optional


def _normalize_text_key(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def _collapse_pipe_duplicates(text: str) -> str:
    if not text:
        return ""
    parts = [part.strip() for part in text.split("|")]
    cleaned: List[str] = []
    for part in parts:
        if not part:
            continue
        if not cleaned or part != cleaned[-1]:
            cleaned.append(part)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    first_key = _normalize_text_key(cleaned[0])
    if first_key and all(_normalize_text_key(part) == first_key for part in cleaned[1:]):
        return cleaned[0]
    return " | ".join(cleaned)


def _dedupe_texts(texts: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for text in texts:
        cleaned = _collapse_pipe_duplicates(text.strip())
        key = _normalize_text_key(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _collect_note_parts(note: Dict[str, Any]) -> Dict[str, List[str]]:
    """Break note into semantic buckets to support various text builders."""
    meta = (note.get("meta") or {}) if isinstance(note, dict) else {}
    attribute = meta.get("attribute") or {}
    subj_profile = meta.get("subject_profile") or {}
    obj_profile = meta.get("object_profile") or {}

    evidence = meta.get("evidence_canonical") or note.get("evidence") or ""
    raw_evidence = note.get("evidence") or ""
    summaries = [attribute.get("summary")] if isinstance(attribute, dict) else []
    descriptions: List[str] = []
    for profile in (subj_profile, obj_profile):
        desc = profile.get("description")
        if isinstance(desc, str) and desc.strip():
            descriptions.append(desc.strip())

    fields = {
        "subj": str(note.get("subj") or "").strip(),
        "pred": str(note.get("pred") or "").strip(),
        "obj": str(note.get("obj") or "").strip(),
        "subj_type": str(note.get("subj_type") or "").strip(),
        "obj_type": str(note.get("obj_type") or "").strip(),
    }

    evidence_texts = [text for text in [evidence, raw_evidence] if isinstance(text, str) and text.strip()]
    return {
        "evidence": _dedupe_texts(evidence_texts),
        "attribute": [
            text.strip()
            for text in (
                attribute.get("name"),
                attribute.get("normalized_name"),
                attribute.get("value_hint"),
            )
            if isinstance(text, str) and text and text.strip()
        ]
        + summaries,
        "descriptions": descriptions,
        "fields": [value for value in fields.values() if value],
    }


def _truncate(text: str, max_len: Optional[int]) -> str:
    if not max_len or max_len <= 0 or len(text) <= max_len:
        return text
    return text[:max_len].rstrip()


def build_note_text_for_embed(note: Dict[str, Any], *, max_len: Optional[int] = None) -> str:
    """Compact text for embedding/ANN usage."""
    parts = _collect_note_parts(note)
    # Prefer canonical evidence + subj/pred/obj tuple
    ordered: List[str] = []
    if parts["fields"]:
        subj, pred, obj = (parts["fields"] + ["", "", ""])[:3]
        tuple_text = " | ".join(token for token in (subj, pred, obj) if token)
        if tuple_text:
            ordered.append(tuple_text)
    ordered.extend(parts["evidence"])
    ordered.extend(parts["attribute"])
    ordered.extend(parts["descriptions"])
    serialized = " \n".join(chunk for chunk in ordered if chunk)
    return _truncate(serialized.strip(), max_len)


def build_note_text_for_rank(note: Dict[str, Any], *, include_metadata: bool = True, max_len: Optional[int] = None) -> str:
    """Verbose text fed to re-ranker/compressor."""
    parts = _collect_note_parts(note)
    lines: List[str] = []
    subj = (note.get("subj") or "").strip()
    pred = (note.get("pred") or "").strip()
    obj = (note.get("obj") or "").strip()
    if subj or pred or obj:
        lines.append(f"[triple] {subj} --{pred}--> {obj}".strip())
    for ev in parts["evidence"]:
        lines.append(f"[evidence] {ev}")
    for desc in parts["descriptions"]:
        lines.append(f"[profile] {desc}")
    if include_metadata:
        meta = (note.get("meta") or {}) if isinstance(note, dict) else {}
        anchor = meta.get("anchor_entity")
        quality = meta.get("quality_score")
        confidence = meta.get("final_conf")
        attribute = (meta.get("attribute") or {}).get("name")
        extra: List[str] = []
        if attribute:
            extra.append(f"attr={attribute}")
        if anchor:
            extra.append(f"anchor={anchor}")
        if quality is not None:
            extra.append(f"quality={quality}")
        if confidence is not None:
            extra.append(f"conf={confidence}")
        if extra:
            lines.append("[meta] " + ", ".join(extra))
    serialized = "\n".join(lines)
    return _truncate(serialized.strip(), max_len)
