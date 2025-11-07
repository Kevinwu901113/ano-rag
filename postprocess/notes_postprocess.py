from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from utils import TextUtils


def build_alias_map(sentences: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Construct document-level alias_map and alias_to_canonical using heuristics.

    Returns (alias_map, alias_to_canonical).
    alias_map: {canonical_name: [aliases...]}
    alias_to_canonical: {alias_lower: canonical}
    """
    # Collect candidates per surface
    variants: Dict[str, int] = {}
    pairs: List[Tuple[str, str]] = []  # (full, abbr)
    for s in sentences:
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(([A-Z]{2,})\)", s):
            full, abbr = m.group(1), m.group(2)
            pairs.append((full, abbr))
            variants[full] = variants.get(full, 0) + 1
            variants[abbr] = variants.get(abbr, 0) + 1
        for m in re.finditer(r"\b([A-Z]{2,})\s*\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\)", s):
            abbr, full = m.group(1), m.group(2)
            pairs.append((full, abbr))
            variants[full] = variants.get(full, 0) + 1
            variants[abbr] = variants.get(abbr, 0) + 1
        for cand in TextUtils.extract_entity_candidates(s):
            variants[cand] = variants.get(cand, 0) + 1

    # Group by lowercase stripped without common suffix tokens
    def canonicalize(name: str) -> str:
        base = (name or "").strip()
        if not base:
            return base
        # Remove company-like suffix (English simple)
        base = re.sub(r"\b(Inc\.|LLC|Ltd\.|Co\.|Corporation)\b", "", base).strip()
        # Normalize spaces
        base = re.sub(r"\s+", " ", base).strip()
        return base

    buckets: Dict[str, List[str]] = {}
    for var in variants.keys():
        canon = canonicalize(var)
        key = canon.lower()
        buckets.setdefault(key, [])
        if var not in buckets[key]:
            buckets[key].append(var)

    alias_map: Dict[str, List[str]] = {}
    alias_to_canonical: Dict[str, str] = {}
    for key, vals in buckets.items():
        # Choose longest variant as canonical
        canonical = sorted(vals, key=lambda x: len(x), reverse=True)[0]
        aliases = [v for v in vals if v != canonical]
        alias_map[canonical] = aliases
        for a in aliases:
            alias_to_canonical[a.lower()] = canonical
        alias_to_canonical[canonical.lower()] = canonical
        # Include parentheses pairs both ways
        for full, abbr in pairs:
            if canonicalize(full).lower() == key:
                if abbr not in alias_map[canonical]:
                    alias_map[canonical].append(abbr)
                alias_to_canonical[abbr.lower()] = canonical

    return alias_map, alias_to_canonical


def _replace_pronoun_subject(sentence: str, subject: str) -> str:
    s = sentence.strip()
    if not s or not subject:
        return sentence
    # English: ^Pronoun + verb → Subject + verb
    parts = s.split()
    if parts and parts[0].lower() in TextUtils.EN_PRONOUNS:
        return subject + " " + " ".join(parts[1:])
    # Chinese: ^Pronoun + trigger
    m = re.match(rf"^({'|'.join(TextUtils.ZH_PRONOUNS)})", s)
    if m:
        return subject + s[m.end():]
    return sentence


def backfill_pronoun_subjects(chunk: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], Dict[str, str]]:
    """
    For notes within a chunk (usually one or several sentences), if the leading subject is a pronoun,
    backfill it using the nearest entity in the previous m sentences (including overlaps).

    Returns updated notes list, alias_map, alias_to_canonical.
    """
    text = chunk.get("text") or ""
    spans = chunk.get("meta", {}).get("sent_spans") or TextUtils.split_with_spans(text)
    sentences = [s.get("text") for s in spans if isinstance(s, dict)]
    alias_map, alias_to_canonical = build_alias_map(sentences)

    m = 3  # window size for backfill search; could be config-driven
    updated_notes: List[Dict[str, Any]] = []

    # We expect upstream generator to produce notes; here we only prepare meta hints
    # If no notes provided, we return empty list; downstream validator/normalizer will consume these fields if present.
    for idx, s in enumerate(sentences):
        original = s
        is_pronoun_lead = TextUtils.is_pronoun_subject_sentence(s)
        subject = None
        if is_pronoun_lead:
            # search backward up to m sentences for entity candidates
            start = max(0, idx - m)
            back_window = sentences[start:idx]
            candidates: List[str] = []
            for bw in reversed(back_window):
                cands = TextUtils.extract_entity_candidates(bw)
                for c in cands:
                    if c not in candidates:
                        candidates.append(c)
                if candidates:
                    break
            if candidates:
                # map to canonical
                for cand in candidates:
                    canonical = alias_to_canonical.get(cand.lower()) or cand
                    subject = canonical
                    break

        # Record meta backfill info as a synthetic note stub for downstream merging
        note_stub = {
            "note_id": chunk.get("chunk_id") + f"#stub#{idx}",
            "subj": subject or None,
            "pred": "__synthetic__",
            "obj": original,
            "subj_type": "CONCEPT",
            "obj_type": "CONCEPT",
            "evidence": original,
            "meta": {
                "source": chunk.get("doc_id") + "#" + chunk.get("chunk_id"),
                "confidence": 0.8 if subject else 0.0,
                "subject_profile": {"type": "CONCEPT", "aliases": []},
                "attribute": {"name": "__synthetic__", "values": [{"value": original, "normalized": None, "confidence": 0.0, "source": chunk.get("doc_id"), "evidence": original}]},
                "has_unresolved_pronoun": bool(is_pronoun_lead and not subject),
                "original_subject": original.split(" ")[0] if is_pronoun_lead else None,
                "subject_source": "window_backfill" if subject else None,
            },
        }
        updated_notes.append(note_stub)

    return updated_notes, alias_map, alias_to_canonical

