from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from utils import TextUtils

SUBJECT_TYPE_HINTS = {
    "born_in": "PERSON",
    "born_on": "PERSON",
    "birth_place": "PERSON",
    "died_in": "PERSON",
    "died_on": "PERSON",
    "married": "PERSON",
    "spouse": "PERSON",
    "spouse_of": "PERSON",
    "parent": "PERSON",
    "served_as": "PERSON",
    "served_in": "PERSON",
    "served_with": "PERSON",
    "joined": "PERSON",
    "member_of": "PERSON",
    "appointed": "PERSON",
    "elected": "PERSON",
}

OBJECT_TYPE_HINTS = {
    "born_in": "PLACE",
    "birth_place": "PLACE",
    "died_in": "PLACE",
    "married": "PERSON",
    "spouse": "PERSON",
    "spouse_of": "PERSON",
    "joined": "ORG",
    "member_of": "ORG",
    "served_in": "ORG",
    "served_with": "ORG",
    "appointed": "ORG",
    "elected": "ORG",
}


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
    s = (sentence or "").strip()
    subj = (subject or "").strip()
    if not s or not subj:
        return sentence
    out = s

    possessives = {"his", "her", "its", "it's", "it’s", "their"}

    def _replace_leading_english(match: re.Match) -> str:
        prefix = match.group("prefix") or ""
        pron = (match.group("pron") or "").lower().replace("’", "'")
        replacement = subj + "'s" if pron in possessives else subj
        return f"{prefix}{replacement}"

    lead_pattern = re.compile(
        r"^(?P<prefix>[\s\"'“”‘’\(\)\[\]]*)(?P<pron>he|she|it|they|him|her|them|his|its|it's|it’s|their)\b",
        flags=re.I,
    )
    out, replaced = lead_pattern.subn(_replace_leading_english, out, count=1)

    if not replaced:
        zh_prefix = "|".join(re.escape(p) for p in TextUtils.ZH_PRONOUNS)
        zh_pattern = re.compile(rf"^({zh_prefix})")
        out = zh_pattern.sub(subj, out, count=1)

    def _replace_clause(match: re.Match) -> str:
        prefix = match.group(1)
        pron = (match.group(2) or "").lower().replace("’", "'")
        replacement = subj + "'s" if pron in possessives else subj
        return f"{prefix}{replacement}"

    out = re.sub(r"([.;:,]\s+)(he|she|it|they)\b", _replace_clause, out, flags=re.I)
    out = re.sub(r"([.;:,]\s+)(his|her|its|it's|it’s|their)\b", _replace_clause, out, flags=re.I)

    # Possessive at bare sentence start (e.g., "His book" / "Its legacy")
    out = re.sub(r"^(his|her|its|it's|it’s|their)\b", lambda m: subj + "'s", out, flags=re.I)
    return out


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
    # Lightweight entity stack within this chunk: push on new entities, reset on apparent subject change
    entity_stack: List[Tuple[str, int]] = []
    updated_notes: List[Dict[str, Any]] = []

    # We expect upstream generator to produce notes; here we only prepare meta hints
    # If no notes provided, we return empty list; downstream validator/normalizer will consume these fields if present.
    for idx, s in enumerate(sentences):
        original = s
        is_pronoun_lead = TextUtils.is_pronoun_subject_sentence(s)
        subject = None
        coref_candidates: List[Dict[str, Any]] = []
        # Heuristic: detect new full-name entity and push stack
        entities = TextUtils.extract_entity_candidates(s)
        # If we see a capitalized multi-word and it's not top of stack, push
        for e in entities:
            if not entity_stack or entity_stack[-1][0] != e:
                entity_stack.append((e, idx))
                break
        # Reset stack on strong subject change: if sentence contains an entity different from top and looks like a title line
        if entities and entity_stack and entities[0] != entity_stack[-1][0]:
            # crude title line check: all caps words or colon
            if any(tok.isupper() and len(tok) >= 2 for tok in s.split()) or (":" in s):
                entity_stack = [(entities[0], idx)]
        if is_pronoun_lead:
            # search backward up to m sentences for entity candidates
            start = max(0, idx - m)
            back_window = sentences[start:idx]
            candidate_map: Dict[str, Dict[str, Any]] = {}

            def _score_candidate(distance: int, alias_hit: bool, entity: str) -> float:
                dist_component = max(0.0, 1.0 - (max(0, distance - 1) / max(1, m)))
                alias_component = 1.0 if alias_hit else 0.0
                type_component = 1.0 if TextUtils.guess_entity_type(entity) else 0.0
                raw = 0.6 * dist_component + 0.3 * alias_component + 0.1 * type_component
                return max(0.0, min(raw, 1.0))

            def _register_candidate(name: str, distance: int, reason: str, alias_hit: bool) -> None:
                if not name:
                    return
                key = name.lower()
                scored = _score_candidate(distance, alias_hit, name)
                entry = {
                    "entity": name,
                    "score": round(scored, 3),
                    "reason": reason,
                    "distance": distance,
                }
                current = candidate_map.get(key)
                if current is None or entry["score"] > current.get("score", 0.0):
                    candidate_map[key] = entry

            for distance, bw in enumerate(reversed(back_window), start=1):
                cands = TextUtils.extract_entity_candidates(bw)
                for c in cands:
                    canon = alias_to_canonical.get(c.lower()) or c
                    alias_hit = bool(alias_to_canonical.get(c.lower()))
                    _register_candidate(canon, distance, f"window_backfill_d{distance}", alias_hit)

            # Stack-based fallback: use top entity if available
            if entity_stack:
                stack_entity, stack_idx = entity_stack[-1]
                stack_distance = max(1, idx - stack_idx)
                _register_candidate(stack_entity, stack_distance, "entity_stack", False)

            coref_candidates = sorted(candidate_map.values(), key=lambda item: item["score"], reverse=True)
            subject = coref_candidates[0]["entity"] if coref_candidates else None

        # Apply inline backfill for evidence canonical if we have subject
        evidence_canonical = original
        if subject:
            evidence_canonical = _replace_pronoun_subject(original, subject)

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
                "coref_confidence": 0.8 if subject else 0.0,
                "subject_profile": {"type": "CONCEPT", "aliases": []},
                "attribute": {"name": "__synthetic__", "values": [{"value": original, "normalized": None, "confidence": 0.0, "source": chunk.get("doc_id"), "evidence": original}]},
                "has_unresolved_pronoun": bool(is_pronoun_lead and not subject),
                "original_subject": original.split(" ")[0] if is_pronoun_lead else None,
                "subject_source": "window_backfill" if subject else None,
                "resolved_subject": subject,
                "evidence_canonical": evidence_canonical if subject else None,
                "coref_candidates": coref_candidates,
            },
        }
        updated_notes.append(note_stub)

    return updated_notes, alias_map, alias_to_canonical


def stitch_pronoun_notes(notes: List[Dict[str, Any]], chunk: Dict[str, Any], max_lookback: int = 3) -> List[Dict[str, Any]]:
    if not notes:
        return notes
    spans = (chunk.get("meta") or {}).get("sent_spans") or TextUtils.split_with_spans(chunk.get("text") or "")
    sentences = [s.get("text") for s in spans if isinstance(s, dict)]
    if not sentences:
        return notes

    for note in notes:
        subj = (note.get("subj") or "").strip()
        if not subj or not TextUtils.is_pronoun(subj):
            continue
        meta = note.get("meta") or {}
        pred = (note.get("pred") or "").lower()
        target_type = SUBJECT_TYPE_HINTS.get(pred)
        anchor_idx = _locate_sentence_index(sentences, note.get("evidence") or "")
        candidate = _find_recent_entity(sentences, anchor_idx, max_lookback, target_type)
        if not candidate:
            continue
        meta.setdefault("original_subject", subj)
        note["subj"] = candidate
        meta["subject_source"] = meta.get("subject_source") or "stitcher_backfill"
        meta["subject_confidence"] = max(meta.get("subject_confidence", 0.0), 0.65)
        meta["coref_confidence"] = max(meta.get("coref_confidence", 0.0), 0.65)
        meta.pop("filter_out_strict", None)
        violations = meta.get("violations")
        if isinstance(violations, dict):
            violations.pop("coref_unresolved", None)
            if not violations:
                meta.pop("violations")
        canonical_target = meta.get("evidence_canonical") or note.get("evidence", "")
        meta["evidence_canonical"] = _replace_pronoun_subject(canonical_target or note.get("evidence", ""), candidate)
        note["meta"] = meta
    return notes


def _locate_sentence_index(sentences: List[str], evidence: str) -> int:
    if not sentences:
        return 0
    for idx, sent in enumerate(sentences):
        if sent and sent in evidence:
            return idx
    return len(sentences) - 1


def _find_recent_entity(
    sentences: List[str],
    anchor_idx: int,
    max_lookback: int,
    target_type: Optional[str],
) -> Optional[str]:
    if not sentences:
        return None
    start = max(0, anchor_idx)
    for idx in range(start, max(-1, start - max_lookback - 1), -1):
        clause = sentences[idx]
        if not clause:
            continue
        candidates = TextUtils.extract_entity_candidates(clause)
        for cand in candidates:
            etype = TextUtils.guess_entity_type(cand) or target_type
            if target_type and etype and etype != target_type:
                continue
            if cand:
                return cand
    return None
