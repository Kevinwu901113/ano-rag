import math
import re
from typing import Any, Dict, List, Tuple

from utils import TextUtils


MIN_RESOLVE_SCORE = 0.6


def _parse_note_position(note_id: str | None) -> Tuple[int, int]:
    """Extract chunk/order indices from note_id."""
    chunk_idx = 0
    local_idx = 0
    if not note_id:
        return chunk_idx, local_idx
    parts = str(note_id).split("#")
    if len(parts) >= 2:
        m = re.search(r"(\d+)", parts[1])
        if m:
            try:
                chunk_idx = int(m.group(1))
            except Exception:
                chunk_idx = 0
    if len(parts) >= 3:
        try:
            local_idx = int(parts[2])
        except Exception:
            local_idx = 0
    return chunk_idx, local_idx


def _score_candidate(pron: Dict[str, Any], cand: Dict[str, Any], freq_map: Dict[str, int]) -> float:
    distance = abs(pron.get("chunk_idx", 0) - cand.get("chunk_idx", 0))
    # Favor closer mentions and higher frequency entities
    distance_score = 1.0 / (1.0 + distance)
    freq_bonus = 0.1 * math.log1p(freq_map.get(cand["subj"], 1))
    same_para_bonus = 0.2 if cand.get("chunk_idx") == pron.get("chunk_idx") else 0.0
    nearby_order = max(0, 3 - abs(pron.get("note_idx", 0) - cand.get("note_idx", 0))) * 0.02
    evidence_bonus = 0.0
    pron_entities = [e.lower() for e in (pron.get("entities") or [])]
    if pron_entities and cand["subj"].lower() in pron_entities:
        evidence_bonus = 0.25
    return distance_score + freq_bonus + same_para_bonus + nearby_order + evidence_bonus


def resolve_pronouns_for_doc(
    doc_id: str, named_notes: List[Dict[str, Any]], pronoun_notes: List[Dict[str, Any]], doc_meta: Dict[str, Any] | None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Doc-level pronoun resolver: backfill pronoun subjects when context exists."""
    if not named_notes and not pronoun_notes:
        return [], []

    freq: Dict[str, int] = {}
    subject_pool: List[Dict[str, Any]] = []

    all_notes = (named_notes or []) + [entry.get("note") for entry in (pronoun_notes or []) if isinstance(entry, dict)]

    for note in all_notes:
        subj = (note.get("subj") or "").strip()
        subj_type = (note.get("subj_type") or "").strip().upper()
        chunk_idx, note_idx = _parse_note_position(note.get("note_id"))
        evidence_text = (note.get("evidence") or "").strip()
        entities = TextUtils.extract_entities(evidence_text) if evidence_text else []
        record = {
            "note": note,
            "subj": subj,
            "subj_type": subj_type,
            "chunk_idx": chunk_idx,
            "note_idx": note_idx,
            "entities": entities,
        }
        if subj and not TextUtils.is_pronoun(subj):
            subject_pool.append(record)
            freq[subj] = freq.get(subj, 0) + 1
        elif record not in subject_pool:
            meta = note.setdefault("meta", {})
            meta["pronoun_subj"] = True

    def _resolve_target(pron: Dict[str, Any]) -> Tuple[str | None, float]:
        if not subject_pool:
            return None, 0.0
        same_type = [cand for cand in subject_pool if cand.get("subj_type") and cand.get("subj_type") == pron.get("subj_type")]
        candidates = same_type or subject_pool
        best_subj = None
        best_score = 0.0
        for cand in candidates:
            score = _score_candidate(pron, cand, freq)
            if score > best_score:
                best_score = score
                best_subj = cand["subj"]
        if best_score < MIN_RESOLVE_SCORE:
            return None, best_score
        return best_subj, best_score

    unresolved: List[Dict[str, Any]] = []
    for entry in pronoun_notes or []:
        record_note = entry.get("note") if isinstance(entry, dict) else None
        if not record_note:
            continue
        chunk_idx, note_idx = _parse_note_position(record_note.get("note_id"))
        pron_record = {
            "note": record_note,
            "subj_type": entry.get("subj_type") or record_note.get("subj_type"),
            "chunk_idx": entry.get("chunk_idx", chunk_idx),
            "note_idx": entry.get("note_idx", note_idx),
            "entities": entry.get("entities") or TextUtils.extract_entities(record_note.get("evidence") or ""),
        }
        resolved, score = _resolve_target(pron_record)
        note = record_note
        if resolved:
            note["subj"] = resolved
            meta = note.setdefault("meta", {})
            meta["subject_source"] = meta.get("subject_source") or "doc_pronoun_resolver"
            meta["coref_confidence"] = max(float(meta.get("coref_confidence") or 0.0), round(score, 3))
            # Feed resolved subject back for downstream pronoun resolution
            subject_pool.append(
                {
                    "note": note,
                    "subj": resolved,
                    "subj_type": pron_record.get("subj_type"),
                    "chunk_idx": pron_record.get("chunk_idx"),
                    "note_idx": pron_record.get("note_idx"),
                    "entities": TextUtils.extract_entities(note.get("evidence") or ""),
                }
            )
            freq[resolved] = freq.get(resolved, 0) + 1
        else:
            meta = note.setdefault("meta", {})
            meta["has_unresolved_pronoun"] = True
            unresolved.append(note)

    resolved_notes: List[Dict[str, Any]] = []
    for note in all_notes:
        subj = (note.get("subj") or "").strip()
        if TextUtils.is_pronoun(subj):
            continue
        resolved_notes.append(note)
    return resolved_notes, unresolved
