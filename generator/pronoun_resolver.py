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
    distance = abs(pron["chunk_idx"] - cand["chunk_idx"])
    # Favor closer mentions and higher frequency entities
    distance_score = 1.0 / (1.0 + distance)
    freq_bonus = 0.1 * math.log1p(freq_map.get(cand["subj"], 1))
    same_para_bonus = 0.2 if cand["chunk_idx"] == pron["chunk_idx"] else 0.0
    nearby_order = max(0, 3 - abs(pron["note_idx"] - cand["note_idx"])) * 0.02
    evidence_bonus = 0.0
    pron_entities = [e.lower() for e in (pron.get("entities") or [])]
    if pron_entities and cand["subj"].lower() in pron_entities:
        evidence_bonus = 0.25
    return distance_score + freq_bonus + same_para_bonus + nearby_order + evidence_bonus


def resolve_pronouns_for_doc(doc_id: str, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Doc-level pronoun resolver: backfill pronoun subjects when context exists."""
    if not notes:
        return []

    freq: Dict[str, int] = {}
    subject_pool: List[Dict[str, Any]] = []
    pronoun_notes: List[Dict[str, Any]] = []

    for note in notes:
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
        else:
            meta = note.setdefault("meta", {})
            meta["pronoun_subj"] = True
            pronoun_notes.append(record)

    def _resolve_target(pron: Dict[str, Any]) -> Tuple[str | None, float]:
        if not subject_pool:
            return None, 0.0
        same_type = [cand for cand in subject_pool if cand["subj_type"] and cand["subj_type"] == pron["subj_type"]]
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

    for entry in pronoun_notes:
        resolved, score = _resolve_target(entry)
        note = entry["note"]
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
                    "subj_type": entry["subj_type"],
                    "chunk_idx": entry["chunk_idx"],
                    "note_idx": entry["note_idx"],
                    "entities": TextUtils.extract_entities(note.get("evidence") or ""),
                }
            )
            freq[resolved] = freq.get(resolved, 0) + 1

    resolved_notes: List[Dict[str, Any]] = []
    for note in notes:
        subj = (note.get("subj") or "").strip()
        if TextUtils.is_pronoun(subj):
            continue
        resolved_notes.append(note)
    return resolved_notes
