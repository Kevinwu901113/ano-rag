from typing import Dict, List

from config import config
from utils import TextUtils


def make_chunks(doc_id: str, text: str, chunk_id_prefix: str = "p") -> List[Dict]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    spans = TextUtils.split_with_spans(cleaned)
    if not spans:
        spans = [{"text": cleaned, "start": 0, "end": len(cleaned)}]

    n_sent = max(1, int(config.get("chunk.n_sent", 4)))
    # Ensure overlap is sentence-based and clamped to 1–2 sentences
    try:
        _overlap_cfg = int(config.get("chunk.overlap", 1))
    except Exception:
        _overlap_cfg = 1
    overlap = max(1, min(2, _overlap_cfg))
    # Sliding step
    step = max(1, n_sent - overlap)
    # Optional token/size cap (simple char cap here; could be bytes/tokens)
    max_total_sent = int(config.get("chunk.max_total_sent", 8))

    chunks: List[Dict] = []
    idx = 0
    cursor = 0

    def _unique_entities(sentences: List[str]) -> List[str]:
        uniq: List[str] = []
        for s in sentences:
            for e in TextUtils.extract_entity_candidates(s):
                if e not in uniq:
                    uniq.append(e)
        return uniq

    while cursor < len(spans):
        # Base window
        window = spans[cursor : cursor + n_sent]
        if not window:
            break

        # Pronoun-aware front merge: if first sentence is pronoun-subject, merge tail of previous
        # up to min(overlap,2) sentences, and continue merging backward until we hit an entity sentence
        # or reach window cap.
        if TextUtils.is_pronoun_subject_sentence(window[0]["text"]) and cursor > 0:
            back = []
            # take min(overlap,2) from previous window tail
            take = min(overlap, 2)
            start_back = max(0, cursor - take)
            back = spans[start_back:cursor]
            merged = back + window
            # If still pronoun subject, keep extending backward until entity sentence or cap
            back_cursor = start_back - 1
            while TextUtils.is_pronoun_subject_sentence(merged[0]["text"]) and back_cursor >= 0 and len(merged) < max_total_sent:
                merged = [spans[back_cursor]] + merged
                if TextUtils.is_entity_sentence(merged[0]["text"]):
                    break
                back_cursor -= 1
            # 伪规则：若上一窗口（或其末句）仅包含唯一实体名，则前置其末句
            if len(merged) > 0 and TextUtils.is_pronoun_subject_sentence(merged[0]["text"]) and cursor > 0:
                prev_block = spans[max(0, cursor - n_sent):cursor]
                prev_sents = [s["text"] for s in prev_block] if prev_block else []
                ents = _unique_entities(prev_sents)
                if len(ents) == 1 and prev_block:
                    # 前置上一块的最后一句（最多1句）
                    merged = [prev_block[-1]] + merged
            window = merged

        # 最小上下文保障：若窗口内出现代词 he/she/they 等，但窗口中没有任何实体句，则强制回拉上一句
        if window:
            has_pronoun_any = any(TextUtils.is_pronoun_subject_sentence(s["text"]) or any(p in s["text"].lower() for p in (" he ", " she ", " they ", " his ", " her ", " their ")) for s in window)
            has_entity_any = any(TextUtils.is_entity_sentence(s["text"]) for s in window)
            if has_pronoun_any and not has_entity_any and cursor > 0:
                prev_idx = cursor - 1
                prev_sent = spans[prev_idx]
                window = [prev_sent] + window

        # Assemble text and metadata
        chunk_text = " ".join([s["text"] for s in window]).strip()
        if not chunk_text:
            cursor += step
            continue
        chunk_meta = {
            "sent_spans": [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in window],
            "has_pronoun_lead": TextUtils.is_pronoun_subject_sentence(window[0]["text"]) if window else False,
        }
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{chunk_id_prefix}{idx:04d}",
                "text": chunk_text,
                "meta": chunk_meta,
            }
        )
        idx += 1
        cursor += step

    if not chunks:
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{chunk_id_prefix}0000",
                "text": cleaned,
                "meta": {"sent_spans": [{"text": cleaned, "start": 0, "end": len(cleaned)}], "has_pronoun_lead": False},
            }
        )

    return chunks
