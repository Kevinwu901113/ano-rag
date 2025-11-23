import re
from typing import Dict, List, Optional, Tuple

from config import config
from utils import TextUtils

CLAUSE_SPLIT_RE = re.compile(r"(?<=,|;)\s+|\s+(?=(?:and|but|which)\b)", re.IGNORECASE)
CLAUSE_LEN_THRESHOLD = 200


def _merge_entity_clauses(clauses: List[str]) -> List[str]:
    cleaned = [cl.strip() for cl in clauses if cl and cl.strip()]
    if not cleaned:
        return []
    merged: List[str] = []
    i = 0
    while i < len(cleaned):
        clause = cleaned[i]
        clause_has_entity = bool(TextUtils.extract_entities(clause)) or TextUtils.is_entity_sentence(clause)
        if clause_has_entity:
            left = merged.pop() if merged else ""
            right = ""
            if (i + 1) < len(cleaned):
                nxt = cleaned[i + 1]
                if not (TextUtils.extract_entities(nxt) or TextUtils.is_entity_sentence(nxt)):
                    right = nxt
                    i += 1
            combined_parts = [part for part in (left, clause, right) if part]
            merged.append(" ".join(combined_parts).strip())
        else:
            merged.append(clause)
        i += 1
    return merged


def _split_into_clauses(spans: List[Dict]) -> List[Dict]:
    clause_threshold = int(config.get("chunk.clause_split_len", CLAUSE_LEN_THRESHOLD))
    expanded: List[Dict] = []
    for span in spans:
        text = (span.get("text") or "").strip()
        if len(text) <= clause_threshold:
            expanded.append(span)
            continue
        parts = CLAUSE_SPLIT_RE.split(span["text"])
        if len(parts) <= 1:
            expanded.append(span)
            continue
        merged = _merge_entity_clauses(parts)
        original = span["text"]
        cursor = 0
        for clause in merged:
            if not clause:
                continue
            local_idx = original.find(clause, cursor)
            if local_idx == -1:
                local_idx = cursor
            clause_span = {
                "text": clause.strip(),
                "start": span["start"] + local_idx,
                "end": span["start"] + local_idx + len(clause.strip()),
            }
            expanded.append(clause_span)
            cursor = local_idx + len(clause)
    return expanded


def _cluster_sentences(spans: List[Dict]) -> List[List[Dict]]:
    cluster_max_len = max(3, int(config.get("chunk.cluster_max_len", 5)))
    pronoun_cap = max(3, int(config.get("chunk.pronoun_run_cap", 5)))
    clusters: List[List[Dict]] = []
    pending: List[Dict] = []
    current: List[Dict] = []
    has_entity = False
    for span in spans:
        text = span.get("text") or ""
        is_entity = bool(TextUtils.extract_entities(text)) or TextUtils.is_entity_sentence(text)
        is_pronoun = TextUtils.is_pronoun_at_start(text)

        if is_entity:
            if not has_entity and pending:
                current = pending + [span]
                pending = []
            else:
                if current:
                    clusters.append(current)
                current = [span]
            has_entity = True
        else:
            if not has_entity:
                pending.append(span)
                if len(pending) > pronoun_cap:
                    pending = pending[-pronoun_cap:]
                continue
            current.append(span)
        if has_entity and len(current) >= cluster_max_len:
            clusters.append(current)
            current = []
            has_entity = False

    if current:
        clusters.append(current)
    if pending:
        if clusters:
            clusters[-1].extend(pending)
        else:
            clusters.append(pending)
    return clusters


def _flatten_clusters(clusters: List[List[Dict]]) -> Tuple[List[Dict], Dict[int, int]]:
    flat: List[Dict] = []
    cluster_start: Dict[int, int] = {}
    idx = 0
    for cluster_id, bucket in enumerate(clusters):
        if cluster_id not in cluster_start:
            cluster_start[cluster_id] = idx
        for span in bucket:
            cloned = dict(span)
            cloned["_cluster_id"] = cluster_id
            flat.append(cloned)
            idx += 1
    return flat, cluster_start


def _cluster_start_indices(flat_spans: List[Dict]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for idx, span in enumerate(flat_spans):
        cid = span.get("_cluster_id")
        if cid is None:
            continue
        try:
            cid_int = int(cid)
        except Exception:
            continue
        if cid_int not in mapping:
            mapping[cid_int] = idx
    return mapping


def split_into_entity_aware_spans(text: str) -> List[Dict]:
    """Split text into sentence/clause spans with light entity awareness."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    spans = TextUtils.split_with_spans(cleaned)
    if not spans:
        spans = [{"text": cleaned, "start": 0, "end": len(cleaned)}]
    clause_spans = _split_into_clauses(spans)
    clusters = _cluster_sentences(clause_spans)
    if not clusters:
        clusters = [clause_spans]
    flat_spans, _ = _flatten_clusters(clusters)
    if flat_spans:
        return flat_spans
    fallback: List[Dict] = []
    for span in clause_spans:
        cloned = dict(span)
        cloned["_cluster_id"] = 0
        fallback.append(cloned)
    return fallback


def make_chunks(doc_id: str, text: str, chunk_id_prefix: str = "p", doc_title: Optional[str] = None) -> List[Dict]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    resolved_title = (doc_title or doc_id or "").strip() or doc_id

    flat_spans = split_into_entity_aware_spans(cleaned)
    cluster_start_indices = _cluster_start_indices(flat_spans)

    n_sent = max(1, int(config.get("chunk.n_sent", 4)))
    # Ensure overlap is sentence-based and clamped to 1–2 sentences
    try:
        _overlap_cfg = int(config.get("chunk.overlap", 1))
    except Exception:
        _overlap_cfg = 1
    overlap = max(1, min(2, _overlap_cfg))
    # Optional token/size cap (simple char cap here; could be bytes/tokens)
    max_total_sent = int(config.get("chunk.max_total_sent", 8))

    adaptive_back_limit = max(1, int(config.get("chunk.adaptive_overlap_cap", 3)))
    chunks: List[Dict] = []
    idx = 0
    cursor = 0

    def _unique_entities(sentences: List[str]) -> List[str]:
        uniq: List[str] = []
        for s in sentences:
            for e in TextUtils.extract_entities(s):
                if e not in uniq:
                    uniq.append(e)
        return uniq

    while cursor < len(flat_spans):
        start = cursor
        # Prevent pronoun-leading spans from starting a chunk when possible
        while start > 0 and TextUtils.is_pronoun_at_start(flat_spans[start]["text"]):
            start -= 1

        window: List[Dict] = []
        end = start
        while end < len(flat_spans) and len(window) < n_sent:
            window.append(flat_spans[end])
            end += 1
        # Do not cut within a cluster
        while (
            end < len(flat_spans)
            and len(window) < max_total_sent
            and flat_spans[end]["_cluster_id"] == flat_spans[end - 1]["_cluster_id"]
        ):
            window.append(flat_spans[end])
            end += 1

        # Ensure at least one entity sentence when possible
        if window and not any(TextUtils.is_entity_sentence(s["text"]) for s in window):
            lookahead = end
            while lookahead < len(flat_spans) and len(window) < max_total_sent:
                window.append(flat_spans[lookahead])
                if TextUtils.is_entity_sentence(flat_spans[lookahead]["text"]):
                    lookahead += 1
                    break
                lookahead += 1
            end = lookahead

        sentences = [s["text"] for s in window]
        chunk_text = " ".join(sentences).strip()
        if not chunk_text:
            cursor = end
            continue
        chunk_meta = {
            "sent_spans": [{"text": s["text"], "start": s.get("start"), "end": s.get("end")} for s in window],
            "has_pronoun_lead": TextUtils.is_pronoun_subject_sentence(window[0]["text"]) if window else False,
            "recent_entities": _unique_entities(sentences)[:3],
            "doc_title": resolved_title,
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
        # Adaptive overlap anchored on nearest entity sentence
        anchor_idx = None
        backtrack = 0
        while backtrack < adaptive_back_limit and (end - backtrack - 1) >= start:
            candidate_idx = end - backtrack - 1
            if TextUtils.is_entity_sentence(flat_spans[candidate_idx]["text"]):
                anchor_idx = candidate_idx
                break
            backtrack += 1

        if anchor_idx is not None and anchor_idx > start:
            anchor_cluster = flat_spans[anchor_idx]["_cluster_id"]
            cursor = cluster_start_indices.get(anchor_cluster, anchor_idx)
        else:
            cursor = max(start, end - overlap)
            if cursor < len(flat_spans):
                cluster_id = flat_spans[cursor]["_cluster_id"]
                cursor = cluster_start_indices.get(cluster_id, cursor)
        if cursor <= start:
            cursor = end
        if cursor <= start:
            cursor = start + 1

    if not chunks:
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{chunk_id_prefix}0000",
                "text": cleaned,
                "meta": {
                    "sent_spans": [{"text": cleaned, "start": 0, "end": len(cleaned)}],
                    "has_pronoun_lead": False,
                    "doc_title": resolved_title,
                },
            }
        )

    return chunks
