from __future__ import annotations

import re
from hashlib import sha1
from typing import Any, Dict, List, Optional


_DOC_BLOCK_RE = re.compile(
    r"### DOC\s+([^\s|]+)(?:\s+\|\s+Title:\s*([^\n]+))?\s*",
    flags=re.MULTILINE,
)


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _text_key(value: str) -> str:
    return sha1(str(value or "").encode("utf-8")).hexdigest()


def parse_doc_blocks(context_text: str) -> List[Dict[str, str]]:
    text = str(context_text or "")
    matches = list(_DOC_BLOCK_RE.finditer(text))
    if not matches:
        return []

    rows: List[Dict[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block_text = text[start:end].strip()
        doc_id = _norm_text(match.group(1))
        title = _norm_text(match.group(2) or "")
        rows.append(
            {
                "id": doc_id,
                "title": title,
                "text": block_text,
            }
        )
    return rows


def build_raptor_ctxs(
    *,
    context_text: str,
    docs: List[Dict[str, Any]],
    top_k: int,
    layer_information: Optional[List[Dict[str, Any]]] = None,
    node_text_by_index: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    max_k = max(0, int(top_k))
    if max_k == 0:
        return []

    docs_by_id: Dict[str, Dict[str, Any]] = {}
    docs_by_title: Dict[str, Dict[str, Any]] = {}
    for row in docs:
        doc_id = _norm_text(row.get("id"))
        title = _norm_text(row.get("title"))
        if doc_id:
            docs_by_id[doc_id] = row
        if title:
            docs_by_title[title.lower()] = row

    ctxs: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add_block(block: Dict[str, str], provenance: Dict[str, Any]) -> None:
        if len(ctxs) >= max_k:
            return

        block_id = _norm_text(block.get("id"))
        block_title = _norm_text(block.get("title"))
        block_text = str(block.get("text") or "").strip()

        resolved = None
        if block_id:
            resolved = docs_by_id.get(block_id)
        if resolved is None and block_title:
            resolved = docs_by_title.get(block_title.lower())

        doc_id = _norm_text((resolved or {}).get("id") or block_id)
        title = _norm_text((resolved or {}).get("title") or block_title)
        text = str((resolved or {}).get("text") or block_text).strip()

        if not text:
            return

        key = doc_id or f"{title.lower()}::{_text_key(text)}"
        if key in seen:
            return
        seen.add(key)

        ctxs.append(
            {
                "id": doc_id or f"raptor_ctx_{len(ctxs) + 1:04d}",
                "title": title,
                "text": text,
                "rank": len(ctxs) + 1,
                "provenance": provenance,
            }
        )

    if layer_information and node_text_by_index:
        for item in layer_information:
            node_index_raw = item.get("node_index")
            layer_number_raw = item.get("layer_number")
            try:
                node_index = int(node_index_raw)
            except (TypeError, ValueError):
                continue
            node_text = str(node_text_by_index.get(node_index) or "")
            if not node_text:
                continue
            for block in parse_doc_blocks(node_text):
                _add_block(
                    block,
                    provenance={
                        "node_index": node_index,
                        "layer_number": layer_number_raw,
                    },
                )
                if len(ctxs) >= max_k:
                    return ctxs

    for block in parse_doc_blocks(context_text):
        _add_block(block, provenance={"source": "context_dump"})
        if len(ctxs) >= max_k:
            return ctxs

    if not ctxs and str(context_text or "").strip():
        ctxs.append(
            {
                "id": "raptor_context_0001",
                "title": "",
                "text": str(context_text).strip(),
                "rank": 1,
                "provenance": {"source": "context_fallback"},
            }
        )

    return ctxs


def build_graphrag_ctxs_from_sources(
    *,
    sources_records: List[Dict[str, Any]],
    text_unit_records: List[Dict[str, Any]],
    document_records: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    max_k = max(0, int(top_k))
    if max_k == 0:
        return []

    text_unit_by_id: Dict[str, Dict[str, Any]] = {}
    text_unit_by_hrid: Dict[str, Dict[str, Any]] = {}
    for row in text_unit_records:
        tid = _norm_text(row.get("id"))
        hrid = row.get("human_readable_id")
        hrid_key = _norm_text(hrid)
        if tid:
            text_unit_by_id[tid] = row
        if hrid_key:
            text_unit_by_hrid[hrid_key] = row

    doc_by_id: Dict[str, Dict[str, Any]] = {}
    for row in document_records:
        did = _norm_text(row.get("id"))
        if did:
            doc_by_id[did] = row

    ctxs: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for idx, source in enumerate(sources_records):
        if len(ctxs) >= max_k:
            break

        source_id = _norm_text(source.get("id"))
        source_text = str(source.get("text") or "").strip()

        text_unit = text_unit_by_hrid.get(source_id) or text_unit_by_id.get(source_id)
        text_unit_id = _norm_text((text_unit or {}).get("id"))
        document_id = _norm_text((text_unit or {}).get("document_id"))
        document = doc_by_id.get(document_id, {})

        title = _norm_text(document.get("title"))
        text = str(source_text or document.get("text") or (text_unit or {}).get("text") or "").strip()
        if not text:
            continue

        ctx_id = (
            document_id
            or text_unit_id
            or source_id
            or f"graphrag_source_{idx + 1:04d}"
        )
        key = f"{ctx_id}::{title.lower()}"
        if key in seen:
            continue
        seen.add(key)

        ctxs.append(
            {
                "id": ctx_id,
                "title": title,
                "text": text,
                "rank": len(ctxs) + 1,
                "provenance": {
                    "source_id": source_id,
                    "text_unit_id": text_unit_id or None,
                    "document_id": document_id or None,
                },
            }
        )

    return ctxs
