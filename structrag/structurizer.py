from __future__ import annotations

import json
from typing import Dict, List

from loguru import logger

from utils import TextUtils

from .llm_client import LLMChatClient
from .prompts import TRIPLE_EXTRACTION_PROMPT


class Structurizer:
    """Build structured views (chunks or graph triples) from documents."""

    def __init__(
        self,
        llm: LLMChatClient,
        *,
        max_chunks_per_doc: int = 8,
        sentences_per_chunk: int = 2,
        max_chars_per_doc: int = 1600,
    ) -> None:
        self.llm = llm
        self.max_chunks_per_doc = max_chunks_per_doc
        self.sentences_per_chunk = max(1, sentences_per_chunk)
        self.max_chars_per_doc = max_chars_per_doc

    def build(self, structure_type: str, docs: List[Dict]) -> Dict[str, List[Dict]]:
        stype = (structure_type or "chunk").lower()
        if stype == "graph":
            triples = self._build_graph(docs)
            return {"triples": triples}
        chunks = self._build_chunks(docs)
        return {"chunks": chunks}

    def _build_chunks(self, docs: List[Dict]) -> List[Dict]:
        chunks: List[Dict] = []
        for doc in docs:
            content = (doc.get("content") or "")[: self.max_chars_per_doc]
            sentences = TextUtils.split_by_sentence(content)
            doc_id = doc.get("doc_id") or ""
            title = doc.get("title") or doc_id
            if not sentences:
                continue
            idx = 0
            cursor = 0
            while cursor < len(sentences) and idx < self.max_chunks_per_doc:
                group = sentences[cursor : cursor + self.sentences_per_chunk]
                text = " ".join(group).strip()
                if text:
                    chunks.append(
                        {
                            "id": f"{doc_id}#c{idx:02d}",
                            "doc_id": doc_id,
                            "doc_title": title,
                            "text": text,
                        }
                    )
                    idx += 1
                cursor += self.sentences_per_chunk
        logger.info("Structurizer produced {} chunks from {} docs", len(chunks), len(docs))
        return chunks

    def _build_graph(self, docs: List[Dict]) -> List[Dict]:
        triples: List[Dict] = []
        for doc in docs:
            doc_id = doc.get("doc_id") or ""
            title = doc.get("title") or doc_id
            content = (doc.get("content") or "")[: self.max_chars_per_doc]
            prompt = TRIPLE_EXTRACTION_PROMPT.format(title=title, content=content)
            try:
                resp = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.0,
                    llm_profile="extract",
                )
                parsed = self._parse_triples(resp.content)
                for idx, t in enumerate(parsed):
                    triples.append(
                        {
                            "id": f"{doc_id}#t{idx:02d}",
                            "doc_id": doc_id,
                            "doc_title": title,
                            "head": t.get("head"),
                            "relation": t.get("relation"),
                            "tail": t.get("tail"),
                        }
                    )
            except Exception as exc:  # noqa: PERF203
                logger.warning("Triple extraction failed for {}: {}", doc_id, exc)
        logger.info("Structurizer produced {} triples from {} docs", len(triples), len(docs))
        return triples

    def _parse_triples(self, text: str) -> List[Dict]:
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Attempt tolerant parsing
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except Exception:
                    return []
            else:
                return []
        if not isinstance(data, list):
            return []
        triples: List[Dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            head = item.get("head")
            relation = item.get("relation")
            tail = item.get("tail")
            if head and relation and tail:
                triples.append({"head": str(head), "relation": str(relation), "tail": str(tail)})
        return triples
