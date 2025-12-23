from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from adapters import mirage as mirage_adapter
from baselines.naive_rag.runner import NaiveIndex
from config import config as config_loader
from utils import TextUtils

from .llm_client import LLMChatClient
from .router import Router
from .structurizer import Structurizer
from .utilizer import Utilizer


class MirageDocStore:
    """Lightweight doc pool accessor for MIRAGE/MuSiQue-style data."""

    def __init__(self, doc_pool_path: str) -> None:
        self.doc_pool_path = Path(doc_pool_path)
        if not self.doc_pool_path.exists():
            raise FileNotFoundError(f"Doc pool not found at {self.doc_pool_path}")
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        records = list(mirage_adapter._load_doc_pool(str(self.doc_pool_path)))  # type: ignore[attr-defined]
        for record in records:
            raw_id = str(
                record.get("id")
                or record.get("doc_id")
                or record.get("_id")
                or record.get("mapped_id")
                or record.get("doc_name")
                or ""
            ).strip()
            if not raw_id:
                continue
            doc_id = f"mirage/{raw_id}"
            title = record.get("title") or record.get("doc_name") or doc_id
            paragraphs = mirage_adapter._paragraphs_from_record(record)  # type: ignore[attr-defined]
            if not paragraphs:
                continue
            text = "\n".join(paragraphs)
            first_sentence = ""
            if paragraphs:
                first_sentence = TextUtils.split_by_sentence(paragraphs[0])[0] if TextUtils.split_by_sentence(paragraphs[0]) else paragraphs[0][:160]
            self._docs[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "paragraphs": paragraphs,
                "content": text,
                "brief": first_sentence,
            }
        logger.info("Loaded {} documents from {}", len(self._docs), self.doc_pool_path)

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._docs.get(doc_id)


class StructRAGBaselineRunner:
    """End-to-end StructRAG-inspired baseline for MIRAGE/MuSiQue datasets."""

    def __init__(
        self,
        *,
        index_path: str,
        chunks_path: str,
        doc_pool_path: str,
        config: Optional[Dict[str, Any]] = None,
        lm_endpoint: Optional[str] = None,
        lm_model: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        structrag_cfg = self.cfg.get("structrag") or {}
        lm_cfg = self.cfg.get("vllm") or {}
        self.top_k = int(top_k or structrag_cfg.get("top_k") or 10)
        supported = structrag_cfg.get("supported_types") or ["chunk", "graph"]
        model_name = lm_model or structrag_cfg.get("llm_model") or lm_cfg.get("model")
        endpoint = lm_endpoint or lm_cfg.get("endpoint")
        if not endpoint or not model_name:
            raise ValueError("vLLM endpoint/model must be provided for StructRAG baseline")

        self.retriever = NaiveIndex(index_path, chunks_path, config=self.cfg)
        self.doc_store = MirageDocStore(doc_pool_path)
        self.llm = LLMChatClient(
            endpoint=endpoint,
            model=model_name,
            llm_profile="generate",
            temperature=float(lm_cfg.get("temperature", 0.0)),
            max_tokens=int(lm_cfg.get("max_tokens", 128)),
        )
        self.router = Router(self.llm, supported_types=supported)
        self.structurizer = Structurizer(self.llm)
        self.utilizer = Utilizer(self.llm)

    def run_dataset(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        work_dir: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        items = list(dataset)
        if limit:
            items = items[:limit]
        out_root = Path(work_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        answers_path = out_root / "answers.jsonl"

        results: List[str] = []
        for idx, item in enumerate(items):
            qid = str(item.get("query_id") or item.get("id") or idx)
            question = str(item.get("query") or item.get("question") or "").strip()
            if not question:
                continue
            logger.info("Processing [{}] {}", qid, question)
            predicted, evidence = self._process_one(question)
            record = {
                "id": qid,
                "question": question,
                "predicted": predicted,
                "evidence": evidence,
            }
            results.append(json.dumps(record, ensure_ascii=False))
        answers_path.write_text("\n".join(results), encoding="utf-8")
        logger.info("StructRAG baseline finished. answers.jsonl -> {}", answers_path)
        return {"answers": str(answers_path)}

    def _process_one(self, question: str) -> Tuple[str, List[str]]:
        doc_candidates = self._retrieve_docs(question)
        structure_type, raw_route = self.router.route(question, doc_candidates)
        structured = self.structurizer.build(structure_type, doc_candidates)
        answer, evidence = self.utilizer.answer(question, structure_type, structured)
        logger.info("Question answered using {} ({} evidences)", structure_type, len(evidence))
        return answer, evidence

    def _retrieve_docs(self, question: str) -> List[Dict]:
        hits = self.retriever.search(question, self.top_k * 2)
        by_doc: Dict[str, Dict[str, Any]] = {}
        for hit in hits:
            doc_id = hit.get("doc_id")
            if not doc_id or doc_id in by_doc:
                continue
            by_doc[doc_id] = {"score": float(hit.get("score") or 0.0), "doc_id": doc_id}
        sorted_docs = sorted(by_doc.values(), key=lambda x: x["score"], reverse=True)[: self.top_k]
        docs: List[Dict] = []
        for item in sorted_docs:
            doc_id = item["doc_id"]
            store_entry = self.doc_store.get(doc_id) or {}
            docs.append(
                {
                    "doc_id": doc_id,
                    "title": store_entry.get("title") or doc_id,
                    "brief": store_entry.get("brief") or store_entry.get("content", "")[:180],
                    "content": store_entry.get("content") or "",
                }
            )
        logger.info("Retriever selected {} docs for structuring", len(docs))
        return docs
