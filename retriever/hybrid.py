from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from config import config as config_loader
from utils.metrics_logger import MetricsLogger
from .bm25_client import BM25Client
from .embedding_client import EmbeddingClient
from .fusion import deduplicate_by_triplet, fuse_rankings
from .note_store import NoteStore
from .rerank import LLMReranker
from . import pipeline as structured_pipeline


class HybridRetriever:
    """Coordinate structured, embedding, and BM25 recalls."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or config_loader.load_config()
        retr_cfg = self.cfg.get("retriever") or {}
        self.struct_cfg = retr_cfg.get("structured", {"enabled": True})
        self.embed_cfg = retr_cfg.get("embedding") or {}
        self.bm25_cfg = retr_cfg.get("bm25") or {}
        self.fusion_cfg = retr_cfg.get("fusion") or {"method": "rrf", "rrf_k": 60, "weights": {}}
        self.rerank_cfg = self.cfg.get("reranker") or {}
        self.final_weights = (self.rerank_cfg.get("final_weights") if self.rerank_cfg else None) or {
            "pre": 0.3,
            "rerank": 0.5,
            "struct": 0.2,
        }
        self.embedding_client = EmbeddingClient(self.embed_cfg) if self.embed_cfg.get("enabled") else None
        self.bm25_client = BM25Client(self.bm25_cfg) if self.bm25_cfg.get("enabled") else None
        self.reranker = LLMReranker(self.rerank_cfg, lm_cfg=self.cfg.get("lmstudio"))
        self.metrics = MetricsLogger()

    def retrieve(
        self,
        question: str,
        ir,
        intent,
        structured_candidates: List[structured_pipeline.Candidate],
        note_store: NoteStore,
    ) -> Optional[Dict[str, Any]]:
        recall_channels = {}
        timings: Dict[str, float] = {}

        if structured_candidates:
            recall_channels["struct"] = self._structured_results(structured_candidates)
        else:
            recall_channels["struct"] = []

        if self.embedding_client is not None:
            start = time.time()
            recall_channels["emb"] = self.embedding_client.search(question, int(self.embed_cfg.get("topn", 200)))
            timings["faiss_ms"] = (time.time() - start) * 1000
        else:
            recall_channels["emb"] = []

        if self.bm25_client is not None:
            start = time.time()
            recall_channels["bm25"] = self.bm25_client.search(question, int(self.bm25_cfg.get("topn", 200)))
            timings["bm25_ms"] = (time.time() - start) * 1000
        else:
            recall_channels["bm25"] = []

        if not any(recall_channels.values()):
            return None

        start = time.time()
        fused = fuse_rankings(
            recall_channels,
            weights=self.fusion_cfg.get("weights", {"struct": 1.0, "emb": 1.0, "bm25": 1.0}),
            rrf_k=int(self.fusion_cfg.get("rrf_k", 60)),
        )
        timings["fusion_ms"] = (time.time() - start) * 1000
        if not fused:
            return None

        pre_top_m = int(self.fusion_cfg.get("pre_topM", 128))
        pre_candidates = fused[:pre_top_m]
        note_ids = [item["note_id"] for item in pre_candidates]
        notes = note_store.get_many(note_ids)
        note_map = {note.get("note_id"): note for note in notes}
        assembled = []
        attribute = getattr(intent, "attribute", None)
        for item in pre_candidates:
            note = note_map.get(item["note_id"])
            if not note:
                continue
            struct_score = structured_pipeline._score_note(note, attribute) if attribute else 0.0
            assembled.append(
                {
                    "note_id": item["note_id"],
                    "note": note,
                    "pre_score": item["score"],
                    "struct_score": struct_score,
                    "sources": item.get("sources", {}),
                }
            )
        if not assembled:
            return None

        start = time.time()
        rerank_scores = self.reranker.score(question, assembled) if self.reranker.enabled else {}
        timings["rerank_ms"] = (time.time() - start) * 1000

        for cand in assembled:
            note_id = cand["note_id"]
            rerank_entry = rerank_scores.get(note_id, {})
            rerank_score = rerank_entry.get("score", 0.0)
            normalized_rerank = rerank_score / 100.0
            cand["rerank_score"] = normalized_rerank
            cand["labels"] = rerank_entry.get("labels", [])
            cand["final_score"] = (
                self.final_weights.get("pre", 0.3) * cand["pre_score"]
                + self.final_weights.get("rerank", 0.5) * normalized_rerank
                + self.final_weights.get("struct", 0.2) * cand["struct_score"]
            )

        deduped = deduplicate_by_triplet(assembled, topk=int(self.fusion_cfg.get("final_topK", 64)))
        if not deduped:
            return None

        support_note_ids = [cand["note_id"] for cand in deduped]
        evidences = structured_pipeline._schedule_evidences(
            note_store,
            support_note_ids,
            keep_at_least=max(3, getattr(ir, "fanout", 4) // 2),
        )

        hybrid_info = [
            {
                "note_id": cand["note_id"],
                "pre_rrf": cand["pre_score"],
                "llm_score": cand.get("rerank_score", 0.0),
                "struct_score": cand.get("struct_score", 0.0),
                "final_score": cand.get("final_score", 0.0),
                "labels": cand.get("labels", []),
            }
            for cand in deduped[:10]
        ]

        self.metrics.log_query(
            question,
            top1_source=_top1_source(recall_channels),
            timings=timings,
            scores={"pre_rrf": hybrid_info[0]["pre_rrf"] if hybrid_info else 0.0},
            notes=support_note_ids[:5],
        )

        paths = [cand.path for cand in structured_candidates[: getattr(ir, "fanout", 4)]]
        answer = structured_candidates[0].answer if structured_candidates else None

        fallback_used = not structured_candidates
        status = "structured_hit" if structured_candidates else "hybrid_hit"

        return {
            "ir": ir.to_dict() if ir else None,
            "answer": answer,
            "paths": paths,
            "support_note_ids": support_note_ids,
            "evidence": evidences,
            "reason": None,
            "fallback": {
                "used": fallback_used,
                "stage": None if structured_candidates else "hybrid",
                "status": status,
                "intent": intent.to_dict() if intent else None,
            },
            "intent": intent.to_dict() if intent else None,
            "hybrid": {
                "pre_candidates": pre_candidates,
                "final": hybrid_info,
            },
        }

    @staticmethod
    def _structured_results(candidates: List[structured_pipeline.Candidate]) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        seen = set()
        for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
            for note_id in cand.note_ids:
                if not note_id or note_id in seen:
                    continue
                ranked.append({"note_id": note_id, "score": cand.score, "rank": len(ranked) + 1, "source": "struct"})
                seen.add(note_id)
        return ranked


def _top1_source(channels: Dict[str, List[Dict[str, Any]]]) -> str:
    best_source = "unknown"
    best_score = -1.0
    for source, items in channels.items():
        if not items:
            continue
        score = items[0].get("score", 0.0)
        if score > best_score:
            best_source = source
            best_score = score
    return best_source
