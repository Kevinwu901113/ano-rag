from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from config import config as config_loader
from schema.vocabulary import normalize_slot_value
from utils.metrics_logger import MetricsLogger
from .bm25_client import BM25Client
from .embedding_client import EmbeddingClient
from .fusion import deduplicate_by_triplet, fuse_rankings
from .note_store import NoteStore
from .rerank import LLMReranker
from .scorer import subject_match
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
        self.hybrid_cfg = retr_cfg.get("hybrid") or {}
        self.rerank_cfg = self.cfg.get("reranker") or {}
        self.final_weights = (self.rerank_cfg.get("final_weights") if self.rerank_cfg else None) or {
            "pre": 0.25,
            "rerank": 0.4,
            "struct": 0.35,
        }
        self.hybrid_weights = self.hybrid_cfg.get("weights") or {
            "bm25": 0.6,
            "embedding": 0.6,
            "structured": 2.5,
            "subject_match": 2.0,
            "source_agree": 1.2,
        }
        self.agreement_threshold = int(self.hybrid_cfg.get("agreement_threshold", 2))
        self.embedding_client = EmbeddingClient(self.embed_cfg) if self.embed_cfg.get("enabled") else None
        self.bm25_client = BM25Client(self.bm25_cfg) if self.bm25_cfg.get("enabled") else None
        self.reranker = LLMReranker(self.rerank_cfg, lm_cfg=self.cfg.get("vllm"))
        self.metrics = MetricsLogger()

    def retrieve(
        self,
        question: str,
        ir,
        intent,
        structured_candidates: List[structured_pipeline.Candidate],
        note_store: NoteStore,
        alias_lookup: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        alias_lookup = alias_lookup or {}
        recall_channels = {}
        timings: Dict[str, float] = {}
        struct_score_map: Dict[str, float] = {}
        for cand in structured_candidates or []:
            for nid in cand.note_ids:
                struct_score_map[nid] = max(struct_score_map.get(nid, 0.0), cand.score)

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
        bm25_scores = {item["note_id"]: item.get("score") for item in recall_channels.get("bm25") or [] if item.get("note_id")}
        emb_scores = {item["note_id"]: item.get("score") for item in recall_channels.get("emb") or [] if item.get("note_id")}
        seed_texts = [seed.text for seed in (ir.seeds or []) if getattr(seed, "text", None)] if ir else []
        for item in pre_candidates:
            note = note_map.get(item["note_id"])
            if not note:
                continue
            struct_score = structured_pipeline._score_note(note, attribute) if attribute else 0.0
            scoring = self._score_candidate(
                note,
                bm25_score=bm25_scores.get(item["note_id"]),
                embed_score=emb_scores.get(item["note_id"]),
                struct_path_score=struct_score_map.get(item["note_id"]),
                alias_lookup=alias_lookup,
                seed_texts=seed_texts,
            )
            if scoring is None:
                continue
            assembled.append(
                {
                    "note_id": item["note_id"],
                    "note": note,
                    "pre_score": item["score"],
                    "struct_score": struct_score,
                    "sources": item.get("sources", {}),
                    "subject_score": scoring["subject_score"],
                    "source_agree": scoring["source_agree"],
                    "struct_path_score": struct_score_map.get(item["note_id"], 0.0),
                    "hybrid_score": scoring["final"],
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
            base_struct = cand.get("struct_score", 0.0)
            cand["final_score"] = (
                self.final_weights.get("pre", 0.3) * cand["pre_score"]
                + self.final_weights.get("rerank", 0.5) * normalized_rerank
                + self.final_weights.get("struct", 0.2) * base_struct
                + cand.get("hybrid_score", 0.0)
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
                "agreement": cand.get("source_agree", 0),
                "subject_score": cand.get("subject_score", 0.0),
            }
            for cand in deduped[:10]
        ]
        best_hybrid = deduped[0] if deduped else {}
        best_meta = {
            "note_id": best_hybrid.get("note_id"),
            "agreement": best_hybrid.get("source_agree", 0),
            "subject_score": best_hybrid.get("subject_score", 0.0),
        }
        consensus = self._aggregate_consensus(structured_candidates, deduped, getattr(intent, "attribute", None))
        best_consensus = consensus[0] if consensus else {}
        if best_consensus:
            best_meta["consensus_label"] = best_consensus.get("label")
            best_meta["consensus_agreement"] = best_consensus.get("agreement", 0)

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
        best_struct = structured_candidates[0] if structured_candidates else None
        meta = {
            "path_consistency": float((best_struct.path_metrics or {}).get("pred_score", 0.0)) if best_struct else 0.0,
            "entity_consistency": float((best_struct.path_metrics or {}).get("entity_score", 0.0)) if best_struct else 0.0,
        }

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
                "best": best_meta,
                "agreement_threshold": self.agreement_threshold,
                "consensus": consensus,
            },
            "meta": meta,
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

    def _aggregate_consensus(
        self,
        structured_candidates: List[structured_pipeline.Candidate],
        deduped_candidates: List[Dict[str, Any]],
        attribute: Optional[str],
    ) -> List[Dict[str, Any]]:
        votes: Dict[str, Dict[str, Any]] = {}

        def _register(label: Optional[str], source: str, note_id: Optional[str]) -> None:
            if not label:
                return
            bucket = votes.setdefault(label, {"sources": set(), "note_ids": []})
            bucket["sources"].add(source)
            if note_id:
                bucket["note_ids"].append(note_id)

        for cand in structured_candidates or []:
            label = self._normalize_label(attribute, cand.answer)
            for nid in cand.note_ids or []:
                _register(label, "struct_path", nid)

        for cand in deduped_candidates or []:
            note = cand.get("note") or {}
            label = self._normalize_label(attribute, structured_pipeline._extract_answer_value(note))
            if not label:
                continue
            source_keys = set((cand.get("sources") or {}).keys())
            if cand.get("struct_path_score"):
                source_keys.add("struct")
            if cand.get("struct_score"):
                source_keys.add("struct_attr")
            if not source_keys:
                source_keys.add("unknown")
            for src in source_keys:
                _register(label, src, cand.get("note_id"))

        consensus: List[Dict[str, Any]] = []
        for label, payload in votes.items():
            sources = sorted(payload["sources"])
            consensus.append(
                {
                    "label": label,
                    "agreement": len(set(sources)),
                    "sources": sources,
                    "support_notes": list(dict.fromkeys(payload["note_ids"])),
                }
            )
        consensus.sort(key=lambda item: (-item["agreement"], item["label"]))
        return consensus

    def _score_candidate(
        self,
        note: Dict[str, Any],
        *,
        bm25_score: Optional[float],
        embed_score: Optional[float],
        struct_path_score: Optional[float],
        alias_lookup: Dict[str, str],
        seed_texts: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        subj_score = subject_match(note.get("subj"), seed_texts, alias_lookup)
        if subj_score <= 0.0:
            return None
        source_agree = int(bm25_score is not None) + int(embed_score is not None) + int(struct_path_score is not None)
        weights = self.hybrid_weights
        final = (
            weights.get("bm25", 1.0) * float(bm25_score or 0.0)
            + weights.get("embedding", 1.0) * float(embed_score or 0.0)
            + weights.get("structured", 1.5) * float(struct_path_score or 0.0)
            + weights.get("subject_match", 2.0) * subj_score
            + weights.get("source_agree", 1.0) * float(source_agree)
        )
        return {"final": final, "subject_score": subj_score, "source_agree": source_agree}

    def _normalize_label(self, attribute: Optional[str], value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if attribute:
            canonical, _ = normalize_slot_value(attribute, text)
            text = canonical or text
        return text.strip() or None


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
