from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

from config import config
from retriever.note_store import NoteStore
from retriever.operators import Indexes
from retriever.pipeline import retrieve_answer
from utils import TextUtils
from schema.vocabulary import normalize_slot_value
from config.attributes_loader import get_selection_priority, allowed_values
from telemetry.metrics import record_answer_outcome

if TYPE_CHECKING:
    from retriever.hybrid import HybridRetriever


class QueryProcessor:
    """面向结构化索引的最小查询管线。"""

    def __init__(
        self,
        *,
        indexes_dir: Optional[str] = None,
        notes_path: Optional[str] = None,
        lmstudio_endpoint: Optional[str] = None,
        lmstudio_model: Optional[str] = None,
    ) -> None:
        self.cfg = config.load_config()
        cfg = self.cfg
        self.indexes_dir = indexes_dir or cfg.get("notes.indexes_dir", "indexes")
        self.notes_path = notes_path or cfg.get("notes.out_path", "notes/notes.jsonl")

        self.lmstudio_endpoint = lmstudio_endpoint or cfg.get("lmstudio.endpoint")
        self.lmstudio_model = lmstudio_model or cfg.get("lmstudio.model")

        if not self.indexes_dir:
            raise ValueError("indexes_dir must be provided")
        if not Path(self.indexes_dir).exists():
            raise FileNotFoundError(
                f"Indexes directory '{self.indexes_dir}' not found. Run 'main.py process' first."
            )

        required = [
            "entity_to_notes.json",
            "predicate_to_notes.json",
            "type_edge_index.json",
            "graph_edges.jsonl",
            "inverse_edges.jsonl",
            "field_index.json",
            "entity_alias_index.json",
        ]
        missing = [name for name in required if not Path(self.indexes_dir, name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing index files: {missing}. Rebuild notes with 'main.py process'."
            )

        if not self.notes_path:
            raise ValueError("notes_path must be provided")
        if not Path(self.notes_path).exists():
            raise FileNotFoundError(
                f"Notes file '{self.notes_path}' not found. Run 'main.py process' first."
            )

        # 当 CLI 传入 indexes_dir 时，将 embedding/BM25 索引路径同步到该目录，确保混合检索使用对应索引
        retr_cfg = self.cfg.setdefault("retriever", {})
        embed_cfg = retr_cfg.setdefault("embedding", {})
        bm25_cfg = retr_cfg.setdefault("bm25", {})
        if indexes_dir:
            base_idx = Path(self.indexes_dir)
            embed_cfg["offline_index_path"] = str(base_idx / "faiss" / "notes.faiss")
            embed_cfg["meta_path"] = str(base_idx / "faiss" / "notes.meta.parquet")
            bm25_cfg["store_path"] = str(base_idx / "bm25" / "notes")

        self.indexes = Indexes(self.indexes_dir)
        preferred_weak = Path(self.notes_path).parent / "weak" / "weak_notes.jsonl"
        legacy_weak = Path(self.notes_path).with_name("weak_notes.jsonl")
        if preferred_weak.exists():
            weak_path: Optional[str] = str(preferred_weak)
        elif legacy_weak.exists():
            weak_path = str(legacy_weak)
        else:
            weak_path = None
        self.note_store = NoteStore(self.notes_path, weak_path)
        self._hybrid: Optional["HybridRetriever"] = None
        self._hybrid_initialized = False
        retr_cfg = self.cfg.get("retriever") or {}
        structured_cfg = retr_cfg.get("structured") or {}
        hybrid_cfg = retr_cfg.get("hybrid") or {}
        self.path_consistency_threshold = float(structured_cfg.get("path_consistency_threshold", 0.9))
        self.entity_consistency_threshold = float(structured_cfg.get("entity_match_threshold", 0.5))
        self.hybrid_agreement_threshold = int(hybrid_cfg.get("agreement_threshold", 2))

    def process(
        self,
        question: str,
        *,
        doc_hint: Optional[str] = None,
        attribute_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info("Running structured retrieval for question: {}", question)
        structured = retrieve_answer(
            question,
            self.indexes,
            self.note_store,
            cfg=self.cfg,
            hybrid=self._get_hybrid_retriever(),
            doc_hint=doc_hint,
            attribute_hint=attribute_hint,
        )

        evidences = structured.get("evidence", []) or []
        # 当判 "Insufficient evidence" 的自检：
        # 如果候选里 70% 以上是代词 evidence，且存在可用 anchor_entity 的邻近 note，则触发一次轻量回补
        try:
            if evidences:
                pronoun_like = 0
                for ev in evidences:
                    text = (ev.get("evidence") or "")
                    if TextUtils.is_pronoun_subject_sentence(text):
                        pronoun_like += 1
                ratio = pronoun_like / float(len(evidences)) if evidences else 0.0
                if ratio >= 0.7:
                    # 回补：把 lead_in_note_id 的一句并入展示（若存在）
                    augmented = []
                    for ev in evidences:
                        augmented.append(ev)
                        lead = ev.get("lead_in_note_id")
                        if lead:
                            stub = self.note_store.get(lead)
                            if stub:
                                augmented.append({
                                    "note_id": stub.get("note_id"),
                                    "evidence": stub.get("evidence", ""),
                                    "canonical": (stub.get("meta", {}) or {}).get("evidence_canonical") or stub.get("evidence", ""),
                                })
                    evidences = augmented
        except Exception:
            pass

        final_answer, diagnostics, decision = self._select_final_answer(question, structured, evidences)
        return {"structured": structured, "answer": final_answer, "decision": decision, "diagnostics": diagnostics}

    def _get_hybrid_retriever(self) -> Optional["HybridRetriever"]:
        if self._hybrid_initialized:
            return self._hybrid
        self._hybrid_initialized = True
        cfg = self.cfg
        retr_cfg = cfg.get("retriever") or {}
        embedding_on = bool((retr_cfg.get("embedding") or {}).get("enabled"))
        bm25_on = bool((retr_cfg.get("bm25") or {}).get("enabled"))
        rerank_on = bool((cfg.get("reranker") or {}).get("enabled"))
        if not (embedding_on or bm25_on or rerank_on):
            self._hybrid = None
            return None
        try:
            from retriever.hybrid import HybridRetriever

            self._hybrid = HybridRetriever(cfg)
        except Exception as exc:
            logger.warning("Hybrid retriever initialization failed, fallback to structured only: {}", exc)
            self._hybrid = None
        return self._hybrid


    def _select_final_answer(self, question: str, structured: Dict[str, Any], evidences: list[Dict[str, Any]]):
        status = (structured.get("fallback") or {}).get("status")
        intent = structured.get("intent") or {}
        attribute = (intent or {}).get("attribute")
        ans = structured.get("answer")
        paths = structured.get("paths") or []
        support_paths = len(paths)
        ir = structured.get("ir") or {}
        try:
            pred_chain_len = len((ir or {}).get("pred_chain") or [])
        except Exception:
            pred_chain_len = 0
        conf = None
        try:
            if paths and isinstance(paths[0], list) and paths[0]:
                last_nid = paths[0][-1].get("note_id")
                if last_nid:
                    note = self.note_store.get(last_nid)
                    conf = ((note.get("meta", {}) or {}).get("final_conf"))
        except Exception:
            pass
        weak = support_paths <= 1 or (len(evidences) < 3) or (isinstance(conf, (int, float)) and conf < 0.5)

        allowed_label_list = self._ordered_allowed_labels(attribute)
        allowed_label_map = {val.lower(): val for val in allowed_label_list}

        candidate_labels = self._collect_candidate_labels(paths, attribute)
        candidate_labels = self._filter_allowed_labels(candidate_labels, allowed_label_map)
        normalized_ans = self._normalize_answer(attribute, ans)
        if normalized_ans and allowed_label_map and normalized_ans.lower() not in allowed_label_map:
            normalized_ans = None
        if normalized_ans:
            candidate_labels = self._merge_answer(candidate_labels, normalized_ans)

        primary_label = candidate_labels[0] if candidate_labels else normalized_ans
        structured_meta = structured.get("meta") or {}
        path_consistency = float(structured_meta.get("path_consistency", 0.0) or 0.0)
        entity_consistency = float(structured_meta.get("entity_consistency", 0.0) or 0.0)
        decision = {"source": "fallback", "reason": "no_reliable_evidence"}
        hybrid_block = structured.get("hybrid") or {}
        consensus_candidates = hybrid_block.get("consensus") or []
        if allowed_label_map:
            consensus_candidates = [
                c for c in consensus_candidates if allowed_label_map.get((c.get("label") or "").lower())
            ]
        top_consensus = next(
            (
                c
                for c in consensus_candidates
                if int(c.get("agreement") or 0) >= self.hybrid_agreement_threshold
                and (c.get("support_notes") or [])
            ),
            None,
        )
        consensus_label = (top_consensus or {}).get("label")
        consensus_agreement = int((top_consensus or {}).get("agreement") or 0)

        if (
            status == "structured_hit"
            and primary_label
            and path_consistency >= self.path_consistency_threshold
            and entity_consistency >= self.entity_consistency_threshold
        ):
            record_answer_outcome(attribute, "hit")
            decision = {"source": "structured", "reason": "path_consistent_entity_consistent"}
            return primary_label, {
                "weak_evidence": weak,
                "support_paths": support_paths,
                "conf": conf,
                "source": "structured_hit",
                "consensus_agreement": consensus_agreement,
            }, decision

        if consensus_label:
            record_answer_outcome(attribute, "hit")
            decision = {"source": "hybrid", "reason": "multi_source_agreement"}
            return consensus_label, {
                "weak_evidence": weak,
                "support_paths": support_paths,
                "conf": conf,
                "source": "hybrid_hit",
                "consensus_agreement": consensus_agreement,
            }, decision

        # 结构化或兜底已限定在 doc_hint 内时，如果有合规标签，直接使用以避免过度缺证
        if status in {"ok", "weak_index", "structured_fallback"} and primary_label:
            record_answer_outcome(attribute, "hit")
            decision = {"source": "fallback", "reason": "doc_constrained_fallback"}
            return primary_label, {
                "weak_evidence": weak,
                "support_paths": support_paths,
                "conf": conf,
                "source": status,
                "consensus_agreement": consensus_agreement,
            }, decision

        # 若已经有 doc 内支持（路径或证据）且标签在允许集合内，避免过度缺证
        if primary_label and allowed_label_map and (support_paths > 0 or len(evidences) > 0):
            record_answer_outcome(attribute, "hit")
            decision = {"source": "fallback", "reason": "doc_supported_label"}
            return primary_label, {
                "weak_evidence": weak,
                "support_paths": support_paths,
                "conf": conf,
                "source": status or "doc_supported",
                "consensus_agreement": consensus_agreement,
            }, decision

        if status != "structured_hit" and pred_chain_len == 0:
            record_answer_outcome(attribute, "reject")
            return "Insufficient evidence", {
                "weak_evidence": True,
                "support_paths": support_paths,
                "conf": conf,
                "source": "no_chain",
                "consensus_agreement": consensus_agreement,
            }, decision

        record_answer_outcome(attribute, "reject")
        return "Insufficient evidence", {
            "weak_evidence": True,
            "support_paths": support_paths,
            "conf": conf,
            "source": "no_consensus",
            "consensus_agreement": consensus_agreement,
        }, decision

    def _normalize_answer(self, attribute: Optional[str], value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if attribute:
            canonical, _ = normalize_slot_value(attribute, text)
            text = canonical or text
        return text.strip() or None

    def _collect_candidate_labels(self, paths: list, attribute: Optional[str]) -> list[str]:
        labels: list[str] = []
        for path in paths:
            if not isinstance(path, list) or not path:
                continue
            obj = path[-1].get("obj")
            label = self._normalize_answer(attribute, obj)
            if label:
                labels.append(label)
        return self._apply_label_priority(labels, attribute)

    def _apply_label_priority(self, labels: list[str], attribute: Optional[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for label in labels:
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(label)
        priority = get_selection_priority(attribute)
        if not priority:
            return deduped
        priority_map = {val: idx for idx, val in enumerate(priority)}
        deduped.sort(key=lambda lbl: priority_map.get(lbl.lower(), len(priority_map)))
        return deduped

    def _merge_answer(self, labels: list[str], answer: Optional[str]) -> list[str]:
        if not answer:
            return labels
        lowered = [lbl.lower() for lbl in labels]
        if answer.lower() in lowered:
            return labels
        return [answer] + labels

    def _ordered_allowed_labels(self, attribute: Optional[str]) -> list[str]:
        pool = allowed_values(attribute)
        seen: set[str] = set()
        deduped: list[str] = []
        for val in pool:
            if not isinstance(val, str):
                continue
            norm = val.strip()
            if not norm:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(norm)
        return deduped

    def _filter_allowed_labels(self, labels: list[str], allowed_map: dict[str, str]) -> list[str]:
        if not labels or not allowed_map:
            return labels
        filtered: list[str] = []
        seen: set[str] = set()
        for label in labels:
            key = label.lower()
            canonical = allowed_map.get(key)
            if not canonical:
                continue
            if canonical.lower() in seen:
                continue
            seen.add(canonical.lower())
            filtered.append(canonical)
        return filtered

    def _build_allowed_pool(self, candidate_labels: list[str], allowed_labels: list[str]) -> list[str]:
        if allowed_labels:
            pool: list[str] = []
            seen: set[str] = set()
            for label in candidate_labels or []:
                if label.lower() in seen:
                    continue
                pool.append(label)
                seen.add(label.lower())
            for label in allowed_labels:
                lowered = label.lower()
                if lowered in seen:
                    continue
                pool.append(label)
                seen.add(lowered)
            return pool
        return candidate_labels or []
