from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

from config import config
from generator.answerer import call_lmstudio
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

        final_answer, diagnostics = self._select_final_answer(question, structured, evidences)
        return {"structured": structured, "answer": final_answer, "diagnostics": diagnostics}

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

        candidate_labels = self._collect_candidate_labels(paths, attribute)
        normalized_ans = self._normalize_answer(attribute, ans)
        if normalized_ans:
            candidate_labels = self._merge_answer(candidate_labels, normalized_ans)

        primary_label = candidate_labels[0] if candidate_labels else normalized_ans

        if status == "structured_hit" and primary_label:
            record_answer_outcome(attribute, "hit")
            return primary_label, {
                "weak_evidence": weak,
                "support_paths": support_paths,
                "conf": conf,
                "source": "structured_hit",
            }

        if primary_label:
            source = "structured_weak" if normalized_ans else "candidate_path"
            record_answer_outcome(attribute, "hit")
            return primary_label, {
                "weak_evidence": weak,
                "support_paths": support_paths,
                "conf": conf,
                "source": source,
            }

        if status != "structured_hit" and pred_chain_len == 0:
            record_answer_outcome(attribute, "reject")
            return "Insufficient evidence", {
                "weak_evidence": True,
                "support_paths": support_paths,
                "conf": conf,
                "source": "no_chain",
            }

        # LM 兜底：限定可选标签，仍然不输出解释
        if self.lmstudio_endpoint and self.lmstudio_model and evidences:
            allowed_pool = candidate_labels or allowed_values(attribute)
            try:
                lm_answer = call_lmstudio(
                    self.lmstudio_endpoint,
                    self.lmstudio_model,
                    question,
                    evidences,
                    allowed_labels=allowed_pool,
                    attribute_name=attribute,
                )
                normalized_lm = self._normalize_answer(attribute, lm_answer)
                if normalized_lm:
                    record_answer_outcome(attribute, "hit")
                    return normalized_lm, {
                        "weak_evidence": True,
                        "support_paths": support_paths,
                        "conf": conf,
                        "source": "lmstudio",
                    }
            except Exception:
                record_answer_outcome(attribute, "reject")
                return "Insufficient evidence", {
                    "weak_evidence": True,
                    "support_paths": support_paths,
                    "conf": conf,
                    "source": "lm_error",
                }

        record_answer_outcome(attribute, "reject")
        return "Insufficient evidence", {
            "weak_evidence": True,
            "support_paths": support_paths,
            "conf": conf,
            "source": "no_answer",
        }

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
