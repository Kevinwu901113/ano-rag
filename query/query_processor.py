from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from config import config
from generator.answerer import call_lmstudio
from retriever.note_store import NoteStore
from retriever.operators import Indexes
from retriever.pipeline import retrieve_answer
from utils import TextUtils


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
        cfg = config.load_config()
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
        self.note_store = NoteStore(self.notes_path)

    def process(self, question: str) -> Dict[str, Any]:
        logger.info("Running structured retrieval for question: {}", question)
        structured = retrieve_answer(question, self.indexes, self.note_store)

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

    @staticmethod
    def _singularize(token: str) -> str:
        t = (token or "").strip().lower()
        if not t:
            return t
        if t.endswith("ses") and not t.endswith("sses"):
            return t[:-2]  # e.g., actresses -> actress
        if t.endswith("ies"):
            return t[:-3] + "y"
        if t.endswith("s") and not t.endswith("ss"):
            return t[:-1]
        return t

    @classmethod
    def _canonicalize_occupation(cls, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return raw
        lowered = raw.lower()
        fillers = {
            "film", "television", "tv", "stage", "screen", "movie",
            "commercial", "radio", "theatre", "theater",
            "former", "retired", "senior", "chief", "award-winning",
        }
        nationalities = {
            "american", "british", "scottish", "english", "canadian", "australian",
            "french", "german", "italian", "spanish", "mexican", "chinese", "japanese",
            "korean", "indian", "russian", "irish", "welsh", "dutch", "swedish",
            "norwegian", "danish", "finnish", "polish", "portuguese", "brazilian",
            "argentinian", "iranian", "iraqi", "egyptian", "turkish", "saudi", "thai",
            "indonesian", "malaysian", "singaporean", "pakistani", "bangladeshi",
            "nepalese", "sri lankan", "afghan", "ethiopian", "kenyan", "nigerian",
            "south african",
        }
        parts: list[str] = []
        for tok in lowered.replace("/", " and ").replace(",", " and ").split():
            if tok in fillers or tok in nationalities:
                continue
            parts.append(tok)
        cleaned = " ".join(parts)
        candidates: list[str] = []
        for chunk in [c.strip() for c in cleaned.split(" and ") if c.strip()]:
            chunk = chunk.replace("-", " ").strip()
            candidates.append(chunk)
        synonyms = {
            # actor family
            "actress": "actor",
            "television actor": "actor",
            "tv actor": "actor",
            "film actor": "actor",
            "screen actor": "actor",
            # business family
            "executive": "businessperson",
            "business executive": "businessperson",
            "advertising executive": "businessperson",
            "businessman": "businessperson",
            "businesswoman": "businessperson",
            "entrepreneur": "businessperson",
            "industrialist": "businessperson",
            # law/judiciary family
            "barrister": "lawyer",
            "solicitor": "lawyer",
            "attorney": "lawyer",
            "advocate": "lawyer",
            "jurist": "judge",
            "justice": "judge",
            "chief justice": "judge",
            "lord chief justice": "judge",
            # politics family
            "statesman": "politician",
            "government minister": "politician",
            "prime minister": "politician",
            "member of parliament": "politician",
            "mp": "politician",
        }
        # 主职业优先级（若并列，优先选择此序中的头部项）
        priority = ["actor", "politician", "judge", "lawyer", "cartoonist", "businessperson"]
        normalized: list[str] = []
        for cand in candidates:
            mapped = synonyms.get(cand) or cand
            mapped = cls._singularize(mapped)
            normalized.append(mapped)
        for head in priority:
            if head in normalized:
                return head
        # handle composite like "cartoonist and illustrator"
        if "illustrator" in normalized and "cartoonist" in normalized:
            return "cartoonist"
        return normalized[0] if normalized else raw.strip().lower()

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
        weak = False
        if support_paths <= 1 or (len(evidences) < 3):
            weak = True
        if isinstance(conf, (int, float)) and conf < 0.5:
            weak = True

        # 1) 结构化命中：直接用结构化答案；occupation 归一化且不改写
        if status == "structured_hit" and ans:
            if attribute == "occupation":
                return self._canonicalize_occupation(ans), {
                    "weak_evidence": weak,
                    "support_paths": support_paths,
                    "conf": conf,
                    "source": "structured_hit",
                }
            return ans, {
                "weak_evidence": weak,
                "support_paths": support_paths,
                "conf": conf,
                "source": "structured_hit",
            }
        # 2) occupation 弱证据：若存在链路且有结构化候选值，尽量归一化使用
        if ans and attribute == "occupation":
            return self._canonicalize_occupation(ans), {
                "weak_evidence": weak,
                "support_paths": support_paths,
                "conf": conf,
                "source": "structured_weak",
            }

        # 3) 仅当 IR 真没命中（无谓词链）才硬失败
        if status != "structured_hit" and pred_chain_len == 0:
            return "Insufficient evidence", {
                "weak_evidence": True,
                "support_paths": support_paths,
                "conf": conf,
                "source": "no_chain",
            }

        # 4) 存在链路但结构化答案为空：如配置了 LM，则尝试一次生成；否则保留空答案并打标
        if (not ans) and self.lmstudio_endpoint and self.lmstudio_model and evidences:
            try:
                lm_answer = call_lmstudio(self.lmstudio_endpoint, self.lmstudio_model, question, evidences)
                return lm_answer, {
                    "weak_evidence": weak,
                    "support_paths": support_paths,
                    "conf": conf,
                    "source": "lmstudio",
                }
            except Exception:
                return "Insufficient evidence", {
                    "weak_evidence": True,
                    "support_paths": support_paths,
                    "conf": conf,
                    "source": "lm_error",
                }

        # 5) 默认保留结构化的原样（不改写），并提供诊断标记
        return ans or "", {
            "weak_evidence": weak,
            "support_paths": support_paths,
            "conf": conf,
            "source": "structured_no_lm",
        }
