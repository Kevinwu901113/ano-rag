from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    SentenceTransformer = None  # type: ignore
    logger.warning("sentence_transformers not available: {}. Vector search disabled.", exc)


def _note_text(note: Dict[str, Any]) -> str:
    """构造用于向量编码的笔记文本。

    优先使用结构化证据与简短字段，避免引入噪声：
    - evidence（原句/规范化句）
    - obj（若为字符串）
    - meta.subject_profile.description / meta.object_profile.description（若存在）
    """
    parts: List[str] = []
    meta = (note.get("meta", {}) or {})
    canonical = meta.get("evidence_canonical")
    if isinstance(canonical, str) and canonical.strip():
        parts.append(canonical.strip())
    else:
        ev = note.get("evidence")
        if isinstance(ev, str) and ev.strip():
            parts.append(ev.strip())
    obj = note.get("obj")
    if isinstance(obj, str) and obj.strip():
        parts.append(obj.strip())
    subj_prof = ((meta.get("subject_profile") or {}) or {}).get("description")
    if isinstance(subj_prof, str) and subj_prof.strip():
        parts.append(subj_prof.strip())
    obj_prof = ((meta.get("object_profile") or {}) or {}).get("description")
    if isinstance(obj_prof, str) and obj_prof.strip():
        parts.append(obj_prof.strip())
    return " \n".join(parts)


class VectorSearcher:
    """向量-only 检索器：用于预筛与兜底。

    注意：仅用于缩小候选集或作为兜底，不参与最终排序，不与结构通道融合。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers is not available")
        try:
            logger.info("Loading vector model: {}", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning("Failed to load model {}: {}. Trying fallback.", self.model_name, e)
            try:
                self._model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
                logger.info("Loaded fallback vector model: paraphrase-MiniLM-L3-v2")
            except Exception as e2:
                logger.error("Failed to load fallback vector model: {}", e2)
                raise e2

    def search_in_notes(
        self,
        question: str,
        notes: Sequence[Dict[str, Any]],
        top_k: int = 64,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """在给定的笔记集合中进行向量检索，返回 top_k (note, score)。"""
        if not notes:
            return []
        self._ensure_model()
        texts: List[str] = []
        valid_notes: List[Dict[str, Any]] = []
        for note in notes:
            text = _note_text(note)
            if text:
                texts.append(text)
                valid_notes.append(note)
        if not valid_notes:
            return []
        try:
            q_emb = self._model.encode([question])[0]
            n_embs = self._model.encode(texts)
        except Exception as exc:
            logger.error("Vector encoding failed: {}", exc)
            return []
        # 余弦相似度
        sims = cosine_similarity(np.array([q_emb]), np.array(n_embs))[0]
        ranked = sorted(zip(valid_notes, sims.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[: top_k]

    def search_note_ids(
        self,
        question: str,
        note_loader: Any,
        candidate_note_ids: Sequence[str],
        top_k: int = 64,
    ) -> List[str]:
        """对指定 note_id 集合做向量检索，返回 top_k 的 note_id 列表。"""
        if not candidate_note_ids:
            return []
        notes = note_loader.get_many(list(candidate_note_ids))
        ranked = self.search_in_notes(question, notes, top_k=top_k)
        return [note.get("note_id") for note, _ in ranked if note.get("note_id")][: top_k]