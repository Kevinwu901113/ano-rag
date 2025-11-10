from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
from .text_builders import build_note_text_for_embed

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    SentenceTransformer = None  # type: ignore
    logger.warning("sentence_transformers not available: {}. Vector search disabled.", exc)


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
            text = build_note_text_for_embed(note, max_len=512)
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
