from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    logger.warning("numpy unavailable: {}. Vector search disabled.", exc)

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as exc:  # pragma: no cover - optional dependency
    cosine_similarity = None  # type: ignore
    logger.warning("sklearn unavailable: {}. Vector search disabled.", exc)

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

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        fallback_to_cpu_on_oom: bool = True,
    ) -> None:
        self.model_name = model_name
        env_device = os.environ.get("EMB_DEVICE")
        chosen = (device or env_device or "").strip().lower()
        self.device: Optional[str] = chosen or None
        self.fallback_to_cpu_on_oom = bool(fallback_to_cpu_on_oom)
        self._resolved_device: Optional[str] = None
        self._model: Optional[SentenceTransformer] = None

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "cuda out of memory" in msg
            or "cublas_status_alloc_failed" in msg
            or "cuda error: out of memory" in msg
            or "out of memory" in msg and "cuda" in msg
        )

    def _switch_to_cpu(self) -> None:
        self.device = "cpu"
        self._resolved_device = "cpu"
        if self._model is None:
            return
        try:
            self._model.to("cpu")
        except Exception:
            self._model = None
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers is not available")
        try:
            logger.info("Loading vector model: {}", self.model_name)
            st_kwargs = {}
            if self.device:
                st_kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **st_kwargs)
            self._resolved_device = str(getattr(self._model, "device", self.device) or "")
        except Exception as e:
            if self.fallback_to_cpu_on_oom and self._is_cuda_oom(e) and (self.device is None or (self.device or "").startswith("cuda")):
                logger.warning("CUDA OOM in vector model init; retrying on CPU")
                self._switch_to_cpu()
                self._model = SentenceTransformer(self.model_name, device="cpu")
                return
            logger.warning("Failed to load model {}: {}. Trying fallback.", self.model_name, e)
            try:
                fb_kwargs = {}
                if self.device:
                    fb_kwargs["device"] = self.device
                self._model = SentenceTransformer("paraphrase-MiniLM-L3-v2", **fb_kwargs)
                self._resolved_device = str(getattr(self._model, "device", self.device) or "")
                logger.info("Loaded fallback vector model: paraphrase-MiniLM-L3-v2")
            except Exception as e2:
                if self.fallback_to_cpu_on_oom and self._is_cuda_oom(e2) and (self.device is None or (self.device or "").startswith("cuda")):
                    logger.warning("CUDA OOM in fallback vector model init; retrying on CPU")
                    self._switch_to_cpu()
                    self._model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
                    logger.info("Loaded fallback vector model on CPU: paraphrase-MiniLM-L3-v2")
                    return
                logger.error("Failed to load fallback vector model: {}", e2)
                raise e2

    def search_in_notes(
        self,
        question: str,
        notes: Sequence[Dict[str, Any]],
        top_k: int = 64,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """在给定的笔记集合中进行向量检索，返回 top_k (note, score)。"""
        if np is None or cosine_similarity is None:
            return []
        if not notes:
            return []
        try:
            self._ensure_model()
        except RuntimeError as exc:
            logger.warning("Vector search unavailable: {}", exc)
            return []
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
            model_device = str(getattr(self._model, "device", self._resolved_device) or "")
            if self.fallback_to_cpu_on_oom and self._is_cuda_oom(exc) and model_device.startswith("cuda"):
                logger.warning("CUDA OOM during vector encoding; switching to CPU and retrying")
                self._switch_to_cpu()
                self._ensure_model()
                q_emb = self._model.encode([question])[0]
                n_embs = self._model.encode(texts)
            else:
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
