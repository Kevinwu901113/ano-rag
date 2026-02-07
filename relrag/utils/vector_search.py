from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    logger.warning("numpy unavailable: {}. Vector search disabled.", exc)

from .text_builders import build_note_text_for_embed
from .embedding_utils import get_shared_encoder

_SentenceTransformer = None  # type: ignore
_SENTENCE_TRANSFORMER_IMPORT_ERROR: Optional[Exception] = None
_SENTENCE_TRANSFORMER_IMPORT_TRIED = False


def _load_sentence_transformer():
    global _SentenceTransformer
    global _SENTENCE_TRANSFORMER_IMPORT_ERROR
    global _SENTENCE_TRANSFORMER_IMPORT_TRIED
    if _SENTENCE_TRANSFORMER_IMPORT_TRIED:
        return _SentenceTransformer
    _SENTENCE_TRANSFORMER_IMPORT_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer as _ST  # type: ignore
        _SentenceTransformer = _ST
        _SENTENCE_TRANSFORMER_IMPORT_ERROR = None
    except Exception as exc:
        _SentenceTransformer = None
        _SENTENCE_TRANSFORMER_IMPORT_ERROR = exc
    return _SentenceTransformer


class VectorSearcher:
    """向量-only 检索器：用于预筛与兜底。

    注意：仅用于缩小候选集或作为兜底，不参与最终排序，不与结构通道融合。
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        fallback_to_cpu_on_oom: bool = True,
        *,
        embedding_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        env_device = os.environ.get("EMB_DEVICE")
        chosen = (device or env_device or "").strip().lower()
        self.device: Optional[str] = chosen or None
        self.fallback_to_cpu_on_oom = bool(fallback_to_cpu_on_oom)
        self.embedding_cfg = embedding_cfg if isinstance(embedding_cfg, dict) else None
        self._encoder = None
        self._encoder_batch_size = 16
        self._encoder_max_len = 256
        self._encoder_normalize = True
        self._resolved_device: Optional[str] = None
        self._model: Optional[Any] = None

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
        if self.embedding_cfg:
            return
        st_cls = _load_sentence_transformer()
        if _SentenceTransformer is None:
            if _SENTENCE_TRANSFORMER_IMPORT_ERROR is not None:
                logger.warning(
                    "sentence_transformers unavailable for provider='st': {}",
                    _SENTENCE_TRANSFORMER_IMPORT_ERROR,
                )
            raise RuntimeError("sentence_transformers is not available")
        try:
            logger.info("Loading vector model: {}", self.model_name)
            st_kwargs = {}
            if self.device:
                st_kwargs["device"] = self.device
            self._model = st_cls(self.model_name, **st_kwargs)
            self._resolved_device = str(getattr(self._model, "device", self.device) or "")
        except Exception as e:
            if self.fallback_to_cpu_on_oom and self._is_cuda_oom(e) and (self.device is None or (self.device or "").startswith("cuda")):
                logger.warning("CUDA OOM in vector model init; retrying on CPU")
                self._switch_to_cpu()
                self._model = st_cls(self.model_name, device="cpu")
                return
            logger.warning("Failed to load model {}: {}. Trying fallback.", self.model_name, e)
            try:
                fb_kwargs = {}
                if self.device:
                    fb_kwargs["device"] = self.device
                self._model = st_cls("paraphrase-MiniLM-L3-v2", **fb_kwargs)
                self._resolved_device = str(getattr(self._model, "device", self.device) or "")
                logger.info("Loaded fallback vector model: paraphrase-MiniLM-L3-v2")
            except Exception as e2:
                if self.fallback_to_cpu_on_oom and self._is_cuda_oom(e2) and (self.device is None or (self.device or "").startswith("cuda")):
                    logger.warning("CUDA OOM in fallback vector model init; retrying on CPU")
                    self._switch_to_cpu()
                    self._model = st_cls("paraphrase-MiniLM-L3-v2", device="cpu")
                    logger.info("Loaded fallback vector model on CPU: paraphrase-MiniLM-L3-v2")
                    return
                logger.error("Failed to load fallback vector model: {}", e2)
                raise e2

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        if not self.embedding_cfg:
            return
        provider = str(self.embedding_cfg.get("provider") or "").strip().lower()
        model_name = str(self.embedding_cfg.get("model") or "").strip()
        if not provider or not model_name:
            return
        self._encoder_batch_size = int(self.embedding_cfg.get("batch_size", 16))
        self._encoder_max_len = int(self.embedding_cfg.get("max_len_note", 256))
        self._encoder_normalize = bool(self.embedding_cfg.get("normalize", True))
        cache_dir = self.embedding_cfg.get("cache_dir")
        device = self.embedding_cfg.get("device")
        dtype = self.embedding_cfg.get("dtype")
        endpoint = self.embedding_cfg.get("endpoint")
        api_key = self.embedding_cfg.get("api_key")
        timeout_s = self.embedding_cfg.get("timeout_s") or self.embedding_cfg.get("request_timeout_s")
        self._encoder = get_shared_encoder(
            provider,
            model_name,
            max_length=self._encoder_max_len,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            endpoint=endpoint,
            api_key=api_key,
            request_timeout_s=timeout_s,
        )
        logger.info("Vector fallback embedding: provider={} model={}", provider, model_name)

    @staticmethod
    def _cosine_similarity_matrix(vectors: "np.ndarray", query: "np.ndarray") -> "np.ndarray":
        denom = (np.linalg.norm(vectors, axis=1) * (np.linalg.norm(query) + 1e-12)) + 1e-12
        return (vectors @ query) / denom

    def search_in_notes(
        self,
        question: str,
        notes: Sequence[Dict[str, Any]],
        top_k: int = 64,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """在给定的笔记集合中进行向量检索，返回 top_k (note, score)。"""
        if np is None:
            return []
        if not notes:
            return []
        use_encoder = bool(self.embedding_cfg and self.embedding_cfg.get("provider") and self.embedding_cfg.get("model"))
        try:
            if use_encoder:
                self._ensure_encoder()
                if self._encoder is None:
                    raise RuntimeError("Vector fallback embedding encoder unavailable")
            else:
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
            if use_encoder and self._encoder is not None:
                q_emb = self._encoder.encode(
                    [question],
                    batch_size=self._encoder_batch_size,
                    max_length=self._encoder_max_len,
                    normalize=self._encoder_normalize,
                )[0]
                n_embs = self._encoder.encode(
                    texts,
                    batch_size=self._encoder_batch_size,
                    max_length=self._encoder_max_len,
                    normalize=self._encoder_normalize,
                )
            else:
                q_emb = self._model.encode([question])[0]
                n_embs = self._model.encode(texts)
        except Exception as exc:
            if not use_encoder:
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
            else:
                logger.error("Vector fallback encoding failed: {}", exc)
                return []
        # 余弦相似度
        sims = self._cosine_similarity_matrix(np.array(n_embs), np.array(q_emb))
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
