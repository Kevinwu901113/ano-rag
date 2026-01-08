from __future__ import annotations

from pathlib import Path
import hashlib
import threading
from typing import Dict, Optional, Sequence

import numpy as np
from loguru import logger


_ENCODER_CACHE: Dict[tuple, "EmbeddingEncoder"] = {}
_ENCODER_LOCK = threading.Lock()


class EmbeddingEncoder:
    """Shared embedding encoder for offline/online stages."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        max_length: int = 256,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        fallback_to_cpu_on_oom: bool = True,
        batch_size: int = 16,
        normalize: bool = False,
    ) -> None:
        self.provider = provider
        self.model_name = self._resolve_model_name(model_name)
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.normalize = bool(normalize)
        self.cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None
        self._device_pref = device.lower() if isinstance(device, str) else None
        self._dtype_pref = dtype.lower() if isinstance(dtype, str) else None
        self._fallback_to_cpu_on_oom = bool(fallback_to_cpu_on_oom)
        self._resolved_device: Optional[str] = None
        self._resolved_torch_dtype = None
        self._model_uses_device_map = False
        self._model = None
        self._tokenizer = None
        self._encode_lock = threading.Lock()

    def encode(
        self,
        texts: Sequence[str],
        *,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        with self._encode_lock:
            return self._encode_impl(
                texts,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                normalize=normalize,
            )

    def _encode_impl(
        self,
        texts: Sequence[str],
        *,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        if self.provider == "mock" or str(self.model_name).strip().lower() == "mock":
            return self._encode_mock(texts, normalize=normalize)
        if self.provider == "qwen3":
            return self._encode_transformers(
                texts,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                normalize=normalize,
            )
        if self.provider == "st":
            return self._encode_sentence_transformers(
                texts,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                normalize=normalize,
            )
        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _encode_mock(self, texts: Sequence[str], *, normalize: Optional[bool]) -> np.ndarray:
        dim = 8
        vectors = np.zeros((len(texts), dim), dtype="float32")
        for i, text in enumerate(texts):
            seed = int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vectors[i] = rng.normal(size=dim)
        if normalize or (normalize is None and self.normalize):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            vectors = vectors / norms
        return vectors

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "cuda out of memory" in msg
            or "cublas_status_alloc_failed" in msg
            or "cuda error: out of memory" in msg
            or "out of memory" in msg and "cuda" in msg
        )

    def _maybe_empty_cuda_cache(self, torch_module) -> None:
        try:
            if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        except Exception:
            return

    def _resolve_device(self, torch_module) -> str:
        """Resolve runtime device from preference + availability."""
        if self._resolved_device is not None:
            return self._resolved_device

        prefer = (self._device_pref or "").strip().lower()
        if prefer in {"", "auto", "best"}:
            prefer = "auto"

        if prefer == "cpu":
            device = "cpu"
        elif prefer.startswith("cuda"):
            if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
                device = prefer
            else:
                logger.warning("Requested CUDA for embeddings but GPU unavailable; falling back to CPU")
                device = "cpu"
        elif prefer == "mps":
            if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
                device = "mps"
            else:
                logger.warning("Requested MPS for embeddings but unavailable; falling back to CPU")
                device = "cpu"
        else:
            if prefer not in {"auto"}:
                logger.warning("Unknown embedding device '{}'; falling back to auto", prefer)
            if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
                device = "cuda"
            elif hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._resolved_device = device
        return device

    def _encode_transformers(
        self,
        texts: Sequence[str],
        *,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers/torch are required for provider 'qwen3'") from exc

        batch_size = max(1, int(batch_size if batch_size is not None else self.batch_size))
        max_length = int(max_length if max_length is not None else self.max_length)
        normalize = bool(self.normalize if normalize is None else normalize)

        target_device_str = str(device or "").strip().lower() or self._resolve_device(torch)
        if target_device_str == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available():
            # Prefer explicit cuda:0 for clearer logs / torch.device parsing.
            target_device_str = "cuda:0"

        if self._model is None or self._tokenizer is None:
            logger.info("Loading transformer embedding model: {}", self.model_name)
            tokenizer_kwargs = {"trust_remote_code": True}
            model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
            
            # Resolve device: Priority GPU
            logger.info("Embedding target device resolved to: {}", target_device_str)
            target_device = torch.device(target_device_str)

            if self.cache_dir:
                tokenizer_kwargs["cache_dir"] = self.cache_dir
                model_kwargs["cache_dir"] = self.cache_dir
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
            torch_dtype = self._resolve_torch_dtype(torch, device=str(target_device_str))
            if torch_dtype is not None:
                # Keep both keys: some remote_code uses `dtype`, transformers uses `torch_dtype`.
                model_kwargs["torch_dtype"] = torch_dtype
                model_kwargs["dtype"] = torch_dtype

            # Load model
            def _load_model_with_kwargs(kwargs):
                try:
                    return AutoModel.from_pretrained(self.model_name, **kwargs)
                except TypeError as exc:
                    msg = str(exc)
                    if "unexpected keyword argument" not in msg:
                        raise
                    cleaned = dict(kwargs)
                    if "dtype" in cleaned and "dtype" in msg:
                        cleaned.pop("dtype", None)
                        return AutoModel.from_pretrained(self.model_name, **cleaned)
                    if "torch_dtype" in cleaned and "torch_dtype" in msg:
                        cleaned.pop("torch_dtype", None)
                        return AutoModel.from_pretrained(self.model_name, **cleaned)
                    raise

            try:
                self._model = _load_model_with_kwargs(model_kwargs)
                self._model.to(target_device)
            except Exception as e:
                if (
                    self._fallback_to_cpu_on_oom
                    and str(target_device_str).startswith("cuda")
                    and self._is_cuda_oom(e)
                    and target_device_str != "cpu"
                ):
                    logger.warning("CUDA OOM while loading embedding model; retrying on CPU")
                else:
                    logger.error("Failed to load model on {}: {}. Falling back to CPU.", target_device, e)

                self._maybe_empty_cuda_cache(torch)
                self._resolved_device = "cpu"
                self._resolved_torch_dtype = None
                cpu_kwargs = dict(model_kwargs)
                # Remove device specific args if any
                self._model = _load_model_with_kwargs(cpu_kwargs)
        
            # Ensure model is on the correct device
            current_device = next(self._model.parameters()).device
            target_device = torch.device(target_device_str)
            if current_device.type != target_device.type:
                logger.info(f"Moving model from {current_device} to {target_device}")
                self._model.to(target_device)

            self._model.eval()

        # Ensure device matches current request (supports cuda->cpu fallback calls).
        assert self._model is not None and self._tokenizer is not None
        current_device = str(self._resolved_device or "cpu").lower()
        if current_device != str(target_device_str).lower():
            try:
                self._model.to(torch.device(target_device_str))
                if str(target_device_str).lower() == "cpu":
                    self._model.to(dtype=torch.float32)
                self._resolved_device = str(target_device_str)
                self._resolved_torch_dtype = None
            except Exception:
                # Let the caller handle device transition errors (run_with_fallback will catch CUDA OOM).
                raise

        vectors = []
        device_str = self._resolved_device or "cpu"
        device_t = torch.device(device_str)
        for start in range(0, len(texts), batch_size):

            batch = texts[start : start + batch_size]
            while True:
                try:
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(device_t)
                    with torch.inference_mode():
                        outputs = self._model(**inputs)
                        pooled = outputs.last_hidden_state.mean(dim=1)
                        if normalize:
                            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    break
                except Exception as exc:
                    if (
                        self._fallback_to_cpu_on_oom
                        and str(device_str).startswith("cuda")
                        and self._is_cuda_oom(exc)
                        and str(device_str) != "cpu"
                    ):
                        logger.warning("CUDA OOM during embedding encode; switching to CPU and retrying")
                        try:
                            self._model.to("cpu")
                            # Force float32 on CPU for max compatibility.
                            self._model.to(dtype=torch.float32)
                        except Exception:
                            pass
                        self._maybe_empty_cuda_cache(torch)
                        self._resolved_device = "cpu"
                        self._resolved_torch_dtype = None
                        device_str = "cpu"
                        device_t = torch.device("cpu")
                        continue
                    raise
            pooled = pooled.to(dtype=torch.float32)
            vectors.extend(pooled.cpu().numpy())
        return np.vstack(vectors).astype("float32")

    def _encode_sentence_transformers(
        self,
        texts: Sequence[str],
        *,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        try:
            import torch  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is required for provider 'st'") from exc

        batch_size = max(1, int(batch_size if batch_size is not None else self.batch_size))
        max_length = int(max_length if max_length is not None else self.max_length)
        normalize = bool(self.normalize if normalize is None else normalize)

        if self._model is None:
            logger.info("Loading SentenceTransformer model: {}", self.model_name)
            st_kwargs = {}
            if self.cache_dir:
                st_kwargs["cache_folder"] = self.cache_dir
            resolved = str(device or "").strip().lower() or self._resolve_device(torch)
            self._resolved_device = resolved
            st_kwargs["device"] = resolved
            try:
                self._model = SentenceTransformer(self.model_name, **st_kwargs)
            except Exception as exc:
                if (
                    self._fallback_to_cpu_on_oom
                    and str(resolved).startswith("cuda")
                    and self._is_cuda_oom(exc)
                    and resolved != "cpu"
                ):
                    logger.warning("CUDA OOM in SentenceTransformer init; retrying on CPU")
                    st_kwargs["device"] = "cpu"
                    self._resolved_device = "cpu"
                    self._resolved_torch_dtype = None
                    self._model = SentenceTransformer(self.model_name, **st_kwargs)
                else:
                    raise

        if max_length and hasattr(self._model, "max_seq_length"):
            try:
                self._model.max_seq_length = max_length
            except Exception:
                pass

        try:
            embeddings = self._model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
        except Exception as exc:
            if (
                self._fallback_to_cpu_on_oom
                and str(self._resolved_device or "").startswith("cuda")
                and self._is_cuda_oom(exc)
                and (self._resolved_device or "") != "cpu"
            ):
                logger.warning("CUDA OOM during SentenceTransformer encode; switching to CPU and retrying")
                try:
                    self._model.to("cpu")
                except Exception:
                    self._model = None
                self._maybe_empty_cuda_cache(torch)
                self._resolved_device = "cpu"
                self._resolved_torch_dtype = None
                if self._model is None:
                    st_kwargs = {}
                    if self.cache_dir:
                        st_kwargs["cache_folder"] = self.cache_dir
                    st_kwargs["device"] = "cpu"
                    self._model = SentenceTransformer(self.model_name, **st_kwargs)
                embeddings = self._model.encode(
                    list(texts),
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
            else:
                raise
        return embeddings.astype("float32")

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        if not model_name:
            raise ValueError("Embedding model name must be provided")
        expanded = Path(model_name).expanduser()
        return str(expanded) if expanded.exists() else model_name

    def _resolve_torch_dtype(self, torch_module, *, device: Optional[str] = None):
        if self._resolved_torch_dtype is not None:
            return self._resolved_torch_dtype
        if not self._dtype_pref:
            return None
        key = self._dtype_pref.lower()
        if device == "cpu" and key not in {"float32", "fp32"}:
            logger.warning("Non-float32 embeddings on CPU are not supported; using float32 instead")
            self._resolved_torch_dtype = None
            return None
        mapping = {
            "float32": "float32",
            "fp32": "float32",
            "float16": "float16",
            "fp16": "float16",
            "half": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
        }
        attr = mapping.get(key)
        if attr and hasattr(torch_module, attr):
            self._resolved_torch_dtype = getattr(torch_module, attr)
        else:
            logger.warning("Unsupported embedding dtype '{}'; using default precision", self._dtype_pref)
            self._resolved_torch_dtype = None
        return self._resolved_torch_dtype

    def _build_device_map(self, device: Optional[str]):
        if not device or device == "cpu":
            return None
        if device.startswith("cuda"):
            return {"": device}
        return None


def get_shared_encoder(
    provider: str,
    model_name: str,
    max_length: int = 256,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> EmbeddingEncoder:
    key = (
        str(provider or "").strip().lower(),
        str(model_name or "").strip(),
        int(max_length),
        str(cache_dir or ""),
        str(device or ""),
        str(dtype or ""),
    )
    with _ENCODER_LOCK:
        cached = _ENCODER_CACHE.get(key)
        if cached is None:
            cached = EmbeddingEncoder(
                provider,
                model_name,
                max_length=max_length,
                cache_dir=cache_dir,
                device=device,
                dtype=dtype,
            )
            _ENCODER_CACHE[key] = cached
        return cached
