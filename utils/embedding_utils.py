from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from loguru import logger


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
    ) -> None:
        self.provider = provider
        self.model_name = self._resolve_model_name(model_name)
        self.max_length = max_length
        self.cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None
        self._device_pref = device.lower() if isinstance(device, str) else None
        self._dtype_pref = dtype.lower() if isinstance(dtype, str) else None
        self._resolved_device: Optional[str] = None
        self._resolved_torch_dtype = None
        self._model_uses_device_map = False
        self._model = None
        self._tokenizer = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        if self.provider == "qwen3":
            return self._encode_transformers(texts)
        if self.provider == "st":
            return self._encode_sentence_transformers(texts)
        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _resolve_device(self, torch) -> str:
        """Resolve the best available device."""
        if self._device_pref:
            return self._device_pref
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _encode_transformers(self, texts: Sequence[str]) -> np.ndarray:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers/torch are required for provider 'qwen3'") from exc
        if self._model is None or self._tokenizer is None:
            logger.info("Loading transformer embedding model: {}", self.model_name)
            tokenizer_kwargs = {"trust_remote_code": True}
            model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
            
            # Resolve device: Priority GPU
            target_device_str = self._resolve_device(torch)
            logger.info(f"Target device resolved to: {target_device_str}")
            target_device = torch.device(target_device_str)
            self._resolved_device = target_device_str

            if self.cache_dir:
                tokenizer_kwargs["cache_dir"] = self.cache_dir
                model_kwargs["cache_dir"] = self.cache_dir
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
            torch_dtype = self._resolve_torch_dtype(torch)
            if torch_dtype is not None:
                model_kwargs["dtype"] = torch_dtype

            # Load model
            try:
                self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
                self._model.to(target_device)
            except Exception as e:
                logger.error(f"Failed to load model on {target_device}: {e}. Falling back to CPU.")
                self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
                self._model.to("cpu")
                self._resolved_device = "cpu"

            self._model.eval()

        assert self._model is not None and self._tokenizer is not None
        vectors = []
        batch_size = 16
        device = torch.device(self._resolved_device or "cpu")
        for start in range(0, len(texts), batch_size):

            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                outputs = self._model(**inputs)
                pooled = outputs.last_hidden_state.mean(dim=1)
            pooled = pooled.to(dtype=torch.float32)
            vectors.extend(pooled.cpu().numpy())
        return np.vstack(vectors).astype("float32")

    def _encode_sentence_transformers(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is required for provider 'st'") from exc
        if self._model is None:
            logger.info("Loading SentenceTransformer model: {}", self.model_name)
            st_kwargs = {}
            if self.cache_dir:
                st_kwargs["cache_folder"] = self.cache_dir
            self._model = SentenceTransformer(self.model_name, **st_kwargs)
        embeddings = self._model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def _resolve_device(self, torch_module):
        if self._resolved_device is not None:
            return self._resolved_device
        device = None
        if self._device_pref:
            if self._device_pref.startswith("cuda"):
                if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
                    device = self._device_pref
                else:
                    logger.warning("Requested CUDA for embeddings but GPU unavailable; falling back to CPU")
            elif self._device_pref == "cpu":
                device = "cpu"
            else:
                logger.warning("Unknown embedding device '{}'; falling back to auto", self._device_pref)
        if device is None and hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            device = "cuda"
        if device is None:
            logger.warning("Requested CUDA for embeddings but GPU unavailable; falling back to CPU")
            device = "cpu"
        self._resolved_device = device
        return device

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        if not model_name:
            raise ValueError("Embedding model name must be provided")
        expanded = Path(model_name).expanduser()
        return str(expanded) if expanded.exists() else model_name

    def _resolve_torch_dtype(self, torch_module):
        if self._resolved_torch_dtype is not None:
            return self._resolved_torch_dtype
        if not self._dtype_pref:
            return None
        key = self._dtype_pref.lower()
        mapping = {
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
