import numpy as np
from typing import List, Optional
from loguru import logger

class EmbeddingEncoder:
    def __init__(self, provider: str = "huggingface", model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu", **kwargs):
        self.provider = provider
        self.model_name = model
        self.device = device
        self._model_instance = None
        
        logger.info(f"Initializing EmbeddingEncoder: {provider} / {model} on {device}")
        
        if provider == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer
                self._model_instance = SentenceTransformer(model, device=device)
            except ImportError:
                logger.warning("sentence-transformers not installed. Using mock embeddings.")
                self._model_instance = None
        elif provider == "mock":
            self._model_instance = None
        else:
            logger.warning(f"Unknown provider {provider}, falling back to mock or implementation needed.")
            self._model_instance = None

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        Returns a numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([])
            
        if self._model_instance:
            return self._model_instance.encode(texts, normalize_embeddings=normalize_embeddings)
        else:
            # Mock fallback
            # Default dimension for MiniLM is 384, for others it might be 768 or 1024
            # We'll use 768 as a safe default for compatibility if unknown
            dim = 384 if "MiniLM" in self.model_name else 768
            logger.debug(f"Generating mock embeddings (dim={dim}) for {len(texts)} texts")
            return np.random.rand(len(texts), dim).astype(np.float32)
