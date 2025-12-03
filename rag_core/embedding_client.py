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
                try:
                    self._model_instance = SentenceTransformer(model, device=device)
                except RuntimeError as e:
                     if "CUDA out of memory" in str(e):
                         logger.warning("CUDA OOM in SentenceTransformer init, retrying on CPU")
                         self._model_instance = SentenceTransformer(model, device="cpu")
                     else:
                         raise e
            except ImportError:
                logger.warning("sentence-transformers not installed. Using mock embeddings.")
                self._model_instance = None
        elif provider == "vllm":
            # OpenAI-compatible vLLM embedding client
            self.endpoint = kwargs.get("endpoint", "http://127.0.0.1:8000/v1").rstrip("/")
            self.api_key = kwargs.get("api_key", "EMPTY")
            self.model_name = model
            logger.info(f"Using vLLM embedding provider at {self.endpoint} with model {self.model_name}")
            self._model_instance = None # Stateless client
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
            
        if self.provider == "huggingface" and self._model_instance:
            return self._model_instance.encode(texts, normalize_embeddings=normalize_embeddings)
        elif self.provider == "vllm":
            import requests
            try:
                # vLLM usually supports batching, but let's handle potential limits if needed
                # For now, send all at once (assuming batch size is reasonable)
                url = f"{self.endpoint}/embeddings"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                payload = {
                    "model": self.model_name,
                    "input": texts
                }
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract embeddings from response
                # Response format: {"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...]}
                # Sort by index to ensure order matches input
                embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                embeddings = [item["embedding"] for item in embeddings_data]
                
                emb_array = np.array(embeddings, dtype=np.float32)
                
                if normalize_embeddings:
                    # L2 normalization
                    norm = np.linalg.norm(emb_array, axis=1, keepdims=True)
                    # Avoid division by zero
                    norm[norm == 0] = 1e-12
                    emb_array = emb_array / norm
                    
                return emb_array
            except Exception as e:
                logger.error(f"vLLM embedding failed: {e}")
                # Fallback to mock if vLLM fails? Or raise? 
                # Let's return empty or mock to avoid crashing the whole pipeline if transient failure
                # But usually we want to know. Let's log and return mock for robustness in this context.
                logger.warning("Falling back to mock embeddings due to vLLM error")
                dim = 768 # Assume 768 for most models like Qwen
                return np.random.rand(len(texts), dim).astype(np.float32)
        else:
            # Mock fallback
            # Default dimension for MiniLM is 384, for others it might be 768 or 1024
            # We'll use 768 as a safe default for compatibility if unknown
            dim = 384 if "MiniLM" in self.model_name else 768
            logger.debug(f"Generating mock embeddings (dim={dim}) for {len(texts)} texts")
            return np.random.rand(len(texts), dim).astype(np.float32)
