from typing import List, Callable, Dict, Any, Tuple
import numpy as np
import threading
import torch
from transformers import AutoModel, AutoTokenizer

# Cache for the closure (get_embedding_model)
_embedding_lock = threading.Lock()
_embedding_model_cache = {}

# Cache for the actual loaded models (tokenizer, model)
_transformers_lock = threading.Lock()
_transformers_cache: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

def get_embedding_model(model_name: str, device: str = "cpu") -> Callable[[List[str]], np.ndarray]:
    """
    Returns a callable encoder.
    Ensures the closure is created once per (model_name, device).
    The underlying model loading is also cached.
    """
    key = (model_name, device)
    if key in _embedding_model_cache:
        return _embedding_model_cache[key]

    with _embedding_lock:
        if key in _embedding_model_cache:
            return _embedding_model_cache[key]

        def _encoder(texts: List[str]) -> np.ndarray:
            return encode_passages(texts, model_name=model_name, device=device)

        _embedding_model_cache[key] = _encoder
        return _encoder

def _load_transformers_model(model_name: str, device: str):
    key = (model_name, device)
    if key in _transformers_cache:
        return _transformers_cache[key]
    
    with _transformers_lock:
        if key in _transformers_cache:
            return _transformers_cache[key]
            
        print(f"Loading embedding model: {model_name} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True)
        
        target_device = torch.device(device)
        model.to(target_device)
        model.eval()
        
        _transformers_cache[key] = (tokenizer, model)
        return tokenizer, model

def encode_passages(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    device: str = "cpu",
    batch_size: int = 32,
    normalize: bool = True
) -> np.ndarray:
    """
    Unified embedding interface using transformers.
    """
    if not texts:
        return np.array([])

    tokenizer, model = _load_transformers_model(model_name, device)
    
    all_embeddings = []
    
    # Batch processing
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512, # Default max length
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            all_embeddings.append(embeddings.cpu().numpy())
            
    if all_embeddings:
        return np.concatenate(all_embeddings, axis=0)
    else:
        return np.array([])
