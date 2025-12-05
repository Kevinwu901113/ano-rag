from typing import Optional, Dict, Any
from loguru import logger
from config.config_loader import config as global_config
from rag_core.embedding_client import EmbeddingEncoder as RagEmbeddingEncoder
from rag_core.llm_client import LLMChatClient
from utils.embedding_utils import EmbeddingEncoder as QwenEmbeddingEncoder

def get_default_embedding_client(config: Optional[Dict[str, Any]] = None):
    """
    Constructs and returns an embedding client for baselines (including simple_raptor).
    When provider == "qwen3", reuses the implementation in utils.embedding_utils,
    allowing local Qwen3 model loading with model_path_override + cache_dir.
    """
    if config is None:
        config = global_config.load_config()

    retriever_cfg = config.get("retriever", {})
    emb_cfg = retriever_cfg.get("embedding", {})

    provider = str(emb_cfg.get("provider", "huggingface")).strip()
    model = str(emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")).strip()
    device = emb_cfg.get("device", "cpu")

    # --- provider == qwen3: use utils.embedding_utils, implementing local cache + local path ---
    if provider == "qwen3":
        from pathlib import Path
        cache_dir = emb_cfg.get("cache_dir")
        cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None

        override = emb_cfg.get("model_path_override")
        base = model or "Qwen/Qwen3-Embedding-8B"
        model_name = str(override or base).strip()
        if override:
            logger.info("Embedding model override detected for qwen3: {}", model_name)
        
        if device is None:
             raise ValueError("Device must be explicitly specified ('cuda' or 'cpu') for qwen3 provider.")

        dtype = emb_cfg.get("dtype")
        max_len = int(emb_cfg.get("max_len_note", 512))

        logger.info(
            f"Initializing Qwen3 Embedding (utils.embedding_utils): model={model_name}, "
            f"cache_dir={cache_dir}, device={device}, dtype={dtype}, max_len={max_len}"
        )

        return QwenEmbeddingEncoder(
            provider="qwen3",
            model_name=model_name,
            max_length=max_len,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
        )

    if provider not in ("huggingface", "vllm", "mock"):
        raise ValueError(f"Unsupported embedding provider: {provider}")

    # --- Other cases: still use the general implementation from rag_core.embedding_client ---
    known_keys = {"provider", "model", "device"}
    extra_kwargs = {k: v for k, v in emb_cfg.items() if k not in known_keys}

    logger.info(f"Initializing Embedding Client (rag_core): provider={provider}, model={model}, device={device}")
    
    return RagEmbeddingEncoder(
        provider=provider,
        model=model,
        device=device,
        **extra_kwargs,
    )

def get_default_llm_client(config: Optional[Dict[str, Any]] = None) -> LLMChatClient:
    """
    Constructs and returns an LLMChatClient instance based on the provided configuration or defaults.
    Prioritizes 'lmstudio' configuration, falls back to 'vllm'.
    """
    if config is None:
        config = global_config.load_config()
        
    # Try LM Studio config first (Preferred for current environment)
    llm_cfg = config.get("lmstudio", {})
    if not llm_cfg:
        # Fallback to vLLM
        llm_cfg = config.get("vllm", {})
        if llm_cfg:
            logger.info("Using vLLM configuration for LLM Client")
        else:
            logger.warning("No LLM configuration found (checked 'lmstudio' and 'vllm'). Using defaults.")
    else:
        logger.info("Using LM Studio configuration for LLM Client")

    endpoint = llm_cfg.get("endpoint")
    model = llm_cfg.get("model")
    temperature = float(llm_cfg.get("temperature", 0.0))
    stop = llm_cfg.get("stop")
    
    # Default fallback values if config is empty but client is requested
    if not endpoint:
        endpoint = "http://127.0.0.1:1234/v1" # Common default
    if not model:
        model = "default-model"
        
    logger.info(f"Initializing LLM Client: endpoint={endpoint}, model={model}")

    return LLMChatClient(
        endpoint=endpoint,
        model=model,
        temperature=temperature,
        stop=stop
    )
