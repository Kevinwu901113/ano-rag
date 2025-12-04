from typing import Optional, Dict, Any
from loguru import logger
from config.config_loader import config as global_config
from rag_core.embedding_client import EmbeddingEncoder
from rag_core.llm_client import LLMChatClient

def get_default_embedding_client(config: Optional[Dict[str, Any]] = None) -> EmbeddingEncoder:
    """
    Constructs and returns an EmbeddingEncoder instance based on the provided configuration or defaults.
    """
    if config is None:
        config = global_config.load_config()

    # Get embedding configuration
    retriever_cfg = config.get("retriever", {})
    emb_cfg = retriever_cfg.get("embedding", {})
    
    # Extract parameters
    provider = emb_cfg.get("provider", "huggingface")
    model = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    device = emb_cfg.get("device", "cpu")
    
    # Extract other kwargs, excluding known keys
    known_keys = {"provider", "model", "device"}
    extra_kwargs = {k: v for k, v in emb_cfg.items() if k not in known_keys}
    
    logger.info(f"Initializing Embedding Client: provider={provider}, model={model}, device={device}")
    
    return EmbeddingEncoder(
        provider=provider,
        model=model,
        device=device,
        **extra_kwargs
    )

def get_default_llm_client(config: Optional[Dict[str, Any]] = None) -> LLMChatClient:
    """
    Constructs and returns an LLMChatClient instance based on the provided configuration or defaults.
    Prioritizes 'vllm' configuration, falls back to 'lmstudio'.
    """
    if config is None:
        config = global_config.load_config()
        
    # Try vLLM config first
    llm_cfg = config.get("vllm", {})
    if not llm_cfg:
        # Fallback to LM Studio
        llm_cfg = config.get("lmstudio", {})
        if llm_cfg:
            logger.info("Using LM Studio configuration for LLM Client")
        else:
            logger.warning("No LLM configuration found (checked 'vllm' and 'lmstudio'). Using defaults.")
    else:
        logger.info("Using vLLM configuration for LLM Client")

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
