from typing import Dict, Any, Optional
from loguru import logger
from config import config

# Import providers
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .lmstudio_client import LMStudioClient
from .vllm_openai_client import VllmOpenAIClient


class LLMFactory:
    """LLM Provider 工厂类
    
    负责根据配置创建和管理不同的 LLM Provider
    支持 Ollama、OpenAI 等多种 Provider
    """
    
    # 注册的 Provider 类型（移除 multi_model 与 hybrid_llm）
    PROVIDERS = {
        'ollama': OllamaClient,
        'openai': OpenAIClient,
        'lmstudio': LMStudioClient,
        'vllm-openai': VllmOpenAIClient,
    }
    
    @classmethod
    def create_provider(cls, provider_type: str = None, route_name: str = None, **kwargs) -> Any:
        """创建 LLM Provider 实例"""
        if provider_type is None:
            provider_type = config.get('llm.provider', 'ollama')
        if provider_type not in cls.PROVIDERS:
            logger.warning(f"Unsupported provider: {provider_type}, falling back to ollama")
            provider_type = 'ollama'
        
        try:
            if provider_type == 'ollama':
                return cls._create_ollama_provider(**kwargs)
            elif provider_type == 'openai':
                return cls._create_openai_provider(**kwargs)
            elif provider_type == 'lmstudio':
                return cls._create_lmstudio_provider(**kwargs)
            elif provider_type == 'vllm-openai':
                return cls._create_vllm_openai_provider(**kwargs)
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")
        except Exception as e:
            logger.error(f"Failed to create {provider_type} provider: {e}")
            if provider_type != 'ollama':
                logger.info("Falling back to Ollama provider")
                return cls._create_ollama_provider(**kwargs)
            raise

    @classmethod
    def _create_ollama_provider(cls, **kwargs) -> OllamaClient:
        base_url = kwargs.get('base_url') or config.get('llm.ollama.base_url', 'http://localhost:11434')
        model = kwargs.get('model') or config.get('llm.ollama.model', 'gpt-oss:latest')
        logger.info(f"Creating Ollama provider with model: {model}")
        return OllamaClient(base_url=base_url, model=model)
    
    @classmethod
    def _create_openai_provider(cls, **kwargs) -> OpenAIClient:
        api_key = kwargs.get('api_key') or config.get('llm.openai.api_key')
        base_url = kwargs.get('base_url') or config.get('llm.openai.base_url')
        model = kwargs.get('model') or config.get('llm.openai.model', 'gpt-3.5-turbo')
        if not api_key:
            raise ValueError("OpenAI API key is required")
        logger.info(f"Creating OpenAI provider with model: {model}")
        return OpenAIClient(api_key=api_key, model=model, base_url=base_url)
    
    @classmethod
    def _create_lmstudio_provider(cls, **kwargs) -> LMStudioClient:
        base_url = kwargs.get('base_url') or config.get('llm.lmstudio.base_url')
        model = kwargs.get('model') or config.get('llm.lmstudio.model', 'default-model')
        port = kwargs.get('port') or config.get('llm.lmstudio.port', 1234)
        concurrent_enabled = config.get('llm.lmstudio.concurrent.enabled', False)
        mode = 'concurrent' if concurrent_enabled else 'single-instance'
        logger.info(f"Creating LM Studio provider in {mode} mode with model: {model} on port: {port}")
        return LMStudioClient(base_url=base_url, model=model, port=port)
    
    @classmethod
    def _create_vllm_openai_provider(cls, **kwargs) -> VllmOpenAIClient:
        endpoints = kwargs.get('endpoints') or config.get('llm.note_generator.endpoints', [])
        model = kwargs.get('model') or config.get('llm.note_generator.model', 'Qwen/Qwen2.5-7B-Instruct')
        if not endpoints:
            raise ValueError("vLLM endpoints are required")
        vllm_config = {
            'max_tokens': kwargs.get('max_tokens') or config.get('llm.note_generator.max_tokens', 96),
            'temperature': kwargs.get('temperature') or config.get('llm.note_generator.temperature', 0.2),
            'top_p': kwargs.get('top_p') or config.get('llm.note_generator.top_p', 0.9),
            'timeout': kwargs.get('timeout') or config.get('llm.note_generator.timeout', 15),
            'concurrency_per_endpoint': kwargs.get('concurrency_per_endpoint') or config.get('llm.note_generator.concurrency_per_endpoint', 32),
            'retry': kwargs.get('retry') or config.get('llm.note_generator.retry', {})
        }
        logger.info(f"Creating vLLM OpenAI provider with {len(endpoints)} endpoints, model: {model}")
        return VllmOpenAIClient(endpoints=endpoints, model=model, **vllm_config)
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        available = {}
        for provider_name in cls.PROVIDERS.keys():
            try:
                provider = cls.create_provider(provider_name)
                available[provider_name] = provider.is_available()
                if hasattr(provider, 'cleanup'):
                    provider.cleanup()
            except Exception as e:
                logger.debug(f"Provider {provider_name} not available: {e}")
                available[provider_name] = False
        return available
    
    @classmethod
    def get_best_available_provider(cls) -> str:
        priority_order = ['ollama', 'lmstudio', 'openai']
        available_providers = cls.get_available_providers()
        for provider_name in priority_order:
            if available_providers.get(provider_name, False):
                logger.info(f"Selected best available provider: {provider_name}")
                return provider_name
        logger.warning("No providers available, falling back to ollama")
        return 'ollama'