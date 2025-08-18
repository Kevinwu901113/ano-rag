from typing import Dict, Any, Optional
from loguru import logger
from config import config

# Import providers
from .providers.vllm_openai import VLLMOpenAIProvider
from .ollama_client import OllamaClient
from .multi_ollama_client import MultiOllamaClient
from .openai_client import OpenAIClient


class LLMFactory:
    """LLM Provider 工厂类
    
    负责根据配置创建和管理不同的 LLM Provider
    支持 vLLM、Ollama、OpenAI 等多种 Provider
    """
    
    # 注册的 Provider 类型
    PROVIDERS = {
        'vllm_openai': VLLMOpenAIProvider,
        'ollama': MultiOllamaClient,  # 使用多实例 Ollama 客户端
        'openai': OpenAIClient,
    }
    
    @classmethod
    def create_provider(cls, provider_type: str = None, route_name: str = None, **kwargs) -> Any:
        """创建 LLM Provider 实例
        
        Args:
            provider_type: Provider 类型，如 'vllm_openai', 'ollama', 'openai'
            route_name: 路由名称，用于获取特定路由的配置
            **kwargs: 额外的参数
            
        Returns:
            Provider 实例
        """
        # 从配置获取 provider 类型
        if provider_type is None:
            provider_type = config.get('llm.provider', 'ollama')
        
        # 检查 provider 是否支持
        if provider_type not in cls.PROVIDERS:
            logger.warning(f"Unsupported provider: {provider_type}, falling back to ollama")
            provider_type = 'ollama'
        
        provider_class = cls.PROVIDERS[provider_type]
        
        try:
            # 根据不同的 provider 类型创建实例
            if provider_type == 'vllm_openai':
                return cls._create_vllm_provider(route_name, **kwargs)
            elif provider_type == 'ollama':
                return cls._create_ollama_provider(**kwargs)
            elif provider_type == 'openai':
                return cls._create_openai_provider(**kwargs)
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")
                
        except Exception as e:
            logger.error(f"Failed to create {provider_type} provider: {e}")
            # 回退到 Ollama
            if provider_type != 'ollama':
                logger.info("Falling back to Ollama provider")
                return cls._create_ollama_provider(**kwargs)
            raise
    
    @classmethod
    def _create_vllm_provider(cls, route_name: str = None, **kwargs) -> VLLMOpenAIProvider:
        """创建 vLLM OpenAI Provider
        
        Args:
            route_name: 路由名称
            **kwargs: 额外参数
            
        Returns:
            VLLMOpenAIProvider 实例
        """
        # 获取默认路由
        if route_name is None:
            route_name = config.get('llm.default_route', 'tiny_qwen')
        
        # 获取路由配置
        route_config = config.get(f'llm.routes.{route_name}', {})
        if not route_config:
            raise ValueError(f"Route '{route_name}' not found in configuration")
        
        # 合并参数
        params = {
            'base_url': route_config.get('base_url', 'http://127.0.0.1:8001/v1'),
            'model': route_config.get('model', 'qwen2_5_0_5b'),
            'api_key': route_config.get('api_key', 'EMPTY'),
            'temperature': config.get('llm.params.temperature', 0.0),
            'max_tokens': config.get('llm.params.max_tokens', 256),
            'top_p': config.get('llm.params.top_p', 1.0),
            'timeout': config.get('llm.params.timeout', 60),
        }
        
        # 覆盖用户提供的参数
        params.update(kwargs)
        
        logger.info(f"Creating vLLM provider with route: {route_name}")
        return VLLMOpenAIProvider(**params)
    
    @classmethod
    def _create_ollama_provider(cls, **kwargs) -> MultiOllamaClient:
        """创建 Ollama Provider
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            MultiOllamaClient 实例
        """
        # 获取 Ollama 配置
        base_url = kwargs.get('base_url') or config.get('llm.ollama.base_url', 'http://localhost:11434')
        model = kwargs.get('model') or config.get('llm.ollama.model', 'llama3.1:8b')
        
        logger.info(f"Creating Ollama provider with model: {model}")
        return MultiOllamaClient(base_url=base_url, model=model)
    
    @classmethod
    def _create_openai_provider(cls, **kwargs) -> OpenAIClient:
        """创建 OpenAI Provider
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            OpenAIClient 实例
        """
        # 获取 OpenAI 配置
        api_key = kwargs.get('api_key') or config.get('llm.openai.api_key')
        base_url = kwargs.get('base_url') or config.get('llm.openai.base_url')
        model = kwargs.get('model') or config.get('llm.openai.model', 'gpt-3.5-turbo')
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        logger.info(f"Creating OpenAI provider with model: {model}")
        return OpenAIClient(api_key=api_key, model=model, base_url=base_url)
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """获取可用的 Provider 列表
        
        Returns:
            Dict[str, bool]: Provider 名称和可用性状态
        """
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
        """获取最佳可用的 Provider
        
        Returns:
            str: 最佳可用的 Provider 名称
        """
        # 优先级顺序：vllm_openai > ollama > openai
        priority_order = ['vllm_openai', 'ollama', 'openai']
        
        available_providers = cls.get_available_providers()
        
        for provider_name in priority_order:
            if available_providers.get(provider_name, False):
                logger.info(f"Selected best available provider: {provider_name}")
                return provider_name
        
        # 如果都不可用，返回默认的 ollama
        logger.warning("No providers available, falling back to ollama")
        return 'ollama'