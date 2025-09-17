from typing import Dict, Any, Optional
from loguru import logger
from config import config

# Import providers
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .lmstudio_client import LMStudioClient


class LLMFactory:
    """LLM Provider 工厂类
    
    负责根据配置创建和管理不同的 LLM Provider
    支持 Ollama、OpenAI 等多种 Provider
    """
    
    # 注册的 Provider 类型
    PROVIDERS = {
        'ollama': OllamaClient,  # 使用单实例 Ollama 客户端
        'openai': OpenAIClient,
        'lmstudio': LMStudioClient,  # 使用统一的 LM Studio 客户端（支持单实例和并发）
    }
    
    @classmethod
    def create_provider(cls, provider_type: str = None, route_name: str = None, **kwargs) -> Any:
        """创建 LLM Provider 实例
        
        Args:
            provider_type: Provider 类型，如 'ollama', 'openai'
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
            if provider_type == 'ollama':
                return cls._create_ollama_provider(**kwargs)
            elif provider_type == 'openai':
                return cls._create_openai_provider(**kwargs)
            elif provider_type == 'lmstudio':
                return cls._create_lmstudio_provider(**kwargs)
            elif provider_type == 'multi_model':
                return cls._create_multi_model_provider(**kwargs)
            elif provider_type == 'hybrid_llm':
                return cls._create_hybrid_llm_provider(**kwargs)
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
    def _create_ollama_provider(cls, **kwargs) -> OllamaClient:
        """创建 Ollama Provider
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            OllamaClient 实例
        """
        # 获取 Ollama 配置
        base_url = kwargs.get('base_url') or config.get('llm.ollama.base_url', 'http://localhost:11434')
        model = kwargs.get('model') or config.get('llm.ollama.model', 'gpt-oss:latest')
        
        logger.info(f"Creating Ollama provider with model: {model}")
        return OllamaClient(base_url=base_url, model=model)
    
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
    def _create_lmstudio_provider(cls, **kwargs) -> LMStudioClient:
        """创建 LM Studio Provider
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            LMStudioClient 实例（统一客户端，自动支持单实例和多实例模式）
        """
        # 获取 LM Studio 配置
        base_url = kwargs.get('base_url') or config.get('llm.lmstudio.base_url')
        model = kwargs.get('model') or config.get('llm.lmstudio.model', 'default-model')
        port = kwargs.get('port') or config.get('llm.lmstudio.port', 1234)
        
        # 检查是否启用并发模式
        concurrent_enabled = config.get('llm.lmstudio.concurrent.enabled', False)
        mode = 'concurrent' if concurrent_enabled else 'single-instance'
        
        logger.info(f"Creating LM Studio provider in {mode} mode with model: {model} on port: {port}")
        return LMStudioClient(base_url=base_url, model=model, port=port)
    
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
        # 优先级顺序：ollama > lmstudio > openai
        priority_order = ['ollama', 'lmstudio', 'openai']
        
        available_providers = cls.get_available_providers()
        
        for provider_name in priority_order:
            if available_providers.get(provider_name, False):
                logger.info(f"Selected best available provider: {provider_name}")
                return provider_name
        
        # 如果都不可用，返回默认的 ollama
        logger.warning("No providers available, falling back to ollama")
        return 'ollama'