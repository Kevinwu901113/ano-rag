from typing import Dict, Any, Optional
from loguru import logger
from config import config

# Import providers
# from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .lmstudio_client import LMStudioClient
from .vllm_openai_client import VllmOpenAIClient


class LLMFactory:
    """LLM Provider 工厂类
    
    负责根据配置创建和管理不同的 LLM Provider
    支持 OpenAI、LM Studio、vLLM-OpenAI 等 Provider
    """
    
    # 注册的 Provider 类型（移除 ollama）
    PROVIDERS = {
        'openai': OpenAIClient,
        'lmstudio': LMStudioClient,
        'vllm-openai': VllmOpenAIClient,
    }

    @classmethod
    def create_provider(cls, provider_type: str = None, route_name: str = None, **kwargs) -> Any:
        """创建 LLM Provider 实例"""
        if provider_type is None:
            provider_type = config.get('llm.provider', 'lmstudio')
        if provider_type not in cls.PROVIDERS:
            logger.warning(f"Unsupported provider: {provider_type}, falling back to lmstudio")
            provider_type = 'lmstudio'
        
        try:
            if provider_type == 'openai':
                return cls._create_openai_provider(**kwargs)
            elif provider_type == 'lmstudio':
                return cls._create_lmstudio_provider(**kwargs)
            elif provider_type == 'vllm-openai':
                return cls._create_vllm_openai_provider(**kwargs)
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")
        except Exception as e:
            logger.error(f"Failed to create {provider_type} provider: {e}")
            # vLLM 客户端不允许回退到 LM Studio，直接抛出异常以便上游硬失败
            if provider_type == 'vllm-openai':
                raise
            # 其他 Provider（如 openai）仍可回退到 LM Studio
            if provider_type != 'lmstudio':
                logger.info("Falling back to LMStudio provider")
                return cls._create_lmstudio_provider(**kwargs)
            raise

    @classmethod
    def _create_openai_provider(cls, **kwargs) -> OpenAIClient:
        base_url = kwargs.get('base_url') or config.get('llm.openai.base_url')
        model = kwargs.get('model') or config.get('llm.openai.model')
        api_key = kwargs.get('api_key') or config.get('llm.openai.api_key')
        logger.info(f"Creating OpenAI provider with model: {model}")
        return OpenAIClient(base_url=base_url, model=model, api_key=api_key)

    @classmethod
    def _create_lmstudio_provider(cls, **kwargs) -> LMStudioClient:
        port = kwargs.get('port') or config.get('llm.lmstudio.port', 1234)
        base_url = kwargs.get('base_url') or config.get('llm.lmstudio.base_url', f"http://localhost:{port}/v1")
        model = kwargs.get('model') or config.get('llm.lmstudio.model')
        logger.info(f"Creating LMStudio provider with model: {model}")
        return LMStudioClient(base_url=base_url, model=model, port=port)

    @classmethod
    def _create_vllm_openai_provider(cls, **kwargs) -> VllmOpenAIClient:
        cfg_prefix = 'llm.note_generator'
        endpoints = kwargs.get('endpoints') or config.get(f'{cfg_prefix}.endpoints', [])
        model = kwargs.get('model') or config.get(f'{cfg_prefix}.model')
        timeout = kwargs.get('timeout') or config.get(f'{cfg_prefix}.timeout', 30)
        # 透传可选的生成参数（如未提供则由客户端使用默认值）
        max_tokens = kwargs.get('max_tokens') or config.get(f'{cfg_prefix}.max_tokens')
        temperature = kwargs.get('temperature') or config.get(f'{cfg_prefix}.temperature')
        top_p = kwargs.get('top_p') or config.get(f'{cfg_prefix}.top_p')
        concurrency_per_endpoint = kwargs.get('concurrency_per_endpoint') or config.get(f'{cfg_prefix}.concurrency_per_endpoint')
        retry = kwargs.get('retry') or config.get(f'{cfg_prefix}.retry')
        logger.info(f"Creating vLLM-OpenAI provider with model: {model}")
        return VllmOpenAIClient(
            endpoints=endpoints,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            concurrency_per_endpoint=concurrency_per_endpoint,
            retry=retry,
        )

    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        providers_status: Dict[str, bool] = {}
        for name, cls_ in cls.PROVIDERS.items():
            try:
                # 尝试仅初始化后检查可用性（避免真实请求）
                client = cls.create_provider(name)
                check_fn = getattr(client, 'is_available', None)
                providers_status[name] = bool(check_fn and check_fn())
            except Exception as e:
                logger.warning(f"Provider {name} availability check failed: {e}")
                providers_status[name] = False
        return providers_status

    @classmethod
    def get_best_available_provider(cls) -> str:
        priority_order = ['lmstudio', 'openai']
        available_providers = cls.get_available_providers()
        for provider_name in priority_order:
            if available_providers.get(provider_name, False):
                logger.info(f"Selected best available provider: {provider_name}")
                return provider_name
        logger.warning("No providers available, falling back to lmstudio")
        return 'lmstudio'