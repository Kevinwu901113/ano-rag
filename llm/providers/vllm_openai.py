import asyncio
from typing import List, Dict, Any, AsyncGenerator, Generator
from openai import OpenAI, AsyncOpenAI
from loguru import logger


class VLLMOpenAIProvider:
    """vLLM OpenAI 兼容接口 Provider
    
    使用 OpenAI SDK 连接 vLLM 的 OpenAI 兼容接口 (/v1)
    支持 chat 和 stream 方法
    """
    
    def __init__(self, base_url: str = None, model: str = None, api_key: str = None, **kwargs):
        self.base_url = base_url or "http://127.0.0.1:8001/v1"
        self.model = model or "qwen2_5_0_5b"
        self.api_key = api_key or "EMPTY"  # vLLM 需要非空的 api_key
        
        # 从配置或参数获取其他设置
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_tokens = kwargs.get('max_tokens', 256)
        self.top_p = kwargs.get('top_p', 1.0)
        self.timeout = kwargs.get('timeout', 60)
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )
        
        # 异步客户端
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )
        
        logger.info(f"VLLMOpenAIProvider initialized with base_url: {self.base_url}, model: {self.model}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """同步聊天接口
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数，如 temperature, max_tokens 等
            
        Returns:
            str: 模型回复内容
        """
        try:
            # 合并参数
            params = {
                'model': kwargs.get('model', self.model),
                'messages': messages,
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'top_p': kwargs.get('top_p', self.top_p),
                'stream': False
            }
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"VLLMOpenAIProvider chat error: {e}")
            raise
    
    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """同步流式聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            str: 流式返回的文本片段
        """
        try:
            # 合并参数
            params = {
                'model': kwargs.get('model', self.model),
                'messages': messages,
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'top_p': kwargs.get('top_p', self.top_p),
                'stream': True
            }
            
            stream = self.client.chat.completions.create(**params)
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"VLLMOpenAIProvider stream error: {e}")
            raise
    
    async def async_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """异步聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            str: 模型回复内容
        """
        try:
            # 合并参数
            params = {
                'model': kwargs.get('model', self.model),
                'messages': messages,
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'top_p': kwargs.get('top_p', self.top_p),
                'stream': False
            }
            
            response = await self.async_client.chat.completions.create(**params)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"VLLMOpenAIProvider async_chat error: {e}")
            raise
    
    async def async_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """异步流式聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            str: 流式返回的文本片段
        """
        try:
            # 合并参数
            params = {
                'model': kwargs.get('model', self.model),
                'messages': messages,
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'top_p': kwargs.get('top_p', self.top_p),
                'stream': True
            }
            
            stream = await self.async_client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"VLLMOpenAIProvider async_stream error: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """简单文本生成接口（兼容现有代码）
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    async def async_generate(self, prompt: str, **kwargs) -> str:
        """异步文本生成接口
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.async_chat(messages, **kwargs)
    
    def is_available(self) -> bool:
        """检查服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        try:
            # 尝试获取模型列表来检查服务可用性
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.warning(f"VLLMOpenAIProvider availability check failed: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """获取可用模型列表
        
        Returns:
            List[str]: 模型名称列表
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            'provider': 'vllm_openai',
            'base_url': self.base_url,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'is_available': self.is_available()
        }
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self.client, 'close'):
            self.client.close()
        if hasattr(self.async_client, 'close'):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.async_client.aclose())
            else:
                loop.create_task(self.async_client.aclose())
        logger.info("VLLMOpenAIProvider cleaned up")