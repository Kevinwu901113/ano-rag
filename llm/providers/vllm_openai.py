import asyncio
import time
import random
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator
from openai import OpenAI, AsyncOpenAI
from loguru import logger
from config import config
import requests


class VLLMOpenAIProvider:
    """vLLM OpenAI 兼容接口 Provider
    
    使用 OpenAI SDK 连接 vLLM 的 OpenAI 兼容接口 (/v1)
    支持 chat 和 stream 方法，包含健康检查、负载均衡、重试机制
    """
    
    def __init__(self, routes: Dict[str, Dict] = None, default_route: str = None, 
                 fallback_route: str = None, lb_policy: str = "round_robin", **kwargs):
        
        # 路由配置
        self.routes = routes or {}
        self.default_route = default_route or "gpt20_a"
        self.fallback_route = fallback_route
        self.lb_policy = lb_policy
        
        # 全局参数
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.top_p = kwargs.get('top_p', 1.0)
        self.timeout = kwargs.get('timeout', 60)
        self.max_retries = kwargs.get('max_retries', 3)
        
        # 负载均衡状态
        self._route_index = 0
        self._route_health = {}
        self._route_latency = {}
        
        # 初始化客户端缓存
        self._clients = {}
        self._async_clients = {}
        
        # 初始化路由健康状态
        self._initialize_routes()
        
        logger.info(f"VLLMOpenAIProvider initialized with {len(self.routes)} routes, default: {self.default_route}")
    
    def _initialize_routes(self):
        """初始化路由健康状态"""
        for route_name in self.routes.keys():
            self._route_health[route_name] = True
            self._route_latency[route_name] = float('inf')
            
    def _get_client(self, route_name: str) -> OpenAI:
        """获取或创建同步客户端"""
        if route_name not in self._clients:
            route_config = self.routes[route_name]
            self._clients[route_name] = OpenAI(
                base_url=route_config['base_url'],
                api_key=route_config.get('api_key', 'EMPTY'),
                timeout=route_config.get('timeout', self.timeout)
            )
        return self._clients[route_name]
    
    def _get_async_client(self, route_name: str) -> AsyncOpenAI:
        """获取或创建异步客户端"""
        if route_name not in self._async_clients:
            route_config = self.routes[route_name]
            self._async_clients[route_name] = AsyncOpenAI(
                base_url=route_config['base_url'],
                api_key=route_config.get('api_key', 'EMPTY'),
                timeout=route_config.get('timeout', self.timeout)
            )
        return self._async_clients[route_name]
    
    def check_health(self, base_url: str) -> bool:
        """检查单个实例的健康状态"""
        try:
            # 移除 /v1 后缀以避免重复
            clean_url = base_url.rstrip('/v1').rstrip('/')
            models_url = f"{clean_url}/v1/models"
            
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return 'data' in data and len(data['data']) > 0
            return False
        except Exception as e:
            logger.debug(f"Health check failed for {base_url}: {e}")
            return False
    
    def _update_route_health(self, route_name: str, is_healthy: bool, latency: float = None):
        """更新路由健康状态"""
        self._route_health[route_name] = is_healthy
        if latency is not None:
            self._route_latency[route_name] = latency
    
    def _get_healthy_routes(self) -> List[str]:
        """获取健康的路由列表"""
        return [route for route, healthy in self._route_health.items() if healthy]
    
    def _select_route(self, preferred_route: str = None) -> str:
        """根据负载均衡策略选择路由"""
        if preferred_route and preferred_route in self.routes and self._route_health.get(preferred_route, False):
            return preferred_route
        
        healthy_routes = self._get_healthy_routes()
        if not healthy_routes:
            # 如果没有健康的路由，尝试使用默认路由
            if self.default_route in self.routes:
                logger.warning(f"No healthy routes available, trying default route: {self.default_route}")
                return self.default_route
            # 如果默认路由也不可用，使用第一个可用路由
            if self.routes:
                route_name = list(self.routes.keys())[0]
                logger.warning(f"No healthy routes available, trying first route: {route_name}")
                return route_name
            raise RuntimeError("No routes available")
        
        if self.lb_policy == "round_robin":
            # 轮询策略
            self._route_index = (self._route_index + 1) % len(healthy_routes)
            return healthy_routes[self._route_index]
        elif self.lb_policy == "least_latency":
            # 最低延迟策略
            return min(healthy_routes, key=lambda r: self._route_latency.get(r, float('inf')))
        else:
            # 随机策略
            return random.choice(healthy_routes)
    
    def _execute_with_retry(self, func, route_name: str, *args, **kwargs):
        """带重试的执行函数"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                
                # 更新健康状态和延迟
                self._update_route_health(route_name, True, latency)
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for route {route_name}: {e}")
                
                # 标记路由为不健康
                self._update_route_health(route_name, False)
                
                if attempt < self.max_retries - 1:
                    # 指数退避
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
        
        raise last_exception
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """同步聊天接口
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数，如 temperature, max_tokens 等
            
        Returns:
            str: 模型回复内容
        """
        # 选择路由
        preferred_route = kwargs.pop('route', None)
        route_name = self._select_route(preferred_route)
        route_config = self.routes[route_name]
        
        # 获取客户端
        client = self._get_client(route_name)
        
        # 合并参数
        params = {
            'model': kwargs.get('model', route_config['model']),
            'messages': messages,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
            'stream': False
        }
        
        def _chat_call():
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        
        try:
            return self._execute_with_retry(_chat_call, route_name)
        except Exception as e:
            logger.error(f"VLLMOpenAIProvider chat error on route {route_name}: {e}")
            # 尝试回退路由
            if self.fallback_route and self.fallback_route != route_name:
                logger.info(f"Trying fallback route: {self.fallback_route}")
                kwargs['route'] = self.fallback_route
                return self.chat(messages, **kwargs)
            raise
    
    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """同步流式聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            str: 流式返回的文本片段
        """
        # 选择路由
        preferred_route = kwargs.pop('route', None)
        route_name = self._select_route(preferred_route)
        route_config = self.routes[route_name]
        
        # 获取客户端
        client = self._get_client(route_name)
        
        # 合并参数
        params = {
            'model': kwargs.get('model', route_config['model']),
            'messages': messages,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
            'stream': True
        }
        
        try:
            start_time = time.time()
            stream = client.chat.completions.create(**params)
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
            
            # 更新健康状态
            latency = time.time() - start_time
            self._update_route_health(route_name, True, latency)
            
        except Exception as e:
            logger.error(f"VLLMOpenAIProvider stream error on route {route_name}: {e}")
            self._update_route_health(route_name, False)
            
            # 尝试回退路由
            if self.fallback_route and self.fallback_route != route_name:
                logger.info(f"Trying fallback route for stream: {self.fallback_route}")
                kwargs['route'] = self.fallback_route
                yield from self.stream(messages, **kwargs)
            else:
                raise
    
    async def async_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """异步聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            str: 模型回复内容
        """
        # 选择路由
        preferred_route = kwargs.pop('route', None)
        route_name = self._select_route(preferred_route)
        route_config = self.routes[route_name]
        
        # 获取异步客户端
        async_client = self._get_async_client(route_name)
        
        # 合并参数
        params = {
            'model': kwargs.get('model', route_config['model']),
            'messages': messages,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
            'stream': False
        }
        
        try:
            start_time = time.time()
            response = await async_client.chat.completions.create(**params)
            latency = time.time() - start_time
            
            # 更新健康状态
            self._update_route_health(route_name, True, latency)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"VLLMOpenAIProvider async_chat error on route {route_name}: {e}")
            self._update_route_health(route_name, False)
            
            # 尝试回退路由
            if self.fallback_route and self.fallback_route != route_name:
                logger.info(f"Trying fallback route for async_chat: {self.fallback_route}")
                kwargs['route'] = self.fallback_route
                return await self.async_chat(messages, **kwargs)
            raise
    
    async def async_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """异步流式聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            str: 流式返回的文本片段
        """
        # 选择路由
        preferred_route = kwargs.pop('route', None)
        route_name = self._select_route(preferred_route)
        route_config = self.routes[route_name]
        
        # 获取异步客户端
        async_client = self._get_async_client(route_name)
        
        # 合并参数
        params = {
            'model': kwargs.get('model', route_config['model']),
            'messages': messages,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
            'stream': True
        }
        
        try:
            start_time = time.time()
            stream = await async_client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
            
            # 更新健康状态
            latency = time.time() - start_time
            self._update_route_health(route_name, True, latency)
            
        except Exception as e:
            logger.error(f"VLLMOpenAIProvider async_stream error on route {route_name}: {e}")
            self._update_route_health(route_name, False)
            
            # 尝试回退路由
            if self.fallback_route and self.fallback_route != route_name:
                logger.info(f"Trying fallback route for async_stream: {self.fallback_route}")
                kwargs['route'] = self.fallback_route
                async for chunk in self.async_stream(messages, **kwargs):
                    yield chunk
            else:
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
        # 检查所有路由的健康状态
        for route_name, route_config in self.routes.items():
            if self.check_health(route_config['base_url']):
                self._update_route_health(route_name, True)
                return True
            else:
                self._update_route_health(route_name, False)
        
        return False
    
    def list_models(self) -> List[str]:
        """获取可用模型列表
        
        Returns:
            List[str]: 模型名称列表
        """
        models = set()
        for route_name, route_config in self.routes.items():
            if self._route_health.get(route_name, False):
                try:
                    client = self._get_client(route_name)
                    route_models = client.models.list()
                    models.update([model.id for model in route_models.data])
                except Exception as e:
                    logger.debug(f"Failed to list models for route {route_name}: {e}")
        
        return list(models)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        healthy_routes = self._get_healthy_routes()
        return {
            'provider': 'vllm_openai',
            'routes': self.routes,
            'default_route': self.default_route,
            'fallback_route': self.fallback_route,
            'lb_policy': self.lb_policy,
            'healthy_routes': healthy_routes,
            'route_health': self._route_health,
            'route_latency': self._route_latency,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'is_available': len(healthy_routes) > 0
        }
    
    def cleanup(self):
        """清理资源"""
        for client in self._clients.values():
            if hasattr(client, 'close'):
                try:
                    client.close()
                except Exception as e:
                    logger.debug(f"Error closing client: {e}")
        
        for async_client in self._async_clients.values():
            if hasattr(async_client, 'aclose'):
                try:
                    asyncio.create_task(async_client.aclose())
                except Exception as e:
                    logger.debug(f"Error closing async client: {e}")
        
        self._clients.clear()
        self._async_clients.clear()
        logger.info("VLLMOpenAIProvider cleaned up")