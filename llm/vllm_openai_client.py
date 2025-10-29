"""
vLLM OpenAI 兼容客户端
支持多端点轮询、并发限流、失败切换和指数回退重试
"""

import asyncio
import json
import random
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from loguru import logger
import httpx
from config import config


@dataclass
class VllmEndpoint:
    """vLLM端点配置"""
    url: str
    semaphore: asyncio.Semaphore
    client: httpx.AsyncClient
    is_healthy: bool = True
    last_error_time: float = 0
    consecutive_errors: int = 0
    served_model_id: Optional[str] = None


class VllmOpenAIClient:
    """
    vLLM OpenAI 兼容客户端
    
    特性：
    - 多端点负载均衡和故障切换
    - 每端点独立并发限流
    - 指数回退重试机制
    - 健康状态监控
    - 与现有LLMClient接口兼容
    """
    
    def __init__(self, endpoints: List[str], model: str, **kwargs):
        """
        初始化vLLM客户端
        
        Args:
            endpoints: vLLM服务端点列表
            model: 模型名称
            **kwargs: 其他配置参数
        """
        self.model = model
        self.endpoints: List[VllmEndpoint] = []
        self.current_endpoint_index = 0
        
        # 配置参数
        self.max_tokens = kwargs.get('max_tokens', 96)
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', 0.9)
        self.timeout = kwargs.get('timeout', 15)
        self.concurrency_per_endpoint = kwargs.get('concurrency_per_endpoint', 32)
        
        # 重试配置
        self.retry_config = kwargs.get('retry', {})
        self.max_attempts = self.retry_config.get('max_attempts', 3)
        self.backoff_base_ms = self.retry_config.get('backoff_base_ms', 200)
        
        # 健康检查配置
        self.health_check_interval = 60  # 秒
        self.error_threshold = 5  # 连续错误阈值
        self.recovery_time = 300  # 恢复时间（秒）
        self.health_check_enabled = kwargs.get(
            'health_check_enabled',
            (config.get('llm.vllm_openai', {}) or {}).get('health_check', {}).get('enabled', True)
        )
        
        # 初始化端点
        self._init_endpoints(endpoints)
        
        logger.info(f"VllmOpenAIClient initialized with {len(self.endpoints)} endpoints")
    
    def _init_endpoints(self, endpoint_urls: List[str]):
        """初始化端点配置"""
        for url in endpoint_urls:
            endpoint = VllmEndpoint(
                url=url,
                semaphore=asyncio.Semaphore(self.concurrency_per_endpoint),
                client=httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout),
                    limits=httpx.Limits(max_connections=self.concurrency_per_endpoint * 2)
                )
            )
            self.endpoints.append(endpoint)
            # 初始快速健康检查：连通性与模型可用性，自动对齐服务端模型名
            try:
                with httpx.Client(timeout=self.timeout) as sync_client:
                    resp = sync_client.get(f"{url}/models", timeout=3.0)
                    resp.raise_for_status()
                    data = resp.json() if hasattr(resp, "json") else {}
                    served_ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
                    if served_ids:
                        # 优先选择与配置匹配的模型，否则取第一个
                        endpoint.served_model_id = self.model if self.model in served_ids else served_ids[0]
                        endpoint.is_healthy = True
                        endpoint.consecutive_errors = 0
                        logger.info(f"Endpoint {url} aligned model to '{endpoint.served_model_id}' (served={served_ids})")
                    else:
                        if self.health_check_enabled:
                            endpoint.is_healthy = False
                            endpoint.consecutive_errors = self.error_threshold
                            endpoint.last_error_time = time.time()
                            logger.warning(f"Endpoint {url} returned empty models; marked unhealthy.")
            except Exception as e:
                if self.health_check_enabled:
                    endpoint.is_healthy = False
                    endpoint.consecutive_errors = self.error_threshold
                    endpoint.last_error_time = time.time()
                    logger.warning(f"Endpoint {url} initial health check failed: {e}; marked unhealthy.")
                else:
                    logger.warning(f"Endpoint {url} initial health check failed: {e}; health-check disabled, keeping endpoint active.")
    
    def _get_next_endpoint(self) -> Optional[VllmEndpoint]:
        """获取下一个可用端点（轮询策略）"""
        if not self.endpoints:
            return None
        
        # 过滤健康的端点
        if self.health_check_enabled:
            healthy_endpoints = [ep for ep in self.endpoints if self._is_endpoint_healthy(ep)]
        else:
            healthy_endpoints = self.endpoints[:]
        
        if not healthy_endpoints:
            # 如果没有健康端点，尝试恢复
            logger.warning("No healthy endpoints available, attempting recovery...")
            if self.health_check_enabled:
                self._attempt_recovery()
                healthy_endpoints = [ep for ep in self.endpoints if self._is_endpoint_healthy(ep)]
            else:
                healthy_endpoints = self.endpoints[:]
        
        if not healthy_endpoints:
            logger.error("All endpoints are unhealthy")
            return None
        
        # 轮询选择
        endpoint = healthy_endpoints[self.current_endpoint_index % len(healthy_endpoints)]
        self.current_endpoint_index += 1
        
        return endpoint
    
    def _is_endpoint_healthy(self, endpoint: VllmEndpoint) -> bool:
        """检查端点是否健康"""
        if not self.health_check_enabled:
            return True
        if endpoint.consecutive_errors < self.error_threshold:
            return True
        
        # 检查是否已过恢复时间
        if time.time() - endpoint.last_error_time > self.recovery_time:
            endpoint.consecutive_errors = 0
            endpoint.is_healthy = True
            logger.info(f"Endpoint {endpoint.url} recovered after {self.recovery_time}s")
            return True
        
        return False
    
    def _refresh_endpoint_health(self, endpoint: VllmEndpoint) -> bool:
        """主动探测 /models，若可达则更新 served_model_id 并标记健康"""
        try:
            with httpx.Client(timeout=self.timeout) as sync_client:
                resp = sync_client.get(f"{endpoint.url}/models", timeout=3.0)
                resp.raise_for_status()
                data = resp.json() if hasattr(resp, "json") else {}
                served_ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
                if served_ids:
                    endpoint.served_model_id = self.model if self.model in served_ids else served_ids[0]
                    endpoint.is_healthy = True
                    endpoint.consecutive_errors = 0
                    return True
        except Exception:
            pass
        return False
    
    def _attempt_recovery(self):
        """尝试恢复不健康的端点：主动探测 /models 并判断模型可用性"""
        if not self.health_check_enabled:
            return
        for endpoint in self.endpoints:
            if not endpoint.is_healthy:
                if self._refresh_endpoint_health(endpoint):
                    logger.info(f"Endpoint {endpoint.url} marked as recovered")
                else:
                    # 渐进衰减错误计数，允许偶尔恢复
                    current_time = time.time()
                    if current_time - endpoint.last_error_time > 60:
                        endpoint.consecutive_errors = max(0, endpoint.consecutive_errors - 1)
    
    def _mark_endpoint_error(self, endpoint: VllmEndpoint, error: Exception):
        """标记端点错误"""
        if self.health_check_enabled:
            endpoint.consecutive_errors += 1
            endpoint.last_error_time = time.time()
            
            if endpoint.consecutive_errors >= self.error_threshold:
                endpoint.is_healthy = False
                logger.warning(f"Endpoint {endpoint.url} marked as unhealthy after {endpoint.consecutive_errors} errors")
        
        logger.debug(f"Endpoint {endpoint.url} error: {error}")
    
    def _mark_endpoint_success(self, endpoint: VllmEndpoint):
        """标记端点成功"""
        if self.health_check_enabled and endpoint.consecutive_errors > 0:
            endpoint.consecutive_errors = 0
            endpoint.is_healthy = True
    
    async def _make_request(self, endpoint: VllmEndpoint, payload: Dict[str, Any]) -> Dict[str, Any]:
        """向指定端点发送请求"""
        async with endpoint.semaphore:  # 并发限流
            url = f"{endpoint.url}/chat/completions"
            # 按端点对齐模型名
            payload["model"] = endpoint.served_model_id or self.model
            try:
                response = await endpoint.client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                self._mark_endpoint_success(endpoint)
                return result
                
            except Exception as e:
                self._mark_endpoint_error(endpoint, e)
                raise
    
    async def _chat_completion_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """带重试的聊天完成请求"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
            "stream": False
        }
        
        last_exception = None
        
        for attempt in range(self.max_attempts):
            endpoint = self._get_next_endpoint()
            if not endpoint:
                raise RuntimeError("No available endpoints")
            
            try:
                result = await self._make_request(endpoint, payload)
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    # 指数回退
                    backoff_time = (self.backoff_base_ms * (2 ** attempt)) / 1000.0
                    backoff_time += random.uniform(0, backoff_time * 0.1)  # 添加抖动
                    logger.debug(f"Request failed (attempt {attempt + 1}/{self.max_attempts}), "
                               f"retrying in {backoff_time:.2f}s: {e}")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"All retry attempts failed: {e}")
        
        raise last_exception or RuntimeError("Request failed after all retries")
    
    async def chat_one(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        单个聊天完成请求
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            result = await self._chat_completion_with_retry(messages, **kwargs)
            content = result["choices"][0]["message"]["content"]
            return content.strip()
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def chat_many(self, prompts: List[str], system_prompt: str = None, **kwargs) -> List[str]:
        """
        批量聊天完成请求
        
        Args:
            prompts: 用户提示列表
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容列表
        """
        tasks = []
        for prompt in prompts:
            task = self.chat_one(prompt, system_prompt, **kwargs)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch request {i} failed: {result}")
                    final_results.append("")  # 返回空字符串作为失败标记
                else:
                    final_results.append(result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch chat completion failed: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        同步生成接口（兼容现有代码）
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        try:
            # 在新的事件循环中运行异步方法
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.chat_one(prompt, system_prompt, **kwargs))
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Sync generation failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查所有端点"""
        results = {}
        
        for i, endpoint in enumerate(self.endpoints):
            try:
                url = f"{endpoint.url}/models"
                response = await endpoint.client.get(url, timeout=5.0)
                response.raise_for_status()
                data = response.json() if hasattr(response, "json") else {}
                served_ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
                # 有可用模型即视为健康，并报告对齐的模型名
                if served_ids:
                    if endpoint.served_model_id is None:
                        endpoint.served_model_id = self.model if self.model in served_ids else served_ids[0]
                    status = {
                        "url": endpoint.url,
                        "status": "healthy",
                        "served_model_id": endpoint.served_model_id,
                        "consecutive_errors": endpoint.consecutive_errors
                    }
                else:
                    status = {
                        "url": endpoint.url,
                        "status": "unhealthy",
                        "error": "empty_models",
                        "consecutive_errors": endpoint.consecutive_errors
                    }
                results[f"endpoint_{i}"] = status
            
            except Exception as e:
                results[f"endpoint_{i}"] = {
                    "url": endpoint.url,
                    "status": "unhealthy",
                    "error": str(e),
                    "consecutive_errors": endpoint.consecutive_errors
                }
        
        return results
    
    async def close(self):
        """关闭所有客户端连接"""
        for endpoint in self.endpoints:
            await endpoint.client.aclose()
        
        logger.info("VllmOpenAIClient closed")
    
    def __del__(self):
        """析构函数"""
        try:
            # 尝试清理资源
            for endpoint in self.endpoints:
                if hasattr(endpoint.client, '_transport') and endpoint.client._transport:
                    # 如果连接还在，尝试关闭
                    pass
        except Exception:
            pass
