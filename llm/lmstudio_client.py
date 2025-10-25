import asyncio
import json
import random
import time
import threading
from typing import Any, Dict, Iterator, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

import openai
import requests
from loguru import logger

from config import config
from .prompts import (
    EVALUATE_ANSWER_SYSTEM_PROMPT,
    EVALUATE_ANSWER_PROMPT,
    get_final_answer_prompts,
)
from .streaming_early_stop import create_early_stop_stream


class LoadBalancingStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_BUSY = "least_busy"
    CONCURRENT_OPTIMAL = "concurrent_optimal"  # 针对并发处理优化的策略


@dataclass
class LMStudioInstance:
    """LM Studio实例配置"""
    base_url: str
    model: str
    port: int
    client: Optional[openai.OpenAI] = None
    is_healthy: bool = True
    last_health_check: float = 0
    active_requests: int = 0
    total_requests: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        if self.client is None:
            self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        client_kwargs = {
            "api_key": "lm-studio",  # LM Studio doesn't require a real API key
            "base_url": self.base_url,
            "timeout": config.get("llm.lmstudio.timeout", 60),
            "max_retries": config.get("llm.lmstudio.max_retries", 3),
        }
        self.client = openai.OpenAI(**client_kwargs)


class LMStudioGenerationError(RuntimeError):
    """Error raised when LM Studio fails to return a completion."""

    def __init__(
        self,
        message: str,
        *,
        original_exception: Exception | None = None,
        is_transport_error: bool = False,
        is_timeout: bool = False,
    ) -> None:
        super().__init__(message)
        self.original_exception = original_exception
        self.is_transport_error = is_transport_error
        self.is_timeout = is_timeout

    def __str__(self) -> str:  # pragma: no cover - delegated to base repr when unused
        base_msg = super().__str__()
        if self.original_exception is None:
            return base_msg
        return f"{base_msg} (caused by {self.original_exception!r})"


class LMStudioClient:
    """LM Studio客户端
    
    利用LM Studio的内置队列功能实现并发处理：
    - 单一模型实例：连接到LM Studio加载的模型
    - 并发处理：利用LM Studio的请求队列机制处理多个并发请求
    - 线程池：使用ThreadPoolExecutor管理并发请求
    
    提供统一的API接口，支持并发原子笔记生成。
    """

    def __init__(self, base_url: str = None, model: str = None, port: int = None):
        # 基础配置
        self.temperature = config.get("llm.lmstudio.temperature", 0.7)
        self.max_tokens = config.get("llm.lmstudio.max_tokens", 4096)
        self.timeout = config.get("llm.lmstudio.timeout", 60)
        self.max_retries = config.get("llm.lmstudio.max_retries", 3)
        self.top_p = config.get("llm.lmstudio.top_p")
        self.stop_sequences = config.get("llm.lmstudio.stop")
        
        # 默认选项
        self.default_options = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            self.default_options["top_p"] = self.top_p
        if self.stop_sequences:
            self.default_options["stop"] = self.stop_sequences
        
        # 初始化单一LM Studio连接
        self._init_connection(base_url, model, port)
        
        # 并发处理配置
        self.concurrent_enabled = config.get("llm.lmstudio.multiple_instances.enabled", False)
        if self.concurrent_enabled:
            self.max_concurrent_requests = config.get("llm.lmstudio.multiple_instances.target_instance_count", 2)
            self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
            logger.info(f"LM Studio concurrent processing enabled with {self.max_concurrent_requests} max concurrent requests")
        else:
            self.max_concurrent_requests = 1
            self.executor = None
        
        logger.info(f"Initialized LM Studio client: {'concurrent' if self.concurrent_enabled else 'single-threaded'} mode")
    
    def _init_connection(self, base_url: str = None, model: str = None, port: int = None):
        """初始化LM Studio连接"""
        port = port or config.get("llm.lmstudio.port", 1234)
        base_url = base_url or config.get("llm.lmstudio.base_url", f"http://localhost:{port}/v1")
        model = model or config.get("llm.lmstudio.model", "default-model")
        
        # 创建单一实例
        self.instance = LMStudioInstance(base_url=base_url, model=model, port=port)
        
        # 为了兼容性，保留instances列表
        self.instances = [self.instance]
        
        logger.info(f"Connected to LM Studio: {base_url} (model: {model})")
    
    def generate_concurrent(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        **kwargs,
    ) -> List[str]:
        """并发生成多个响应"""
        if not self.concurrent_enabled or len(prompts) == 1:
            # 单线程处理
            return [
                self.generate(prompt, system_prompt=system_prompt, **kwargs)
                for prompt in prompts
            ]

        # 使用线程池并发处理
        executor = self.executor or ThreadPoolExecutor(
            max_workers=min(self.max_concurrent_requests, len(prompts)) or 1
        )
        created_local_executor = executor is not self.executor

        futures = {
            executor.submit(
                self.generate,
                prompt,
                system_prompt,
                **kwargs,
            ): idx
            for idx, prompt in enumerate(prompts)
        }

        results: List[str] = [""] * len(prompts)
        try:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Concurrent generation failed: {e}")
                    results[idx] = ""
        finally:
            if created_local_executor:
                executor.shutdown(wait=True)

        return results

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成单个响应"""
        try:
            # 提取system_prompt参数，避免传递给OpenAI API
            options = {k: v for k, v in self.default_options.items() if v is not None}
            for key, value in kwargs.items():
                if key == 'system_prompt' or value is None:
                    continue
                options[key] = value

            # 准备消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # 调用OpenAI客户端生成响应
            response = self.instance.client.chat.completions.create(
                model=self.instance.model,
                messages=messages,
                **options
            )

            message_content = response.choices[0].message.content
            return message_content.strip() if message_content else ""

        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")

            # 判断异常类型，以便上游根据错误类型采取不同策略
            request_exception_types = (requests.RequestException,)
            openai_timeout_type = getattr(openai, "APITimeoutError", None)
            openai_connection_type = getattr(openai, "APIConnectionError", None)

            timeout_types = (requests.Timeout,)
            if isinstance(openai_timeout_type, type):
                timeout_types = timeout_types + (openai_timeout_type,)

            transport_types = request_exception_types
            if isinstance(openai_connection_type, type):
                transport_types = transport_types + (openai_connection_type,)

            is_timeout = isinstance(e, timeout_types) or "timeout" in str(e).lower()
            is_transport_error = isinstance(e, transport_types) or is_timeout

            raise LMStudioGenerationError(
                "LM Studio generation failed",
                original_exception=e,
                is_transport_error=is_transport_error,
                is_timeout=is_timeout,
            ) from e

    async def chat_many(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        **kwargs,
    ) -> List[str]:
        """提供异步批量对话接口，兼容需要chat_many的上游组件"""

        if not prompts:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_concurrent(
                prompts,
                system_prompt=system_prompt,
                **kwargs,
            ),
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """生成流式响应"""
        try:
            # 合并默认选项和传入的参数
            options = {k: v for k, v in self.default_options.items() if v is not None}
            for key, value in kwargs.items():
                if value is None:
                    continue
                options[key] = value
            
            # 准备消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # 调用OpenAI客户端生成流式响应
            response = self.instance.client.chat.completions.create(
                model=self.instance.model,
                messages=messages,
                stream=True,
                **options
            )
            
            # 创建原始流
            def original_stream():
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
            # 检查是否启用早停机制
            stream_early_stop = config.get('notes_llm.stream_early_stop', False)
            if stream_early_stop:
                sentinel_char = config.get('notes_llm.sentinel_char', '~')
                # 应用早停机制
                yield from create_early_stop_stream(original_stream(), sentinel_char, 16)
            else:
                # 直接返回原始流
                yield from original_stream()
                    
        except Exception as e:
            logger.error(f"LM Studio stream generation failed: {e}")
            yield ""
    
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """与模型进行对话"""
        try:
            # 合并默认选项和传入的参数
            options = {**self.default_options, **kwargs}
            
            response = self.instance.client.chat.completions.create(
                model=self.instance.model,
                messages=messages,
                stream=stream,
                **options,
            )

            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return self._clean_response(response.choices[0].message.content or "")
        
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            if stream:
                return iter([])
            return ""
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> List[str]:
        """批量生成响应"""
        if len(prompts) <= 1:
            # 单个提示，使用串行处理
            results = []
            for prompt in prompts:
                try:
                    result = self.generate(prompt, system_prompt, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch generation failed for prompt: {e}")
                    results.append("")
            return results
        
        # 使用线程池并行处理
        if self.executor:
            return self.generate_concurrent(
                prompts,
                system_prompt=system_prompt,
                **kwargs,
            )
        else:
            # 回退到串行处理
            results = []
            for prompt in prompts:
                try:
                    result = self.generate(prompt, system_prompt, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch generation failed for prompt: {e}")
                    results.append("")
            return results
    
    def generate_final_answer(
        self,
        prompt_or_context: str,
        query: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> str:
        """Generate final answer; accepts either a preformatted prompt or raw context."""
        system_prompt, prompt_template = get_final_answer_prompts(dataset)

        prompt_text = prompt_or_context or ""
        # Heuristic: if the incoming text does not look preformatted, rebuild with template
        if "OUTPUT FORMAT" not in prompt_text and query is not None:
            prompt_text = prompt_template.format(context=prompt_text, query=query)

        return self.generate(prompt_text, system_prompt)
    
    def evaluate_answer(self, query: str, context: str, answer: str) -> Dict[str, float]:
        """Evaluate answer quality."""
        prompt = EVALUATE_ANSWER_PROMPT.format(query=query, context=context, answer=answer)
        response = self.generate(prompt, EVALUATE_ANSWER_SYSTEM_PROMPT)
        
        try:
            return json.loads(self._clean_json_response(response))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse evaluation response: {response}")
            return {"relevance": 0.5, "completeness": 0.5, "accuracy": 0.5}
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize response text."""
        if not response:
            return ""
        
        # Remove common artifacts
        response = response.strip()
        
        # Remove control characters
        response = self._clean_control_characters(response)
        
        return response
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response by extracting valid JSON."""
        response = response.strip()
        
        # Try to find JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            
            # Clean up common issues
            json_str = json_str.replace('\n', ' ')
            json_str = json_str.replace('\t', ' ')
            
            # Remove multiple spaces
            import re
            json_str = re.sub(r'\s+', ' ', json_str)
            
            return json_str
        
        return response
    
    def _clean_control_characters(self, text: str) -> str:
        """Remove control characters from text."""
        import re
        # Remove control characters except newlines and tabs
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return cleaned
    
    def _quick_health_check(self, instance: LMStudioInstance) -> bool:
        """Quick health check for LM Studio API."""
        try:
            # Simple test request
            response = instance.client.chat.completions.create(
                model=instance.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                temperature=0,
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            logger.debug(f"Health check failed for LM Studio at {instance.base_url}: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if LM Studio API is available."""
        if not self.concurrent_enabled:
            return self._quick_health_check(self.instances[0])
        
        return any(instance.is_healthy and self._quick_health_check(instance) for instance in self.instances)
    
    def list_models(self) -> List[str]:
        """List available models from LM Studio."""
        if not self.multi_instance_enabled:
            try:
                models = self.instances[0].client.models.list()
                return [model.id for model in models.data]
            except Exception as e:
                logger.error(f"Failed to list models from LM Studio: {e}")
                return []
        
        # 多实例模式：列出所有实例的可用模型
        all_models = set()
        for instance in self.instances:
            if instance.is_healthy:
                try:
                    models = instance.client.models.list()
                    all_models.update([model.id for model in models.data])
                except Exception as e:
                    logger.error(f"Failed to list models from {instance.base_url}: {e}")
        return list(all_models)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        base_info = {
            'provider': 'lmstudio',
            'multi_instance_enabled': self.multi_instance_enabled,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'is_available': self.is_available()
        }
        
        if not self.multi_instance_enabled:
            # 单实例模式
            instance = self.instances[0]
            base_info.update({
                'base_url': instance.base_url,
                'port': instance.port,
                'model_name': instance.model,
            })
        else:
            # 多实例模式
            base_info.update({
                'instances_count': len(self.instances),
                'healthy_instances_count': sum(1 for inst in self.instances if inst.is_healthy),
                'load_balancing_strategy': self.load_balancing_strategy.value,
                'stats': self.get_stats()
            })
        
        return base_info
    
    def get_stats(self) -> Dict[str, Any]:
        """获取实例统计信息（仅多实例模式）"""
        if not self.multi_instance_enabled:
            return {}
        
        total_requests = sum(inst.total_requests for inst in self.instances)
        total_errors = sum(inst.error_count for inst in self.instances)
        healthy_count = sum(1 for inst in self.instances if inst.is_healthy)
        
        return {"total_instances": len(self.instances),
            "healthy_instances": healthy_count,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
            "load_balancing_strategy": self.load_balancing_strategy.value if hasattr(self, 'load_balancing_strategy') else 'round_robin',
            "instances": [
                {
                    "base_url": inst.base_url,
                    "model": inst.model,
                    "port": inst.port,
                    "is_healthy": inst.is_healthy,
                    "active_requests": inst.active_requests,
                    "total_requests": inst.total_requests,
                    "error_count": inst.error_count,
                }
                for inst in self.instances
            ]
        }
    
    def _quick_health_check(self, instance=None):
        """Quick health check for LM Studio instances."""
        if instance is None:
            # Check all instances
            all_models = set()
            for inst in self.instances:
                try:
                    # Test connection by listing models
                    response = inst.client.models.list()
                    models = [model.id for model in response.data]
                    all_models.update(models)
                    inst.is_healthy = True
                    logger.debug(f"Health check passed for {inst.base_url}")
                except Exception as e:
                    inst.is_healthy = False
                    inst.error_count += 1
                    logger.error(f"Health check failed for {inst.base_url}: {e}")
            
            self.all_models = all_models
            return len([inst for inst in self.instances if inst.is_healthy]) > 0
        else:
            # Check specific instance
            try:
                response = instance.client.models.list()
                models = [model.id for model in response.data]
                instance.is_healthy = True
                logger.debug(f"Health check passed for {instance.base_url}")
                return True
            except Exception as e:
                instance.is_healthy = False
                instance.error_count += 1
                logger.error(f"Health check failed for {instance.base_url}: {e}")
                return False


# 为了向后兼容，保留原有的类名别名
MultiLMStudioClient = LMStudioClient
