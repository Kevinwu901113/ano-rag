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
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT,
)


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
        
        # 测试连接 - 使用更宽松的健康检查
        self.is_healthy = self._quick_health_check_with_retry()
        if self.is_healthy:
            logger.info(f"LM Studio connection test successful for {self.base_url} (model: {self.model})")
        else:
            logger.warning(f"LM Studio connection test failed for {self.base_url}, but instance will remain available for retry")
    
    def _quick_health_check_with_retry(self, max_retries: int = 3, backoff_ms: int = 200) -> bool:
        """带重试的快速健康检查"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=1,
                    temperature=0,
                )
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Health check attempt {attempt + 1} failed for {self.base_url}: {e}, retrying in {backoff_ms}ms")
                    time.sleep(backoff_ms / 1000.0)  # 转换为秒
                    backoff_ms *= 2  # 指数退避
                else:
                    logger.error(f"Health check failed for {self.base_url} after {max_retries} attempts: {e}")
        return False


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
        
        # 默认选项
        self.default_options = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # 初始化单一LM Studio连接
        self._init_connection(base_url, model, port)
        
        # 并发处理配置
        self.concurrent_enabled = config.get("llm.lmstudio.multiple_instances.enabled", False)
        self.multi_instance_enabled = self.concurrent_enabled  # 添加兼容性属性
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
    
    def generate_concurrent(self, prompts: List[str], **kwargs) -> List[str]:
        """并发生成多个响应"""
        if not self.concurrent_enabled or len(prompts) == 1:
            # 单线程处理
            return [self.generate(prompt, **kwargs) for prompt in prompts]
        
        # 使用线程池并发处理
        futures = []
        with self.executor as executor:
            for prompt in prompts:
                future = executor.submit(self.generate, prompt, **kwargs)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Concurrent generation failed: {e}")
                    results.append("")
        
        return results
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成单个响应 - JSON模式作为软约束，失败时不判实例失效"""
        try:
            # 提取system_prompt和try_json_mode参数，避免传递给OpenAI API
            options = {**self.default_options}
            for key, value in kwargs.items():
                if key not in ['system_prompt', 'try_json_mode']:
                    options[key] = value
            
            # JSON模式作为软约束 - 无论是否开启都走宽松JSON提取
            try_json_mode = kwargs.get('try_json_mode', True)
            json_mode_enabled = False
            
            if try_json_mode and 'response_format' not in options:
                # 降低temperature以提高JSON输出稳定性
                options['temperature'] = min(options.get('temperature', 0.7), 0.1)
                
                # 尝试使用JSON Schema格式，但不强制
                try:
                    options['response_format'] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "strict": False,  # 放宽严格性要求
                            "schema": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": True
                            }
                        }
                    }
                    json_mode_enabled = True
                    logger.debug("Enabled relaxed JSON schema mode for LM Studio request")
                except Exception as e:
                    logger.debug(f"JSON schema mode not supported, falling back to text mode: {e}")
            
            # 准备消息，如果启用JSON模式，添加更宽松的提示
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # 无论是否启用JSON模式，都添加宽松的JSON提示
            if try_json_mode:
                enhanced_prompt = f"{prompt}\n\n请只输出JSON数组或对象。如果无法严格匹配格式，也不要输出额外文字，系统会自动纠错。"
                messages.append({"role": "user", "content": enhanced_prompt})
            else:
                messages.append({"role": "user", "content": prompt})
            
            # 调用OpenAI客户端生成响应
            response = self.instance.client.chat.completions.create(
                model=self.instance.model,
                messages=messages,
                **options
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # 无论是否开启JSON模式，都走宽松JSON提取
            if try_json_mode:
                from utils.json_utils import extract_json_from_response
                try:
                    cleaned_json = extract_json_from_response(raw_response)
                    if cleaned_json:
                        return cleaned_json
                    else:
                        logger.warning("No valid JSON found in LM Studio response, returning raw response")
                        return raw_response
                except Exception as e:
                    logger.warning(f"JSON extraction failed: {e}, returning raw response")
                    return raw_response
            
            return raw_response
            
        except Exception as e:
            # 失败时不把实例判为"失效"，仅记录错误
            logger.error(f"LM Studio generation failed: {e}")
            return ""
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """生成流式响应"""
        try:
            # 合并默认选项和传入的参数
            options = {**self.default_options, **kwargs}
            
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
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
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
            
            # 尝试启用JSON模式（如果模型支持且不是流式）
            try_json_mode = kwargs.get('try_json_mode', True)
            if try_json_mode and not stream and 'response_format' not in options:
                # 降低temperature以提高JSON输出稳定性
                options['temperature'] = min(options.get('temperature', 0.7), 0.1)
                
                # 使用LM Studio支持的JSON Schema格式
                try:
                    options['response_format'] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": True
                            }
                        }
                    }
                    logger.debug("Enabled JSON schema mode for LM Studio chat request")
                except Exception as e:
                    logger.debug(f"JSON schema mode not supported, falling back to text mode: {e}")
            
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
            return self.generate_concurrent(prompts, system_prompt, **kwargs)
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
    
    def generate_final_answer(self, context: str, query: str) -> str:
        """Generate final answer using context and query."""
        prompt = FINAL_ANSWER_PROMPT.format(context=context, query=query)
        return self.generate(prompt, FINAL_ANSWER_SYSTEM_PROMPT)
    
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
            # 对于单实例，使用更宽松的可用性判断
            instance = self.instances[0]
            current_time = time.time()
            
            # 如果最近5秒内有成功的健康检查，认为可用
            if instance.is_healthy and (current_time - instance.last_health_check) < 5:
                return True
            
            # 否则进行新的健康检查
            is_healthy = self._quick_health_check(instance)
            instance.last_health_check = current_time
            return is_healthy
        
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
        """Quick health check for LM Studio instances with retry and backoff."""
        if instance is None:
            # Check all instances
            all_models = set()
            for inst in self.instances:
                is_healthy = self._quick_health_check_single_instance(inst)
                if is_healthy:
                    try:
                        response = inst.client.models.list()
                        models = [model.id for model in response.data]
                        all_models.update(models)
                    except Exception as e:
                        logger.debug(f"Failed to list models for {inst.base_url}: {e}")
            
            self.all_models = all_models
            return len([inst for inst in self.instances if inst.is_healthy]) > 0
        else:
            # Check specific instance
            return self._quick_health_check_single_instance(instance)
    
    def _quick_health_check_single_instance(self, instance: LMStudioInstance) -> bool:
        """Quick health check for a single instance with retry mechanism."""
        max_retries = 3
        backoff_ms = 200
        
        for attempt in range(max_retries):
            try:
                response = instance.client.models.list()
                models = [model.id for model in response.data]
                instance.is_healthy = True
                instance.last_health_check = time.time()
                logger.debug(f"Health check passed for {instance.base_url}")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Health check attempt {attempt + 1} failed for {instance.base_url}: {e}, retrying in {backoff_ms}ms")
                    time.sleep(backoff_ms / 1000.0)
                    backoff_ms *= 2  # 指数退避
                else:
                    # 失败时不将实例移出池，仅降低优先级
                    instance.is_healthy = False
                    instance.error_count += 1
                    logger.warning(f"Health check failed for {instance.base_url} after {max_retries} attempts: {e}. Instance marked as low priority but kept in pool.")
        
        return False


# 为了向后兼容，保留原有的类名别名
MultiLMStudioClient = LMStudioClient