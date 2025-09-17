import json
from typing import Any, Dict, Iterator, List, Union, Optional
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
import requests
from loguru import logger

from config import config
from .prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT,
)


class OllamaClient:
    """Wrapper around the Ollama Python SDK with LightRAG-inspired improvements."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or config.get("llm.ollama.base_url", "http://localhost:11434")
        self.model = model or config.get("llm.ollama.model", "qwen2.5:14b")  # 更实用的默认模型
        self.temperature = config.get("llm.ollama.temperature", 0.3)  # 降低温度提高一致性
        self.max_tokens = config.get("llm.ollama.max_tokens", 4096)  # 适中的token限制
        
        # LightRAG-inspired configuration with optimized defaults
        self.num_ctx = config.get("llm.ollama.num_ctx", 16384)  # 减少上下文窗口以提高性能
        self.max_async = config.get("llm.ollama.max_async", 8)  # 减少并发数避免资源竞争
        self.timeout = config.get("llm.ollama.timeout", 90)  # 增加超时时间适应复杂任务
        
        # Initialize client with host
        self.client = ollama.Client(host=self.base_url)
        
        # Default options for all requests with optimized settings
        self.default_options = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "top_k": config.get("llm.ollama.top_k", 40),  # 添加top_k控制
            "top_p": config.get("llm.ollama.top_p", 0.9),  # 添加top_p控制
            "repeat_penalty": config.get("llm.ollama.repeat_penalty", 1.1),  # 避免重复
        }
        
        # 并发处理配置 - 更保守的默认设置
        self.concurrent_enabled = config.get("llm.ollama.concurrent.enabled", False)
        if self.concurrent_enabled:
            self.max_concurrent_requests = config.get(
                "llm.ollama.concurrent.max_workers", min(self.max_async, 4)
            )
            logger.info(
                f"Ollama concurrent processing enabled with {self.max_concurrent_requests} max concurrent requests"
            )
        else:
            self.max_concurrent_requests = 1

        self.executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()
        
        logger.info(f"Initialized Ollama client: {'concurrent' if self.concurrent_enabled else 'single-threaded'} mode")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        timeout: int | None = None,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt with improved error handling."""
        # Merge default options with kwargs
        options = self.default_options.copy()
        options.update({
            "temperature": kwargs.get("temperature", self.temperature),
            "num_predict": kwargs.get("max_tokens", self.max_tokens),
            "num_ctx": kwargs.get("num_ctx", self.num_ctx),
        })
        
        # Use provided timeout or default
        request_timeout = timeout or self.timeout
        
        # Prepare the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # 实现重试机制
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt, 10)  # 指数退避，最大10秒
                    logger.info(f"Retrying Ollama request (attempt {attempt + 1}/{max_retries + 1}) after {wait_time}s")
                    time.sleep(wait_time)
                
                # 直接调用，不使用线程超时机制来避免并发问题
                try:
                    result = self.client.generate(
                        model=self.model,
                        prompt=full_prompt,
                        stream=stream,
                        options=options,
                    )
                except Exception as e:
                    logger.warning(f"Ollama request failed (attempt {attempt + 1}): {e}")
                    last_exception = e
                    if attempt < max_retries and isinstance(e, (TimeoutError, ConnectionError)):
                        continue
                    raise e
                
                
                # Get the result
                text: str = ""
                if stream:
                    for chunk in result:
                        if hasattr(chunk, "response"):
                            text += chunk.response
                else:
                    if hasattr(result, "response"):
                        text = result.response
                    else:
                        text = str(result)

                if not text or text.strip() == "":
                    last_exception = ValueError("Ollama returned empty response")
                    if attempt < max_retries:
                        continue
                    raise last_exception

                return self._clean_response(text)
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries and isinstance(e, (TimeoutError, ConnectionError, ValueError)):
                    continue
                raise e
        
        # 如果所有重试都失败了，抛出最后一个异常
        if last_exception:
            raise last_exception
        
        # 如果没有异常但也没有返回结果，抛出错误
        raise RuntimeError("Unexpected error: no result and no exception")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Chat with the model using a list of messages with improved error handling."""
        # Merge default options with kwargs
        options = self.default_options.copy()
        options.update({
            "temperature": kwargs.get("temperature", self.temperature),
            "num_predict": kwargs.get("max_tokens", self.max_tokens),
            "num_ctx": kwargs.get("num_ctx", self.num_ctx),
        })
        
        try:
            # Make the request directly - let ollama handle connection errors
            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=stream,
                options=options,
            )
            
            if stream:
                return (chunk.message.content for chunk in response)
            
            # Clean and return the response
            result = response.message.content if hasattr(response, 'message') else str(response)
            return self._clean_response(result)
            
        except Exception as e:
            # Log the specific error for debugging
            error_msg = str(e)
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                logger.error(f"Connection error in chat: Ollama service is not responding - {e}")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                logger.error(f"Model error in chat: Model '{self.model}' not found - {e}")
            else:
                logger.error(f"Chat failed: {e}")
            return ""

    def generate_final_answer(self, prompt: str, **kwargs) -> str:
        """Generate final answer from a combined prompt"""
        return self.generate(prompt, FINAL_ANSWER_SYSTEM_PROMPT, **kwargs)

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """批量生成响应，支持并发处理"""
        if not self.concurrent_enabled or len(prompts) <= 1:
            # 单线程处理
            results: List[str] = []
            for prompt in prompts:
                try:
                    results.append(self.generate(prompt, system_prompt, stream=stream, **kwargs))
                except Exception as e:  # pragma: no cover
                    logger.error(f"Batch generation failed for prompt: {e}")
                    results.append("")
            return results
        
        # 使用线程池并发处理
        return self.generate_concurrent(prompts, system_prompt, stream=stream, **kwargs)
    
    def generate_concurrent(self, prompts: List[str], system_prompt: str | None = None, **kwargs) -> List[str]:
        """并发生成多个响应"""
        if not self.concurrent_enabled or len(prompts) == 1:
            # 单线程处理
            return [self.generate(prompt, system_prompt, **kwargs) for prompt in prompts]
        
        executor = self._ensure_executor()
        futures = []
        for prompt in prompts:
            future = executor.submit(self.generate, prompt, system_prompt, **kwargs)
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

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self.executor is None:
            with self._executor_lock:
                if self.executor is None:
                    self.executor = ThreadPoolExecutor(
                        max_workers=self.max_concurrent_requests
                    )
        return self.executor

    def close(self):
        if not hasattr(self, "_executor_lock"):
            self._executor_lock = threading.Lock()
        with self._executor_lock:
            if self.executor is not None:
                self.executor.shutdown(wait=False)
                self.executor = None

    def cleanup(self):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _quick_health_check(self) -> bool:
        """Quick health check with minimal timeout."""
        try:
            # First check if the API endpoint is reachable
            response = requests.get(f"{self.base_url}/api/version", timeout=3)
            if response.status_code != 200:
                return False
            
            # Then check if we can list models (this tests the ollama client)
            models_response = self.client.list()
            return hasattr(models_response, 'models')
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize the response text."""
        if not response:
            return ""
        
        # Strip whitespace
        result = response.strip()
        
        # Remove BOM character if present
        if result.startswith('\ufeff'):
            result = result[1:]
        
        # Remove any trailing null characters
        result = result.rstrip('\x00')
        
        return result
    
    def _clean_json_response(self, response: str) -> str:
        """清理LLM响应，移除markdown代码块标记和其他格式"""
        if not response:
            return "{}"
        
        # 清理控制字符
        response = self._clean_control_characters(response)
        
        # 移除markdown代码块标记
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        
        if response.endswith('```'):
            response = response[:-3]
        
        # 移除可能的前后空白和换行
        response = response.strip()
        
        # 尝试提取JSON对象
        import re
        # 查找第一个完整的JSON对象
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        if json_match:
            response = json_match.group(0)
        
        # 如果响应为空或不是JSON格式，返回空对象
        if not response or not (response.startswith('{') or response.startswith('[')):
            return "{}"
        
        return response
    
    def _clean_control_characters(self, text: str) -> str:
        """清理字符串中的无效控制字符"""
        import re
        
        # 移除或替换无效的控制字符，但保留有效的空白字符（空格、制表符、换行符）
        # 保留 \t (\x09), \n (\x0A), \r (\x0D) 和普通空格 (\x20)
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # 替换一些常见的问题字符
        cleaned = cleaned.replace('\u0000', '')  # NULL字符
        cleaned = cleaned.replace('\u0001', '')  # SOH字符
        cleaned = cleaned.replace('\u0002', '')  # STX字符
        
        return cleaned
    
    def is_available(self) -> bool:
        """Comprehensive availability check."""
        try:
            # Quick health check first
            if not self._quick_health_check():
                return False
            
            # Test model availability
            models = self.client.list()
            available_models = [m.model for m in models.models] if hasattr(models, 'models') else []
            
            # Check if our model is available
            if self.model not in available_models:
                logger.warning(f"Model {self.model} not found in available models: {available_models}")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Ollama service check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        try:
            response = self.client.list()
            return [m.model for m in response.models]
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to list models: {e}")
            return []
