import json
from typing import Any, Dict, Iterator, List, Union

import ollama
import requests
from loguru import logger

from config import config
from .prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT,
    EVALUATE_ANSWER_SYSTEM_PROMPT,
    EVALUATE_ANSWER_PROMPT,
)


class OllamaClient:
    """Wrapper around the Ollama Python SDK with LightRAG-inspired improvements."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or config.get("llm.ollama.base_url", "http://localhost:11434")
        self.model = model or config.get("llm.ollama.model", "gemma3:4b-it-fp16")
        self.temperature = config.get("llm.ollama.temperature", 0.7)
        self.max_tokens = config.get("llm.ollama.max_tokens", 4096)
        
        # LightRAG-inspired configuration
        self.num_ctx = config.get("llm.ollama.num_ctx", 32768)  # Context window size
        self.max_async = config.get("llm.ollama.max_async", 4)  # Max concurrent requests
        self.timeout = config.get("llm.ollama.timeout", 60)  # Request timeout
        
        # Initialize client with host
        self.client = ollama.Client(host=self.base_url)
        
        # Default options for all requests
        # Reduce num_ctx to avoid 503 errors with smaller models
        safe_num_ctx = min(self.num_ctx, 8192)  # Use smaller context window
        safe_max_tokens = min(self.max_tokens, 2048)  # Limit output tokens
        
        self.default_options = {
            "num_ctx": safe_num_ctx,
            "temperature": self.temperature,
            "num_predict": safe_max_tokens,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt with improved error handling."""
        # Merge default options with kwargs
        options = self.default_options.copy()
        # Use safe limits to avoid 503 errors
        safe_max_tokens = min(kwargs.get("max_tokens", self.max_tokens), 2048)
        safe_num_ctx = min(kwargs.get("num_ctx", self.num_ctx), 8192)
        
        options.update({
            "temperature": kwargs.get("temperature", self.temperature),
            "num_predict": safe_max_tokens,
            "num_ctx": safe_num_ctx,
        })
        
        # Get timeout from kwargs or use default
        timeout = kwargs.get("timeout", self.timeout)
        
        try:
            # Prepare the request
            request_params = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": options,
            }
            
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Make the request directly - let ollama handle connection errors
            response = self.client.generate(**request_params)

            text: str = ""
            if stream:
                for chunk in response:
                    chunk_text = getattr(chunk, "response", None)
                    if chunk_text is not None:
                        text += chunk_text
                    elif hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                        text += chunk.message.content
            else:
                if hasattr(response, "response"):
                    text = response.response
                elif hasattr(response, "message") and hasattr(response.message, "content"):
                    text = response.message.content
                else:
                    text = str(response)

            return self._clean_response(text)
            
        except Exception as e:
            # Log the specific error for debugging
            error_msg = str(e)
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                logger.error(f"Connection error: Ollama service is not responding - {e}")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                logger.error(f"Model error: Model '{self.model}' not found - {e}")
            else:
                logger.error(f"Generation failed: {e}")
            return ""

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
        # Use safe limits to avoid 503 errors
        safe_max_tokens = min(kwargs.get("max_tokens", self.max_tokens), 2048)
        safe_num_ctx = min(kwargs.get("num_ctx", self.num_ctx), 8192)
        
        options.update({
            "temperature": kwargs.get("temperature", self.temperature),
            "num_predict": safe_max_tokens,
            "num_ctx": safe_num_ctx,
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

    def generate_final_answer(self, context: str, query: str) -> str:
        system_prompt = FINAL_ANSWER_SYSTEM_PROMPT
        prompt = FINAL_ANSWER_PROMPT.format(context=context, query=query)
        return self.generate(prompt, system_prompt)

    def evaluate_answer(self, query: str, context: str, answer: str) -> Dict[str, float]:
        system_prompt = EVALUATE_ANSWER_SYSTEM_PROMPT
        prompt = EVALUATE_ANSWER_PROMPT.format(query=query, context=context, answer=answer)
        try:
            text = self.generate(prompt, system_prompt)
            # 清理响应，移除可能的markdown代码块标记
            cleaned_text = self._clean_json_response(text)
            return json.loads(cleaned_text)
        except Exception as e:  # pragma: no cover - runtime parsing or connection error
            logger.error(f"Answer evaluation failed: {e}")
            return {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        results: List[str] = []
        for prompt in prompts:
            try:
                results.append(self.generate(prompt, system_prompt, stream=stream, **kwargs))
            except Exception as e:  # pragma: no cover
                logger.error(f"Batch generation failed for prompt: {e}")
                results.append("")
        return results

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
