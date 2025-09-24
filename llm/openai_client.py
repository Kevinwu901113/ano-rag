import json
from typing import Any, Dict, Iterator, List, Union, Optional

import openai
from loguru import logger

from config import config
from .prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT,
    EVALUATE_ANSWER_SYSTEM_PROMPT,
    EVALUATE_ANSWER_PROMPT,
)
from .streaming_early_stop import create_early_stop_stream


class OpenAIClient:
    """Wrapper around the OpenAI API with similar interface to OllamaClient."""

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or config.get("llm.openai.api_key")
        self.model = model or config.get("llm.openai.model", "gpt-3.5-turbo")
        self.base_url = base_url or config.get("llm.openai.base_url")
        self.temperature = config.get("llm.openai.temperature", 0.7)
        self.max_tokens = config.get("llm.openai.max_tokens", 4096)
        
        # OpenAI specific configuration
        self.timeout = config.get("llm.openai.timeout", 60)
        self.max_retries = config.get("llm.openai.max_retries", 3)
        
        # Initialize OpenAI client
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = openai.OpenAI(**client_kwargs)
        
        # Default options for all requests
        self.default_options = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
        options.update({
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        })
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **options,
            )

            if stream:
                text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        text += chunk.choices[0].delta.content
                return self._clean_response(text)
            else:
                return self._clean_response(response.choices[0].message.content or "")
            
        except Exception as e:
            # Log the specific error for debugging
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                logger.error(f"API Key error: Invalid or missing OpenAI API key - {e}")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                logger.error(f"Model error: Model '{self.model}' not found - {e}")
            elif "rate_limit" in error_msg.lower():
                logger.error(f"Rate limit error: OpenAI API rate limit exceeded - {e}")
            else:
                logger.error(f"Generation failed: {e}")
            return ""

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """生成流式响应"""
        # Merge default options with kwargs
        options = self.default_options.copy()
        options.update({
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        })
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **options,
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
            # Log the specific error for debugging
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                logger.error(f"API Key error: Invalid or missing OpenAI API key - {e}")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                logger.error(f"Model error: Model '{self.model}' not found - {e}")
            elif "rate_limit" in error_msg.lower():
                logger.error(f"Rate limit error: OpenAI API rate limit exceeded - {e}")
            else:
                logger.error(f"Generation failed: {e}")
            yield ""

    def chat(
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Chat completion with message history."""
        options = self.default_options.copy()
        options.update({
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
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
            logger.error(f"Chat completion failed: {e}")
            if stream:
                return iter([])
            return ""

    def generate_final_answer(self, context: str, query: str) -> str:
        """Generate final answer based on context and query."""
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
            return {"relevance": 0.0, "accuracy": 0.0, "completeness": 0.0, "clarity": 0.0}

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, system_prompt, stream=stream, **kwargs)
            results.append(result)
        return results

    def _quick_health_check(self) -> bool:
        """Quick health check for OpenAI API."""
        try:
            # Simple test request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                temperature=0,
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

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

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.api_key:
            logger.warning("OpenAI API key not configured")
            return False
            
        return self._quick_health_check()

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []