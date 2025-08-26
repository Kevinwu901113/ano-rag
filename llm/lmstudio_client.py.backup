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


class LMStudioClient:
    """LM Studio客户端，提供与OpenAI兼容的API接口。
    
    LM Studio使用本地1234端口提供OpenAI兼容的API服务。
    支持chat completions、embeddings等功能。
    """

    def __init__(self, base_url: str = None, model: str = None, port: int = None):
        # 设置默认配置
        self.port = port or config.get("llm.lmstudio.port", 1234)
        self.base_url = base_url or config.get("llm.lmstudio.base_url", f"http://localhost:{self.port}/v1")
        self.model = model or config.get("llm.lmstudio.model", "default-model")
        self.temperature = config.get("llm.lmstudio.temperature", 0.7)
        self.max_tokens = config.get("llm.lmstudio.max_tokens", 4096)
        
        # LM Studio specific configuration
        self.timeout = config.get("llm.lmstudio.timeout", 60)
        self.max_retries = config.get("llm.lmstudio.max_retries", 3)
        
        # Initialize OpenAI client pointing to LM Studio
        client_kwargs = {
            "api_key": "lm-studio",  # LM Studio doesn't require a real API key
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
            
        self.client = openai.OpenAI(**client_kwargs)
        
        # Default options for all requests
        self.default_options = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        logger.info(f"Initialized LM Studio client: {self.base_url}, model: {self.model}")

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
            if "connection" in error_msg.lower():
                logger.error(f"Connection error: Failed to connect to LM Studio at {self.base_url} - {e}")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                logger.error(f"Model error: Model '{self.model}' not found in LM Studio - {e}")
            elif "timeout" in error_msg.lower():
                logger.error(f"Timeout error: LM Studio request timed out - {e}")
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
        """Chat with the model using a list of messages."""
        # Merge default options with kwargs
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
            logger.error(f"Chat failed: {e}")
            if stream:
                return iter([])
            return ""

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

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, system_prompt, stream=stream, **kwargs) for prompt in prompts]

    def _quick_health_check(self) -> bool:
        """Quick health check for LM Studio API."""
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
            logger.debug(f"Health check failed for LM Studio at {self.base_url}: {e}")
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
        """Check if LM Studio API is available."""
        return self._quick_health_check()

    def list_models(self) -> List[str]:
        """List available models from LM Studio."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list models from LM Studio: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'provider': 'lmstudio',
            'base_url': self.base_url,
            'port': self.port,
            'model_name': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'is_available': self.is_available()
        }