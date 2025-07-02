import json
from typing import Any, Dict, Iterator, List, Union

import ollama
from loguru import logger

from config import config
from .prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT,
    EVALUATE_ANSWER_SYSTEM_PROMPT,
    EVALUATE_ANSWER_PROMPT,
)


class OllamaClient:
    """Wrapper around the Ollama Python SDK."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or config.get("llm.ollama.base_url", "http://localhost:11434")
        self.model = model or config.get("llm.ollama.model", "gemma3:4b-it-fp16")
        self.temperature = config.get("llm.ollama.temperature", 0.7)
        self.max_tokens = config.get("llm.ollama.max_tokens", 4096)
        self.client = ollama.Client(host=self.base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Generate text from a prompt."""
        options = {
            "temperature": kwargs.get("temperature", self.temperature),
            "num_predict": kwargs.get("max_tokens", self.max_tokens),
        }
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                stream=stream,
                options=options,
            )
            if stream:
                return (chunk.response for chunk in response)
            return response.response
        except Exception as e:  # pragma: no cover - runtime connection error
            logger.error(f"Generation failed: {e}")
            return "" if not stream else iter(())

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Chat with the model using a list of messages."""
        options = {
            "temperature": kwargs.get("temperature", self.temperature),
            "num_predict": kwargs.get("max_tokens", self.max_tokens),
        }
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=stream,
                options=options,
            )
            if stream:
                return (chunk.message.content or "" for chunk in response)
            return response.message.content or ""
        except Exception as e:  # pragma: no cover - runtime connection error
            logger.error(f"Chat failed: {e}")
            return "" if not stream else iter(())

    def generate_final_answer(self, context: str, query: str) -> str:
        system_prompt = FINAL_ANSWER_SYSTEM_PROMPT
        prompt = FINAL_ANSWER_PROMPT.format(context=context, query=query)
        result = self.generate(prompt, system_prompt)
        return result if isinstance(result, str) else "".join(result)

    def evaluate_answer(self, query: str, context: str, answer: str) -> Dict[str, float]:
        system_prompt = EVALUATE_ANSWER_SYSTEM_PROMPT
        prompt = EVALUATE_ANSWER_PROMPT.format(query=query, context=context, answer=answer)
        try:
            result = self.generate(prompt, system_prompt)
            text = result if isinstance(result, str) else "".join(result)
            return json.loads(text)
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
    ) -> List[Union[str, Iterator[str]]]:
        results: List[Union[str, Iterator[str]]] = []
        for prompt in prompts:
            try:
                results.append(self.generate(prompt, system_prompt, stream=stream, **kwargs))
            except Exception as e:  # pragma: no cover
                logger.error(f"Batch generation failed for prompt: {e}")
                results.append("" if not stream else iter(()))
        return results

    def is_available(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            response = self.client.list()
            return [m.model for m in response.models]
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to list models: {e}")
            return []
