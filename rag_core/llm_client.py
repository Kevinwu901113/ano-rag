import time
import requests
from typing import List, Dict, Optional, Any
from loguru import logger

class LLMChatClient:
    """
    A unified client for chatting with LLM via OpenAI-compatible API.
    Support for LM Studio, vLLM, etc.
    """
    def __init__(
        self, 
        endpoint: str, 
        model: str, 
        temperature: float = 0.0, 
        api_key: str = "sk-no-key-required",
        stop: Optional[List[str]] = None,
        retries: int = 3
    ):
        self._mock = str(endpoint).strip().lower() == "mock"
        self.endpoint = endpoint.rstrip("/")
        # Auto-correct endpoint format if needed
        # Most OpenAI clients expect base_url to be just the host, but requests needs full path
        # If user passes http://localhost:1234/v1, we keep it.
        # If user passes http://localhost:1234, we might append /v1 or not depending on usage.
        # Here we assume the user provides the base API URL (e.g. .../v1).
        
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.stop = stop
        self.retries = retries

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: Optional[float] = None, **kwargs) -> str:
        """
        Send chat completion request with retries.
        messages: List of {"role": "...", "content": "..."}
        """
        if self._mock:
            last_msg = messages[-1]["content"] if messages else ""
            if "json" in str(last_msg).lower():
                return "{}"
            return "Mock Answer"
        url = f"{self.endpoint}/chat/completions"
        logger.info(f"[RAPTOR-LLM] POST {url} model={self.model}")
        # Ensure we don't double-slash if endpoint already had trailing slash (handled by rstrip above)
        # But if endpoint is http://localhost:8000 and we want http://localhost:8000/v1/chat/completions
        # Check if /v1 is missing?
        # Standard practice: assume endpoint is base URL like http://localhost:8000/v1
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        if self.stop:
            payload["stop"] = self.stop
            
        # Merge extra kwargs
        payload.update(kwargs)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        for attempt in range(self.retries + 1):
            if attempt == 0:
                logger.info(f"[RAPTOR-LLM] payload_preview={str(payload)[:300]}")
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                if response.status_code == 400:
                    logger.error(f"400 Bad Request: {response.text}")
                    # 400 usually means bad payload (too long context?), retrying won't help usually.
                    # But for now let's just log it and break to avoid wasting time.
                    return f"[Error: 400 Client Error: {response.text}]"
                response.raise_for_status()
                data = response.json()
                
                # Parse response
                if "choices" not in data or not data["choices"]:
                     logger.error(f"Invalid LLM response: {data}")
                     # If response is malformed, retry?
                     if attempt < self.retries:
                         continue
                     return "[Error: Invalid response format]"
                     
                content = data["choices"][0]["message"]["content"]
                return content
                
            except requests.RequestException as e:
                if attempt < self.retries:
                    wait_time = 2 ** attempt # Exponential backoff: 1s, 2s, 4s...
                    logger.warning(f"LLM request failed (attempt {attempt+1}/{self.retries+1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM request failed after {self.retries+1} attempts: {e}")
                    # Fallback or re-raise? 
                    # For baselines, returning an error string is safer than crashing the whole run.
                    return f"[Error: {str(e)}]"
            except Exception as e:
                 logger.error(f"Unexpected error during LLM call: {e}")
                 return f"[Error: {str(e)}]"
                 
        return "[Error: Max retries exceeded]"
