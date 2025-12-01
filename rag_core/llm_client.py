import requests
import json
from typing import List, Dict, Optional, Any
from loguru import logger

class LLMChatClient:
    def __init__(self, endpoint: str, model: str, temperature: float = 0.0, **kwargs):
        """
        Initialize LLM Chat Client.
        endpoint: URL to the OpenAI-compatible API (e.g., http://localhost:1234/v1)
        model: Model identifier string
        """
        self.endpoint = endpoint.rstrip('/')
        if not self.endpoint.endswith("/v1"):
             # Some endpoints might be passed without /v1, but let's assume standard OpenAI format
             # If the user passed "http://localhost:1234", we append "/v1"
             # If they passed "http://localhost:1234/v1", we keep it.
             # But safer to just rely on what's passed or append if it looks like base URL.
             # Let's assume endpoint should be the full base URL for the client, usually ending in /v1
             pass
             
        self.model = model
        self.temperature = temperature
        self.api_key = kwargs.get("api_key", "lm-studio") # Default dummy key for local
        
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: Optional[float] = None, **kwargs) -> str:
        """
        Send chat completion request.
        messages: List of {"role": "...", "content": "..."}
        """
        url = f"{self.endpoint}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Merge extra kwargs
        payload.update(kwargs)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            # OpenAI format: choices[0].message.content
            content = data["choices"][0]["message"]["content"]
            return content
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            # Fallback for robustness? Or re-raise?
            # For baseline, maybe return empty or error message
            return f"[Error: {str(e)}]"
