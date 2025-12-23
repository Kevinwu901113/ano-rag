import json
import time

from utils.llm_client import LLMChatClient

def test_llm(endpoint="http://127.0.0.1:8000/v1", model="qwen3-30b-a3b"):
    client = LLMChatClient(endpoint=endpoint, model=model, llm_profile="generate", retries=0, timeout=10)
    
    try:
        print(f"Sending request to {endpoint}/chat/completions...")
        response = client.chat(
            [{"role": "user", "content": "What is the capital of France?"}],
            temperature=0.0,
            max_tokens=100,
        )
        data = response.raw
        print("Response received:")
        print(json.dumps(data, indent=2))
        content = response.content
        print(f"Content: {content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm()
