import requests
import json
import time

def test_llm(endpoint="http://127.0.0.1:8001/v1", model="Qwen/Qwen2.5-7B-Instruct"):
    url = f"{endpoint}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.0,
        "max_tokens": 100
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("Response received:")
        print(json.dumps(data, indent=2))
        content = data['choices'][0]['message']['content']
        print(f"Content: {content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm()
