import json
import requests

def debug_llm_response(endpoint="http://127.0.0.1:8000/v1", model="Qwen/Qwen2.5-7B-Instruct"):
    url = f"{endpoint}/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    context = """John Dawson Mayne

Mayne served as the Professor of law, logic and moral philosophy at the Presidency College, Madras from 1857 throughout the 1860s. He also served as Assistant Legal Secretary to the Madras government from 1860 to 1872 and as a Clerk of the Crown during the 1860s. He served as Advocate-General of Madras from 1862 to 1872."""

    question = "What is John Mayne's occupation?"
    
    prompt_template = """Answer the question based on the context below. Keep the answer short and concise. Do not output reasoning.

Context:
{context}

Question: {question}
Answer:"""

    prompt = prompt_template.format(context=context, question=question)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100
    }
    
    print(f"Sending prompt:\n{prompt}\n")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']
        print(f"Raw LLM Response: {content!r}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_llm_response()
