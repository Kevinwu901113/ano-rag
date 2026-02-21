import openai
import os

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

try:
    resp = client.embeddings.create(
        model="bge-m3",
        input="hello world"
    )
    if not resp.data:
        raise Exception("No embedding data received")
    print("Embedding service is reachable!")
    print(resp.data[0].embedding[:5])
except Exception as e:
    print(f"Embedding service failed: {e}")
