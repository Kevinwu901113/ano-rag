import requests


ANS_PROMPT = """You are a factual answerer. Use ONLY the provided evidence sentences to answer the question. If the evidence is insufficient, say "Insufficient evidence".
Question: {q}
Evidence:
{ev}
Instruction: Provide a concise answer. Do not add facts not present in the evidence.
Answer:
"""


def call_lmstudio(
    endpoint: str,
    model: str,
    question: str,
    evidences: list,
    temperature: float = 0.2,
    max_tokens: int = 64,
) -> str:
    ev_text = "\n".join(f"{idx + 1}) {item['evidence']}" for idx, item in enumerate(evidences))
    prompt = ANS_PROMPT.format(q=question, ev=ev_text)

    response = requests.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
