import time

import requests
from loguru import logger


ANS_PROMPT = """You are a factual answerer. Use ONLY the provided evidence sentences to answer the question. If the evidence is insufficient, say "Insufficient evidence".
Question: {q}
Evidence (canonical | original):
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
    retries: int = 2,
) -> str:
    def _fmt(item, idx):
        canon = item.get("canonical") or item.get("evidence") or ""
        raw = item.get("evidence") or ""
        nid = item.get("note_id") or ""
        return f"{idx + 1}) [{nid}] {canon} | {raw}"
    ev_text = "\n".join(_fmt(item, idx) for idx, item in enumerate(evidences))
    ev_text = ev_text.replace("{", "{{").replace("}", "}}")
    prompt = ANS_PROMPT.format(q=question.replace("{", "{{").replace("}", "}}"), ev=ev_text)

    for attempt in range(retries + 1):
        try:
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
        except requests.RequestException as exc:  # noqa: PERF203
            if attempt == retries:
                raise
            wait = 2 ** attempt
            logger.warning("Answerer call failed (attempt={}): {}", attempt + 1, exc)
            time.sleep(wait)
