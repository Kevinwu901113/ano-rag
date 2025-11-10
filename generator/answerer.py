import time

import requests
from loguru import logger

from generator.extractor import judge_and_compress


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
    compressed = _compress_evidence(question, evidences, endpoint, model)

    def _fmt(item, idx):
        canon = item.get("canonical") or item.get("evidence") or ""
        raw = item.get("evidence") or ""
        nid = item.get("note_id") or ""
        summary = item.get("summary")
        if summary:
            canon = summary
        return f"{idx + 1}) [{nid}] {canon} | {raw}"

    display_evs = compressed or evidences
    ev_text = "\n".join(_fmt(item, idx) for idx, item in enumerate(display_evs))
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


def _compress_evidence(question: str, evidences: list, endpoint: str, model: str) -> list:
    if not evidences:
        return []
    llm_cfg = {"endpoint": endpoint, "model": model, "timeout_s": 10}
    notes = []
    for ev in evidences:
        notes.append(
            {
                "note_id": ev.get("note_id"),
                "evidence": ev.get("canonical") or ev.get("evidence"),
                "meta": {"evidence_canonical": ev.get("canonical")},
            }
        )
    try:
        compressed = judge_and_compress(question, notes, llm_cfg)
        # merge with originals for formatting fallback
        merged = []
        for comp in compressed:
            match = next((ev for ev in evidences if ev.get("note_id") == comp.get("note_id")), {})
            merged.append({**match, **comp})
        return merged
    except Exception as exc:  # noqa: PERF203
        logger.warning("Evidence compression failed: {}", exc)
        return []
