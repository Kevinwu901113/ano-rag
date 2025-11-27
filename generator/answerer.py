import time

from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from generator.extractor import judge_and_compress


ANS_PROMPT = """You are a factual answerer. Use the provided evidence sentences to answer the question.
If the evidence is insufficient, respond EXACTLY with "Insufficient evidence".
Prioritize high-confidence evidence, but do not ignore weak evidence if it provides a reasonable answer and does not conflict with strong evidence.
{label_instruction}
Question: {q}
[STRUCTURED EVIDENCE]
{strong_block}
[WEAK EVIDENCE – lower confidence]
{weak_block}
Rules:
- Prefer answers supported by strong evidence.
- You can use weak evidence if it directly answers the question and is not contradicted by strong evidence.
- If weak evidence conflicts with strong evidence, ignore the weak evidence.
- If no sufficient evidence exists, answer "Insufficient evidence".
Respond with exactly one label and nothing else.
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
    allowed_labels: Optional[List[str]] = None,
    attribute_name: Optional[str] = None,
) -> str:
    compressed = _compress_evidence(question, evidences, endpoint, model)

    display_evs = compressed or evidences
    strong_items = [item for item in display_evs if not item.get("weak")]
    weak_items = [item for item in display_evs if item.get("weak")]
    strong_block = "\n".join(_fmt_strong(item, idx) for idx, item in enumerate(strong_items))
    weak_block = "\n".join(_fmt_weak(item) for item in weak_items)
    strong_block = (strong_block or "None").replace("{", "{{").replace("}", "}}")
    weak_block = (weak_block or "None").replace("{", "{{").replace("}", "}}")
    sanitized_labels = _prepare_allowed_labels(allowed_labels)
    prompt = ANS_PROMPT.format(
        q=question.replace("{", "{{").replace("}", "}}"),
        strong_block=strong_block,
        weak_block=weak_block,
        label_instruction=_label_instruction(sanitized_labels, attribute_name),
    )

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
            raw_answer = data["choices"][0]["message"]["content"].strip()
            return _enforce_single_label(raw_answer, sanitized_labels)
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


def _label_instruction(labels: List[str], attribute_name: Optional[str]) -> str:
    if labels:
        joined = ", ".join(labels)
        return f"Allowed labels ({attribute_name or 'answer'}): {joined}."
    return "If you can answer, output the canonical label only."


def _prepare_allowed_labels(labels: Optional[List[str]]) -> List[str]:
    prepared: List[str] = []
    if not labels:
        return prepared
    seen: set[str] = set()
    for label in labels:
        text = str(label).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        prepared.append(text)
    return prepared


def _enforce_single_label(text: str, allowed: List[str]) -> str:
    candidate = _strip_reasoning(text)
    candidate = candidate.splitlines()[0].strip()
    if not candidate:
        return ""
    if allowed:
        lowered = candidate.lower()
        for label in allowed:
            if lowered == label.lower():
                return label
        for label in allowed:
            if label.lower() in lowered:
                return label
        return allowed[0]
    return candidate


def _strip_reasoning(text: str) -> str:
    output = text or ""
    while True:
        start = output.find("<think>")
        if start == -1:
            break
        end = output.find("</think>", start + len("<think>"))
        if end == -1:
            output = output[:start] + output[start + len("<think>") :]
            break
        output = output[:start] + output[end + len("</think>") :]
    return output.strip()


def _fmt_strong(item: Dict[str, Any], idx: int) -> str:
    canon = item.get("canonical") or item.get("evidence") or ""
    raw = item.get("evidence") or ""
    nid = item.get("note_id") or ""
    summary = item.get("summary")
    if summary:
        canon = summary
    return f"{idx + 1}) [{nid}] {canon} | {raw}"


def _fmt_weak(item: Dict[str, Any]) -> str:
    canon = item.get("canonical") or item.get("evidence") or ""
    raw = item.get("evidence") or ""
    nid = item.get("note_id") or ""
    score = item.get("score")
    subj_hint = item.get("subj") or item.get("anchor_entity") or "?"
    try:
        score_text = f"{float(score):.2f}"
    except (TypeError, ValueError):
        score_text = "~0.30"
    return f"- [{nid}] (score≈{score_text}, subj≈{subj_hint}) {canon} | {raw}"
