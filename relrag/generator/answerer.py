from typing import Any, Dict, List, Optional

from loguru import logger

from relrag.generator.extractor import judge_and_compress
from relrag.prompt import render_prompt
from relrag.utils.llm_client import LLMChatClient
from relrag.utils.output_protocol import build_final_instruction


ANSWER_PROMPT_NAME = "answerer.txt"


def call_llm(
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
    strong_block = strong_block or "None"
    weak_block = weak_block or "None"
    sanitized_labels = _prepare_allowed_labels(allowed_labels)
    prompt = render_prompt(
        ANSWER_PROMPT_NAME,
        q=question,
        strong_block=strong_block,
        weak_block=weak_block,
        label_instruction=_label_instruction(sanitized_labels, attribute_name),
        final_instruction=build_final_instruction(),
    )

    client = LLMChatClient(
        endpoint=endpoint,
        model=model,
        llm_profile="generate",
        retries=retries,
        timeout=60,
    )
    try:
        response = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content.strip()
    except Exception as exc:  # noqa: PERF203
        logger.warning("Answerer call failed: {}", exc)
        raise


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
