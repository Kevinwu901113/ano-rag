from __future__ import annotations

from typing import Dict, List, Tuple

from loguru import logger

from .llm_client import LLMChatClient
from .prompts import ROUTER_PROMPT


class Router:
    """Decide whether to use chunk-based or graph-based structuring."""

    def __init__(self, llm: LLMChatClient, supported_types: List[str]) -> None:
        self.llm = llm
        self.supported_types = [t.lower() for t in supported_types]

    def route(self, question: str, doc_briefs: List[Dict]) -> Tuple[str, str]:
        documents_block = "\n".join(
            f"- [{idx}] {doc.get('title') or doc.get('doc_id')}: {doc.get('brief') or ''}"
            for idx, doc in enumerate(doc_briefs)
        )
        prompt = ROUTER_PROMPT.format(question=question, documents=documents_block)
        logger.info("Router prompt prepared with {} doc briefs", len(doc_briefs))
        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
            llm_profile="extract",
        )
        raw = (resp.content or "").strip().lower()
        chosen = self._normalize_choice(raw)
        logger.info("Router result: raw='{}' -> {}", raw, chosen)
        return chosen, raw

    def _normalize_choice(self, text: str) -> str:
        lowered = (text or "").strip().lower()
        for cand in self.supported_types:
            if lowered.startswith(cand):
                return cand
        # simple heuristics
        if "graph" in lowered:
            return "graph"
        return "chunk"
