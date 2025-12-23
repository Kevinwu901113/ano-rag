from __future__ import annotations

import json
from typing import Dict, List, Tuple

from loguru import logger

from .llm_client import LLMChatClient
from .prompts import DECOMPOSITION_PROMPT, FINAL_ANSWER_PROMPT, SELECTION_PROMPT


class Utilizer:
    """Run sub-question reasoning over structured data and generate final answers."""

    def __init__(
        self,
        llm: LLMChatClient,
        *,
        max_candidates: int = 32,
        max_tokens_answer: int = 96,
    ) -> None:
        self.llm = llm
        self.max_candidates = max_candidates
        self.max_tokens_answer = max_tokens_answer

    def answer(self, question: str, structure_type: str, structured: Dict[str, List[Dict]]) -> Tuple[str, List[str]]:
        subquestions = self._decompose(question)
        candidates, candidate_map = self._prepare_candidates(structure_type, structured)
        selections = self._select(subquestions, candidates, structure_type)
        evidence_lines, evidence_map = self._gather_evidence(selections, candidate_map)
        answer = self._generate_answer(question, selections, evidence_map)
        return answer, evidence_lines

    def _decompose(self, question: str) -> List[str]:
        prompt = DECOMPOSITION_PROMPT.format(question=question.replace("{", "{{").replace("}", "}}"))
        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=196,
            temperature=0.0,
            llm_profile="extract",
        )
        parsed = self._parse_json_array(resp.content, fallback=[question])
        subqs = [str(item).strip() for item in parsed if str(item).strip()]
        if not subqs:
            subqs = [question]
        logger.info("Decomposed into {} sub-questions", len(subqs))
        return subqs

    def _prepare_candidates(self, structure_type: str, structured: Dict[str, List[Dict]]) -> Tuple[List[str], Dict[int, str]]:
        items: List[str] = []
        mapping: Dict[int, str] = {}
        if structure_type == "graph":
            triples = structured.get("triples") or []
            for idx, t in enumerate(triples[: self.max_candidates]):
                text = f"{t.get('head')} --{t.get('relation')}--> {t.get('tail')} (source: {t.get('doc_title') or t.get('doc_id')})"
                items.append(f"[{idx}] {text}")
                mapping[idx] = text
        else:
            chunks = structured.get("chunks") or []
            for idx, chunk in enumerate(chunks[: self.max_candidates]):
                prefix = chunk.get("doc_title") or chunk.get("doc_id") or ""
                text = f"{prefix}: {chunk.get('text')}"
                items.append(f"[{idx}] {text}")
                mapping[idx] = text
        logger.info("Prepared {} {} candidates for selection", len(items), structure_type)
        return items, mapping

    def _select(self, subquestions: List[str], candidates: List[str], structure_type: str) -> List[Dict]:
        if not subquestions:
            subquestions = ["Use provided evidence to answer."]
        subq_lines = "\n".join(f"- {q}" for q in subquestions)
        cand_block = "\n".join(candidates)
        prompt = SELECTION_PROMPT.format(
            item_type=structure_type if structure_type in {"chunk", "graph"} else "chunk",
            subquestions=subq_lines,
            candidates=cand_block,
        )
        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
            llm_profile="extract",
        )
        parsed = self._parse_json_array(resp.content, fallback=[])
        selections: List[Dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            sq = str(item.get("subquestion") or "").strip()
            ids = item.get("evidence_ids")
            if not isinstance(ids, list):
                continue
            cleaned_ids: List[int] = []
            for val in ids:
                try:
                    cleaned_ids.append(int(val))
                except Exception:
                    continue
            selections.append({"subquestion": sq or subquestions[0], "evidence_ids": cleaned_ids})
        if not selections:
            # Fallback: attach top-2 candidates to the first subquestion
            selections = [{"subquestion": subquestions[0], "evidence_ids": list(range(min(2, len(candidates))))}]
        logger.info("Selection complete for {} sub-questions", len(selections))
        return selections

    def _gather_evidence(self, selections: List[Dict], candidate_map: Dict[int, str]) -> Tuple[List[str], Dict[int, str]]:
        evidence_list: List[str] = []
        evidence_map: Dict[int, str] = {}
        for sel in selections:
            for cid in sel.get("evidence_ids", []):
                if cid in evidence_map:
                    continue
                text = candidate_map.get(cid)
                if text:
                    evidence_map[cid] = text
                    evidence_list.append(text)
        return evidence_list, evidence_map

    def _generate_answer(self, question: str, selections: List[Dict], evidence_map: Dict[int, str]) -> str:
        block_lines: List[str] = []
        for sel in selections:
            block_lines.append(f"Sub-question: {sel.get('subquestion')}")
            ev_ids = sel.get("evidence_ids") or []
            if not ev_ids:
                block_lines.append("Evidence: None")
                continue
            for cid in ev_ids:
                text = evidence_map.get(cid, "")
                block_lines.append(f"- [{cid}] {text}")
        evidence_block = "\n".join(block_lines) if block_lines else "None"
        prompt = FINAL_ANSWER_PROMPT.format(question=question, evidence_block=evidence_block)
        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens_answer,
            temperature=0.2,
            llm_profile="generate",
        )
        answer = (resp.content or "").strip()
        logger.info("Final answer generated (len={}): {}", len(answer), answer[:120])
        return answer or "Insufficient evidence"

    def _parse_json_array(self, text: str, fallback: List) -> List:
        if not text:
            return fallback
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        # Try to locate JSON array substring
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, list):
                    return data
            except Exception:
                return fallback
        return fallback
