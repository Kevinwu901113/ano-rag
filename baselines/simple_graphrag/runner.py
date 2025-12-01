from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
import re

from baselines.simple_graphrag.retriever import GraphRetriever
from config import config as config_loader
from structrag.llm_client import LLMChatClient


def _strip_reasoning(text: str) -> str:
    output = text or ""
    # Strip standard <think> tags if present
    while True:
        start = output.find("<think>")
        if start == -1:
            break
        end = output.find("</think>", start + len("<think>"))
        if end == -1:
            output = output[:start]
            break
        output = output[:start] + output[end + len("</think>") :]
    
    # Heuristic: If the output contains "Answer:", it might be "Reasoning... Answer: Result"
    # We only keep the part after "Answer:" if it appears in the last few lines
    # BUT we must be careful not to strip if "Answer:" is part of the context or question repetition
    # A safer heuristic for reasoning models that don't use <think> is looking for double newlines + "Answer:"
    
    cleaned = output.strip()
    return cleaned or "Insufficient evidence"


def _enforce_short_answer(text: str) -> str:
    """
    Clean up the LLM output to ensure it's just the answer.
    Handles conversational fillers and reasoning that might have slipped through.
    """
    if not text:
        return "Insufficient evidence"
    
    # 1. Handle common conversational prefixes (case-insensitive)
    # We use regex to match these at the start of the string
    conversational_patterns = [
        r"^(okay|ok|so|well|hmm|let's see|let me see|let me look|let me try|i need to|the user is asking|the question is asking|first, i need to|let me go through|let me check|determine)[\.,]?",
        r"^based on the (provided )?context,?",
        r"^the answer is",
        r"^the answer appears to be",
        r"^according to the context,?",
        r"^it seems that",
        # r"^is\s+",  # Removed: too aggressive (e.g. "is a city in France" -> "a city in France" might be okay, but "is 42" -> "42" is risky if answer is "is")
        # r"^occupation is", # Removed: too aggressive
        r"^about",
        r"^(i need to|i must|i should)",
        r"^(from the|in the) (given|provided)? ?context",
        r"^(to find|to determine|to answer|to figure out)",
        # r"^what\s+.*?\s+is", # e.g. "what John's occupation is" -> REMOVED, too risky
        r"^what the answer is",
        r"^the question is asking",
        # r"^\.", # Removed: might strip decimal points?
        r"^the user provided",
        r"^his job, right\?",
        r"^check the context provided",
        # r"^John Floyd's occupation", # Removed specific pattern
        r"^([a-zA-Z0-9' \.-]+)'s occupation is", # e.g. "John Mayne's occupation is"
        r"^the occupation of [a-zA-Z0-9' \.-]+ is", # e.g. "The occupation of John Mayne is"
        # r"^\*\*", # Moved to dedicated markdown cleanup section to avoid breaking paired check
    ]
    
    cleaned = text.strip()
    
    # Iteratively remove prefixes until no more matches found
    # Limit iterations to prevent infinite loops
    for _ in range(10):
        original = cleaned
        for pattern in conversational_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        
        # Also strip Markdown bold/italic markers from the ends if they exist
        # e.g. "**Answer**" -> "Answer"
        # Repeatedly strip to handle cases like "** **Answer** **"
        # Limit inner loop as well
        for _ in range(5):
            prev = cleaned
            # Handle paired markers
            if cleaned.startswith("**") and cleaned.endswith("**") and len(cleaned) >= 4:
                cleaned = cleaned[2:-2].strip()
            elif cleaned.startswith("*") and cleaned.endswith("*") and len(cleaned) >= 2:
                cleaned = cleaned[1:-1].strip()
            
            # Handle unmatched markers (leftover from other cleanups or malformed output)
            if cleaned.startswith("**"):
                cleaned = cleaned[2:].strip()
            if cleaned.endswith("**"):
                cleaned = cleaned[:-2].strip()
                
            if cleaned == prev:
                break
            
        if cleaned == original:
            break

    # Fallback: if the answer is empty after cleaning (e.g. was just "**"), return Insufficient evidence
    if not cleaned:
        # Check if raw text had <think> but no answer outside it
        if "<think>" in text and not cleaned:
             return "Insufficient evidence"
        # If text was short and just punctuation/markers
        return "Insufficient evidence"

    # 2. Handle "Answer:" markers if present
    # Sometimes models output "Reasoning... Answer: X"
    if "Answer:" in cleaned:
        parts = cleaned.split("Answer:")
        # Take the last part as the likely answer
        cleaned = parts[-1].strip()
    elif "answer:" in cleaned.lower():
        # Case-insensitive split if exact case not found
        parts = re.split(r"answer:", cleaned, flags=re.IGNORECASE)
        cleaned = parts[-1].strip()

    # Additional cleanup for conversational endings that might remain
    # e.g. "The answer is X. I hope this helps." -> "X"
    # This is risky but necessary if the model is very chatty
    # We stop at the first newline if multiple lines exist
    
    lines = [L.strip() for L in cleaned.splitlines() if L.strip()]
    if not lines:
        return "Insufficient evidence"
    
    first_line = lines[0]
    
    # If the first line is still very long, it might be a sentence.
    # Try to extract the last few words if it ends with a period?
    # Or if it contains "is a", split there.
    if len(first_line) > 100:
        # Emergency: try to find "is a" / "was a" again
        match = re.search(r"\b(is|was) (a|an|the) (.+?)(\.|$)", first_line, re.IGNORECASE)
        if match:
             candidate = match.group(3).strip()
             if len(candidate) < 50:
                 first_line = candidate

    cleaned = first_line
    # Remove surrounding quotes if present
    if len(cleaned) >= 2 and ((cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'"))):
        cleaned = cleaned[1:-1].strip()
        
    # Remove trailing period if it looks like a sentence end (but be careful with abbreviations)
    # Heuristic: only remove trailing dot if the string is somewhat long or clearly a sentence
    if cleaned.endswith(".") and not cleaned.endswith("Inc.") and not cleaned.endswith("St."):
        cleaned = cleaned[:-1].strip()

    # 4. Final check for multiline output
    # If multiple lines remain, take the first non-empty one
    # (Already handled above by taking first_line)
    final_answer = cleaned
    
    # 5. Check for "Insufficient evidence" variations
    if "insufficient evidence" in final_answer.lower():
        # return "Insufficient evidence" # Don't force it if it's part of a sentence like "not insufficient evidence" (rare but possible)
        # Better: Exact match or close to it
        if len(final_answer) < 30 and "insufficient evidence" in final_answer.lower():
             return "Insufficient evidence"

    return final_answer or "Insufficient evidence"


class SimpleGraphRAGRunner:
    """End-to-end Simple GraphRAG runner over MIRAGE dataset.json."""

    def __init__(
        self,
        graph_path: str,
        chunk_store_path: str,
        *,
        lm_endpoint: Optional[str] = None,
        lm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        
        lm_cfg = self.cfg.get("lmstudio", {}) or {}
        endpoint = lm_endpoint or lm_cfg.get("endpoint")
        model = lm_model or lm_cfg.get("model")
        
        if not endpoint or not model:
             raise ValueError("LM Studio endpoint/model must be configured")
             
        temp = temperature if temperature is not None else lm_cfg.get("temperature", 0.0)
        max_new_tokens = max_tokens if max_tokens is not None else lm_cfg.get("max_tokens", 8192)
        
        llm_client = LLMChatClient(endpoint=endpoint, model=model, temperature=float(temp), max_tokens=int(max_new_tokens))
        
        if not os.path.exists(graph_path) or not os.path.exists(chunk_store_path):
            raise FileNotFoundError(f"Graph files not found: {graph_path}, {chunk_store_path}")
            
        self.retriever = GraphRetriever(graph_path, chunk_store_path, llm_client)

    def run_dataset(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        work_dir: str,
        limit: Optional[int] = None,
        debug: bool = True,
    ) -> Dict[str, Any]:
        items = list(dataset)
        if limit:
            items = items[:limit]
            
        answers: List[Dict[str, Any]] = []
        qa_lines: List[str] = []
        
        for i, item in enumerate(items):
            question = item.get("query") or item.get("question")
            qid = item.get("query_id") or str(i)
            
            try:
                logger.info(f"Processing Q{i}: {question}")
                ans = self.retriever.answer(question)
                cleaned_ans = _strip_reasoning(ans)
                final_ans = _enforce_short_answer(cleaned_ans)
                
                answers.append({
                    "query_id": qid,
                    "question": question,
                    "answer": final_ans,
                    "raw_answer": ans
                })
                
                clean_qa_ans = " ".join(final_ans.split())
                qa_lines.append(f"{question}\t{clean_qa_ans}")
                
            except Exception as e:
                logger.error(f"Error processing Q{i}: {e}")
                answers.append({
                    "query_id": qid,
                    "question": question,
                    "answer": "Error",
                    "error": str(e)
                })

        # Save results
        out_root = Path(work_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        
        output_path = out_root / "answers.json"
        qa_log_path = out_root / "qa.tsv"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
            
        with open(qa_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(qa_lines))
            
        logger.info(f"Simple GraphRAG finished. Results saved to {out_root}")
        
        return {
            "answers": str(output_path),
            "qa": str(qa_log_path),
        }
