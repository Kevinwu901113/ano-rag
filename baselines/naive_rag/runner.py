from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from loguru import logger

from config import config as config_loader
from utils.embedding_utils import EmbeddingEncoder

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    logger.warning("FAISS unavailable: {}", exc)


class NaiveIndex:
    """In-memory wrapper around naive chunks + FAISS index."""

    def __init__(
        self,
        index_path: str,
        chunks_path: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if faiss is None:
            raise RuntimeError("FAISS is required for naive retrieval")
        self.cfg = config or config_loader.load_config()
        retr_cfg = self.cfg.get("retriever", {}) or {}
        self.embed_cfg = retr_cfg.get("embedding", {}) or {}
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self._index = self._load_index()
        self._chunks = self._load_chunks()
        self._by_vec: Dict[int, Dict[str, Any]] = {
            int(c["vector_id"]): c for c in self._chunks if c.get("vector_id") is not None
        }
        if not self._by_vec:
            logger.warning("No vector_id found in chunks; assigning sequential ids.")
            for idx, chunk in enumerate(self._chunks):
                chunk["vector_id"] = idx
                self._by_vec[idx] = chunk
        self._encoder = self._init_encoder()

    def search(self, question: str, topk: int) -> List[Dict[str, Any]]:
        q = (question or "").strip()
        if not q:
            return []
        encoded = self._encoder.encode([q])
        if encoded.size == 0:
            return []
        if bool(self.embed_cfg.get("normalize", True)):
            faiss.normalize_L2(encoded)
        limit = min(topk, self._index.ntotal)
        if limit <= 0:
            return []
        scores, ids = self._index.search(encoded.astype("float32"), limit)
        ranked: List[Dict[str, Any]] = []
        for rank, (score, vec_id) in enumerate(zip(scores[0], ids[0]), start=1):
            if vec_id < 0:
                continue
            meta = self._by_vec.get(int(vec_id))
            if not meta:
                continue
            ranked.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "doc_id": meta.get("doc_id"),
                    "text": meta.get("text"),
                    "doc_title": meta.get("meta", {}).get("doc_title") or meta.get("doc_id"),
                    "score": float(score),
                    "rank": rank,
                    "vector_id": int(vec_id),
                }
            )
        return ranked

    def _load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        return faiss.read_index(str(self.index_path))

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")
        records: List[Dict[str, Any]] = []
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _init_encoder(self) -> EmbeddingEncoder:
        provider = self.embed_cfg.get("provider", "qwen3")
        model = self._resolve_model_name()
        cache_dir = self._clean_path(self.embed_cfg.get("cache_dir"))
        device = self._resolve_device()
        dtype = self.embed_cfg.get("dtype")
        max_len = int(self.embed_cfg.get("max_len_note", 384))
        return EmbeddingEncoder(provider, model, max_len, cache_dir=cache_dir, device=device, dtype=dtype)

    def _resolve_model_name(self) -> str:
        override = self.embed_cfg.get("model_path_override")
        base = self.embed_cfg.get("model", "Qwen/Qwen3-Embedding-8B")
        candidate = str(override or base).strip()
        if not candidate:
            raise ValueError("Embedding model name is not configured")
        if override:
            logger.info("Embedding model override detected: {}", candidate)
        return candidate

    def _resolve_device(self) -> Optional[str]:
        device = self.embed_cfg.get("device")
        if device:
            return str(device)
        system_cfg = self.cfg.get("system") or {}
        return system_cfg.get("device")

    def _clean_path(self, value: Any) -> Optional[str]:
        if not value:
            return None
        return str(Path(str(value)).expanduser())


class LLMClient:
    """Minimal LM Studio/OpenAI-compatible chat client."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        stop: Optional[List[str]] = None,
        retries: int = 2,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("Both endpoint and model are required for LLM calls")
        self.endpoint = endpoint.rstrip("/")
        if self.endpoint.endswith("/v1"):
            self.endpoint = self.endpoint[:-3]
        
        # Fix: Ensure endpoint has scheme
        if not self.endpoint.startswith("http://") and not self.endpoint.startswith("https://"):
            self.endpoint = "http://" + self.endpoint
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Default: do not set stop tokens. For some models a leading newline is common;
        # forcing "\n" as a stop can truncate the answer to empty content.
        self.stop = [] if stop is None else stop
        self.retries = max(0, retries)

    def answer(self, question: str, context: str) -> str:
        # Allow empty context if needed (e.g. summarization where prompt contains text)
        # But typically Naive RAG uses context.
        # Raptor baseline sometimes passes empty context string if it's doing pure summarization 
        # (where question is the prompt and context is empty string).
        # So we should relax the check:
        if not context.strip() and not question.strip():
             return "Insufficient evidence"

        # Truncate context if it's too long to avoid 400 Bad Request
        # Qwen context window is large, but let's be safe around 30k chars ~ 8k tokens
        if len(context) > 20000:
             logger.warning(f"Context too long ({len(context)} chars), truncating to 20000 chars")
             context = context[:20000] + "..."
             
        # If context is empty, prompt is just the question (summarization use case)
        # If context is present, we build the RAG prompt.
        if context.strip():
            prompt = _build_prompt(question, context)
        else:
            prompt = question

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.stop and len(self.stop) > 0:
            payload["stop"] = self.stop

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-no-key-required"
        }

        for attempt in range(self.retries + 1):
            try:
                url = f"{self.endpoint}/v1/chat/completions"
                # Ensure we don't double-slash if endpoint already had trailing slash (we rstrip it above but just in case)
                if self.endpoint.endswith("/"):
                    url = f"{self.endpoint}v1/chat/completions"
                
                # DEBUG: Log payload on first attempt
                if attempt == 0:
                    logger.debug(f"LLM Payload: {json.dumps(payload, ensure_ascii=False)}")

                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                if "choices" not in data:
                    logger.error(f"Invalid LLM response: {data}")
                    return "Insufficient evidence"
                content = data["choices"][0]["message"]["content"]
                
                # Handle <think> blocks
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content.startswith("<think>"):
                    content = re.sub(r"^<think>.*", "", content, flags=re.DOTALL).strip()
                    
                cleaned = _strip_reasoning(content)
                return _enforce_short_answer(cleaned)
            except requests.RequestException as exc:  # noqa: PERF203
                if attempt >= self.retries:
                    logger.error("LLM call failed after {} attempts: {}", attempt + 1, exc)
                    return "Insufficient evidence"
                backoff = 2 ** attempt
                logger.warning("LLM call failed (attempt {}): {}; retrying in {}s", attempt + 1, exc, backoff)
                time.sleep(backoff)
        return "Insufficient evidence"


class NaiveRAGRunner:
    """End-to-end naive RAG over MIRAGE dataset.json."""

    def __init__(
        self,
        index_path: str,
        chunks_path: str,
        *,
        topk: int = 5,
        lm_endpoint: Optional[str] = None,
        lm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        self.topk = max(1, int(topk))
        lm_cfg = self.cfg.get("lmstudio", {}) or {}
        endpoint = lm_endpoint or lm_cfg.get("endpoint")
        model = lm_model or lm_cfg.get("model")
        
        # Ensure strings
        endpoint = str(endpoint) if endpoint else None
        model = str(model) if model else None
        
        if model and "qwen" in model.lower() and "instruct" in model.lower():
             # Fix potential model name mismatch if user config has lowercase but server has MixedCase
             # This is a heuristic; ideally we list models from server
             pass
             
        temp = temperature if temperature is not None else lm_cfg.get("temperature", 0.0)
        max_new_tokens = max_tokens if max_tokens is not None else lm_cfg.get("max_tokens", 8192)
        self.retriever = NaiveIndex(index_path, chunks_path, config=self.cfg)
        self.lm = LLMClient(endpoint, model, temperature=float(temp or 0.0), max_tokens=int(max_new_tokens or 8192), stop=stop)

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
        qa_rows: List[str] = []
        qa_rows_no_header: List[str] = []
        qa_with_q: List[str] = []
        debug_records: List[str] = []
        for idx, item in enumerate(items):
            question = item.get("query") or item.get("question") or ""
            qid = item.get("query_id") or str(idx)
            hits = self.retriever.search(question, self.topk)
            context = _format_context(hits)
            answer = self.lm.answer(question, context)
            answers.append(
                {
                    "query_id": qid,
                    "question": question,
                    "answer": answer,
                    "hits": hits,
                }
            )
            qa_rows.append(f"{question}\t{answer}")
            qa_rows_no_header.append(f"{question}\t{answer}")
            qa_with_q.append(f"{question}\t{answer}")
            if debug:
                debug_records.append(
                    json.dumps(
                        {
                            "query_id": qid,
                            "question": question,
                            "answer": answer,
                            "hits": hits,
                        },
                        ensure_ascii=False,
                    )
                )

        out_root = Path(work_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        answers_path = out_root / "answers.json"
        qa_path = out_root / "qa.tsv"
        qa_no_header_path = out_root / "qa.no_header.tsv"
        qa_q_path = out_root / "qa_with_question.tsv"
        debug_dir = out_root / "debug"
        answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
        qa_path.write_text("\n".join(qa_rows), encoding="utf-8")
        qa_no_header_path.write_text("\n".join(qa_rows_no_header), encoding="utf-8")
        qa_q_path.write_text("\n".join(qa_with_q), encoding="utf-8")
        if debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "retrieval.jsonl").write_text("\n".join(debug_records), encoding="utf-8")
        logger.info("Naive RAG run complete: {} examples -> {}", len(answers), qa_path)
        return {
            "answers": str(answers_path),
            "qa": str(qa_path),
            "qa_no_header": str(qa_no_header_path),
            "qa_with_question": str(qa_q_path),
            "debug": str(debug_dir) if debug else None,
        }


PROMPT_TEMPLATE = """Answer the question based on the context below. Keep the answer short and concise. Do not output reasoning.

Context:
{context}

Question: {question}
Answer:"""


def _build_prompt(question: str, context: str) -> str:
    safe_context = context.replace("{", "{{").replace("}", "}}")
    safe_question = question.replace("{", "{{").replace("}", "}}")
    return PROMPT_TEMPLATE.format(context=safe_context, question=safe_question)


def _format_context(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, hit in enumerate(hits, start=1):
        text = str(hit.get("text") or "")
        lines.append(f"[{idx}] {text}")
    return "\n\n".join(lines)


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
    ]
    
    cleaned = text.strip()
    
    # Iteratively remove prefixes until no more matches found
    while True:
        original = cleaned
        for pattern in conversational_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == original:
            break
            
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
