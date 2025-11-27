from __future__ import annotations

import json
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
        max_tokens: int = 128,
        stop: Optional[List[str]] = None,
        retries: int = 2,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("Both endpoint and model are required for LLM calls")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Default: do not set stop tokens. For some models a leading newline is common;
        # forcing "\n" as a stop can truncate the answer to empty content.
        self.stop = [] if stop is None else stop
        self.retries = max(0, retries)

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "Insufficient evidence"
        prompt = _build_prompt(question, context)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.stop:
            payload["stop"] = self.stop

        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
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
        temp = temperature if temperature is not None else lm_cfg.get("temperature", 0.0)
        max_new_tokens = max_tokens if max_tokens is not None else lm_cfg.get("max_tokens", 128)
        self.retriever = NaiveIndex(index_path, chunks_path, config=self.cfg)
        self.lm = LLMClient(endpoint, model, temperature=float(temp or 0.0), max_tokens=int(max_new_tokens or 128), stop=stop)

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


PROMPT_TEMPLATE = """You are a factual answerer. Use ONLY the provided context to answer the question.
If the context is insufficient, respond EXACTLY with "Insufficient evidence".
Return only the occupation title without nationality or adjectives (e.g., "lawyer"), with no extra words, no punctuation, no quotes, and no restating the question.

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
    while True:
        start = output.find("<think>")
        if start == -1:
            break
        end = output.find("</think>", start + len("<think>"))
        if end == -1:
            output = output[:start] + output[start + len("<think>") :]
            break
        output = output[:start] + output[end + len("</think>") :]
    cleaned = output.strip()
    return cleaned or "Insufficient evidence"


def _enforce_short_answer(text: str) -> str:
    # Keep only the first non-empty line, strip leading labels/prefixes.
    if not text:
        return "Insufficient evidence"
    candidates = [line.strip() for line in text.splitlines() if line.strip()]
    if not candidates:
        return "Insufficient evidence"
    first = candidates[0]
    # Drop leading "Answer:" or similar prefixes
    for prefix in ("answer:", "ans:", "output:", "prediction:"):
        if first.lower().startswith(prefix):
            first = first[len(prefix) :].strip()
            break
    lowered = first.lower()
    normalized = lowered.strip().rstrip(".")
    if normalized == "insufficient evidence":
        return "Insufficient evidence"
    # Trim surrounding quotes
    if len(first) >= 2 and ((first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'"))):
        first = first[1:-1].strip()
    return first or "Insufficient evidence"
