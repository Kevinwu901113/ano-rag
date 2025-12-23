from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from config import config as config_loader
from utils.device import run_with_fallback
from utils.embedding_utils import EmbeddingEncoder
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.output_protocol import build_final_instruction
from utils.llm_client import LLMChatClient
from utils.retrieval_logger import log_retrieval

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    logger.warning("FAISS unavailable: {}", exc)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for multi-hop question answering.\n"
    "You are given several pieces of context that may come from different Wikipedia articles.\n"
    "You may need to combine information from multiple pieces to answer the question.\n"
    "Answer the question with a short phrase. If the answer is not contained in the context, say \"unknown\"."
)

# We relax the prompt slightly to allow chain of thought
# but we still want concise final answers if possible.
PROMPT_TEMPLATE = """Context:
{context}

Question: {question}
{final_instruction}

Answer the question with a short phrase. If the answer is not contained in the context, say "unknown"."""

def _paragraphs_from_record(record: Dict[str, Any]) -> List[str]:
    """Extract text paragraphs from a doc_pool record."""
    text = record.get("text") or record.get("doc_chunk") or ""
    if isinstance(text, list):
        return [str(t) for t in text if str(t).strip()]
    
    # Split by double newline as naive paragraph separator
    return [p.strip() for p in str(text).split("\n\n") if p.strip()]

def _load_doc_pool(path: str) -> Iterable[Dict[str, Any]]:
    """Stream records from doc_pool.json or .jsonl."""
    with open(path, "r", encoding="utf-8") as f:
        # Try to read as list of dicts
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # Load all at once (careful with memory)
            data = json.load(f)
            for item in data:
                yield item
        else:
            # Assume JSONL
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

class NaiveChunker:
    """Simple chunker that splits text by tokens (naive implementation)."""
    
    def __init__(
        self, 
        target_tokens: int = 320,
        max_tokens: int = 384,
        overlap_tokens: int = 64,
        append_title: bool = True
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.append_title = append_title
        
    def chunk(self, text: str, title: str = "") -> List[str]:
        # Naive whitespace splitting approximation
        words = text.split()
        chunks = []
        current_chunk: List[str] = []
        current_len = 0
        
        # Prepend title
        prefix = f"{title}\n" if self.append_title and title else ""
        prefix_len = len(prefix.split()) # Rough
        
        # If text is short, just return it
        if len(words) + prefix_len <= self.max_tokens:
            return [prefix + text]
            
        step = self.target_tokens - self.overlap_tokens
        if step <= 0:
            step = self.target_tokens // 2
            
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.target_tokens]
            chunk_text = " ".join(chunk_words)
            if prefix:
                chunk_text = prefix + chunk_text
            chunks.append(chunk_text)
            
        return chunks


class NaiveIndex:
    """In-memory FAISS index wrapper for retrieval."""

    def __init__(
        self, 
        index_path: str, 
        chunks_path: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        embed_model: Optional[str] = None,
        embed_device: str = "auto",
        embed_batch_size: Optional[int] = None,
        embed_max_length: Optional[int] = None,
        embed_normalize: Optional[bool] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self._embed_model_override = embed_model
        self._embed_device_prefer = embed_device
        if self._embed_device_prefer == "auto":
             self._embed_device_prefer = self.embed_cfg.get("device", "auto")
             
        self._embed_batch_size_override = embed_batch_size
        self._embed_max_length_override = embed_max_length
        self._embed_normalize_override = embed_normalize
        self.embed_device_used: Optional[str] = None
        self.fallback_reason: Optional[str] = None
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")
            
        logger.info("Loading FAISS index from {}", index_path)
        self.index = faiss.read_index(str(index_path))
        
        logger.info("Loading chunks from {}", chunks_path)
        self.chunks: List[Dict[str, Any]] = []
        with open(chunks_path, "rb") as f:
            # Check if it's a pickle file first (magic bytes)
            header = f.read(2)
            f.seek(0)
            if header == b'\x80\x04' or header == b'\x80\x03' or header == b'\x80\x02': # Pickle magic bytes
                import pickle
                try:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        # SimpleSelfRAG stores chunk_store as dict {id: text}
                        # We need to convert to list of dicts for NaiveIndex compatibility
                        # But wait, NaiveIndex expects chunks to align with FAISS indices
                        # If we are loading a SimpleSelfRAG chunk store, we need the meta file to map indices to IDs
                        # This NaiveIndex class is for "naive_rag", not "simple_selfrag".
                        # The evaluator is trying to use NaiveIndex to evaluate SimpleSelfRAG output?
                        # No, the evaluator `evaluate_mirage_retrieval.py` uses `NaiveIndex` when mode is `naive`.
                        # But I pointed it to `result/mirage_selfrag_strict` which has `simple_selfrag` artifacts (pickle).
                        # NaiveIndex expects `chunks.jsonl`.
                        # I symlinked `simple_selfrag_chunk_store.pkl` to `chunks.jsonl`.
                        # So `chunks.jsonl` is actually a pickle file.
                        
                        # I should adapt NaiveIndex to handle pickle chunk stores if I want to use it for evaluation.
                        # OR, I should use a different evaluator or different index class for SelfRAG.
                        # But `evaluate_mirage_retrieval.py` is hardcoded to use `NaiveIndex` for `naive` mode.
                        # So let's make `NaiveIndex` smarter.
                        
                        # Load meta if exists to map index -> chunk_id
                        meta_path = str(index_path) + ".meta.pkl"
                        chunk_ids = []
                        if Path(meta_path).exists():
                            with open(meta_path, "rb") as mf:
                                chunk_ids = pickle.load(mf)
                        
                        # Convert dict store to list based on chunk_ids order (if available) or just values
                        if chunk_ids:
                            self.chunks = []
                            for cid in chunk_ids:
                                text = data.get(cid, "")
                                self.chunks.append({"text": text, "id": cid})
                        else:
                            # Fallback: just list values? No, FAISS index alignment matters.
                            # If no meta, we can't align.
                            logger.error("Loaded pickle chunk store but missing .meta.pkl for alignment. Results will be wrong.")
                            self.chunks = [{"text": v, "id": k} for k,v in data.items()] 
                    elif isinstance(data, list):
                        self.chunks = data
                except Exception as e:
                    logger.error(f"Failed to load pickle: {e}")
            else:
                # Assume JSONL (utf-8 text)
                try:
                    f_text = open(chunks_path, "r", encoding="utf-8")
                    for line in f_text:
                        if line.strip():
                            self.chunks.append(json.loads(line))
                    f_text.close()
                except Exception as e:
                     logger.error(f"Failed to load JSONL: {e}")
        self.last_hits: List[Dict[str, Any]] = []
        
        # Initialize encoder for query embedding
        self.encoder = self._init_encoder()
        
    def _init_encoder(self) -> EmbeddingEncoder:
        provider = self.embed_cfg.get("provider", "qwen3")
        model = self._embed_model_override or self._resolve_model_name()
        cache_dir = self._clean_path(self.embed_cfg.get("cache_dir"))
        dtype = self.embed_cfg.get("dtype")
        max_len = int(self._embed_max_length_override or self.embed_cfg.get("max_len_note", 384))
        batch_size = int(self._embed_batch_size_override or self.embed_cfg.get("batch_size", 4))
        normalize = bool(
            self.embed_cfg.get("normalize", True)
            if self._embed_normalize_override is None
            else self._embed_normalize_override
        )
        return EmbeddingEncoder(
            provider,
            model,
            max_len,
            cache_dir=cache_dir,
            device=self._embed_device_prefer,
            dtype=dtype,
            fallback_to_cpu_on_oom=False,
            batch_size=batch_size,
            normalize=normalize,
        )

    @property
    def embed_cfg(self) -> Dict[str, Any]:
        return (self.cfg.get("retriever", {}) or {}).get("embedding", {}) or {}
        
    def _resolve_model_name(self) -> str:
        override = self.embed_cfg.get("model_path_override")
        base = self.embed_cfg.get("model", "Qwen/Qwen3-Embedding-8B")
        return str(override or base).strip()

    def _resolve_device(self) -> Optional[str]:
        device = self.embed_cfg.get("device")
        if device:
            return str(device)
        return (self.cfg.get("system") or {}).get("device")

    def _clean_path(self, value: Any) -> Optional[str]:
        if not value: return None
        return str(Path(str(value)).expanduser())

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        normalize = bool(
            self.embed_cfg.get("normalize", True)
            if self._embed_normalize_override is None
            else self._embed_normalize_override
        )
        emb, used_device, fallback_reason = run_with_fallback(
            lambda device: self.encoder.encode([query], device=device),
            prefer=self._embed_device_prefer,
        )
        self.embed_device_used = used_device
        self.fallback_reason = fallback_reason
        if emb is None or len(emb) == 0:
            return []

        if normalize:
            faiss.normalize_L2(emb)
            
        # Ensure dimension match
        if emb.shape[1] != self.index.d:
             logger.warning(f"Dimension mismatch: query {emb.shape[1]} vs index {self.index.d}")
             # Simple fix: padding or truncation
             if emb.shape[1] < self.index.d:
                 import numpy as np
                 padding = np.zeros((emb.shape[0], self.index.d - emb.shape[1]), dtype=emb.dtype)
                 emb = np.hstack([emb, padding])
             else:
                 emb = emb[:, :self.index.d]

        scores, indices = self.index.search(emb, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            doc_id = chunk.get("doc_id")
            if not doc_id and chunk_id and "::" in str(chunk_id):
                doc_id = str(chunk_id).split("::", 1)[0]
            passage_id = None
            if chunk_id:
                if "::" in str(chunk_id):
                    passage_id = str(chunk_id).split("::", 1)[1]
                else:
                    passage_id = str(chunk_id)
            results.append(
                {
                    "text": chunk.get("text", ""),
                    "score": float(score),
                    "metadata": chunk,
                    "title": ((chunk.get("meta") or {}).get("doc_title")) if isinstance(chunk.get("meta"), dict) else None,
                    "doc_id": doc_id,
                    "sent_ids": None,
                    "passage_id": passage_id,
                }
            )
        self.last_hits = results
        return results

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve(query, k)


class LLMClient:
    """Minimal vLLM OpenAI-compatible chat client."""

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
        self.client = LLMChatClient(
            endpoint=endpoint,
            model=model,
            llm_profile="generate",
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            retries=retries,
            timeout=120,
        )
        self.model = self.client.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Default: do not set stop tokens. For some models a leading newline is common;
        # forcing "\n" as a stop can truncate the answer to empty content.
        self.stop = [] if stop is None else stop

    def chat(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            llm_profile="generate",
        )
        return str(response.content)


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
        context_budget: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        self.topk = max(1, int(topk))
        self.context_budget = int(context_budget or 0)
        lm_cfg = self.cfg.get("vllm", {}) or {}
        endpoint = lm_endpoint or lm_cfg.get("endpoint")
        model = lm_model or lm_cfg.get("model")
        
        # Ensure strings
        endpoint = str(endpoint) if endpoint else None
        model = str(model) if model else None
        
        temp = temperature if temperature is not None else lm_cfg.get("temperature", 0.0)
        max_new_tokens = max_tokens if max_tokens is not None else lm_cfg.get("max_tokens", 8192)
        self.retriever = NaiveIndex(index_path, chunks_path, config=self.cfg)
        
        if llm_client:
            self.lm = llm_client
        elif endpoint and model:
            self.lm = LLMClient(endpoint, model, temperature=float(temp or 0.0), max_tokens=int(max_new_tokens or 8192), stop=stop)
        else:
            raise ValueError("LLM Client configuration missing")

    def run_dataset(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        work_dir: str,
        limit: Optional[int] = None,
        debug: bool = True,
        dataset_name: str = "mirage",
        run_name: Optional[str] = None,
        resume: bool = False,
        save_every: int = 0,
    ) -> Dict[str, Any]:
        items = list(dataset)
        if limit:
            items = items[:limit]
        work_path = Path(work_dir)
        preds_dir = work_path / "preds"
        artifacts_dir = work_path / "artifacts"
        preds_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        resolved_run_name = run_name or work_path.name
        answers: List[Dict[str, Any]] = []
        qa_rows: List[str] = []
        debug_records: List[str] = []
        pred_raw_records: List[Dict[str, Any]] = []
        completed: set[str] = set()

        answers_path = preds_dir / "answers.json"
        qa_path = preds_dir / "qa.tsv"
        pred_raw_path = preds_dir / "pred_raw.jsonl"
        debug_dir = artifacts_dir / "debug"
        debug_path = debug_dir / "retrieval_raw.jsonl"

        if resume:
            if answers_path.exists():
                try:
                    payload = json.loads(answers_path.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        answers = payload
                        completed = {str(a.get("query_id")) for a in answers if a.get("query_id") is not None}
                except Exception as exc:
                    logger.warning("Failed to load existing answers from {}: {}", answers_path, exc)
            if qa_path.exists():
                try:
                    qa_rows = qa_path.read_text(encoding="utf-8").splitlines()
                except Exception as exc:
                    logger.warning("Failed to load existing QA log from {}: {}", qa_path, exc)
            if debug and debug_path.exists():
                try:
                    debug_records = debug_path.read_text(encoding="utf-8").splitlines()
                except Exception as exc:
                    logger.warning("Failed to load existing debug log from {}: {}", debug_path, exc)
            if pred_raw_path.exists():
                try:
                    pred_raw_records = [
                        json.loads(line)
                        for line in pred_raw_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                except Exception as exc:
                    logger.warning("Failed to load existing pred_raw.jsonl: {}", exc)

        def _flush() -> None:
            answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_path.write_text("\n".join(qa_rows), encoding="utf-8")
            write_jsonl(pred_raw_path, pred_raw_records)
            if debug:
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path.write_text("\n".join(debug_records), encoding="utf-8")

        for idx, item in enumerate(items):
            question = item.get("query") or item.get("question") or ""
            qid = item.get("query_id") or str(idx)
            qid_key = str(qid)
            if resume and qid_key in completed:
                continue
            hits = self.retriever.retrieve(question, k=self.topk)

            annotated_hits = []
            for i, doc in enumerate(hits):
                annotated_hits.append({**doc, "text": f"[{i+1}] {doc['text']}"})
            context_str, contexts_used, context_tokens = pack_contexts(
                annotated_hits, self.context_budget
            )

            prompt = PROMPT_TEMPLATE.format(
                context=context_str,
                question=question,
                final_instruction=build_final_instruction(),
            )
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                raw_answer = self.lm.chat(messages)
                ans_text = raw_answer
            except Exception as exc:
                logger.error("LLM call failed for {}: {}", qid, exc)
                ans_text = "Insufficient evidence"

            answers.append(
                {
                    "query_id": qid,
                    "question": question,
                    "answer": ans_text,
                    "hits": hits,
                    "contexts_used": contexts_used,
                    "context_tokens_used": context_tokens,
                    "context_budget_tokens": self.context_budget or None,
                }
            )
            pred_raw_records.append(
                {
                    "id": str(qid),
                    "question": question,
                    "pred_raw": ans_text,
                    "contexts_used": contexts_used,
                    "context_tokens_used": context_tokens,
                    "context_budget_tokens": self.context_budget or None,
                }
            )
            safe_q = " ".join(str(question).replace("\t", " ").split())
            safe_a = " ".join(str(ans_text).replace("\t", " ").replace('"', "'").split())
            qa_rows.append(f"{safe_q}\t{safe_a}")
            completed.add(qid_key)
            if debug:
                debug_records.append(
                    json.dumps(
                        {"query_id": qid, "question": question, "answer": ans_text, "hits": hits},
                        ensure_ascii=False,
                    )
                )
            try:
                log_retrieval(
                    sample_id=qid,
                    dataset=dataset_name,
                    run_name=resolved_run_name,
                    retrieved=[{**hit, "rank": i + 1} for i, hit in enumerate(hits)],
                    topk=len(hits),
                    final_context=contexts_used,
                    final_context_tokens=context_tokens,
                    context_budget_tokens=self.context_budget or None,
                    log_dir=artifacts_dir,
                )
            except Exception as log_exc:
                logger.error("retrieval logging failed for {}: {}", qid, log_exc)

            if save_every and len(answers) % max(1, int(save_every)) == 0:
                _flush()

        _flush()
        return {"answers": str(answers_path), "qa": str(qa_path)}

    def answer(self, question: str) -> str:
        # 1. Retrieve
        docs = self.retriever.retrieve(question, k=self.topk)
        
        # 2. Construct context
        annotated_hits = []
        for i, doc in enumerate(docs):
            annotated_hits.append({**doc, "text": f"[{i+1}] {doc['text']}"})
        context_str, _, _ = pack_contexts(annotated_hits, self.context_budget)
        
        # 3. Prompt
        prompt = PROMPT_TEMPLATE.format(
            context=context_str,
            question=question,
            final_instruction=build_final_instruction(),
        )
        
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # 4. Generate
        ans = self.lm.chat(messages)
        return ans

def answer(
    question: str, 
    index_path: str, 
    chunk_store_path: str, 
    top_k: int = 5,
    llm_client: Optional[LLMClient] = None,
    context_budget: Optional[int] = None,
) -> str:
    """Convenience function for external scripts."""
    # This instantiates a new runner every time, which is inefficient for loops.
    # But fine for simple scripts.
    runner = NaiveRAGRunner(
        index_path, 
        chunk_store_path, 
        topk=top_k,
        llm_client=llm_client,
        context_budget=context_budget,
    )
    return runner.answer(question)
