import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from scripts.hotpotqa.baselines.baseline_utils import (
    build_passage_entries,
    save_predictions_and_qa,
    TransformerEmbedder,
)
from utils.bm25 import BM25Index, rrf_fuse
from utils.context_budget import pack_contexts
from utils.device import run_with_fallback
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.output_protocol import build_final_instruction
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved
from config.config_loader import DEFAULT_EMBED_MODEL, DEFAULT_EMBED_DEVICE

_CACHE_KEY_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sha256_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _slugify(text: str) -> str:
    return _CACHE_KEY_SAFE.sub("_", text or "").strip("_") or "unknown"


def _infer_dataset_id(dataset_path: Path) -> str:
    stem = dataset_path.stem
    parent = dataset_path.parent.name
    if stem in {"dataset", "data"}:
        return parent or stem
    if stem.startswith("dataset_") and parent:
        return f"{parent}_{stem[len('dataset_'):]}"
    return stem


def _build_cache_key(dataset_path: Path, args) -> Tuple[str, Dict[str, Any]]:
    dataset_id = _infer_dataset_id(dataset_path)
    dataset_hash = _sha256_file(dataset_path)
    model = str(args.embed_model or DEFAULT_EMBED_MODEL)
    device = str(args.embed_device or DEFAULT_EMBED_DEVICE)
    max_len = int(args.embed_max_length or 0)
    normalize = bool(args.embed_normalize)
    dtype = args.emb_dtype or "auto"
    batch_size = int(args.embed_batch_size or 0)
    max_context = int(args.max_context or 0)
    index_kind = "in_memory_cosine"
    key = "__".join(
        [
            _slugify(dataset_id),
            _slugify(model),
            _slugify(device),
            f"len{max_len}",
            f"norm{1 if normalize else 0}",
            f"dtype{_slugify(str(dtype))}",
            f"batch{batch_size}",
            f"maxctx{max_context}",
            f"index{index_kind}",
            f"data{dataset_hash[:12]}",
        ]
    )
    meta = {
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "embed_model": model,
        "embed_device": device,
        "embed_max_length": max_len,
        "embed_normalize": normalize,
        "embed_dtype": dtype,
        "embed_batch_size": batch_size,
        "max_context": max_context,
        "index_kind": index_kind,
    }
    return key, meta

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class InMemoryVanillaRetriever:
    def __init__(self, embedder: TransformerEmbedder, *, embed_device: str):
        self.embedder = embedder
        self.embed_device = embed_device
        self.passages: List[str] = []
        self.entries: List[Dict[str, Any]] = []
        self.vectors = None
        self.last_hits: List[Dict[str, Any]] = []
        self.dim: Optional[int] = None
        self.embed_device_used_build: Optional[str] = None
        self.embed_device_used_query: Optional[str] = None
        self.fallback_reason_build: Optional[str] = None
        self.fallback_reason_query: Optional[str] = None
        self.last_scores: Optional[np.ndarray] = None
        
    def build_index_for_question(self, entries: List[Dict[str, Any]]):
        """
        Build a temporary index for the paragraphs provided in the distractor setting.
        """
        self.entries = entries
        self.passages = [entry["text"] for entry in entries]
        if not self.passages:
            self.vectors = None
            return

        self.vectors, used_device, fallback_reason = run_with_fallback(
            lambda device: self.embedder.encode(self.passages, device=device),
            prefer=self.embed_device,
        )
        self.embed_device_used_build = used_device
        self.fallback_reason_build = fallback_reason
        if self.vectors is not None and getattr(self.vectors, "size", 0) > 0:
            self.dim = int(self.vectors.shape[1])
        # Normalize for cosine similarity
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def load_cached_index(
        self,
        entries: List[Dict[str, Any]],
        vectors: Optional[np.ndarray],
        *,
        embed_device_used: Optional[str] = None,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """
        Load precomputed (already normalized) vectors for the current question.
        """
        self.entries = entries
        self.passages = [entry["text"] for entry in entries]
        if vectors is None or getattr(vectors, "size", 0) == 0:
            self.vectors = None
            return
        self.vectors = vectors
        self.dim = int(vectors.shape[1])
        self.embed_device_used_build = embed_device_used
        self.fallback_reason_build = fallback_reason

    def score(self, query: str) -> np.ndarray:
        if self.vectors is None or len(self.passages) == 0:
            self.last_scores = None
            return np.array([])
            
        query_vec, used_device, fallback_reason = run_with_fallback(
            lambda device: self.embedder.encode([query], device=device),
            prefer=self.embed_device,
        )
        self.embed_device_used_query = used_device
        self.fallback_reason_query = fallback_reason
        if query_vec is not None and getattr(query_vec, "size", 0) > 0 and self.dim is None:
            self.dim = int(query_vec.shape[1])
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / (norm + 1e-10)
        
        scores = np.dot(self.vectors, query_vec.T).flatten()
        self.last_scores = scores
        return scores

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.passages) == 0:
            return []

        scores = self.score(query)
        # Get top k
        indices = np.argsort(scores)[::-1][:k]
        
        results: List[Dict[str, Any]] = []
        for idx in indices:
            entry = self.entries[idx]
            results.append(
                {
                    "text": self.passages[idx],
                    "score": float(scores[idx]),
                    "doc_id": entry.get("doc_id"),
                    "sent_ids": entry.get("sent_ids"),
                    "passage_id": entry.get("passage_id"),
                }
            )
        self.last_hits = results
        return results

def process_example(item: Dict[str, Any], 
                    llm: Optional[Any], 
                    embedder: Optional[TransformerEmbedder],
                    args,
                    *,
                    run_name: str,
                    dataset_name: str,
                    log_dir: Path,
                    embed_meta: Dict[str, Any],
                    embed_meta_lock: threading.Lock,
                    cache_embeddings: Optional[np.ndarray] = None,
                    cache_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
                    cache_embed_device: Optional[str] = None,
                    cache_fallback_reason: Optional[str] = None,
                    ) -> Tuple[str, str, str, List[List[Any]], List[Dict[str, Any]], int]:
    """
    Process a single HotpotQA example.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passage_entries = build_passage_entries(context, max_passages=args.max_context)
    hits: List[Dict[str, Any]] = []
    if args.retriever in ("dense", "hybrid"):
        if embedder is None:
            raise RuntimeError("Embedding model required for dense/hybrid retrieval.")
        retriever = InMemoryVanillaRetriever(embedder, embed_device=args.embed_device)

        # 1. Build small index for this question
        cached_vectors = None
        if cache_embeddings is not None and cache_offsets is not None:
            qid_key = str(qid)
            offset = cache_offsets.get(qid_key)
            if offset:
                start, count = offset
                if count > 0:
                    cached_vectors = cache_embeddings[start : start + count]
        if cached_vectors is not None:
            retriever.load_cached_index(
                passage_entries,
                cached_vectors,
                embed_device_used=cache_embed_device,
                fallback_reason=cache_fallback_reason,
            )
        else:
            retriever.build_index_for_question(passage_entries)

        # 2. Dense scores
        dense_scores = retriever.score(question)
        if args.retriever == "dense":
            hits = retriever.retrieve(question, k=args.topk)
        else:
            bm25 = BM25Index([entry["text"] for entry in passage_entries])
            bm25_scores = bm25.get_scores(question)
            dense_rank = list(np.argsort(dense_scores)[::-1])
            bm25_rank = list(np.argsort(bm25_scores)[::-1])
            fused = rrf_fuse([dense_rank, bm25_rank], k=int(getattr(args, "hybrid_rrf_k", 60)))
            indices = sorted(fused, key=fused.get, reverse=True)[: args.topk]
            for idx in indices:
                entry = passage_entries[idx]
                hits.append(
                    {
                        "text": entry["text"],
                        "score": float(fused.get(idx, 0.0)),
                        "doc_id": entry.get("doc_id"),
                        "sent_ids": entry.get("sent_ids"),
                        "passage_id": entry.get("passage_id"),
                    }
                )

        with embed_meta_lock:
            for used in (retriever.embed_device_used_build, retriever.embed_device_used_query):
                if used:
                    embed_meta["used_devices"].add(str(used))
            for reason in (retriever.fallback_reason_build, retriever.fallback_reason_query):
                if reason:
                    embed_meta["fallback_reasons"].append(str(reason))
            if embed_meta.get("dim") is None and retriever.dim:
                embed_meta["dim"] = int(retriever.dim)
    else:
        bm25 = BM25Index([entry["text"] for entry in passage_entries])
        scores = bm25.get_scores(question)
        indices = list(np.argsort(scores)[::-1][: args.topk])
        for idx in indices:
            entry = passage_entries[idx]
            hits.append(
                {
                    "text": entry["text"],
                    "score": float(scores[idx]),
                    "doc_id": entry.get("doc_id"),
                    "sent_ids": entry.get("sent_ids"),
                    "passage_id": entry.get("passage_id"),
                }
            )

    context_str, contexts_used, context_tokens = pack_contexts(
        hits, int(getattr(args, "context_budget", 0) or 0)
    )
    # Log retrieval regardless of answer generation
    try:
        log_retrieval(
            sample_id=qid,
            dataset=dataset_name,
            run_name=run_name,
            retrieved=[
                {**hit, "rank": idx + 1} for idx, hit in enumerate(hits)
            ],
            topk=len(hits),
            final_context=contexts_used,
            final_context_tokens=context_tokens,
            context_budget_tokens=int(getattr(args, "context_budget", 0) or 0) or None,
            log_dir=log_dir,
        )
    except Exception as log_exc:
        logger.error(f"retrieval logging failed for {qid}: {log_exc}")

    if getattr(args, "retrieval_only", False):
        sp = []
        return qid, question, "", sp, contexts_used, context_tokens
    
    # 4. Prompt
    prompt = f"""Answer the question based on the selected paragraphs.
Keep the answer concise.
{build_final_instruction()}

{context_str}

Question: {question}
Answer:"""

    # 5. Generate
    try:
        if llm is None:
            raise RuntimeError("LLM client is required unless --retrieval-only is set")
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = resp.content
        
        sp = [] # Not predicting supporting facts for now
        return qid, question, ans, sp, contexts_used, context_tokens
        
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        return qid, question, "error", [], contexts_used, context_tokens

def main():
    parser = argparse.ArgumentParser(description="Run Vanilla RAG Baseline on HotpotQA Distractor Setting")
    parser.add_argument("--dataset", required=True, help="Path to hotpotqa distractor dev/test json")
    parser.add_argument("--output", default=None, help="Output path for prediction json (default: work_dir/pred.json)")
    parser.add_argument("--qa-path", default=None, help="Optional QA log path (default: work_dir/qa.tsv)")
    parser.add_argument("--result-root", default="result_relrag", help="Root directory for auto workspace creation")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Workspace directory (default: auto under result-root)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--lm-endpoint", default="http://127.0.0.1:8000/v1", help="LLM API endpoint")
    parser.add_argument("--lm-model", default="qwen3-30b-a3b", help="LLM model name")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for LLM decoding")
    parser.add_argument(
        "--embed-model",
        "--emb-model",
        dest="embed_model",
        default=DEFAULT_EMBED_MODEL,
        help="Embedding model name or path",
    )
    parser.add_argument(
        "--embed-device",
        "--emb-device",
        dest="embed_device",
        choices=["auto", "cuda", "cpu"],
        default=DEFAULT_EMBED_DEVICE,
        help="Embedding device preference (auto prefers CUDA, falls back to CPU)",
    )
    parser.add_argument("--embed-batch-size", type=int, default=4, help="Embedding batch size")
    parser.add_argument("--embed-max-length", type=int, default=512, help="Embedding max sequence length")
    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--embed-normalize", dest="embed_normalize", action="store_true", help="L2-normalize embeddings")
    norm.add_argument(
        "--no-embed-normalize",
        dest="embed_normalize",
        action="store_false",
        help="Disable L2-normalization",
    )
    parser.set_defaults(embed_normalize=True)
    parser.add_argument("--emb-dtype", default=None, help="Embedding torch dtype (e.g., float16, bfloat16)")
    parser.add_argument("--topk", type=int, default=3, help="Number of paragraphs to retrieve from the 10 distractors")
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Retrieval mode (dense, bm25, hybrid)",
    )
    parser.add_argument("--hybrid-rrf-k", type=int, default=60, help="RRF k for hybrid fusion")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls; only run retrieval and log retrieval.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Test on N examples")
    parser.add_argument("--max-context", type=int, default=10, help="Max number of paragraphs from context to keep")
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    # Workspace setup
    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="hotpotqa")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    run_name = work_dir.name
    dataset_name = "hotpotqa"
    output_path = Path(args.output) if args.output else preds_dir / "pred.json"
    qa_path = Path(args.qa_path) if args.qa_path else preds_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir}")
    setup_logging(str(work_dir / "run.log"))
    cache_key: Optional[str] = None
    cache_path: Optional[str] = None
    cache_hit: Optional[bool] = None
    cache_meta: Dict[str, Any] = {}
    cache_embeddings: Optional[np.ndarray] = None
    cache_offsets: Optional[Dict[str, Tuple[int, int]]] = None
    cache_embed_device: Optional[str] = None
    cache_fallback_reason: Optional[str] = None
    cache_paths: Dict[str, Path] = {}
    cache_enabled = args.retriever in ("dense", "hybrid")
    cache_dir: Optional[Path] = None
    if cache_enabled:
        dataset_path = Path(args.dataset)
        cache_key, cache_meta = _build_cache_key(dataset_path, args)
        cache_dir = Path(args.result_root) / "cache" / cache_key
        cache_path = str(cache_dir)
        cache_paths = {
            "embeddings": cache_dir / "embeddings.npy",
            "offsets": cache_dir / "offsets.json",
            "metadata": cache_dir / "metadata.json",
            "build_log": cache_dir / "build.log",
            "index": cache_dir / "faiss.index",
        }
        cache_hit = all(path.exists() for path in cache_paths.values())
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="hotpotqa",
            model=args.lm_model,
            endpoint=args.lm_endpoint,
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            context_budget=args.context_budget or None,
            topk=args.topk,
            decode={
                "temperature": 0.0,
                "top_p": None,
                "repetition_penalty": None,
                "max_tokens": args.max_new_tokens,
            },
            embedding={
                "model": args.embed_model,
                "device": args.embed_device,
                "batch_size": args.embed_batch_size,
                "max_length": args.embed_max_length,
                "normalize": args.embed_normalize,
                "dtype": args.emb_dtype,
            },
            budgets={
                "context_budget_tokens": args.context_budget or None,
                "topk": args.topk,
            },
            extra={
                "max_context": args.max_context,
                "embed_model": args.embed_model,
                "retriever": args.retriever,
                "cache_key": cache_key,
                "cache_path": cache_path,
                "cache_hit": cache_hit,
            },
        ),
    )

    # Initialize Clients
    llm = None
    if not args.retrieval_only:
        from structrag.llm_client import LLMChatClient

        llm = LLMChatClient(
            endpoint=args.lm_endpoint,
            model=args.lm_model,
            temperature=0.0,
            max_tokens=args.max_new_tokens if args.max_new_tokens is not None else 8192,
        )

    embedder = None
    if args.retriever in ("dense", "hybrid"):
        embedder = TransformerEmbedder(
            args.embed_model,
            torch_dtype=args.emb_dtype,
            batch_size=args.embed_batch_size,
            max_length=args.embed_max_length,
            normalize=args.embed_normalize,
        )
        logger.info(
            "Embedding model {} (prefer={}, batch_size={}, max_length={}, normalize={}, dtype={})",
            args.embed_model,
            args.embed_device,
            args.embed_batch_size,
            args.embed_max_length,
            args.embed_normalize,
            args.emb_dtype or "auto",
        )
    else:
        logger.info("BM25 retrieval selected; skipping embedding model init.")
    embed_meta: Dict[str, Any] = {"used_devices": set(), "fallback_reasons": [], "dim": None}
    embed_meta_lock = threading.Lock()

    if cache_enabled and cache_dir is not None and cache_paths:
        if cache_hit:
            logger.info("Embedding cache hit: {} -> {}", cache_key, cache_path)
            cache_embeddings = np.load(cache_paths["embeddings"])
            offsets_payload = json.loads(cache_paths["offsets"].read_text(encoding="utf-8"))
            cache_offsets = {str(k): (int(v[0]), int(v[1])) for k, v in offsets_payload.items()}
            cache_meta_loaded = json.loads(cache_paths["metadata"].read_text(encoding="utf-8"))
            cache_embed_device = cache_meta_loaded.get("embed_device_used")
            cache_fallback_reason = cache_meta_loaded.get("fallback_reason")
            if cache_meta_loaded.get("vector_dim") and embed_meta.get("dim") is None:
                embed_meta["dim"] = int(cache_meta_loaded.get("vector_dim") or 0)
        else:
            logger.info("Embedding cache miss: {} -> {}", cache_key, cache_path)
            if embedder is None:
                raise RuntimeError("Embedding model required to build cache.")
            build_started = time.time()
            cache_dir.mkdir(parents=True, exist_ok=True)
            all_vectors: List[np.ndarray] = []
            offsets: Dict[str, Tuple[int, int]] = {}
            used_devices: List[str] = []
            fallback_reasons: List[str] = []
            dim: Optional[int] = None
            cursor = 0
            for item in data:
                qid = item.get("_id") or item.get("id")
                if not qid:
                    continue
                passage_entries = build_passage_entries(item["context"], max_passages=args.max_context)
                texts = [entry["text"] for entry in passage_entries]
                key = str(qid)
                if not texts:
                    offsets[key] = (cursor, 0)
                    continue
                vectors, used_device, fallback_reason = run_with_fallback(
                    lambda device: embedder.encode(texts, device=device),
                    prefer=args.embed_device,
                )
                used_devices.append(str(used_device))
                if fallback_reason:
                    fallback_reasons.append(str(fallback_reason))
                if vectors is None or getattr(vectors, "size", 0) == 0:
                    offsets[key] = (cursor, 0)
                    continue
                norm = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / (norm + 1e-10)
                if dim is None:
                    dim = int(vectors.shape[1])
                all_vectors.append(vectors)
                count = int(vectors.shape[0])
                offsets[key] = (cursor, count)
                cursor += count
            if all_vectors:
                cache_embeddings = np.vstack(all_vectors).astype("float32")
            else:
                cache_embeddings = np.zeros((0, 0), dtype="float32")
            cache_offsets = offsets
            np.save(cache_paths["embeddings"], cache_embeddings)
            cache_paths["offsets"].write_text(
                json.dumps({k: [v[0], v[1]] for k, v in offsets.items()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cache_embed_device = sorted(set(used_devices))[0] if used_devices else None
            cache_fallback_reason = fallback_reasons[0] if fallback_reasons else None
            meta_payload = {
                **cache_meta,
                "cache_key": cache_key,
                "cache_path": cache_path,
                "dataset_path": str(Path(args.dataset)),
                "samples": len(offsets),
                "total_vectors": int(cache_embeddings.shape[0]),
                "vector_dim": int(dim or 0),
                "embed_device_used": cache_embed_device,
                "fallback_reason": cache_fallback_reason,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            cache_paths["metadata"].write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cache_paths["index"].write_text("in_memory_cosine\n", encoding="utf-8")
            duration_s = time.time() - build_started
            cache_paths["build_log"].write_text(
                f"cache_key={cache_key}\n"
                f"dataset_path={args.dataset}\n"
                f"samples={len(offsets)}\n"
                f"vectors={cache_embeddings.shape[0]}\n"
                f"dim={int(dim or 0)}\n"
                f"duration_s={duration_s:.2f}\n",
                encoding="utf-8",
            )
    
    predictions = {"answer": {}, "sp": {}}
    qa_rows: List[Tuple[str, str]] = []
    pred_raw_records: List[Dict[str, Any]] = []
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running Vanilla RAG on {len(data)} examples with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(
                process_example,
                item,
                llm,
                embedder,
                args,
                run_name=run_name,
                dataset_name=dataset_name,
                log_dir=artifacts_dir,
                embed_meta=embed_meta,
                embed_meta_lock=embed_meta_lock,
                cache_embeddings=cache_embeddings,
                cache_offsets=cache_offsets,
                cache_embed_device=cache_embed_device,
                cache_fallback_reason=cache_fallback_reason,
            )
            for item in data
        ]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                qid, question, ans, sp, contexts_used, context_tokens = fut.result()
                if not qid:
                    continue
                predictions["answer"][qid] = ans
                predictions["sp"][qid] = sp
                qa_rows.append((question, ans))
                pred_raw_records.append(
                    {
                        "id": str(qid),
                        "question": question,
                        "pred_raw": ans,
                        "contexts_used": contexts_used,
                        "context_tokens_used": context_tokens,
                        "context_budget_tokens": int(args.context_budget or 0) or None,
                    }
                )
            except Exception as e:
                logger.error(f"Error in worker: {e}")

    logger.info(f"Predictions generated for {len(predictions['answer'])} examples")

    out_path, qa_file = save_predictions_and_qa(
        work_dir,
        predictions,
        qa_rows,
        output_path=output_path,
        qa_path=qa_path,
    )
    write_jsonl(preds_dir / "pred_raw.jsonl", pred_raw_records)
    logger.info(f"Saved predictions to {out_path}")
    logger.info(f"Saved QA log to {qa_file}")
    devices = sorted(str(d) for d in embed_meta["used_devices"])
    used_device = "cpu" if "cpu" in devices else ("cuda" if "cuda" in devices else (devices[0] if devices else "cpu"))
    fallback_reason = embed_meta["fallback_reasons"][0] if embed_meta["fallback_reasons"] else None
    meta_payload = {
        "embed_model": args.embed_model,
        "embed_device_used": used_device,
        "fallback_reason": fallback_reason,
        "normalize": bool(args.embed_normalize),
        "dim": int(embed_meta["dim"] or 0),
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
