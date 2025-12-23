import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hotpotqa.baselines.baseline_utils import (
    build_passage_entries,
    save_predictions_and_qa,
    TransformerEmbedder,
)
from utils.context_budget import pack_contexts
from utils.device import run_with_fallback
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.output_protocol import build_final_instruction
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class SelfReflectiveRetriever:
    def __init__(self, embedder: TransformerEmbedder, llm: Optional[Any], *, embed_device: str):
        self.embedder = embedder
        self.embed_device = embed_device
        self.llm = llm
        self.passages: List[str] = []
        self.entries: List[Dict[str, Any]] = []
        self.vectors = None
        self.last_hits: List[Dict[str, Any]] = []
        self.dim: Optional[int] = None
        self.embed_device_used_build: Optional[str] = None
        self.embed_device_used_query: Optional[str] = None
        self.fallback_reason_build: Optional[str] = None
        self.fallback_reason_query: Optional[str] = None
        
    def build_index(self, passages: List[Dict[str, Any]]):
        self.entries = passages
        self.passages = [p["text"] for p in passages]
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
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def retrieve_and_reflect(
        self,
        query: str,
        topk: int = 3,
        *,
        use_llm_reflection: bool = True,
    ) -> List[Dict[str, Any]]:
        if self.vectors is None:
            return []
            
        # 1. Initial Retrieval
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
        
        # Get top K candidates
        limit = max(1, int(topk))
        indices = np.argsort(scores)[::-1][:min(limit * 2, len(self.passages))]
        candidates = [self.entries[i] for i in indices]
        candidate_hits: List[Dict[str, Any]] = []
        for rank, i in enumerate(indices):
            candidate_hits.append(
                {
                    "text": self.passages[i],
                    "score": float(scores[i]),
                    "doc_id": self.entries[i].get("doc_id"),
                    "sent_ids": self.entries[i].get("sent_ids"),
                    "passage_id": self.entries[i].get("passage_id"),
                    "rank": rank + 1,
                }
            )

        if (not use_llm_reflection) or self.llm is None:
            self.last_hits = candidate_hits[:limit]
            return self.last_hits
        
        # 2. Reflection / Re-ranking using LLM
        # Simple implementation: ask LLM to select relevant paragraphs from candidates
        cand_str = "\n\n".join([f"[{i}] {c['text']}" for i, c in enumerate(candidates)])
        
        prompt = f"""Identify the paragraphs that are most relevant to answering the question: "{query}"
Return only the indices (e.g., 0, 2) of the relevant paragraphs. If none are relevant, return nothing.

Candidates:
{cand_str}

Relevant Indices:"""

        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}])
            # Parse indices
            selected_indices = []
            nums = re.findall(r'\d+', resp.content)
            for n in nums:
                idx = int(n)
                if 0 <= idx < len(candidates):
                    selected_indices.append(idx)
            
            # If nothing selected or parse failed, fall back to top-k vector search
            if not selected_indices:
                self.last_hits = candidate_hits[:limit]
                return self.last_hits
                
            # Deduplicate while preserving order
            seen = set()
            uniq: List[int] = []
            for idx in selected_indices:
                if idx not in seen:
                    uniq.append(idx)
                    seen.add(idx)
            selected_hits = [candidate_hits[i] for i in uniq[:limit]]
            # Fix rank to reflect post-reflection order
            self.last_hits = [{**hit, "rank": ridx + 1} for ridx, hit in enumerate(selected_hits)]
            return self.last_hits
            
        except Exception:
            self.last_hits = candidate_hits[:limit]
            return self.last_hits

def process_example(item: Dict[str, Any], 
                    llm: Optional[Any], 
                    embedder: TransformerEmbedder,
                    args,
                    *,
                    run_name: str,
                    dataset_name: str,
                    log_dir: Path,
                    embed_meta: Dict[str, Any],
                    embed_meta_lock: threading.Lock) -> Tuple[str, str, str, List[List[Any]], List[Dict[str, Any]], int]:
    """
    Process a single HotpotQA example.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passage_entries(context, max_passages=args.max_context)
    
    # Use a new retriever instance to avoid state conflicts, 
    # but share the encoder and llm client.
    retriever = SelfReflectiveRetriever(embedder, llm, embed_device=args.embed_device)
    
    retriever.build_index(passages)
    retrieval_only = bool(getattr(args, "retrieval_only", False))
    hits = retriever.retrieve_and_reflect(question, topk=int(getattr(args, "topk", 3)), use_llm_reflection=not retrieval_only)
    with embed_meta_lock:
        for used in (retriever.embed_device_used_build, retriever.embed_device_used_query):
            if used:
                embed_meta["used_devices"].add(str(used))
        for reason in (retriever.fallback_reason_build, retriever.fallback_reason_query):
            if reason:
                embed_meta["fallback_reasons"].append(str(reason))
        if embed_meta.get("dim") is None and retriever.dim:
            embed_meta["dim"] = int(retriever.dim)
    
    context_str, contexts_used, context_tokens = pack_contexts(
        hits, int(getattr(args, "context_budget", 0) or 0)
    )

    try:
        log_retrieval(
            sample_id=qid,
            dataset=dataset_name,
            run_name=run_name,
            retrieved=hits,
            topk=len(hits),
            final_context=contexts_used,
            final_context_tokens=context_tokens,
            context_budget_tokens=int(getattr(args, "context_budget", 0) or 0) or None,
            log_dir=log_dir,
        )
    except Exception as log_exc:
        logger.error(f"retrieval logging failed for {qid}: {log_exc}")

    if retrieval_only:
        sp = []
        return qid, question, "", sp, contexts_used, context_tokens
    
    prompt = f"""Answer the question using the provided context.
Keep the answer concise.
{build_final_instruction()}

{context_str}

Question: {question}
Answer:"""

    try:
        if llm is None:
            raise RuntimeError("LLM client is required unless --retrieval-only is set")
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = resp.content
        sp = [] # Not predicting supporting facts for now
        return qid, question, ans, sp, contexts_used, context_tokens
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, question, "error", [], contexts_used, context_tokens

def main():
    parser = argparse.ArgumentParser(description="Run Self-RAG Baseline on HotpotQA Distractor Setting")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None, help="Output path for prediction json (default: work_dir/pred.json)")
    parser.add_argument("--qa-path", default=None, help="Optional QA log path (default: work_dir/qa.tsv)")
    parser.add_argument("--result-root", default="result_relrag", help="Root directory for auto workspace creation")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Workspace directory (default: auto under result-root)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--lm-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--lm-model", default="qwen3-30b-a3b")
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
    parser.add_argument("--topk", type=int, default=3, help="Number of paragraphs to retrieve")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls; only run retrieval and log retrieval.jsonl")
    parser.add_argument("--limit", type=int, default=0)
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
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="hotpotqa",
            model=args.lm_model,
            endpoint=args.lm_endpoint,
            temperature=0.0,
            max_tokens=None,
            context_budget=args.context_budget or None,
            topk=args.topk,
            extra={"max_context": args.max_context, "embed_model": args.embed_model},
        ),
    )

    llm = None
    if not args.retrieval_only:
        from structrag.llm_client import LLMChatClient

        llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)

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
    embed_meta: Dict[str, Any] = {"used_devices": set(), "fallback_reasons": [], "dim": None}
    embed_meta_lock = threading.Lock()
    
    predictions = {"answer": {}, "sp": {}}
    qa_rows: List[Tuple[str, str]] = []
    pred_raw_records: List[Dict[str, Any]] = []
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running Self-RAG on {len(data)} examples with {num_workers} workers...")
    
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
