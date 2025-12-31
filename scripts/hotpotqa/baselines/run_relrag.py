from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config_loader import DEFAULT_EMBED_MODEL, DEFAULT_EMBED_DEVICE
from scripts.hotpotqa.baselines.baseline_utils import (
    build_passage_entries,
    save_predictions_and_qa,
    TransformerEmbedder,
)
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.output_protocol import build_final_instruction
from utils.device import run_with_fallback
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_qid(item: Dict[str, Any]) -> str:
    return str(item.get("_id") or item.get("id") or "")

def _load_resume_state(
    output_path: Path,
    qa_path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str]], Set[str]]:
    predictions: Dict[str, Dict[str, Any]] = {"answer": {}, "sp": {}}
    qa_rows: List[Tuple[str, str]] = []
    completed: Set[str] = set()

    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                answers = payload.get("answer")
                sps = payload.get("sp")
                if isinstance(answers, dict):
                    predictions["answer"] = answers
                    completed.update(str(k) for k in answers.keys())
                if isinstance(sps, dict):
                    predictions["sp"] = sps
        except Exception as exc:
            logger.warning("Failed to load existing predictions from {}: {}", output_path, exc)

    if qa_path.exists():
        try:
            for line in qa_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if "\t" in line:
                    q, a = line.split("\t", 1)
                else:
                    q, a = line, ""
                qa_rows.append((q, a))
        except Exception as exc:
            logger.warning("Failed to load existing QA log from {}: {}", qa_path, exc)

    return predictions, qa_rows, completed

class RelRAG:
    """
    RelRAG: Relation-aware retrieval.
    Builds a relation graph between paragraphs (based on entity overlap or similarity)
    and re-ranks them using PageRank or similar centrality on top of vector scores.
    """
    def __init__(self, embedder: TransformerEmbedder, llm: Optional[Any], *, embed_device: str):
        self.embedder = embedder
        self.embed_device = embed_device
        self.llm = llm
        self.last_hits: List[Dict[str, Any]] = []
        self.last_context: List[Dict[str, Any]] = []
        self.last_context_tokens: int = 0
        self.embed_used_devices: Set[str] = set()
        self.embed_fallback_reasons: List[str] = []
        self.dim: Optional[int] = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs, used_device, fallback_reason = run_with_fallback(
            lambda device: self.embedder.encode(texts, device=device),
            prefer=self.embed_device,
        )
        self.embed_used_devices.add(str(used_device))
        if fallback_reason:
            self.embed_fallback_reasons.append(str(fallback_reason))
        if getattr(vecs, "size", 0) > 0 and self.dim is None:
            self.dim = int(vecs.shape[1])
        return vecs
        
    def solve(
        self,
        passages: List[Dict[str, Any]],
        question: str,
        *,
        topk: int = 3,
        retrieval_only: bool = False,
        context_budget: int = 0,
        hop: int = 1,
        use_scheduler: bool = True,
    ) -> str:
        texts = passages
        
        if not texts:
            return "Insufficient evidence"
            
        vecs = self._encode([p["text"] for p in passages])
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norm + 1e-10)
        
        # 2. Build Adjacency Matrix (Similarity > threshold)
        sim_matrix = np.dot(vecs, vecs.T)
        threshold = 0.7
        adj = (sim_matrix > threshold).astype(float)
        hop_steps = max(1, int(hop))
        if hop_steps > 1:
            adj_hop = adj.copy()
            for _ in range(1, hop_steps):
                adj_hop = (adj_hop @ adj) > 0
            adj = adj_hop.astype(float)
        
        # 3. Vector Search for Question
        q_vec = self._encode([question])
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        q_vec = q_vec / (q_norm + 1e-10)
        
        initial_scores = np.dot(vecs, q_vec.T).flatten()
        
        # 4. Spread Activation / PageRank-like re-ranking
        # Final Score = alpha * Initial + (1-alpha) * Neighbor_Avg
        if use_scheduler:
            alpha = 0.6
            neighbor_scores = np.dot(adj, initial_scores) / (np.sum(adj, axis=1) + 1e-10)
            final_scores = alpha * initial_scores + (1 - alpha) * neighbor_scores
        else:
            final_scores = initial_scores
        
        # Select Top K
        limit = max(1, int(topk))
        indices = np.argsort(final_scores)[::-1][:limit]
        
        selected = [passages[i] for i in indices]
        self.last_hits = [
            {
                "text": passages[i]["text"],
                "score": float(final_scores[i]),
                "doc_id": passages[i].get("doc_id"),
                "sent_ids": passages[i].get("sent_ids"),
                "passage_id": passages[i].get("passage_id"),
                "rank": rank + 1,
            }
            for rank, i in enumerate(indices)
        ]
        context_str, contexts_used, context_tokens = pack_contexts(selected, int(context_budget or 0))
        self.last_context = contexts_used
        self.last_context_tokens = context_tokens

        if retrieval_only:
            return ""
        
        # 5. Answer
        if self.llm is None:
            raise RuntimeError("LLM client is required unless --retrieval-only is set")
        prompt = f"""Answer the question based on the context.
{build_final_instruction()}
        
{context_str}

Question: {question}
Answer:"""

        resp = self.llm.chat([{"role": "user", "content": prompt}])
        return resp.content

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
    Process a single HotpotQA example using RelRAG.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passage_entries(context, max_passages=args.max_context)
    
    # New relrag instance per thread
    relrag = RelRAG(embedder, llm, embed_device=args.embed_device)
    
    try:
        retrieval_only = bool(getattr(args, "retrieval_only", False))
        ans = relrag.solve(
            passages,
            question,
            topk=int(getattr(args, "topk", 3)),
            retrieval_only=retrieval_only,
            context_budget=int(getattr(args, "context_budget", 0) or 0),
            hop=int(getattr(args, "hop", 1)),
            use_scheduler=not bool(getattr(args, "no_scheduler", False)),
        )
        if retrieval_only:
            ans = ""
        try:
            log_retrieval(
                sample_id=qid,
                dataset=dataset_name,
                run_name=run_name,
                retrieved=relrag.last_hits,
                topk=len(relrag.last_hits),
                final_context=relrag.last_context,
                final_context_tokens=relrag.last_context_tokens,
                context_budget_tokens=int(getattr(args, "context_budget", 0) or 0) or None,
                log_dir=log_dir,
            )
        except Exception as log_exc:
            logger.error(f"retrieval logging failed for {qid}: {log_exc}")
        with embed_meta_lock:
            for used in relrag.embed_used_devices:
                embed_meta["used_devices"].add(str(used))
            for reason in relrag.embed_fallback_reasons:
                embed_meta["fallback_reasons"].append(str(reason))
            if embed_meta.get("dim") is None and relrag.dim:
                embed_meta["dim"] = int(relrag.dim)
        sp = []
        return qid, question, ans, sp, relrag.last_context, relrag.last_context_tokens
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, question, "error", [], [], 0

def main():
    parser = argparse.ArgumentParser(description="Run RelRAG on HotpotQA Distractor")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None, help="Output path for prediction json (default: work_dir/pred.json)")
    parser.add_argument("--qa-path", default=None, help="Optional QA log path (default: work_dir/qa.tsv)")
    parser.add_argument("--result-root", default="result_relrag", help="Root directory for auto workspace creation")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Workspace directory (default: auto under result-root)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--lm-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--lm-model", default="qwen3-30b-a3b")
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
    parser.add_argument("--topk", type=int, default=3, help="Number of paragraphs to retrieve")
    parser.add_argument("--hop", type=int, default=1, help="Graph hop expansion for neighbor re-ranking")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable graph-aware re-ranking scheduler")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls; only run retrieval and log retrieval.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=10, help="Max number of paragraphs from context to keep")
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs in workdir")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint every N samples (0 disables)")
    
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
                "graph_hop": args.hop,
                "scheduler_enabled": not args.no_scheduler,
            },
        ),
    )

    llm = None
    if not args.retrieval_only:
        from structrag.llm_client import LLMChatClient

        llm = LLMChatClient(
            endpoint=args.lm_endpoint,
            model=args.lm_model,
            temperature=0.0,
            max_tokens=args.max_new_tokens if args.max_new_tokens is not None else 8192,
        )

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
    completed: Set[str] = set()
    if args.resume:
        predictions, qa_rows, completed = _load_resume_state(output_path, qa_path)
        if completed:
            logger.info("Resuming with {} existing predictions", len(completed))
    
    num_workers = max(1, args.num_workers)
    work_items: List[Dict[str, Any]] = []
    skipped = 0
    for item in data:
        qid = _get_qid(item)
        if qid and qid in completed:
            skipped += 1
            continue
        work_items.append(item)
    if skipped:
        logger.info("Skipping {} already completed examples", skipped)
    logger.info(f"Running RelRAG on {len(work_items)} examples with {num_workers} workers...")

    if not work_items:
        logger.info("No new items to process; keeping existing outputs.")
        save_predictions_and_qa(
            work_dir,
            predictions,
            qa_rows,
            output_path=output_path,
            qa_path=qa_path,
        )
        return
    
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
            for item in work_items
        ]

        save_every = max(0, int(args.save_every))
        processed = 0
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                qid, question, ans, sp, contexts_used, context_tokens = fut.result()
                if not qid:
                    continue
                predictions["answer"][qid] = ans
                predictions["sp"][qid] = sp
                completed.add(str(qid))
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
                processed += 1
                if save_every and processed % save_every == 0:
                    save_predictions_and_qa(
                        work_dir,
                        predictions,
                        qa_rows,
                        output_path=output_path,
                        qa_path=qa_path,
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
