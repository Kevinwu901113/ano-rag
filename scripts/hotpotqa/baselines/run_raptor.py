from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from sklearn.cluster import KMeans

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

class MiniRaptor:
    def __init__(self, embedder: TransformerEmbedder, llm: Optional[Any], *, embed_device: str):
        self.embedder = embedder
        self.embed_device = embed_device
        self.llm = llm
        self.tree_nodes: List[Dict[str, Any]] = []  # Nodes with text + metadata
        self.leaf_entries: List[Dict[str, Any]] = []
        self.last_hits: List[Dict[str, Any]] = []
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
        
    def build_tree(self, passages: List[Dict[str, Any]], *, use_summaries: bool = True):
        """
        Build a small RAPTOR tree from the paragraphs.
        1. Leaf layer: paragraphs
        2. Cluster and summarize -> Higher level
        """
        self.leaf_entries = passages
        leaf_texts = [p["text"] for p in passages]
            
        if not leaf_texts:
            self.tree_nodes = []
            return

        leaf_nodes = [
            {
                "text": text,
                "doc_id": entry.get("doc_id"),
                "sent_ids": entry.get("sent_ids"),
                "passage_id": entry.get("passage_id"),
            }
            for text, entry in zip(leaf_texts, self.leaf_entries)
        ]

        if (not use_summaries) or self.llm is None:
            self.tree_nodes = leaf_nodes
            return
            
        # If too few nodes, just use leaves
        if len(leaf_texts) < 3:
            self.tree_nodes = leaf_nodes
            return
            
        # Level 1: Cluster leaves
        vecs = self._encode(leaf_texts)
        # Dynamic cluster count
        n_clusters = max(1, len(leaf_texts) // 3)
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        labels = kmeans.fit_predict(vecs)
        
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters: clusters[label] = []
            clusters[label].append(leaf_texts[i])
            
        summaries = []
        for label, texts in clusters.items():
            # Summarize cluster
            joint_text = "\n\n".join(texts)
            prompt = f"Summarize the following texts into a single concise paragraph containing key information:\n\n{joint_text}\n\nSummary:"
            try:
                summ = self.llm.chat([{"role": "user", "content": prompt}])
                summaries.append(summ.content)
            except Exception:
                pass
                
        # Tree = Leaves + Summaries
        summary_nodes = [
            {
                "text": summary,
                "doc_id": None,
                "sent_ids": None,
                "passage_id": f"summary_{idx}",
            }
            for idx, summary in enumerate(summaries)
        ]
        self.tree_nodes = leaf_nodes + summary_nodes
        
    def retrieve(self, query: str, k: int = 5):
        if not self.tree_nodes:
            return []
            
        texts = [node["text"] for node in self.tree_nodes]
        vecs = self._encode(texts)
        q_vec = self._encode([query])
        
        scores = np.dot(vecs, q_vec.T).flatten()
        indices = np.argsort(scores)[::-1][:k]
        
        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices):
            node = self.tree_nodes[idx]
            hits.append(
                {
                    **node,
                    "score": float(scores[idx]),
                    "rank": rank + 1,
                }
            )
        self.last_hits = hits
        return hits

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
    Process a single HotpotQA example using RAPTOR.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passage_entries(context, max_passages=args.max_context)
    
    try:
        retrieval_only = bool(getattr(args, "retrieval_only", False))
        # New raptor instance per thread
        raptor = MiniRaptor(embedder, llm, embed_device=args.embed_device)
        
        raptor.build_tree(passages, use_summaries=not retrieval_only)
        hits = raptor.retrieve(question, k=int(getattr(args, "topk", 5)))
        with embed_meta_lock:
            for used in raptor.embed_used_devices:
                embed_meta["used_devices"].add(str(used))
            for reason in raptor.embed_fallback_reasons:
                embed_meta["fallback_reasons"].append(str(reason))
            if embed_meta.get("dim") is None and raptor.dim:
                embed_meta["dim"] = int(raptor.dim)
        
        budget_tokens = int(getattr(args, "context_budget", 0) or 0)
        if budget_tokens <= 0 and getattr(args, "max_prompt_tokens", 0):
            budget_tokens = int(args.max_prompt_tokens)
        context_str, contexts_used, context_tokens = pack_contexts(hits, budget_tokens)
        try:
            log_retrieval(
                sample_id=qid,
                dataset=dataset_name,
                run_name=run_name,
                retrieved=hits,
                topk=len(hits),
                final_context=contexts_used,
                final_context_tokens=context_tokens,
                context_budget_tokens=budget_tokens or None,
                log_dir=log_dir,
            )
        except Exception as log_exc:
            logger.error(f"retrieval logging failed for {qid}: {log_exc}")

        if retrieval_only:
            return qid, question, "", [], contexts_used, context_tokens
        
        prompt = f"""Answer the question based on the context (which may include summaries).
{build_final_instruction()}

{context_str}

Question: {question}
Answer:"""
	    
        if llm is None:
            raise RuntimeError("LLM client is required unless --retrieval-only is set")
        ans = llm.chat([{"role": "user", "content": prompt}])
        ans = ans.content
        sp = []
        return qid, question, ans, sp, contexts_used, context_tokens
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, question, "error", [], [], 0

def main():
    parser = argparse.ArgumentParser(description="Run Mini RAPTOR on HotpotQA Distractor")
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
    parser.add_argument("--topk", type=int, default=5, help="Number of nodes to retrieve from the RAPTOR tree")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls; only run retrieval and log retrieval.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=10, help="Max number of paragraphs from context to keep")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=3000,
        help="Legacy prompt token cap (used only when --context-budget is 0).",
    )
    
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
            extra={"max_context": args.max_context, "embed_model": args.embed_model},
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
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running RAPTOR on {len(data)} examples with {num_workers} workers...")
    
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
