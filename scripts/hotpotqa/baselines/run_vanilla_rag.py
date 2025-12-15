import argparse
import json
import os
import sys
import threading
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
    clean_hotpot_answer,
    format_context,
    save_predictions_and_qa,
    select_workspace,
    TransformerEmbedder,
)
from utils.device import run_with_fallback
from utils.retrieval_logger import log_retrieval

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

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.passages) == 0:
            return []
            
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
                    embedder: TransformerEmbedder,
                    args,
                    *,
                    run_name: str,
                    dataset_name: str,
                    log_dir: Path,
                    embed_meta: Dict[str, Any],
                    embed_meta_lock: threading.Lock) -> Tuple[str, str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passage_entries = build_passage_entries(context, max_passages=args.max_context)
    
    retriever = InMemoryVanillaRetriever(embedder, embed_device=args.embed_device)
    
    # 1. Build small index for this question
    retriever.build_index_for_question(passage_entries)
    
    # 2. Retrieve
    hits = retriever.retrieve(question, k=args.topk)
    with embed_meta_lock:
        for used in (retriever.embed_device_used_build, retriever.embed_device_used_query):
            if used:
                embed_meta["used_devices"].add(str(used))
        for reason in (retriever.fallback_reason_build, retriever.fallback_reason_query):
            if reason:
                embed_meta["fallback_reasons"].append(str(reason))
        if embed_meta.get("dim") is None and retriever.dim:
            embed_meta["dim"] = int(retriever.dim)

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
            final_context=[
                {
                    "doc_id": hit.get("doc_id"),
                    "sent_ids": hit.get("sent_ids"),
                    "passage_id": hit.get("passage_id"),
                    "text": hit.get("text"),
                }
                for hit in hits
            ],
            log_dir=log_dir,
        )
    except Exception as log_exc:
        logger.error(f"retrieval logging failed for {qid}: {log_exc}")

    if getattr(args, "retrieval_only", False):
        sp = []
        return qid, question, "", sp
    
    # 3. Format Context
    context_str = format_context([hit["text"] for hit in hits])
    
    # 4. Prompt
    prompt = f"""Answer the question based on the selected paragraphs.
Keep the answer concise.

{context_str}

Question: {question}
Answer:"""

    # 5. Generate
    try:
        if llm is None:
            raise RuntimeError("LLM client is required unless --retrieval-only is set")
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = clean_hotpot_answer(resp.content)
        
        sp = [] # Not predicting supporting facts for now
        return qid, question, ans, sp
        
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        return qid, question, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run Vanilla RAG Baseline on HotpotQA Distractor Setting")
    parser.add_argument("--dataset", required=True, help="Path to hotpotqa distractor dev/test json")
    parser.add_argument("--output", default=None, help="Output path for prediction json (default: work_dir/pred.json)")
    parser.add_argument("--qa-path", default=None, help="Optional QA log path (default: work_dir/qa.tsv)")
    parser.add_argument("--result-root", default="result/hotpotqa", help="Root directory for auto workspace creation")
    parser.add_argument("--work-dir", default=None, help="Workspace directory (default: auto under result-root)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1", help="LLM API endpoint")
    parser.add_argument("--lm-model", default="model-identifier", help="LLM model name")
    parser.add_argument(
        "--embed-model",
        "--emb-model",
        dest="embed_model",
        default="Qwen/Qwen3-Embedding-8B",
        help="Embedding model name or path",
    )
    parser.add_argument(
        "--embed-device",
        "--emb-device",
        dest="embed_device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
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
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls; only run retrieval and log retrieval.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Test on N examples")
    parser.add_argument("--max-context", type=int, default=10, help="Max number of paragraphs from context to keep")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    # Workspace setup
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = select_workspace(Path(args.result_root), "hotpot_vanilla_rag", args.new)
    run_name = work_dir.name
    dataset_name = "hotpotqa"
    output_path = Path(args.output) if args.output else work_dir / "pred.json"
    qa_path = Path(args.qa_path) if args.qa_path else work_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir}")

    # Initialize Clients
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
                    log_dir=work_dir,
                    embed_meta=embed_meta,
                    embed_meta_lock=embed_meta_lock,
                )
            for item in data
        ]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                qid, question, ans, sp = fut.result()
                if not qid:
                    continue
                predictions["answer"][qid] = ans
                predictions["sp"][qid] = sp
                qa_rows.append((question, ans))
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
    (work_dir / "meta.json").write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
