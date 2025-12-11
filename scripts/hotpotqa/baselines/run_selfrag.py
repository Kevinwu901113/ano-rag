import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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
    detect_device,
    format_context,
    get_embedding_model,
    save_predictions_and_qa,
    select_workspace,
)
from utils.retrieval_logger import log_retrieval

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class SelfReflectiveRetriever:
    def __init__(self, encoder: Callable[[List[str]], np.ndarray], llm: LLMChatClient):
        self.encoder = encoder
        self.llm = llm
        self.passages: List[str] = []
        self.entries: List[Dict[str, Any]] = []
        self.vectors = None
        self.last_hits: List[Dict[str, Any]] = []
        
    def build_index(self, passages: List[Dict[str, Any]]):
        self.entries = passages
        self.passages = [p["text"] for p in passages]
        if not self.passages:
            self.vectors = None
            return

        self.vectors = self.encoder(self.passages)
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def retrieve_and_reflect(self, query: str, topk: int = 3) -> List[Dict[str, Any]]:
        if self.vectors is None:
            return []
            
        # 1. Initial Retrieval
        query_vec = self.encoder([query])
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / (norm + 1e-10)
        scores = np.dot(self.vectors, query_vec.T).flatten()
        
        # Get top K candidates
        indices = np.argsort(scores)[::-1][:min(topk * 2, len(self.passages))]
        candidates = [self.entries[i] for i in indices]
        self.last_hits = [
            {
                "text": self.passages[i],
                "score": float(scores[i]),
                "doc_id": self.entries[i].get("doc_id"),
                "sent_ids": self.entries[i].get("sent_ids"),
                "passage_id": self.entries[i].get("passage_id"),
                "rank": rank + 1,
            }
            for rank, i in enumerate(indices)
        ]
        
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
                return candidates[:topk]
                
            return [candidates[i] for i in selected_indices[:topk]]
            
        except Exception:
            return candidates[:topk]

def process_example(item: Dict[str, Any], 
                    llm: LLMChatClient, 
                    encoder: Callable[[List[str]], np.ndarray],
                    args,
                    *,
                    run_name: str,
                    dataset_name: str,
                    log_dir: Path) -> Tuple[str, str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passage_entries(context, max_passages=args.max_context)
    
    # Use a new retriever instance to avoid state conflicts, 
    # but share the encoder and llm client.
    retriever = SelfReflectiveRetriever(encoder, llm)
    
    retriever.build_index(passages)
    hits = retriever.retrieve_and_reflect(question)
    
    context_str = format_context([hit["text"] for hit in hits])
    
    prompt = f"""Answer the question using the provided context.
    
{context_str}

Question: {question}
Answer:"""

    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = clean_hotpot_answer(resp.content)
        try:
            log_retrieval(
                sample_id=qid,
                dataset=dataset_name,
                run_name=run_name,
                retrieved=hits,
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
        sp = [] # Not predicting supporting facts for now
        return qid, question, ans, sp
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, question, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run Self-RAG Baseline on HotpotQA Distractor Setting")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None, help="Output path for prediction json (default: work_dir/pred.json)")
    parser.add_argument("--qa-path", default=None, help="Optional QA log path (default: work_dir/qa.tsv)")
    parser.add_argument("--result-root", default="result/hotpotqa", help="Root directory for auto workspace creation")
    parser.add_argument("--work-dir", default=None, help="Workspace directory (default: auto under result-root)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--emb-device", default=None, help="Force embedding device (e.g., cpu, cuda)")
    parser.add_argument("--emb-dtype", default=None, help="Embedding torch dtype (e.g., float16, bfloat16)")
    parser.add_argument("--limit", type=int, default=0)
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
        work_dir = select_workspace(Path(args.result_root), "hotpot_selfrag", args.new)
    run_name = work_dir.name
    dataset_name = "hotpotqa"
    output_path = Path(args.output) if args.output else work_dir / "pred.json"
    qa_path = Path(args.qa_path) if args.qa_path else work_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir}")

    llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)
    device = args.emb_device or detect_device()
    encoder = get_embedding_model(args.emb_model, device, torch_dtype=args.emb_dtype)
    logger.info(f"Embedding model {args.emb_model} on {device} (dtype={args.emb_dtype or 'auto'})")
    
    predictions = {"answer": {}, "sp": {}}
    qa_rows: List[Tuple[str, str]] = []
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running Self-RAG on {len(data)} examples with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(
                process_example,
                item,
                llm,
                encoder,
                args,
                run_name=run_name,
                dataset_name=dataset_name,
                log_dir=work_dir,
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

if __name__ == "__main__":
    main()
