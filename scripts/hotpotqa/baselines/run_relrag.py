import argparse
import json
import os
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

class RelRAG:
    """
    RelRAG: Relation-aware retrieval.
    Builds a relation graph between paragraphs (based on entity overlap or similarity)
    and re-ranks them using PageRank or similar centrality on top of vector scores.
    """
    def __init__(self, encoder: Callable[[List[str]], np.ndarray], llm: LLMChatClient):
        self.encoder = encoder
        self.llm = llm
        self.last_hits: List[Dict[str, Any]] = []
        
    def solve(self, passages: List[Dict[str, Any]], question: str) -> str:
        texts = passages
        
        if not texts:
            return "Insufficient evidence"
            
        vecs = self.encoder([p["text"] for p in passages])
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norm + 1e-10)
        
        # 2. Build Adjacency Matrix (Similarity > threshold)
        sim_matrix = np.dot(vecs, vecs.T)
        threshold = 0.7
        adj = (sim_matrix > threshold).astype(float)
        
        # 3. Vector Search for Question
        q_vec = self.encoder([question])
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        q_vec = q_vec / (q_norm + 1e-10)
        
        initial_scores = np.dot(vecs, q_vec.T).flatten()
        
        # 4. Spread Activation / PageRank-like re-ranking
        # Final Score = alpha * Initial + (1-alpha) * Neighbor_Avg
        alpha = 0.6
        neighbor_scores = np.dot(adj, initial_scores) / (np.sum(adj, axis=1) + 1e-10)
        final_scores = alpha * initial_scores + (1 - alpha) * neighbor_scores
        
        # Select Top K
        k = 3
        indices = np.argsort(final_scores)[::-1][:k]
        
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
        selected_texts = [item["text"] for item in selected]
        
        # 5. Answer
        context_str = format_context(selected_texts)
        prompt = f"""Answer the question based on the context.
        
{context_str}

Question: {question}
Answer:"""

        resp = self.llm.chat([{"role": "user", "content": prompt}])
        return resp.content

def process_example(item: Dict[str, Any], 
                    llm: LLMChatClient, 
                    encoder: Callable[[List[str]], np.ndarray],
                    args,
                    *,
                    run_name: str,
                    dataset_name: str,
                    log_dir: Path) -> Tuple[str, str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example using RelRAG.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passage_entries(context, max_passages=args.max_context)
    
    # New relrag instance per thread
    relrag = RelRAG(encoder, llm)
    
    try:
        ans = relrag.solve(passages, question)
        ans = clean_hotpot_answer(ans)
        try:
            log_retrieval(
                sample_id=qid,
                dataset=dataset_name,
                run_name=run_name,
                retrieved=relrag.last_hits,
                topk=len(relrag.last_hits),
                final_context=[
                    {
                        "doc_id": hit.get("doc_id"),
                        "sent_ids": hit.get("sent_ids"),
                        "passage_id": hit.get("passage_id"),
                        "text": hit.get("text"),
                    }
                    for hit in relrag.last_hits
                ],
                log_dir=log_dir,
            )
        except Exception as log_exc:
            logger.error(f"retrieval logging failed for {qid}: {log_exc}")
        sp = []
        return qid, question, ans, sp
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, question, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run RelRAG on HotpotQA Distractor")
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
        work_dir = select_workspace(Path(args.result_root), "hotpot_relrag", args.new)
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
    logger.info(f"Running RelRAG on {len(data)} examples with {num_workers} workers...")
    
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
