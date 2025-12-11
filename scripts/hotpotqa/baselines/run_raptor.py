import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from sklearn.cluster import KMeans

from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from scripts.hotpotqa.baselines.baseline_utils import (
    build_passages_from_context,
    clean_hotpot_answer,
    detect_device,
    format_context,
    get_embedding_model,
    save_predictions_and_qa,
    select_workspace,
    truncate_text,
)

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class MiniRaptor:
    def __init__(self, encoder: Callable[[List[str]], np.ndarray], llm: LLMChatClient):
        self.encoder = encoder
        self.llm = llm
        self.tree_nodes = [] # List of text
        
    def build_tree(self, passages: List[str]):
        """
        Build a small RAPTOR tree from the paragraphs.
        1. Leaf layer: paragraphs
        2. Cluster and summarize -> Higher level
        """
        leaf_texts = passages
            
        if not leaf_texts:
            self.tree_nodes = []
            return
            
        # If too few nodes, just use leaves
        if len(leaf_texts) < 3:
            self.tree_nodes = leaf_texts
            return
            
        # Level 1: Cluster leaves
        vecs = self.encoder(leaf_texts)
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
        self.tree_nodes = leaf_texts + summaries
        
    def retrieve(self, query: str, k: int = 5):
        if not self.tree_nodes:
            return []
            
        vecs = self.encoder(self.tree_nodes)
        q_vec = self.encoder([query])
        
        scores = np.dot(vecs, q_vec.T).flatten()
        indices = np.argsort(scores)[::-1][:k]
        
        return [self.tree_nodes[i] for i in indices]

def process_example(item: Dict[str, Any], 
                    llm: LLMChatClient, 
                    encoder: Callable[[List[str]], np.ndarray],
                    args) -> Tuple[str, str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example using RAPTOR.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passages_from_context(context, max_passages=args.max_context)
    
    try:
        # New raptor instance per thread
        raptor = MiniRaptor(encoder, llm)
        
        raptor.build_tree(passages)
        context_nodes = raptor.retrieve(question)
        
        context_str = format_context(context_nodes)
        # Clip context to avoid exceeding small ctx-length models (approx 4 chars per token)
        max_chars = args.max_prompt_tokens * 4 if args.max_prompt_tokens and args.max_prompt_tokens > 0 else None
        context_str = truncate_text(context_str, max_chars)
        
        prompt = f"""Answer the question based on the context (which may include summaries).
    
{context_str}

Question: {question}
Answer:"""
    
        ans = llm.chat([{"role": "user", "content": prompt}])
        ans = clean_hotpot_answer(ans.content)
        # Hard to map back to original paragraphs from summaries
        sp = []
        return qid, question, ans, sp
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, question, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run Mini RAPTOR on HotpotQA Distractor")
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
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=3000,
        help="Approx upper bound for prompt tokens; context will be truncated to avoid ctx overflow.",
    )
    
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
        work_dir = select_workspace(Path(args.result_root), "hotpot_raptor", args.new)
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
    logger.info(f"Running RAPTOR on {len(data)} examples with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(process_example, item, llm, encoder, args)
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
