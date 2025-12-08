from pathlib import Path
from typing import Any, Dict, List, Tuple
from loguru import logger
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import os

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from scripts.hotpotqa.baselines.baseline_utils import get_embedding_model
from typing import Callable

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_passages_from_context(context: Any) -> List[str]:
    """
    支持两种格式的 context：
    1) dict: {"title": [...], "sentences": [...]}
    2) list: [[title, [sent1, ...]], ...]  (兼容官方原始格式)
    返回：每段 "Title: xxx\nContent: yyy" 的列表
    """
    passages: List[str] = []

    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences)
            passages.append(f"Title: {title}\nContent: {text}")
    else:
        for title, sentences in context:
            text = " ".join(sentences)
            passages.append(f"Title: {title}\nContent: {text}")

    return passages

class RelRAG:
    """
    RelRAG: Relation-aware retrieval.
    Builds a relation graph between paragraphs (based on entity overlap or similarity)
    and re-ranks them using PageRank or similar centrality on top of vector scores.
    """
    def __init__(self, encoder: Callable[[List[str]], np.ndarray], llm: LLMChatClient):
        self.encoder = encoder
        self.llm = llm
        
    def solve(self, passages: List[str], question: str) -> str:
        texts = passages
        
        if not texts:
            return "Insufficient evidence"
            
        vecs = self.encoder(texts)
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
        
        selected_texts = [texts[i] for i in indices]
        
        # 5. Answer
        context_str = "\n\n".join(selected_texts)
        prompt = f"""Answer the question based on the context.
        
{context_str}

Question: {question}
Answer:"""

        resp = self.llm.chat([{"role": "user", "content": prompt}])
        return resp.content

def process_example(item: Dict[str, Any], 
                    llm: LLMChatClient, 
                    encoder: Callable[[List[str]], np.ndarray],
                    args) -> Tuple[str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example using RelRAG.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passages_from_context(context)
    
    # New relrag instance per thread
    relrag = RelRAG(encoder, llm)
    
    try:
        ans = relrag.solve(passages, question)
        ans = ans.strip().replace("Answer:", "").strip()
        sp = []
        return qid, ans, sp
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run RelRAG on HotpotQA Distractor")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    encoder = get_embedding_model(args.emb_model, device)
    
    predictions = {"answer": {}, "sp": {}}
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running RelRAG on {len(data)} examples with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(process_example, item, llm, encoder, args)
            for item in data
        ]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                qid, ans, sp = fut.result()
                if not qid:
                    continue
                predictions["answer"][qid] = ans
                predictions["sp"][qid] = sp
            except Exception as e:
                logger.error(f"Error in worker: {e}")
            
    logger.info(f"Predictions generated for {len(predictions['answer'])} examples")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
