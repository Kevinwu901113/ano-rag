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
from sklearn.cluster import KMeans

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from utils.embedding_utils import EmbeddingEncoder

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

class MiniRaptor:
    def __init__(self, encoder: EmbeddingEncoder, llm: LLMChatClient):
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
        vecs = self.encoder.encode(leaf_texts)
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
            
        vecs = self.encoder.encode(self.tree_nodes)
        q_vec = self.encoder.encode([query])
        
        scores = np.dot(vecs, q_vec.T).flatten()
        indices = np.argsort(scores)[::-1][:k]
        
        return [self.tree_nodes[i] for i in indices]

def process_example(item: Dict[str, Any], 
                    llm: LLMChatClient, 
                    encoder: EmbeddingEncoder,
                    args) -> Tuple[str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example using RAPTOR.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passages_from_context(context)
    
    try:
        # New raptor instance per thread
        raptor = MiniRaptor(encoder, llm)
        
        raptor.build_tree(passages)
        context_nodes = raptor.retrieve(question)
        
        context_str = "\n\n".join(context_nodes)
        
        prompt = f"""Answer the question based on the context (which may include summaries).
    
{context_str}

Question: {question}
Answer:"""
    
        ans = llm.chat([{"role": "user", "content": prompt}])
        ans = ans.content.strip().replace("Answer:", "").strip()
        # Hard to map back to original paragraphs from summaries
        sp = []
        return qid, ans, sp
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run Mini RAPTOR on HotpotQA Distractor")
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
    encoder = EmbeddingEncoder(provider="qwen3", model_name=args.emb_model, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    
    predictions = {"answer": {}, "sp": {}}
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running RAPTOR on {len(data)} examples with {num_workers} workers...")
    
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
