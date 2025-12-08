import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from utils.embedding_utils import EmbeddingEncoder

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class MiniRaptor:
    def __init__(self, encoder: EmbeddingEncoder, llm: LLMChatClient):
        self.encoder = encoder
        self.llm = llm
        self.tree_nodes = [] # List of text
        
    def build_tree(self, context_data: List[List[Any]]):
        """
        Build a small RAPTOR tree from 10 paragraphs.
        1. Leaf layer: 10 paragraphs
        2. Cluster and summarize -> Higher level
        """
        if len(context_data) > 10:
             context_data = context_data[:10]
             
        leaf_texts = []
        for title, sentences in context_data:
            leaf_texts.append(f"{title}\n{''.join(sentences)}")
            
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
                summaries.append(summ)
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

def main():
    parser = argparse.ArgumentParser(description="Run Mini RAPTOR on HotpotQA Distractor")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--limit", type=int, default=0)
    
    args = parser.parse_args()

    llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)
    encoder = EmbeddingEncoder(provider="qwen3", model_name=args.emb_model, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    
    raptor = MiniRaptor(encoder, llm)
    
    data = load_dataset(args.dataset)
    if args.limit > 0:
        data = data[:args.limit]
        
    predictions = {"answer": {}, "sp": {}}
    
    for item in tqdm(data):
        qid = item["_id"]
        question = item["question"]
        
        raptor.build_tree(item["context"])
        context_nodes = raptor.retrieve(question)
        
        context_str = "\n\n".join(context_nodes)
        
        prompt = f"""Answer the question based on the context (which may include summaries).
        
{context_str}

Question: {question}
Answer:"""
        
        try:
            ans = llm.chat([{"role": "user", "content": prompt}])
            ans = ans.strip().replace("Answer:", "").strip()
            predictions["answer"][qid] = ans
            predictions["sp"][qid] = [] # Hard to map back to original paragraphs from summaries
        except Exception as e:
            logger.error(f"Error Q {qid}: {e}")
            predictions["answer"][qid] = "error"
            
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)

if __name__ == "__main__":
    main()
