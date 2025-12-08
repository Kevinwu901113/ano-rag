import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from utils.embedding_utils import EmbeddingEncoder

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class RelRAG:
    """
    RelRAG: Relation-aware retrieval.
    Builds a relation graph between paragraphs (based on entity overlap or similarity)
    and re-ranks them using PageRank or similar centrality on top of vector scores.
    """
    def __init__(self, encoder: EmbeddingEncoder, llm: LLMChatClient):
        self.encoder = encoder
        self.llm = llm
        
    def solve(self, context_data: List[List[Any]], question: str) -> str:
        if len(context_data) > 10:
             context_data = context_data[:10]
             
        # 1. Encode paragraphs
        texts = [f"{t}\n{''.join(s)}" for t, s in context_data]
        titles = [t for t, s in context_data]
        
        if not texts:
            return "Insufficient evidence"
            
        vecs = self.encoder.encode(texts)
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norm + 1e-10)
        
        # 2. Build Adjacency Matrix (Similarity > threshold)
        sim_matrix = np.dot(vecs, vecs.T)
        threshold = 0.7
        adj = (sim_matrix > threshold).astype(float)
        
        # 3. Vector Search for Question
        q_vec = self.encoder.encode([question])
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

        return self.llm.chat([{"role": "user", "content": prompt}])

def main():
    parser = argparse.ArgumentParser(description="Run RelRAG on HotpotQA Distractor")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--limit", type=int, default=0)
    
    args = parser.parse_args()

    llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)
    encoder = EmbeddingEncoder(provider="qwen3", model_name=args.emb_model, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    
    relrag = RelRAG(encoder, llm)
    
    data = load_dataset(args.dataset)
    if args.limit > 0:
        data = data[:args.limit]
        
    predictions = {"answer": {}, "sp": {}}
    
    for item in tqdm(data):
        qid = item["_id"]
        question = item["question"]
        
        try:
            ans = relrag.solve(item["context"], question)
            ans = ans.strip().replace("Answer:", "").strip()
            predictions["answer"][qid] = ans
            predictions["sp"][qid] = []
        except Exception as e:
            logger.error(f"Error Q {qid}: {e}")
            predictions["answer"][qid] = "error"
            
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)

if __name__ == "__main__":
    main()
