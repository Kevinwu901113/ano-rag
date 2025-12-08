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

class SelfReflectiveRetriever:
    def __init__(self, encoder: EmbeddingEncoder, llm: LLMChatClient):
        self.encoder = encoder
        self.llm = llm
        self.paragraphs = []
        self.titles = []
        self.vectors = None
        
    def build_index(self, context_data: List[List[Any]]):
        if len(context_data) > 10:
             context_data = context_data[:10]
        
        self.paragraphs = []
        self.titles = []
        texts = []
        for title, sentences in context_data:
            text = "".join(sentences)
            self.paragraphs.append(text)
            self.titles.append(title)
            texts.append(f"{title}\n{text}")
            
        if not texts:
            self.vectors = None
            return

        self.vectors = self.encoder.encode(texts)
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def retrieve_and_reflect(self, query: str, topk: int = 3) -> List[Tuple[str, str]]:
        if self.vectors is None:
            return []
            
        # 1. Initial Retrieval
        query_vec = self.encoder.encode([query])
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / (norm + 1e-10)
        scores = np.dot(self.vectors, query_vec.T).flatten()
        
        # Get top K candidates
        indices = np.argsort(scores)[::-1][:min(topk * 2, len(self.paragraphs))]
        candidates = [(self.titles[i], self.paragraphs[i]) for i in indices]
        
        # 2. Reflection / Re-ranking using LLM
        # Simple implementation: ask LLM to select relevant paragraphs from candidates
        cand_str = "\n\n".join([f"[{i}] Title: {c[0]}\nContent: {c[1]}" for i, c in enumerate(candidates)])
        
        prompt = f"""Identify the paragraphs that are most relevant to answering the question: "{query}"
Return only the indices (e.g., 0, 2) of the relevant paragraphs. If none are relevant, return nothing.

Candidates:
{cand_str}

Relevant Indices:"""

        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}])
            # Parse indices
            selected_indices = []
            import re
            nums = re.findall(r'\d+', resp)
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

def main():
    parser = argparse.ArgumentParser(description="Run Self-RAG Baseline on HotpotQA Distractor Setting")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--limit", type=int, default=0)
    
    args = parser.parse_args()

    llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)
    encoder = EmbeddingEncoder(provider="qwen3", model_name=args.emb_model, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    
    retriever = SelfReflectiveRetriever(encoder, llm)
    
    data = load_dataset(args.dataset)
    if args.limit > 0:
        data = data[:args.limit]
        
    predictions = {"answer": {}, "sp": {}}
    
    logger.info(f"Running Self-RAG on {len(data)} examples...")
    
    for item in tqdm(data):
        qid = item["_id"]
        question = item["question"]
        
        retriever.build_index(item["context"])
        hits = retriever.retrieve_and_reflect(question)
        
        context_str = "\n\n".join([f"Title: {t}\nContent: {p}" for t, p in hits])
        
        prompt = f"""Answer the question using the provided context.
        
{context_str}

Question: {question}
Answer:"""

        try:
            ans = llm.chat([{"role": "user", "content": prompt}])
            ans = ans.strip().replace("Answer:", "").strip()
            predictions["answer"][qid] = ans
            predictions["sp"][qid] = [[t, 0] for t, _ in hits]
        except Exception as e:
            logger.error(f"Error Q {qid}: {e}")
            predictions["answer"][qid] = "error"
            
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)

if __name__ == "__main__":
    main()
