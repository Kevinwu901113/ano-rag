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
import re

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

class SelfReflectiveRetriever:
    def __init__(self, encoder: EmbeddingEncoder, llm: LLMChatClient):
        self.encoder = encoder
        self.llm = llm
        self.passages = []
        self.vectors = None
        
    def build_index(self, passages: List[str]):
        self.passages = passages
        if not self.passages:
            self.vectors = None
            return

        self.vectors = self.encoder.encode(self.passages)
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def retrieve_and_reflect(self, query: str, topk: int = 3) -> List[str]:
        if self.vectors is None:
            return []
            
        # 1. Initial Retrieval
        query_vec = self.encoder.encode([query])
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / (norm + 1e-10)
        scores = np.dot(self.vectors, query_vec.T).flatten()
        
        # Get top K candidates
        indices = np.argsort(scores)[::-1][:min(topk * 2, len(self.passages))]
        candidates = [self.passages[i] for i in indices]
        
        # 2. Reflection / Re-ranking using LLM
        # Simple implementation: ask LLM to select relevant paragraphs from candidates
        cand_str = "\n\n".join([f"[{i}] {c}" for i, c in enumerate(candidates)])
        
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
                    encoder: EmbeddingEncoder,
                    args) -> Tuple[str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passages_from_context(context)
    
    # Use a new retriever instance to avoid state conflicts, 
    # but share the encoder and llm client.
    retriever = SelfReflectiveRetriever(encoder, llm)
    
    retriever.build_index(passages)
    hits = retriever.retrieve_and_reflect(question)
    
    context_str = "\n\n".join(hits)
    
    prompt = f"""Answer the question using the provided context.
    
{context_str}

Question: {question}
Answer:"""

    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = resp.content
        ans = ans.strip().replace("Answer:", "").strip()
        sp = [] # Not predicting supporting facts for now
        return qid, ans, sp
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run Self-RAG Baseline on HotpotQA Distractor Setting")
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
    logger.info(f"Running Self-RAG on {len(data)} examples with {num_workers} workers...")
    
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
