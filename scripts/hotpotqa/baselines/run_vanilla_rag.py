import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from utils.embedding_utils import EmbeddingEncoder

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class InMemoryVanillaRetriever:
    def __init__(self, encoder: EmbeddingEncoder):
        self.encoder = encoder
        self.paragraphs = []
        self.vectors = None
        self.titles = []
        
    def build_index_for_question(self, context_data: List[List[Any]]):
        """
        Build a temporary index for the 10 paragraphs provided in the distractor setting.
        context_data: List of [title, sentences]
        """
        self.paragraphs = []
        self.titles = []
        texts = []
        
        # Verify limit
        if len(context_data) > 10:
             context_data = context_data[:10]
             
        for title, sentences in context_data:
            text = "".join(sentences)
            self.paragraphs.append(text)
            self.titles.append(title)
            texts.append(f"{title}\n{text}")
            
        if not texts:
            self.vectors = None
            return

        self.vectors = self.encoder.encode(texts)
        # Normalize for cosine similarity
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        if self.vectors is None or len(self.paragraphs) == 0:
            return []
            
        query_vec = self.encoder.encode([query])
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / (norm + 1e-10)
        
        scores = np.dot(self.vectors, query_vec.T).flatten()
        
        # Get top k
        indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in indices:
            results.append((self.titles[idx], self.paragraphs[idx], float(scores[idx])))
        return results

def main():
    parser = argparse.ArgumentParser(description="Run Vanilla RAG Baseline on HotpotQA Distractor Setting")
    parser.add_argument("--dataset", required=True, help="Path to hotpotqa distractor dev/test json")
    parser.add_argument("--output", required=True, help="Output path for prediction json")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1", help="LLM API endpoint")
    parser.add_argument("--lm-model", default="model-identifier", help="LLM model name")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B", help="Embedding model name or path")
    parser.add_argument("--topk", type=int, default=3, help="Number of paragraphs to retrieve from the 10 distractors")
    parser.add_argument("--limit", type=int, default=0, help="Test on N examples")
    
    args = parser.parse_args()

    # Initialize Clients
    llm = LLMChatClient(
        endpoint=args.lm_endpoint,
        model=args.lm_model,
        temperature=0.0
    )
    
    logger.info("Loading embedding model...")
    encoder = EmbeddingEncoder(
        provider="qwen3",
        model_name=args.emb_model,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    
    retriever = InMemoryVanillaRetriever(encoder)

    data = load_dataset(args.dataset)
    if args.limit > 0:
        data = data[:args.limit]

    predictions = {"answer": {}, "sp": {}}
    
    logger.info(f"Running Vanilla RAG on {len(data)} examples...")

    for item in tqdm(data):
        qid = item["_id"]
        question = item["question"]
        context_data = item["context"]
        
        # 1. Build small index for this question
        retriever.build_index_for_question(context_data)
        
        # 2. Retrieve
        hits = retriever.retrieve(question, k=args.topk)
        
        # 3. Format Context
        context_str = "\n\n".join([f"Title: {title}\nContent: {text}" for title, text, score in hits])
        
        # 4. Prompt
        prompt = f"""Answer the question based on the selected paragraphs.
Keep the answer concise.

{context_str}

Question: {question}
Answer:"""

        # 5. Generate
        try:
            ans = llm.chat([{"role": "user", "content": prompt}])
            ans = ans.strip()
            if ans.lower().startswith("answer:"):
                ans = ans[7:].strip()
            predictions["answer"][qid] = ans
            
            # Use top retrieved docs as supporting facts (title + sentence index 0 as placeholder)
            # HotpotQA requires [title, sent_id]
            sp = []
            for title, text, _ in hits:
                # We simply cite the first sentence or all sentences? 
                # HotpotQA gold SP is granular. Here we just point to the retrieved paragraphs.
                # Since we don't have sentence mapping easily here without re-splitting, 
                # we'll skip detailed SP or just cite sentence 0 for now.
                sp.append([title, 0]) 
            predictions["sp"][qid] = sp
            
        except Exception as e:
            logger.error(f"Error processing {qid}: {e}")
            predictions["answer"][qid] = "error"
            predictions["sp"][qid] = []

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
