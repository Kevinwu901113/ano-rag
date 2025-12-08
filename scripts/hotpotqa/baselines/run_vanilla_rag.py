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

class InMemoryVanillaRetriever:
    def __init__(self, encoder: EmbeddingEncoder):
        self.encoder = encoder
        self.passages = []
        self.vectors = None
        
    def build_index_for_question(self, passages: List[str]):
        """
        Build a temporary index for the paragraphs provided in the distractor setting.
        """
        self.passages = passages
        if not self.passages:
            self.vectors = None
            return

        self.vectors = self.encoder.encode(self.passages)
        # Normalize for cosine similarity
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if self.vectors is None or len(self.passages) == 0:
            return []
            
        query_vec = self.encoder.encode([query])
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / (norm + 1e-10)
        
        scores = np.dot(self.vectors, query_vec.T).flatten()
        
        # Get top k
        indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in indices:
            results.append((self.passages[idx], float(scores[idx])))
        return results

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
    
    retriever = InMemoryVanillaRetriever(encoder)
    
    # 1. Build small index for this question
    retriever.build_index_for_question(passages)
    
    # 2. Retrieve
    hits = retriever.retrieve(question, k=args.topk)
    
    # 3. Format Context
    context_str = "\n\n".join([text for text, score in hits])
    
    # 4. Prompt
    prompt = f"""Answer the question based on the selected paragraphs.
Keep the answer concise.

{context_str}

Question: {question}
Answer:"""

    # 5. Generate
    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = resp.content
        ans = ans.strip()
        if ans.lower().startswith("answer:"):
            ans = ans[7:].strip()
        
        sp = [] # Not predicting supporting facts for now
        return qid, ans, sp
        
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        return qid, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run Vanilla RAG Baseline on HotpotQA Distractor Setting")
    parser.add_argument("--dataset", required=True, help="Path to hotpotqa distractor dev/test json")
    parser.add_argument("--output", required=True, help="Output path for prediction json")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1", help="LLM API endpoint")
    parser.add_argument("--lm-model", default="model-identifier", help="LLM model name")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B", help="Embedding model name or path")
    parser.add_argument("--topk", type=int, default=3, help="Number of paragraphs to retrieve from the 10 distractors")
    parser.add_argument("--limit", type=int, default=0, help="Test on N examples")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

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
    
    predictions = {"answer": {}, "sp": {}}
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running Vanilla RAG on {len(data)} examples with {num_workers} workers...")
    
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

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
