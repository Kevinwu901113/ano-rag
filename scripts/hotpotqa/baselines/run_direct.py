import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_context(paragraphs: List[str]) -> str:
    # Join paragraphs with clear delimiters
    return "\n\n".join([f"Paragraph {i+1}: {p}" for i, p in enumerate(paragraphs)])

def main():
    parser = argparse.ArgumentParser(description="Run Direct/Naive Baseline on HotpotQA Distractor Setting")
    parser.add_argument("--dataset", required=True, help="Path to hotpotqa distractor dev/test json")
    parser.add_argument("--output", required=True, help="Output path for prediction json")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1", help="LLM API endpoint")
    parser.add_argument("--lm-model", default="model-identifier", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Test on N examples")
    
    args = parser.parse_args()

    # Initialize LLM Client
    llm = LLMChatClient(
        endpoint=args.lm_endpoint,
        model=args.lm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    data = load_dataset(args.dataset)
    if args.limit > 0:
        data = data[:args.limit]

    predictions = {"answer": {}, "sp": {}}
    
    logger.info(f"Running Direct/Naive Baseline on {len(data)} examples...")

    for item in tqdm(data):
        qid = item["_id"]
        question = item["question"]
        context_data = item["context"]  # List of [title, sentences]
        
        # 1. Flatten context to 10 paragraphs max (HotpotQA distractor usually has 10)
        # Verify limit
        if len(context_data) > 10:
             logger.warning(f"QID {qid} has {len(context_data)} paragraphs. Truncating to 10.")
             context_data = context_data[:10]
             
        # Flatten sentences to paragraphs
        paragraphs = []
        for title, sentences in context_data:
            text = "".join(sentences)
            paragraphs.append(f"Title: {title}\nContent: {text}")
            
        context_text = format_context(paragraphs)
        
        # 2. Prompt
        prompt = f"""Answer the question based on the following paragraphs. 
Keep the answer concise.

{context_text}

Question: {question}
Answer:"""

        messages = [{"role": "user", "content": prompt}]
        
        # 3. Generate
        try:
            ans = llm.chat(messages)
            # Cleanup answer
            ans = ans.strip()
            if ans.lower().startswith("answer:"):
                ans = ans[7:].strip()
            predictions["answer"][qid] = ans
            # Direct baseline doesn't predict supporting facts usually, or we can just leave empty
            predictions["sp"][qid] = [] 
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
