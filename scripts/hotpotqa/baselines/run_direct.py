from pathlib import Path
from typing import Any, Dict, List, Tuple
from loguru import logger
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient

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

def process_example(
    item: Dict[str, Any],
    llm: LLMChatClient,
    args,
) -> Tuple[str, str, List[List[Any]]]:
    """
    返回 (qid, answer, sp)
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passages_from_context(context)
    
    # 1. Format context
    context_text = "\n\n".join(passages)
    
    # 2. Prompt
    prompt = f"""Answer the question based on the following paragraphs. 
Keep the answer concise.

{context_text}

Question: {question}
Answer:"""

    messages = [{"role": "user", "content": prompt}]
    
    # 3. Generate
    try:
        resp = llm.chat(messages)
        ans = resp.content
        # Cleanup answer
        ans = ans.strip()
        if ans.lower().startswith("answer:"):
            ans = ans[7:].strip()
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        ans = "error"

    sp: List[List[Any]] = []  # Direct baseline doesn't predict supporting facts
    return qid, ans, sp

def main():
    parser = argparse.ArgumentParser(description="Run Direct/Naive Baseline on HotpotQA Distractor Setting")
    parser.add_argument("--dataset", required=True, help="Path to hotpotqa distractor dev/test json")
    parser.add_argument("--output", required=True, help="Output path for prediction json")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1", help="LLM API endpoint")
    parser.add_argument("--lm-model", default="model-identifier", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Test on N examples")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并发 worker 数，使用线程并发调用 LM，默认 1（串行）",
    )
    
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    # Initialize LLM Client
    llm = LLMChatClient(
        endpoint=args.lm_endpoint,
        model=args.lm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    predictions = {"answer": {}, "sp": {}}
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(process_example, item, llm, args)
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
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
