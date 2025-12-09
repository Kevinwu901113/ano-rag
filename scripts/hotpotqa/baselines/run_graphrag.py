from pathlib import Path
from typing import Any, Dict, List, Tuple
from loguru import logger
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from scripts.hotpotqa.baselines.baseline_utils import encode_passages

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

class MiniGraphRAG:
    def __init__(self, llm: LLMChatClient):
        self.llm = llm
        
    def build_and_query(self, passages: List[str], question: str) -> str:
        """
        1. Extract entities/relations from paragraphs.
        2. Build a mini text-based graph.
        3. Answer.
        """
        
        triples = []
        
        # Simplified: Concat all text, then extract graph (might be too long for extraction prompt output)
        # Better: Extract from each paragraph
        
        for text in passages:
            # Extract
            prompt = f"""Extract knowledge triples (Subject, Relation, Object) from the text. Return as JSON list.
Text: {text}
JSON:"""
            try:
                resp = self.llm.chat([{"role": "user", "content": prompt}])
                # Heuristic parsing
                try:
                    # Try to find JSON list
                    match = re.search(r'\[.*\]', resp.content, re.DOTALL)
                    if match:
                        extracted = json.loads(match.group(0))
                        if isinstance(extracted, list):
                            triples.extend(extracted)
                except:
                    pass
            except:
                pass
                
        # Format graph
        graph_desc = "\n".join([f"{t.get('Subject', '')} -- {t.get('Relation', '')} --> {t.get('Object', '')}" for t in triples if isinstance(t, dict)])
        
        # Also provide original text for grounding? GraphRAG usually uses community summaries.
        # Here we just use the graph + original text as fallback or combined.
        # Strict GraphRAG relies on the graph.
        
        full_text = "\n\n".join(passages)
        
        prompt = f"""Answer the question using the knowledge graph and text below.
        
Graph:
{graph_desc[:4000]} 

Original Text:
{full_text[:4000]}

Question: {question}
Answer:"""

        resp = self.llm.chat([{"role": "user", "content": prompt}])
        return resp.content

def process_example(item: Dict[str, Any], 
                    llm: LLMChatClient, 
                    args) -> Tuple[str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example using GraphRAG.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passages_from_context(context)
    
    # New graph_rag instance per thread
    graph_rag = MiniGraphRAG(llm)
    
    try:
        ans = graph_rag.build_and_query(passages, question)
        ans = ans.strip().replace("Answer:", "").strip()
        sp = []
        return qid, ans, sp
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run Mini GraphRAG on HotpotQA Distractor")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)
    
    predictions = {"answer": {}, "sp": {}}
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running GraphRAG on {len(data)} examples with {num_workers} workers...")
    
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
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
