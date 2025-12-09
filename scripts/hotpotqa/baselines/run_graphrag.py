import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import re
import sys

from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from scripts.hotpotqa.baselines.baseline_utils import (
    build_passages_from_context,
    clean_hotpot_answer,
    format_context,
    save_predictions_and_qa,
    select_workspace,
)

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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
        
        full_text = format_context(passages)
        
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
                    args) -> Tuple[str, str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example using GraphRAG.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passages_from_context(context, max_passages=args.max_context)
    
    # New graph_rag instance per thread
    graph_rag = MiniGraphRAG(llm)
    
    try:
        ans = graph_rag.build_and_query(passages, question)
        ans = clean_hotpot_answer(ans)
        sp = []
        return qid, question, ans, sp
    except Exception as e:
        logger.error(f"Error Q {qid}: {e}")
        return qid, question, "error", []

def main():
    parser = argparse.ArgumentParser(description="Run Mini GraphRAG on HotpotQA Distractor")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None, help="Output path for prediction json (default: work_dir/pred.json)")
    parser.add_argument("--qa-path", default=None, help="Optional QA log path (default: work_dir/qa.tsv)")
    parser.add_argument("--result-root", default="result/hotpotqa", help="Root directory for auto workspace creation")
    parser.add_argument("--work-dir", default=None, help="Workspace directory (default: auto under result-root)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=10, help="Max number of paragraphs from context to keep")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    # Workspace setup
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = select_workspace(Path(args.result_root), "hotpot_graphrag", args.new)
    output_path = Path(args.output) if args.output else work_dir / "pred.json"
    qa_path = Path(args.qa_path) if args.qa_path else work_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir}")

    llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)
    
    predictions = {"answer": {}, "sp": {}}
    qa_rows: List[Tuple[str, str]] = []
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running GraphRAG on {len(data)} examples with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(process_example, item, llm, args)
            for item in data
        ]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                qid, question, ans, sp = fut.result()
                if not qid:
                    continue
                predictions["answer"][qid] = ans
                predictions["sp"][qid] = sp
                qa_rows.append((question, ans))
            except Exception as e:
                logger.error(f"Error in worker: {e}")
            
    logger.info(f"Predictions generated for {len(predictions['answer'])} examples")

    out_path, qa_file = save_predictions_and_qa(
        work_dir,
        predictions,
        qa_rows,
        output_path=output_path,
        qa_path=qa_path,
    )
    logger.info(f"Saved predictions to {out_path}")
    logger.info(f"Saved QA log to {qa_file}")

if __name__ == "__main__":
    main()
