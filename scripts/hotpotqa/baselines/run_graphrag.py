import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from scripts.hotpotqa.baselines.baseline_utils import (
    build_passage_entries,
    clean_hotpot_answer,
    format_context,
    save_predictions_and_qa,
    select_workspace,
)
from utils.retrieval_logger import log_retrieval

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class MiniGraphRAG:
    def __init__(self, llm_extract: LLMChatClient, llm_answer: LLMChatClient):
        self.llm_extract = llm_extract
        self.llm_answer = llm_answer
        
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
                resp = self.llm_extract.chat([{"role": "user", "content": prompt}])
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

        resp = self.llm_answer.chat([{"role": "user", "content": prompt}])
        return resp.content

def process_example(item: Dict[str, Any], 
                    llm_extract: Optional[LLMChatClient], 
                    llm_answer: Optional[LLMChatClient],
                    args,
                    *,
                    run_name: str,
                    dataset_name: str,
                    log_dir: Path) -> Tuple[str, str, str, List[List[Any]]]:
    """
    Process a single HotpotQA example using GraphRAG.
    """
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    context = item["context"]
    passages = build_passage_entries(context, max_passages=args.max_context)
    
    # GraphRAG baseline uses all provided paragraphs; log them regardless of generation success.
    try:
        log_retrieval(
            sample_id=qid,
            dataset=dataset_name,
            run_name=run_name,
            retrieved=[
                {
                    "rank": idx + 1,
                    "score": None,
                    "doc_id": entry.get("doc_id"),
                    "sent_ids": entry.get("sent_ids"),
                    "passage_id": entry.get("passage_id"),
                }
                for idx, entry in enumerate(passages)
            ],
            topk=len(passages),
            final_context=[
                {
                    "doc_id": entry.get("doc_id"),
                    "sent_ids": entry.get("sent_ids"),
                    "passage_id": entry.get("passage_id"),
                    "text": entry.get("text"),
                }
                for entry in passages
            ],
            log_dir=log_dir,
        )
    except Exception as log_exc:
        logger.error(f"retrieval logging failed for {qid}: {log_exc}")

    if bool(getattr(args, "retrieval_only", False)):
        return qid, question, "", []

    if llm_extract is None or llm_answer is None:
        raise RuntimeError("LLM clients are required unless --retrieval-only is set")

    # New graph_rag instance per thread
    graph_rag = MiniGraphRAG(llm_extract, llm_answer)
    
    try:
        ans = graph_rag.build_and_query([p["text"] for p in passages], question)
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
    parser.add_argument("--lm-timeout", type=int, default=120, help="LLM HTTP timeout in seconds")
    parser.add_argument("--lm-max-tokens", type=int, default=512, help="Max new tokens per LLM call")
    parser.add_argument("--extract-endpoint", default=None, help="LLM endpoint for triple extraction (defaults to lm-endpoint)")
    parser.add_argument("--extract-model", default=None, help="LLM model for triple extraction (defaults to lm-model)")
    parser.add_argument("--answer-endpoint", default=None, help="LLM endpoint for answering (defaults to lm-endpoint)")
    parser.add_argument("--answer-model", default=None, help="LLM model for answering (defaults to lm-model)")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM calls; only run retrieval and log retrieval.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=10, help="Max number of paragraphs from context to keep")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    
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
    run_name = work_dir.name
    dataset_name = "hotpotqa"
    output_path = Path(args.output) if args.output else work_dir / "pred.json"
    qa_path = Path(args.qa_path) if args.qa_path else work_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir}")

    extract_endpoint = args.extract_endpoint or args.lm_endpoint
    extract_model = args.extract_model or args.lm_model
    answer_endpoint = args.answer_endpoint or args.lm_endpoint
    answer_model = args.answer_model or args.lm_model

    llm_extract = None
    llm_answer = None
    if not args.retrieval_only:
        llm_extract = LLMChatClient(
            endpoint=extract_endpoint,
            model=extract_model,
            temperature=0.0,
            timeout=args.lm_timeout,
            max_tokens=args.lm_max_tokens,
        )
        llm_answer = LLMChatClient(
            endpoint=answer_endpoint,
            model=answer_model,
            temperature=0.0,
            timeout=args.lm_timeout,
            max_tokens=args.lm_max_tokens,
        )
    
    predictions = {"answer": {}, "sp": {}}
    qa_rows: List[Tuple[str, str]] = []
    
    num_workers = max(1, args.num_workers)
    logger.info(f"Running GraphRAG on {len(data)} examples with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(
                process_example,
                item,
                llm_extract,
                llm_answer,
                args,
                run_name=run_name,
                dataset_name=dataset_name,
                log_dir=work_dir,
            )
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
