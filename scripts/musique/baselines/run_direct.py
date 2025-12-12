import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    select_workspace,
)
from scripts.musique.baselines.musique_utils import load_dataset, save_musique_results_and_qa


def process_example(
    item: Dict[str, Any],
    llm: LLMChatClient,
    args,
) -> Tuple[str, str, str, List[str]]:
    qid = str(item.get("id") or item.get("_id") or item.get("query_id") or "")
    question = str(item.get("question") or item.get("query") or "").strip()
    paragraphs = item.get("paragraphs") or item.get("contexts") or item.get("passages") or []
    passage_entries = build_passage_entries(paragraphs, max_passages=args.max_context)

    context_text = format_context([p["text"] for p in passage_entries])
    prompt = f"""Answer the question based on the following paragraphs.
Keep the answer concise.

{context_text}

Question: {question}
Answer:"""
    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = clean_hotpot_answer(resp.content)
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        ans = "error"

    # Direct baseline does not select evidence explicitly.
    return qid, question, ans, []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Direct/Naive baseline on MuSiQue (provided paragraphs)")
    parser.add_argument("--dataset", required=True, help="Path to MuSiQue dev/test jsonl/json")
    parser.add_argument("--output", default=None, help="Output path for musique_results.jsonl")
    parser.add_argument("--qa-path", default=None, help="Optional QA tsv path")
    parser.add_argument("--result-root", default="result/musique", help="Root directory for auto-created workspaces")
    parser.add_argument("--work-dir", default=None, help="Workspace directory (default: auto under result-root)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace under result-root")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1", help="LLM API endpoint")
    parser.add_argument("--lm-model", default="model-identifier", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0, help="Test on N examples")
    parser.add_argument("--max-context", type=int, default=20, help="Max number of paragraphs to use")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = select_workspace(Path(args.result_root), "musique_direct", args.new)
    run_name = work_dir.name
    output_path = Path(args.output) if args.output else work_dir / "musique_results.jsonl"
    qa_path = Path(args.qa_path) if args.qa_path else work_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir} (run={run_name})")

    llm = LLMChatClient(
        endpoint=args.lm_endpoint,
        model=args.lm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    results: List[Dict[str, Any]] = []
    qa_rows: List[Tuple[str, str]] = []

    num_workers = max(1, args.num_workers)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_example, item, llm, args) for item in data]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            qid, question, ans, pred_evidence = fut.result()
            if not qid:
                continue
            results.append(
                {"id": qid, "predicted_answer": ans, "predicted_evidence": pred_evidence}
            )
            qa_rows.append((question, ans))

    out_path, qa_file = save_musique_results_and_qa(
        work_dir,
        results,
        qa_rows,
        output_path=output_path,
        qa_path=qa_path,
    )
    logger.info(f"Saved results to {out_path}")
    logger.info(f"Saved QA log to {qa_file}")


if __name__ == "__main__":
    main()

