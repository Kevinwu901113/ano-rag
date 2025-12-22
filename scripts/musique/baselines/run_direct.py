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
from scripts.hotpotqa.baselines.baseline_utils import build_passage_entries
from scripts.musique.baselines.musique_utils import load_dataset, save_musique_results_and_qa
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.output_protocol import build_final_instruction
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved


def process_example(
    item: Dict[str, Any],
    llm: LLMChatClient,
    args,
) -> Tuple[str, str, str, List[str], List[Dict[str, Any]], int]:
    qid = str(item.get("id") or item.get("_id") or item.get("query_id") or "")
    question = str(item.get("question") or item.get("query") or "").strip()
    paragraphs = item.get("paragraphs") or item.get("contexts") or item.get("passages") or []
    passage_entries = build_passage_entries(paragraphs, max_passages=args.max_context)

    context_text, contexts_used, context_tokens = pack_contexts(
        passage_entries, int(getattr(args, "context_budget", 0) or 0)
    )
    prompt = f"""Answer the question based on the following paragraphs.
Keep the answer concise.
{build_final_instruction()}

{context_text}

Question: {question}
Answer:"""
    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = resp.content
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        ans = "error"

    # Direct baseline does not select evidence explicitly.
    try:
        log_retrieval(
            sample_id=qid,
            dataset="musique",
            run_name=args.run_name,
            retrieved=[
                {
                    "rank": idx + 1,
                    "score": None,
                    "doc_id": entry.get("doc_id"),
                    "sent_ids": entry.get("sent_ids"),
                    "passage_id": entry.get("passage_id"),
                }
                for idx, entry in enumerate(passage_entries)
            ],
            topk=len(passage_entries),
            final_context=contexts_used,
            final_context_tokens=context_tokens,
            context_budget_tokens=int(getattr(args, "context_budget", 0) or 0) or None,
            log_dir=args.artifacts_dir,
        )
    except Exception as log_exc:
        logger.error(f"retrieval logging failed for {qid}: {log_exc}")
    return qid, question, ans, [], contexts_used, context_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Direct/Naive baseline on MuSiQue (provided paragraphs)")
    parser.add_argument("--dataset", required=True, help="Path to MuSiQue dev/test jsonl/json")
    parser.add_argument("--output", default=None, help="Output path for musique_results.jsonl")
    parser.add_argument("--qa-path", default=None, help="Optional QA tsv path")
    parser.add_argument("--result-root", default="result_relrag", help="Root directory for auto-created workspaces")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None, help="Workspace directory (default: auto under result-root)")
    parser.add_argument("--new", action="store_true", help="Force creating a new workspace under result-root")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1", help="LLM API endpoint")
    parser.add_argument("--lm-model", default="model-identifier", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    parser.add_argument("--limit", type=int, default=0, help="Test on N examples")
    parser.add_argument("--max-context", type=int, default=20, help="Max number of paragraphs to use")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="musique")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    run_name = work_dir.name
    output_path = Path(args.output) if args.output else preds_dir / "musique_results.jsonl"
    qa_path = Path(args.qa_path) if args.qa_path else preds_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir} (run={run_name})")
    setup_logging(str(work_dir / "run.log"))
    args.run_name = run_name
    args.artifacts_dir = artifacts_dir
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="musique",
            model=args.lm_model,
            endpoint=args.lm_endpoint,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            context_budget=args.context_budget or None,
            extra={"max_context": args.max_context},
        ),
    )

    llm = LLMChatClient(
        endpoint=args.lm_endpoint,
        model=args.lm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    results: List[Dict[str, Any]] = []
    qa_rows: List[Tuple[str, str]] = []
    pred_raw_records: List[Dict[str, Any]] = []

    num_workers = max(1, args.num_workers)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_example, item, llm, args) for item in data]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            qid, question, ans, pred_evidence, contexts_used, context_tokens = fut.result()
            if not qid:
                continue
            results.append(
                {"id": qid, "predicted_answer": ans, "predicted_evidence": pred_evidence}
            )
            qa_rows.append((question, ans))
            pred_raw_records.append(
                {
                    "id": str(qid),
                    "question": question,
                    "pred_raw": ans,
                    "contexts_used": contexts_used,
                    "context_tokens_used": context_tokens,
                    "context_budget_tokens": int(args.context_budget or 0) or None,
                }
            )

    out_path, qa_file = save_musique_results_and_qa(
        work_dir,
        results,
        qa_rows,
        output_path=output_path,
        qa_path=qa_path,
    )
    write_jsonl(preds_dir / "pred_raw.jsonl", pred_raw_records)
    logger.info(f"Saved results to {out_path}")
    logger.info(f"Saved QA log to {qa_file}")


if __name__ == "__main__":
    main()
