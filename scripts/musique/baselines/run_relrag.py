import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from scripts.hotpotqa.baselines.baseline_utils import (
    build_passage_entries,
    detect_device,
    get_embedding_model,
)
from scripts.musique.baselines.musique_utils import load_dataset, save_musique_results_and_qa
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.output_protocol import build_final_instruction
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved
from config.config_loader import DEFAULT_EMBED_MODEL, DEFAULT_EMBED_DEVICE


def _get_qid(item: Dict[str, Any]) -> str:
    return str(item.get("id") or item.get("_id") or item.get("query_id") or "")

def _load_resume_state(
    output_path: Path,
    qa_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], set[str]]:
    results: List[Dict[str, Any]] = []
    qa_rows: List[Tuple[str, str]] = []
    pred_raw_records: List[Dict[str, Any]] = []
    pred_raw_path = preds_dir / "pred_raw.jsonl"
    if pred_raw_path.exists():
        try:
            for line in pred_raw_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                pred_raw_records.append(json.loads(line))
        except Exception as exc:
            logger.warning("Failed to load existing pred_raw.jsonl: {}", exc)
    completed: set[str] = set()

    if output_path.exists():
        try:
            for line in output_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                results.append(row)
                qid = str(row.get("id") or row.get("_id") or row.get("query_id") or "")
                if qid:
                    completed.add(qid)
        except Exception as exc:
            logger.warning("Failed to load existing results from {}: {}", output_path, exc)

    if qa_path.exists():
        try:
            for line in qa_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if "\t" in line:
                    q, a = line.split("\t", 1)
                else:
                    q, a = line, ""
                qa_rows.append((q, a))
        except Exception as exc:
            logger.warning("Failed to load existing QA log from {}: {}", qa_path, exc)

    return results, qa_rows, completed

class MiniRelRAG:
    def __init__(self, encoder: Callable[[List[str]], np.ndarray]):
        self.encoder = encoder
        self.entries: List[Dict[str, Any]] = []
        self.vectors = None

    def build(self, entries: List[Dict[str, Any]]):
        self.entries = entries
        texts = [e["text"] for e in entries]
        if not texts:
            self.vectors = None
            return
        self.vectors = self.encoder(texts)
        self.vectors = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.vectors is None:
            return []
        q_vec = self.encoder([query])
        q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10)
        sims = np.dot(self.vectors, q_vec.T).flatten()

        # Relation score: average similarity to others (centrality)
        rel_scores = np.dot(self.vectors, self.vectors.T).mean(axis=1)
        combined = 0.7 * sims + 0.3 * rel_scores

        indices = np.argsort(combined)[::-1][:k]
        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices):
            e = self.entries[idx]
            hits.append(
                {
                    "text": e["text"],
                    "score": float(combined[idx]),
                    "doc_id": e.get("doc_id"),
                    "sent_ids": e.get("sent_ids"),
                    "passage_id": e.get("passage_id"),
                    "rank": rank + 1,
                }
            )
        return hits


def process_example(
    item: Dict[str, Any],
    llm: LLMChatClient,
    encoder: Callable[[List[str]], np.ndarray],
    args,
    *,
    run_name: str,
    dataset_name: str,
    log_dir: Path,
) -> Tuple[str, str, str, List[str], List[Dict[str, Any]], int]:
    qid = str(item.get("id") or item.get("_id") or item.get("query_id") or "")
    question = str(item.get("question") or item.get("query") or "").strip()
    paragraphs = item.get("paragraphs") or item.get("contexts") or item.get("passages") or []
    entries = build_passage_entries(paragraphs, max_passages=args.max_context)

    relrag = MiniRelRAG(encoder)
    relrag.build(entries)
    hits = relrag.retrieve(question, k=args.topk)

    context_str, contexts_used, context_tokens = pack_contexts(
        hits, int(getattr(args, "context_budget", 0) or 0)
    )
    prompt = f"""Answer the question based on the selected paragraphs.
Keep the answer concise.
{build_final_instruction()}

{context_str}

Question: {question}
Answer:"""
    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = resp.content
        try:
            log_retrieval(
                sample_id=qid,
                dataset=dataset_name,
                run_name=run_name,
                retrieved=hits,
                topk=len(hits),
                final_context=contexts_used,
                final_context_tokens=context_tokens,
                context_budget_tokens=int(getattr(args, "context_budget", 0) or 0) or None,
                log_dir=log_dir,
            )
        except Exception as log_exc:
            logger.error(f"retrieval logging failed for {qid}: {log_exc}")

        pred_evidence = [str(h.get("passage_id")) for h in hits if h.get("passage_id")]
        seen = set()
        pred_evidence = [p for p in pred_evidence if not (p in seen or seen.add(p))]
        return qid, question, ans, pred_evidence, contexts_used, context_tokens
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        return qid, question, "error", [], contexts_used, context_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RelRAG baseline on MuSiQue")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--qa-path", default=None)
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--lm-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--lm-model", default="qwen3-30b-a3b")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for LLM decoding")
    parser.add_argument("--emb-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--emb-device", default=DEFAULT_EMBED_DEVICE)
    parser.add_argument("--emb-dtype", default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=20)
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs in workdir")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint every N samples (0 disables)")
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
    dataset_name = "musique"
    output_path = Path(args.output) if args.output else preds_dir / "musique_results.jsonl"
    qa_path = Path(args.qa_path) if args.qa_path else preds_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir}")
    setup_logging(str(work_dir / "run.log"))
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="musique",
            model=args.lm_model,
            endpoint=args.lm_endpoint,
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            context_budget=args.context_budget or None,
            topk=args.topk,
            decode={
                "temperature": 0.0,
                "top_p": None,
                "repetition_penalty": None,
                "max_tokens": args.max_new_tokens,
            },
            embedding={
                "model": args.emb_model,
                "device": args.emb_device,
                "batch_size": None,
                "max_length": None,
                "normalize": None,
                "dtype": args.emb_dtype,
            },
            budgets={
                "context_budget_tokens": args.context_budget or None,
                "topk": args.topk,
            },
            extra={"max_context": args.max_context, "emb_model": args.emb_model},
        ),
    )

    llm = LLMChatClient(
        endpoint=args.lm_endpoint,
        model=args.lm_model,
        temperature=0.0,
        max_tokens=args.max_new_tokens if args.max_new_tokens is not None else 8192,
    )
    device = args.emb_device or detect_device()
    encoder = get_embedding_model(args.emb_model, device, torch_dtype=args.emb_dtype)
    logger.info(f"Embedding model {args.emb_model} on {device} (dtype={args.emb_dtype or 'auto'})")

    results: List[Dict[str, Any]] = []
    qa_rows: List[Tuple[str, str]] = []
    completed: set[str] = set()
    if args.resume:
        results, qa_rows, completed = _load_resume_state(output_path, qa_path)
        if completed:
            logger.info("Resuming with {} existing predictions", len(completed))

    num_workers = max(1, args.num_workers)
    work_items: List[Dict[str, Any]] = []
    skipped = 0
    for item in data:
        qid = _get_qid(item)
        if qid and qid in completed:
            skipped += 1
            continue
        work_items.append(item)
    if skipped:
        logger.info("Skipping {} already completed examples", skipped)
    if not work_items:
        logger.info("No new items to process; keeping existing outputs.")
        save_musique_results_and_qa(
            work_dir,
            results,
            qa_rows,
            output_path=output_path,
            qa_path=qa_path,
        )
        return

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(
                process_example,
                item,
                llm,
                encoder,
                args,
                run_name=run_name,
                dataset_name=dataset_name,
                log_dir=artifacts_dir,
            )
            for item in work_items
        ]
        save_every = max(0, int(args.save_every))
        processed = 0
        for fut in tqdm(as_completed(futures), total=len(futures)):
            qid, question, ans, pred_evidence, contexts_used, context_tokens = fut.result()
            if not qid:
                continue
            results.append(
                {"id": qid, "predicted_answer": ans, "predicted_evidence": pred_evidence}
            )
            completed.add(str(qid))
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
            processed += 1
            if save_every and processed % save_every == 0:
                save_musique_results_and_qa(
                    work_dir,
                    results,
                    qa_rows,
                    output_path=output_path,
                    qa_path=qa_path,
                )
                write_jsonl(pred_raw_path, pred_raw_records)

    out_path, qa_file = save_musique_results_and_qa(
        work_dir,
        results,
        qa_rows,
        output_path=output_path,
        qa_path=qa_path,
    )
    write_jsonl(pred_raw_path, pred_raw_records)
    logger.info(f"Saved results to {out_path}")
    logger.info(f"Saved QA log to {qa_file}")


if __name__ == "__main__":
    main()
