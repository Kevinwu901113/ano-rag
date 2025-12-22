import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from utils.bm25 import BM25Index, rrf_fuse
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.output_protocol import build_final_instruction
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved


class InMemoryVanillaRetriever:
    def __init__(self, encoder: Callable[[List[str]], np.ndarray], emb_dtype: Optional[str] = None):
        self.encoder = encoder
        self.passages: List[str] = []
        self.entries: List[Dict[str, Any]] = []
        self.vectors = None
        self.last_hits: List[Dict[str, Any]] = []
        self.emb_dtype = emb_dtype
        self.last_scores: Optional[np.ndarray] = None

    def build_index_for_question(self, entries: List[Dict[str, Any]]):
        self.entries = entries
        self.passages = [entry["text"] for entry in entries]
        if not self.passages:
            self.vectors = None
            return
        self.vectors = self.encoder(self.passages, torch_dtype=self.emb_dtype)
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def score(self, query: str) -> np.ndarray:
        if self.vectors is None or len(self.passages) == 0:
            self.last_scores = None
            return np.array([])
        query_vec = self.encoder([query])
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / (norm + 1e-10)
        scores = np.dot(self.vectors, query_vec.T).flatten()
        self.last_scores = scores
        return scores

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.passages) == 0:
            return []
        scores = self.score(query)
        indices = np.argsort(scores)[::-1][:k]
        results: List[Dict[str, Any]] = []
        for idx in indices:
            entry = self.entries[idx]
            results.append(
                {
                    "text": self.passages[idx],
                    "score": float(scores[idx]),
                    "doc_id": entry.get("doc_id"),
                    "sent_ids": entry.get("sent_ids"),
                    "passage_id": entry.get("passage_id"),
                }
            )
        self.last_hits = results
        return results


def process_example(
    item: Dict[str, Any],
    llm: LLMChatClient,
    encoder: Optional[Callable[[List[str]], np.ndarray]],
    args,
    *,
    run_name: str,
    dataset_name: str,
    log_dir: Path,
) -> Tuple[str, str, str, List[str], List[Dict[str, Any]], int]:
    qid = str(item.get("id") or item.get("_id") or item.get("query_id") or "")
    question = str(item.get("question") or item.get("query") or "").strip()
    paragraphs = item.get("paragraphs") or item.get("contexts") or item.get("passages") or []
    passage_entries = build_passage_entries(paragraphs, max_passages=args.max_context)

    hits: List[Dict[str, Any]] = []
    if args.retriever in ("dense", "hybrid"):
        if encoder is None:
            raise RuntimeError("Embedding model required for dense/hybrid retrieval.")
        retriever = InMemoryVanillaRetriever(encoder, emb_dtype=getattr(args, "emb_dtype", None))
        retriever.build_index_for_question(passage_entries)
        dense_scores = retriever.score(question)
        if args.retriever == "dense":
            hits = retriever.retrieve(question, k=args.topk)
        else:
            bm25 = BM25Index([entry["text"] for entry in passage_entries])
            bm25_scores = bm25.get_scores(question)
            dense_rank = list(np.argsort(dense_scores)[::-1])
            bm25_rank = list(np.argsort(bm25_scores)[::-1])
            fused = rrf_fuse([dense_rank, bm25_rank], k=int(getattr(args, "hybrid_rrf_k", 60)))
            indices = sorted(fused, key=fused.get, reverse=True)[: args.topk]
            for idx in indices:
                entry = passage_entries[idx]
                hits.append(
                    {
                        "text": entry["text"],
                        "score": float(fused.get(idx, 0.0)),
                        "doc_id": entry.get("doc_id"),
                        "sent_ids": entry.get("sent_ids"),
                        "passage_id": entry.get("passage_id"),
                    }
                )
    else:
        bm25 = BM25Index([entry["text"] for entry in passage_entries])
        scores = bm25.get_scores(question)
        indices = list(np.argsort(scores)[::-1][: args.topk])
        for idx in indices:
            entry = passage_entries[idx]
            hits.append(
                {
                    "text": entry["text"],
                    "score": float(scores[idx]),
                    "doc_id": entry.get("doc_id"),
                    "sent_ids": entry.get("sent_ids"),
                    "passage_id": entry.get("passage_id"),
                }
            )

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
                retrieved=[{**hit, "rank": i + 1} for i, hit in enumerate(hits)],
                topk=len(hits),
                final_context=contexts_used,
                final_context_tokens=context_tokens,
                context_budget_tokens=int(getattr(args, "context_budget", 0) or 0) or None,
                log_dir=log_dir,
            )
        except Exception as log_exc:
            logger.error(f"retrieval logging failed for {qid}: {log_exc}")

        pred_evidence = [str(h.get("passage_id")) for h in hits if h.get("passage_id")]
        # unique while preserving order
        seen = set()
        pred_evidence = [p for p in pred_evidence if not (p in seen or seen.add(p))]
        return qid, question, ans, pred_evidence, contexts_used, context_tokens
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        return qid, question, "error", [], contexts_used, context_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vanilla RAG baseline on MuSiQue")
    parser.add_argument("--dataset", required=True, help="Path to MuSiQue jsonl/json")
    parser.add_argument("--output", default=None, help="Output path for musique_results.jsonl")
    parser.add_argument("--qa-path", default=None, help="Optional QA log path")
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for LLM decoding")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--emb-device", default=None)
    parser.add_argument("--emb-dtype", default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Retrieval mode (dense, bm25, hybrid)",
    )
    parser.add_argument("--hybrid-rrf-k", type=int, default=60, help="RRF k for hybrid fusion")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=20)
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    parser.add_argument("--num-workers", type=int, default=1)
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
            extra={
                "max_context": args.max_context,
                "emb_model": args.emb_model,
                "retriever": args.retriever,
            },
        ),
    )

    llm = LLMChatClient(
        endpoint=args.lm_endpoint,
        model=args.lm_model,
        temperature=0.0,
        max_tokens=args.max_new_tokens if args.max_new_tokens is not None else 8192,
    )
    device = args.emb_device or detect_device()
    encoder = None
    if args.retriever in ("dense", "hybrid"):
        encoder = get_embedding_model(args.emb_model, device, torch_dtype=args.emb_dtype)
        logger.info(f"Embedding model {args.emb_model} on {device} (dtype={args.emb_dtype or 'auto'})")
    else:
        logger.info("BM25 retrieval selected; skipping embedding model init.")

    results: List[Dict[str, Any]] = []
    qa_rows: List[Tuple[str, str]] = []
    pred_raw_records: List[Dict[str, Any]] = []

    num_workers = max(1, args.num_workers)
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
            for item in data
        ]
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
