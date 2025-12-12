import argparse
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
    clean_hotpot_answer,
    detect_device,
    format_context,
    get_embedding_model,
    select_workspace,
)
from scripts.musique.baselines.musique_utils import load_dataset, save_musique_results_and_qa
from utils.retrieval_logger import log_retrieval


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
) -> Tuple[str, str, str, List[str]]:
    qid = str(item.get("id") or item.get("_id") or item.get("query_id") or "")
    question = str(item.get("question") or item.get("query") or "").strip()
    paragraphs = item.get("paragraphs") or item.get("contexts") or item.get("passages") or []
    entries = build_passage_entries(paragraphs, max_passages=args.max_context)

    relrag = MiniRelRAG(encoder)
    relrag.build(entries)
    hits = relrag.retrieve(question, k=args.topk)

    context_str = format_context([h["text"] for h in hits])
    prompt = f"""Answer the question based on the selected paragraphs.
Keep the answer concise.

{context_str}

Question: {question}
Answer:"""
    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = clean_hotpot_answer(resp.content)
        try:
            log_retrieval(
                sample_id=qid,
                dataset=dataset_name,
                run_name=run_name,
                retrieved=hits,
                topk=len(hits),
                final_context=[
                    {
                        "doc_id": h.get("doc_id"),
                        "sent_ids": h.get("sent_ids"),
                        "passage_id": h.get("passage_id"),
                        "text": h.get("text"),
                    }
                    for h in hits
                ],
                log_dir=log_dir,
            )
        except Exception as log_exc:
            logger.error(f"retrieval logging failed for {qid}: {log_exc}")

        pred_evidence = [str(h.get("passage_id")) for h in hits if h.get("passage_id")]
        seen = set()
        pred_evidence = [p for p in pred_evidence if not (p in seen or seen.add(p))]
        return qid, question, ans, pred_evidence
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        return qid, question, "error", []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RelRAG baseline on MuSiQue")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--qa-path", default=None)
    parser.add_argument("--result-root", default="result/musique")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--emb-device", default=None)
    parser.add_argument("--emb-dtype", default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = select_workspace(Path(args.result_root), "musique_relrag", args.new)
    run_name = work_dir.name
    dataset_name = "musique"
    output_path = Path(args.output) if args.output else work_dir / "musique_results.jsonl"
    qa_path = Path(args.qa_path) if args.qa_path else work_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir}")

    llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)
    device = args.emb_device or detect_device()
    encoder = get_embedding_model(args.emb_model, device, torch_dtype=args.emb_dtype)
    logger.info(f"Embedding model {args.emb_model} on {device} (dtype={args.emb_dtype or 'auto'})")

    results: List[Dict[str, Any]] = []
    qa_rows: List[Tuple[str, str]] = []

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
                log_dir=work_dir,
            )
            for item in data
        ]
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

